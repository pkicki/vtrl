import datetime
import functools
import gc
import math
import os

import jax
import mujoco_playground
from mujoco_playground import registry, wrapper
from mujoco_playground.config import dm_control_suite_params
from brax.io import model
from brax.training import acting
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
import mediapy as media
import wandb

from experiment_launcher import single_experiment, run_experiment

from utils.render import record_final_evaluation_video


@single_experiment
def experiment(
    env_name: str = "CheetahRun",
    num_timesteps: int = 1_638_400,
    num_envs: int = 128,
    batch_size: int = 128,
    unroll_length: int = 32,
    num_minibatches: int = 4,
    train_action_repeat: int = 1,
    eval_action_repeat: int = 1,
    num_eval_envs: int = 64,
    num_resets_per_eval: int = 0,
    gamma: float = 0.995,
    results_dir: str = "results",
    seed: int = 444,
):
    gamma = gamma ** train_action_repeat  # Adjust gamma for action repeat
    unroll_length = int(32 / train_action_repeat) # Keep total unroll length constant regardless of action repeat


    print(f"Loading environment: {env_name}")
    # Single RNG source for this run: controls policy init, training env,
    # separate eval env rollouts and final video rollout.
    master_key = jax.random.PRNGKey(seed)
    eval_key, final_video_key = jax.random.split(master_key)
    
    # 2. Load the environment via the registry
    env_cfg = registry.get_default_config(env_name)
    env = registry.load(env_name, config=env_cfg)

    ppo_params = dm_control_suite_params.brax_ppo_config(env_name, impl="jax")
    
    ppo_params.num_envs = num_envs # 512 is safe for 6GB VRAM
    ppo_params.batch_size = batch_size
    ppo_params.num_timesteps = num_timesteps
    ppo_params.action_repeat = train_action_repeat
    ppo_params.num_minibatches = num_minibatches
    ppo_params.unroll_length = unroll_length
    ppo_params.num_resets_per_eval = num_resets_per_eval
    ppo_params.discounting = gamma

    # Ensure exact total timesteps and one callback/eval after each epoch.
    # In Brax PPO, each training step advances this many env steps:
    # batch_size * unroll_length * num_minibatches * action_repeat
    env_steps_per_training_step = (
        int(ppo_params.batch_size)
        * int(ppo_params.unroll_length)
        * int(ppo_params.num_minibatches)
        * int(ppo_params.action_repeat)
    )
    if num_timesteps % env_steps_per_training_step != 0:
        raise ValueError(
            "NUM_TIMESTEPS must be divisible by "
            "(batch_size * unroll_length * num_minibatches * action_repeat) "
            "to get an exact total number of timesteps. "
            f"Got NUM_TIMESTEPS={num_timesteps} and step_size={env_steps_per_training_step}."
        )

    num_epochs = num_timesteps // env_steps_per_training_step
    if num_epochs < 1:
        raise ValueError(
            "NUM_TIMESTEPS is too small for current PPO batch settings. "
            f"Need at least {env_steps_per_training_step} timesteps."
        )

    # Brax uses num_evals_after_init = max(num_evals - 1, 1),
    # and calls policy_params_fn once per epoch. Setting num_evals this way
    # gives one epoch step per iteration and an eval callback after each epoch.
    ppo_params.num_evals = int(num_epochs + 1)

    # Separate eval env (frame/action skip = 1)
    eval_env_ = registry.load(env_name, config=env_cfg)
    eval_env = wrapper.wrap_for_brax_training(
        eval_env_,
        episode_length=int(ppo_params.episode_length),
        action_repeat=eval_action_repeat,
    )
    eval_state = {
        "last_eval_step": -1,
        "evaluator": None,
        "last_eval_metrics": {},
    }

    print(f"Starting training (this will JIT compile first, which takes a minute)...")
    wandb.init(
        project=f"playground_{env_name.lower()}_ppo",
        name=f"{env_name}_{datetime.datetime.now().strftime('%m%d-%H%M')}_ar{train_action_repeat}_bs{batch_size}",
        config=ppo_params.to_dict()
    )

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128), # Standard for Cheetah
        policy_obs_key="state",
        value_obs_key="state"
    )

    def progress(num_steps, metrics):
        if wandb.run is not None:
            wandb.log({k: float(v) for k, v in metrics.items()}, step=num_steps)
        sps = metrics.get("training/sps", metrics.get("eval/sps", 0.0))
        print(f"Step: {num_steps} | Training SPS: {float(sps):.0f}")

    def periodic_eval(step, make_policy, params):
        # Called at step=0 (initial params) and then once after each epoch.
        if step == 0 or step == eval_state["last_eval_step"]:
            return

        if eval_state["evaluator"] is None:
            eval_state["evaluator"] = acting.Evaluator(
                eval_env=eval_env,
                eval_policy_fn=functools.partial(make_policy, deterministic=True),
                num_eval_envs=num_eval_envs,
                episode_length=int(ppo_params.episode_length),
                action_repeat=eval_action_repeat,
                key=eval_key,
            )

        metrics = eval_state["evaluator"].run_evaluation(params, training_metrics={})
        eval_metrics = {
            k.replace("eval/", "eval_fs1/"): float(v)
            for k, v in metrics.items()
            if k.startswith("eval/")
        }
        if wandb.run is not None and eval_metrics:
            wandb.log(eval_metrics, step=step)

        eval_state["last_eval_metrics"] = eval_metrics

        reward = eval_metrics.get("eval_fs1/episode_reward")
        sps = eval_metrics.get("eval_fs1/sps")
        if reward is not None and sps is not None:
            print(f"[Separate Eval] Step: {step} | Reward(fs=1): {reward:.2f} | SPS: {sps:.0f}")

        eval_state["last_eval_step"] = step
    
    train_fn = functools.partial(
        ppo.train,
        **ppo_params,
        network_factory=network_factory,
        seed=seed,
        environment=env,
        progress_fn=progress,
        policy_params_fn=periodic_eval,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        run_evals=False,
    )
    # 3. Run training 
    # In Brax, calling ppo.train starts the training process immediately.
    make_inference_fn, params, metrics = train_fn()

    print("\nTraining complete.")
    
    # metrics contains the training history
    final_reward = eval_state["last_eval_metrics"].get("eval_fs1/episode_reward")
    if final_reward is not None:
        print(f"Final Separate Eval Reward (fs=1): {final_reward:.2f}")
    else:
        print("Final Separate Eval Reward (fs=1): not available")

    # Release heavy training objects before rendering video.
    # Keep `env` alive, because we reuse it for video rollout to avoid
    # re-initializing fresh CUDA graph captures.
    del train_fn, env
    gc.collect()
    jax.clear_caches()  # Free XLA compilation caches to reclaim GPU memory

    # 4. Record final deterministic evaluation video.
    run_name = f"{env_name}_{datetime.datetime.now().strftime('%m%d-%H%M')}_ar{train_action_repeat}_bs{batch_size}_seed{seed}"
    video_path = os.path.join("videos", run_name, "final_eval.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    saved_video = record_final_evaluation_video(
        make_inference_fn=make_inference_fn,
        params=params,
        render_env=eval_env_,
        #episode_length=int(ppo_params.episode_length),
        episode_length=int(ppo_params.episode_length / 4),
        action_repeat=eval_action_repeat,
        seed=int(jax.random.randint(final_video_key, (), 0, 2**31 - 1)),
        output_path=video_path,
    )
    print(f"Final evaluation video saved to {saved_video}")
    if wandb.run is not None:
        wandb.log({"final_eval/video": wandb.Video(saved_video, format="mp4")})

    # 5. Save the policy
    model.save_params("cheetah_params", params)
    print("Model saved to 'cheetah_params'")

if __name__ == "__main__":
    run_experiment(experiment)