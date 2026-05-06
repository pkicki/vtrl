# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import sys
import time
from dataclasses import dataclass, fields

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agent import Agent
from hopper import Hopper
#from pendulum_env import PendulumRK4
#from stiff_contact_oscilator import StiffContactOscillator
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pointmass2d_env import PointMass2D


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    record_video_enabled: bool = False
    """whether to record videos of the agent during training and evaluation"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Pendulum-v1"
    """the id of the environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    eval_num_steps: int = 200
    """the number of steps to run in each environment per evaluation""" 
    eval_episodes: int = 64
    """the number of episodes to run for evaluation"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    sim_dt: float = 0.01
    actor_train_dt: float = 0.01
    actor_eval_dt: float = 0.01
    eval_time: float = 2.0

#ENV = StiffContactOscillator
#ENV = PendulumRK4
#ENV = Hopper
ENV = PointMass2D

def evaluate(agent, sim_dt, actor_dt, eval_episodes, num_steps, gamma, device):
    with torch.no_grad():
        envs = ENV(sim_dt=sim_dt, actor_dt=actor_dt, max_episode_steps=num_steps, num_envs=eval_episodes, device=device)
        agent.eval()

        #obs, _ = envs.reset()
        obs, _ = envs.reset(seed=42)
        episodic_rewards = torch.zeros(eval_episodes, device=device)
        episodic_returns = torch.zeros(eval_episodes, device=device)
        for step in range(num_steps):
            #actions, _, _, _ = agent.get_action_and_value(obs)
            actions = agent.actor_mean(obs)
            next_obs, reward, _, _, infos = envs.step(actions)
            episodic_rewards += reward
            episodic_returns += reward * gamma ** step
            obs = next_obs

    agent.train()
    return episodic_returns, episodic_rewards


def record_video(agent, num_steps, device, video_path):
    """Run a single deterministic episode and save a rendered video.

    Uses the pygame ``rgb_array`` renderer from StiffContactOscillator –
    much faster than the old matplotlib animation path.
    """
    import imageio

    with torch.no_grad():
        env = ENV(
            sim_dt=args.sim_dt,
            actor_dt=args.actor_eval_dt,
            num_envs=1,
            device=device,
            render_mode="rgb_array"
        )
        agent.eval()

        obs, _ = env.reset(seed=42)
        frames = []
        rewards = 0

        with torch.no_grad():
            for _ in range(num_steps):
                #action, _, _, _ = agent.get_action_and_value(obs)
                action = agent.actor_mean(obs)
                obs, reward, _, _, _ = env.step(action)
                rewards += reward
                frame = env.render()   # ndarray (H, W, 3), uint8
                if frame is not None:
                    frames.append(frame)

        env.close()
    agent.train()

    saved_path = video_path
    try:
        imageio.mimwrite(video_path, frames, fps=20, quality=8)
    except Exception:
        gif_path = os.path.splitext(video_path)[0] + ".gif"
        imageio.mimwrite(gif_path, frames, fps=20)
        saved_path = gif_path

    return saved_path


if __name__ == "__main__":
    # ── Config-file pre-processing ──────────────────────────────────────
    # Allow --config path/to/config.yaml (or --config=path/to/config.yaml).
    # Values in the file become new defaults; CLI flags still override them.
    _default_args = Args()
    _argv = sys.argv[1:]
    for _i, _arg in enumerate(_argv):
        if _arg == "--config" and _i + 1 < len(_argv):
            _config_path = _argv[_i + 1]
            sys.argv.pop(_i + 1)  # remove '--config'
            sys.argv.pop(_i + 1)  # remove the path (now at the same index)
            with open(_config_path) as _f:
                _cfg = yaml.safe_load(_f) or {}
            _field_names = {f.name for f in fields(Args)}
            for _k, _v in _cfg.items():
                if _k in _field_names:
                    setattr(_default_args, _k, _v)
                else:
                    print(f"[config] Unknown field '{_k}' – ignored.")
            break
        elif _arg.startswith("--config="):
            _config_path = _arg.split("=", 1)[1]
            sys.argv.pop(_i + 1)  # remove '--config=path'
            with open(_config_path) as _f:
                _cfg = yaml.safe_load(_f) or {}
            _field_names = {f.name for f in fields(Args)}
            for _k, _v in _cfg.items():
                if _k in _field_names:
                    setattr(_default_args, _k, _v)
                else:
                    print(f"[config] Unknown field '{_k}' – ignored.")
            break

    args = tyro.cli(Args, default=_default_args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    video_dir = f"videos/{run_name}"
    os.makedirs(video_dir, exist_ok=True)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = ENV(sim_dt=args.sim_dt, actor_dt=args.actor_train_dt,
               num_envs=args.num_envs, device=device)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    total_eval_time = 0.0
    total_train_time = 0.0
    total_env_time = 0.0
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    pbar = tqdm(range(1, args.num_iterations + 1), desc="Training", unit="iter")
    for iteration in pbar:
        print("Actor std:", torch.exp(agent.actor_logstd).mean().item())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        env_metrics_acc: dict[str, float] = {}
        env_metrics_counts: dict[str, int] = {}

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            _t_env = time.perf_counter()
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            total_env_time += time.perf_counter() - _t_env
            # Only mark done (nextnonterminal=0) for true termination.
            # Truncated episodes should still have their value bootstrapped.
            next_done = terminations
            rewards[step] = reward.to(device).view(-1)
            next_obs = next_obs.to(device)
            next_done = next_done.to(device)

            # For truncated envs, add the bootstrap value to the reward so
            # that the GAE target is correct even though next_done=0 here.
            if truncations.any():
                with torch.no_grad():
                    final_obs = infos["final_observation"].to(device)
                    bootstrap_val = agent.get_value(final_obs).flatten()
                    rewards[step] += args.gamma * bootstrap_val * truncations.float()

            if "env_metrics" in infos:
                for k, v in infos["env_metrics"].items():
                    env_metrics_acc[k] = env_metrics_acc.get(k, 0.0) + float(v)
                    env_metrics_counts[k] = env_metrics_counts.get(k, 0) + 1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        mean_reward = rewards.mean().item()

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.type(torch.float32)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].type(torch.float32)
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        _t_train = time.perf_counter()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        total_train_time += time.perf_counter() - _t_train
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        _t_eval = time.perf_counter()
        episodic_returns, episodic_rewards = evaluate(
            agent,
            sim_dt=args.sim_dt,
            actor_dt=args.actor_eval_dt,
            eval_episodes=args.eval_episodes,
            num_steps=args.eval_num_steps,
            gamma=args.gamma,
            device=device
        )
        total_eval_time += time.perf_counter() - _t_eval
        print(f"""eval_episodic_return={episodic_returns.mean().item()},
              eval_episodic_reward={episodic_rewards.mean().item()}""")
        writer.add_scalar("eval/episodic_return_mean", episodic_returns.mean().item(), global_step)
        writer.add_scalar("eval/episodic_reward_mean", episodic_rewards.mean().item(), global_step)
        if args.track:
            wandb.log({
                "eval/episodic_return_mean": episodic_returns.mean().item(),
                "eval/episodic_reward_mean": episodic_rewards.mean().item(),
                "eval/episodic_return_std": episodic_returns.std().item(),
                "eval/episodic_reward_std": episodic_rewards.std().item(),
            }, step=global_step)
        if args.record_video_enabled:
            iter_video_path = os.path.join(video_dir, f"iter_{iteration:04d}_step_{global_step}.mp4")
            saved_video = record_video(
                agent,
                num_steps=int(args.eval_time / args.actor_eval_dt),
                device=device,
                video_path=iter_video_path,
            )
            pbar.write(f"Video saved: {saved_video}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/mean_reward", mean_reward, global_step)

        # Log env-specific metrics (works for any env that populates infos["env_metrics"])
        env_metrics_mean = {
            k: env_metrics_acc[k] / env_metrics_counts[k]
            for k in env_metrics_acc
        }
        for k, v in env_metrics_mean.items():
            writer.add_scalar(f"env/{k}", v, global_step)
        if args.track and env_metrics_mean:
            wandb.log({f"env/{k}": v for k, v in env_metrics_mean.items()}, step=global_step)
        if args.track:
            wandb.log({"charts/mean_reward": mean_reward}, step=global_step)

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        pbar.set_postfix({
            "steps": global_step,
            "SPS": sps,
            "rew": f"{mean_reward:.4f}",
            "v_loss": f"{v_loss.item():.3f}",
            "pg_loss": f"{pg_loss.item():.3f}",
            **{k: f"{v:.3f}" for k, v in env_metrics_mean.items()},
        })
        writer.add_scalar("charts/SPS", sps, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        _t_eval = time.perf_counter()
        episodic_returns, episodic_rewards = evaluate(
            agent,
            sim_dt=args.sim_dt,
            actor_dt=args.actor_eval_dt,
            eval_episodes=args.eval_episodes,
            num_steps=args.eval_num_steps,
            gamma=args.gamma,
            device=device
        )
        total_eval_time += time.perf_counter() - _t_eval
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        if args.track:
            wandb.log({
                "eval/final_episodic_return_mean": episodic_returns.mean().item(),
                "eval/final_episodic_reward_mean": episodic_rewards.mean().item(),
                "eval/final_episodic_return_std": episodic_returns.std().item(),
                "eval/final_episodic_reward_std": episodic_rewards.std().item(),
            })

        if args.record_video_enabled:
            final_video_path = os.path.join(video_dir, "final.mp4")
            saved_video = record_video(
                agent,
                num_steps=int(args.eval_time / args.actor_eval_dt),
                device=device,
                video_path=final_video_path,
            )
            print(f"Final evaluation video saved to {saved_video}")

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    print("\nTiming summary:")
    print(f"  Total environment interaction time: {total_env_time:.2f}s")
    print(f"  Total training (optimization) time: {total_train_time:.2f}s")
    print(f"  Total evaluation time:              {total_eval_time:.2f}s")
    if args.track:
        wandb.log({
            "timing/total_env_time": total_env_time,
            "timing/total_train_time": total_train_time,
            "timing/total_eval_time": total_eval_time,
        })
    envs.close()
    writer.close()
