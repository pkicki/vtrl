# PPO for Gymnasium vectorized continuous-control environments (e.g. HalfCheetah-v4)
import os
import random
import sys
import time
from dataclasses import dataclass, fields
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import Agent


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
    """the wandb project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb project"""
    save_model: bool = True
    """whether to save model into runs/{run_name}"""
    record_video_enabled: bool = False
    """whether to record videos of deterministic policy rollouts"""

    # Environment arguments (Gymnasium vectorized env only)
    env_id: str = "HalfCheetah-v4"
    """the id of the Gymnasium environment"""
    num_envs: int = 256
    """number of parallel environments"""
    vector_env: str = "sync"
    """vectorized env backend: sync or async"""
    max_episode_steps: int = 1000
    """time limit for each environment instance"""

    # PPO arguments
    total_timesteps: int = 1_000_000
    """total timesteps of experiments"""
    learning_rate: float = 3e-4
    """learning rate of optimizer"""
    num_steps: int = 2048
    """steps to run in each environment per rollout"""
    eval_num_steps: int = 1000
    """max steps per evaluation episode"""
    eval_episodes: int = 64
    """number of episodes to run for evaluation"""
    anneal_lr: bool = True
    """toggle learning rate annealing"""
    gamma: float = 0.99
    """discount factor"""
    gae_lambda: float = 0.95
    """lambda for GAE"""
    num_minibatches: int = 32
    """number of minibatches"""
    update_epochs: int = 10
    """policy update epochs"""
    norm_adv: bool = True
    """normalize advantages"""
    clip_coef: float = 0.2
    """surrogate clipping coefficient"""
    clip_vloss: bool = True
    """use clipped value loss"""
    ent_coef: float = 0.0
    """entropy coefficient"""
    vf_coef: float = 0.5
    """value function coefficient"""
    max_grad_norm: float = 0.5
    """gradient clipping max norm"""
    target_kl: float | None = None
    """target KL divergence threshold"""

    # runtime-computed
    batch_size: int = 0
    """batch size (computed at runtime)"""
    minibatch_size: int = 0
    """minibatch size (computed at runtime)"""
    num_iterations: int = 0
    """iterations (computed at runtime)"""


def _load_args_with_yaml() -> Args:
    default_args = Args()
    argv = sys.argv[1:]

    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            # Remove the two arguments so tyro never sees them.
            sys.argv.pop(i + 1)
            sys.argv.pop(i + 1)
            break
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            # Remove only the current argument from sys.argv.
            sys.argv.pop(i + 1)
            break
    else:
        config_path = None

    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        field_names = {f.name for f in fields(Args)}
        for k, v in cfg.items():
            if k in field_names:
                setattr(default_args, k, v)
            else:
                print(f"[config] Unknown field '{k}' – ignored.")

    return tyro.cli(Args, default=default_args)


def _to_tensor(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _to_bool_tensor(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.bool)
    return torch.as_tensor(x, device=device, dtype=torch.bool)


def _make_single_env_fn(env_id: str, max_episode_steps: int, seed: int, idx: int) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        env = gym.make(env_id)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        return env

    return thunk


def make_vector_env(env_id: str, num_envs: int, max_episode_steps: int, seed: int, vector_env: str):
    env_fns = [_make_single_env_fn(env_id, max_episode_steps, seed, i) for i in range(num_envs)]
    if vector_env == "sync":
        envs = gym.vector.SyncVectorEnv(env_fns)
    elif vector_env == "async":
        envs = gym.vector.AsyncVectorEnv(env_fns)
    else:
        raise ValueError(f"vector_env must be 'sync' or 'async', got '{vector_env}'")

    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise ValueError("This script supports only continuous Box action spaces.")

    return envs


def _extract_final_observation_batch(infos, truncations_t: torch.Tensor, device: torch.device, obs_shape):
    if "final_observation" not in infos:
        return None

    final_obs = infos["final_observation"]
    trunc_np = truncations_t.detach().cpu().numpy().astype(bool)

    out = np.zeros((trunc_np.shape[0],) + obs_shape, dtype=np.float32)
    has_any = False

    if isinstance(final_obs, np.ndarray) and final_obs.dtype == object:
        for i, fo in enumerate(final_obs):
            if trunc_np[i] and fo is not None:
                out[i] = np.asarray(fo, dtype=np.float32)
                has_any = True
    else:
        arr = np.asarray(final_obs, dtype=np.float32)
        if arr.ndim >= 1 and arr.shape[0] == trunc_np.shape[0]:
            out = arr
            has_any = True
        elif arr.ndim >= 1 and arr.shape[0] == int(trunc_np.sum()):
            out[trunc_np] = arr
            has_any = True

    if not has_any:
        return None
    return torch.as_tensor(out, device=device, dtype=torch.float32)


def evaluate(agent: Agent, args: Args, device: torch.device):
    eval_envs = make_vector_env(
        env_id=args.env_id,
        num_envs=args.eval_episodes,
        max_episode_steps=args.eval_num_steps,
        seed=args.seed + 100_000,
        vector_env="sync",
    )

    action_low = torch.as_tensor(eval_envs.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(eval_envs.single_action_space.high, device=device, dtype=torch.float32)

    with torch.no_grad():
        agent.eval()
        obs_np, _ = eval_envs.reset(seed=args.seed + 1234)
        obs = _to_tensor(obs_np, device)

        episodic_rewards = torch.zeros(args.eval_episodes, device=device)
        episodic_returns = torch.zeros(args.eval_episodes, device=device)

        for step in range(args.eval_num_steps):
            actions = agent.actor_mean(obs)
            clipped_actions = torch.clamp(actions, action_low, action_high)
            next_obs_np, reward_np, _, _, _ = eval_envs.step(clipped_actions.cpu().numpy())

            reward_t = _to_tensor(reward_np, device).view(-1)
            episodic_rewards += reward_t
            episodic_returns += reward_t * (args.gamma ** step)
            obs = _to_tensor(next_obs_np, device)

    eval_envs.close()
    agent.train()
    return episodic_returns, episodic_rewards


def record_video(agent: Agent, args: Args, device: torch.device, video_path: str):
    import imageio

    env = gym.make(args.env_id, render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.eval_num_steps)

    action_low = torch.as_tensor(env.action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space.high, device=device, dtype=torch.float32)

    frames = []
    obs_np, _ = env.reset(seed=args.seed + 42)

    with torch.no_grad():
        agent.eval()
        for _ in range(args.eval_num_steps):
            obs_t = _to_tensor(obs_np, device).unsqueeze(0)
            action = agent.actor_mean(obs_t).squeeze(0)
            action = torch.clamp(action, action_low, action_high)
            obs_np, _, terminated, truncated, _ = env.step(action.cpu().numpy())

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if terminated or truncated:
                break

    env.close()
    agent.train()

    saved_path = video_path
    try:
        imageio.mimwrite(video_path, frames, fps=30, quality=8)
    except Exception:
        gif_path = os.path.splitext(video_path)[0] + ".gif"
        imageio.mimwrite(gif_path, frames, fps=30)
        saved_path = gif_path

    return saved_path


if __name__ == "__main__":
    args = _load_args_with_yaml()
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = make_vector_env(
        env_id=args.env_id,
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        vector_env=args.vector_env,
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    action_low = torch.as_tensor(envs.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(envs.single_action_space.high, device=device, dtype=torch.float32)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    total_eval_time = 0.0
    total_train_time = 0.0
    total_env_time = 0.0

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = _to_tensor(next_obs_np, device)
    next_done = torch.zeros(args.num_envs, device=device)

    pbar = tqdm(range(1, args.num_iterations + 1), desc="Training", unit="iter")
    for iteration in pbar:
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            env_action = torch.clamp(action, action_low, action_high).cpu().numpy()

            t_env = time.perf_counter()
            next_obs_np, reward_np, terminations, truncations, infos = envs.step(env_action)
            total_env_time += time.perf_counter() - t_env

            terminations_t = _to_bool_tensor(terminations, device)
            truncations_t = _to_bool_tensor(truncations, device)

            next_done = terminations_t
            rewards[step] = _to_tensor(reward_np, device).view(-1)
            next_obs = _to_tensor(next_obs_np, device)

            if truncations_t.any():
                final_obs = _extract_final_observation_batch(
                    infos, truncations_t, device, envs.single_observation_space.shape
                )
                if final_obs is not None:
                    with torch.no_grad():
                        bootstrap_val = agent.get_value(final_obs).flatten()
                    rewards[step] += args.gamma * bootstrap_val * truncations_t.float()

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(info["episode"]["r"])
                        ep_r = np.mean(info["episode"]["r"])
                        ep_l = np.mean(info["episode"]["l"])
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", ep_l, global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        t_train = time.perf_counter()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        total_train_time += time.perf_counter() - t_train

        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        t_eval = time.perf_counter()
        episodic_returns, episodic_rewards = evaluate(agent, args, device)
        total_eval_time += time.perf_counter() - t_eval

        writer.add_scalar("eval/episodic_return_mean", episodic_returns.mean().item(), global_step)
        writer.add_scalar("eval/episodic_reward_mean", episodic_rewards.mean().item(), global_step)

        if args.track:
            wandb.log(
                {
                    "eval/episodic_return_mean": episodic_returns.mean().item(),
                    "eval/episodic_reward_mean": episodic_rewards.mean().item(),
                    "eval/episodic_return_std": episodic_returns.std().item(),
                    "eval/episodic_reward_std": episodic_rewards.std().item(),
                },
                step=global_step,
            )

        if args.record_video_enabled:
            iter_video_path = os.path.join(video_dir, f"iter_{iteration:04d}_step_{global_step}.mp4")
            saved_video = record_video(agent, args, device, iter_video_path)
            pbar.write(f"Video saved: {saved_video}")

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        pbar.set_postfix({"steps": global_step, "SPS": sps, "v_loss": f"{v_loss.item():.3f}"})

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        t_eval = time.perf_counter()
        episodic_returns, episodic_rewards = evaluate(agent, args, device)
        total_eval_time += time.perf_counter() - t_eval

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return.item(), idx)

        if args.track:
            wandb.log(
                {
                    "eval/final_episodic_return_mean": episodic_returns.mean().item(),
                    "eval/final_episodic_reward_mean": episodic_rewards.mean().item(),
                    "eval/final_episodic_return_std": episodic_returns.std().item(),
                    "eval/final_episodic_reward_std": episodic_rewards.std().item(),
                }
            )

        if args.record_video_enabled:
            final_video_path = os.path.join(video_dir, "final.mp4")
            saved_video = record_video(agent, args, device, final_video_path)
            print(f"Final evaluation video saved to {saved_video}")

    print("\nTiming summary:")
    print(f"  Total environment interaction time: {total_env_time:.2f}s")
    print(f"  Total training (optimization) time: {total_train_time:.2f}s")
    print(f"  Total evaluation time:              {total_eval_time:.2f}s")

    if args.track:
        wandb.log(
            {
                "timing/total_env_time": total_env_time,
                "timing/total_train_time": total_train_time,
                "timing/total_eval_time": total_eval_time,
            }
        )

    envs.close()
    writer.close()
