import os
# Must be set before JAX/XLA initializes to prevent it from
# pre-allocating all GPU memory, which conflicts with PyTorch's CUDA context.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from time import perf_counter

import torch
import wandb
from tqdm import tqdm

from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import BraxEnv, StepCounter, TransformedEnv, RewardScaling
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators # Use the Enum

from tensordict.nn import NormalParamExtractor

import jax
print(f"JAX devices: {jax.devices()}")


#import gymnasium as gym
#import mujoco_playground
#from torch_playground_wrapper import TorchPlaygroundWrapper

# 1. Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
num_envs = 192
frames_per_batch = 16 * num_envs
total_frames = 1_000_000
num_epochs = 10
minibatch_size = 512
lr = 3e-4

# 2. Environment Setup
def make_env():
    # MJX backend is used by default in BraxEnv
    torch_env = BraxEnv(env_name="halfcheetah", batch_size=(num_envs,), device=device)
    #env = mujoco_playground.make("HalfCheetah-v4")
    #torch_env = TorchPlaygroundWrapper(env)
    env = TransformedEnv(torch_env)
    env.append_transform(RewardScaling(loc=0.0, scale=0.1))
    env.append_transform(StepCounter())
    return env

train_env = make_env()

# 3. Policy and Value Networks
obs_shape = train_env.observation_spec["observation"].shape
action_shape = train_env.action_spec.shape

actor_net = nn.Sequential(
    nn.Linear(obs_shape[-1], 256), nn.Tanh(),
    nn.Linear(256, 256), nn.Tanh(),
    nn.Linear(256, 2 * action_shape[-1]),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
policy = ProbabilisticActor(
    module=policy_module,
    in_keys=["loc", "scale"],
    spec=train_env.action_spec,
    distribution_class=TanhNormal,
    return_log_prob=True,
).to(device)

value_net = nn.Sequential(
    nn.Linear(obs_shape[-1], 256), nn.Tanh(),
    nn.Linear(256, 256), nn.Tanh(),
    nn.Linear(256, 1),
)
value_module = ValueOperator(module=value_net, in_keys=["observation"]).to(device)

# 4. Logger
wandb.init(
    project="torchrl_mjx",
    name="ppo_halfcheetah",
    config={
        "num_envs": num_envs,
        "frames_per_batch": frames_per_batch,
        "total_frames": total_frames,
        "num_epochs": num_epochs,
        "minibatch_size": minibatch_size,
        "lr": lr,
    },
)

# 5. Loss and GAE Setup
loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=value_module,
    clip_epsilon=0.2,
    entropy_bonus=True,
    entropy_coeff=0.01,
)
# This replaces the manual GAE import:
loss_module.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)
loss_module.to(device)

optim = torch.optim.Adam(loss_module.parameters(), lr=lr)

collector = SyncDataCollector(
    train_env,
    policy,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    device=device,
    storing_device=device,
)

replay_buffer = ReplayBuffer(storage=LazyTensorStorage(frames_per_batch, device=device))

# 6. Training Loop
pbar = tqdm(total=total_frames)
t0 = perf_counter()
for i, tensordict_data in enumerate(collector):
    t1 = perf_counter()
    
    # Move to GPU once, then calculate advantages
    tensordict_data = tensordict_data.to(device)
    with torch.no_grad():
        tensordict_data = loss_module.value_estimator(tensordict_data)

    # Optimization
    replay_buffer.extend(tensordict_data.reshape(-1))
    t2 = perf_counter()
    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample(minibatch_size)
            loss_vals = loss_module(subdata)
            loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
            optim.step()
            optim.zero_grad()
    t3 = perf_counter()

    # Logging
    train_reward = tensordict_data["next", "reward"].mean().item()
    wandb.log({"train/reward_step": train_reward}, step=collector._frames)
    
    pbar.update(tensordict_data.numel())
    print("Data collection time: {:.2f}s,"
          "Advantage calculation time: {:.2f}s,"
          "Optimization time: {:.2f}s".format(t1 - t0, t2 - t1, t3 - t2))
    t0 = perf_counter()

wandb.finish()