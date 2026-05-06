import torch
from torch.utils import dlpack as torch_dlpack
import jax.dlpack as jax_dlpack

class TorchPlaygroundWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def _to_torch(self, x):
        # Zero-copy conversion from JAX to PyTorch
        return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x))

    def reset(self):
        obs, info = self.env.reset()
        return self._to_torch(obs), info

    def step(self, action):
        # If action is a torch tensor, move it to JAX
        if isinstance(action, torch.Tensor):
            action = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(action))
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        return (
            self._to_torch(obs),
            self._to_torch(reward),
            terminated,
            truncated,
            info
        )