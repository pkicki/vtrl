import torch
import numpy as np
import gymnasium as gym

class Hopper:
    def __init__(
        self,
        num_envs=1024,
        max_episode_steps=1000,
        dt=0.01,
        frame_skip=1,
        device="cpu",
        g=9.81,
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.frame_skip = frame_skip
        self.device = device

        self.g = g

        # State: [height, velocity]
        self.state = torch.zeros(num_envs, 2, device=device)

        # Contact parameters
        self.ground_height = 0.0
        self.k_spring = 200.0
        self.damping = 2.0

        self.max_steps = max_episode_steps
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.single_observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(2,), dtype=np.float32)
        self.single_action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        h = torch.zeros(self.num_envs, device=self.device) * 1.0
        v = torch.zeros(self.num_envs, device=self.device)

        self.state = torch.stack([h, v], dim=-1)
        self.steps.zero_()

        return self.state.clone(), {}

    def step(self, action):
        """
        action: [B, 1] thrust applied during contact
        """
        total_reward = torch.zeros(self.num_envs, device=self.device)

        for _ in range(self.frame_skip):
            h, v = self.state[:, 0], self.state[:, 1]

            in_contact = h <= self.ground_height

            # Spring force when in contact
            spring_force = torch.zeros_like(h)
            spring_force[in_contact] = (
                -self.k_spring * (h[in_contact] - self.ground_height)
                - self.damping * v[in_contact]
            )

            # Control force only in contact
            control_force = torch.zeros_like(h)
            control_force[in_contact] = action[in_contact, 0]

            # Total acceleration
            a = (spring_force + control_force) - self.g

            # Integrate
            v = v + self.dt * a
            h = h + self.dt * v

            # Prevent sinking too deep
            h = torch.maximum(h, torch.tensor(self.ground_height, device=self.device))

            self.state = torch.stack([h, v], dim=-1)

            # Reward: stay high + smooth motion
            reward = h - 0.001 * (action[:, 0] ** 2)
            total_reward += reward

        self.steps += 1

        truncated = self.steps >= self.max_steps
        terminated = torch.zeros_like(truncated)

        # Save final observation before auto-reset (for GAE value bootstrapping)
        final_obs = self.state.clone()

        # Auto-reset truncated environments
        if truncated.any():
            idx = truncated.nonzero(as_tuple=True)[0]
            self.state[idx] = torch.tensor([0., 0.], device=self.device)
            self.steps[idx] = 0.

        infos = {"final_observation": final_obs}

        return self.state.clone(), total_reward, terminated, truncated, infos

    def close(self):
        pass