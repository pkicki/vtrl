import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import pendulum_env  # registers PendulumRK4-v1

def obs2state(obs):
    obs = obs[0]
    cos_theta, sin_theta, theta_dot = obs
    theta = np.arctan2(sin_theta, cos_theta)
    return np.array([theta, theta_dot])

# Create the environment
env_id = "PendulumRK4-v1"
env = make_vec_env(env_id, n_envs=1)

e = env.envs[0].unwrapped

dt1 = 0.2
dt2 = 0.001

e.dt = dt1 
e.reset()
e.state = np.array([0.0, 0.0])
next_s = obs2state(e.step(np.array([1.0])))
print(next_s)

e.state = np.array([0.0, 0.0])
e.dt = dt2 
for i in range(int(dt1 / dt2)):
    next_s = obs2state(e.step(np.array([1.0])))

if dt1 % dt2 != 0:
    e.dt = dt1 % dt2
    next_s = obs2state(e.step(np.array([1.0])))

print(next_s)
a = 0

# Instantiate the agent
model = PPO(
    "MlpPolicy",
    env,
    gamma=0.98,
    # Using https://proceedings.mlr.press/v164/raffin22a.html
    #use_sde=True,
    #sde_sample_freq=4,
    learning_rate=1e-3,
    verbose=1,
    device="cpu",
)

# Train the agent
model.learn(total_timesteps=int(1e5))