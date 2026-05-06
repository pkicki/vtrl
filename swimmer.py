import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def obs2state(obs):
    obs = obs[0]
    return obs

# Create the environment
env_id = "Swimmer-v4"
env = make_vec_env(env_id, n_envs=1)

e = env.envs[0].unwrapped

e.frame_skip = 1

#dt1 = 0.05
#dt1 = 0.01
#dt2 = 0.001
dt1 = 0.1
dt2 = 0.01

control = np.array([1.0, 0.0])

e.model.opt.timestep = dt1 
e.reset()
e.data.qpos = np.zeros(5)
e.data.qvel = np.zeros(5)
next_s_1 = obs2state(e.step(control))
print(next_s_1)

e.reset()
e.data.qpos = np.zeros(5)
e.data.qvel = np.zeros(5)
e.model.opt.timestep = dt2 
for i in range(int(dt1 / dt2)):
    next_s_2 = obs2state(e.step(control))

if dt1 % dt2 != 0:
    e.model.opt.timestep = dt1 % dt2
    next_s_2 = obs2state(e.step(control))

print(next_s_2)
print()
print(np.abs(next_s_1 - next_s_2))
a = 0