import hydrogym.firedrake as hgym
from hydrogym import FlowEnv
import numpy as np
from envs.pinball_env import PinballEnv

env = PinballEnv({"Re": 100, "warmup_steps": 50})
obs = env.reset()

obs = np.array(obs, dtype=np.float32)
print("obs shape:", obs.shape)   # expect (6,)
print("obs:", obs)               # should be small pressure values


for i in range(5):
    action = env.action_space.sample()
    raw_obs, reward, done, info = env.step(action)
    obs = np.array(raw_obs, dtype=np.float32)
    print(f"step {i}  reward={reward:.4f}  obs[:6]={obs[:6].round(3)}")