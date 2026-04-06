# scripts/train_local.py (stub)
import hydrogym.firedrake as hgym
from hydrogym import FlowEnv
import numpy as np

env = FlowEnv({
    'flow': hgym.Pinball,
    'flow_config': {'mesh': 'medium', 'Re': 100},
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-2},
    'actuation_config': {'num_substeps': 2},
})

obs = env.reset()
obs = np.array(obs, dtype=np.float32)
print("obs shape:", obs.shape)   # expect (6,)
print("obs:", obs)               # should be small pressure values

for i in range(10):
    action = env.action_space.sample()
    raw_obs, reward, done, info = env.step(action)
    obs = np.array(raw_obs, dtype=np.float32)
    print(f"step {i:02d}  obs={obs}  reward={reward:.4f}")