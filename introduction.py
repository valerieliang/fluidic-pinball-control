import hydrogym.firedrake as hgym

env_config = {
    "flow": hgym.Pinball,
    "flow_config": {},
    "solver": hgym.SemiImplicitBDF,
    "solver_config": {"dt": 1e-2},
}
env = hgym.FlowEnv(env_config)

num_steps = 100
for i in range(num_steps):
    action = [0.0, 0.0, 0.0]  # Pinball has 3 cylinders, so 3 actions
    obs, reward, done, info = env.step(action)
    print(f"Step {i}: obs={obs}, reward={reward}")