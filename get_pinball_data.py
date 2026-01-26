from hydrogym.firedrake import Pinball, IPCS
import numpy as np
import h5py
import os
import time

# --- CREATE FLOW AND SOLVER ---
print("Creating flow and solver...")
flow = Pinball()
solver = IPCS(flow, dt=flow.DEFAULT_DT)

SAVE_DIR = "pinball_dataset_h5"
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, "fluidic_pinball_data.h5")

EPISODES = 1000
MAX_STEPS = 500  # Maximum steps per episode

print(f"Flow configuration:")
print(f"  Action dimension: {flow.ACT_DIM}")
print(f"  Observation dimension: {flow.OBS_DIM}")
print(f"  Max control: {flow.MAX_CONTROL}")
print(f"  Reynolds number: {flow.Re}")
print(f"  Time step: {flow.DEFAULT_DT}")

# Extract Reynolds number once (it's constant)
try:
    Re_value = float(flow.Re)
except:
    try:
        Re_value = float(flow.DEFAULT_REYNOLDS)
    except:
        Re_value = 30.0

print(f"  Using Re = {Re_value}")
print(f"\nStarting data collection for {EPISODES} episodes...")

with h5py.File(save_path, "w") as f:

    for ep in range(EPISODES):
        ep_start_time = time.time()
        
        # Reset the flow
        print(f"\nEpisode {ep+1}/{EPISODES}: Resetting flow...", end='', flush=True)
        flow.reset()
        print(" Done.", flush=True)
        
        grp = f.create_group(f"episode_{ep}")

        obs_list = []
        act_list = []
        rew_list = []
        cd_list = []
        cl_list = []
        re_list = []

        step_start_time = time.time()
        for step in range(MAX_STEPS):
            if step % 50 == 0:
                print(f"  Step {step}/{MAX_STEPS}...", end='', flush=True)
            
            # Get current observation
            obs = flow.get_observations()
            
            # Sample random action (control)
            action = np.random.uniform(-flow.MAX_CONTROL, flow.MAX_CONTROL, size=flow.ACT_DIM)
            
            # Apply control
            flow.set_control(action)
            
            # Step the simulation forward
            solver.step(step)
            
            # Compute reward/objective
            reward = flow.evaluate_objective()
            
            # Compute forces
            forces = flow.compute_forces()
            cd = forces.get('Cd', np.nan) if isinstance(forces, dict) else np.nan
            cl = forces.get('Cl', np.nan) if isinstance(forces, dict) else np.nan

            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(reward)
            cd_list.append(cd)
            cl_list.append(cl)
            re_list.append(Re_value)
            
            if step % 50 == 0:
                elapsed = time.time() - step_start_time
                print(f" ({elapsed:.2f}s)", flush=True)
                step_start_time = time.time()

        print(f"  Saving episode data...", end='', flush=True)
        grp.create_dataset("obs", data=np.array(obs_list), compression="gzip")
        grp.create_dataset("actions", data=np.array(act_list), compression="gzip")
        grp.create_dataset("rewards", data=np.array(rew_list), compression="gzip")
        grp.create_dataset("Cd", data=np.array(cd_list), compression="gzip")
        grp.create_dataset("Cl", data=np.array(cl_list), compression="gzip")
        grp.create_dataset("Re", data=np.array(re_list), compression="gzip")
        print(" Done.", flush=True)

        ep_time = time.time() - ep_start_time
        print(f"Episode {ep+1} completed in {ep_time:.2f}s ({len(obs_list)} steps)")

print(f"\n{'='*50}")
print(f"Dataset saved to {save_path}")
print(f"{'='*50}")