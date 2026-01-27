from hydrogym.firedrake import Pinball, IPCS
import numpy as np
import h5py
import os
import time

print("="*60)
print("Enhanced Data Collection for RL Training")
print("="*60)

# Initialize flow and solver
try:
    flow = Pinball()
    solver = IPCS(flow, dt=flow.DEFAULT_DT)
    print("Flow and solver created successfully")
except Exception as e:
    print(f"Failed to create flow/solver: {e}")
    raise

# Extract Reynolds number
try:
    Re_value = float(flow.Re)
except:
    try:
        Re_value = float(flow.DEFAULT_REYNOLDS)
    except:
        Re_value = 30.0
        print(f"⚠ Could not extract Re, using default: {Re_value}")

print(f"\nFlow configuration:")
print(f"  Action dimension: {flow.ACT_DIM}")
print(f"  Observation dimension: {flow.OBS_DIM}")
print(f"  Reynolds number: {Re_value}")

SAVE_DIR = "pinball_dataset_h5"
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, "fluidic_pinball_data.h5")

EPISODES = 100
MAX_STEPS = 200

print(f"\nGenerating dataset:")
print(f"  Episodes: {EPISODES}")
print(f"  Steps per episode: {MAX_STEPS}")

with h5py.File(save_path, "w") as f:
    
    total_start = time.time()
    
    for ep in range(EPISODES):
        ep_start = time.time()
        
        try:
            flow.reset()
        except Exception as e:
            print(f"Episode {ep}: Failed to reset flow: {e}")
            continue
        
        grp = f.create_group(f"episode_{ep}")

        # Standard data
        obs_list = []
        next_obs_list = []
        act_list = []
        rew_list = []
        cd_list = []
        cl_list = []
        re_list = []
        done_list = []
        vorticity_list = []
        pressure_list = []
        kinetic_energy_list = []
        timestep_list = []
        cumulative_reward_list = []
        
        cumulative_reward = 0.0
        
        for step in range(MAX_STEPS):
            # Get current observation
            try:
                obs = flow.get_observations()
            except Exception as e:
                print(f"Episode {ep}, Step {step}: Failed to get observation: {e}")
                obs = np.nan
            
            # Generate action with some structure
            try:
                if np.random.random() < 0.3:
                    action = np.random.uniform(-flow.MAX_CONTROL, flow.MAX_CONTROL, size=flow.ACT_DIM)
                else:
                    freq = 0.5 + 0.5 * np.random.random()
                    action = 0.5 * flow.MAX_CONTROL * np.sin(2 * np.pi * freq * step * flow.DEFAULT_DT)
            except Exception as e:
                print(f"Episode {ep}, Step {step}: Failed to generate action: {e}")
                action = np.zeros(flow.ACT_DIM)
            
            # Apply control
            try:
                flow.set_control(action)
            except Exception as e:
                print(f"Episode {ep}, Step {step}: Failed to set control: {e}")
            
            # Step simulation
            try:
                solver.step(step)
            except Exception as e:
                print(f"Episode {ep}, Step {step}: Failed to step solver: {e}")
                break
            
            # Get next observation
            try:
                next_obs = flow.get_observations()
            except Exception as e:
                print(f"Episode {ep}, Step {step}: Failed to get next observation: {e}")
                next_obs = np.nan
            
            # Compute reward
            try:
                reward = flow.evaluate_objective()
            except Exception as e:
                print(f"Episode {ep}, Step {step}: Failed to evaluate objective: {e}")
                reward = np.nan
            
            # Compute forces
            try:
                forces = flow.compute_forces()
                cd = forces.get('Cd', np.nan) if isinstance(forces, dict) else np.nan
                cl = forces.get('Cl', np.nan) if isinstance(forces, dict) else np.nan
            except Exception as e:
                print(f"Episode {ep}, Step {step}: Failed to compute forces: {e}")
                cd = np.nan
                cl = np.nan
            
            # Compute vorticity
            try:
                vort = flow.vorticity()
                vort_value = np.linalg.norm(vort) if hasattr(vort, '__len__') else float(vort)
            except Exception as e:
                vort_value = np.nan
            
            # Compute pressure
            try:
                pressure_value = float(np.mean(flow.p.dat.data))
            except Exception as e:
                pressure_value = np.nan
            
            # Compute kinetic energy
            try:
                u_data = flow.u.dat.data
                ke = 0.5 * np.mean(u_data**2)
            except Exception as e:
                ke = np.nan
            
            # Episode termination
            done = (step == MAX_STEPS - 1)
            
            # Cumulative reward
            try:
                cumulative_reward += reward if not np.isnan(reward) else 0.0
            except:
                pass
            
            # Store everything
            obs_list.append(obs)
            next_obs_list.append(next_obs)
            act_list.append(action)
            rew_list.append(reward)
            cd_list.append(cd)
            cl_list.append(cl)
            re_list.append(Re_value)
            done_list.append(done)
            vorticity_list.append(vort_value)
            pressure_list.append(pressure_value)
            kinetic_energy_list.append(ke)
            timestep_list.append(step)
            cumulative_reward_list.append(cumulative_reward)

        # Save all datasets
        try:
            grp.create_dataset("obs", data=np.array(obs_list), compression="gzip")
            grp.create_dataset("next_obs", data=np.array(next_obs_list), compression="gzip")
            grp.create_dataset("actions", data=np.array(act_list), compression="gzip")
            grp.create_dataset("rewards", data=np.array(rew_list), compression="gzip")
            grp.create_dataset("Cd", data=np.array(cd_list), compression="gzip")
            grp.create_dataset("Cl", data=np.array(cl_list), compression="gzip")
            grp.create_dataset("Re", data=np.array(re_list), compression="gzip")
            grp.create_dataset("done", data=np.array(done_list), compression="gzip")
            grp.create_dataset("vorticity", data=np.array(vorticity_list), compression="gzip")
            grp.create_dataset("pressure", data=np.array(pressure_list), compression="gzip")
            grp.create_dataset("kinetic_energy", data=np.array(kinetic_energy_list), compression="gzip")
            grp.create_dataset("timestep", data=np.array(timestep_list), compression="gzip")
            grp.create_dataset("cumulative_reward", data=np.array(cumulative_reward_list), compression="gzip")
        except Exception as e:
            print(f"Episode {ep}: Failed to save datasets: {e}")
            continue

        ep_time = time.time() - ep_start
        if (ep + 1) % 10 == 0 or ep == 0:
            total_time = time.time() - total_start
            avg_time = total_time / (ep + 1)
            remaining = avg_time * (EPISODES - ep - 1)
            print(f"Episode {ep+1}/{EPISODES}: {ep_time:.1f}s | ETA: {remaining/60:.1f}min")

total_time = time.time() - total_start
print(f"\n" + "="*60)
print(f"  Dataset complete!")
print(f"  Total time: {total_time/60:.1f} minutes")
print(f"  Features per timestep: 13")
print(f"  Saved to: {save_path}")
print("="*60)

print("\nDataset contains:")
print("  Standard RL: obs, next_obs, actions, rewards, done")
print("  Aerodynamics: Cd, Cl")  
print("  Flow physics: vorticity, pressure, kinetic_energy")
print("  Metadata: timestep, cumulative_reward, Re")