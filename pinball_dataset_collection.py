from hydrogym.firedrake import Pinball, IPCS
import numpy as np
import h5py
import os
import time
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Collect fluidic pinball dataset')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
args = parser.parse_args()
VERBOSE = args.verbose

if VERBOSE:
    print("="*60)
    print("Data Collection for RL Training")
    print("="*60)

# Initialize flow and solver
try:
    flow = Pinball()
    solver = IPCS(flow, dt=flow.DEFAULT_DT)
    if VERBOSE:
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
        if VERBOSE:
            print(f"Could not extract Re, using default: {Re_value}")

# Test observation to get actual dimensions
try:
    flow.reset()
    test_obs = flow.get_observations()
    actual_obs_dim = len(test_obs) if isinstance(test_obs, (list, np.ndarray)) else 1
    if VERBOSE:
        print(f"Actual observation dimension: {actual_obs_dim}")
except Exception as e:
    if VERBOSE:
        print(f"Could not determine observation dimension: {e}")
    actual_obs_dim = flow.OBS_DIM

if VERBOSE:
    print(f"\nFlow configuration:")
    print(f"  Action dimension: {flow.ACT_DIM}")
    print(f"  Observation dimension: {actual_obs_dim} (actual) vs {flow.OBS_DIM} (constant)")
    print(f"  Reynolds number: {Re_value}")

SAVE_DIR = "pinball_dataset_h5"
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, "fluidic_pinball_data.h5")

EPISODES = 100
MAX_STEPS = 200

if VERBOSE:
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
                # Convert to numpy array and flatten to ensure consistent shape
                if isinstance(obs, (list, tuple)):
                    obs = np.array(obs)
                obs = np.atleast_1d(obs).flatten()
                # Ensure correct dimension
                if len(obs) != actual_obs_dim:
                    if VERBOSE:
                        print(f"Episode {ep}, Step {step}: Observation dim mismatch: {len(obs)} vs {actual_obs_dim}")
                    obs = np.resize(obs, actual_obs_dim)
            except Exception as e:
                if VERBOSE:
                    print(f"Episode {ep}, Step {step}: Failed to get observation: {e}")
                obs = np.full(actual_obs_dim, np.nan)
            
            # Generate action with some structure
            try:
                if np.random.random() < 0.3:
                    action = np.random.uniform(-flow.MAX_CONTROL, flow.MAX_CONTROL, size=flow.ACT_DIM)
                else:
                    freq = 0.5 + 0.5 * np.random.random()
                    action_value = 0.5 * flow.MAX_CONTROL * np.sin(2 * np.pi * freq * step * flow.DEFAULT_DT)
                    # Ensure action is always an array with correct dimension
                    action = np.full(flow.ACT_DIM, action_value)
            except Exception as e:
                if VERBOSE:
                    print(f"Episode {ep}, Step {step}: Failed to generate action: {e}")
                action = np.zeros(flow.ACT_DIM)
            
            # Apply control
            try:
                flow.set_control(action)
            except Exception as e:
                if VERBOSE:
                    print(f"Episode {ep}, Step {step}: Failed to set control: {e}")
            
            # Step simulation
            try:
                solver.step(step)
            except Exception as e:
                if VERBOSE:
                    print(f"Episode {ep}, Step {step}: Failed to step solver: {e}")
                break
            
            # Get next observation
            try:
                next_obs = flow.get_observations()
                # Convert to numpy array and flatten to ensure consistent shape
                if isinstance(next_obs, (list, tuple)):
                    next_obs = np.array(next_obs)
                next_obs = np.atleast_1d(next_obs).flatten()
                # Ensure correct dimension
                if len(next_obs) != actual_obs_dim:
                    if VERBOSE:
                        print(f"Episode {ep}, Step {step}: Next obs dim mismatch: {len(next_obs)} vs {actual_obs_dim}")
                    next_obs = np.resize(next_obs, actual_obs_dim)
            except Exception as e:
                if VERBOSE:
                    print(f"Episode {ep}, Step {step}: Failed to get next observation: {e}")
                next_obs = np.full(actual_obs_dim, np.nan)
            
            # Compute reward
            try:
                reward = flow.evaluate_objective()
            except Exception as e:
                if VERBOSE:
                    print(f"Episode {ep}, Step {step}: Failed to evaluate objective: {e}")
                reward = np.nan
            
            # Compute forces
            try:
                forces = flow.compute_forces()
                cd = forces.get('Cd', np.nan) if isinstance(forces, dict) else np.nan
                cl = forces.get('Cl', np.nan) if isinstance(forces, dict) else np.nan
            except Exception as e:
                if VERBOSE:
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
            # Ensure consistent array shapes
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            if not isinstance(next_obs, np.ndarray):
                next_obs = np.array(next_obs)
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            
            # Flatten if needed to ensure 1D
            obs = obs.flatten()
            next_obs = next_obs.flatten()
            action = action.flatten()
            
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
            # For variable-length arrays, we need to use special dtype or store as object arrays
            # Option 1: Store flattened with length metadata
            # Option 2: Use HDF5 variable-length dtype
            
            # Create arrays to track dimensions
            obs_lengths = np.array([len(o) for o in obs_list])
            next_obs_lengths = np.array([len(o) for o in next_obs_list])
            act_lengths = np.array([len(a) for a in act_list])
            
            # Flatten all variable-length data into 1D arrays
            obs_flat = np.concatenate(obs_list)
            next_obs_flat = np.concatenate(next_obs_list)
            act_flat = np.concatenate(act_list)
            
            if VERBOSE:
                print(f"Episode {ep}: Variable-length data - obs lengths: {np.unique(obs_lengths)}, action lengths: {np.unique(act_lengths)}")
            
            # Save flattened data and length metadata
            grp.create_dataset("obs_flat", data=obs_flat, compression="gzip")
            grp.create_dataset("obs_lengths", data=obs_lengths, compression="gzip")
            grp.create_dataset("next_obs_flat", data=next_obs_flat, compression="gzip")
            grp.create_dataset("next_obs_lengths", data=next_obs_lengths, compression="gzip")
            grp.create_dataset("actions_flat", data=act_flat, compression="gzip")
            grp.create_dataset("actions_lengths", data=act_lengths, compression="gzip")
            
            # Scalar data can be stored normally
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
            
            # Store metadata about array structure
            grp.attrs['num_steps'] = len(obs_list)
            grp.attrs['obs_dim_range'] = f"{obs_lengths.min()}-{obs_lengths.max()}"
            grp.attrs['act_dim_range'] = f"{act_lengths.min()}-{act_lengths.max()}"
            
        except Exception as e:
            print(f"Episode {ep}: Failed to save datasets: {e}")
            if VERBOSE:
                print(f"  obs_list length: {len(obs_list)}, sample shapes: {[o.shape for o in obs_list[:3]]}")
                print(f"  act_list length: {len(act_list)}, sample shapes: {[a.shape for a in act_list[:3]]}")
            continue

        ep_time = time.time() - ep_start
        if VERBOSE:
            if (ep + 1) % 10 == 0 or ep == 0:
                total_time = time.time() - total_start
                avg_time = total_time / (ep + 1)
                remaining = avg_time * (EPISODES - ep - 1)
                print(f"Episode {ep+1}/{EPISODES}: {ep_time:.1f}s | ETA: {remaining/60:.1f}min")
        else:
            # Non-verbose: just show progress at milestones
            if (ep + 1) % 25 == 0:
                print(f"Progress: {ep+1}/{EPISODES} episodes complete")

total_time = time.time() - total_start

if VERBOSE:
    print(f"\n" + "="*60)
    print(f"Dataset complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Total transitions: {EPISODES * MAX_STEPS}")
    print(f"  Saved to: {save_path}")
    print("="*60)
    
    print("\nDataset storage format:")
    print("  Variable-length arrays stored as:")
    print("    - obs_flat + obs_lengths (reconstruct with cumsum)")
    print("    - next_obs_flat + next_obs_lengths")
    print("    - actions_flat + actions_lengths")
    print("  Fixed-length scalars:")
    print("    - rewards, Cd, Cl, Re, done")
    print("    - vorticity, pressure, kinetic_energy")
    print("    - timestep, cumulative_reward")
    print("\nTo reconstruct variable-length arrays:")
    print("  offsets = np.concatenate([[0], np.cumsum(lengths[:-1])])")
    print("  arrays = [flat[offset:offset+length] for offset, length in zip(offsets, lengths)]")
else:
    print(f"Dataset complete: {EPISODES * MAX_STEPS} transitions saved to {save_path}")