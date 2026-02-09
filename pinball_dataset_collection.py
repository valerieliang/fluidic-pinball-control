from hydrogym.firedrake import Pinball, IPCS
import numpy as np
import h5py
import os
import time
import argparse
from scipy import signal

# Parse command line arguments
parser = argparse.ArgumentParser(description='Collect fluidic pinball dataset')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('--episodes', type=int, default=2, help='Number of episodes')
parser.add_argument('--steps', type=int, default=200, help='Steps per episode')
parser.add_argument('--reynolds', type=float, default=30.0, help='Reynolds number')
args = parser.parse_args()

VERBOSE = args.verbose
EPISODES = args.episodes
MAX_STEPS = args.steps
TARGET_RE = args.reynolds

if VERBOSE:
    print("="*60)
    print("Data Collection for RL Training - Fluidic Pinball")
    print("="*60)
    print(f"Target Reynolds Number: {TARGET_RE}")

# Initialize flow and solver
try:
    flow = Pinball()
    solver = IPCS(flow, dt=flow.DEFAULT_DT)
    if VERBOSE:
        print("Flow and solver created successfully")
        print(f"  Timestep: {flow.DEFAULT_DT}")
except Exception as e:
    print(f"Failed to create flow/solver: {e}")
    raise

# Extract Reynolds number
try:
    Re_value = float(flow.Re) if hasattr(flow, 'Re') else TARGET_RE
    if VERBOSE:
        print(f"\nReynolds number: {Re_value}")
except:
    Re_value = TARGET_RE
    if VERBOSE:
        print(f"Using target Reynolds: {Re_value}")

# Get timesteps per control action
try:
    timesteps_per_action = flow.num_steps if hasattr(flow, 'num_steps') else 1
    if VERBOSE:
        print(f"  Timesteps per control action: {timesteps_per_action}")
except:
    timesteps_per_action = 1

# Test observation to get actual dimensions
try:
    flow.reset()
    test_obs = flow.get_observations()
    actual_obs_dim = len(test_obs) if isinstance(test_obs, (list, np.ndarray)) else 1
    if VERBOSE:
        print(f"\nActual observation dimension: {actual_obs_dim}")
except Exception as e:
    if VERBOSE:
        print(f"Could not determine observation dimension: {e}")
    actual_obs_dim = flow.OBS_DIM if hasattr(flow, 'OBS_DIM') else 6

if VERBOSE:
    print(f"Flow configuration:")
    print(f"  Action dimension: {flow.ACT_DIM}")

SAVE_DIR = "pinball_dataset_h5"
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, f"fluidic_pinball_Re{int(Re_value)}_data.h5")

if VERBOSE:
    print(f"\nGenerating dataset:")
    print(f"  Episodes: {EPISODES}")
    print(f"  Steps per episode: {MAX_STEPS}")
    print(f"  Total timesteps per episode: {MAX_STEPS * timesteps_per_action}")

def compute_f0_from_lift(cl_history, dt):
    """
    Compute dominant frequency f0 from lift coefficient time series using FFT
    
    Args:
        cl_history: List of lift coefficient values over time
        dt: Timestep size
        
    Returns:
        f0: Dominant frequency (Hz)
    """
    if len(cl_history) < 10:
        return np.nan
    
    try:
        # Convert to numpy array
        cl_array = np.array(cl_history)
        
        # Remove mean (detrend)
        cl_detrended = cl_array - np.mean(cl_array)
        
        # Compute FFT
        fft = np.fft.rfft(cl_detrended)
        freqs = np.fft.rfftfreq(len(cl_detrended), d=dt)
        
        # Find dominant frequency (excluding DC component)
        power = np.abs(fft[1:])**2
        dominant_idx = np.argmax(power) + 1
        f0 = freqs[dominant_idx]
        
        return f0
    except Exception as e:
        if VERBOSE:
            print(f"Warning: Failed to compute f0: {e}")
        return np.nan

with h5py.File(save_path, "w") as f:
    
    total_start = time.time()
    
    for ep in range(EPISODES):
        ep_start = time.time()
        
        try:
            flow.reset()
            if VERBOSE:
                print(f"\nEpisode {ep}: Flow initialized")
        except Exception as e:
            print(f"Episode {ep}: Failed to reset flow: {e}")
            continue
        
        grp = f.create_group(f"episode_{ep}")

        # Standard data
        obs_list = []
        next_obs_list = []
        act_list = []
        rew_list = []
        cd_vec_list = []  # 3-element vector (one per cylinder)
        cl_vec_list = []  # 3-element vector (one per cylinder)
        re_list = []
        done_list = []
        vorticity_list = []
        pressure_list = []
        kinetic_energy_list = []
        timestep_list = []
        cumulative_reward_list = []
        
        # For f0 computation (need lift history for FFT)
        cl2_history = []  # Lift on cylinder 2
        cl3_history = []  # Lift on cylinder 3
        
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
                    if VERBOSE and step == 0:
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
                    if VERBOSE and step == 0:
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
            
            # Compute forces - returns tuple of (cd_vec, cl_vec) with 3 elements each
            try:
                forces = flow.compute_forces()
                if isinstance(forces, (tuple, list)) and len(forces) >= 2:
                    cd_vec = np.array(forces[0])  # 3-element array
                    cl_vec = np.array(forces[1])  # 3-element array
                    
                    # Ensure they are 3-element arrays
                    if len(cd_vec) != 3:
                        cd_vec = np.full(3, np.nan)
                    if len(cl_vec) != 3:
                        cl_vec = np.full(3, np.nan)
                    
                    # Track lift history for f0 computation
                    # CL[1] is cylinder 2, CL[2] is cylinder 3
                    if len(cl_vec) >= 3:
                        cl2_history.append(cl_vec[1])
                        cl3_history.append(cl_vec[2])
                else:
                    cd_vec = np.full(3, np.nan)
                    cl_vec = np.full(3, np.nan)
            except Exception as e:
                if VERBOSE:
                    print(f"Episode {ep}, Step {step}: Failed to compute forces: {e}")
                cd_vec = np.full(3, np.nan)
                cl_vec = np.full(3, np.nan)
            
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
            cd_vec_list.append(cd_vec)  # Store full 3-element vector
            cl_vec_list.append(cl_vec)  # Store full 3-element vector
            re_list.append(Re_value)
            done_list.append(done)
            vorticity_list.append(vort_value)
            pressure_list.append(pressure_value)
            kinetic_energy_list.append(ke)
            timestep_list.append(step)
            cumulative_reward_list.append(cumulative_reward)

        # Compute f0 from lift history
        dt = flow.DEFAULT_DT * timesteps_per_action
        f0_cyl2 = compute_f0_from_lift(cl2_history, dt)
        f0_cyl3 = compute_f0_from_lift(cl3_history, dt)
        # Take the average or use one as representative
        if not np.isnan(f0_cyl2) and not np.isnan(f0_cyl3):
            f0 = (f0_cyl2 + f0_cyl3) / 2  # Average if both available
        elif not np.isnan(f0_cyl2):
            f0 = f0_cyl2  # Use cylinder 2 if only one available
        elif not np.isnan(f0_cyl3):
            f0 = f0_cyl3  # Use cylinder 3 if only one available
        else:
            f0 = np.nan


        if VERBOSE:
            print(f"Episode {ep}: Computed f0 = {f0:.4f} Hz (cyl2: {f0_cyl2:.4f}, cyl3: {f0_cyl3:.4f})")

        # Store f0 as a scalar dataset
        grp.create_dataset("f0", data=f0)
        grp.attrs['f0'] = float(f0) 
        grp.create_dataset("f0_cyl2", data=f0_cyl2)
        grp.create_dataset("f0_cyl3", data=f0_cyl3)

        # Save all datasets
        try:
            # Create arrays to track dimensions
            obs_lengths = np.array([len(o) for o in obs_list])
            next_obs_lengths = np.array([len(o) for o in next_obs_list])
            act_lengths = np.array([len(a) for a in act_list])
            
            # Flatten all variable-length data into 1D arrays
            obs_flat = np.concatenate(obs_list)
            next_obs_flat = np.concatenate(next_obs_list)
            act_flat = np.concatenate(act_list)
            
            # CD and CL are fixed 3-element vectors, store as 2D array (steps x 3)
            cd_array = np.array(cd_vec_list)  # Shape: (MAX_STEPS, 3)
            cl_array = np.array(cl_vec_list)  # Shape: (MAX_STEPS, 3)
            
            if VERBOSE:
                print(f"Episode {ep}: Data shapes - obs: {obs_flat.shape}, CD: {cd_array.shape}, CL: {cl_array.shape}")
            
            # Save flattened observation/action data and length metadata
            grp.create_dataset("obs_flat", data=obs_flat, compression="gzip")
            grp.create_dataset("obs_lengths", data=obs_lengths, compression="gzip")
            grp.create_dataset("next_obs_flat", data=next_obs_flat, compression="gzip")
            grp.create_dataset("next_obs_lengths", data=next_obs_lengths, compression="gzip")
            grp.create_dataset("actions_flat", data=act_flat, compression="gzip")
            grp.create_dataset("actions_lengths", data=act_lengths, compression="gzip")
            
            # Save force vectors as 2D arrays
            grp.create_dataset("CD_vec", data=cd_array, compression="gzip")  # (steps, 3)
            grp.create_dataset("CL_vec", data=cl_array, compression="gzip")  # (steps, 3)
            
            # Also save individual components for convenience
            grp.create_dataset("CD_total", data=np.sum(cd_array, axis=1), compression="gzip")
            grp.create_dataset("CL2", data=cl_array[:, 1], compression="gzip")  # Cylinder 2
            grp.create_dataset("CL3", data=cl_array[:, 2], compression="gzip")  # Cylinder 3
            
            # Scalar data
            grp.create_dataset("rewards", data=np.array(rew_list), compression="gzip")
            grp.create_dataset("Re", data=np.array(re_list), compression="gzip")
            grp.create_dataset("done", data=np.array(done_list), compression="gzip")
            grp.create_dataset("vorticity", data=np.array(vorticity_list), compression="gzip")
            grp.create_dataset("pressure", data=np.array(pressure_list), compression="gzip")
            grp.create_dataset("kinetic_energy", data=np.array(kinetic_energy_list), compression="gzip")
            grp.create_dataset("timestep", data=np.array(timestep_list), compression="gzip")
            grp.create_dataset("cumulative_reward", data=np.array(cumulative_reward_list), compression="gzip")
            
            # Store metadata about array structure and f0
            grp.attrs['num_steps'] = len(obs_list)
            grp.attrs['obs_dim'] = actual_obs_dim
            grp.attrs['act_dim'] = flow.ACT_DIM
            grp.attrs['Re'] = Re_value
            grp.attrs['f0'] = f0
            grp.attrs['f0_cyl2'] = f0_cyl2
            grp.attrs['f0_cyl3'] = f0_cyl3
            grp.attrs['dt'] = dt
            
        except Exception as e:
            print(f"Episode {ep}: Failed to save datasets: {e}")
            if VERBOSE:
                import traceback
                traceback.print_exc()
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
    print("  Force vectors (fixed 3-element per step):")
    print("    - CD_vec (steps x 3): drag on each cylinder")
    print("    - CL_vec (steps x 3): lift on each cylinder")
    print("    - CD_total: sum of CD_vec")
    print("    - CL2: lift on cylinder 2")
    print("    - CL3: lift on cylinder 3")
    print("  Scalars per step:")
    print("    - rewards, Re, done, vorticity, pressure")
    print("    - kinetic_energy, timestep, cumulative_reward")
    print("  Episode attributes:")
    print("    - f0: dominant frequency (Hz)")
    print("    - f0_cyl2, f0_cyl3: per-cylinder frequencies")
    print("\nTo reconstruct variable-length arrays:")
    print("  offsets = np.concatenate([[0], np.cumsum(lengths[:-1])])")
    print("  arrays = [flat[offset:offset+length] for offset, length in zip(offsets, lengths)]")
else:
    print(f"Dataset complete: {EPISODES * MAX_STEPS} transitions saved to {save_path}")