import sys
import os

sys.path.insert(0, os.getcwd()) 

import warnings
warnings.filterwarnings("ignore")

# Suppress Gym warning
os.environ["GYM_WARNINGS"] = "false"

from envs.pinball_env_baseline import PinballEnvBaseline
import numpy as np

def test_with_long_warmup():
    """Test with sufficient warmup to develop flow"""
    print("=" * 70)
    print("TEST 1: Long warmup to develop flow")
    print("=" * 70)
    
    env = PinballEnvBaseline({
        "Re": 100,
        "mesh": "medium",
        "num_substeps": 5,
        "n_probes": 6,
        "warmup_steps": 100,  # Much longer warmup
        "verbose": False,
        "save_warmup_plots": False,
        "save_episode_snapshots": False,
        "save_episode_h5": False,
    })
    
    print(f"Warmup steps: {env.warmup_steps}")
    print(f"Substeps per action: {env.num_substeps}")
    print(f"Total simulation steps before first action: {env.warmup_steps * env.num_substeps}")
    
    # Reset with long warmup
    print("\n[1] Resetting with long warmup (this may take a minute)...")
    obs = env.reset()
    print(f"    Initial observation (probe values): {obs}")
    print(f"    Initial observation magnitude: {np.linalg.norm(obs):.6f}")
    
    # Take one step
    print("\n[2] Taking one action step...")
    action = np.array([0.0, 0.0, 0.0])  # Zero action to see natural flow
    print(f"    Action: {action}")
    
    obs, reward, done, info = env.step(action)
    
    # Analyze drag/lift from this step
    print("\n[3] DRAG/LIFT ANALYSIS FOR FIRST STEP:")
    print(f"    Number of drag samples: {len(env._episode_drag_buffer)}")
    print(f"    Drag values: {env._episode_drag_buffer}")
    print(f"    Lift values: {env._episode_lift_buffer}")
    
    if len(env._episode_drag_buffer) > 0:
        print(f"\n    Drag stats:")
        print(f"        Min: {min(env._episode_drag_buffer):.6f}")
        print(f"        Max: {max(env._episode_drag_buffer):.6f}")
        print(f"        Mean: {np.mean(env._episode_drag_buffer):.6f}")
        print(f"        Std: {np.std(env._episode_drag_buffer):.6f}")
        
        print(f"\n    Lift stats:")
        print(f"        Min: {min(env._episode_lift_buffer):.6f}")
        print(f"        Max: {max(env._episode_lift_buffer):.6f}")
        print(f"        Mean: {np.mean(env._episode_lift_buffer):.6f}")
        print(f"        Std: {np.std(env._episode_lift_buffer):.6f}")
        
        # Check if forces are actually non-zero
        max_drag = max(abs(d) for d in env._episode_drag_buffer)
        max_lift = max(abs(l) for l in env._episode_lift_buffer)
        
        if max_drag < 1e-6 and max_lift < 1e-6:
            print("\n    WARNING: All forces are zero! CFD may not be computing forces.")
        elif max_drag < 0.01 and max_lift < 0.01:
            print("\n    Forces are very small (<0.01). Flow may still be developing.")
        else:
            print("\n    Forces are non-zero and reasonable.")
    
    env.close()
    return env

def test_multiple_steps_with_monitoring():
    """Test multiple steps and monitor force development over time"""
    print("\n" + "=" * 70)
    print("TEST 2: Force development over multiple steps")
    print("=" * 70)
    
    env = PinballEnvBaseline({
        "Re": 100,
        "mesh": "medium",
        "num_substeps": 10,  # More substeps for better resolution
        "n_probes": 6,
        "warmup_steps": 200,  # Even longer warmup
        "verbose": False,
        "save_warmup_plots": False,
        "save_episode_snapshots": False,
    })
    
    obs = env.reset()
    
    print("\nTracking forces over 5 actions (each with 10 substeps)...")
    print(f"{'Action':<8} {'Step':<8} {'Mean Drag':<12} {'Mean |Lift|':<12} {'Max Drag':<12} {'Reward':<12}")
    print("-" * 70)
    
    all_drags = []
    all_lifts = []
    
    for action_idx in range(5):
        action = np.random.uniform(-5, 5, size=3)  # Random actions
        obs, reward, done, info = env.step(action)
        
        # Get forces from this step's substeps
        step_drags = env._episode_drag_buffer[-env.num_substeps:]
        step_lifts = env._episode_lift_buffer[-env.num_substeps:]
        
        mean_drag = np.mean(step_drags)
        mean_abs_lift = np.mean(np.abs(step_lifts))
        max_drag = np.max(np.abs(step_drags))
        
        all_drags.extend(step_drags)
        all_lifts.extend(step_lifts)
        
        print(f"{action_idx+1:<8} {action_idx+1:<8} {mean_drag:<12.6f} {mean_abs_lift:<12.6f} {max_drag:<12.6f} {reward:<12.6f}")
    
    # Overall statistics
    print("\n[4] OVERALL FORCE STATISTICS:")
    print(f"    Total drag samples: {len(all_drags)}")
    print(f"    Total lift samples: {len(all_lifts)}")
    print(f"    Drag - Mean: {np.mean(all_drags):.6f} +- {np.std(all_drags):.6f}")
    print(f"    Drag - Range: [{np.min(all_drags):.6f}, {np.max(all_drags):.6f}]")
    print(f"    Lift - Mean: {np.mean(all_lifts):.6f} +- {np.std(all_lifts):.6f}")
    print(f"    Lift - Range: [{np.min(all_lifts):.6f}, {np.max(all_lifts):.6f}]")
    
    # Check if forces are non-zero
    if np.max(np.abs(all_drags)) < 1e-6 and np.max(np.abs(all_lifts)) < 1e-6:
        print("\n    CRITICAL: All forces are zero!")
        print("    The CFD simulation is NOT computing drag/lift forces.")
        print("    Check:")
        print("      1. Is the flow solver actually running?")
        print("      2. Are the force probes properly configured?")
        print("      3. Is the simulation time advancing?")
    elif np.max(np.abs(all_drags)) < 0.001:
        print("\n    Forces are very small. Possible issues:")
        print("      - Re=100 may produce very small forces")
        print("      - Mesh might be too coarse")
        print("      - Need even longer warmup")
    else:
        print("\n    Forces are non-zero and within expected range.")
    
    env.close()

def test_force_scaling():
    """Test if forces scale properly with Reynolds number"""
    print("\n" + "=" * 70)
    print("TEST 3: Force scaling with Reynolds number")
    print("=" * 70)
    
    reynolds_numbers = [100, 200, 500]
    results = {}
    
    for Re in reynolds_numbers:
        print(f"\nTesting Re = {Re}...")
        
        env = PinballEnvBaseline({
            "Re": Re,
            "mesh": "medium",
            "num_substeps": 5,
            "n_probes": 6,
            "warmup_steps": 150,
            "verbose": False,
            "save_warmup_plots": False,
            "save_episode_snapshots": False,
        })
        
        obs = env.reset()
        
        # Take 3 steps and average forces
        all_drags = []
        all_lifts = []
        
        for _ in range(3):
            action = np.zeros(3)  # Zero action
            obs, reward, done, info = env.step(action)
            all_drags.extend(env._episode_drag_buffer[-env.num_substeps:])
            all_lifts.extend(env._episode_lift_buffer[-env.num_substeps:])
        
        results[Re] = {
            'mean_drag': np.mean(all_drags),
            'std_drag': np.std(all_drags),
            'mean_lift': np.mean(np.abs(all_lifts)),
            'max_drag': np.max(np.abs(all_drags))
        }
        
        print(f"    Mean drag: {results[Re]['mean_drag']:.6f} +- {results[Re]['std_drag']:.6f}")
        print(f"    Mean |lift|: {results[Re]['mean_lift']:.6f}")
        
        env.close()
    
    # Compare scaling
    print("\n[5] REYNOLDS NUMBER SCALING:")
    print(f"{'Re':<8} {'Mean Drag':<15} {'Expected Scaling':<20}")
    print("-" * 50)
    base_drag = results[100]['mean_drag']
    for Re in reynolds_numbers:
        if base_drag != 0:
            scaling = results[Re]['mean_drag'] / base_drag
            expected_scaling = Re / 100
            print(f"{Re:<8} {results[Re]['mean_drag']:<15.6f} {scaling:.2f}x (expected {expected_scaling:.2f}x)")
        else:
            print(f"{Re:<8} {results[Re]['mean_drag']:<15.6f} {'N/A (zero forces)'}")

def debug_cfd_solver():
    """Direct debug of CFD solver state"""
    print("\n" + "=" * 70)
    print("TEST 4: CFD SOLVER DEBUG")
    print("=" * 70)
    
    env = PinballEnvBaseline({
        "Re": 100,
        "mesh": "medium",  # Try "coarse" for faster debugging
        "num_substeps": 1,  # Single substep for clarity
        "n_probes": 6,
        "warmup_steps": 0,  # No warmup
        "verbose": True,  # Turn on verbosity
        "save_warmup_plots": False,
        "save_episode_snapshots": False,
    })
    
    print("\n[DEBUG] Checking environment internals...")
    print(f"    Has solver: {hasattr(env, '_solver')}")
    print(f"    Has flow: {hasattr(env, '_flow')}")
    
    if hasattr(env, '_solver'):
        print(f"    Solver type: {type(env._solver)}")
        
        # Try to get force information
        try:
            if hasattr(env._solver, 'get_forces'):
                forces = env._solver.get_forces()
                print(f"    Direct solver forces: {forces}")
            else:
                print("    Solver has no 'get_forces' method")
        except Exception as e:
            print(f"    Error getting forces: {e}")
    
    print("\n[DEBUG] Resetting and checking initial state...")
    obs = env.reset()
    print(f"    Initial observation (probe values): {obs}")
    
    # Try to manually compute forces after one simulation step
    print("\n[DEBUG] Advancing one simulation step manually...")
    try:
        # This depends on your specific solver interface
        if hasattr(env, '_step_simulation'):
            env._step_simulation()
            print("    Manual step successful")
        else:
            print("    No manual step method found")
    except Exception as e:
        print(f"    Error during manual step: {e}")
    
    env.close()

if __name__ == "__main__":
    print("COMPREHENSIVE DRAG/LIFT DIAGNOSTIC SUITE")
    print("=" * 70)
    
    # Run all tests
    test_with_long_warmup()
    test_multiple_steps_with_monitoring()
    test_force_scaling()
    debug_cfd_solver()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)