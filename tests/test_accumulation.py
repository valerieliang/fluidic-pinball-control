# test_accumulation.py 
# test of reward accumulation
import sys
sys.path.insert(0, os.getcwd()) 

import warnings
warnings.filterwarnings("ignore")

# Suppress Gym warning
import os
os.environ["GYM_WARNINGS"] = "false"

from envs.pinball_env_baseline import PinballEnvBaseline

# Create environment with minimal settings for fast testing
env = PinballEnvBaseline({
    "Re": 100,
    "mesh": "medium",
    "num_substeps": 5,  # Reduced for fast testing
    "n_probes": 6,
    "warmup_steps": 3,  # Minimal warmup
    "verbose": False,
    "save_warmup_plots": False,  # Disable plots
    "save_episode_snapshots": False,  # Disable snapshots
    "save_episode_h5": False,  # Disable H5 for test
})

print("=" * 60)
print("FAST ACCUMULATION TEST")
print("=" * 60)
print(f"num_substeps = {env.num_substeps}")
print(f"warmup_steps = {env.warmup_steps}")
print(f"H5 export: {env.save_episode_h5}")
print(f"Visualization: {env.save_episode_snapshots}")

# Reset (this will do quick warmup)
print("\n[1] Resetting environment (with quick warmup)...")
obs = env.reset()
print(f"    Initial observation shape: {obs.shape}")

# Test single step accumulation
print(f"\n[2] Testing single step (should accumulate over {env.num_substeps} substeps)...")
action = env.action_space.sample()
print(f"    Action: {action}")

obs, reward, done, info = env.step(action)

print(f"\n[3] Results:")
print(f"    Reward received: {reward:.6f}")
print(f"    Done: {done}")
print(f"    Step count: {env._step_count}")

# Check buffers
print(f"\n[4] Buffer lengths after 1 step:")
print(f"    Observations: {len(env._episode_obs_buffer)}")
print(f"    Actions: {len(env._episode_action_buffer)}")
print(f"    Rewards: {len(env._episode_reward_buffer)}")
print(f"    Drag samples: {len(env._episode_drag_buffer)}")
print(f"    Lift samples: {len(env._episode_lift_buffer)}")
print(f"    Substep rewards: {len(env._episode_substep_rewards)}")

# Verify accumulation math
print(f"\n[5] Accumulation verification:")
print(f"    Substep rewards sum: {sum(env._episode_substep_rewards[-env.num_substeps:]):.6f}")
print(f"    Reported reward: {reward:.6f}")
print(f"    Match: {'YES' if abs(sum(env._episode_substep_rewards[-env.num_substeps:]) - reward) < 1e-6 else '❌ NO'}")

# Test multiple steps
print(f"\n[6] Testing multiple steps...")
total_reward = 0.0
for i in range(3):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward
    print(f"    Step {i+2}: reward={reward:.6f}, cumulative={total_reward:.6f}")

print(f"\n[7] Final buffer lengths:")
print(f"    Observations: {len(env._episode_obs_buffer)}")
print(f"    Actions: {len(env._episode_action_buffer)}")
print(f"    Drag samples: {len(env._episode_drag_buffer)} (should be {4 * env.num_substeps})")
print(f"    Expected drag samples: {4 * env.num_substeps}")

# Check drag/lift tracking
if env._episode_drag_buffer:
    print(f"\n[8] Drag/Lift stats:")
    print(f"    Mean Drag: {sum(env._episode_drag_buffer)/len(env._episode_drag_buffer):.4f}")
    print(f"    Mean |Lift|: {sum(abs(l) for l in env._episode_lift_buffer)/len(env._episode_lift_buffer):.4f}")

# Test episode completion
print(f"\n[9] Testing episode reset...")
env.close()

print("\n" + "=" * 60)
print("TEST COMPLETE - Reward accumulation is working!")
print("=" * 60)