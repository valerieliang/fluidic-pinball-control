"""
train_pinball.py
----------------
PPO training on the HydroGym 2-D fluidic pinball (Re = 100).
Uses the Firedrake backend, matching the simulation script exactly.

Run
  python train_pinball.py [--episodes 500] [--device cuda] [--resume ckpt.pt]

Outputs
  checkpoints/pinball_ep{N}.pt   -- periodic checkpoints
  checkpoints/pinball_best.pt    -- best policy so far
  logs/training_log.csv          -- per-episode metrics
"""

import sys
import argparse

# Parse our flags BEFORE hydrogym import (which triggers PETSc).
# PETSc scans sys.argv at import time and warns about unrecognised flags.
# We parse first (stdlib only), store values in _args, then give PETSc
# a clean sys.argv containing only its own flags.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--episodes",   type=int,  default=500)
_parser.add_argument("--re",         type=int,  default=100)
_parser.add_argument("--device",     type=str,  default="cuda")
_parser.add_argument("--resume",     type=str,  default=None)
_parser.add_argument("--save_every", type=int,  default=50)
_args, _petsc_leftovers = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _petsc_leftovers
del _parser, _petsc_leftovers

import os
import csv
import time
import numpy as np
import hydrogym.firedrake as hgym

from ppo_agent import PPOAgent, PPOConfig


# Constants  (from paper + your HDF5 baseline)

BASELINE_DRAG  = 3.60    # mean CD_sum from pinball_timeseries.h5
TARGET_DRAG    = 0.36    # 90% reduction target

# Observation layout returned by Firedrake FlowEnv (same as your sim script):
#   obs = [CL1, CL2, CL3, CD1, CD2, CD3]
#   indices: 0    1    2    3    4    5
OBS_DIM = 6
ACT_DIM = 3   # [omega_1, omega_2, omega_3] cylinder rotation rates

# RL episode timing
# Paper (Table SI, Re=100 2D): 170 CFD steps per control action
# Each CFD step = dt = 0.01 s  -> 1 control action = 1.7 s simulated time
N_SKIP         = 170     # CFD substeps per RL action
DT_CFD         = 0.01   # must match solver_config below
DT_CONTROL     = N_SKIP * DT_CFD   # 1.7 s -- used for f0 FFT axis
EPISODE_LENGTH = 200     # control actions per episode (about 10 shedding periods)


# Environment

def make_env(re: int = 100) -> hgym.FlowEnv:
    """
    Build a FlowEnv using the exact same config as the simulation script.

    Observation : [CL1, CL2, CL3, CD1, CD2, CD3]  -- shape (6,)
    Action      : [omega_1, omega_2, omega_3]       -- cylinder rotation rates
    """
    env_config = {
        "flow":          hgym.Pinball,
        "flow_config":   {"Re": re},
        "solver":        hgym.SemiImplicitBDF,
        "solver_config": {"dt": DT_CFD},
    }
    return hgym.FlowEnv(env_config)


def env_reset(env):
    """
    Reset the environment, handling both old Gym (returns obs)
    and new Gymnasium (returns obs, info) APIs.
    """
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result[0], result[1]
    else:
        obs, info = result, {}
    return np.array(obs, dtype=np.float32), info


def env_step(env, action):
    """
    Step the environment once (one CFD timestep), handling both
    old Gym 4-return (obs, reward, done, info) and
    new Gymnasium 5-return (obs, reward, terminated, truncated, info).
    """
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:                         # old Gym API -- matches your simulation script
        obs, reward, done, info = result
    return np.array(obs, dtype=np.float32), float(reward), bool(done), info


def extract_forces(obs: np.ndarray):
    """
    Unpack per-cylinder forces from the observation vector.
    obs = [CL1, CL2, CL3, CD1, CD2, CD3]
    """
    cl1, cl2, cl3 = float(obs[0]), float(obs[1]), float(obs[2])
    cd1, cd2, cd3 = float(obs[3]), float(obs[4]), float(obs[5])
    return cd1, cd2, cd3, cl1, cl2, cl3


# f0 from FFT

def compute_f0(signal: np.ndarray, dt_control: float) -> float:
    """
    Dominant shedding frequency from the drag time series.
    Paper Table SI4 reference values: Re=30->0.064, Re=100->0.088, Re=150->0.120
    """
    if len(signal) < 8:
        return float("nan")
    sig = signal - signal.mean()
    fft = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(signal), d=dt_control)
    return float(freqs[1:][np.argmax(np.abs(fft[1:]))])


# Logging

class CSVLogger:
    FIELDS = [
        "episode", "elapsed_s",
        "ep_reward", "cd", "drag_reduction_pct",
        "f0", "cd1", "cd2", "cd3", "cl2", "cl3",
        "policy_loss", "value_loss", "entropy",
    ]

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def write(self, row: dict):
        self._writer.writerow({k: row.get(k, "") for k in self.FIELDS})
        self._f.flush()

    def close(self):
        self._f.close()


def print_row(ep: int, row: dict):
    print(
        f"ep {ep:4d} | "
        f"reward {row['ep_reward']:+.4f} | "
        f"CD {row['cd']:.4f} | "
        f"f0 {row['f0']:.4f} | "
        f"CL2 {row['cl2']:+.4f}  CL3 {row['cl3']:+.4f} | "
        f"reduction {row['drag_reduction_pct']:5.1f}% | "
        f"pi_loss {row['policy_loss']:+.4f} | "
        f"H {row['entropy']:.4f}"
    )


# Rollout -- one episode

def run_episode(env, agent: PPOAgent, collect: bool = True) -> dict:
    """
    Roll out one episode of EPISODE_LENGTH control actions.

    Each control action = N_SKIP CFD substeps.
    Forces are read directly from obs: [CL1, CL2, CL3, CD1, CD2, CD3].

    collect=True  : store transitions for PPO update
    collect=False : evaluation only, no buffer writes
    """
    obs, _ = env_reset(env)

    ep_reward = 0.0
    cd1_list, cd2_list, cd3_list = [], [], []
    cl2_list, cl3_list = [], []
    drag_list = []

    for step in range(EPISODE_LENGTH):
        action, log_prob, value = agent.select_action(obs)

        # N_SKIP CFD substeps with the same action
        step_reward = 0.0
        for _ in range(N_SKIP):
            obs_next, r, done, info = env_step(env, action.tolist())
            step_reward += r
            if done:
                break

        # read forces from the final observation of this substep block
        cd1, cd2, cd3, cl1, cl2, cl3 = extract_forces(obs_next)

        if collect:
            agent.store(obs, action, log_prob, step_reward, value, done)

        ep_reward += step_reward
        cd1_list.append(cd1); cd2_list.append(cd2); cd3_list.append(cd3)
        cl2_list.append(cl2); cl3_list.append(cl3)
        drag_list.append(cd1 + cd2 + cd3)

        obs = obs_next
        if done:
            break

    cd = float(np.mean(drag_list))
    drag_red_pct = 100.0 * (1.0 - cd / BASELINE_DRAG)
    f0 = compute_f0(np.array(drag_list), DT_CONTROL)

    return {
        "last_obs":           obs,
        "ep_reward":          ep_reward,
        "cd":                 cd,
        "drag_reduction_pct": drag_red_pct,
        "f0":                 f0,
        "cd1":                float(np.mean(cd1_list)),
        "cd2":                float(np.mean(cd2_list)),
        "cd3":                float(np.mean(cd3_list)),
        "cl2":                float(np.mean(cl2_list)),
        "cl3":                float(np.mean(cl3_list)),
    }


# Training loop

def train(args):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Building environment ...")
    env = make_env(re=args.re)
    print(f"  obs_dim = {OBS_DIM}   act_dim = {ACT_DIM}")
    print(f"  N_SKIP  = {N_SKIP}    DT_control = {DT_CONTROL:.2f} s")

    cfg = PPOConfig(
        hidden_sizes   = [256, 256],
        log_std_init   = -0.5,
        lr             = 3e-4,
        clip_eps       = 0.2,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        n_epochs       = 10,
        batch_size     = 32,
        value_coef     = 0.5,
        entropy_coef   = 0.01,
        max_grad_norm  = 0.5,
        lr_anneal      = True,
        total_episodes = args.episodes,
    )

    agent = PPOAgent(OBS_DIM, ACT_DIM, cfg, device=args.device)
    if args.resume:
        agent.load(args.resume)

    logger = CSVLogger("logs/training_log.csv")
    t0 = time.time()
    best_red = -float("inf")

    for ep in range(1, args.episodes + 1):

        ep_info = run_episode(env, agent, collect=True)
        loss_info = agent.update(ep_info["last_obs"])

        row = {
            "episode":   ep,
            "elapsed_s": round(time.time() - t0, 1),
            **ep_info,
            **loss_info,
        }
        logger.write(row)
        print_row(ep, row)

        if ep % args.save_every == 0:
            agent.save(f"checkpoints/pinball_ep{ep:04d}.pt")

        if ep_info["drag_reduction_pct"] > best_red:
            best_red = ep_info["drag_reduction_pct"]
            agent.save("checkpoints/pinball_best.pt")
            print(f"  * new best: {best_red:.1f}%")

        if best_red >= 90.0:
            print(f"\n90% drag reduction reached at episode {ep}.")
            break

    logger.close()
    try:
        env.close()
    except AttributeError:
        pass   # older FlowEnv versions don't have close()
    print(f"\nDone. Best reduction: {best_red:.2f}%")
    print(f"Best checkpoint: checkpoints/pinball_best.pt")


# Entry point

if __name__ == "__main__":
    train(_args)