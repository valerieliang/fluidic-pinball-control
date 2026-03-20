"""
train_pinball.py
----------------
End-to-end PPO training on the HydroGym 2-D fluidic pinball (Re = 100).

Run
  python train_pinball.py [--episodes 500] [--device cpu] [--resume ckpt.pt]

Outputs
  checkpoints/pinball_ep{N}.pt            -- periodic checkpoints
  logs/training_log.csv                   -- per-episode metrics
  checkpoints/pinball_best.pt             -- best drag-reduction policy so far
"""

import sys
import argparse

# Parse our flags NOW, before hydrogym triggers PETSc 
# PETSc scans sys.argv the moment 'import hydrogym' runs. We parse first
# (stdlib only), save values into _args, then hand PETSc a clean sys.argv.
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
import hydrogym

from ppo_agent import PPOAgent, PPOConfig


# Constants (from paper + HDF5 analysis)

BASELINE_DRAG   = 3.60     # mean CD_sum from your pinball_timeseries.h5 baseline
TARGET_DRAG     = 0.36     # 90 % reduction
REWARD_SCALE    = 100.0    # HydroGym internally divides reward by this
EPISODE_LENGTH  = 200      # control actions per episode (≈ 10 shedding periods)
N_SKIP_DT       = 170 * 0.01   # 1.7 s between control actions (Re=100, 2-D)
                                # used for f0 FFT frequency axis

# Probe grid: 6 rows × 8 columns centred on the pinball cluster
_PROBE_X = np.linspace(6.0, 14.0, 8)    # streamwise positions (in cylinder diameters)
_PROBE_Y = np.linspace(-3.0, 3.0, 6)    # transverse positions
PROBE_LOCATIONS = [(float(x), float(y))
                   for x in _PROBE_X for y in _PROBE_Y]   # 48 probes


# Natural frequency (f0) from FFT of a force signal

def compute_f0(signal: np.ndarray, dt_control: float) -> float:
    """
    Estimate the dominant shedding frequency f0 from a drag or lift time series
    collected during one episode.

    Parameters
    ----------
    signal      : 1-D array of a periodic force (e.g. drag_total per step)
    dt_control  : time in seconds between consecutive control actions
                  = N_skip × CFD_dt = 170 × 0.01 = 1.7 s  (Re = 100, 2-D)

    Returns
    -------
    f0 : dominant frequency in dimensionless units  (f × D / U∞, D=U=1 in HydroGym)

    Paper reference values (Table SI4)
      Re = 30  -> f0 ≈ 0.064
      Re = 100 -> f0 ≈ 0.088
      Re = 150 -> f0 ≈ 0.120
    """
    n    = len(signal)
    if n < 8:
        return float("nan")
    # remove DC component before FFT so the mean doesn't dominate
    sig  = signal - signal.mean()
    fft  = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=dt_control)
    # ignore the DC bin (index 0)
    mag  = np.abs(fft[1:])
    f0   = float(freqs[1:][np.argmax(mag)])
    return f0




def make_env(re: int = 100) -> hydrogym.FlowEnv:
    """
    Build a HydroGym FlowEnv for the 2-D fluidic pinball.

    Observation  : (u, v, p) at each probe -> shape (144,)
    Action       : [omega_1, omega_2, omega_3] ∈ [-1, 1]   (surface rotation)
    N_skip       : 170 CFD timesteps / control action       (Re = 100 2-D table)
    """
    cfg = {
        "flow":               hydrogym.Pinball,
        "solver":             hydrogym.MAIA,
        "Re":                 re,
        "observation":        "pressure",       # returns (u, v, p) at probes
        "probe_locations":    PROBE_LOCATIONS,
        # n_skip is set inside config_pinball.yaml; override here if supported:
        # "n_skip": 170,
        "configuration_file": "config_pinball.yaml",
    }
    return hydrogym.FlowEnv(cfg)


def obs_dim_from_env(env) -> int:
    return int(np.prod(env.observation_space.shape))


def act_dim_from_env(env) -> int:
    return int(np.prod(env.action_space.shape))


# Logging helpers

class CSVLogger:
    # Columns match the paper's Table SI4 validation metrics:
    #   f0  -- dominant shedding frequency (dimensionless)
    #   cd  -- time-averaged total drag  (CD1 + CD2 + CD3)
    #   cl2 -- time-averaged lift on rear-upper cylinder
    #   cl3 -- time-averaged lift on rear-lower cylinder
    FIELDS = [
        "episode", "elapsed_s",
        "ep_reward", "cd", "drag_reduction_pct",
        "f0", "cd1", "cd2", "cd3", "cl2", "cl3",
        "policy_loss", "value_loss", "entropy",
    ]

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f      = open(path, "w", newline="")
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

# Rollout -- one full episode

def run_episode(env, agent: PPOAgent, collect: bool = True) -> dict:
    """
    Roll out one episode.

    Parameters
    ----------
    collect : bool
        If True, store transitions in agent.buffer for later PPO update.
        Set False for pure evaluation (no gradient storage).

    Returns
    -------
    dict with per-episode scalars matching Table SI4:
        f0   -- dominant shedding frequency (FFT of drag_total signal)
        cd   -- time-averaged total drag (CD1 + CD2 + CD3)
        cl2  -- time-averaged lift on rear-upper cylinder
        cl3  -- time-averaged lift on rear-lower cylinder
    """
    obs, info = env.reset()
    obs = np.array(obs, dtype=np.float32)

    ep_reward = 0.0
    cd1_list, cd2_list, cd3_list = [], [], []
    cl2_list, cl3_list           = [], []
    drag_total_list               = []   # for f0 FFT

    for step in range(EPISODE_LENGTH):
        action, log_prob, value = agent.select_action(obs)

        obs_next, reward, terminated, truncated, info = env.step(action)
        obs_next = np.array(obs_next, dtype=np.float32)
        done     = terminated or truncated

        # HydroGym populates info with raw per-cylinder CFD quantities
        cd1 = float(info.get("CD1", 0.0))
        cd2 = float(info.get("CD2", 0.0))
        cd3 = float(info.get("CD3", 0.0))
        cl2 = float(info.get("CL2", 0.0))   # rear-upper cylinder (paper Table SI4)
        cl3 = float(info.get("CL3", 0.0))   # rear-lower cylinder

        if collect:
            agent.store(obs, action, log_prob, reward, value, done)

        ep_reward += reward
        cd1_list.append(cd1)
        cd2_list.append(cd2)
        cd3_list.append(cd3)
        cl2_list.append(cl2)
        cl3_list.append(cl3)
        drag_total_list.append(cd1 + cd2 + cd3)

        obs = obs_next
        if done:
            break

    cd           = float(np.mean(drag_total_list))
    drag_red_pct = 100.0 * (1.0 - cd / BASELINE_DRAG)
    f0           = compute_f0(np.array(drag_total_list), dt_control=N_SKIP_DT)

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

# Main training loop 

def train(args):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Building environment ...")
    env = make_env(re=args.re)

    obs_dim = obs_dim_from_env(env)
    act_dim = act_dim_from_env(env)
    print(f"  obs_dim = {obs_dim}   act_dim = {act_dim}")

    cfg = PPOConfig(
        hidden_sizes  = [256, 256],
        log_std_init  = -0.5,
        lr            = 3e-4,
        clip_eps      = 0.2,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        n_epochs      = 10,
        batch_size    = 32,
        value_coef    = 0.5,
        entropy_coef  = 0.01,
        max_grad_norm = 0.5,
        lr_anneal     = True,
        total_episodes= args.episodes,
    )

    agent = PPOAgent(obs_dim, act_dim, cfg, device=args.device)
    if args.resume:
        agent.load(args.resume)

    logger   = CSVLogger("logs/training_log.csv")
    t0       = time.time()
    best_red = -np.inf

    for ep in range(1, args.episodes + 1):

        # collect one episode 
        ep_info = run_episode(env, agent, collect=True)

        # PPO update 
        loss_info = agent.update(ep_info["last_obs"])

        # logging 
        row = {
            "episode":            ep,
            "elapsed_s":          round(time.time() - t0, 1),
            **ep_info,
            **loss_info,
        }
        logger.write(row)
        print_row(ep, row)

        # checkpointing 
        if ep % args.save_every == 0:
            agent.save(f"checkpoints/pinball_ep{ep:04d}.pt")

        if ep_info["drag_reduction_pct"] > best_red:
            best_red = ep_info["drag_reduction_pct"]
            agent.save("checkpoints/pinball_best.pt")
            print(f"  * new best drag reduction: {best_red:.1f}%")

        if best_red >= 90.0:
            print(f"\n[OK] Target reached (>= 90% drag reduction) at episode {ep}.")
            break

    logger.close()
    env.terminate_run()
    print("\nTraining complete.")
    print(f"Best drag reduction: {best_red:.2f}%")
    print(f"Best checkpoint   : checkpoints/pinball_best.pt")


# Entry point

if __name__ == "__main__":
    train(_args)   # _args already parsed before hydrogym import