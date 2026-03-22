"""
train_pinball.py
----------------
PPO training on the HydroGym 2-D fluidic pinball (Re = 100).
Uses the Firedrake backend with MPI parallelism.

MPI design:
  - ALL ranks run the Firedrake solver (mesh is decomposed across ranks)
  - Only rank 0 has the PyTorch agent, buffer, and logger
  - Rank 0 selects an action and broadcasts it to all other ranks
  - All ranks call env.step() with that action
  - Only rank 0 stores transitions, does PPO updates, logs, and checkpoints
  - best_red is broadcast so all ranks can check the stop condition

Run:
  python train_pinball.py                         # single process
  mpiexec -n 8 python train_pinball.py            # 8 MPI ranks (use nproc)
  mpiexec -n 8 python train_pinball.py --episodes 500
"""

import sys
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--episodes",   type=int, default=500)
_parser.add_argument("--re",         type=int, default=100)
_parser.add_argument("--device",     type=str, default="cuda")
_parser.add_argument("--resume",     type=str, default=None)
_parser.add_argument("--save_every", type=int, default=50)
_args, _petsc_leftovers = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _petsc_leftovers
del _parser, _petsc_leftovers

import os
import csv
import time
import numpy as np

import hydrogym.firedrake as hgym

# MPI setup -- must happen after hydrogym import (which initialises MPI)
try:
    from mpi4py import MPI
    _COMM     = MPI.COMM_WORLD
    _MPI_RANK = _COMM.Get_rank()
    _MPI_SIZE = _COMM.Get_size()
except ImportError:
    _COMM     = None
    _MPI_RANK = 0
    _MPI_SIZE = 1

def is_rank0():
    return _MPI_RANK == 0

def bcast(value):
    """Broadcast value from rank 0 to all other ranks."""
    if _COMM is not None:
        return _COMM.bcast(value, root=0)
    return value

# Only rank 0 imports PyTorch and creates the agent
if is_rank0():
    from ppo_agent import PPOAgent, PPOConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_DRAG  = 3.60
TARGET_DRAG    = 0.36

OBS_DIM = 6   # [CL1, CL2, CL3, CD1, CD2, CD3]
ACT_DIM = 3   # [omega_1, omega_2, omega_3]

# N_SKIP=10 for Firedrake FEM (paper used N_SKIP=170 for GPU LBM solver)
# dt=0.01 s, so each control action = 0.10 s simulated
# EPISODE_LENGTH=600: 60 s simulated = ~5 shedding periods at Re=100
N_SKIP         = 10
DT_CFD         = 0.01
DT_CONTROL     = N_SKIP * DT_CFD
EPISODE_LENGTH = 600

# Warmup: run uncontrolled until vortex shedding is fully developed
# 500 steps x 0.10 s = 50 s simulated (~4 shedding periods)
WARMUP_STEPS   = 500


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def make_env(re=100):
    env_config = {
        "flow":                  hgym.Pinball,
        "flow_config":           {"Re": re},
        "solver":                hgym.SemiImplicitBDF,
        "solver_config":         {"dt": DT_CFD},
        "n_steps_per_actuation": N_SKIP,
    }
    return hgym.FlowEnv(env_config)


def env_step(env, action):
    """Step env, handle both 4-return (old Gym) and 5-return (Gymnasium) APIs."""
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, _ = result
        done = terminated or truncated
    else:
        obs, reward, done, _ = result
    return np.array(obs, dtype=np.float32), float(reward), bool(done)


def extract_forces(obs):
    """obs = [CL1, CL2, CL3, CD1, CD2, CD3]"""
    cd1, cd2, cd3 = float(obs[3]), float(obs[4]), float(obs[5])
    cl1, cl2, cl3 = float(obs[0]), float(obs[1]), float(obs[2])
    return cd1, cd2, cd3, cl1, cl2, cl3


def get_initial_obs(env):
    """
    Warm up the flow with WARMUP_STEPS zero-action steps so vortex shedding
    is fully developed before training starts. All ranks run the solver;
    only rank 0 prints progress.
    """
    if is_rank0():
        print("  Warming up for %d steps (%.0f s simulated) ..." % (
              WARMUP_STEPS, WARMUP_STEPS * DT_CONTROL))
    obs = None
    for i in range(WARMUP_STEPS):
        result = env.step([0.0, 0.0, 0.0])
        obs = result[0]
        if is_rank0() and (i + 1) % 100 == 0:
            cd_sum = float(obs[3]) + float(obs[4]) + float(obs[5])
            print("    warmup step %4d / %d   CD_sum = %.4f" % (
                  i + 1, WARMUP_STEPS, cd_sum))
    return np.array(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# f0 from FFT
# ---------------------------------------------------------------------------

def compute_f0(signal, dt_control):
    if len(signal) < 8:
        return float("nan")
    sig   = signal - signal.mean()
    freqs = np.fft.rfftfreq(len(signal), d=dt_control)
    mag   = np.abs(np.fft.rfft(sig))
    return float(freqs[1:][np.argmax(mag[1:])])


# ---------------------------------------------------------------------------
# Logging (rank 0 only)
# ---------------------------------------------------------------------------

class CSVLogger:
    FIELDS = [
        "episode", "elapsed_s",
        "ep_reward", "cd", "drag_reduction_pct",
        "f0", "cd1", "cd2", "cd3", "cl2", "cl3",
        "policy_loss", "value_loss", "entropy",
    ]

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f      = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def write(self, row):
        self._writer.writerow({k: row.get(k, "") for k in self.FIELDS})
        self._f.flush()

    def close(self):
        self._f.close()


def print_row(ep, row):
    print(
        "ep %4d | reward %+.4f | CD %.4f | f0 %.4f | "
        "CL2 %+.4f  CL3 %+.4f | reduction %5.1f%% | "
        "pi_loss %+.4f | H %.4f" % (
            ep,
            row["ep_reward"],
            row["cd"],
            row["f0"],
            row["cl2"],
            row["cl3"],
            row["drag_reduction_pct"],
            row["policy_loss"],
            row["entropy"],
        )
    )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_episode(env, agent, obs, collect=True):
    """
    Roll out one episode.

    MPI behaviour:
      - Rank 0 selects the action from the policy and broadcasts it
      - All ranks call env.step() with that action (Firedrake parallelism)
      - Only rank 0 stores transitions in the buffer
      - Only rank 0 accumulates stats (other ranks return None)

    Returns a dict on rank 0, None on other ranks.
    """
    if is_rank0():
        ep_reward = 0.0
        cd1_list, cd2_list, cd3_list = [], [], []
        cl2_list, cl3_list = [], []
        drag_list = []

    for _ in range(EPISODE_LENGTH):
        # Rank 0 selects action, all ranks receive it
        if is_rank0():
            action, log_prob, value = agent.select_action(obs)
            action_list = action.tolist()
        else:
            action_list = None
            log_prob    = None
            value       = None
            action      = None

        action_list = bcast(action_list)

        obs_next, reward, done = env_step(env, action_list)

        if is_rank0():
            cd1, cd2, cd3, cl1, cl2, cl3 = extract_forces(obs_next)

            if collect:
                agent.store(obs, action, log_prob, reward, value, done)

            ep_reward += reward
            cd1_list.append(cd1)
            cd2_list.append(cd2)
            cd3_list.append(cd3)
            cl2_list.append(cl2)
            cl3_list.append(cl3)
            drag_list.append(cd1 + cd2 + cd3)

        obs = obs_next

        if done:
            break

    if is_rank0():
        cd           = float(np.mean(drag_list))
        drag_red_pct = 100.0 * (1.0 - cd / BASELINE_DRAG)
        return {
            "last_obs":           obs,
            "ep_reward":          ep_reward,
            "cd":                 cd,
            "drag_reduction_pct": drag_red_pct,
            "f0":                 compute_f0(np.array(drag_list), DT_CONTROL),
            "cd1":                float(np.mean(cd1_list)),
            "cd2":                float(np.mean(cd2_list)),
            "cd3":                float(np.mean(cd3_list)),
            "cl2":                float(np.mean(cl2_list)),
            "cl3":                float(np.mean(cl3_list)),
        }
    else:
        return {"last_obs": obs}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    if is_rank0():
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    if is_rank0():
        print("Building environment ...")
    env = make_env(re=args.re)
    if is_rank0():
        print("  obs_dim = %d   act_dim = %d" % (OBS_DIM, ACT_DIM))
        print("  N_SKIP = %d   DT_control = %.2f s" % (N_SKIP, DT_CONTROL))
        print("  MPI ranks = %d" % _MPI_SIZE)

    if is_rank0():
        cfg = PPOConfig(
            hidden_sizes   = [256, 256],
            log_std_init   = 0.0,
            lr             = 3e-4,
            clip_eps       = 0.2,
            gamma          = 0.99,
            gae_lambda     = 0.95,
            n_epochs       = 10,
            batch_size     = 32,
            value_coef     = 0.5,
            entropy_coef   = 0.05,
            max_grad_norm  = 0.5,
            lr_anneal      = True,
            total_episodes = args.episodes,
        )
        agent = PPOAgent(OBS_DIM, ACT_DIM, cfg, device=args.device)
        if args.resume:
            agent.load(args.resume)
        logger   = CSVLogger("logs/training_log.csv")
        t0       = time.time()
        best_red = -float("inf")
    else:
        agent = None

    if is_rank0():
        print("Initialising flow (one-time setup) ...")
        t_init = time.time()
    obs = get_initial_obs(env)
    if is_rank0():
        print("  Ready in %.1fs. Initial obs: %s" % (time.time() - t_init, obs))

    for ep in range(1, args.episodes + 1):

        ep_info = run_episode(env, agent, obs, collect=True)
        obs     = ep_info["last_obs"]

        if is_rank0():
            loss_info = agent.update(ep_info["last_obs"])

            row = {"episode": ep, "elapsed_s": round(time.time() - t0, 1)}
            row.update(ep_info)
            row.update(loss_info)

            logger.write(row)
            print_row(ep, row)

            if ep % args.save_every == 0:
                agent.save("checkpoints/pinball_ep%04d.pt" % ep)

            if ep_info["drag_reduction_pct"] > best_red:
                best_red = ep_info["drag_reduction_pct"]
                agent.save("checkpoints/pinball_best.pt")
                print("  new best: %.1f%%" % best_red)

        # Broadcast best_red so all ranks can check the stop condition
        best_red = bcast(best_red if is_rank0() else None)

        if best_red >= 90.0:
            if is_rank0():
                print("90%% drag reduction reached at episode %d." % ep)
            break

    if is_rank0():
        logger.close()
        print("Done. Best reduction: %.2f%%" % best_red)
        print("Best checkpoint: checkpoints/pinball_best.pt")

    try:
        env.close()
    except AttributeError:
        pass


if __name__ == "__main__":
    train(_args)