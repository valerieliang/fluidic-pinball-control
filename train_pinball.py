"""
train_pinball.py
----------------
PPO training on the HydroGym 2-D fluidic pinball (Re = 100).
Uses the Firedrake backend with MPI parallelism.

Run:
  python train_pinball.py
  mpiexec -n 8 --bind-to none python train_pinball.py
  mpiexec -n 8 --bind-to none python train_pinball.py --resume checkpoints/pinball_best.pt
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

# MPI setup -- after hydrogym so MPI is already initialised
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
    if _COMM is not None:
        return _COMM.bcast(value, root=0)
    return value

if is_rank0():
    from ppo_agent import PPOAgent, PPOConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BASELINE_DRAG is computed empirically during warmup, not hardcoded.
# This adapts to whatever flow state the solver settles into.
BASELINE_DRAG  = None   # set during warmup
TARGET_DRAG    = None   # set after warmup

# obs = [CL1, CL2, CL3, CD1, CD2, CD3]
# Augmented with last ACTION_HISTORY_LEN actions: obs_dim = 6 + 5*3 = 21
RAW_OBS_DIM        = 6
ACT_DIM            = 3
ACTION_HISTORY_LEN = 5
OBS_DIM            = RAW_OBS_DIM + ACTION_HISTORY_LEN * ACT_DIM   # 21

N_SKIP         = 10
DT_CFD         = 0.01
DT_CONTROL     = N_SKIP * DT_CFD   # 0.10 s
EPISODE_LENGTH = 600               # ~5.3 shedding periods


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(cd, cd_base, cl2, cl3):
    """
    Three-component shaped reward designed to create a gradient toward
    the symmetric counter-rotation strategy.

    Component 1 -- drag reduction (main objective, weight 1.0):
        (cd_base - cd) / cd_base
        Positive when drag is below baseline, proportional to reduction.

    Component 2 -- symmetry penalty (weight 0.5):
        -(cl2 + cl3)^2
        Zero when CL2 = -CL3 (symmetric wake, counter-rotation).
        Negative when wake is asymmetric. Penalises the local optimum
        the agent currently exploits.

    Component 3 -- lift magnitude bonus (weight 0.3):
        (|cl2| + |cl3|) / 2
        Encourages strong cylinder rotation. Both the asymmetric and
        symmetric strategies generate lift, but this bonus combined with
        the symmetry penalty steers toward balanced lift (counter-rotation).
    """
    r_drag = 1.0 * (cd_base - cd) / cd_base
    r_sym  = -0.5 * (cl2 + cl3) ** 2
    r_mag  = 0.3  * (abs(cl2) + abs(cl3)) / 2.0
    return r_drag + r_sym + r_mag


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


def build_obs(raw_obs, action_history):
    """Concatenate raw obs with flattened action history."""
    history_flat = np.array(action_history, dtype=np.float32).flatten()
    return np.concatenate([raw_obs, history_flat])


# ---------------------------------------------------------------------------
# Warmup: develop flow and compute empirical baseline
# ---------------------------------------------------------------------------

# Two-phase warmup:
#   WARMUP_DEVELOP steps: let vortex shedding fully develop (no recording)
#   WARMUP_MEASURE steps: record drag to compute empirical BASELINE_DRAG
WARMUP_DEVELOP = 1500
WARMUP_MEASURE = 200


def get_initial_obs(env):
    """
    Run uncontrolled flow until shedding is developed, then measure
    the baseline drag. Returns raw 6-dim obs.
    """
    global BASELINE_DRAG, TARGET_DRAG

    total = WARMUP_DEVELOP + WARMUP_MEASURE
    if is_rank0():
        print("  Warming up for %d steps (%.0f s simulated) ..." % (
              total, total * DT_CONTROL))

    obs = None
    drag_measurements = []

    for i in range(total):
        result = env.step([0.0, 0.0, 0.0])
        obs = result[0]

        phase = "develop" if i < WARMUP_DEVELOP else "measure"

        if i >= WARMUP_DEVELOP:
            cd_sum = float(obs[3]) + float(obs[4]) + float(obs[5])
            drag_measurements.append(cd_sum)

        if is_rank0() and (i + 1) % 200 == 0:
            cd_sum = float(obs[3]) + float(obs[4]) + float(obs[5])
            print("    warmup step %4d / %d  [%s]  CD_sum = %.4f" % (
                  i + 1, total, phase, cd_sum))

    if is_rank0():
        BASELINE_DRAG = float(np.mean(drag_measurements))
        TARGET_DRAG   = BASELINE_DRAG * 0.10
        print("  Computed BASELINE_DRAG = %.4f  (mean of last %d steps)" % (
              BASELINE_DRAG, WARMUP_MEASURE))

    BASELINE_DRAG = bcast(BASELINE_DRAG)
    TARGET_DRAG   = bcast(TARGET_DRAG)

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
# Logging
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
        "ep %4d | reward %+.4f | CD %.4f (base %.4f) | f0 %.4f | "
        "CL2 %+.4f  CL3 %+.4f | reduction %5.1f%% | "
        "pi_loss %+.4f | H %.4f" % (
            ep,
            row["ep_reward"],
            row["cd"],
            BASELINE_DRAG,
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

def run_episode(env, agent, raw_obs, collect=True):
    """
    Roll out one episode starting from raw_obs (6-dim).

    Action history is reset to zeros at the start of each episode.
    Augmented obs (21-dim) is built internally and never leaked out --
    raw_obs_next (6-dim) is returned as "last_obs" for the next episode,
    and obs_next (21-dim) is returned as "last_obs_aug" for agent.update().

    MPI: rank 0 selects actions and accumulates stats.
         All ranks run env.step() in parallel (Firedrake mesh decomposition).
    """
    zero_action    = [0.0] * ACT_DIM
    action_history = [zero_action[:] for _ in range(ACTION_HISTORY_LEN)]
    obs            = build_obs(raw_obs, action_history)

    if is_rank0():
        ep_reward = 0.0
        cd1_list, cd2_list, cd3_list = [], [], []
        cl2_list, cl3_list = [], []
        drag_list = []

    raw_obs_next = raw_obs
    obs_next     = obs

    for _ in range(EPISODE_LENGTH):
        if is_rank0():
            action, log_prob, value = agent.select_action(obs)
            action_list = action.tolist()
        else:
            action_list = None
            log_prob    = None
            value       = None
            action      = None

        action_list = bcast(action_list)

        raw_obs_next, _env_reward, done = env_step(env, action_list)

        action_history = action_history[1:] + [action_list]
        obs_next       = build_obs(raw_obs_next, action_history)

        if is_rank0():
            cd1, cd2, cd3, cl1, cl2, cl3 = extract_forces(raw_obs_next)

            reward = compute_reward(cd1 + cd2 + cd3, BASELINE_DRAG, cl2, cl3)

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
            "last_obs":     raw_obs_next,   # 6-dim for next episode
            "last_obs_aug": obs_next,       # 21-dim for agent.update()
            "ep_reward":    ep_reward,
            "cd":           cd,
            "drag_reduction_pct": drag_red_pct,
            "f0":           compute_f0(np.array(drag_list), DT_CONTROL),
            "cd1":          float(np.mean(cd1_list)),
            "cd2":          float(np.mean(cd2_list)),
            "cd3":          float(np.mean(cd3_list)),
            "cl2":          float(np.mean(cl2_list)),
            "cl3":          float(np.mean(cl3_list)),
        }
    else:
        return {"last_obs": raw_obs_next}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    if is_rank0():
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        print("Building environment ...")

    env = make_env(re=args.re)

    if is_rank0():
        print("  obs_dim = %d (raw %d + %d actions x %d history)" % (
              OBS_DIM, RAW_OBS_DIM, ACT_DIM, ACTION_HISTORY_LEN))
        print("  act_dim = %d" % ACT_DIM)
        print("  N_SKIP = %d   DT_control = %.2f s" % (N_SKIP, DT_CONTROL))
        print("  EPISODE_LENGTH = %d   (%.0f s = %.1f shedding periods)" % (
              EPISODE_LENGTH, EPISODE_LENGTH * DT_CONTROL,
              EPISODE_LENGTH * DT_CONTROL / (1.0 / 0.088)))
        print("  MPI ranks = %d" % _MPI_SIZE)

    if is_rank0():
        cfg = PPOConfig(
            hidden_sizes   = [256, 256],
            log_std_init   = 0.0,       # higher initial exploration
            lr             = 3e-4,
            clip_eps       = 0.2,
            gamma          = 0.99,
            gae_lambda     = 0.95,
            n_epochs       = 10,
            batch_size     = 32,
            value_coef     = 0.5,
            entropy_coef   = 0.05,      # strong entropy to prevent premature convergence
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
    raw_obs = get_initial_obs(env)
    if is_rank0():
        print("  Ready in %.1fs." % (time.time() - t_init))
        print("  Initial obs: %s" % raw_obs)
        print("  BASELINE_DRAG (empirical) = %.4f" % BASELINE_DRAG)

    for ep in range(1, args.episodes + 1):

        ep_info = run_episode(env, agent, raw_obs, collect=True)
        raw_obs = ep_info["last_obs"]

        if is_rank0():
            loss_info = agent.update(ep_info["last_obs_aug"])

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