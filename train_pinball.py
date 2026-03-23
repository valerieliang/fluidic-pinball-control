"""
train_pinball.py
----------------
PPO training on the HydroGym 2-D fluidic pinball (Re = 100).
Uses the Firedrake backend with MPI parallelism.

MPI design
----------
  All ranks participate in every env.step() call — Firedrake's FEM solver
  is a collective MPI operation and will deadlock if any rank skips it.

  Only rank 0 runs the PPO agent (neural network, optimizer, buffer).
  After rank 0 selects an action it broadcasts it to all other ranks so
  that every rank calls env.step() with the same action.

  All printing, logging, and checkpointing is guarded by `if RANK == 0`.

Run:
  mpiexec -n 8 --bind-to none python train_pinball.py
  mpiexec -n 8 --bind-to none python train_pinball.py --episodes 500
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
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

# PPOAgent is only imported and instantiated on rank 0 to avoid every rank
# allocating a GPU model.
if RANK == 0:
    from ppo_agent import PPOAgent, PPOConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBS_DIM = 6
ACT_DIM = 3

N_SKIP         = 10
DT_CFD         = 0.01
DT_CONTROL     = N_SKIP * DT_CFD
EPISODE_LENGTH = 600


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


WARMUP_STEPS_DEVELOP = 1500
WARMUP_STEPS_MEASURE =  200
WARMUP_STEPS = WARMUP_STEPS_DEVELOP + WARMUP_STEPS_MEASURE


def get_initial_obs(env):
    """
    All ranks step through warmup together (env.step is collective).
    Only rank 0 prints progress and accumulates the baseline drag estimate.
    Returns (obs, baseline_drag). baseline_drag is None on non-zero ranks.
    """
    if RANK == 0:
        print("  Warming up flow for %d steps (%.0f s simulated) ..." %
              (WARMUP_STEPS, WARMUP_STEPS * DT_CONTROL))

    obs        = None
    cd_measure = []

    for i in range(WARMUP_STEPS):
        # COLLECTIVE: all ranks must call this together.
        result = env.step([0.0, 0.0, 0.0])
        obs    = result[0]

        if i >= WARMUP_STEPS_DEVELOP:
            cd_sum = float(obs[3]) + float(obs[4]) + float(obs[5])
            cd_measure.append(cd_sum)

        if RANK == 0 and (i + 1) % 200 == 0:
            cd_sum = float(obs[3]) + float(obs[4]) + float(obs[5])
            phase  = "develop" if i < WARMUP_STEPS_DEVELOP else "measure"
            print("    warmup step %4d / %d  [%s]  CD_sum = %.4f" %
                  (i + 1, WARMUP_STEPS, phase, cd_sum))

    baseline = float(np.mean(cd_measure)) if cd_measure else None
    if RANK == 0:
        print("  Computed BASELINE_DRAG = %.4f  (mean of last %d steps)" %
              (baseline, WARMUP_STEPS_MEASURE))

    return np.array(obs, dtype=np.float32), baseline


def env_step(env, action):
    """
    COLLECTIVE: all ranks call env.step with the same action.
    action is a plain Python list — already broadcast by the caller.
    """
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, _ = result
        done = terminated or truncated
    else:
        obs, reward, done, _ = result
    return np.array(obs, dtype=np.float32), float(reward), bool(done)


def extract_forces(obs):
    cd1, cd2, cd3 = float(obs[3]), float(obs[4]), float(obs[5])
    cl1, cl2, cl3 = float(obs[0]), float(obs[1]), float(obs[2])
    return cd1, cd2, cd3, cl1, cl2, cl3


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
# Logging  (rank 0 only)
# ---------------------------------------------------------------------------

class CSVLogger:
    FIELDS = [
        "episode", "elapsed_s",
        "ep_reward", "cd", "drag_reduction_pct",
        "f0", "cd1", "cd2", "cd3", "cl2", "cl3",
        "policy_loss", "value_loss", "entropy",
        "baseline_drag",
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
            row["baseline_drag"],
            row["f0"],
            row["cl2"],
            row["cl3"],
            row["drag_reduction_pct"],
            row["policy_loss"],
            row["entropy"],
        )
    )


# ---------------------------------------------------------------------------
# Rollout  (collective env.step, rank-0-only agent logic)
# ---------------------------------------------------------------------------

def run_episode(env, agent, obs, baseline_drag):
    """
    One episode of EPISODE_LENGTH control steps.

    Control flow per step:
      1. Rank 0 calls agent.select_action(obs)   -> action (numpy array)
      2. Rank 0 broadcasts action to all ranks   (MPI Bcast)
      3. ALL ranks call env.step(action)          (collective FEM solve)
      4. Rank 0 calls agent.store(...)
      5. Rank 0 broadcasts done flag             (MPI Bcast, so all ranks
                                                  exit the loop together)
    """
    ep_reward = 0.0
    cd1_list, cd2_list, cd3_list = [], [], []
    cl2_list, cl3_list = [], []
    drag_list = []

    action_buf = np.zeros(ACT_DIM, dtype=np.float64)
    done_flag  = np.array([0], dtype=np.int32)

    for _ in range(EPISODE_LENGTH):

        # Rank 0: select action.
        if RANK == 0:
            action_np, log_prob, value = agent.select_action(obs)
            action_buf[:] = action_np.astype(np.float64)

        # COLLECTIVE: broadcast action so all ranks advance with the same input.
        COMM.Bcast(action_buf, root=0)

        # COLLECTIVE: advance CFD.
        obs_next, reward, done = env_step(env, action_buf.tolist())
        # --- compute drag-based reward ---
        cd1, cd2, cd3, cl1, cl2, cl3 = extract_forces(obs_next)
        drag = cd1 + cd2 + cd3

        # normalized reward
        reward = (baseline_drag - drag) / baseline_drag

        # Rank 0: record transition.
        if RANK == 0:
            agent.store(obs, action_np, log_prob, reward, value, done)
            ep_reward += reward
            cd1_list.append(cd1);  cd2_list.append(cd2);  cd3_list.append(cd3)
            cl2_list.append(cl2);  cl3_list.append(cl3)
            drag_list.append(cd1 + cd2 + cd3)
            done_flag[0] = int(done)

        obs = obs_next

        # COLLECTIVE: broadcast done so all ranks exit together.
        COMM.Bcast(done_flag, root=0)
        if done_flag[0]:
            break

    if RANK == 0:
        cd           = float(np.mean(drag_list))
        drag_red_pct = 100.0 * (1.0 - cd / baseline_drag)
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
            "baseline_drag":      baseline_drag,
        }
    else:
        return {"last_obs": obs}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    if RANK == 0:
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        print("Building environment ...")

    env = make_env(re=args.re)

    if RANK == 0:
        print("  obs_dim = %d   act_dim = %d" % (OBS_DIM, ACT_DIM))
        print("  N_SKIP = %d   DT_control = %.2f s" % (N_SKIP, DT_CONTROL))
        print("  EPISODE_LENGTH = %d   (%.0f s = %.1f shedding periods)" % (
              EPISODE_LENGTH,
              EPISODE_LENGTH * DT_CONTROL,
              EPISODE_LENGTH * DT_CONTROL / 11.4))
        print("  MPI ranks = %d" % SIZE)

        cfg = PPOConfig(
            hidden_sizes   = [256, 256],
            log_std_init   = -0.5,
            lr             = 1e-4,
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
    else:
        agent = None

    if RANK == 0:
        print("Initialising flow (one-time setup) ...")
        t_init = time.time()

    # COLLECTIVE: all ranks warm up together.
    obs, baseline_drag = get_initial_obs(env)

    if RANK == 0:
        print("  Ready in %.1fs." % (time.time() - t_init))
        print("  Initial obs: %s" % obs)
        print("  BASELINE_DRAG (empirical) = %.4f" % baseline_drag)
        logger   = CSVLogger("logs/training_log.csv")
        t0       = time.time()
        best_red = -float("inf")

    stop = np.array([0], dtype=np.int32)

    for ep in range(1, args.episodes + 1):

        # COLLECTIVE episode rollout.
        ep_info = run_episode(env, agent, obs, baseline_drag)
        obs     = ep_info["last_obs"]

        if RANK == 0:
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

            stop[0] = int(best_red >= 90.0)

        # COLLECTIVE: broadcast stop so all ranks exit together.
        COMM.Bcast(stop, root=0)
        if stop[0]:
            if RANK == 0:
                print("90%% drag reduction reached at episode %d." % ep)
            break

    if RANK == 0:
        logger.close()
        print("Done. Best reduction: %.2f%%" % best_red)
        print("Best checkpoint: checkpoints/pinball_best.pt")

    try:
        env.close()
    except AttributeError:
        pass


if __name__ == "__main__":
    train(_args)