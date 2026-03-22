"""
train_pinball.py
----------------
PPO training on the HydroGym 2-D fluidic pinball (Re = 100).
Uses the Firedrake backend.

Key decisions:
  - env.reset() is called ONCE at startup only. The Firedrake FlowEnv
    re-solves a steady-state problem inside reset() which is very slow.
    Between episodes we continue from the last observation.
  - n_steps_per_actuation is set in the env config so Firedrake advances
    N_SKIP CFD steps per env.step() call internally.

Run:
  python train_pinball.py
  python train_pinball.py --episodes 500 --device cuda
  python train_pinball.py --resume checkpoints/pinball_best.pt
"""

import sys
import argparse

# Parse our flags before hydrogym import triggers PETSc.
# PETSc scans sys.argv at import time and warns about unrecognised flags.
# We parse first, save values in _args, then hand PETSc a clean sys.argv.
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
from ppo_agent import PPOAgent, PPOConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BASELINE_DRAG is no longer hardcoded here.
# It is computed empirically at the end of warmup (see get_initial_obs).
# The hardcoded 3.60 was from pinball_timeseries.h5 measured after >500 s
# of simulation, but our 50 s warmup ends at CD_sum ~4.27 — a different
# operating point that miscalibrates every reward and drag-reduction %.
TARGET_DRAG    = 0.36   # 90% reduction target (still relative to computed baseline)

# obs layout from Firedrake FlowEnv: [CL1, CL2, CL3, CD1, CD2, CD3]
OBS_DIM = 6
ACT_DIM = 3   # [omega_1, omega_2, omega_3]

# 170 CFD steps per control action at Re=100 (paper Table SI)
# dt=0.01 s per CFD step, so 1 control action = 1.7 s simulated time
# N_SKIP=10 chosen for Firedrake (CPU FEM solver).
# The paper used N_SKIP=170 for the m-AIA GPU lattice-Boltzmann solver.
# Firedrake is much slower per timestep, so we use fewer substeps per action
# and more actions per episode to still cover ~5 shedding periods.
#
# Shedding period at Re=100: St~0.088 -> period = 1/0.088 ~= 11.4 s
# N_SKIP=10, dt=0.01 -> dt_control=0.10 s -> 114 steps/period
# EPISODE_LENGTH=600 -> ~5.3 shedding periods per episode  (matches paper)
# Total FEM solves per episode: 10 x 600 = 6,000
N_SKIP         = 10
DT_CFD         = 0.01
DT_CONTROL     = N_SKIP * DT_CFD   # 0.10 s, used for f0 FFT
# FIX 1: was 1200 (120 s, ~10 periods), but the paper targets 5 periods.
# Halving episode length doubles how often PPO parameters update for the
# same wall-clock budget, which is strictly better early in training.
EPISODE_LENGTH = 600                # control actions per episode (60 s sim)


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


# Warmup strategy:
#   Phase 1 (WARMUP_STEPS_DEVELOP):  let vortex shedding develop fully.
#     At Re=100, transient lasts ~30-50 s; we use 1500 steps = 150 s to
#     be conservative.  Your previous 500-step warmup ended at CD_sum~4.27,
#     well above the saturated-shedding value — the transient was still live.
#   Phase 2 (WARMUP_STEPS_MEASURE):  record uncontrolled CD_sum to use as
#     BASELINE_DRAG in the reward function.  200 steps = 20 s ~= 1.8 periods,
#     enough to average out one full shedding cycle cleanly.
WARMUP_STEPS_DEVELOP = 1500   # 150 s simulated — develop shedding
WARMUP_STEPS_MEASURE =  200   # 20 s simulated  — measure baseline drag
WARMUP_STEPS = WARMUP_STEPS_DEVELOP + WARMUP_STEPS_MEASURE


def get_initial_obs(env):
    """
    Step the flow through WARMUP_STEPS control actions with zero actuation.

    Phase 1: develop vortex shedding (WARMUP_STEPS_DEVELOP steps).
    Phase 2: measure uncontrolled drag to calibrate the reward function
             (WARMUP_STEPS_MEASURE steps), returning both the final obs
             and the computed baseline CD_sum.
    """
    total = WARMUP_STEPS_DEVELOP + WARMUP_STEPS_MEASURE
    print("  Warming up flow for %d steps (%.0f s simulated) ..." % (
          total, total * DT_CONTROL))

    obs = None
    cd_measure = []

    for i in range(total):
        result = env.step([0.0, 0.0, 0.0])
        obs    = result[0]

        if i >= WARMUP_STEPS_DEVELOP:
            # Phase 2: accumulate drag for baseline estimate
            cd_sum = float(obs[3]) + float(obs[4]) + float(obs[5])
            cd_measure.append(cd_sum)

        if (i + 1) % 200 == 0:
            cd_sum = float(obs[3]) + float(obs[4]) + float(obs[5])
            phase  = "develop" if i < WARMUP_STEPS_DEVELOP else "measure"
            print("    warmup step %4d / %d  [%s]  CD_sum = %.4f" % (
                  i + 1, total, phase, cd_sum))

    baseline = float(np.mean(cd_measure))
    print("  Computed BASELINE_DRAG = %.4f  (mean of last %d steps)" % (
          baseline, WARMUP_STEPS_MEASURE))
    return np.array(obs, dtype=np.float32), baseline


def env_step(env, action):
    """
    Single step advancing N_SKIP CFD substeps.
    Handles both old Gym 4-return and Gymnasium 5-return APIs.
    """
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, _ = result
        done = terminated or truncated
    else:
        obs, reward, done, _ = result
    return np.array(obs, dtype=np.float32), float(reward), bool(done)


def extract_forces(obs):
    """obs = [CL1, CL2, CL3, CD1, CD2, CD3]  (first 6 dims even if augmented)"""
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
# Logging
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
# Rollout
# ---------------------------------------------------------------------------

def run_episode(env, agent, obs, baseline_drag, collect=True):
    """
    Roll out one episode of EPISODE_LENGTH control actions starting from obs.

    obs is passed in rather than obtained from reset() so we never call
    reset() between episodes. The flow state continues from the previous
    episode end point.

    baseline_drag is the empirically measured uncontrolled CD_sum used to
    compute drag_reduction_pct consistently throughout training.

    Returns a dict including "last_obs" for the next episode.
    """
    ep_reward = 0.0
    cd1_list, cd2_list, cd3_list = [], [], []
    cl2_list, cl3_list = [], []
    drag_list = []

    for _ in range(EPISODE_LENGTH):
        action, log_prob, value = agent.select_action(obs)
        obs_next, reward, done  = env_step(env, action.tolist())

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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Building environment ...")
    env = make_env(re=args.re)
    print("  obs_dim = %d   act_dim = %d" % (OBS_DIM, ACT_DIM))
    print("  N_SKIP = %d   DT_control = %.2f s" % (N_SKIP, DT_CONTROL))
    print("  EPISODE_LENGTH = %d   (%.0f s = %.1f shedding periods)" % (
          EPISODE_LENGTH,
          EPISODE_LENGTH * DT_CONTROL,
          EPISODE_LENGTH * DT_CONTROL / 11.4))

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
        # FIX 2: was 0.01 but paper specifies 0.05.  Higher entropy
        # regularisation keeps the policy exploratory for longer, which
        # matters for a reward landscape this flat early in training.
        entropy_coef   = 0.05,
        max_grad_norm  = 0.5,
        lr_anneal      = True,
        total_episodes = args.episodes,
    )

    agent = PPOAgent(OBS_DIM, ACT_DIM, cfg, device=args.device)
    if args.resume:
        agent.load(args.resume)

    print("Initialising flow (one-time setup) ...")
    t_init = time.time()
    # FIX 3: get_initial_obs now returns (obs, baseline_drag) instead of
    # just obs.  baseline_drag replaces the hardcoded 3.60 constant.
    obs, baseline_drag = get_initial_obs(env)
    print("  Ready in %.1fs." % (time.time() - t_init))
    print("  Initial obs: %s" % obs)
    print("  BASELINE_DRAG (empirical) = %.4f" % baseline_drag)

    logger   = CSVLogger("logs/training_log.csv")
    t0       = time.time()
    best_red = -float("inf")

    for ep in range(1, args.episodes + 1):

        ep_info   = run_episode(env, agent, obs, baseline_drag, collect=True)
        obs       = ep_info["last_obs"]
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

        if best_red >= 90.0:
            print("90%% drag reduction reached at episode %d." % ep)
            break

    logger.close()
    try:
        env.close()
    except AttributeError:
        pass

    print("Done. Best reduction: %.2f%%" % best_red)
    print("Best checkpoint: checkpoints/pinball_best.pt")


if __name__ == "__main__":
    train(_args)