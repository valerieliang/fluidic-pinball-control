"""
evaluate_drag.py
----------------
Load a trained PPO checkpoint and compute drag reduction metrics.
Also analyses the HDF5 baseline file.

Usage:
  python evaluate_drag.py --hdf5_only --baseline pinball_timeseries.h5
  python evaluate_drag.py --checkpoint checkpoints/pinball_best.pt --episodes 10
"""

import sys
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--checkpoint", type=str,  default=None)
_parser.add_argument("--episodes",   type=int,  default=5)
_parser.add_argument("--baseline",   type=str,  default="pinball_timeseries.h5")
_parser.add_argument("--hdf5_only",  action="store_true")
_parser.add_argument("--plot",       action="store_true")
_parser.add_argument("--device",     type=str,  default="cuda")
_args, _petsc_leftovers = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _petsc_leftovers
del _parser, _petsc_leftovers

import numpy as np
import h5py
import torch

from ppo_agent import PPOAgent, PPOConfig
from train_pinball import (
    make_env, run_episode, get_initial_obs,
    BASELINE_DRAG, EPISODE_LENGTH, OBS_DIM, ACT_DIM,
)


# ---------------------------------------------------------------------------
# HDF5 baseline analysis
# ---------------------------------------------------------------------------

def load_baseline(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        ts   = f["timeseries"]
        t    = ts["time"][:]
        mask = t > 1.0
        cd1  = ts["CD1"][:][mask]
        cd2  = ts["CD2"][:][mask]
        cd3  = ts["CD3"][:][mask]
        cl1  = ts["CL1"][:][mask]
        cl2  = ts["CL2"][:][mask]
        cl3  = ts["CL3"][:][mask]
        drag = (cd1 + cd2 + cd3)
        rwd  = ts["reward"][:][mask]

    return {
        "drag_mean":   float(drag.mean()),
        "drag_std":    float(drag.std()),
        "drag_min":    float(drag.min()),
        "drag_max":    float(drag.max()),
        "cd1_mean":    float(cd1.mean()),
        "cd2_mean":    float(cd2.mean()),
        "cd3_mean":    float(cd3.mean()),
        "cl_sum_mean": float(np.abs(cl1 + cl2 + cl3).mean()),
        "reward_mean": float(rwd.mean()),
        "n_steps":     int(mask.sum()),
    }


def analyse_hdf5(hdf5_path):
    b = load_baseline(hdf5_path)
    bar = "-" * 55
    print("")
    print("%s" % "HDF5 Baseline Analysis".center(55))
    print(bar)
    print("  File             : %s" % hdf5_path)
    print("  Steps analysed   : %s  (t > 1 s)" % "{:,}".format(b["n_steps"]))
    print("  drag_total mean  : %.4f" % b["drag_mean"])
    print("  drag_total std   : %.4f" % b["drag_std"])
    print("  drag_total range : %.4f - %.4f" % (b["drag_min"], b["drag_max"]))
    print("  CD1 / CD2 / CD3  : %.4f / %.4f / %.4f" % (
          b["cd1_mean"], b["cd2_mean"], b["cd3_mean"]))
    print("  |CL_sum| mean    : %.4f" % b["cl_sum_mean"])
    print("  Reward mean      : %.5f" % b["reward_mean"])
    print("  90%% reduction target")
    print("    drag_total <=  %.4f" % (b["drag_mean"] * 0.10))
    print("    reward     >=  %.5f" % (-(b["drag_mean"] * 0.10) / 100.0))
    print(bar)
    return b


# ---------------------------------------------------------------------------
# Drag reduction metrics
# ---------------------------------------------------------------------------

def compute_drag_reduction(controlled_drag, baseline_drag):
    mean_c = float(controlled_drag.mean())
    std_c  = float(controlled_drag.std())
    min_c  = float(controlled_drag.min())
    abs_red      = baseline_drag - mean_c
    pct_red      = 100.0 * abs_red / baseline_drag
    pct_red_std  = 100.0 * std_c   / baseline_drag
    pct_red_peak = 100.0 * (baseline_drag - min_c) / baseline_drag
    return {
        "baseline_drag":      baseline_drag,
        "controlled_drag":    mean_c,
        "drag_std":           std_c,
        "absolute_reduction": abs_red,
        "pct_reduction":      pct_red,
        "pct_reduction_std":  pct_red_std,
        "pct_reduction_peak": pct_red_peak,
        "reward_mean":        -(mean_c) / 100.0,
        "target_met":         pct_red >= 90.0,
    }


def print_drag_report(metrics, baseline, per_cyl=None):
    bar = "-" * 55
    print("")
    print("%s" % "Drag Reduction Report".center(55))
    print(bar)
    print("  Baseline (uncontrolled)")
    print("    drag_total       : %.4f  +/-  %.4f" % (
          baseline["drag_mean"], baseline["drag_std"]))
    print("    CD1 / CD2 / CD3  : %.4f / %.4f / %.4f" % (
          baseline["cd1_mean"], baseline["cd2_mean"], baseline["cd3_mean"]))
    print("    reward           : %.5f" % baseline["reward_mean"])
    print("")
    print("  Controlled (trained PPO)")
    print("    drag_total       : %.4f  +/-  %.4f" % (
          metrics["controlled_drag"], metrics["drag_std"]))
    if per_cyl:
        print("    CD1 / CD2 / CD3  : %.4f / %.4f / %.4f" % (
              per_cyl["cd1"], per_cyl["cd2"], per_cyl["cd3"]))
        print("    CL2 / CL3        : %+.4f / %+.4f" % (
              per_cyl.get("cl2", 0), per_cyl.get("cl3", 0)))
        print("    f0               : %.4f  (uncontrolled ~0.088)" % (
              per_cyl.get("f0", float("nan"))))
    print(bar)
    print("  Drag reduction     : %.2f%%  +/-  %.2f%%" % (
          metrics["pct_reduction"], metrics["pct_reduction_std"]))
    print("  Peak reduction     : %.2f%%" % metrics["pct_reduction_peak"])
    print("  Absolute reduction : %.4f (CD units)" % metrics["absolute_reduction"])
    if metrics["target_met"]:
        print("  TARGET MET (>=90%%, got %.1f%%)" % metrics["pct_reduction"])
    else:
        print("  below 90%% target (got %.1f%%)" % metrics["pct_reduction"])
    print(bar)


# ---------------------------------------------------------------------------
# Live evaluation
# ---------------------------------------------------------------------------

def evaluate_live(checkpoint, n_episodes, baseline, device="cuda"):
    env     = make_env(re=100)
    cfg     = PPOConfig()
    agent   = PPOAgent(OBS_DIM, ACT_DIM, cfg, device=device)
    agent.load(checkpoint)

    all_drag = []
    all_cd1, all_cd2, all_cd3 = [], [], []
    all_cl2, all_cl3, all_f0  = [], [], []

    print("\nEvaluating %d episode(s) ..." % n_episodes)
    obs = get_initial_obs(env)
    for ep in range(1, n_episodes + 1):
        info = run_episode(env, agent, obs, collect=False)
        obs  = info["last_obs"]
        all_drag.append(info["cd"])
        all_cd1.append(info["cd1"])
        all_cd2.append(info["cd2"])
        all_cd3.append(info["cd3"])
        all_cl2.append(info["cl2"])
        all_cl3.append(info["cl3"])
        all_f0.append(info["f0"])
        print("  ep %3d  CD=%.4f  f0=%.4f  CL2=%+.4f  CL3=%+.4f  reduction=%.1f%%" % (
              ep, info["cd"], info["f0"], info["cl2"], info["cl3"],
              info["drag_reduction_pct"]))

    try:
        env.close()
    except AttributeError:
        pass

    metrics = compute_drag_reduction(np.array(all_drag), baseline["drag_mean"])
    per_cyl = {
        "cd1": float(np.mean(all_cd1)),
        "cd2": float(np.mean(all_cd2)),
        "cd3": float(np.mean(all_cd3)),
        "cl2": float(np.mean(all_cl2)),
        "cl3": float(np.mean(all_cl3)),
        "f0":  float(np.nanmean(all_f0)),
    }
    print_drag_report(metrics, baseline, per_cyl)
    return metrics


# ---------------------------------------------------------------------------
# Training curve plot
# ---------------------------------------------------------------------------

def plot_training_log(csv_path="logs/training_log.csv"):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("pandas/matplotlib not installed - skipping plot")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPO Training - 2D Fluidic Pinball Re=100", fontsize=14)

    axes[0, 0].plot(df["episode"], df["drag_reduction_pct"], color="steelblue")
    axes[0, 0].axhline(90, color="red", linestyle="--", label="90% target")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Drag Reduction (%)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(df["episode"], df["ep_reward"], color="darkorange")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Episode Reward")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(df["episode"], df["cd1"], label="CD1")
    axes[1, 0].plot(df["episode"], df["cd2"], label="CD2")
    axes[1, 0].plot(df["episode"], df["cd3"], label="CD3")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Mean Drag Coefficient")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(df["episode"], df["policy_loss"], label="Policy loss")
    axes[1, 1].plot(df["episode"], df["value_loss"],  label="Value loss")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/training_curves.png", dpi=150)
    print("Saved logs/training_curves.png")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _args

    baseline = analyse_hdf5(args.baseline)

    if args.hdf5_only:
        pass

    elif args.checkpoint:
        evaluate_live(args.checkpoint, args.episodes, baseline, args.device)

    else:
        print("\nNo --checkpoint given. Run with --hdf5_only to see baseline stats,")
        print("or pass --checkpoint path/to/pinball_best.pt after training.")

    if args.plot:
        plot_training_log()