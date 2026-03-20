"""
evaluate_drag.py
----------------
Load a trained PPO checkpoint, run N evaluation episodes, and compute
drag reduction metrics — including comparisons against your HDF5 baseline.

Usage
  # Against live HydroGym environment
  python evaluate_drag.py --checkpoint checkpoints/pinball_best.pt --episodes 10

  # Against the stored HDF5 baseline only (no environment needed)
  python evaluate_drag.py --hdf5_only --baseline pinball_timeseries.h5
"""

import argparse
import numpy as np
import h5py
import torch

from ppo_agent import PPOAgent, PPOConfig
from train_pinball import (
    make_env, obs_dim_from_env, act_dim_from_env,
    run_episode, BASELINE_DRAG, EPISODE_LENGTH, PROBE_LOCATIONS
)


# Load baseline statistics from HDF5

def load_baseline(hdf5_path: str) -> dict:
    """
    Parse the pinball_timeseries.h5 file you already have.

    Returns a dict of scalars that characterise the UNCONTROLLED flow,
    skipping the initialisation spike (t < 1 s).
    """
    with h5py.File(hdf5_path, "r") as f:
        ts   = f["timeseries"]
        t    = ts["time"][:]
        mask = t > 1.0                      # skip CFD start-up spike

        cd1      = ts["CD1"][:][mask]
        cd2      = ts["CD2"][:][mask]
        cd3      = ts["CD3"][:][mask]
        cl1      = ts["CL1"][:][mask]
        cl2      = ts["CL2"][:][mask]
        cl3      = ts["CL3"][:][mask]
        drag     = ts["drag_total"][:][mask]
        reward   = ts["reward"][:][mask]

    drag_total  = cd1 + cd2 + cd3          # re-sum to match per-cylinder granularity
    cl_sum_abs  = np.abs(cl1 + cl2 + cl3)

    baseline = {
        "drag_mean":     float(drag_total.mean()),
        "drag_std":      float(drag_total.std()),
        "drag_min":      float(drag_total.min()),
        "drag_max":      float(drag_total.max()),
        "cd1_mean":      float(cd1.mean()),
        "cd2_mean":      float(cd2.mean()),
        "cd3_mean":      float(cd3.mean()),
        "cl_sum_mean":   float(cl_sum_abs.mean()),
        "reward_mean":   float(reward.mean()),
        "n_steps":       int(mask.sum()),
    }
    return baseline


# Drag reduction metrics

def compute_drag_reduction(controlled_drag: np.ndarray,
                           baseline_drag:   float) -> dict:
    """
    All standard drag reduction metrics used in the HydroGym paper.

    Parameters
    ----------
    controlled_drag : 1-D array of drag_total values during controlled episodes
    baseline_drag   : scalar mean drag from the uncontrolled baseline

    Returns
    -------
    dict of floats, all in consistent units (drag is dimensionless CD)
    """
    mean_c  = float(controlled_drag.mean())
    std_c   = float(controlled_drag.std())
    min_c   = float(controlled_drag.min())

    abs_red        = baseline_drag - mean_c            # absolute reduction
    pct_red        = 100.0 * abs_red / baseline_drag   # percentage reduction
    pct_red_std    = 100.0 * std_c   / baseline_drag   # ± variability

    # Peak (best single-step) drag reduction
    pct_red_peak   = 100.0 * (baseline_drag - min_c) / baseline_drag

    # Reward at convergence  (paper formula, /100 normalisation)
    reward_mean    = -(mean_c) / 100.0

    return {
        "baseline_drag":      baseline_drag,
        "controlled_drag":    mean_c,
        "drag_std":           std_c,
        "absolute_reduction": abs_red,
        "pct_reduction":      pct_red,
        "pct_reduction_std":  pct_red_std,
        "pct_reduction_peak": pct_red_peak,
        "reward_mean":        reward_mean,
        "target_met":         pct_red >= 90.0,
    }


def print_drag_report(metrics: dict, baseline: dict, controlled_per_cyl: dict):
    """Pretty-print the full drag reduction report."""
    bar = "─" * 55

    print(f"\n{'Drag Reduction Report':^55}")
    print(bar)

    print(f"  Baseline (uncontrolled)")
    print(f"    drag_total       : {baseline['drag_mean']:.4f}  ±  {baseline['drag_std']:.4f}")
    print(f"    CD1 / CD2 / CD3  : {baseline['cd1_mean']:.4f} / "
          f"{baseline['cd2_mean']:.4f} / {baseline['cd3_mean']:.4f}")
    print(f"    |CL_sum|         : {baseline['cl_sum_mean']:.4f}")
    print(f"    reward           : {baseline['reward_mean']:.5f}")

    print(f"\n  Controlled (trained PPO)")
    print(f"    drag_total       : {metrics['controlled_drag']:.4f}  ±  {metrics['drag_std']:.4f}")
    if controlled_per_cyl:
        print(f"    CD1 / CD2 / CD3  : {controlled_per_cyl['cd1']:.4f} / "
              f"{controlled_per_cyl['cd2']:.4f} / {controlled_per_cyl['cd3']:.4f}")

    print(bar)
    print(f"  Drag reduction     : {metrics['pct_reduction']:.2f}%  ±  "
          f"{metrics['pct_reduction_std']:.2f}%")
    print(f"  Peak reduction     : {metrics['pct_reduction_peak']:.2f}%")
    print(f"  Absolute reduction : {metrics['absolute_reduction']:.4f} (CD units)")
    print(f"  Reward (controlled): {metrics['reward_mean']:.5f}")
    status = "✓  TARGET MET" if metrics["target_met"] else "✗  below 90% target"
    print(f"\n  {status}  (need ≥ 90%, got {metrics['pct_reduction']:.1f}%)")
    print(bar)


# Live evaluation against HydroGym

def evaluate_live(checkpoint: str, n_episodes: int,
                  baseline: dict, device: str = "cpu"):
    """Roll out the trained policy and collect drag statistics."""

    # rebuild env just for dims
    env     = make_env(re=100)
    obs_dim = obs_dim_from_env(env)
    act_dim = act_dim_from_env(env)

    cfg   = PPOConfig()                    # defaults match training
    agent = PPOAgent(obs_dim, act_dim, cfg, device=device)
    agent.load(checkpoint)

    all_drag = []
    all_cd1  = []
    all_cd2  = []
    all_cd3  = []
    all_red  = []

    print(f"\nEvaluating {n_episodes} episode(s) …")
    for ep in range(1, n_episodes + 1):
        info = run_episode(env, agent, collect=False)
        all_drag.append(info["mean_drag"])
        all_cd1.append(info["cd1"])
        all_cd2.append(info["cd2"])
        all_cd3.append(info["cd3"])
        all_red.append(info["drag_reduction_pct"])
        print(f"  ep {ep:3d}  drag={info['mean_drag']:.4f}  "
              f"reduction={info['drag_reduction_pct']:.1f}%")

    env.terminate_run()

    controlled_drag = np.array(all_drag)
    metrics = compute_drag_reduction(controlled_drag, baseline["drag_mean"])

    controlled_per_cyl = {
        "cd1": float(np.mean(all_cd1)),
        "cd2": float(np.mean(all_cd2)),
        "cd3": float(np.mean(all_cd3)),
    }

    print_drag_report(metrics, baseline, controlled_per_cyl)
    return metrics


# HDF5-only analysis (no environment)

def analyse_hdf5(hdf5_path: str):
    """
    Full drag statistics from the stored baseline.
    Use this to verify the baseline before running live evaluation.
    """
    baseline = load_baseline(hdf5_path)

    print(f"\n{'HDF5 Baseline Analysis':^55}")
    print("─" * 55)
    print(f"  File             : {hdf5_path}")
    print(f"  Steps analysed   : {baseline['n_steps']:,}  (t > 1 s)")
    print(f"  drag_total mean  : {baseline['drag_mean']:.4f}")
    print(f"  drag_total std   : {baseline['drag_std']:.4f}")
    print(f"  drag_total range : {baseline['drag_min']:.4f} – {baseline['drag_max']:.4f}")
    print(f"  CD1 / CD2 / CD3  : {baseline['cd1_mean']:.4f} / "
          f"{baseline['cd2_mean']:.4f} / {baseline['cd3_mean']:.4f}")
    print(f"  |CL_sum| mean    : {baseline['cl_sum_mean']:.4f}")
    print(f"  Reward mean      : {baseline['reward_mean']:.5f}")

    print(f"\n  90% reduction target")
    target_drag   = baseline["drag_mean"] * 0.10
    target_reward = -target_drag / 100.0
    print(f"    drag_total ≤  {target_drag:.4f}")
    print(f"    reward     ≥  {target_reward:.5f}")
    print("─" * 55)

    return baseline


# Plot from training log

def plot_training_log(csv_path: str = "logs/training_log.csv"):
    """Quick matplotlib plot of drag reduction over training episodes."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("pandas/matplotlib not installed — skipping plot")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPO Training — 2-D Fluidic Pinball Re=100", fontsize=14)

    # Drag reduction
    axes[0, 0].plot(df["episode"], df["drag_reduction_pct"], color="steelblue")
    axes[0, 0].axhline(90, color="red", linestyle="--", label="90% target")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Drag Reduction (%)")
    axes[0, 0].set_title("Drag Reduction")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Reward
    axes[0, 1].plot(df["episode"], df["ep_reward"], color="darkorange")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Episode Reward")
    axes[0, 1].set_title("PPO Training Reward")
    axes[0, 1].grid(alpha=0.3)

    # Per-cylinder drag
    axes[1, 0].plot(df["episode"], df["cd1"], label="CD1 (front)")
    axes[1, 0].plot(df["episode"], df["cd2"], label="CD2 (rear-upper)")
    axes[1, 0].plot(df["episode"], df["cd3"], label="CD3 (rear-lower)")
    axes[1, 0].axhline(0.97 * 0.1, color="gray", linestyle=":", label="CD1 target")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Mean Drag Coefficient")
    axes[1, 0].set_title("Per-Cylinder Drag")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    # Policy and value losses
    axes[1, 1].plot(df["episode"], df["policy_loss"], label="Policy loss")
    axes[1, 1].plot(df["episode"], df["value_loss"],  label="Value loss")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("PPO Losses")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out = "logs/training_curves.png"
    plt.savefig(out, dpi=150)
    print(f"Training curves saved → {out}")
    plt.show()


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to trained .pt checkpoint for live evaluation")
    parser.add_argument("--episodes",   type=int, default=5,
                        help="number of eval episodes (live mode)")
    parser.add_argument("--baseline",   type=str, default="pinball_timeseries.h5",
                        help="path to HDF5 baseline file")
    parser.add_argument("--hdf5_only",  action="store_true",
                        help="only analyse the HDF5 file, do not run live evaluation")
    parser.add_argument("--plot",       action="store_true",
                        help="plot training log from logs/training_log.csv")
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    # always analyse baseline first
    baseline = analyse_hdf5(args.baseline)

    if args.hdf5_only:
        pass  # done

    elif args.checkpoint:
        evaluate_live(args.checkpoint, args.episodes, baseline, args.device)

    else:
        print("\nNo --checkpoint given. Run with --hdf5_only to just see baseline stats,")
        print("or pass --checkpoint path/to/pinball_best.pt after training.")

    if args.plot:
        plot_training_log()