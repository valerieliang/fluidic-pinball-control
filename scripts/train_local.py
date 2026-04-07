#!/usr/bin/env python
# scripts/train_local.py
"""
Single-node training entrypoint. Thin wrapper around ppo_joint.py
so you can run from the repo root without worrying about module paths.

Usage (from repo root, inside venv-firedrake):
    python scripts/train_local.py --config configs/pinball_re100.yaml
    python scripts/train_local.py --config configs/pinball_re150.yaml --resume checkpoints/hr_ssa_re150/checkpoint_ep00050.pt
    python scripts/train_local.py  # uses PPOConfig defaults (Re=100)
"""

import sys
import os

# Disable CUDA visibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ensure repo root is on sys.path regardless of where script is called from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from training.ppo_joint import PPOConfig, HRSSATrainer
from training.callbacks import default_callbacks


def main():
    parser = argparse.ArgumentParser(description="HR-SSA local training")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config (e.g. configs/pinball_re100.yaml). "
             "If omitted, PPOConfig defaults are used (Re=100, 2D)."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a .pt checkpoint to resume from."
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None,
        help="Weights & Biases project name. If omitted, W&B is disabled."
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total_timesteps from config."
    )
    args = parser.parse_args()

    cfg = PPOConfig.from_yaml(args.config) if args.config else PPOConfig()
    if args.timesteps is not None:
        cfg.total_timesteps = args.timesteps

    print("=" * 60)
    print(f"  HR-SSA Training")
    print(f"  Re={cfg.Re}  mesh={cfg.mesh}  substeps={cfg.num_substeps}")
    print(f"  total_timesteps={cfg.total_timesteps:,}")
    print(f"  run_name={cfg.run_name}")
    print(f"  save_dir={cfg.save_dir}/{cfg.run_name}")
    print("=" * 60)

    callbacks = default_callbacks(
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        wandb_project=args.wandb_project,
        wandb_config=cfg.__dict__,
    )

    trainer = HRSSATrainer(cfg)
    trainer.train(resume_from=args.resume, callbacks=callbacks)


if __name__ == "__main__":
    main()