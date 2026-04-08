#!/usr/bin/env python3
# scripts/train_local.py
"""
Single-node MPI launcher for HR-SSA.

Usage
-----
  # 8 single-rank Firedrake sims  (fastest per-step on a medium mesh)
  mpiexec -n 8 --bind-to none python -m scripts.train_local \
      --config configs/pinball_re100.yaml --timesteps 5000

  # 4 two-rank Firedrake sims  (better PETSc scaling for fine meshes)
  mpiexec -n 8 --bind-to none python -m scripts.train_local \
      --config configs/pinball_re100.yaml --timesteps 5000 --ranks-per-env 2

The --ranks-per-env value MUST be set before ppo_joint (and therefore
Firedrake/PETSc) is imported, so it is written to an environment variable
read at module-load time by ppo_joint.py.
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="HR-SSA MPI Training")
    parser.add_argument("--config",         type=str, required=True)
    parser.add_argument("--timesteps",      type=int, default=5000)
    parser.add_argument("--resume",         type=str, default=None)
    parser.add_argument(
        "--ranks-per-env", type=int, default=1,
        help="Number of MPI ranks per Firedrake simulation. "
             "Total world size must be divisible by this value. "
             "Default 1 gives one independent sim per rank.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set env var BEFORE importing ppo_joint so PETSc sees the subcommunicator.
    os.environ["HRSSA_RANKS_PER_ENV"] = str(args.ranks_per_env)

    # Only import after the env var is set
    from training.ppo_joint import (
        HRSSATrainer, PPOConfig, is_global_rank0, _WORLD_RANK
    )

    cfg = PPOConfig.from_yaml(args.config)
    cfg.total_timesteps = args.timesteps

    trainer = HRSSATrainer(cfg)

    if args.resume:
        trainer.load(args.resume)

    obs = trainer.env.reset()
    h   = trainer.manager.init_hidden(1).to(trainer.device)

    while trainer.global_step < cfg.total_timesteps:
        obs, h, ep_rewards_list = trainer.collect_rollout(obs, h)
        loss = trainer.update()

        if is_global_rank0() and trainer.episode > 0 and trainer.episode % cfg.save_interval == 0:
            trainer.save()

        if is_global_rank0() and ep_rewards_list:
            mean_ep = sum(ep_rewards_list) / len(ep_rewards_list)
            print(
                f"[Step {trainer.global_step:7d}] "
                f"loss={loss:.4f}  mean_ep_reward={mean_ep:.3f}",
                flush=True,
            )

    if is_global_rank0():
        trainer.save(tag="final")
        print("Training complete. Checkpoints saved to:", cfg.save_dir, flush=True)


if __name__ == "__main__":
    main()