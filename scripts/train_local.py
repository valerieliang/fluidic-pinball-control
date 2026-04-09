#!/usr/bin/env python
# scripts/train_local.py
"""
Single-node training entrypoint.

MPI topology
------------
ALL ranks run the full training loop because every env.step() / env.reset()
triggers a collective PETSc solve that every rank must participate in.

Only rank 0 does meaningful RL work (policy updates, logging, saving).
Non-root ranks call the same env methods but ignore return values, and
skip all bookkeeping that doesn't involve a collective operation.

Usage (from repo root, inside venv-firedrake):
    mpiexec -n 8 python scripts/train_local.py --config configs/pinball_re100.yaml
"""

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import time
from training.ppo_joint import PPOConfig, HRSSATrainer
from training.callbacks import default_callbacks


def get_mpi_rank() -> int:
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0


def get_mpi_size() -> int:
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size()
    except ImportError:
        return 1


def rlog(rank: int, msg: str):
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    rank = get_mpi_rank()
    size = get_mpi_size()

    rlog(rank, f"started (pid={os.getpid()}, {size} total ranks)")

    parser = argparse.ArgumentParser(description="HR-SSA local training")
    parser.add_argument("--config",        type=str, default=None)
    parser.add_argument("--resume",        type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--timesteps",     type=int, default=None)
    args = parser.parse_args()

    cfg = PPOConfig.from_yaml(args.config) if args.config else PPOConfig()
    if args.timesteps is not None:
        cfg.total_timesteps = args.timesteps

    if rank == 0:
        print("=" * 60)
        print(f"  HR-SSA Training  (rank 0 of {size} MPI processes)")
        print(f"  Re={cfg.Re}  mesh={cfg.mesh}  substeps={cfg.num_substeps}")
        print(f"  total_timesteps={cfg.total_timesteps:,}")
        print(f"  run_name={cfg.run_name}")
        print(f"  save_dir={cfg.save_dir}/{cfg.run_name}")
        print("=" * 60, flush=True)

    rlog(rank, "building HRSSATrainer...")
    t0 = time.time()
    trainer = HRSSATrainer(cfg)
    rlog(rank, f"HRSSATrainer ready ({time.time()-t0:.1f}s)")

    # All ranks run train() together so that every collective PETSc solve
    # inside env.step()/env.reset() has all ranks present.
    # Rank-gating of logging/saving happens inside train() and save().
    callbacks = default_callbacks(
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        wandb_project=args.wandb_project if rank == 0 else None,
        wandb_config=cfg.__dict__,
    ) if rank == 0 else []

    rlog(rank, "entering train()")
    trainer.train(resume_from=args.resume, callbacks=callbacks)
    rlog(rank, "train() returned, exiting")


if __name__ == "__main__":
    main()