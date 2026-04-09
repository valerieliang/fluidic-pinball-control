#!/usr/bin/env python
# baseline/run_baseline.py
"""
Entry point for flat PPO baseline training with MPI support.
"""

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add parent directory to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from baseline.flat_ppo import FlatPPOConfig, FlatPPOTrainer


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


def main():
    rank = get_mpi_rank()
    size = get_mpi_size()
    
    print(f"[rank {rank}] started (pid={os.getpid()}, {size} total ranks)", flush=True)
    
    parser = argparse.ArgumentParser(description="Flat PPO Baseline Training")
    parser.add_argument("--config", type=str, default="baseline/config_flat.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    cfg = FlatPPOConfig.from_yaml(args.config)
    
    if rank == 0:
        print("=" * 60)
        print(f"  Flat PPO Baseline (rank 0 of {size} MPI processes)")
        print(f"  Re={cfg.Re}  mesh={cfg.mesh}  substeps={cfg.num_substeps}")
        print(f"  total_timesteps={cfg.total_timesteps:,}")
        print(f"  run_name={cfg.run_name}")
        print("=" * 60, flush=True)
    
    trainer = FlatPPOTrainer(cfg)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()