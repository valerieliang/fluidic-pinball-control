#!/usr/bin/env python
# baseline/run_baseline.py
"""
Entry point for flat PPO baseline training with MPI and GPU support.

Run from the project root:
    mpiexec -n <N> python -m baseline.run_baseline --config baseline/config_flat.yaml

Or single-process:
    python -m baseline.run_baseline --config baseline/config_flat.yaml
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import os

# GPU setup: allow CUDA for PyTorch (policy network).
# PETSc/Firedrake uses CPU-side MPI; GPU-aware MPI is disabled via
# PETSC_OPTIONS="-use_gpu_aware_mpi 0" (set in ~/.bashrc per setup_gpu.md).
# Do NOT set CUDA_VISIBLE_DEVICES="" here — that would break GPU inference.
os.environ.setdefault("PETSC_OPTIONS", "-use_gpu_aware_mpi 0")
os.environ["GYM_WARNINGS"] = "false"

import argparse
import torch

# Self-contained: all imports from within baseline/
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

    parser = argparse.ArgumentParser(description="Flat PPO Baseline Training")
    parser.add_argument("--config", type=str, default="baseline/config_flat.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")
    args = parser.parse_args()

    # Allow forcing CPU via flag (useful for debugging)
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    cfg = FlatPPOConfig.from_yaml(args.config)

    if args.verbose:
        cfg.verbose = True

    if rank == 0:
        device_str = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if device_str == "cuda" else "N/A"
        print("=" * 60)
        print(f"  Flat PPO Baseline (rank 0 of {size} MPI processes)")
        print(f"  Re={cfg.Re}  mesh={cfg.mesh}  substeps={cfg.num_substeps}")
        print(f"  total_timesteps={cfg.total_timesteps:,}")
        print(f"  run_name={cfg.run_name}")
        print(f"  verbose={cfg.verbose}")
        print(f"  device={device_str}  gpu={gpu_name}")
        print("=" * 60, flush=True)

    trainer = FlatPPOTrainer(cfg)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()