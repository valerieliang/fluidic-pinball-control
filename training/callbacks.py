# training/callbacks.py
"""
Callbacks for the HR-SSA training loop.

Each callback receives a shared `state` dict populated by HRSSATrainer
at the end of every rollout. Callbacks are composable -- pass a list to
the trainer and it calls them in order.

State keys available to callbacks
----------------------------------
episode         : int   -- completed episode count
global_step     : int   -- total env steps taken
ep_reward       : float -- reward of the episode just completed
mean_reward     : float -- rolling mean over last 100 episodes
total_loss      : float -- PPO loss from the most recent update
manager         : nn.Module
sub_policies    : nn.Module
encoder         : nn.Module  (env._buf.encoder)
optimizer       : optim.Optimizer
obs_mean        : np.ndarray
obs_std         : np.ndarray
env             : PinballEnv
save_dir        : Path

Optional trajectory keys (populated when a rollout buffer is attached)
-----------------------------------------------------------------------
obs_buffer      : np.ndarray  shape (n_steps, n_probes, n_features)
action_buffer   : np.ndarray  shape (n_steps, n_actuators)
reward_buffer   : np.ndarray  shape (n_steps,)
drag_buffer     : np.ndarray  shape (n_steps,)
lift_buffer     : np.ndarray  shape (n_steps,)

Optional HDF5 metadata keys (all written as HDF5 root attributes when present)
-------------------------------------------------------------------------------
Re              : float  -- Reynolds number
geometry        : str
actuation       : str
algorithm       : str    -- defaults to "PPO"
convergence_episode : int
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Callback:
    def on_rollout_end(self, state: Dict[str, Any]) -> None:
        pass

    def on_training_end(self, state: Dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------

class CSVLoggerCallback(Callback):
    """
    Appends one row per completed episode to a CSV file.
    Columns: episode, global_step, ep_reward, mean_reward_100, total_loss, wall_time
    """
    def __init__(self, log_interval: int = 1):
        self.log_interval = log_interval
        self._path: Optional[Path] = None
        self._writer = None
        self._file = None
        self._t0 = time.time()

    def _init(self, state: Dict[str, Any]):
        self._path = state["save_dir"] / "training_log.csv"
        self._file = open(self._path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=[
            "episode", "global_step", "ep_reward",
            "mean_reward_100", "total_loss", "wall_time_s",
        ])
        self._writer.writeheader()
        self._file.flush()

    def on_rollout_end(self, state: Dict[str, Any]) -> None:
        if self._writer is None:
            self._init(state)

        ep = state["episode"]
        if ep % self.log_interval != 0:
            return

        row = {
            "episode":        ep,
            "global_step":    state["global_step"],
            "ep_reward":      f"{state['ep_reward']:.6f}",
            "mean_reward_100": f"{state['mean_reward']:.6f}",
            "total_loss":     f"{state['total_loss']:.6f}",
            "wall_time_s":    f"{time.time() - self._t0:.1f}",
        }
        self._writer.writerow(row)
        self._file.flush()

        print(
            f"[ep {ep:4d} | step {state['global_step']:7d}] "
            f"ep_reward={state['ep_reward']:8.4f}  "
            f"mean100={state['mean_reward']:8.4f}  "
            f"loss={state['total_loss']:.4f}"
        )

    def on_training_end(self, state: Dict[str, Any]) -> None:
        if self._file is not None:
            self._file.close()
            print(f"Training log saved -> {self._path}")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class CheckpointCallback(Callback):
    """
    Saves a full checkpoint every `save_interval` episodes and always on
    training end. Checkpoints are written as HDF5 (.h5) files and contain:

    Datasets
    --------
    /weights/manager        -- raw bytes of manager.state_dict() (torch format)
    /weights/sub_policies   -- raw bytes of sub_policies.state_dict()
    /weights/encoder        -- raw bytes of encoder.state_dict()
    /weights/optimizer      -- raw bytes of optimizer.state_dict()
    /norm/obs_mean          -- observation normalisation mean  (float32)
    /norm/obs_std           -- observation normalisation std   (float32)

    Trajectory datasets (written only when the corresponding buffer is
    present in `state`; shapes follow the HydroGym/paper convention)
    --------
    /trajectory/observations  -- (n_steps, n_probes, n_features)
    /trajectory/actions       -- (n_steps, n_actuators)
    /trajectory/rewards       -- (n_steps,)
    /trajectory/drags         -- (n_steps,)
    /trajectory/lifts         -- (n_steps,)

    Root attributes
    ---------------
    episode, global_step, ep_reward, mean_reward_100, total_loss
    Re, geometry, actuation, algorithm, convergence_episode
    (the last five are written only when present in `state`)

    Keeps only the last `keep_last` checkpoints to avoid filling disk on
    HPC runs.  The "final" checkpoint is never pruned.
    """

    def __init__(self, save_interval: int = 10, keep_last: int = 5):
        self.save_interval = save_interval
        self.keep_last     = keep_last
        self._saved: List[Path] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mpi_rank() -> int:
        """Return MPI world rank, or 0 if MPI is not in use."""
        try:
            from mpi4py import MPI
            return MPI.COMM_WORLD.Get_rank()
        except ImportError:
            return 0

    @staticmethod
    def _state_dict_to_bytes(module_or_optim) -> np.ndarray:
        """Serialise a PyTorch state-dict to a 1-D uint8 numpy array."""
        buf = io.BytesIO()
        torch.save(module_or_optim.state_dict(), buf)
        return np.frombuffer(buf.getvalue(), dtype=np.uint8)

    def _save(self, state: Dict[str, Any], tag: str) -> None:
        # Only rank 0 writes -- prevents all MPI ranks racing to lock the
        # same HDF5 file (BlockingIOError / errno 11 under Open MPI).
        if self._mpi_rank() != 0:
            return

        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required for HDF5 checkpointing. "
                "Install it with: pip install h5py"
            ) from exc

        path = state["save_dir"] / f"checkpoint_{tag}.h5"

        with h5py.File(path, "w") as f:

            # ---- model weights (serialised PyTorch state dicts) ----------
            wg = f.create_group("weights")
            wg.create_dataset(
                "manager",
                data=self._state_dict_to_bytes(state["manager"]),
            )
            wg.create_dataset(
                "sub_policies",
                data=self._state_dict_to_bytes(state["sub_policies"]),
            )
            wg.create_dataset(
                "encoder",
                data=self._state_dict_to_bytes(state["encoder"]),
            )
            wg.create_dataset(
                "optimizer",
                data=self._state_dict_to_bytes(state["optimizer"]),
            )

            # ---- observation normalisation stats -------------------------
            ng = f.create_group("norm")
            ng.create_dataset(
                "obs_mean",
                data=np.asarray(state["obs_mean"], dtype=np.float32),
            )
            ng.create_dataset(
                "obs_std",
                data=np.asarray(state["obs_std"], dtype=np.float32),
            )

            # ---- trajectory buffers (optional) ---------------------------
            traj_keys = {
                "obs_buffer":    "observations",
                "action_buffer": "actions",
                "reward_buffer": "rewards",
                "drag_buffer":   "drags",
                "lift_buffer":   "lifts",
            }
            traj_data = {
                h5_name: state[state_key]
                for state_key, h5_name in traj_keys.items()
                if state_key in state and state[state_key] is not None
            }
            if traj_data:
                tg = f.create_group("trajectory")
                for h5_name, arr in traj_data.items():
                    tg.create_dataset(
                        h5_name,
                        data=np.asarray(arr, dtype=np.float32),
                        compression="gzip",
                        compression_opts=4,
                    )

            # ---- scalar training metrics as root attributes --------------
            f.attrs["episode"]          = state["episode"]
            f.attrs["global_step"]      = state["global_step"]
            f.attrs["ep_reward"]        = float(state["ep_reward"])
            f.attrs["mean_reward_100"]  = float(state["mean_reward"])
            f.attrs["total_loss"]       = float(state["total_loss"])

            # ---- optional experiment metadata ----------------------------
            for meta_key in (
                "Re", "geometry", "actuation", "algorithm", "convergence_episode"
            ):
                if meta_key in state:
                    f.attrs[meta_key] = state[meta_key]
            # Provide a sensible default for algorithm if not set
            if "algorithm" not in f.attrs:
                f.attrs["algorithm"] = "PPO"

        self._saved.append(path)
        print(f"  checkpoint -> {path}")

        # Prune old checkpoints (never prune the "final" one)
        while len(self._saved) > self.keep_last:
            old = self._saved.pop(0)
            if old.exists() and "final" not in old.name:
                old.unlink()

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_rollout_end(self, state: Dict[str, Any]) -> None:
        ep = state["episode"]
        if ep > 0 and ep % self.save_interval == 0:
            self._save(state, tag=f"ep{ep:05d}")

    def on_training_end(self, state: Dict[str, Any]) -> None:
        self._save(state, tag="final")


# ---------------------------------------------------------------------------
# Paraview export
# ---------------------------------------------------------------------------

class ParaviewExportCallback(Callback):
    """
    Triggers a Paraview-compatible .pvd snapshot at a fixed episode interval
    by calling env.render() if the underlying HydroGym FlowEnv supports it.

    HydroGym writes .pvd/.vtu files to a configurable output path. This
    callback checks for a `paraview_output_dir` key in state (set by the
    trainer from config) and passes it through to the env if present.

    If the env does not support Paraview export (e.g. during unit tests),
    the callback is a no-op.
    """
    def __init__(self, export_interval: int = 25):
        self.export_interval = export_interval

    def on_rollout_end(self, state: Dict[str, Any]) -> None:
        ep = state["episode"]
        if ep == 0 or ep % self.export_interval != 0:
            return

        env = state["env"]
        inner = getattr(env, "_env", None)
        if inner is None:
            return

        # HydroGym FlowEnv exposes flow.write_checkpoint() for .h5 and
        # flow.save_pvd() (solver-dependent). Try both gracefully.
        flow = getattr(inner, "flow", None)
        if flow is None:
            return

        pvd_dir = state.get("paraview_output_dir")
        if pvd_dir is not None:
            pvd_path = Path(pvd_dir) / f"flow_ep{ep:05d}.pvd"
            try:
                flow.save_checkpoint(str(pvd_path))
                print(f"  Paraview snapshot -> {pvd_path}")
            except AttributeError:
                pass   # solver doesn't implement save_checkpoint


# ---------------------------------------------------------------------------
# W&B (optional -- only active if wandb is installed and initialized)
# ---------------------------------------------------------------------------

class WandBCallback(Callback):
    """
    Logs scalar metrics to Weights & Biases if available.
    Falls back to a no-op if wandb is not installed or not logged in.

    Usage:
        cb = WandBCallback(project="hr-ssa", config=cfg.__dict__)
    """
    def __init__(self, project: str, config: dict, log_interval: int = 1):
        self.log_interval = log_interval
        self._active = False
        try:
            import wandb
            wandb.init(project=project, config=config)
            self._wandb = wandb
            self._active = True
            print(f"W&B logging active -> project: {project}")
        except Exception as e:
            print(f"W&B not available ({e}), skipping.")

    def on_rollout_end(self, state: Dict[str, Any]) -> None:
        if not self._active:
            return
        ep = state["episode"]
        if ep % self.log_interval != 0:
            return
        self._wandb.log({
            "episode":         ep,
            "ep_reward":       state["ep_reward"],
            "mean_reward_100": state["mean_reward"],
            "total_loss":      state["total_loss"],
            "global_step":     state["global_step"],
        }, step=state["global_step"])

    def on_training_end(self, state: Dict[str, Any]) -> None:
        if self._active:
            self._wandb.finish()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def default_callbacks(
    log_interval: int = 1,
    save_interval: int = 10,
    export_interval: int = 25,
    keep_last: int = 5,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[dict] = None,
) -> List[Callback]:
    """
    Returns the standard callback stack. Pass to HRSSATrainer.train().
    W&B callback is only added if wandb_project is specified.
    """
    cbs: List[Callback] = [
        CSVLoggerCallback(log_interval=log_interval),
        CheckpointCallback(save_interval=save_interval, keep_last=keep_last),
        ParaviewExportCallback(export_interval=export_interval),
    ]
    if wandb_project is not None:
        cbs.append(WandBCallback(
            project=wandb_project,
            config=wandb_config or {},
            log_interval=log_interval,
        ))
    return cbs