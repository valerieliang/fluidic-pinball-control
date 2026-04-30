# baseline/pinball_env_baseline.py
"""
Baseline-only Pinball Environment with Visualization and H5 Data Export.
Self-contained inside the baseline/ package.
"""

import numpy as np
import gym
from gym import spaces
import hydrogym.firedrake as hgym
from hydrogym import FlowEnv
import time
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for MPI/headless
import matplotlib.pyplot as plt
from collections import deque
import json
from datetime import datetime

# Optional: H5 saving
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("[Warning] h5py not available. H5 saving disabled.")

try:
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _rank() -> int:
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0


def _mpi_comm():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return None


class PinballEnvBaseline(gym.Env):
    """
    Baseline-only Gym-compatible wrapper around HydroGym's FlowEnv for the fluidic pinball.

    Features:
        - Flat observations (raw probes only, no HR-SSA embedding)
        - Reward accumulation over control interval (matching HydroGym paper)
        - Warmup visualization and statistics plotting
        - Turbulence/flow field snapshot saving during training
        - H5 data export at episode end (probes, drag, lift, rewards, actions)
        - Reduced logging for cleaner training output

    Config options:
        Re: int = 100                # Reynolds number
        mesh: str = "medium"         # Mesh resolution
        dt: float = 1e-2             # Time step
        num_substeps: int = 170      # CFD steps per control action
        n_probes: int = 6            # Number of observation probes
        warmup_steps: int = 500      # Warmup steps for normalization
        reward_omega: float = 1.0    # Lift penalty weight
        verbose: bool = False        # Control logging verbosity
        viz_dir: str = "visualizations"  # Directory for saving plots
        data_dir: str = "episode_data"   # Directory for saving H5 files
        save_warmup_plots: bool = True   # Save warmup statistics plots
        save_episode_snapshots: bool = True  # Save flow snapshots during training
        save_episode_h5: bool = True   # Save H5 data at episode end
        snapshot_freq: int = 10      # Save snapshots every N episodes
    """

    metadata = {"render.modes": []}

    def __init__(self, config: dict = None):
        super().__init__()

        cfg = config or {}

        # --- Core configuration ---
        self.Re = cfg.get("Re", 100)
        self.mesh = cfg.get("mesh", "medium")
        self.dt = cfg.get("dt", 1e-2)
        self.num_substeps = cfg.get("num_substeps", 170)
        self.n_probes = cfg.get("n_probes", 6)
        self.warmup_steps = cfg.get("warmup_steps", 500)
        self.reward_omega = cfg.get("reward_omega", 1.0)
        self.verbose = cfg.get("verbose", False)

        # --- Data export configuration ---
        self.data_dir = Path(cfg.get("data_dir", "episode_data"))
        self.save_episode_h5 = cfg.get("save_episode_h5", True) and H5PY_AVAILABLE
        self.viz_dir = Path(cfg.get("viz_dir", "visualizations"))
        self.save_warmup_plots = cfg.get("save_warmup_plots", True)
        self.save_episode_snapshots = cfg.get("save_episode_snapshots", True)
        self.snapshot_freq = cfg.get("snapshot_freq", 10)

        # Create directories
        self._setup_directories()

        # Store rank info for conditional logging
        self._rank = _rank()
        self._is_root = self._rank == 0

        # --- Reward accumulation setup ---
        self._accumulated_reward = 0.0

        # --- Episode data buffers (for H5 export) ---
        self._episode_obs_buffer: list = []
        self._episode_action_buffer: list = []
        self._episode_reward_buffer: list = []
        self._episode_drag_buffer: list = []
        self._episode_lift_buffer: list = []
        self._episode_substep_rewards: list = []

        # --- Warmup tracking for visualization ---
        self._warmup_rewards: list = []
        self._warmup_drag: list = []
        self._warmup_lift: list = []
        self._warmup_obs_history: list = []

        # --- Episode tracking ---
        self._episode_count = 0
        self._episode_rewards: deque = deque(maxlen=100)
        self._episode_drag_history: deque = deque(maxlen=100)
        self._episode_lift_history: deque = deque(maxlen=100)
        self._episode_action_magnitudes: deque = deque(maxlen=100)

        self._log(f"[PinballEnvBaseline] Initializing Re={self.Re}, mesh={self.mesh}, substeps={self.num_substeps}")

        # --- Create underlying FlowEnv ---
        env_config = {
            "flow": hgym.Pinball,
            "flow_config": {
                "mesh": self.mesh,
                "Re": self.Re,
                "reward_omega": self.reward_omega,
            },
            "solver": hgym.SemiImplicitBDF,
            "solver_config": {
                "dt": self.dt,
            },
            "actuation_config": {
                "num_substeps": self.num_substeps,
            },
        }

        self._env = FlowEnv(env_config)

        # --- Observation and Action spaces ---
        obs_dim = self.n_probes
        action_dim = 3  # Three cylinders

        raw_act = self._env.action_space
        self.action_space = spaces.Box(
            low=raw_act.low,
            high=raw_act.high,
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # --- Normalization ---
        self._normalizer_fitted = False
        self._post_warmup_done = False
        self._obs_mean = np.zeros(self.n_probes)
        self._obs_std = np.ones(self.n_probes)

        # --- Episode termination ---
        self._max_episode_steps = 200
        self._step_count = 0

        # --- Episode info for tracking ---
        self._current_episode_drag = []
        self._current_episode_lift = []
        self._current_episode_reward = 0.0
        self._current_episode_actions = []

        # --- Cached episode summary (populated at episode end, before buffers clear) ---
        self._last_episode_summary: dict = {}

        self._log(f"[PinballEnvBaseline] Ready (obs_dim={obs_dim}, action_dim={action_dim})")
        if self.save_episode_h5:
            self._log(f"[PinballEnvBaseline] H5 export enabled -> {self.data_dir}")

    # ------------------------------------------------------------------
    # Directory Setup
    # ------------------------------------------------------------------

    def _setup_directories(self):
        """Create all necessary directories."""
        dirs_to_create = []

        if self.save_warmup_plots or self.save_episode_snapshots:
            dirs_to_create.extend([
                self.viz_dir,
                self.viz_dir / "warmup",
                self.viz_dir / "episodes",
                self.viz_dir / "turbulence",
            ])

        if self.save_episode_h5:
            dirs_to_create.extend([
                self.data_dir,
                self.data_dir / f"Re{self.Re}",
            ])

        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, msg: str, force: bool = False):
        """Conditional logging based on verbose flag and root rank."""
        if self._is_root and (self.verbose or force):
            print(f"{msg}", flush=True)

    # ------------------------------------------------------------------
    # H5 Export
    # ------------------------------------------------------------------

    def _save_episode_h5(self):
        """Save episode data to H5 file."""
        if not self.save_episode_h5 or not self._is_root:
            return

        if len(self._episode_obs_buffer) == 0:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"Re{self.Re}" / f"episode_{self._episode_count:04d}_{timestamp}.h5"

        try:
            with h5py.File(filename, 'w') as f:
                f.attrs['episode'] = self._episode_count
                f.attrs['Re'] = self.Re
                f.attrs['mesh'] = self.mesh
                f.attrs['num_substeps'] = self.num_substeps
                f.attrs['n_probes'] = self.n_probes
                f.attrs['total_reward'] = self._current_episode_reward
                f.attrs['mean_drag'] = np.mean(self._current_episode_drag) if self._current_episode_drag else 0.0
                f.attrs['mean_abs_lift'] = np.mean(np.abs(self._current_episode_lift)) if self._current_episode_lift else 0.0
                f.attrs['timestamp'] = timestamp

                obs_array = np.array(self._episode_obs_buffer, dtype=np.float32)
                f.create_dataset('observations', data=obs_array, compression='gzip')

                actions_array = np.array(self._episode_action_buffer, dtype=np.float32)
                f.create_dataset('actions', data=actions_array, compression='gzip')

                rewards_array = np.array(self._episode_reward_buffer, dtype=np.float32)
                f.create_dataset('rewards', data=rewards_array, compression='gzip')

                if self._episode_drag_buffer:
                    drag_array = np.array(self._episode_drag_buffer, dtype=np.float32)
                    f.create_dataset('drag', data=drag_array, compression='gzip')

                if self._episode_lift_buffer:
                    lift_array = np.array(self._episode_lift_buffer, dtype=np.float32)
                    f.create_dataset('lift', data=lift_array, compression='gzip')

                if self._episode_substep_rewards:
                    substep_rewards = np.array(self._episode_substep_rewards, dtype=np.float32)
                    f.create_dataset('substep_rewards', data=substep_rewards, compression='gzip')

                f.create_dataset('obs_mean', data=self._obs_mean)
                f.create_dataset('obs_std', data=self._obs_std)

            self._log(f"[H5] Saved episode data to {filename}", force=False)

        except Exception as e:
            self._log(f"[H5 ERROR] Failed to save episode H5: {e}", force=True)

    def _get_episode_summary(self) -> dict:
        """Get summary statistics for the current episode."""
        if not self._episode_drag_buffer:
            return {}

        drag_array = np.array(self._episode_drag_buffer)
        lift_array = np.array(self._episode_lift_buffer)
        actions_array = np.array(self._episode_action_buffer)

        return {
            'episode': self._episode_count,
            'total_reward': float(self._current_episode_reward),
            'mean_drag': float(np.mean(drag_array)),
            'std_drag': float(np.std(drag_array)),
            'min_drag': float(np.min(drag_array)),
            'max_drag': float(np.max(drag_array)),
            'mean_abs_lift': float(np.mean(np.abs(lift_array))),
            'std_lift': float(np.std(lift_array)),
            'mean_action_magnitude': float(np.mean(np.linalg.norm(actions_array, axis=1))),
            'num_steps': len(self._episode_action_buffer),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_to_array(self, raw_obs) -> np.ndarray:
        return np.array(raw_obs, dtype=np.float32)

    def _get_drag_lift(self) -> tuple:
        """Extract current drag and lift coefficients from environment."""
        try:
            flow = self._env.flow
            if hasattr(flow, 'compute_forces'):
                drag_list, lift_list = flow.compute_forces()
                CD = float(sum(drag_list))
                CL = float(sum(lift_list))
                return CD, CL
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] _get_drag_lift failed: {e}")
        return 0.0, 0.0

    def _plot_warmup_statistics(self):
        """Generate and save warmup statistics plots."""
        if not self.save_warmup_plots or not self._is_root:
            return

        self._log("[WARMUP] Generating statistics plots...", force=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Warmup Statistics (Re={self.Re}, mesh={self.mesh})', fontsize=14)

        if self._warmup_rewards:
            axes[0, 0].hist(self._warmup_rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].axvline(np.mean(self._warmup_rewards), color='red', linestyle='--',
                              label=f'Mean: {np.mean(self._warmup_rewards):.3f}')
            axes[0, 0].set_xlabel('Reward')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Reward Distribution During Warmup')
            axes[0, 0].legend()

        if self._warmup_drag:
            axes[0, 1].plot(self._warmup_drag, 'b-', alpha=0.7, linewidth=0.5)
            axes[0, 1].axhline(np.mean(self._warmup_drag), color='red', linestyle='--',
                              label=f'Mean CD: {np.mean(self._warmup_drag):.3f}')
            axes[0, 1].set_xlabel('Warmup Step')
            axes[0, 1].set_ylabel('Drag Coefficient (CD)')
            axes[0, 1].set_title('Drag Evolution During Warmup')
            axes[0, 1].legend()

        if self._warmup_lift:
            axes[0, 2].plot(self._warmup_lift, 'g-', alpha=0.7, linewidth=0.5)
            axes[0, 2].axhline(np.mean(self._warmup_lift), color='red', linestyle='--',
                              label=f'Mean CL: {np.mean(self._warmup_lift):.3f}')
            axes[0, 2].set_xlabel('Warmup Step')
            axes[0, 2].set_ylabel('Lift Coefficient (CL)')
            axes[0, 2].set_title('Lift Evolution During Warmup')
            axes[0, 2].legend()

        if self._warmup_obs_history:
            obs_array = np.array(self._warmup_obs_history)
            for i in range(min(self.n_probes, 6)):
                axes[1, 0].plot(obs_array[:, i], alpha=0.5, linewidth=0.5, label=f'Probe {i}')
            axes[1, 0].set_xlabel('Warmup Step')
            axes[1, 0].set_ylabel('Observation Value')
            axes[1, 0].set_title('Probe Observations During Warmup')
            axes[1, 0].legend(loc='upper right', fontsize=8)

        if self._warmup_obs_history:
            obs_array = np.array(self._warmup_obs_history)
            axes[1, 1].boxplot([obs_array[:, i] for i in range(self.n_probes)])
            axes[1, 1].set_xlabel('Probe Index')
            axes[1, 1].set_ylabel('Observation Value')
            axes[1, 1].set_title('Probe Value Distributions')

        axes[1, 2].bar(range(self.n_probes), self._obs_mean, alpha=0.7, label='Mean', color='blue')
        axes[1, 2].bar(range(self.n_probes), self._obs_std, alpha=0.7, label='Std', color='orange',
                      bottom=self._obs_mean)
        axes[1, 2].set_xlabel('Probe Index')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].set_title('Normalization Statistics')
        axes[1, 2].legend()

        plt.tight_layout()
        save_path = self.viz_dir / "warmup" / f"warmup_statistics_Re{self.Re}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        summary = {
            'Re': self.Re,
            'mesh': self.mesh,
            'num_substeps': self.num_substeps,
            'warmup_steps': self.warmup_steps,
            'obs_mean': self._obs_mean.tolist(),
            'obs_std': self._obs_std.tolist(),
            'mean_reward': float(np.mean(self._warmup_rewards)) if self._warmup_rewards else 0.0,
            'mean_drag': float(np.mean(self._warmup_drag)) if self._warmup_drag else 0.0,
            'mean_abs_lift': float(np.mean(np.abs(self._warmup_lift))) if self._warmup_lift else 0.0,
        }
        json_path = self.viz_dir / "warmup" / f"warmup_summary_Re{self.Re}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self._log(f"[WARMUP] Statistics saved to {save_path}", force=True)

    def _save_turbulence_snapshot(self, episode: int, step=None):
        """Save a snapshot of the current flow field."""
        if not self.save_episode_snapshots or not self._is_root:
            return

        try:
            flow = self._env.flow
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            if hasattr(flow, 'mesh'):
                mesh = flow.mesh
                coords = mesh.coordinates.dat.data_ro

                if hasattr(flow, 'vorticity'):
                    vorticity = flow.vorticity()
                else:
                    vorticity = np.zeros(len(coords))

                scatter = ax.scatter(coords[:, 0], coords[:, 1],
                                   c=vorticity.flatten()[:len(coords)],
                                   cmap='RdBu_r', s=1, alpha=0.8)
                plt.colorbar(scatter, ax=ax, label='Vorticity')

            cylinder_positions = [(-1.299, 0.0), (0.0, 0.75), (0.0, -0.75)]
            for x, y in cylinder_positions:
                circle = plt.Circle((x, y), 0.5, color='black', fill=False, linewidth=2)
                ax.add_patch(circle)

            CD, CL = self._get_drag_lift()
            ax.set_xlabel('x/D')
            ax.set_ylabel('y/D')
            ax.set_title(f'Episode {episode} - Flow Field\nCD={CD:.3f}, CL={CL:.3f}')
            ax.set_aspect('equal')

            step_str = f"_step{step}" if step is not None else ""
            save_path = self.viz_dir / "turbulence" / f"ep{episode:04d}{step_str}_vorticity.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self._log(f"[VIZ] Failed to save turbulence snapshot: {e}")

    def _plot_episode_progress(self):
        """Plot episode-level progress."""
        if not self.save_episode_snapshots or not self._is_root:
            return

        if len(self._episode_rewards) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Training Progress (Episode {self._episode_count})', fontsize=14)

        episodes = list(range(max(1, self._episode_count - len(self._episode_rewards) + 1),
                              self._episode_count + 1))

        rewards = list(self._episode_rewards)
        axes[0, 0].plot(episodes[-len(rewards):], rewards, 'b-', alpha=0.7)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Reward History')
        axes[0, 0].grid(True, alpha=0.3)

        if self._episode_drag_history:
            drags = list(self._episode_drag_history)
            axes[0, 1].plot(episodes[-len(drags):], drags, 'r-', alpha=0.7)
            axes[0, 1].axhline(2.9, color='gray', linestyle='--', alpha=0.5, label='Uncontrolled CD≈2.9')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Mean Drag (CD)')
            axes[0, 1].set_title('Drag Coefficient History')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        if self._episode_lift_history:
            lifts = list(self._episode_lift_history)
            axes[1, 0].plot(episodes[-len(lifts):], lifts, 'g-', alpha=0.7)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Mean |Lift|')
            axes[1, 0].set_title('Lift Coefficient History')
            axes[1, 0].grid(True, alpha=0.3)

        if len(rewards) >= 10:
            window = min(10, len(rewards))
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ma_episodes = episodes[window-1:]
            axes[1, 1].plot(ma_episodes, ma, 'b-', linewidth=2, label=f'{window}-ep MA')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward (Moving Avg)')
            axes[1, 1].set_title('Smoothed Reward')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.viz_dir / "episodes" / f"progress_ep{self._episode_count:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _fit_normalizer(self):
        """Fit observation normalizer with visualization."""
        comm = _mpi_comm()
        rank = _rank()

        self._log(f"[WARMUP] Starting normalizer fitting with {self.warmup_steps} steps", force=True)
        start_time = time.time()

        self._warmup_rewards = []
        self._warmup_drag = []
        self._warmup_lift = []
        self._warmup_obs_history = []

        raw = self._env.reset()
        raw_array = self._raw_to_array(raw)
        self._warmup_obs_history.append(raw_array)

        for i in range(self.warmup_steps):
            if rank == 0 and i % max(1, self.warmup_steps // 10) == 0:
                self._log(f"[WARMUP] Step {i}/{self.warmup_steps}", force=True)

            if rank == 0:
                action = self._env.action_space.sample()
            else:
                action = None

            if comm is not None:
                action = comm.bcast(action, root=0)

            step_reward = 0.0
            final_CD, final_CL = 0.0, 0.0

            for substep_idx in range(self.num_substeps):
                raw, reward, done, info = self._env.step(action)
                step_reward += reward

                if substep_idx == self.num_substeps - 1:
                    final_CD, final_CL = self._get_drag_lift()

                if done:
                    break

            raw_array = self._raw_to_array(raw)
            self._warmup_obs_history.append(raw_array)

            self._warmup_drag.append(final_CD)
            self._warmup_lift.append(final_CL)
            self._warmup_rewards.append(step_reward)

            if done:
                raw = self._env.reset()
                raw_array = self._raw_to_array(raw)
                self._warmup_obs_history.append(raw_array)

        if rank == 0:
            if self._warmup_obs_history:
                stacked = np.stack(self._warmup_obs_history, axis=0)
                self._obs_mean = stacked.mean(axis=0)
                self._obs_std = stacked.std(axis=0).clip(1e-6)
                self._log(f"[WARMUP] Stats: mean={self._obs_mean}, std={self._obs_std}", force=True)
                if self.save_warmup_plots:
                    self._plot_warmup_statistics()
            else:
                self._obs_mean = np.zeros(self.n_probes)
                self._obs_std = np.ones(self.n_probes)
                self._log("[WARMUP] WARNING: No warmup data collected, using default stats", force=True)

        if comm is not None:
            if rank == 0:
                stats = np.array([*self._obs_mean, *self._obs_std])
            else:
                stats = np.zeros(2 * self.n_probes)
            stats = comm.bcast(stats, root=0)
            if rank != 0:
                self._obs_mean = stats[:self.n_probes]
                self._obs_std = stats[self.n_probes:]

        self._normalizer_fitted = True
        elapsed = time.time() - start_time
        self._log(f"[WARMUP] Normalizer fitting COMPLETE after {elapsed:.1f}s", force=True)

        return raw_array

    def reset(self):
        """Reset environment and return initial observation."""
        comm = _mpi_comm()

        if len(self._episode_obs_buffer) > 0:
            self._save_episode_h5()

        if not self._normalizer_fitted:
            self._log("[RESET] Fitting normalizer...", force=True)
            obs = self._fit_normalizer()
            self._log("[RESET] Normalizer fitted", force=True)

            self._episode_obs_buffer = [obs.copy()]
            self._accumulated_reward = 0.0
            self._step_count = 0
            self._current_episode_drag = []
            self._current_episode_lift = []
            self._current_episode_reward = 0.0
            self._current_episode_actions = []
            self._post_warmup_done = True

            return obs.astype(np.float32)

        raw = self._env.reset()

        if raw is None:
            self._log("[RESET] ERROR: raw observation is None!", force=True)
            raw = self._env.reset()
            if raw is None:
                raise RuntimeError("Failed to get valid observation from environment")

        if not self._post_warmup_done:
            self._log("[RESET] Post-normalizer warmup...", force=True)
            for i in range(50):
                if self._is_root:
                    action = np.zeros(3)
                else:
                    action = None
                if comm is not None:
                    action = comm.bcast(action, root=0)
                for _ in range(self.num_substeps):
                    raw, _, _, _ = self._env.step(action)
            self._post_warmup_done = True
            self._log("[RESET] Post-normalizer warmup COMPLETE", force=True)

        self._episode_obs_buffer = []
        self._episode_action_buffer = []
        self._episode_reward_buffer = []
        self._episode_drag_buffer = []
        self._episode_lift_buffer = []
        self._episode_substep_rewards = []

        self._accumulated_reward = 0.0
        self._step_count = 0
        self._current_episode_drag = []
        self._current_episode_lift = []
        self._current_episode_reward = 0.0
        self._current_episode_actions = []

        obs = self._raw_to_array(raw)
        self._episode_obs_buffer.append(obs.copy())

        if self.save_episode_snapshots and self._episode_count % self.snapshot_freq == 0:
            self._save_turbulence_snapshot(self._episode_count, step=0)

        return obs.astype(np.float32)

    def step(self, action):
        """Execute one control step with reward accumulation over substeps."""
        action = np.asarray(action, dtype=np.float32)

        self._episode_action_buffer.append(action.copy())
        self._current_episode_actions.append(action.copy())

        self._accumulated_reward = 0.0
        final_raw_obs = None
        done = False

        for substep in range(self.num_substeps):
            raw_obs, reward, step_done, info = self._env.step(action)

            self._accumulated_reward += reward
            self._episode_substep_rewards.append(reward)
            final_raw_obs = raw_obs

            CD, CL = self._get_drag_lift()

            self._episode_drag_buffer.append(CD)
            self._episode_lift_buffer.append(CL)
            self._current_episode_drag.append(CD)
            self._current_episode_lift.append(CL)

            if step_done:
                done = True
                break

        obs = self._raw_to_array(final_raw_obs)
        self._episode_obs_buffer.append(obs.copy())
        self._episode_reward_buffer.append(self._accumulated_reward)

        self._step_count += 1
        self._current_episode_reward += self._accumulated_reward

        if self.save_episode_snapshots and self._episode_count % self.snapshot_freq == 0 and self._step_count % 50 == 0:
            self._save_turbulence_snapshot(self._episode_count, step=self._step_count)

        if self._step_count >= self._max_episode_steps:
            done = True

        if done:
            self._episode_count += 1
            self._episode_rewards.append(self._current_episode_reward)

            if self._current_episode_drag:
                self._episode_drag_history.append(np.mean(self._current_episode_drag))
            if self._current_episode_lift:
                self._episode_lift_history.append(np.mean(np.abs(self._current_episode_lift)))
            if self._current_episode_actions:
                self._episode_action_magnitudes.append(
                    np.mean(np.linalg.norm(np.array(self._current_episode_actions), axis=1))
                )

            self._last_episode_summary = self._get_episode_summary()

            self._save_episode_h5()

            if self.save_episode_snapshots:
                self._save_turbulence_snapshot(self._episode_count, step="final")
                if self._episode_count % (self.snapshot_freq * 5) == 0:
                    self._plot_episode_progress()

        return obs, float(self._accumulated_reward), bool(done), info

    def get_episode_summary(self) -> dict:
        return self._last_episode_summary

    def render(self, mode="human"):
        pass

    def close(self):
        if len(self._episode_obs_buffer) > 0:
            self._save_episode_h5()
        self._env.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_mean(self):
        return self._obs_mean

    @property
    def obs_std(self):
        return self._obs_std

    @property
    def episode_count(self):
        return self._episode_count

    @property
    def best_metrics(self) -> dict:
        return {
            'best_reward': max(self._episode_rewards) if self._episode_rewards else -np.inf,
            'best_drag': min(self._episode_drag_history) if self._episode_drag_history else np.inf,
            'best_lift': min(self._episode_lift_history) if self._episode_lift_history else np.inf,
        }