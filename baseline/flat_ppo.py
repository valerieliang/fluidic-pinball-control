# baseline/flat_ppo.py
"""
Flat PPO training loop matching HydroGym paper baseline.
Uses standard PPO with GAE, without hierarchical structure.
Includes GPU support, best model tracking, and comprehensive logging.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Self-contained imports within baseline/
from baseline.pinball_env_baseline import PinballEnvBaseline as PinballEnv
from baseline.flat_policy import FlatActorCritic


def _select_device() -> torch.device:
    """
    Select the best available device.

    Priority: CUDA > CPU.
    The PETSc/Firedrake side uses -use_gpu_aware_mpi 0 (set via PETSC_OPTIONS),
    so CUDA is safe to use for PyTorch even without GPU-aware MPI.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


@dataclass
class FlatPPOConfig:
    """Configuration for flat PPO baseline."""

    # Environment
    Re: int = 100
    mesh: str = "medium"
    dt: float = 1e-2
    num_substeps: int = 170
    n_probes: int = 6
    warmup_steps: int = 500
    reward_omega: float = 1.0

    # Model
    hidden_dim: int = 64

    # PPO hyperparameters (matching paper baselines)
    total_timesteps: int = 500_000
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4

    # Logging and Visualization
    log_interval: int = 1
    save_interval: int = 50
    save_dir: str = "checkpoints"
    run_name: str = "flat_ppo_baseline"
    verbose: bool = False
    viz_dir: str = "visualizations"
    data_dir: str = "episode_data"
    save_warmup_plots: bool = True
    save_episode_snapshots: bool = True
    save_episode_h5: bool = True
    snapshot_freq: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> "FlatPPOConfig":
        with open(path, 'r') as f:
            d = yaml.safe_load(f)

        numeric_fields = [
            'Re', 'dt', 'num_substeps', 'n_probes', 'warmup_steps', 'reward_omega',
            'total_timesteps', 'n_steps', 'n_epochs', 'batch_size', 'gamma',
            'gae_lambda', 'clip_eps', 'vf_coef', 'ent_coef', 'max_grad_norm', 'lr',
            'log_interval', 'save_interval', 'hidden_dim', 'snapshot_freq'
        ]

        for f_name in numeric_fields:
            if f_name in d and isinstance(d[f_name], str):
                try:
                    if '.' in d[f_name]:
                        d[f_name] = float(d[f_name])
                    else:
                        d[f_name] = int(d[f_name])
                except (ValueError, TypeError):
                    pass

        bool_fields = ['verbose', 'save_warmup_plots', 'save_episode_snapshots', 'save_episode_h5']
        for f_name in bool_fields:
            if f_name in d and isinstance(d[f_name], str):
                d[f_name] = d[f_name].lower() == 'true'

        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class RolloutBuffer:
    """Stores trajectories for PPO update."""

    def __init__(self, n_steps: int, obs_dim: int, action_dim: int):
        self.n_steps = n_steps
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.advantages = None
        self.returns = None
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, done, value):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.n_steps

    def reset(self):
        self.ptr = 0

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """Compute GAE without reward normalization."""
        n = self.ptr

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values[:n]

        if n > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


class FlatPPOTrainer:
    """Standard PPO trainer with flat policy network and GPU support."""

    def __init__(self, cfg: FlatPPOConfig):
        self.cfg = cfg
        self.verbose = cfg.verbose

        # ------------------------------------------------------------------
        # Device selection
        # Only rank 0 runs the PyTorch policy; non-root ranks only step the
        # Firedrake environment.  We still select the device on all ranks so
        # checkpoint loading works uniformly.
        # ------------------------------------------------------------------
        self.device = _select_device()

        rank = self._mpi_rank()
        if rank == 0:
            print(f"[FlatPPO] Device: {self.device}", flush=True)
            if self.device.type == "cuda":
                print(f"[FlatPPO] GPU: {torch.cuda.get_device_name(0)}", flush=True)
                print(f"[FlatPPO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

        # Create environment
        self.env = PinballEnv({
            "Re": cfg.Re,
            "mesh": cfg.mesh,
            "dt": cfg.dt,
            "num_substeps": cfg.num_substeps,
            "n_probes": cfg.n_probes,
            "warmup_steps": cfg.warmup_steps,
            "reward_omega": cfg.reward_omega,
            "verbose": cfg.verbose,
            "viz_dir": cfg.viz_dir,
            "data_dir": cfg.data_dir,
            "save_warmup_plots": cfg.save_warmup_plots,
            "save_episode_snapshots": cfg.save_episode_snapshots,
            "save_episode_h5": cfg.save_episode_h5,
            "snapshot_freq": cfg.snapshot_freq,
        })

        obs_dim = cfg.n_probes
        action_dim = 3  # Three cylinders

        # Create policy on GPU (or CPU if unavailable)
        self.policy = FlatActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)

        n_rollouts = cfg.total_timesteps // cfg.n_steps
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=n_rollouts,
        )

        self.buffer = RolloutBuffer(
            n_steps=cfg.n_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        self.save_dir = Path(cfg.save_dir) / cfg.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Best model tracking
        self.best_reward = -np.inf
        self.best_drag = np.inf
        self.best_lift = np.inf
        self.best_episode = 0
        self.best_model_dir = self.save_dir / "best_models"
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.ep_rewards: deque[float] = deque(maxlen=100)
        self.ep_drags: deque[float] = deque(maxlen=100)
        self.ep_lifts: deque[float] = deque(maxlen=100)
        self.global_step = 0
        self.episode = 0

        # Loss tracking
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0
        self.last_approx_kl = 0.0

    @staticmethod
    def _mpi_rank() -> int:
        try:
            from mpi4py import MPI
            return MPI.COMM_WORLD.Get_rank()
        except ImportError:
            return 0

    def _check_and_save_best(self, episode_reward: float, episode_drag: float = None, episode_lift: float = None):
        """Check if current model is best and save if so."""
        is_best = False
        reasons = []

        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            is_best = True
            reasons.append("reward")

        if episode_drag is not None and episode_drag < self.best_drag:
            self.best_drag = episode_drag
            if "drag" not in reasons:
                is_best = True
                reasons.append("drag")

        if episode_lift is not None and episode_lift < self.best_lift:
            self.best_lift = episode_lift
            if "lift" not in reasons and "drag" not in reasons:
                is_best = True
                reasons.append("lift")

        if is_best:
            self.best_episode = self.episode
            reason_str = "_".join(reasons)
            path = self.best_model_dir / f"best_model_{reason_str}_ep{self.episode:05d}.pt"
            torch.save({
                "episode": self.episode,
                "global_step": self.global_step,
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_mean": self.env.obs_mean,
                "obs_std": self.env.obs_std,
                "cfg": self.cfg,
                "best_reward": self.best_reward,
                "best_drag": self.best_drag,
                "best_lift": self.best_lift,
                "episode_reward": episode_reward,
                "episode_drag": episode_drag,
                "episode_lift": episode_lift,
            }, path)
            print(f"  New best model ({reason_str}) saved -> {path}", flush=True)

    def collect_rollout(self, obs: np.ndarray):
        """Collect n_steps of experience."""
        rank = self._mpi_rank()
        is_root = rank == 0

        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except ImportError:
            comm = None

        if is_root:
            self.buffer.reset()

        if comm is not None:
            comm.Barrier()

        ep_reward = 0.0
        ep_rewards_list = []
        _zero_action = np.zeros(3, dtype=np.float32)

        need_reset = False

        for step_idx in range(self.cfg.n_steps):
            if is_root:
                obs_raw = obs[:self.cfg.n_probes]
                # Move obs to GPU for inference
                obs_t = torch.as_tensor(obs_raw, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    action, log_prob, value = self.policy.get_action(obs_t)

                # Move results back to CPU for environment interaction
                action_np = action.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)

                next_obs, reward, done, info = self.env.step(action_np)
                ep_reward += reward

                self.buffer.add(
                    obs=obs_raw,
                    action=action_np,
                    log_prob=log_prob.item(),
                    reward=reward,
                    done=float(done),
                    value=value.item(),
                )

                self.global_step += 1
                obs = next_obs

                if done:
                    self.episode += 1
                    ep_rewards_list.append(ep_reward)
                    self.ep_rewards.append(ep_reward)
                    ep_summary = self.env.get_episode_summary()
                    if ep_summary:
                        if 'mean_drag' in ep_summary:
                            self.ep_drags.append(ep_summary['mean_drag'])
                        if 'mean_abs_lift' in ep_summary:
                            self.ep_lifts.append(ep_summary['mean_abs_lift'])

                    if self.verbose:
                        print(f"[COLLECT] Episode {self.episode} finished, reward={ep_reward:.4f}", flush=True)
                    ep_reward = 0.0
                    need_reset = True
            else:
                _, _, done, _ = self.env.step(_zero_action)
                if done:
                    need_reset = True

            if comm is not None:
                need_reset_arr = np.array([need_reset], dtype=np.int32)
                comm.Bcast(need_reset_arr, root=0)
                need_reset = bool(need_reset_arr[0])

            if need_reset:
                if self.verbose and is_root:
                    print(f"[COLLECT] Resetting environment (step {step_idx})", flush=True)
                if is_root:
                    obs = self.env.reset()
                else:
                    _ = self.env.reset()
                need_reset = False

                if comm is not None:
                    comm.Barrier()

        # Bootstrap value at end of rollout
        if is_root:
            obs_raw = obs[:self.cfg.n_probes]
            obs_t = torch.as_tensor(obs_raw, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, _, last_value = self.policy.get_action(obs_t)
            self.buffer.compute_returns_and_advantages(
                last_value=last_value.item(),
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )

        if comm is not None:
            if is_root:
                obs_to_bcast = obs
            else:
                obs_to_bcast = np.zeros_like(obs)
            comm.Bcast(obs_to_bcast, root=0)
            if not is_root:
                obs = obs_to_bcast

        return obs, ep_rewards_list

    def update(self) -> float:
        """Perform PPO update on collected rollout. Runs entirely on GPU."""
        if self._mpi_rank() != 0:
            return 0.0

        n = self.buffer.ptr

        # Move entire rollout to GPU once; all minibatch ops stay on device
        obs = torch.as_tensor(self.buffer.obs[:n], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.buffer.actions[:n], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs[:n], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(self.buffer.advantages[:n], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(self.buffer.returns[:n], dtype=torch.float32, device=self.device)
        values = torch.as_tensor(self.buffer.values[:n], dtype=torch.float32, device=self.device)

        indices = np.arange(n)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.cfg.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n, self.cfg.batch_size):
                idx = indices[start:start + self.cfg.batch_size]
                if len(idx) == 0:
                    continue

                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_old_values = values[idx]

                log_probs, entropy, new_values = self.policy.evaluate(batch_obs, batch_actions)
                new_values = new_values.squeeze(-1)

                # CleanRL: normalize advantages per minibatch
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # PPO clipped objective
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                value_pred_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values,
                    -self.cfg.clip_eps,
                    self.cfg.clip_eps,
                )
                value_loss_unclipped = (new_values - batch_returns) ** 2
                value_loss_clipped = (value_pred_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.cfg.vf_coef * value_loss + self.cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1

        with torch.no_grad():
            log_probs, _, _ = self.policy.evaluate(obs, actions)
            approx_kl = ((log_probs - old_log_probs) ** 2).mean().item()

        if n_updates > 0:
            self.last_policy_loss = total_policy_loss / n_updates
            self.last_value_loss = total_value_loss / n_updates
            self.last_entropy = total_entropy / n_updates
            self.last_approx_kl = approx_kl
            return total_loss / n_updates

        return 0.0

    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        rank = self._mpi_rank()
        is_root = rank == 0

        if is_root:
            print(f"[FlatPPO] Starting training", flush=True)
            print(f"[FlatPPO] Configuration:")
            print(f"  Re={self.cfg.Re}, mesh={self.cfg.mesh}, substeps={self.cfg.num_substeps}")
            print(f"  total_timesteps={self.cfg.total_timesteps:,}, n_steps={self.cfg.n_steps}")
            print(f"  lr={self.cfg.lr}, ent_coef={self.cfg.ent_coef}")
            print(f"  device={self.device}")
            print(f"  save_dir={self.save_dir}")

        if resume_from and is_root:
            self.load(resume_from)

        if self.verbose and is_root:
            print(f"[TRAIN] Initial reset...", flush=True)
        obs = self.env.reset()
        if self.verbose and is_root:
            print(f"[TRAIN] Initial reset complete", flush=True)

        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            comm.Barrier()
        except ImportError:
            comm = None

        t0 = time.time()
        rollout_count = 0

        while True:
            if comm is not None:
                comm.Barrier()

            keep_going = np.array([self.global_step < self.cfg.total_timesteps], dtype=np.int32)
            if comm is not None:
                comm.Bcast(keep_going, root=0)
            if not keep_going[0]:
                if is_root:
                    print(f"[FlatPPO] Training complete - reached {self.cfg.total_timesteps} steps", flush=True)
                break

            obs, ep_rewards_list = self.collect_rollout(obs)
            mean_loss = self.update()

            rollout_count += 1

            if is_root:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]

            if is_root and ep_rewards_list:
                ep_r = ep_rewards_list[-1]
                mean100 = np.mean(self.ep_rewards) if self.ep_rewards else ep_r

                ep_summary = self.env.get_episode_summary()
                ep_drag = ep_summary.get('mean_drag', None)
                ep_lift = ep_summary.get('mean_abs_lift', None)

                self._check_and_save_best(ep_r, ep_drag, ep_lift)

                # GPU memory info (if available)
                gpu_info = ""
                if self.device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(0) / 1e6
                    reserved = torch.cuda.memory_reserved(0) / 1e6
                    gpu_info = f"\n  GPU Memory:     {allocated:.0f} MB allocated / {reserved:.0f} MB reserved"

                output_lines = [
                    f"\n{'='*60}",
                    f"EPISODE {self.episode:4d} COMPLETE | Step {self.global_step:7d}/{self.cfg.total_timesteps}",
                    f"  Episode Reward: {ep_r:8.4f}",
                    f"  100-Ep Average: {mean100:8.4f}",
                ]

                if ep_drag is not None:
                    drag100 = np.mean(self.ep_drags) if self.ep_drags else ep_drag
                    output_lines.append(f"  Mean Drag:      {ep_drag:8.4f}  (100-ep avg: {drag100:8.4f})")

                if ep_lift is not None:
                    lift100 = np.mean(self.ep_lifts) if self.ep_lifts else ep_lift
                    output_lines.append(f"  Mean |Lift|:    {ep_lift:8.4f}  (100-ep avg: {lift100:8.4f})")

                output_lines.extend([
                    f"  PPO Loss:       {mean_loss:.4f}",
                    f"  Learning Rate:  {current_lr:.2e}",
                    f"  Best Reward:    {self.best_reward:.4f} (ep {self.best_episode})",
                ])

                if gpu_info:
                    output_lines.append(gpu_info)

                output_lines.append(f"{'='*60}")
                print("\n".join(output_lines), flush=True)

                if self.episode > 0 and self.episode % self.cfg.save_interval == 0:
                    self.save()

        if is_root:
            self.save(tag="final")
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETE")
            print(f"  Best Reward: {self.best_reward:.4f} (episode {self.best_episode})")
            print(f"  Best Drag:   {self.best_drag:.4f}")
            print(f"  Best |Lift|: {self.best_lift:.4f}")
            print(f"  Total time:  {time.time() - t0:.1f}s")
            print(f"{'='*60}", flush=True)

        self.env.close()

    def save(self, tag: str = None):
        """Save model checkpoint."""
        if self._mpi_rank() != 0:
            return
        label = tag or f"ep{self.episode:05d}"
        path = self.save_dir / f"flat_ppo_re{self.cfg.Re}_{label}.pt"

        torch.save({
            "episode": self.episode,
            "global_step": self.global_step,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_mean": self.env.obs_mean,
            "obs_std": self.env.obs_std,
            "cfg": self.cfg,
            "best_reward": self.best_reward,
            "best_drag": self.best_drag,
            "best_lift": self.best_lift,
        }, path)
        print(f"  Saved checkpoint -> {path}", flush=True)

    def load(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.episode = ckpt["episode"]
        self.global_step = ckpt["global_step"]
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        if hasattr(self.env, '_obs_mean'):
            self.env._obs_mean = ckpt["obs_mean"]
            self.env._obs_std = ckpt["obs_std"]
        self.env._normalizer_fitted = True

        if "best_reward" in ckpt:
            self.best_reward = ckpt["best_reward"]
        if "best_drag" in ckpt:
            self.best_drag = ckpt["best_drag"]
        if "best_lift" in ckpt:
            self.best_lift = ckpt["best_lift"]

        print(f"  Loaded checkpoint from {path}", flush=True)