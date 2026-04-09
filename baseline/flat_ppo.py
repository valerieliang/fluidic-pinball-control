# baseline/flat_ppo.py
"""
Flat PPO training loop matching HydroGym paper baseline.
Uses standard PPO with GAE, without hierarchical structure.
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

from envs.pinball_env import PinballEnv
from baseline.flat_policy import FlatActorCritic


@dataclass
class FlatPPOConfig:
    """Configuration for flat PPO baseline."""
    
    # Environment
    Re: int = 100
    mesh: str = "medium"
    dt: float = 1e-2
    num_substeps: int = 170
    n_probes: int = 6
    warmup_steps: int = 200
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
    
    # Logging
    log_interval: int = 1
    save_interval: int = 10
    save_dir: str = "checkpoints"
    run_name: str = "flat_ppo_baseline"
    
    @classmethod
    def from_yaml(cls, path: str) -> "FlatPPOConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
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
        
    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns."""
        n = self.ptr
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 1.0  # Terminal state after rollout
            else:
                next_value = self.values[t + 1]
                next_done = 1.0 - self.dones[t + 1]
                
            delta = self.rewards[t] + gamma * next_value * next_done - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_done * last_gae
            advantages[t] = last_gae
            
        self.advantages = advantages
        self.returns = advantages + self.values[:n]
        
        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


class FlatPPOTrainer:
    """Standard PPO trainer with flat policy network."""
    
    def __init__(self, cfg: FlatPPOConfig):
        self.cfg = cfg
        self.device = torch.device("cpu")
        print(f"[FlatPPO] Device: {self.device}")
        
        # Create environment
        self.env = PinballEnv({
            "Re": cfg.Re,
            "mesh": cfg.mesh,
            "dt": cfg.dt,
            "num_substeps": cfg.num_substeps,
            "n_probes": cfg.n_probes,
            "buffer_len": 1,  # Not used in flat PPO
            "embed_dim": 0,   # No embedding for flat PPO
            "warmup_steps": cfg.warmup_steps,
            "use_hrssa": False,  # Force flat PPO mode
            "use_regime_encoder": False,  # Disable encoder
        })
        
        # Override observation to use only raw probes (no embedding)
        obs_dim = cfg.n_probes
        action_dim = 3  # Three cylinders
        
        # Create policy
        self.policy = FlatActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        
        # Create buffer
        self.buffer = RolloutBuffer(
            n_steps=cfg.n_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        
        self.save_dir = Path(cfg.save_dir) / cfg.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.ep_rewards: deque[float] = deque(maxlen=100)
        self.global_step = 0
        self.episode = 0
        
    @staticmethod
    def _mpi_rank() -> int:
        try:
            from mpi4py import MPI
            return MPI.COMM_WORLD.Get_rank()
        except ImportError:
            return 0
            
    def collect_rollout(self, obs: np.ndarray):
        """Collect n_steps of experience."""
        rank = self._mpi_rank()
        is_root = rank == 0
        
        if is_root:
            self.buffer.reset()
            
        ep_reward = 0.0
        ep_rewards_list = []
        _zero_action = np.zeros(3, dtype=np.float32)
        
        for _ in range(self.cfg.n_steps):
            if is_root:
                # Use only raw probes (first cfg.n_probes elements)
                obs_raw = obs[:self.cfg.n_probes]
                obs_t = torch.as_tensor(obs_raw, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    action, log_prob, value = self.policy.get_action(obs_t)
                    
                action_np = action.squeeze(0).cpu().numpy()
                
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
                    ep_reward = 0.0
                    obs = self.env.reset()
            else:
                # Non-root: just step environment for MPI collectives
                _, _, done, _ = self.env.step(_zero_action)
                if done:
                    self.env.reset()
                    
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
            
        return obs, ep_rewards_list
    
    def update(self) -> float:
        """Perform PPO update on collected rollout."""
        if self._mpi_rank() != 0:
            return 0.0
            
        n = self.buffer.ptr
        
        # Convert to tensors
        obs = torch.as_tensor(self.buffer.obs[:n], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.buffer.actions[:n], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs[:n], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(self.buffer.advantages[:n], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(self.buffer.returns[:n], dtype=torch.float32, device=self.device)
        
        indices = np.arange(n)
        total_loss = 0.0
        n_updates = 0
        
        for _ in range(self.cfg.n_epochs):
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
                
                # Evaluate current policy
                log_probs, entropy, values = self.policy.evaluate(batch_obs, batch_actions)
                values = values.squeeze(-1)
                
                # PPO clipped objective
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.ent_coef * entropy_loss
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_updates += 1
                
        return total_loss / max(n_updates, 1)
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        rank = self._mpi_rank()
        is_root = rank == 0
        
        if resume_from and is_root:
            self.load(resume_from)
            
        obs = self.env.reset()
        t0 = time.time()
        
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except ImportError:
            comm = None
            
        while True:
            keep_going = np.array([self.global_step < self.cfg.total_timesteps], dtype=np.bool_)
            if comm is not None:
                comm.Bcast(keep_going, root=0)
            if not keep_going[0]:
                break
                
            obs, ep_rewards_list = self.collect_rollout(obs)
            mean_loss = self.update()
            
            if is_root and ep_rewards_list:
                ep_r = ep_rewards_list[-1]
                mean100 = np.mean(self.ep_rewards) if self.ep_rewards else ep_r
                print(
                    f"[FlatPPO] ep {self.episode:4d} | step {self.global_step:7d} | "
                    f"reward={ep_r:8.4f} | mean100={mean100:8.4f} | loss={mean_loss:.4f}",
                    flush=True
                )
                
                if self.episode > 0 and self.episode % self.cfg.save_interval == 0:
                    self.save()
                    
        if is_root:
            self.save(tag="final")
            
        print(f"[FlatPPO rank {rank}] Training complete in {time.time() - t0:.1f}s", flush=True)
        self.env.close()
        
    def save(self, tag: str = None):
        """Save model checkpoint."""
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
        }, path)
        print(f"  checkpoint -> {path}", flush=True)
        
    def load(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.episode = ckpt["episode"]
        self.global_step = ckpt["global_step"]
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.env._use_hrssa and self.env._buf is not None:
            self.env._buf.obs_mean = ckpt["obs_mean"]
            self.env._buf.obs_std = ckpt["obs_std"]
        else:
            self.env._obs_mean = ckpt["obs_mean"]
            self.env._obs_std = ckpt["obs_std"]
        self.env._normalizer_fitted = True
        print(f"  loaded checkpoint from {path}", flush=True)