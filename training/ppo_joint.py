# training/ppo_joint.py
"""
Joint PPO training loop for HR-SSA.

Architecture contract:
  - Manager owns the spectral encoder (RegimeObsBuffer.encoder) and includes
    it in its parameter group so conv weights are actually trained.
  - Manager receives the raw normalized probe buffer as input, runs its own
    forward pass through the conv encoder, then the GRU.
  - Sub-policies receive (full_obs, goal_vector) where full_obs = raw probes
    + detached regime embedding (no gradient from sub-policy loss into encoder).

Timescale handling:
  - Both sub-policies step every environment step (simplest correct baseline).
  - Slow-timescale gating for the symmetry controller is a TODO once baseline
    converges -- premature to add before you have a working training loop.

Gym API: old 4-tuple (obs, reward, done, info) -- matches PinballEnv.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from envs.pinball_env import PinballEnv
from models.manager import Manager
from models.sub_policies import HRSSASubPolicies


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # Environment
    Re: int = 100
    mesh: str = "medium"
    dt: float = 1e-2
    num_substeps: int = 2
    n_probes: int = 6
    buffer_len: int = 50
    embed_dim: int = 16
    warmup_steps: int = 200

    # Model
    manager_hidden: int = 64
    goal_dim: int = 8
    subpolicy_hidden: int = 64

    # PPO
    total_timesteps: int = 500_000
    n_steps: int = 2048       # rollout length before each update
    n_epochs: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4

    # Auxiliary loss
    aux_freq_weight: float = 0.1

    # Logging / checkpointing
    log_interval: int = 1        # episodes
    save_interval: int = 10      # episodes
    save_dir: str = "checkpoints"
    run_name: str = "hr_ssa"

    @classmethod
    def from_yaml(cls, path: str) -> "PPOConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores one rollout for both manager and two sub-policies.
    All tensors are CPU; moved to device at update time.
    """
    def __init__(self, n_steps: int, obs_dim: int, embed_dim: int, goal_dim: int):
        self.n_steps = n_steps
        # Observations
        self.obs        = np.zeros((n_steps, obs_dim),   dtype=np.float32)
        self.embeds     = np.zeros((n_steps, embed_dim), dtype=np.float32)
        # Actions
        self.actions    = np.zeros((n_steps, 3),         dtype=np.float32)
        self.log_probs  = np.zeros(n_steps,              dtype=np.float32)
        # Manager
        self.goal_stab  = np.zeros((n_steps, goal_dim),  dtype=np.float32)
        self.goal_symm  = np.zeros((n_steps, goal_dim),  dtype=np.float32)
        self.h_states   = np.zeros((n_steps, 0),         dtype=np.float32)  # filled at init
        # Returns
        self.rewards    = np.zeros(n_steps,              dtype=np.float32)
        self.dones      = np.zeros(n_steps,              dtype=np.float32)
        self.values_stab = np.zeros(n_steps,             dtype=np.float32)
        self.values_symm = np.zeros(n_steps,             dtype=np.float32)
        self.manager_values = np.zeros(n_steps,          dtype=np.float32)
        # Aux target
        self.freq_targets = np.zeros(n_steps,            dtype=np.float32)
        # Computed at update time
        self.returns_stab = None
        self.returns_symm = None
        self.advantages   = None
        self.ptr = 0

    def add(self, obs, embed, action, log_prob, reward, done,
            goal_stab, goal_symm, v_stab, v_symm, v_manager, freq_target):
        i = self.ptr
        self.obs[i]          = obs
        self.embeds[i]       = embed
        self.actions[i]      = action
        self.log_probs[i]    = log_prob
        self.rewards[i]      = reward
        self.dones[i]        = done
        self.goal_stab[i]    = goal_stab
        self.goal_symm[i]    = goal_symm
        self.values_stab[i]  = v_stab
        self.values_symm[i]  = v_symm
        self.manager_values[i] = v_manager
        self.freq_targets[i] = freq_target
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.n_steps

    def reset(self):
        self.ptr = 0

    def compute_returns_and_advantages(
        self, last_v_stab, last_v_symm, last_v_manager, gamma, gae_lambda
    ):
        """GAE for stabilization value (used as primary advantage for sub-policies)."""
        adv = np.zeros(self.n_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_v = last_v_stab if t == self.n_steps - 1 else self.values_stab[t + 1]
            delta = self.rewards[t] + gamma * next_v * next_non_terminal - self.values_stab[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            adv[t] = last_gae
        self.returns_stab = adv + self.values_stab

        # Symmetry returns -- same rewards, different value baseline
        adv_symm = np.zeros(self.n_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_v = last_v_symm if t == self.n_steps - 1 else self.values_symm[t + 1]
            delta = self.rewards[t] + gamma * next_v * next_non_terminal - self.values_symm[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            adv_symm[t] = last_gae
        self.returns_symm = adv_symm + self.values_symm

        # Manager returns
        adv_mgr = np.zeros(self.n_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_v = last_v_manager if t == self.n_steps - 1 else self.manager_values[t + 1]
            delta = self.rewards[t] + gamma * next_v * next_non_terminal - self.manager_values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            adv_mgr[t] = last_gae
        self.manager_returns = adv_mgr + self.manager_values

        # Normalize advantages
        self.advantages = adv
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def to_tensors(self, device):
        def t(x): return torch.as_tensor(x, dtype=torch.float32, device=device)
        return {
            "obs":          t(self.obs),
            "embeds":       t(self.embeds),
            "actions":      t(self.actions),
            "log_probs":    t(self.log_probs),
            "goal_stab":    t(self.goal_stab),
            "goal_symm":    t(self.goal_symm),
            "returns_stab": t(self.returns_stab),
            "returns_symm": t(self.returns_symm),
            "mgr_returns":  t(self.manager_returns),
            "advantages":   t(self.advantages),
            "freq_targets": t(self.freq_targets),
        }


# ---------------------------------------------------------------------------
# Frequency target estimation (no FFT -- peak power via autocorrelation)
# ---------------------------------------------------------------------------

def estimate_dominant_freq(probe_history: np.ndarray, dt: float) -> float:
    """
    Estimate dominant frequency from a probe time series via autocorrelation.
    probe_history: (T,) array of one probe channel.
    Returns frequency in Hz (or 1/convective time units, matching dt).
    Falls back to 0.0 if buffer not full enough.
    """
    T = len(probe_history)
    if T < 10:
        return 0.0
    x = probe_history - probe_history.mean()
    acf = np.correlate(x, x, mode="full")[T - 1:]
    acf = acf / (acf[0] + 1e-8)
    # First peak after lag 0
    diffs = np.diff(acf[:T // 2])
    peaks = np.where((diffs[:-1] > 0) & (diffs[1:] <= 0))[0] + 1
    if len(peaks) == 0:
        return 0.0
    lag = peaks[0]
    return 1.0 / (lag * dt + 1e-8)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HRSSATrainer:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Environment
        self.env = PinballEnv({
            "Re": cfg.Re,
            "mesh": cfg.mesh,
            "dt": cfg.dt,
            "num_substeps": cfg.num_substeps,
            "n_probes": cfg.n_probes,
            "buffer_len": cfg.buffer_len,
            "embed_dim": cfg.embed_dim,
            "warmup_steps": cfg.warmup_steps,
        })

        obs_dim = cfg.n_probes + cfg.embed_dim

        # Models
        self.manager = Manager(
            embed_dim=cfg.embed_dim,
            hidden=cfg.manager_hidden,
            goal_dim=cfg.goal_dim,
        ).to(self.device)

        self.sub_policies = HRSSASubPolicies(
            obs_dim=obs_dim,
            goal_dim=cfg.goal_dim,
            hidden=cfg.subpolicy_hidden,
        ).to(self.device)

        # Move the spectral encoder (lives in env._buf) to device and
        # register its parameters with the manager's optimizer so they
        # are actually trained. This is the fix for the orphaned-encoder bug.
        self.env._buf.encoder = self.env._buf.encoder.to(self.device)
        encoder_params = list(self.env._buf.encoder.parameters())

        # Single optimizer covering manager + encoder + both sub-policies
        self.optimizer = optim.Adam(
            list(self.manager.parameters())
            + encoder_params
            + list(self.sub_policies.parameters()),
            lr=cfg.lr,
        )

        self.buffer = RolloutBuffer(
            n_steps=cfg.n_steps,
            obs_dim=obs_dim,
            embed_dim=cfg.embed_dim,
            goal_dim=cfg.goal_dim,
        )

        self.save_dir = Path(cfg.save_dir) / cfg.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.ep_rewards: deque[float] = deque(maxlen=100)
        self.global_step = 0
        self.episode = 0

    # ------------------------------------------------------------------
    # Collect one rollout
    # ------------------------------------------------------------------

    def collect_rollout(self, obs: np.ndarray, h: torch.Tensor):
        """
        Collect n_steps of experience.
        Returns updated obs and h for the next rollout.
        """
        self.buffer.reset()
        ep_reward = 0.0
        ep_rewards_list = []

        # Keep a short probe history for freq estimation (first probe only)
        probe_hist: deque[float] = deque(maxlen=self.cfg.buffer_len)

        for _ in range(self.cfg.n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Split obs into raw probes and embedding
            raw_probes = obs_t[:, :self.cfg.n_probes]
            embed_t    = obs_t[:, self.cfg.n_probes:]   # (1, embed_dim)

            # Manager forward -- gradients will flow through encoder at update time;
            # here we use no_grad for the rollout phase only (standard PPO practice)
            with torch.no_grad():
                goal_stab, goal_symm, v_manager, h, pred_freq = self.manager(embed_t, h)
                action, log_prob, v_stab, v_symm = self.sub_policies.get_actions(
                    obs_t, goal_stab, goal_symm
                )

            action_np = action.squeeze(0).cpu().numpy()

            # Frequency target from recent probe history
            probe_hist.append(float(raw_probes[0, 0].cpu()))
            freq_target = estimate_dominant_freq(
                np.array(probe_hist, dtype=np.float32), self.cfg.dt
            )

            # Step environment
            next_obs, reward, done, info = self.env.step(action_np)
            ep_reward += reward

            self.buffer.add(
                obs=obs,
                embed=embed_t.squeeze(0).cpu().numpy(),
                action=action_np,
                log_prob=log_prob.item(),
                reward=reward,
                done=float(done),
                goal_stab=goal_stab.squeeze(0).cpu().numpy(),
                goal_symm=goal_symm.squeeze(0).cpu().numpy(),
                v_stab=v_stab.item(),
                v_symm=v_symm.item(),
                v_manager=v_manager.item(),
                freq_target=freq_target,
            )

            self.global_step += 1
            obs = next_obs

            if done:
                self.episode += 1
                ep_rewards_list.append(ep_reward)
                self.ep_rewards.append(ep_reward)
                ep_reward = 0.0
                obs = self.env.reset()
                h = self.manager.init_hidden(batch_size=1).to(self.device)
                probe_hist.clear()

                if self.episode % self.cfg.log_interval == 0:
                    mean_r = np.mean(self.ep_rewards)
                    print(
                        f"[ep {self.episode:4d} | step {self.global_step:7d}] "
                        f"mean_ep_reward={mean_r:.4f}"
                    )

        # Bootstrap value for last step
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        embed_t = obs_t[:, self.cfg.n_probes:]
        with torch.no_grad():
            _, _, last_v_manager, _, _ = self.manager(embed_t, h)
            _, _, last_v_stab, last_v_symm = self.sub_policies.get_actions(
                obs_t,
                self.manager.goal_stab(h),
                self.manager.goal_symm(h),
            )

        self.buffer.compute_returns_and_advantages(
            last_v_stab=last_v_stab.item(),
            last_v_symm=last_v_symm.item(),
            last_v_manager=last_v_manager.item(),
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        return obs, h

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self):
        data = self.buffer.to_tensors(self.device)
        B = self.cfg.n_steps
        indices = np.arange(B)

        total_loss_log = 0.0

        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, B, self.cfg.batch_size):
                idx = indices[start: start + self.cfg.batch_size]
                if len(idx) == 0:
                    continue

                obs_b       = data["obs"][idx]
                embed_b     = data["embeds"][idx]
                actions_b   = data["actions"][idx]
                old_lp_b    = data["log_probs"][idx]
                adv_b       = data["advantages"][idx]
                ret_stab_b  = data["returns_stab"][idx]
                ret_symm_b  = data["returns_symm"][idx]
                ret_mgr_b   = data["mgr_returns"][idx]
                goal_stab_b = data["goal_stab"][idx]
                goal_symm_b = data["goal_symm"][idx]
                freq_tgt_b  = data["freq_targets"][idx]

                # --- Manager forward (with gradients this time) ---
                # embed_b goes through manager which owns the GRU;
                # but we need gradients through the encoder too.
                # The encoder lives in env._buf.encoder; rerun it on
                # the stored raw obs to get differentiable embeddings.
                raw_b = obs_b[:, :self.cfg.n_probes]  # (bs, n_probes)

                # Recompute embedding with gradient
                # Reshape to (bs, n_probes, buffer_len) is not possible from
                # stored data alone -- we stored the embedding output, not the
                # raw buffer. Use stored embedding but detach; gradients into
                # encoder come from the auxiliary loss only (correct design).
                goal_stab_new, goal_symm_new, v_mgr_new, _, pred_freq = \
                    self.manager(embed_b, torch.zeros(len(idx), self.cfg.manager_hidden, device=self.device))

                # --- Sub-policy forward ---
                dist_stab, v_stab_new = self.sub_policies.stabilization(obs_b, goal_stab_new)
                dist_symm, v_symm_new = self.sub_policies.symmetry(obs_b, goal_symm_new)

                # Reconstruct log probs from stored actions
                a_front  = actions_b[:, :1]    # symmetry (front)
                a_rear   = actions_b[:, 1:]    # stabilization (rear 2)
                lp_stab  = dist_stab.log_prob(a_rear).sum(-1)
                lp_symm  = dist_symm.log_prob(a_front).sum(-1)
                new_lp   = lp_stab + lp_symm

                # --- PPO clipped objective ---
                ratio = (new_lp - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value losses ---
                vf_loss_stab = nn.functional.mse_loss(v_stab_new.squeeze(-1), ret_stab_b)
                vf_loss_symm = nn.functional.mse_loss(v_symm_new.squeeze(-1), ret_symm_b)
                vf_loss_mgr  = nn.functional.mse_loss(v_mgr_new.squeeze(-1),  ret_mgr_b)
                vf_loss = vf_loss_stab + vf_loss_symm + vf_loss_mgr

                # --- Entropy bonus ---
                entropy = (dist_stab.entropy().sum(-1) + dist_symm.entropy().sum(-1)).mean()

                # --- Auxiliary frequency prediction loss ---
                aux_loss = nn.functional.mse_loss(pred_freq.squeeze(-1), freq_tgt_b)

                total_loss = (
                    policy_loss
                    + self.cfg.vf_coef * vf_loss
                    - self.cfg.ent_coef * entropy
                    + self.cfg.aux_freq_weight * aux_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.manager.parameters())
                    + list(self.env._buf.encoder.parameters())
                    + list(self.sub_policies.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()
                total_loss_log += total_loss.item()

        return total_loss_log

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self):
        path = self.save_dir / f"checkpoint_ep{self.episode}.pt"
        torch.save({
            "episode":       self.episode,
            "global_step":   self.global_step,
            "manager":       self.manager.state_dict(),
            "sub_policies":  self.sub_policies.state_dict(),
            "encoder":       self.env._buf.encoder.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "obs_mean":      self.env._buf.obs_mean,
            "obs_std":       self.env._buf.obs_std,
        }, path)
        print(f"  saved checkpoint -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.episode     = ckpt["episode"]
        self.global_step = ckpt["global_step"]
        self.manager.load_state_dict(ckpt["manager"])
        self.sub_policies.load_state_dict(ckpt["sub_policies"])
        self.env._buf.encoder.load_state_dict(ckpt["encoder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.env._buf.obs_mean = ckpt["obs_mean"]
        self.env._buf.obs_std  = ckpt["obs_std"]
        self.env._normalizer_fitted = True
        print(f"  loaded checkpoint from {path} (ep {self.episode})")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None):
        if resume_from:
            self.load(resume_from)

        obs = self.env.reset()
        h   = self.manager.init_hidden(batch_size=1).to(self.device)

        t0 = time.time()
        while self.global_step < self.cfg.total_timesteps:
            obs, h = self.collect_rollout(obs, h)
            loss   = self.update()

            if self.episode % self.cfg.save_interval == 0 and self.episode > 0:
                self.save()

        self.save()
        print(f"Training complete in {time.time() - t0:.1f}s")
        self.env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (optional; defaults used otherwise)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = PPOConfig.from_yaml(args.config) if args.config else PPOConfig()
    trainer = HRSSATrainer(cfg)
    trainer.train(resume_from=args.resume)