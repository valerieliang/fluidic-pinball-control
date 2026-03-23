"""
ppo_agent.py
------------
Self-contained PPO implementation for HydroGym continuous-action environments.
No TorchRL / CleanRL dependency -- pure PyTorch + numpy.

Architecture  (paper sec4.1)
  Actor  : MLP with ReLU activations -> mu (unbounded) + state-independent log_std
           -> TanhNormal action distribution (tanh applied to sample, NOT to mean)
  Critic : MLP with ReLU activations -> scalar value estimate

Key hyper-parameters for the 2-D fluidic pinball at Re = 100  (paper sec4.1)
  lr           : 5e-5 -- 1e-3  (default 3e-4)
  batch_size   : 16 -- 48      (default 32)
  clip_eps     : 0.2
  gae_lambda   : 0.95
  entropy_coef : 0.05
  episode_len  : 600 actions   (~5 shedding periods at Re=100 with N_SKIP=10)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # network
    hidden_sizes:   List[int] = field(default_factory=lambda: [256, 256])
    log_std_init:   float     = -0.5      # initial log std ≈ 0.6

    # PPO core
    lr:             float     = 3e-4
    clip_eps:       float     = 0.2       # surrogate objective clip
    gamma:          float     = 0.99      # discount
    gae_lambda:     float     = 0.95      # GAE smoothing
    n_epochs:       int       = 10        # gradient passes per update
    batch_size:     int       = 32        # mini-batch size
    value_coef:     float     = 0.5       # critic loss weight
    entropy_coef:   float     = 0.05      # entropy bonus (paper sec4.1)
    max_grad_norm:  float     = 0.5       # gradient clipping

    # schedule
    lr_anneal:      bool      = True      # linearly decay lr to 0
    total_episodes: int       = 500       # for annealing denominator

    # observation normalisation
    obs_norm:       bool      = True      # running mean/var normalisation


# ---------------------------------------------------------------------------
# Running observation normaliser
# ---------------------------------------------------------------------------

class RunningNorm(nn.Module):
    """
    Welford running mean / variance normaliser.

    Keeps statistics on CPU (updated with numpy arrays) and exposes a
    torch-compatible forward() for use inside the network forward pass.
    Not in the paper, but standard practice for PPO on environments whose
    observation components have different scales (CD ~1-5, CL ~0.1-0.3).
    Stats are preserved in checkpoints alongside the network weights.
    """

    def __init__(self, dim: int, clip: float = 5.0):
        super().__init__()
        self.clip = clip
        # Register as buffers so they travel with .to(device) and are saved
        # in state_dict automatically.
        self.register_buffer("mean",  torch.zeros(dim))
        self.register_buffer("var",   torch.ones(dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, obs: np.ndarray):
        """Update running stats with a single observation (1-D numpy array)."""
        x     = torch.FloatTensor(obs).to(self.mean.device)
        self.count += 1
        delta      = x - self.mean
        self.mean  = self.mean + delta / self.count.float()
        delta2     = x - self.mean
        self.var   = self.var + delta * delta2   # unnormalised M2

    @property
    def std(self) -> torch.Tensor:
        # For count < 2 return ones to avoid dividing by near-zero.
        n = self.count.item()
        if n < 2:
            return torch.ones_like(self.var)
        return (self.var / (n - 1)).clamp(min=1e-4).sqrt()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normed = (obs - self.mean) / self.std
        return normed.clamp(-self.clip, self.clip)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: List[int], out_dim: int,
         act=nn.ReLU) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    TanhNormal actor.

    The network outputs an unbounded mean mu.  Tanh is applied only to the
    *sample*, not to the mean.  This is the standard formulation used in SAC
    and PPO literature.

    Previous version applied tanh to mu before passing it to Normal(), which
    meant:
      1. Gradients through the mean vanished whenever |net_out| was large.
      2. The Normal was centred inside (-1, 1), so almost every sample after
         the second tanh ended up near ±1 -- the policy saturated.

    Correct formulation:
        mu  = net(obs)          -- unbounded
        raw ~ Normal(mu, std)
        act = tanh(raw)         -- squash to (-1, 1) once
        log_prob = Normal.log_prob(raw) - log(1 - act² + ε)
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        super().__init__()
        self.net     = _mlp(obs_dim, cfg.hidden_sizes, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), cfg.log_std_init))

        # Small init on the output layer keeps early actions near zero,
        # which avoids large early actuation that can destabilise the flow.
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor):
        # FIX: mu is unbounded -- tanh is NOT applied here.
        # The previous code had:  mu = torch.tanh(self.net(obs))
        # which caused gradient vanishing and policy saturation.
        mu  = self.net(obs)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_action(self, obs: torch.Tensor):
        """Sample an action and return (action, log_prob, entropy)."""
        mu, std = self(obs)
        dist    = Normal(mu, std)
        raw     = dist.rsample()           # raw ~ Normal(mu_unbounded, std)
        action  = torch.tanh(raw)          # squash once to (-1, 1)

        # Change-of-variables correction for tanh:
        #   log p(action) = log p_Normal(raw) - log |d(tanh)/d(raw)|
        #                 = log p_Normal(raw) - log(1 - tanh(raw)²)
        log_prob = (dist.log_prob(raw)
                    - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Compute log_prob and entropy for a stored action (PPO update)."""
        mu, std = self(obs)
        dist    = Normal(mu, std)
        # Invert tanh to recover the pre-squash sample.
        raw     = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
        log_prob = (dist.log_prob(raw)
                    - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, obs_dim: int, cfg: PPOConfig):
        super().__init__()
        self.net = _mlp(obs_dim, cfg.hidden_sizes, 1)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores one full episode of transitions, then computes GAE returns
    before each PPO update.

    Pre-allocates numpy arrays at first `add()` call to avoid repeated
    list appends and final np.array() copies over long episodes.
    """

    def __init__(self):
        self._capacity = 0
        self._ptr      = 0
        # arrays allocated lazily on first add()
        self.obs       = None
        self.actions   = None
        self.log_probs = None
        self.rewards   = None
        self.values    = None
        self.dones     = None

    def reset(self):
        self._ptr = 0
        # Keep the arrays; just reset the write pointer.

    def _alloc(self, obs, action, capacity=2048):
        """Allocate storage on first call."""
        obs_dim = obs.shape[0]
        act_dim = action.shape[0]
        self._capacity = capacity
        self.obs       = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.actions   = np.zeros((capacity, act_dim),  dtype=np.float32)
        self.log_probs = np.zeros(capacity,              dtype=np.float32)
        self.rewards   = np.zeros(capacity,              dtype=np.float32)
        self.values    = np.zeros(capacity,              dtype=np.float32)
        self.dones     = np.zeros(capacity,              dtype=np.float32)

    def add(self, obs, action, log_prob, reward, value, done):
        if self.obs is None:
            self._alloc(obs, action)
        if self._ptr >= self._capacity:
            # Grow by doubling (rare).
            self._alloc(obs, action, capacity=self._capacity * 2)
            # Re-alloc wipes data; caller should not hit this in practice.
        i = self._ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.values[i]    = value
        self.dones[i]     = float(done)
        self._ptr += 1

    def compute_returns(self, last_value: float,
                        gamma: float, gae_lambda: float):
        """Generalised Advantage Estimation (GAE)."""
        T          = self._ptr
        advantages = np.zeros(T, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(T)):
            non_terminal  = 1.0 - self.dones[t]
            next_value    = last_value if t == T - 1 else self.values[t + 1]
            delta         = (self.rewards[t]
                             + gamma * next_value * non_terminal
                             - self.values[t])
            last_gae      = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:T]
        return advantages, returns

    def to_tensors(self, advantages, returns, device):
        T   = self._ptr
        obs       = torch.FloatTensor(self.obs[:T]).to(device)
        actions   = torch.FloatTensor(self.actions[:T]).to(device)
        log_probs = torch.FloatTensor(self.log_probs[:T]).to(device)
        advs      = torch.FloatTensor(advantages).to(device)
        rets      = torch.FloatTensor(returns).to(device)

        # Normalise advantages within the batch (standard PPO).
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return obs, actions, log_probs, advs, rets


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------

class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig,
                 device: str = "cuda"):
        self.cfg = cfg
        if device == "cuda" and not torch.cuda.is_available():
            print("  [PPOAgent] CUDA requested but not available -- using CPU")
            device = "cpu"
        self.device = torch.device(device)
        print(f"  [PPOAgent] device: {self.device}"
              + (f"  ({torch.cuda.get_device_name(0)})"
                 if self.device.type == "cuda" else ""))

        self.obs_norm = (RunningNorm(obs_dim).to(self.device)
                         if cfg.obs_norm else None)
        self.actor    = Actor(obs_dim, act_dim, cfg).to(self.device)
        self.critic   = Critic(obs_dim, cfg).to(self.device)
        self.optim    = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.lr, eps=1e-5,
        )
        self.buffer   = RolloutBuffer()
        self._episode = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise(self, obs_t):
        if self.obs_norm is not None:
            return self.obs_norm(obs_t)
        return obs_t

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        # Update running stats with raw observation before normalising.
        if self.obs_norm is not None and self._episode < 5:
            self.obs_norm.update(obs)

        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        obs_n  = self._normalise(obs_t)
        action, log_prob, _ = self.actor.get_action(obs_n)
        value  = self.critic(obs_n)
        return (action.squeeze(0).cpu().numpy(),
                log_prob.item(),
                value.item())

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(self, obs, action, log_prob, reward, value, done):
        self.buffer.add(obs, action, log_prob, reward, value, done)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, last_obs: np.ndarray) -> dict:
        """
        Compute GAE, then run n_epochs of mini-batch PPO updates.
        Returns a dict of scalar losses for logging.
        """
        with torch.no_grad():
            last_obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            last_v     = self.critic(self._normalise(last_obs_t)).item()

        advantages, returns = self.buffer.compute_returns(
            last_v, self.cfg.gamma, self.cfg.gae_lambda
        )

        # Normalise all stored raw observations before training.
        # We build a normalised obs tensor here so the buffer's raw obs
        # are unchanged (useful for debugging).
        raw_obs   = torch.FloatTensor(
            self.buffer.obs[:self.buffer._ptr]).to(self.device)
        norm_obs  = self._normalise(raw_obs)

        acts, old_lps, advs, rets = [
            t.to(self.device) for t in (
                torch.FloatTensor(self.buffer.actions[:self.buffer._ptr]),
                torch.FloatTensor(self.buffer.log_probs[:self.buffer._ptr]),
                torch.FloatTensor(advantages),
                torch.FloatTensor(returns),
            )
        ]
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        T       = norm_obs.shape[0]
        indices = np.arange(T)

        policy_losses, value_losses, entropy_losses = [], [], []

        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.cfg.batch_size):
                idx = indices[start: start + self.cfg.batch_size]

                new_lp, entropy = self.actor.evaluate(norm_obs[idx], acts[idx])

                approx_kl = (old_lps[idx] - new_lp).mean().item()
                if approx_kl > 0.02:
                    continue

                new_v           = self.critic(norm_obs[idx])

                ratio       = (new_lp - old_lps[idx]).exp()
                surr1       = ratio * advs[idx]
                surr2       = ratio.clamp(1 - self.cfg.clip_eps,
                                          1 + self.cfg.clip_eps) * advs[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(new_v, rets[idx])
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.cfg.value_coef   * value_loss
                        + self.cfg.entropy_coef * entropy_loss)

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.optim.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        self.buffer.reset()
        self._episode += 1

        if self.cfg.lr_anneal:
            frac = max(1.0 - self._episode / self.cfg.total_episodes, 0.0)
            for pg in self.optim.param_groups:
                pg["lr"] = self.cfg.lr * frac

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss":  float(np.mean(value_losses)),
            "entropy":     float(-np.mean(entropy_losses)),
            "total_loss":  float(np.mean(policy_losses)
                                 + self.cfg.value_coef * np.mean(value_losses)
                                 + self.cfg.entropy_coef * np.mean(entropy_losses)),
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str):
        payload = {
            "actor":    self.actor.state_dict(),
            "critic":   self.critic.state_dict(),
            "optim":    self.optim.state_dict(),
            "episode":  self._episode,
        }
        if self.obs_norm is not None:
            payload["obs_norm"] = self.obs_norm.state_dict()
        torch.save(payload, path)
        print(f"  [ckpt] saved -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optim.load_state_dict(ckpt["optim"])
        self._episode = ckpt.get("episode", 0)
        if self.obs_norm is not None and "obs_norm" in ckpt:
            self.obs_norm.load_state_dict(ckpt["obs_norm"])
        print(f"  [ckpt] loaded <- {path} (episode {self._episode})")