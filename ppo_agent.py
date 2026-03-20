"""
ppo_agent.py
------------
Self-contained PPO implementation for HydroGym continuous-action environments.
No TorchRL / CleanRL dependency -- pure PyTorch + numpy.

Architecture  (paper sec4.1)
  Actor  : MLP with ReLU activations -> (mu, log_std) -> TanhNormal action distribution
  Critic : MLP with ReLU activations -> scalar value estimate

Key hyper-parameters for the 2-D fluidic pinball at Re = 100  (paper sec4.1)
  lr           : 5e-5 -- 1e-3  (default 3e-4)
  batch_size   : 16 -- 48      (default 32)
  clip_eps     : 0.2
  gae_lambda   : 0.95
  episode_len  : 200 actions  (~ 10 shedding periods at Re = 100)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import List, Tuple


# Configuration

@dataclass
class PPOConfig:
    # network
    hidden_sizes: List[int]  = field(default_factory=lambda: [256, 256])
    log_std_init: float      = -0.5      # initial log std approx 0.6 std

    # PPO core
    lr:           float      = 3e-4
    clip_eps:     float      = 0.2       # surrogate objective clip
    gamma:        float      = 0.99      # discount
    gae_lambda:   float      = 0.95      # GAE smoothing
    n_epochs:     int        = 10        # gradient passes per update
    batch_size:   int        = 32        # mini-batch size
    value_coef:   float      = 0.5       # critic loss weight
    entropy_coef: float      = 0.01      # entropy bonus
    max_grad_norm: float     = 0.5       # gradient clipping

    # schedule
    lr_anneal:    bool       = True      # linearly decay lr to 0
    total_episodes: int      = 500       # for annealing denominator


# Networks

def _mlp(in_dim: int, hidden: List[int], out_dim: int, act=nn.ReLU) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Outputs a (mu, std) Gaussian over actions, then squashes with tanh so that
    sampled actions live in (-1, 1) -- matching HydroGym's normalised action space.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        super().__init__()
        self.net     = _mlp(obs_dim, cfg.hidden_sizes, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), cfg.log_std_init))

        # small init on output layer keeps early actions near zero
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor):
        mu      = torch.tanh(self.net(obs))           # pre-squashed mean
        std     = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_action(self, obs: torch.Tensor):
        """Sample an action and return (action, log_prob, entropy)."""
        mu, std  = self(obs)
        dist     = Normal(mu, std)
        raw      = dist.rsample()
        action   = torch.tanh(raw)

        # log-prob accounting for tanh squashing
        log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return action, log_prob, entropy

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Compute log_prob and entropy for a stored action (for PPO update)."""
        mu, std  = self(obs)
        dist     = Normal(mu, std)
        raw      = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
        log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, obs_dim: int, cfg: PPOConfig):
        super().__init__()
        self.net = _mlp(obs_dim, cfg.hidden_sizes, 1)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# Rollout buffer

class RolloutBuffer:
    """
    Stores one full episode of transitions, then computes GAE returns
    before each PPO update.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs       : List[np.ndarray] = []
        self.actions   : List[np.ndarray] = []
        self.log_probs : List[float]      = []
        self.rewards   : List[float]      = []
        self.values    : List[float]      = []
        self.dones     : List[bool]       = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, last_value: float, gamma: float, gae_lambda: float):
        """Generalised Advantage Estimation (GAE)."""
        T          = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns    = np.zeros(T, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value        = self.values[t + 1]

            delta       = (self.rewards[t]
                           + gamma * next_value * next_non_terminal
                           - self.values[t])
            last_gae    = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def to_tensors(self, advantages, returns, device):
        obs       = torch.FloatTensor(np.array(self.obs)).to(device)
        actions   = torch.FloatTensor(np.array(self.actions)).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        advs      = torch.FloatTensor(advantages).to(device)
        rets      = torch.FloatTensor(returns).to(device)

        # normalise advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return obs, actions, log_probs, advs, rets


# PPO agent

class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig,
                 device: str = "cuda"):
        self.cfg    = cfg
        # fall back to CPU silently if the requested device is unavailable
        if device == "cuda" and not torch.cuda.is_available():
            print("  [PPOAgent] CUDA requested but not available -- using CPU")
            device = "cpu"
        self.device = torch.device(device)
        print(f"  [PPOAgent] device: {self.device}"
              + (f"  ({torch.cuda.get_device_name(0)})"
                 if self.device.type == "cuda" else ""))

        self.actor  = Actor(obs_dim, act_dim, cfg).to(self.device)
        self.critic = Critic(obs_dim, cfg).to(self.device)
        self.optim  = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.lr, eps=1e-5
        )
        self.buffer = RolloutBuffer()
        self._episode = 0

    # inference 

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        obs_t                    = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, _      = self.actor.get_action(obs_t)
        value                    = self.critic(obs_t)
        return (action.squeeze(0).cpu().numpy(),
                log_prob.item(),
                value.item())

    # storage 

    def store(self, obs, action, log_prob, reward, value, done):
        self.buffer.add(obs, action, log_prob, reward, value, done)

    # update 

    def update(self, last_obs: np.ndarray) -> dict:
        """
        Compute GAE, then run n_epochs of mini-batch PPO updates.
        Returns a dict of scalar losses for logging.
        """
        # bootstrap value for last observation
        with torch.no_grad():
            last_v = self.critic(
                torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            ).item()

        advantages, returns = self.buffer.compute_returns(
            last_v, self.cfg.gamma, self.cfg.gae_lambda
        )
        obs, acts, old_lps, advs, rets = self.buffer.to_tensors(
            advantages, returns, self.device
        )

        T = obs.shape[0]
        indices = np.arange(T)

        policy_losses, value_losses, entropy_losses, total_losses = [], [], [], []

        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.cfg.batch_size):
                idx = indices[start: start + self.cfg.batch_size]
                b_obs   = obs[idx]
                b_acts  = acts[idx]
                b_adv   = advs[idx]
                b_ret   = rets[idx]
                b_oldlp = old_lps[idx]

                new_lp, entropy = self.actor.evaluate(b_obs, b_acts)
                new_v           = self.critic(b_obs)

                ratio        = (new_lp - b_oldlp).exp()
                surr1        = ratio * b_adv
                surr2        = torch.clamp(ratio, 1 - self.cfg.clip_eps,
                                                   1 + self.cfg.clip_eps) * b_adv
                policy_loss  = -torch.min(surr1, surr2).mean()
                value_loss   = F.mse_loss(new_v, b_ret)
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.cfg.value_coef  * value_loss
                        + self.cfg.entropy_coef * entropy_loss)

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.cfg.max_grad_norm
                )
                self.optim.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())

        self.buffer.reset()
        self._episode += 1

        # optional lr annealing
        if self.cfg.lr_anneal:
            frac = 1.0 - self._episode / self.cfg.total_episodes
            for pg in self.optim.param_groups:
                pg["lr"] = self.cfg.lr * max(frac, 0.0)

        return {
            "policy_loss":  np.mean(policy_losses),
            "value_loss":   np.mean(value_losses),
            "entropy":      -np.mean(entropy_losses),
            "total_loss":   np.mean(total_losses),
        }

    # checkpointing 

    def save(self, path: str):
        torch.save({
            "actor":   self.actor.state_dict(),
            "critic":  self.critic.state_dict(),
            "optim":   self.optim.state_dict(),
            "episode": self._episode,
        }, path)
        print(f"  [ckpt] saved -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optim.load_state_dict(ckpt["optim"])
        self._episode = ckpt.get("episode", 0)
        print(f"  [ckpt] loaded <- {path} (episode {self._episode})")