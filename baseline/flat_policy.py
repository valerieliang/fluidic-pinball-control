# baseline/flat_policy.py
"""
Simple feedforward actor-critic network for flat PPO baseline.
Matches the architecture used in the HydroGym paper.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class FlatActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.action_dim = action_dim
        self.action_scale = 1.0  # Actions are clipped to [-1, 1]

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: outputs raw (unbounded) mean.
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # Learnable log standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        import numpy as np

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Actor mean gets smaller initial weights for stable early exploration
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

        # Initialize log std to give reasonable initial exploration (~0.6 std)
        nn.init.constant_(self.actor_logstd, -0.5)

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        mean = self.actor_mean(features)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)

        value = self.critic(features)

        return dist, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        dist, value = self.forward(obs)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        # Clamp to valid action range [-1, 1]
        action = torch.clamp(action, -self.action_scale, self.action_scale)

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Evaluate log probability and entropy for given actions.
        Used during PPO update.

        Args:
            obs: (batch, obs_dim) observations
            action: (batch, action_dim) actions to evaluate — already clamped
                    to [-1, 1] as stored in the rollout buffer.

        Returns:
            log_prob: (batch,) log probabilities
            entropy: (batch,) distribution entropy
            value: (batch, 1) state values
        """
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value