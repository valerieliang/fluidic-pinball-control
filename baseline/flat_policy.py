# baseline/flat_policy.py
"""
Simple feedforward actor-critic network for flat PPO baseline.
Matches the architecture used in the HydroGym paper.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class FlatActorCritic(nn.Module):
    """
    Single feedforward network with separate actor and critic heads.
    
    Architecture: 2 hidden layers of 64 neurons with Tanh activation,
    as commonly used in the HydroGym paper baselines.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Actor head: outputs mean of Gaussian policy
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # Learnable log standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head: outputs state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (batch, obs_dim) observations
            
        Returns:
            dist: Normal distribution over actions
            value: (batch, 1) state value estimates
        """
        features = self.shared(obs)
        
        # Action distribution
        mean = self.actor_mean(features)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        # Value estimate
        value = self.critic(features)
        
        return dist, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample action from policy.
        
        Args:
            obs: (batch, obs_dim) observations
            deterministic: if True, return mean instead of sampling
            
        Returns:
            action: (batch, action_dim) sampled actions
            log_prob: (batch,) log probabilities
            value: (batch, 1) state values
        """
        dist, value = self.forward(obs)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Evaluate log probability and entropy for given action.
        Used during PPO update.
        
        Args:
            obs: (batch, obs_dim) observations
            action: (batch, action_dim) actions to evaluate
            
        Returns:
            log_prob: (batch,) log probabilities
            entropy: (batch,) distribution entropy
            value: (batch, 1) state values
        """
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value