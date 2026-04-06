# models/sub_policies.py
import torch
import torch.nn as nn
from torch.distributions import Normal

class SubPolicy(nn.Module):
    """
    Single actor-critic head for one sub-policy (stabilization or symmetry).
    
    Input:  probe obs (6,) + regime embedding (embed_dim,) + goal (goal_dim,)
    Output: action mean/std for its assigned cylinders, and a value estimate
    """
    def __init__(self, obs_dim, action_dim, goal_dim, hidden=64):
        super().__init__()
        inp = obs_dim + goal_dim

        self.actor = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.action_mean = nn.Linear(hidden, action_dim)
        self.action_logstd = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        h = self.actor(x)
        mean = self.action_mean(h)
        std  = self.action_logstd.exp().expand_as(mean)
        return Normal(mean, std), self.critic(x)

    def get_action(self, obs, goal):
        dist, value = self.forward(obs, goal)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value


class HRSSASubPolicies(nn.Module):
    """
    Two sub-policies with distinct action dimensions:
      - stabilization: rear two cylinders  (action_dim=2)
      - symmetry:      front cylinder      (action_dim=1)
    """
    def __init__(self, obs_dim, goal_dim, hidden=64):
        super().__init__()
        self.stabilization = SubPolicy(obs_dim, action_dim=2,
                                       goal_dim=goal_dim, hidden=hidden)
        self.symmetry      = SubPolicy(obs_dim, action_dim=1,
                                       goal_dim=goal_dim, hidden=hidden)

    def get_actions(self, obs, goal_stab, goal_symm):
        a_stab, lp_stab, v_stab = self.stabilization.get_action(obs, goal_stab)
        a_symm, lp_symm, v_symm = self.symmetry.get_action(obs, goal_symm)

        # Concatenate into full 3-cylinder action: [front, rear_1, rear_2]
        action   = torch.cat([a_symm, a_stab], dim=-1)
        log_prob = lp_stab + lp_symm
        return action, log_prob, v_stab, v_symm