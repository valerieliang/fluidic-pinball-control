import torch
from models.manager import Manager
from models.sub_policies import HRSSASubPolicies

embed_dim = 16
goal_dim  = 8
obs_dim   = 6 + embed_dim   # probes + embed
batch     = 4

manager     = Manager(embed_dim=embed_dim, hidden=64, goal_dim=goal_dim)
sub_policies = HRSSASubPolicies(obs_dim=obs_dim, goal_dim=goal_dim, hidden=64)

embed  = torch.randn(batch, embed_dim)
obs    = torch.randn(batch, obs_dim)
h      = manager.init_hidden(batch)

goal_stab, goal_symm, v_mgr, h_next, pred_freq = manager(embed, h)
action, log_prob, v_stab, v_symm = sub_policies.get_actions(
    obs, goal_stab, goal_symm
)

print("goal_stab:  ", goal_stab.shape)    # (4, 8)
print("goal_symm:  ", goal_symm.shape)    # (4, 8)
print("action:     ", action.shape)       # (4, 3)
print("log_prob:   ", log_prob.shape)     # (4,)
print("v_manager:  ", v_mgr.shape)        # (4, 1)
print("v_stab:     ", v_stab.shape)       # (4, 1)
print("v_symm:     ", v_symm.shape)       # (4, 1)
print("pred_freq:  ", pred_freq.shape)    # (4, 1)
print("h_next:     ", h_next.shape)       # (4, 64)
print("All shapes correct.")