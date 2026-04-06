# models/manager.py
import torch
import torch.nn as nn


class Manager(nn.Module):
    """
    GRU-based manager. Consumes the spectral regime embedding from
    RegimeObsBuffer and emits goal vectors for each sub-policy.

    Also has an auxiliary head that predicts the next dominant probe
    frequency — this gives a dense learning signal without extra CFD rollouts.
    """
    def __init__(self, embed_dim=16, hidden=64, goal_dim=8):
        super().__init__()
        self.hidden = hidden
        self.goal_dim = goal_dim

        self.gru = nn.GRUCell(embed_dim, hidden)

        # Two goal heads — one per sub-policy
        self.goal_stab = nn.Linear(hidden, goal_dim)
        self.goal_symm = nn.Linear(hidden, goal_dim)

        # Manager's own value function
        self.value_head = nn.Linear(hidden, 1)

        # Aux head: predict next Strouhal frequency from current hidden state
        # Target: scalar dominant frequency estimated from probe FFT
        self.aux_freq_head = nn.Linear(hidden, 1)

    def forward(self, embed, h):
        """
        embed: (batch, embed_dim) — from RegimeObsBuffer.embed()
        h:     (batch, hidden)    — GRU hidden state, carried across steps
        Returns: goal_stab, goal_symm, value, h_next, predicted_freq
        """
        h_next = self.gru(embed, h)
        goal_stab     = self.goal_stab(h_next)
        goal_symm     = self.goal_symm(h_next)
        value         = self.value_head(h_next)
        predicted_freq = self.aux_freq_head(h_next)
        return goal_stab, goal_symm, value, h_next, predicted_freq

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden)