# models/manager.py
import torch
import torch.nn as nn


class Manager(nn.Module):
    """
    GRU-based manager. Consumes the spectral regime embedding from
    RegimeObsBuffer and emits goal vectors for each sub-policy.

    Also has an auxiliary head that predicts the next dominant probe
    frequency -- this gives a dense learning signal without extra CFD rollouts.

    Naming note: the linear projection layers are prefixed with `_head_` to
    avoid collision with the tensor outputs returned by forward(), which are
    named goal_stab / goal_symm for readability at the call site.
    """
    def __init__(self, embed_dim=16, hidden=64, goal_dim=8):
        super().__init__()
        self.hidden   = hidden
        self.goal_dim = goal_dim

        self.gru = nn.GRUCell(embed_dim, hidden)

        # Renamed from goal_stab/goal_symm to avoid shadowing the forward() outputs
        self._head_goal_stab  = nn.Linear(hidden, goal_dim)
        self._head_goal_symm  = nn.Linear(hidden, goal_dim)
        self._head_value      = nn.Linear(hidden, 1)
        self._head_aux_freq   = nn.Linear(hidden, 1)

    def forward(self, embed: torch.Tensor, h: torch.Tensor):
        """
        embed : (batch, embed_dim) -- from RegimeObsBuffer.embed()
        h     : (batch, hidden)   -- GRU hidden state, carried across steps

        Returns
        -------
        goal_stab     : (batch, goal_dim)
        goal_symm     : (batch, goal_dim)
        value         : (batch, 1)
        h_next        : (batch, hidden)
        predicted_freq: (batch, 1)
        """
        h_next         = self.gru(embed, h)
        goal_stab      = self._head_goal_stab(h_next)
        goal_symm      = self._head_goal_symm(h_next)
        value          = self._head_value(h_next)
        predicted_freq = self._head_aux_freq(h_next)
        return goal_stab, goal_symm, value, h_next, predicted_freq

    def bootstrap_goals(self, h: torch.Tensor):
        """
        Compute goal vectors from a hidden state without stepping the GRU.
        Used at the end of a rollout to bootstrap the last value estimate.

        h : (batch, hidden)
        Returns: goal_stab (batch, goal_dim), goal_symm (batch, goal_dim)
        """
        return self._head_goal_stab(h), self._head_goal_symm(h)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden)