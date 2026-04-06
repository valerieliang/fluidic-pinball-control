# envs/regime_obs.py
import numpy as np
import torch
import torch.nn as nn

class RegimeObsBuffer:
    """
    Rolling buffer of probe observations + learned spectral embedding.
    Buffer shape: (T, n_probes) — time axis first for 1D conv.
    """
    def __init__(self, n_probes=6, buffer_len=50, embed_dim=16):
        self.n_probes = n_probes
        self.buffer_len = buffer_len
        self.embed_dim = embed_dim

        # Running stats for per-probe normalization
        self.obs_mean = np.zeros(n_probes, dtype=np.float32)
        self.obs_std  = np.ones(n_probes,  dtype=np.float32)

        # 1D conv: input (batch, n_probes, T), output (batch, embed_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_probes, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, embed_dim, kernel_size=5, padding=2),
            nn.AdaptiveAvgPool1d(1),   # collapse time dim
            nn.Flatten(),
        )

        self._buf = np.zeros((buffer_len, n_probes), dtype=np.float32)
        self._ptr = 0
        self._full = False

    def update_normalization(self, obs_history: np.ndarray):
        """Call once on a warm-up rollout before training."""
        self.obs_mean = obs_history.mean(axis=0)
        self.obs_std  = obs_history.std(axis=0).clip(1e-6)

    def push(self, obs: np.ndarray):
        normed = (obs - self.obs_mean) / self.obs_std
        self._buf[self._ptr % self.buffer_len] = normed
        self._ptr += 1
        self._full = self._ptr >= self.buffer_len

    def embed(self) -> torch.Tensor:
        """Returns (embed_dim,) regime embedding, or zeros if buffer not full."""
        if not self._full:
            return torch.zeros(self.embed_dim)
        # Arrange as circular buffer in time order
        idx = self._ptr % self.buffer_len
        ordered = np.concatenate([self._buf[idx:], self._buf[:idx]], axis=0)
        x = torch.from_numpy(ordered).T.unsqueeze(0)  # (1, n_probes, T)
        with torch.no_grad():
            return self.encoder(x).squeeze(0)          # (embed_dim,)

    def reset(self):
        self._buf[:] = 0.0
        self._ptr = 0
        self._full = False