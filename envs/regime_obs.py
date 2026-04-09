# envs/regime_obs.py
import numpy as np
import torch
import torch.nn as nn

class RegimeObsBuffer:
    """
    Rolling buffer of probe observations + learned spectral embedding.
    Buffer shape: (T, n_probes) -- time axis first for 1D conv.
    """
    def __init__(self, n_probes=6, buffer_len=50, embed_dim=16, use_encoder=True):
        self.n_probes = n_probes
        self.buffer_len = buffer_len
        self.embed_dim = embed_dim
        self.use_encoder = use_encoder  # Toggle to enable/disable encoder

        # Running stats for per-probe normalization
        self.obs_mean = np.zeros(n_probes, dtype=np.float32)
        self.obs_std  = np.ones(n_probes,  dtype=np.float32)

        self.device = torch.device("cpu")  # Force CPU

        if self.use_encoder:
            # Validate dimensions
            if n_probes <= 0:
                raise ValueError(f"n_probes must be positive, got {n_probes}")
            if embed_dim <= 0:
                raise ValueError(f"embed_dim must be positive, got {embed_dim}")
            
            # 1D conv: input (batch, n_probes, T), output (batch, embed_dim)
            self.encoder = nn.Sequential(
                nn.Conv1d(n_probes, max(32, embed_dim), kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(max(32, embed_dim), embed_dim, kernel_size=5, padding=2),
                nn.AdaptiveAvgPool1d(1),   # collapse time dim
                nn.Flatten(),
            ).to(self.device)
            
            # Verify encoder has valid parameters
            self._validate_encoder()
        else:
            self.encoder = None
            print(f"[RegimeObsBuffer] Encoder disabled, using zero embeddings (dim={embed_dim})")

        self._buf = np.zeros((buffer_len, n_probes), dtype=np.float32)
        self._ptr = 0
        self._full = False

    def _validate_encoder(self):
        """Check that all Conv1d layers have positive channels."""
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Conv1d):
                if module.out_channels <= 0 or module.in_channels <= 0:
                    raise ValueError(
                        f"Conv1d {name} has invalid dimensions: "
                        f"in={module.in_channels}, out={module.out_channels}"
                    )
        print(f"[RegimeObsBuffer] Encoder validated: {self.encoder}")

    def update_normalization(self, obs_history: np.ndarray):
        """Call once on a warm-up rollout before training."""
        if len(obs_history) > 0:
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
            return torch.zeros(self.embed_dim, device=self.device)

        if not self.use_encoder or self.encoder is None:
            # Return zero embedding when encoder is disabled
            return torch.zeros(self.embed_dim, device=self.device)

        idx = self._ptr % self.buffer_len
        ordered = np.concatenate([self._buf[idx:], self._buf[:idx]], axis=0)

        x = torch.from_numpy(ordered).float().to(self.device)
        x = x.T.unsqueeze(0)  # (1, n_probes, T)

        with torch.no_grad():
            self.encoder.to(self.device)
            try:
                embedding = self.encoder(x).squeeze(0)
                # Validate embedding
                if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                    print("[RegimeObsBuffer] Warning: NaN/Inf in encoder output, returning zeros")
                    return torch.zeros(self.embed_dim, device=self.device)
                return embedding
            except RuntimeError as e:
                print(f"[RegimeObsBuffer] Encoder error: {e}, returning zeros")
                return torch.zeros(self.embed_dim, device=self.device)

    def reset(self):
        self._buf[:] = 0.0
        self._ptr = 0
        self._full = False
        
    def disable_encoder(self):
        """Disable the encoder and use zero embeddings."""
        self.use_encoder = False
        self.encoder = None
        print("[RegimeObsBuffer] Encoder disabled")
    
    def enable_encoder(self):
        """Re-enable the encoder if it was disabled."""
        if not self.use_encoder:
            self.use_encoder = True
            # Reinitialize encoder
            self.encoder = nn.Sequential(
                nn.Conv1d(self.n_probes, max(32, self.embed_dim), kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(max(32, self.embed_dim), self.embed_dim, kernel_size=5, padding=2),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            ).to(self.device)
            self._validate_encoder()
            print("[RegimeObsBuffer] Encoder re-enabled and reinitialized")