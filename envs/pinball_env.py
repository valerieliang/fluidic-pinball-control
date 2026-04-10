# envs/pinball_env.py
import numpy as np
import gym
from gym import spaces
import hydrogym.firedrake as hgym
from hydrogym import FlowEnv

from envs.regime_obs import RegimeObsBuffer


def _rank() -> int:
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0


class PinballEnv(gym.Env):
    """
    Gym-compatible wrapper around HydroGym's FlowEnv for the fluidic pinball.
    
    Two modes:
      - use_hrssa=True  (default): Full observation = raw probes + regime embedding
      - use_hrssa=False (flat PPO): Observation = raw probes only
    
    Config options:
        use_hrssa: bool = True   # Toggle HR-SSA vs flat mode
        debug: bool = False      # Verbose per-step logging
    """

    metadata = {"render.modes": []}

    def __init__(self, config: dict = None):
        super().__init__()
        
        cfg = config or {}
        
        # --- Mode toggles ---
        self._use_hrssa = cfg.get("use_hrssa", True)
        self._debug = cfg.get("debug", False)
        
        # Store num_substeps for potential reward scaling
        self.num_substeps = cfg.get("num_substeps", 2)
        
        # Only print essential info when not debugging
        if self._debug:
            print(f"[rank {_rank()}][PinballEnv] __init__ start (use_hrssa={self._use_hrssa})", flush=True)

        env_config = {
            "flow": hgym.Pinball,
            "flow_config": {
                "mesh": cfg.get("mesh", "medium"),
                "Re":   cfg.get("Re", 100),
                "reward_omega": cfg.get("reward_omega", 1.0),
            },
            "solver": hgym.SemiImplicitBDF,
            "solver_config": {
                "dt": cfg.get("dt", 1e-2),
            },
            "actuation_config": {
                "num_substeps": self.num_substeps,
            },
        }

        self._env = FlowEnv(env_config)

        # --- Buffer / spectral embedding (only if HR-SSA mode) ---
        self.n_probes   = cfg.get("n_probes", 6)
        self.buffer_len = cfg.get("buffer_len", 50)
        self.embed_dim  = cfg.get("embed_dim", 16)
        
        if self._use_hrssa:
            # Fix zero or negative embed_dim ONLY if using HR-SSA
            if self.embed_dim <= 0:
                print(f"[rank {_rank()}][PinballEnv] Warning: embed_dim={self.embed_dim} is invalid, setting to 16")
                self.embed_dim = 16
            
            self._buf = RegimeObsBuffer(
                n_probes=self.n_probes,
                buffer_len=self.buffer_len,
                embed_dim=self.embed_dim,
                use_encoder=cfg.get("use_regime_encoder", True)
            )
            obs_dim = self.n_probes + self.embed_dim
        else:
            self._buf = None
            self.embed_dim = 0  # Explicitly set to 0 for flat mode
            obs_dim = self.n_probes

        # --- Spaces ---
        raw_act = self._env.action_space
        self.action_space = spaces.Box(
            low=raw_act.low,
            high=raw_act.high,
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._warmup_steps      = cfg.get("warmup_steps", 200)
        self._normalizer_fitted = False
        self._post_warmup_done  = False  # Track if we've done post-normalizer warmup

        # --- Episode termination ---
        self._max_episode_steps = 200
        self._step_count = 0

        if not self._debug:
            # Single line summary per rank
            print(f"[rank {_rank()}][PinballEnv] ready (obs_dim={obs_dim}, use_hrssa={self._use_hrssa})", flush=True)
        else:
            print(f"[rank {_rank()}][PinballEnv] __init__ done  obs_dim={obs_dim}", flush=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_to_array(self, raw_obs) -> np.ndarray:
        return np.array(raw_obs, dtype=np.float32)

    def _make_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        if self._use_hrssa and self._buf is not None:
            self._buf.push(raw_obs)
            embed = self._buf.embed().detach().cpu().numpy()
            return np.concatenate([raw_obs, embed])
        else:
            normed = (raw_obs - self.obs_mean) / (self.obs_std + 1e-8)
            return normed.astype(np.float32)

    def _fit_normalizer(self):
        if self._debug:
            print(f"[rank {_rank()}][PinballEnv] warmup start ({self._warmup_steps} steps)", flush=True)
            
        history = []
        raw = self._env.reset()
        history.append(self._raw_to_array(raw))

        for i in range(self._warmup_steps):
            action = self._env.action_space.sample()
            raw, _, done, _ = self._env.step(action)
            history.append(self._raw_to_array(raw))
            if done:
                raw = self._env.reset()
                history.append(self._raw_to_array(raw))

        if self._use_hrssa and self._buf is not None:
            self._buf.update_normalization(np.stack(history, axis=0))
        else:
            stacked = np.stack(history, axis=0)
            self._obs_mean = stacked.mean(axis=0)
            self._obs_std = stacked.std(axis=0).clip(1e-6)
            
        self._normalizer_fitted = True
        
        if self._debug:
            print(f"[rank {_rank()}][PinballEnv] warmup done", flush=True)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self):
        if not self._normalizer_fitted:
            self._fit_normalizer()
            # After fitting normalizer, reset to get a clean state
            raw = self._env.reset()
        else:
            raw = self._env.reset()

        # Additional warmup: run for a few shedding cycles without recording
        # to let transients die out. Only do this ONCE after normalizer fitting.
        if not self._post_warmup_done:
            print(f"[rank {_rank()}][PinballEnv] Post-normalizer warmup (discarding transients)...", flush=True)
            for i in range(50):  # ~5 shedding cycles
                action = np.zeros(3)  # Zero action
                raw, _, _, _ = self._env.step(action)
            self._post_warmup_done = True
            print(f"[rank {_rank()}][PinballEnv] Post-warmup complete", flush=True)

        if self._use_hrssa and self._buf is not None:
            self._buf.reset()

        obs = self._make_obs(self._raw_to_array(raw))
        self._step_count = 0
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        raw_obs, reward, done, info = self._env.step(action)
        
        # Optional: Scale reward if FlowEnv returns time-averaged instead of cumulative
        # Uncomment the line below if you want cumulative reward over the control interval
        # reward = reward * self.num_substeps
        
        obs = self._make_obs(self._raw_to_array(raw_obs))

        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            done = True
            
        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        pass

    def close(self):
        self._env.close()
        
    # ------------------------------------------------------------------
    # Properties for checkpointing
    # ------------------------------------------------------------------
    
    @property
    def obs_mean(self):
        if self._use_hrssa and self._buf is not None:
            return self._buf.obs_mean
        else:
            return getattr(self, '_obs_mean', np.zeros(self.n_probes))
            
    @property
    def obs_std(self):
        if self._use_hrssa and self._buf is not None:
            return self._buf.obs_std
        else:
            return getattr(self, '_obs_std', np.ones(self.n_probes))