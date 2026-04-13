# envs/pinball_env.py
import numpy as np
import gym
from gym import spaces
import hydrogym.firedrake as hgym
from hydrogym import FlowEnv
import time

from envs.regime_obs import RegimeObsBuffer


def _rank() -> int:
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0


def _mpi_comm():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return None


class PinballEnv(gym.Env):
    """
    Gym-compatible wrapper around HydroGym's FlowEnv for the fluidic pinball.
    
    Two modes:
      - use_hrssa=True  (default): Full observation = raw probes + regime embedding
      - use_hrssa=False (flat PPO): Observation = raw probes only
    
    Config options:
        use_hrssa: bool = True   # Toggle HR-SSA vs flat mode
        debug: bool = False      # Verbose per-step logging
        verbose: bool = False    # Control general logging
    """

    metadata = {"render.modes": []}

    def __init__(self, config: dict = None):
        super().__init__()
        
        cfg = config or {}
        
        # --- Mode toggles ---
        self._use_hrssa = cfg.get("use_hrssa", True)
        self._debug = cfg.get("debug", False)
        self._verbose = cfg.get("verbose", False)
        
        # Store rank info for conditional logging
        self._rank = _rank()
        self._is_root = self._rank == 0
        
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
                if self._is_root:
                    print(f"[PinballEnv] Warning: embed_dim={self.embed_dim} is invalid, setting to 16")
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

        if self._is_root and not self._debug:
            print(f"[PinballEnv] ready (obs_dim={obs_dim}, use_hrssa={self._use_hrssa})", flush=True)
        elif self._debug:
            print(f"[rank {_rank()}][PinballEnv] __init__ done  obs_dim={obs_dim}", flush=True)

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------
    
    def _log(self, msg: str, force: bool = False):
        """Conditional logging based on verbose flag and root rank."""
        if self._is_root and (self._verbose or force or self._debug):
            print(f"{msg}", flush=True)

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
            # FLAT MODE: Return raw observations (no normalization here)
            # Normalization already applied to raw_obs by environment
            return raw_obs.astype(np.float32)

    def _fit_normalizer(self):
        """Fit observation normalizer - coordinated across all MPI ranks."""
        comm = _mpi_comm()
        rank = _rank()
        
        self._log(f"[WARMUP] Starting normalizer fitting with {self._warmup_steps} steps", force=True)
        start_time = time.time()
        
        if self._debug:
            self._log(f"[WARMUP] warmup start ({self._warmup_steps} steps)")
            
        history = []
        raw = self._env.reset()
        raw_array = self._raw_to_array(raw)
        history.append(raw_array)
        
        self._log(f"[WARMUP] Initial reset complete, starting collection loop", force=True)

        for i in range(self._warmup_steps):
            # Progress reporting every 50 steps on rank 0
            if rank == 0 and i % 50 == 0:
                self._log(f"[WARMUP] Step {i}/{self._warmup_steps}", force=True)
            
            if rank == 0:
                action = self._env.action_space.sample()
            else:
                action = None
            
            if comm is not None:
                action = comm.bcast(action, root=0)
            
            raw, _, done, _ = self._env.step(action)
            raw_array = self._raw_to_array(raw)
            history.append(raw_array)
            
            if done:
                self._log(f"[WARMUP] Episode done at step {i}, resetting")
                raw = self._env.reset()

        self._log(f"[WARMUP] Collection loop complete, collected {len(history)} samples", force=True)

        if self._use_hrssa and self._buf is not None:
            if rank == 0:
                self._log(f"[WARMUP] Computing normalization stats for HR-SSA mode")
                self._buf.update_normalization(np.stack(history, axis=0))
            if comm is not None:
                self._log(f"[WARMUP] Broadcasting HR-SSA normalization stats")
                self._buf.obs_mean = comm.bcast(self._buf.obs_mean, root=0)
                self._buf.obs_std = comm.bcast(self._buf.obs_std, root=0)
        else:
            # For flat mode: store normalized stats but don't apply in _make_obs
            if rank == 0:
                self._log(f"[WARMUP] Computing normalization stats for FLAT mode")
                stacked = np.stack(history, axis=0)
                self._obs_mean = stacked.mean(axis=0)
                self._obs_std = stacked.std(axis=0).clip(1e-6)
                self._log(f"[WARMUP] Flat mode stats: mean range [{self._obs_mean.min():.3f}, {self._obs_mean.max():.3f}], std range [{self._obs_std.min():.3f}, {self._obs_std.max():.3f}]")
            
            if comm is not None:
                self._log(f"[WARMUP] Broadcasting flat mode normalization stats")
                if rank == 0:
                    stats = np.array([*self._obs_mean, *self._obs_std])
                else:
                    stats = np.zeros(2 * self.n_probes)
                stats = comm.bcast(stats, root=0)
                if rank != 0:
                    self._obs_mean = stats[:self.n_probes]
                    self._obs_std = stats[self.n_probes:]
        
        self._normalizer_fitted = True
        elapsed = time.time() - start_time
        self._log(f"[WARMUP] Normalizer fitting COMPLETE after {elapsed:.1f}s", force=True)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self):
        comm = _mpi_comm()
        
        self._log(f"[RESET] Entering reset (normalizer_fitted={self._normalizer_fitted}, post_warmup_done={self._post_warmup_done})")
        
        if not self._normalizer_fitted:
            self._log(f"[RESET] Normalizer not fitted, calling _fit_normalizer()")
            self._fit_normalizer()
            self._log(f"[RESET] After normalizer fitting, resetting env")
            raw = self._env.reset()
        else:
            self._log(f"[RESET] Normalizer already fitted, resetting env")
            raw = self._env.reset()

        # Additional warmup: run for a few shedding cycles without recording
        # to let transients die out. Only do this ONCE after normalizer fitting.
        if not self._post_warmup_done:
            self._log(f"[RESET] Starting post-normalizer warmup (discarding transients)...", force=True)
            if self._is_root:
                self._log(f"[RESET] Post-normalizer warmup - discarding transients for 50 steps", force=True)
            
            # All ranks participate in the warmup steps
            for i in range(50):  # ~5 shedding cycles
                if i % 10 == 0 and self._is_root:
                    self._log(f"[RESET] Post-warmup step {i}/50", force=True)
                    
                if self._is_root:
                    action = np.zeros(3)
                else:
                    action = None
                
                # Broadcast action
                if comm is not None:
                    action = comm.bcast(action, root=0)
                
                raw, _, _, _ = self._env.step(action)
            
            self._post_warmup_done = True
            self._log(f"[RESET] Post-normalizer warmup COMPLETE", force=True)

        if self._use_hrssa and self._buf is not None:
            self._log(f"[RESET] Resetting HR-SSA buffer")
            self._buf.reset()

        # Get final observation
        obs = self._make_obs(self._raw_to_array(raw))
        self._step_count = 0
        self._log(f"[RESET] Reset COMPLETE, obs shape={obs.shape}")
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        raw_obs, reward, done, info = self._env.step(action)
        
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