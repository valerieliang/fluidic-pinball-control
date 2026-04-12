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
            # FLAT MODE: Return raw observations (no normalization here)
            # Normalization already applied to raw_obs by environment
            return raw_obs.astype(np.float32)

    def _fit_normalizer(self):
        """Fit observation normalizer - coordinated across all MPI ranks."""
        comm = _mpi_comm()
        rank = _rank()
        
        rank_str = f"[rank {rank}]"
        print(f"{rank_str} [WARMUP] Starting normalizer fitting with {self._warmup_steps} steps", flush=True)
        start_time = time.time()
        
        if self._debug:
            print(f"{rank_str} [WARMUP] warmup start ({self._warmup_steps} steps)", flush=True)
            
        history = []
        raw = self._env.reset()
        raw_array = self._raw_to_array(raw)
        history.append(raw_array)
        
        print(f"{rank_str} [WARMUP] Initial reset complete, starting collection loop", flush=True)

        for i in range(self._warmup_steps):
            # Progress reporting every 50 steps on rank 0
            if rank == 0 and i % 50 == 0:
                print(f"{rank_str} [WARMUP] Step {i}/{self._warmup_steps}", flush=True)
            
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
                print(f"{rank_str} [WARMUP] Episode done at step {i}, resetting", flush=True)
                raw = self._env.reset()

        print(f"{rank_str} [WARMUP] Collection loop complete, collected {len(history)} samples", flush=True)

        if self._use_hrssa and self._buf is not None:
            if rank == 0:
                print(f"{rank_str} [WARMUP] Computing normalization stats for HR-SSA mode", flush=True)
                self._buf.update_normalization(np.stack(history, axis=0))
            if comm is not None:
                print(f"{rank_str} [WARMUP] Broadcasting HR-SSA normalization stats", flush=True)
                self._buf.obs_mean = comm.bcast(self._buf.obs_mean, root=0)
                self._buf.obs_std = comm.bcast(self._buf.obs_std, root=0)
        else:
            # For flat mode: store normalized stats but don't apply in _make_obs
            if rank == 0:
                print(f"{rank_str} [WARMUP] Computing normalization stats for FLAT mode", flush=True)
                stacked = np.stack(history, axis=0)
                self._obs_mean = stacked.mean(axis=0)
                self._obs_std = stacked.std(axis=0).clip(1e-6)
                print(f"{rank_str} [WARMUP] Flat mode stats: mean range [{self._obs_mean.min():.3f}, {self._obs_mean.max():.3f}], std range [{self._obs_std.min():.3f}, {self._obs_std.max():.3f}]", flush=True)
            
            if comm is not None:
                print(f"{rank_str} [WARMUP] Broadcasting flat mode normalization stats", flush=True)
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
        print(f"{rank_str} [WARMUP] Normalizer fitting COMPLETE after {elapsed:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self):
        comm = _mpi_comm()
        rank = _rank()
        rank_str = f"[rank {rank}]"
        
        print(f"{rank_str} [RESET] Entering reset (normalizer_fitted={self._normalizer_fitted}, post_warmup_done={self._post_warmup_done})", flush=True)
        
        if not self._normalizer_fitted:
            print(f"{rank_str} [RESET] Normalizer not fitted, calling _fit_normalizer()", flush=True)
            self._fit_normalizer()
            print(f"{rank_str} [RESET] After normalizer fitting, resetting env", flush=True)
            raw = self._env.reset()
        else:
            print(f"{rank_str} [RESET] Normalizer already fitted, resetting env", flush=True)
            raw = self._env.reset()

        # Additional warmup: run for a few shedding cycles without recording
        # to let transients die out. Only do this ONCE after normalizer fitting.
        # Must be coordinated across all ranks!
        if not self._post_warmup_done:
            print(f"{rank_str} [RESET] Starting post-normalizer warmup (discarding transients)...", flush=True)
            if rank == 0:
                print(f"{rank_str} [RESET] Post-normalizer warmup - discarding transients for 50 steps", flush=True)
            
            # All ranks participate in the warmup steps
            for i in range(50):  # ~5 shedding cycles
                if i % 10 == 0 and rank == 0:
                    print(f"{rank_str} [RESET] Post-warmup step {i}/50", flush=True)
                    
                if rank == 0:
                    action = np.zeros(3)
                else:
                    action = None
                
                # Broadcast action
                if comm is not None:
                    action = comm.bcast(action, root=0)
                
                raw, _, _, _ = self._env.step(action)
            
            self._post_warmup_done = True
            print(f"{rank_str} [RESET] Post-normalizer warmup COMPLETE", flush=True)

        if self._use_hrssa and self._buf is not None:
            print(f"{rank_str} [RESET] Resetting HR-SSA buffer", flush=True)
            self._buf.reset()

        # Get final observation
        obs = self._make_obs(self._raw_to_array(raw))
        self._step_count = 0
        print(f"{rank_str} [RESET] Reset COMPLETE, obs shape={obs.shape}", flush=True)
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