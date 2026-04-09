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


def _rlog(msg: str, force: bool = False):
    """Log only if debug is enabled or force=True."""
    # We'll set the debug flag on the class instance, but need a fallback here
    # This function gets replaced in __init__ with a bound method
    pass


class PinballEnv(gym.Env):
    """
    Gym-compatible wrapper around HydroGym's FlowEnv for the fluidic pinball.
    
    Set debug=True in config to enable verbose per-step logging.
    Example: PinballEnv({"Re": 100, "debug": True})
    """

    metadata = {"render.modes": []}

    def __init__(self, config: dict = None):
        super().__init__()
        
        cfg = config or {}
        self._debug = cfg.get("debug", False)
        
        # Replace the global _rlog with a bound method that respects self._debug
        self._log = lambda msg, force=False: print(
            f"[rank {_rank()}][PinballEnv] {msg}", flush=True
        ) if (self._debug or force) else None
        
        if self._debug:
            self._log("__init__ start", force=True)
        else:
            print(f"[rank {_rank()}][PinballEnv] __init__ start", flush=True)

        env_config = {
            "flow": hgym.Pinball,
            "flow_config": {
                "mesh": cfg.get("mesh", "medium"),
                "Re":   cfg.get("Re", 100),
            },
            "solver": hgym.SemiImplicitBDF,
            "solver_config": {
                "dt": cfg.get("dt", 1e-2),
            },
            "actuation_config": {
                "num_substeps": cfg.get("num_substeps", 2),
            },
        }

        self._log("calling FlowEnv() (mesh partition + PETSc init)...")
        self._env = FlowEnv(env_config)
        self._log("FlowEnv() done")

        # --- Buffer / spectral embedding ---
        self.n_probes   = cfg.get("n_probes",   6)
        self.buffer_len = cfg.get("buffer_len", 50)
        self.embed_dim  = cfg.get("embed_dim",  16)
        self._buf = RegimeObsBuffer(
            n_probes=self.n_probes,
            buffer_len=self.buffer_len,
            embed_dim=self.embed_dim,
        )

        # --- Spaces ---
        raw_act = self._env.action_space
        self.action_space = spaces.Box(
            low=raw_act.low,
            high=raw_act.high,
            dtype=np.float32,
        )

        obs_dim = self.n_probes + self.embed_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._warmup_steps      = cfg.get("warmup_steps", 200)
        self._normalizer_fitted = False

        # --- Episode termination ---
        self._max_episode_steps = 200
        self._step_count = 0

        if not self._debug:
            print(f"[rank {_rank()}][PinballEnv] __init__ done  obs_dim={obs_dim}  warmup_steps={self._warmup_steps}", flush=True)
        else:
            self._log(f"__init__ done  obs_dim={obs_dim}  warmup_steps={self._warmup_steps}", force=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_to_array(self, raw_obs) -> np.ndarray:
        return np.array(raw_obs, dtype=np.float32)

    def _make_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        self._buf.push(raw_obs)
        embed = self._buf.embed().detach().cpu().numpy()
        return np.concatenate([raw_obs, embed])

    def _fit_normalizer(self):
        self._log(f"_fit_normalizer start  ({self._warmup_steps} warmup steps)")
        history = []
        raw = self._env.reset()
        history.append(self._raw_to_array(raw))

        for i in range(self._warmup_steps):
            if i % 50 == 0 and self._debug:
                self._log(f"  warmup step {i}/{self._warmup_steps}")
            action = self._env.action_space.sample()
            raw, _, done, _ = self._env.step(action)
            history.append(self._raw_to_array(raw))
            if done:
                self._log(f"  warmup reset at step {i}")
                raw = self._env.reset()
                history.append(self._raw_to_array(raw))

        self._buf.update_normalization(np.stack(history, axis=0))
        self._normalizer_fitted = True
        self._log("_fit_normalizer done")

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self):
        self._log("reset() called")
        if not self._normalizer_fitted:
            self._fit_normalizer()

        self._buf.reset()
        raw = self._env.reset()
        obs = self._make_obs(self._raw_to_array(raw))

        self._step_count = 0
        self._log("reset() done")
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        raw_obs, reward, done, info = self._env.step(action)
        obs = self._make_obs(self._raw_to_array(raw_obs))

        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            done = True
            self._log(f"step() -> episode limit reached, done=True  reward={reward:.4f}")

        # Only log done=True messages if debug is on
        if done and self._debug:
            self._log(f"step() -> done=True  reward={reward:.4f}")
            
        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        pass

    def close(self):
        self._log("close() called")
        self._env.close()
        self._log("close() done")