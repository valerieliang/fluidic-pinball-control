# envs/pinball_env.py
import os
import sys
import time
import numpy as np
import gym
from gym import spaces
import hydrogym.firedrake as hgym
from hydrogym import FlowEnv

from envs.regime_obs import RegimeObsBuffer

# Pull world rank for print labels without importing MPI here
# (MPI is already initialised by ppo_joint before this module loads)
try:
    from mpi4py import MPI as _MPI
    _W_RANK = _MPI.COMM_WORLD.Get_rank()
except ImportError:
    _W_RANK = 0

def _log(msg: str):
    """Timestamped, rank-labelled, immediately-flushed diagnostic print."""
    print(f"[R{_W_RANK} | {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class PinballEnv(gym.Env):
    """
    Gym-compatible wrapper around HydroGym's FlowEnv for the fluidic pinball.
    Accepts an optional `comm` key in the config dict for MPI subcommunicators.
    """

    metadata = {"render.modes": []}

    def __init__(self, config: dict = None):
        super().__init__()
        cfg  = config or {}
        comm = cfg.get("comm", None)

        _log(f"PinballEnv.__init__ start  (Re={cfg.get('Re',100)}, "
             f"mesh={cfg.get('mesh','medium')}, comm={'subcommunicator' if comm else 'COMM_WORLD'})")

        flow_config = {
            "mesh": cfg.get("mesh", "medium"),
            "Re":   cfg.get("Re",   100),
        }
        if comm is not None:
            flow_config["comm"] = comm

        env_config = {
            "flow":        hgym.Pinball,
            "flow_config": flow_config,
            "solver":      hgym.SemiImplicitBDF,
            "solver_config":    {"dt": cfg.get("dt", 1e-2)},
            "actuation_config": {"num_substeps": cfg.get("num_substeps", 2)},
        }

        _log("FlowEnv init start (mesh load + PETSc setup) ...")
        self._env = FlowEnv(env_config)
        _log("FlowEnv init done")

        self.n_probes   = cfg.get("n_probes",   6)
        self.buffer_len = cfg.get("buffer_len", 50)
        self.embed_dim  = cfg.get("embed_dim",  16)
        self._buf = RegimeObsBuffer(
            n_probes=self.n_probes,
            buffer_len=self.buffer_len,
            embed_dim=self.embed_dim,
        )

        raw_act = self._env.action_space
        self.action_space = spaces.Box(
            low=raw_act.low, high=raw_act.high, dtype=np.float32,
        )
        obs_dim = self.n_probes + self.embed_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        self._warmup_steps      = cfg.get("warmup_steps", 200)
        self._normalizer_fitted = False
        _log("PinballEnv.__init__ complete")

    # ------------------------------------------------------------------

    def _raw_to_array(self, raw_obs) -> np.ndarray:
        return np.array(raw_obs, dtype=np.float32)

    def _make_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        self._buf.push(raw_obs)
        embed = self._buf.embed().detach().cpu().numpy()
        return np.concatenate([raw_obs, embed])

    def _fit_normalizer(self):
        _log(f"Normalizer warmup start ({self._warmup_steps} steps) ...")
        t0      = time.time()
        history = []

        _log("  warmup env.reset() ...")
        raw = self._env.reset()
        history.append(self._raw_to_array(raw))
        _log("  warmup env.reset() done")

        log_every = max(1, self._warmup_steps // 10)   # log 10 progress marks
        for i in range(self._warmup_steps):
            if i % log_every == 0:
                elapsed = time.time() - t0
                _log(f"  warmup step {i}/{self._warmup_steps}  ({elapsed:.1f}s elapsed)")
            action = self._env.action_space.sample()
            raw, _, done, _ = self._env.step(action)
            history.append(self._raw_to_array(raw))
            if done:
                _log(f"  warmup episode done at step {i}, resetting ...")
                raw = self._env.reset()
                history.append(self._raw_to_array(raw))

        self._buf.update_normalization(np.stack(history, axis=0))
        self._normalizer_fitted = True
        _log(f"Normalizer warmup complete ({time.time()-t0:.1f}s total)")

    # ------------------------------------------------------------------

    def reset(self):
        if not self._normalizer_fitted:
            self._fit_normalizer()
        _log("env.reset() called")
        self._buf.reset()
        raw = self._env.reset()
        _log("env.reset() done")
        return self._make_obs(self._raw_to_array(raw))

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        raw_obs, reward, done, info = self._env.step(action)
        obs = self._make_obs(self._raw_to_array(raw_obs))
        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        pass

    def close(self):
        self._env.close()