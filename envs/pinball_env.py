# envs/pinball_env.py
import numpy as np
import gym
from gym import spaces
import hydrogym.firedrake as hgym
from hydrogym import FlowEnv

from envs.regime_obs import RegimeObsBuffer


class PinballEnv(gym.Env):
    """
    Gym-compatible wrapper around HydroGym's FlowEnv for the fluidic pinball.
    Adapts the old 4-tuple step API and adds the regime spectral embedding
    to the observation.

    Observation: concatenation of raw probe readings (6,) and the
                 regime spectral embedding (embed_dim,) from RegimeObsBuffer.
    Action:      three cylinder angular velocities, passed through as-is.
    """

    metadata = {"render.modes": []}

    def __init__(self, config: dict = None):
        super().__init__()

        cfg = config or {}

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

        self._env = FlowEnv(env_config)

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_to_array(self, raw_obs) -> np.ndarray:
        return np.array(raw_obs, dtype=np.float32)

    def _make_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        self._buf.push(raw_obs)
        embed = self._buf.embed().numpy()
        return np.concatenate([raw_obs, embed])

    def _fit_normalizer(self):
        history = []
        raw = self._env.reset()
        history.append(self._raw_to_array(raw))

        for _ in range(self._warmup_steps):
            action = self._env.action_space.sample()
            raw, _, done, _ = self._env.step(action)
            history.append(self._raw_to_array(raw))
            if done:
                raw = self._env.reset()
                history.append(self._raw_to_array(raw))

        self._buf.update_normalization(np.stack(history, axis=0))
        self._normalizer_fitted = True

    # ------------------------------------------------------------------
    # Gym API (old 4-tuple style to match HydroGym internals)
    # ------------------------------------------------------------------

    def reset(self):
        if not self._normalizer_fitted:
            self._fit_normalizer()

        self._buf.reset()
        raw = self._env.reset()
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