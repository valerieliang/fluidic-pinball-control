# training/ppo_joint.py
"""
Joint PPO training loop for HR-SSA.

Architecture contract:
  - Manager owns the GRU and goal heads. The spectral encoder (in env._buf)
    is registered in the optimizer so its conv weights are actually trained.
  - Encoder gradients come only from the auxiliary frequency prediction loss
    (embed stored as numpy during rollout -> no policy gradient into encoder).
  - Sub-policies receive (full_obs, goal_vector); full_obs = raw probes +
    detached regime embedding.
  - Symmetry controller goal is updated every symm_timescale_ratio steps;
    stabilization goal updates every step.

Gym API: old 4-tuple (obs, reward, done, info) -- matches PinballEnv.

Paper references (Table SI 4):
  Re=100 2D: num_substeps=170, St~0.088, C_D_bar~2.904
  Re=150 2D: num_substeps=100, St~0.120, C_D_bar~2.922
  Re=150 3D: num_substeps=200, St~0.113, C_D_bar~2.949
  Reward: r = -|sum C_D,i| - omega*|sum C_L,i|, omega=1.0
  Episode length: 200 control actions (~10 shedding cycles)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from envs.pinball_env import PinballEnv
from models.manager import Manager
from models.sub_policies import HRSSASubPolicies


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # Environment
    Re: int = 100
    mesh: str = "medium"
    refinement_level: int = 12
    dt: float = 1e-2
    num_substeps: int = 170       # paper Table SI 4: Re=100 2D -> 170
    n_probes: int = 6
    buffer_len: int = 35          # ~5 shedding cycles at Re=100
    embed_dim: int = 16
    warmup_steps: int = 200

    # Reward (paper: r = -|sum C_D,i| - omega*|sum C_L,i|)
    reward_omega: float = 1.0

    # Model
    manager_hidden: int = 64
    goal_dim: int = 8
    subpolicy_hidden: int = 64

    # Timescale
    symm_timescale_ratio: int = 3  # symmetry goal updates every N stab steps

    # PPO
    total_timesteps: int = 500_000
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4

    # Auxiliary loss
    aux_freq_weight: float = 0.1
    strouhal_ref: float = 0.088   # ground truth for sanity checks only

    # Logging / checkpointing
    log_interval: int = 1
    save_interval: int = 10
    save_dir: str = "checkpoints"
    run_name: str = "hr_ssa"

    @classmethod
    def from_yaml(cls, path: str) -> "PPOConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, embed_dim: int, goal_dim: int):
        self.n_steps      = n_steps
        self.obs          = np.zeros((n_steps, obs_dim),   dtype=np.float32)
        self.embeds       = np.zeros((n_steps, embed_dim), dtype=np.float32)
        self.actions      = np.zeros((n_steps, 3),         dtype=np.float32)
        self.log_probs    = np.zeros(n_steps,              dtype=np.float32)
        self.goal_stab    = np.zeros((n_steps, goal_dim),  dtype=np.float32)
        self.goal_symm    = np.zeros((n_steps, goal_dim),  dtype=np.float32)
        self.rewards      = np.zeros(n_steps,              dtype=np.float32)
        self.dones        = np.zeros(n_steps,              dtype=np.float32)
        self.values_stab  = np.zeros(n_steps,              dtype=np.float32)
        self.values_symm  = np.zeros(n_steps,              dtype=np.float32)
        self.manager_values = np.zeros(n_steps,            dtype=np.float32)
        self.freq_targets = np.zeros(n_steps,              dtype=np.float32)
        self.returns_stab    = None
        self.returns_symm    = None
        self.manager_returns = None
        self.advantages      = None
        self.ptr = 0

    def add(self, obs, embed, action, log_prob, reward, done,
            goal_stab, goal_symm, v_stab, v_symm, v_manager, freq_target):
        i = self.ptr
        self.obs[i]            = obs
        self.embeds[i]         = embed
        self.actions[i]        = action
        self.log_probs[i]      = log_prob
        self.rewards[i]        = reward
        self.dones[i]          = done
        self.goal_stab[i]      = goal_stab
        self.goal_symm[i]      = goal_symm
        self.values_stab[i]    = v_stab
        self.values_symm[i]    = v_symm
        self.manager_values[i] = v_manager
        self.freq_targets[i]   = freq_target
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.n_steps

    def reset(self):
        self.ptr = 0

    def _gae(self, rewards, values, dones, last_v, gamma, lam):
        n = len(rewards)
        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            nxt   = 1.0 - dones[t]
            nxt_v = last_v if t == n - 1 else values[t + 1]
            delta = rewards[t] + gamma * nxt_v * nxt - values[t]
            last_gae = delta + gamma * lam * nxt * last_gae
            adv[t] = last_gae
        return adv

    def compute_returns_and_advantages(
        self, last_v_stab, last_v_symm, last_v_manager, gamma, gae_lambda
    ):
        adv_stab = self._gae(
            self.rewards, self.values_stab,    self.dones,
            last_v_stab,    gamma, gae_lambda
        )
        adv_symm = self._gae(
            self.rewards, self.values_symm,    self.dones,
            last_v_symm,    gamma, gae_lambda
        )
        adv_mgr  = self._gae(
            self.rewards, self.manager_values, self.dones,
            last_v_manager, gamma, gae_lambda
        )
        self.returns_stab    = adv_stab + self.values_stab
        self.returns_symm    = adv_symm + self.values_symm
        self.manager_returns = adv_mgr  + self.manager_values
        # Normalize stab advantages -- used for policy loss of both sub-policies
        self.advantages = (adv_stab - adv_stab.mean()) / (adv_stab.std() + 1e-8)

    def to_tensors(self, device):
        def t(x): return torch.as_tensor(x, dtype=torch.float32, device=device)
        return {
            "obs":          t(self.obs),
            "embeds":       t(self.embeds),
            "actions":      t(self.actions),
            "log_probs":    t(self.log_probs),
            "goal_stab":    t(self.goal_stab),
            "goal_symm":    t(self.goal_symm),
            "returns_stab": t(self.returns_stab),
            "returns_symm": t(self.returns_symm),
            "mgr_returns":  t(self.manager_returns),
            "advantages":   t(self.advantages),
            "freq_targets": t(self.freq_targets),
        }


# ---------------------------------------------------------------------------
# Frequency target estimation
# ---------------------------------------------------------------------------

def estimate_dominant_freq(probe_history: np.ndarray, dt: float) -> float:
    """
    Estimate dominant frequency via autocorrelation.
    probe_history: (T,) normalized probe readings (units cancel in ratio).
    Returns frequency in units of 1/control_step.
    Falls back to 0.0 if signal is flat or buffer too short.
    """
    T = len(probe_history)
    if T < 10:
        return 0.0
    x = probe_history - probe_history.mean()
    if x.std() < 1e-8:
        return 0.0
    acf   = np.correlate(x, x, mode="full")[T - 1:]
    acf   = acf / (acf[0] + 1e-8)
    diffs = np.diff(acf[:T // 2])
    peaks = np.where((diffs[:-1] > 0) & (diffs[1:] <= 0))[0] + 1
    if len(peaks) == 0:
        return 0.0
    return 1.0 / (peaks[0] + 1e-8)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HRSSATrainer:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device("cpu")
        print(f"Device: {self.device}")

        # --- Environment ---
        self.env = PinballEnv({
            "Re":           cfg.Re,
            "mesh":         cfg.mesh,
            "dt":           cfg.dt,
            "num_substeps": cfg.num_substeps,
            "n_probes":     cfg.n_probes,
            "buffer_len":   cfg.buffer_len,
            "embed_dim":    cfg.embed_dim,
            "warmup_steps": cfg.warmup_steps,
        })

        obs_dim = cfg.n_probes + cfg.embed_dim

        # --- Models ---
        self.manager = Manager(
            embed_dim=cfg.embed_dim,
            hidden=cfg.manager_hidden,
            goal_dim=cfg.goal_dim,
        ).to(self.device)

        self.sub_policies = HRSSASubPolicies(
            obs_dim=obs_dim,
            goal_dim=cfg.goal_dim,
            hidden=cfg.subpolicy_hidden,
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.manager.parameters())
            + list(self.env._buf.encoder.parameters())
            + list(self.sub_policies.parameters()),
            lr=cfg.lr,
        )

        self.buffer = RolloutBuffer(
            n_steps=cfg.n_steps,
            obs_dim=obs_dim,
            embed_dim=cfg.embed_dim,
            goal_dim=cfg.goal_dim,
        )

        self.save_dir = Path(cfg.save_dir) / cfg.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.ep_rewards: deque[float] = deque(maxlen=100)
        self.global_step = 0
        self.episode     = 0

    # ------------------------------------------------------------------
    # Collect one rollout
    # ------------------------------------------------------------------

    @staticmethod
    def _mpi_rank() -> int:
        try:
            from mpi4py import MPI
            return MPI.COMM_WORLD.Get_rank()
        except ImportError:
            return 0

    # ------------------------------------------------------------------
    # Collect one rollout
    # ------------------------------------------------------------------

    def collect_rollout(self, obs: np.ndarray, h: torch.Tensor):
        """
        All MPI ranks call this method together so that every env.step()
        and env.reset() -- which are collective PETSc solves -- has all
        ranks present.

        Only rank 0 does RL bookkeeping (buffer writes, reward tracking,
        policy inference). Non-root ranks step the env with a zero action
        (value doesn't matter; the solver uses rank 0's action broadcast
        through PETSc) and discard the return values.
        """
        rank = self._mpi_rank()
        is_root = rank == 0

        if is_root:
            self.buffer.reset()

        ep_reward       = 0.0
        ep_rewards_list = []
        probe_hist: deque[float] = deque(maxlen=self.cfg.buffer_len)
        symm_counter   = 0
        goal_symm_held = None

        _zero_action = np.zeros(3, dtype=np.float32)

        for _ in range(self.cfg.n_steps):
            if is_root:
                obs_t   = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                embed_t = obs_t[:, self.cfg.n_probes:]

                with torch.no_grad():
                    goal_stab, goal_symm_new, v_manager, h, _ = self.manager(embed_t, h)
                    if symm_counter % self.cfg.symm_timescale_ratio == 0:
                        goal_symm_held = goal_symm_new
                    symm_counter += 1
                    goal_symm = goal_symm_held
                    action, log_prob, v_stab, v_symm = self.sub_policies.get_actions(obs_t, goal_stab, goal_symm)

                action_np = action.squeeze(0).cpu().numpy()
                probe_hist.append(float(obs[0]))
                freq_target = estimate_dominant_freq(np.array(probe_hist, dtype=np.float32), self.cfg.dt)

                next_obs, reward, done, info = self.env.step(action_np)
                ep_reward += reward

                self.buffer.add(
                    obs=obs, embed=embed_t.squeeze(0).cpu().numpy(),
                    action=action_np, log_prob=log_prob.item(),
                    reward=reward, done=float(done),
                    goal_stab=goal_stab.squeeze(0).cpu().numpy(),
                    goal_symm=goal_symm.squeeze(0).cpu().numpy(),
                    v_stab=v_stab.item(), v_symm=v_symm.item(),
                    v_manager=v_manager.item(), freq_target=freq_target,
                )
                self.global_step += 1
                obs = next_obs

                if done:
                    self.episode += 1
                    ep_rewards_list.append(ep_reward)
                    self.ep_rewards.append(ep_reward)
                    ep_reward = 0.0
                    obs = self.env.reset()
                    h = self.manager.init_hidden(1).to(self.device)
                    probe_hist.clear()
                    symm_counter = 0
                    goal_symm_held = None
                    if self.episode % self.cfg.log_interval == 0:
                        print(
                            f"[rank 0][ep {self.episode:4d} | step {self.global_step:7d}] "
                            f"ep_reward={ep_rewards_list[-1]:8.4f}  "
                            f"mean100={np.mean(self.ep_rewards):8.4f}",
                            flush=True,
                        )
            else:
                # Non-root: step the env so PETSc collectives complete,
                # then discard the result. The action value is irrelevant --
                # Firedrake uses rank 0's action from the shared solve.
                _, _, done, _ = self.env.step(_zero_action)
                if done:
                    self.env.reset()

        # Bootstrap (rank 0 only -- pure Python/PyTorch, not collective)
        if is_root:
            obs_t   = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            embed_t = obs_t[:, self.cfg.n_probes:]
            with torch.no_grad():
                _, _, last_v_mgr, _, _ = self.manager(embed_t, h)
                g_stab_b, g_symm_b    = self.manager.bootstrap_goals(h)
                _, _, last_v_stab, last_v_symm = self.sub_policies.get_actions(obs_t, g_stab_b, g_symm_b)
            self.buffer.compute_returns_and_advantages(
                last_v_stab=last_v_stab.item(),
                last_v_symm=last_v_symm.item(),
                last_v_manager=last_v_mgr.item(),
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )

        return obs, h, ep_rewards_list

    # ------------------------------------------------------------------
    # PPO update  (rank 0 only -- pure PyTorch, not collective)
    # ------------------------------------------------------------------

    def update(self) -> float:
        if self._mpi_rank() != 0:
            return 0.0
        data      = self.buffer.to_tensors(self.device)
        B         = self.cfg.n_steps
        indices   = np.arange(B)
        loss_sum  = 0.0
        n_updates = 0

        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, B, self.cfg.batch_size):
                idx = indices[start: start + self.cfg.batch_size]
                if len(idx) == 0:
                    continue

                obs_b      = data["obs"][idx]
                embed_b    = data["embeds"][idx]
                actions_b  = data["actions"][idx]
                old_lp_b   = data["log_probs"][idx]
                adv_b      = data["advantages"][idx]
                ret_stab_b = data["returns_stab"][idx]
                ret_symm_b = data["returns_symm"][idx]
                ret_mgr_b  = data["mgr_returns"][idx]
                freq_tgt_b = data["freq_targets"][idx]

                # Manager forward with gradients.
                # embed_b detached from encoder (stored as numpy in rollout) --
                # encoder gradients flow only through aux_loss (intended).
                h_zeros = torch.zeros(
                    len(idx), self.cfg.manager_hidden, device=self.device
                )
                goal_stab_new, goal_symm_new, v_mgr_new, _, pred_freq = \
                    self.manager(embed_b, h_zeros)

                dist_stab, v_stab_new = self.sub_policies.stabilization(
                    obs_b, goal_stab_new
                )
                dist_symm, v_symm_new = self.sub_policies.symmetry(
                    obs_b, goal_symm_new
                )

                # Action order: [front/symm, rear1, rear2/stab]
                a_front = actions_b[:, :1]
                a_rear  = actions_b[:, 1:]
                lp_stab = dist_stab.log_prob(a_rear).sum(-1)
                lp_symm = dist_symm.log_prob(a_front).sum(-1)
                new_lp  = lp_stab + lp_symm

                ratio = (new_lp - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(
                    1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps
                ) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                vf_loss = (
                    nn.functional.mse_loss(v_stab_new.squeeze(-1), ret_stab_b)
                    + nn.functional.mse_loss(v_symm_new.squeeze(-1), ret_symm_b)
                    + nn.functional.mse_loss(v_mgr_new.squeeze(-1),  ret_mgr_b)
                )

                entropy = (
                    dist_stab.entropy().sum(-1)
                    + dist_symm.entropy().sum(-1)
                ).mean()

                aux_loss = nn.functional.mse_loss(
                    pred_freq.squeeze(-1), freq_tgt_b
                )

                total_loss = (
                    policy_loss
                    + self.cfg.vf_coef        * vf_loss
                    - self.cfg.ent_coef       * entropy
                    + self.cfg.aux_freq_weight * aux_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.manager.parameters())
                    + list(self.env._buf.encoder.parameters())
                    + list(self.sub_policies.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()
                loss_sum  += total_loss.item()
                n_updates += 1

        return loss_sum / max(n_updates, 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, tag: str = None) -> Path:
        import h5py, io
        label = tag or f"ep{self.episode:05d}"
        path  = self.save_dir / f"pinball_re{self.cfg.Re}_{label}.h5"
        if self._mpi_rank() != 0:
            return path

        def state_dict_bytes(obj) -> np.ndarray:
            buf = io.BytesIO()
            torch.save(obj.state_dict(), buf)
            return np.frombuffer(buf.getvalue(), dtype=np.uint8)

        n = self.buffer.ptr  # steps actually filled in current rollout

        with h5py.File(path, "w") as f:

            # ----------------------------------------------------------------
            # /metadata   -- human-readable run description and hyperparams
            # ----------------------------------------------------------------
            mg = f.create_group("metadata")
            mg.attrs["description"]  = (
                f"Fluidic Pinball Re={self.cfg.Re} PPO training run"
            )
            mg.attrs["Re"]           = self.cfg.Re
            mg.attrs["dt"]           = self.cfg.dt
            mg.attrs["mesh"]         = self.cfg.mesh
            mg.attrs["num_substeps"] = self.cfg.num_substeps
            mg.attrs["algorithm"]    = "PPO"
            mg.attrs["geometry"]     = "fluidic_pinball"
            mg.attrs["actuation"]    = "cylinder_rotation"
            mg.attrs["run_name"]     = self.cfg.run_name
            mg.attrs["episode"]      = self.episode
            mg.attrs["global_step"]  = self.global_step
            mg.attrs["num_steps"]    = n

            # ----------------------------------------------------------------
            # /timeseries   -- one row per env step, named like the ref file
            # ----------------------------------------------------------------
            tg = f.create_group("timeseries")

            # Simulated time axis (control steps × dt × num_substeps)
            step_dt = self.cfg.dt * self.cfg.num_substeps
            time_arr = np.arange(n, dtype=np.float32) * step_dt
            tg.create_dataset("time",   data=time_arr, compression="gzip")

            tg.create_dataset("reward", data=self.buffer.rewards[:n].astype(np.float32), compression="gzip")

            # Raw probe observations: columns 0..n_probes-1
            obs = self.buffer.obs[:n]           # (n, obs_dim)
            probes = obs[:, :self.cfg.n_probes] # (n, 6)

            # Drag proxies: probes 0-2 track cylinder drag signals
            tg.create_dataset("drag",   data=probes[:, 0], compression="gzip")  # front cylinder
            tg.create_dataset("drag_2", data=probes[:, 1], compression="gzip")  # rear upper
            tg.create_dataset("drag_3", data=probes[:, 2], compression="gzip")  # rear lower
            # CD = mean drag proxy across all three cylinders
            tg.create_dataset("CD",     data=probes[:, :3].mean(axis=1).astype(np.float32), compression="gzip")

            # Lift proxies: probes 3-5
            tg.create_dataset("CL1",    data=probes[:, 3], compression="gzip")
            tg.create_dataset("CL2",    data=probes[:, 4], compression="gzip")
            tg.create_dataset("CL3",    data=probes[:, 5], compression="gzip")

            # Actions (cylinder angular velocities)
            tg.create_dataset("action_front",    data=self.buffer.actions[:n, 0], compression="gzip")
            tg.create_dataset("action_rear_top", data=self.buffer.actions[:n, 1], compression="gzip")
            tg.create_dataset("action_rear_bot", data=self.buffer.actions[:n, 2], compression="gzip")

            # Regime embedding (full latent for analysis)
            tg.create_dataset("regime_embedding",
                              data=self.buffer.embeds[:n].astype(np.float32),
                              compression="gzip")

            # ----------------------------------------------------------------
            # /training   -- RL scalars for learning curve analysis
            # ----------------------------------------------------------------
            trg = f.create_group("training")
            trg.attrs["mean_reward_100"] = float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0
            trg.attrs["ep_reward_last"]  = float(self.ep_rewards[-1]) if self.ep_rewards else 0.0
            trg.attrs["total_loss"]      = float(getattr(self, "_last_loss", 0.0))

            # ----------------------------------------------------------------
            # /weights   -- model state dicts (byte-packed for resuming)
            # ----------------------------------------------------------------
            wg = f.create_group("weights")
            wg.attrs["note"] = (
                "Load with: buf=io.BytesIO(f['weights/manager'][:].tobytes()); "
                "model.load_state_dict(torch.load(buf))"
            )
            wg.create_dataset("manager",      data=state_dict_bytes(self.manager))
            wg.create_dataset("sub_policies", data=state_dict_bytes(self.sub_policies))
            wg.create_dataset("encoder",      data=state_dict_bytes(self.env._buf.encoder))
            wg.create_dataset("optimizer",    data=state_dict_bytes(self.optimizer))

            ng = f.create_group("norm")
            ng.attrs["note"] = "Observation normalisation stats for inference"
            ng.create_dataset("obs_mean", data=np.asarray(self.env._buf.obs_mean, dtype=np.float32))
            ng.create_dataset("obs_std",  data=np.asarray(self.env._buf.obs_std,  dtype=np.float32))

        print(f"  checkpoint -> {path}", flush=True)
        return path

    def load(self, path: str):
        import h5py, io
        p = Path(path)
        if p.suffix == ".pt":
            # Legacy .pt support
            ckpt = torch.load(path, map_location=self.device)
            self.episode                = ckpt["episode"]
            self.global_step            = ckpt["global_step"]
            self.manager.load_state_dict(ckpt["manager"])
            self.sub_policies.load_state_dict(ckpt["sub_policies"])
            self.env._buf.encoder.load_state_dict(ckpt["encoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.env._buf.obs_mean      = ckpt["obs_mean"]
            self.env._buf.obs_std       = ckpt["obs_std"]
        else:
            # HDF5 checkpoint (new readable format)
            def load_state_dict(obj, dataset):
                buf = io.BytesIO(dataset[:].tobytes())
                obj.load_state_dict(torch.load(buf, map_location=self.device))
            with h5py.File(path, "r") as f:
                meta = f["metadata"]
                self.episode     = int(meta.attrs["episode"])
                self.global_step = int(meta.attrs["global_step"])
                load_state_dict(self.manager,          f["weights/manager"])
                load_state_dict(self.sub_policies,     f["weights/sub_policies"])
                load_state_dict(self.env._buf.encoder, f["weights/encoder"])
                load_state_dict(self.optimizer,        f["weights/optimizer"])
                self.env._buf.obs_mean = f["norm/obs_mean"][:]
                self.env._buf.obs_std  = f["norm/obs_std"][:]
        self.env._normalizer_fitted = True
        print(f"  loaded checkpoint from {path} (ep {self.episode})", flush=True)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        resume_from: Optional[str] = None,
        callbacks=None,
    ):
        rank = self._mpi_rank()
        is_root = rank == 0

        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except ImportError:
            comm = None

        if resume_from and is_root:
            self.load(resume_from)

        callbacks = callbacks or []
        obs = self.env.reset()
        h   = self.manager.init_hidden(1).to(self.device)
        cb_state = {}

        t0 = time.time()
        while True:
            # Rank 0 decides whether to keep going; broadcast to all ranks
            # so non-root ranks don't loop forever on a stale global_step.
            # Must use a numpy array -- mpi4py Bcast rejects plain Python lists.
            keep_going = np.array([self.global_step < self.cfg.total_timesteps], dtype=np.bool_)
            if comm is not None:
                comm.Bcast(keep_going, root=0)
            if not keep_going[0]:
                break

            obs, h, ep_rewards_list = self.collect_rollout(obs, h)
            mean_loss = self.update()   # no-op on non-root ranks
            if is_root:
                self._last_loss = mean_loss

            if is_root:
                cb_state = {
                    "episode":       self.episode,
                    "global_step":   self.global_step,
                    "ep_reward":     ep_rewards_list[-1] if ep_rewards_list else 0.0,
                    "mean_reward":   float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0,
                    "total_loss":    mean_loss,
                    "manager":       self.manager,
                    "sub_policies":  self.sub_policies,
                    "encoder":       self.env._buf.encoder,
                    "optimizer":     self.optimizer,
                    "obs_mean":      self.env._buf.obs_mean,
                    "obs_std":       self.env._buf.obs_std,
                    "env":           self.env,
                    "save_dir":      self.save_dir,
                    "obs_buffer":    self.buffer.obs[:self.buffer.ptr],
                    "action_buffer": self.buffer.actions[:self.buffer.ptr],
                    "reward_buffer": self.buffer.rewards[:self.buffer.ptr],
                    "Re":            self.cfg.Re,
                    "geometry":      "fluidic_pinball",
                    "actuation":     "cylinder_rotation",
                    "algorithm":     "PPO",
                }
                for cb in callbacks:
                    cb.on_rollout_end(cb_state)

                if not any(hasattr(cb, "save_interval") for cb in callbacks):
                    if self.episode > 0 and self.episode % self.cfg.save_interval == 0:
                        self.save()

        if is_root:
            self.save(tag="final")
            for cb in callbacks:
                cb.on_training_end(cb_state)

        print(f"[rank {rank}] Training complete in {time.time() - t0:.1f}s", flush=True)

        # All ranks close the env so PETSc shuts down cleanly
        self.env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from training.callbacks import default_callbacks

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (optional)")
    args = parser.parse_args()

    cfg = PPOConfig.from_yaml(args.config) if args.config else PPOConfig()

    callbacks = default_callbacks(
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        wandb_project=args.wandb_project,
        wandb_config=cfg.__dict__,
    )

    trainer = HRSSATrainer(cfg)
    trainer.train(resume_from=args.resume, callbacks=callbacks)