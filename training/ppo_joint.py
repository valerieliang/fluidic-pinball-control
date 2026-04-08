# training/ppo_joint.py
"""
Joint PPO training loop for HR-SSA with MPI subcommunicator support.
See prior revision comments for full architecture rationale.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Timestamped diagnostic logger  (always prints, all ranks, immediate flush)
# ---------------------------------------------------------------------------

def _log(rank: int, msg: str):
    print(f"[R{rank} | {time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------------------------------------------------------------------------
# MPI subcommunicator setup  — BEFORE any Firedrake / PETSc import
# ---------------------------------------------------------------------------

try:
    from mpi4py import MPI as _MPI

    _WORLD      = _MPI.COMM_WORLD
    _WORLD_RANK = _WORLD.Get_rank()
    _WORLD_SIZE = _WORLD.Get_size()
    _RANKS_PER_ENV = int(os.environ.get("HRSSA_RANKS_PER_ENV", "1"))

    _log(_WORLD_RANK, f"MPI init: world_rank={_WORLD_RANK}/{_WORLD_SIZE}, ranks_per_env={_RANKS_PER_ENV}")

    if _WORLD_SIZE % _RANKS_PER_ENV != 0:
        if _WORLD_RANK == 0:
            _log(0, f"ERROR: world size {_WORLD_SIZE} not divisible by ranks_per_env {_RANKS_PER_ENV}")
        _WORLD.Abort(1)

    _N_ENVS     = _WORLD_SIZE // _RANKS_PER_ENV
    _ENV_GROUP  = _WORLD_RANK // _RANKS_PER_ENV
    _LOCAL_RANK = _WORLD_RANK %  _RANKS_PER_ENV

    _log(_WORLD_RANK, f"Splitting COMM_WORLD -> ENV_COMM (group={_ENV_GROUP}, local_rank={_LOCAL_RANK}) ...")
    ENV_COMM = _WORLD.Split(color=_ENV_GROUP, key=_LOCAL_RANK)
    _log(_WORLD_RANK, "ENV_COMM split done")

    _is_env_leader = (_LOCAL_RANK == 0)
    _log(_WORLD_RANK, f"Splitting TRAIN_COMM (is_leader={_is_env_leader}) ...")
    TRAIN_COMM = _WORLD.Split(
        color=(0 if _is_env_leader else _MPI.UNDEFINED),
        key=_ENV_GROUP,
    )
    _TRAIN_RANK = TRAIN_COMM.Get_rank() if _is_env_leader else -1
    _TRAIN_SIZE = TRAIN_COMM.Get_size() if _is_env_leader else 0
    _log(_WORLD_RANK, f"TRAIN_COMM split done (train_rank={_TRAIN_RANK}/{_TRAIN_SIZE})")
    _MPI_AVAILABLE = True

except ImportError:
    _log(0, "mpi4py not found, running single-process")
    _MPI           = None
    _WORLD         = None
    _WORLD_RANK    = 0
    _WORLD_SIZE    = 1
    _RANKS_PER_ENV = 1
    _N_ENVS        = 1
    _ENV_GROUP     = 0
    _LOCAL_RANK    = 0
    ENV_COMM       = None
    TRAIN_COMM     = None
    _is_env_leader = True
    _TRAIN_RANK    = 0
    _TRAIN_SIZE    = 1
    _MPI_AVAILABLE = False


def is_global_rank0() -> bool:
    return _WORLD_RANK == 0

def is_env_leader() -> bool:
    return _is_env_leader

def _R() -> int:   # shorthand for log calls
    return _WORLD_RANK

# ---------------------------------------------------------------------------
# Import Firedrake-dependent code AFTER comm split
# ---------------------------------------------------------------------------

_log(_WORLD_RANK, "Importing torch ...")
import torch
import torch.nn as nn
import torch.optim as optim
_log(_WORLD_RANK, "torch imported")

_log(_WORLD_RANK, "Importing PinballEnv (triggers Firedrake/PETSc import) ...")
from envs.pinball_env import PinballEnv
_log(_WORLD_RANK, "PinballEnv imported")

from models.manager import Manager
from models.sub_policies import HRSSASubPolicies
_log(_WORLD_RANK, "All imports done")

# ---------------------------------------------------------------------------
# Communication helpers
# ---------------------------------------------------------------------------

def _tc_bcast_obj(x, root: int = 0):
    if TRAIN_COMM is None or _TRAIN_SIZE <= 1:
        return x
    return TRAIN_COMM.bcast(x, root=root)

def _tc_bcast_action(action_np, n_actions: int = 3, root: int = 0) -> np.ndarray:
    if TRAIN_COMM is None or _TRAIN_SIZE <= 1:
        return action_np
    buf = action_np.copy() if (_TRAIN_RANK == root) else np.empty(n_actions, dtype=np.float32)
    TRAIN_COMM.Bcast(buf, root=root)
    return buf

def _tc_gather_reward(reward: float, root: int = 0):
    if TRAIN_COMM is None or _TRAIN_SIZE <= 1:
        return [reward]
    send = np.array([reward], dtype=np.float32)
    recv = np.empty(_TRAIN_SIZE, dtype=np.float32) if (_TRAIN_RANK == root) else None
    TRAIN_COMM.Gather(send, recv, root=root)
    return recv.tolist() if (_TRAIN_RANK == root) else None

def _env_bcast_action(action_np, n_actions: int = 3) -> np.ndarray:
    if ENV_COMM is None or _RANKS_PER_ENV <= 1:
        return action_np
    buf = action_np.copy() if _is_env_leader else np.empty(n_actions, dtype=np.float32)
    ENV_COMM.Bcast(buf, root=0)
    return buf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    Re: int = 100
    mesh: str = "medium"
    refinement_level: int = 12
    dt: float = 1e-2
    num_substeps: int = 170
    n_probes: int = 6
    buffer_len: int = 35
    embed_dim: int = 16
    warmup_steps: int = 200
    reward_omega: float = 1.0
    manager_hidden: int = 64
    goal_dim: int = 8
    subpolicy_hidden: int = 64
    symm_timescale_ratio: int = 3
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
    aux_freq_weight: float = 0.1
    strouhal_ref: float = 0.088
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
    def __init__(self, n_steps, obs_dim, embed_dim, goal_dim):
        self.n_steps         = n_steps
        self.obs             = np.zeros((n_steps, obs_dim),   dtype=np.float32)
        self.embeds          = np.zeros((n_steps, embed_dim), dtype=np.float32)
        self.actions         = np.zeros((n_steps, 3),         dtype=np.float32)
        self.log_probs       = np.zeros(n_steps,              dtype=np.float32)
        self.goal_stab       = np.zeros((n_steps, goal_dim),  dtype=np.float32)
        self.goal_symm       = np.zeros((n_steps, goal_dim),  dtype=np.float32)
        self.rewards         = np.zeros(n_steps,              dtype=np.float32)
        self.dones           = np.zeros(n_steps,              dtype=np.float32)
        self.values_stab     = np.zeros(n_steps,              dtype=np.float32)
        self.values_symm     = np.zeros(n_steps,              dtype=np.float32)
        self.manager_values  = np.zeros(n_steps,              dtype=np.float32)
        self.freq_targets    = np.zeros(n_steps,              dtype=np.float32)
        self.returns_stab = self.returns_symm = self.manager_returns = self.advantages = None
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

    def full(self):   return self.ptr >= self.n_steps
    def reset(self):  self.ptr = 0

    def _gae(self, rewards, values, dones, last_v, gamma, lam):
        n, adv, last_gae = len(rewards), np.zeros(len(rewards), dtype=np.float32), 0.0
        for t in reversed(range(n)):
            nxt = 1.0 - dones[t]
            nxt_v = last_v if t == n - 1 else values[t + 1]
            delta = rewards[t] + gamma * nxt_v * nxt - values[t]
            last_gae = delta + gamma * lam * nxt * last_gae
            adv[t] = last_gae
        return adv

    def compute_returns_and_advantages(self, lv_stab, lv_symm, lv_mgr, gamma, lam):
        a_s = self._gae(self.rewards, self.values_stab,    self.dones, lv_stab, gamma, lam)
        a_y = self._gae(self.rewards, self.values_symm,    self.dones, lv_symm, gamma, lam)
        a_m = self._gae(self.rewards, self.manager_values, self.dones, lv_mgr,  gamma, lam)
        self.returns_stab    = a_s + self.values_stab
        self.returns_symm    = a_y + self.values_symm
        self.manager_returns = a_m + self.manager_values
        self.advantages      = (a_s - a_s.mean()) / (a_s.std() + 1e-8)

    def to_tensors(self, device):
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        return {
            "obs": t(self.obs), "embeds": t(self.embeds), "actions": t(self.actions),
            "log_probs": t(self.log_probs), "goal_stab": t(self.goal_stab),
            "goal_symm": t(self.goal_symm), "returns_stab": t(self.returns_stab),
            "returns_symm": t(self.returns_symm), "mgr_returns": t(self.manager_returns),
            "advantages": t(self.advantages), "freq_targets": t(self.freq_targets),
        }

# ---------------------------------------------------------------------------
# Frequency estimation
# ---------------------------------------------------------------------------

def estimate_dominant_freq(probe_history: np.ndarray, dt: float) -> float:
    T = len(probe_history)
    if T < 10: return 0.0
    x = probe_history - probe_history.mean()
    if x.std() < 1e-8: return 0.0
    acf   = np.correlate(x, x, mode="full")[T - 1:]
    acf   = acf / (acf[0] + 1e-8)
    diffs = np.diff(acf[:T // 2])
    peaks = np.where((diffs[:-1] > 0) & (diffs[1:] <= 0))[0] + 1
    return 0.0 if len(peaks) == 0 else 1.0 / (peaks[0] + 1e-8)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HRSSATrainer:
    def __init__(self, cfg: PPOConfig):
        self.cfg    = cfg
        self.device = torch.device("cpu")

        _log(_R(), f"HRSSATrainer.__init__ start")

        if is_global_rank0():
            _log(0, f"world={_WORLD_SIZE} | n_envs={_N_ENVS} | ranks_per_env={_RANKS_PER_ENV}")

        env_cfg = {
            "Re": cfg.Re, "mesh": cfg.mesh, "dt": cfg.dt,
            "num_substeps": cfg.num_substeps, "n_probes": cfg.n_probes,
            "buffer_len": cfg.buffer_len, "embed_dim": cfg.embed_dim,
            "warmup_steps": cfg.warmup_steps,
        }
        if ENV_COMM is not None:
            env_cfg["comm"] = ENV_COMM

        _log(_R(), "Constructing PinballEnv ...")
        self.env = PinballEnv(env_cfg)
        _log(_R(), "PinballEnv constructed")

        obs_dim = cfg.n_probes + cfg.embed_dim
        self.manager = Manager(embed_dim=cfg.embed_dim, hidden=cfg.manager_hidden, goal_dim=cfg.goal_dim).to(self.device)
        self.sub_policies = HRSSASubPolicies(obs_dim=obs_dim, goal_dim=cfg.goal_dim, hidden=cfg.subpolicy_hidden).to(self.device)
        self.optimizer = optim.Adam(
            list(self.manager.parameters()) + list(self.env._buf.encoder.parameters()) + list(self.sub_policies.parameters()),
            lr=cfg.lr,
        )

        _log(_R(), "Initial _sync_params (TRAIN_COMM + ENV_COMM broadcast) ...")
        self._sync_params()
        _log(_R(), "_sync_params done")

        self.buffer = RolloutBuffer(n_steps=cfg.n_steps, obs_dim=obs_dim, embed_dim=cfg.embed_dim, goal_dim=cfg.goal_dim)
        self.save_dir = Path(cfg.save_dir) / cfg.run_name
        if is_global_rank0():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.ep_rewards: deque = deque(maxlen=100)
        self.global_step = 0
        self.episode     = 0
        _log(_R(), "HRSSATrainer.__init__ complete")

    # -----------------------------------------------------------------------
    # Parameter sync
    # -----------------------------------------------------------------------

    def _sync_params(self):
        modules = [self.manager, self.sub_policies, self.env._buf.encoder]
        if _is_env_leader and _TRAIN_SIZE > 1:
            _log(_R(), "  _sync_params: TRAIN_COMM bcast ...")
            for mod in modules:
                sd = mod.state_dict() if (_TRAIN_RANK == 0) else None
                sd = TRAIN_COMM.bcast(sd, root=0)
                if _TRAIN_RANK != 0:
                    mod.load_state_dict(sd)
            _log(_R(), "  _sync_params: TRAIN_COMM bcast done")
        if _RANKS_PER_ENV > 1 and ENV_COMM is not None:
            _log(_R(), "  _sync_params: ENV_COMM intra-group bcast ...")
            for mod in modules:
                sd = mod.state_dict() if _is_env_leader else None
                sd = ENV_COMM.bcast(sd, root=0)
                if not _is_env_leader:
                    mod.load_state_dict(sd)
            _log(_R(), "  _sync_params: ENV_COMM intra-group bcast done")

    # -----------------------------------------------------------------------
    # Rollout
    # -----------------------------------------------------------------------

    def collect_rollout(self, obs: np.ndarray, h: torch.Tensor):
        _log(_R(), f"collect_rollout start (n_steps={self.cfg.n_steps})")
        self.buffer.reset()
        ep_reward, ep_rewards_list = 0.0, []
        probe_hist     = deque(maxlen=self.cfg.buffer_len)
        symm_counter   = 0
        goal_symm_held = None
        embed_t = goal_stab = goal_symm = log_prob = v_stab = v_symm = v_manager = None

        for step_i in range(self.cfg.n_steps):

            # Diagnostic: print first step and every 100 after
            if step_i == 0 or (step_i + 1) % 100 == 0:
                _log(_R(), f"  rollout step {step_i+1}/{self.cfg.n_steps}")

            # 1. Compute action on global rank 0
            if _is_env_leader and _TRAIN_RANK == 0:
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
            else:
                action_np = None

            # 2. Broadcast action: TRAIN_COMM → ENV_COMM
            if step_i == 0:
                _log(_R(), "  first step: TRAIN_COMM action bcast ...")
            if _is_env_leader:
                action_np = _tc_bcast_action(action_np, n_actions=3, root=0)
            if step_i == 0:
                _log(_R(), "  first step: ENV_COMM intra-group action bcast ...")
            action_np = _env_bcast_action(action_np, n_actions=3)
            if step_i == 0:
                _log(_R(), "  first step: action bcasts done, calling env.step() ...")

            # 3. Step env (Firedrake collective within ENV_COMM)
            next_obs, reward, done, _ = self.env.step(action_np)

            if step_i == 0:
                _log(_R(), "  first step: env.step() returned")

            # 4. Gather rewards to TRAIN_COMM root
            mean_reward = None
            if _is_env_leader:
                all_rewards = _tc_gather_reward(float(reward), root=0)
                if _TRAIN_RANK == 0:
                    mean_reward = float(np.mean(all_rewards))

            # 5. Sync done flag across all env groups
            if _is_env_leader:
                if TRAIN_COMM is not None and _TRAIN_SIZE > 1:
                    all_dones = TRAIN_COMM.allgather(bool(done))
                    done = any(all_dones)
                else:
                    done = bool(done)
            if _RANKS_PER_ENV > 1 and ENV_COMM is not None:
                done = ENV_COMM.bcast(done if _is_env_leader else None, root=0)
            done = bool(done)

            # 6. Store transition
            if _is_env_leader and _TRAIN_RANK == 0:
                ep_reward += mean_reward
                probe_hist.append(float(obs[0]))
                freq_target = estimate_dominant_freq(np.array(probe_hist, dtype=np.float32), self.cfg.dt)
                self.buffer.add(
                    obs=obs, embed=embed_t.squeeze(0).cpu().numpy(), action=action_np,
                    log_prob=log_prob.item(), reward=mean_reward, done=float(done),
                    goal_stab=goal_stab.squeeze(0).cpu().numpy(), goal_symm=goal_symm.squeeze(0).cpu().numpy(),
                    v_stab=v_stab.item(), v_symm=v_symm.item(), v_manager=v_manager.item(),
                    freq_target=freq_target,
                )

            self.global_step += 1

            # 7. Episode reset
            if done:
                _log(_R(), f"  episode done at rollout step {step_i+1}, resetting env ...")
                next_obs = self.env.reset()
                _log(_R(), "  env reset after episode done")
                if _is_env_leader and _TRAIN_RANK == 0:
                    self.episode += 1
                    ep_rewards_list.append(ep_reward)
                    self.ep_rewards.append(ep_reward)
                    ep_reward, h = 0.0, self.manager.init_hidden(1).to(self.device)
                    probe_hist.clear(); symm_counter = 0; goal_symm_held = None
                    if self.episode % self.cfg.log_interval == 0:
                        print(
                            f"[ep {self.episode:4d} | step {self.global_step:7d}] "
                            f"ep_reward={ep_rewards_list[-1]:8.4f}  "
                            f"mean100={np.mean(self.ep_rewards):8.4f}",
                            flush=True,
                        )
            obs = next_obs

        # Bootstrap
        if _is_env_leader and _TRAIN_RANK == 0:
            _log(_R(), "  bootstrapping final values ...")
            obs_t   = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            embed_t = obs_t[:, self.cfg.n_probes:]
            with torch.no_grad():
                _, _, lv_mgr, _, _ = self.manager(embed_t, h)
                g_s, g_y = self.manager.bootstrap_goals(h)
                _, _, lv_stab, lv_symm = self.sub_policies.get_actions(obs_t, g_s, g_y)
            self.buffer.compute_returns_and_advantages(lv_stab.item(), lv_symm.item(), lv_mgr.item(), self.cfg.gamma, self.cfg.gae_lambda)

        _log(_R(), "collect_rollout done")
        return obs, h, ep_rewards_list

    # -----------------------------------------------------------------------
    # PPO update
    # -----------------------------------------------------------------------

    def update(self) -> float:
        _log(_R(), "update() start")
        loss_val = 0.0
        if _is_env_leader and _TRAIN_RANK == 0:
            data = self.buffer.to_tensors(self.device)
            B, indices, loss_sum, n_updates = self.cfg.n_steps, np.arange(self.cfg.n_steps), 0.0, 0

            for epoch in range(self.cfg.n_epochs):
                _log(_R(), f"  PPO epoch {epoch+1}/{self.cfg.n_epochs}")
                np.random.shuffle(indices)
                for start in range(0, B, self.cfg.batch_size):
                    idx = indices[start: start + self.cfg.batch_size]
                    if len(idx) == 0: continue

                    obs_b      = data["obs"][idx];       embed_b    = data["embeds"][idx]
                    actions_b  = data["actions"][idx];   old_lp_b   = data["log_probs"][idx]
                    adv_b      = data["advantages"][idx]; ret_stab_b = data["returns_stab"][idx]
                    ret_symm_b = data["returns_symm"][idx]; ret_mgr_b = data["mgr_returns"][idx]
                    freq_tgt_b = data["freq_targets"][idx]

                    h_z = torch.zeros(len(idx), self.cfg.manager_hidden, device=self.device)
                    goal_stab_n, goal_symm_n, v_mgr_n, _, pred_freq = self.manager(embed_b, h_z)
                    dist_stab, v_stab_n = self.sub_policies.stabilization(obs_b, goal_stab_n)
                    dist_symm, v_symm_n = self.sub_policies.symmetry(obs_b, goal_symm_n)

                    lp_stab = dist_stab.log_prob(actions_b[:, 1:]).sum(-1)
                    lp_symm = dist_symm.log_prob(actions_b[:, :1]).sum(-1)
                    new_lp  = lp_stab + lp_symm
                    ratio   = (new_lp - old_lp_b).exp()
                    policy_loss = -torch.min(ratio * adv_b, ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv_b).mean()
                    vf_loss = (
                        nn.functional.mse_loss(v_stab_n.squeeze(-1), ret_stab_b)
                        + nn.functional.mse_loss(v_symm_n.squeeze(-1), ret_symm_b)
                        + nn.functional.mse_loss(v_mgr_n.squeeze(-1),  ret_mgr_b)
                    )
                    entropy  = (dist_stab.entropy().sum(-1) + dist_symm.entropy().sum(-1)).mean()
                    aux_loss = nn.functional.mse_loss(pred_freq.squeeze(-1), freq_tgt_b)
                    total_loss = policy_loss + self.cfg.vf_coef * vf_loss - self.cfg.ent_coef * entropy + self.cfg.aux_freq_weight * aux_loss

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.manager.parameters()) + list(self.env._buf.encoder.parameters()) + list(self.sub_policies.parameters()),
                        self.cfg.max_grad_norm,
                    )
                    self.optimizer.step()
                    loss_sum += total_loss.item(); n_updates += 1

            loss_val = loss_sum / max(n_updates, 1)
            _log(_R(), f"  PPO update done, loss={loss_val:.4f}")

        _log(_R(), "  _sync_params after update ...")
        self._sync_params()
        _log(_R(), "update() complete")
        return loss_val

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save(self, tag: str = None):
        if not is_global_rank0(): return
        label = tag or f"ep{self.episode:05d}"
        path  = self.save_dir / f"checkpoint_{label}.pt"
        torch.save({
            "episode": self.episode, "global_step": self.global_step,
            "manager": self.manager.state_dict(), "sub_policies": self.sub_policies.state_dict(),
            "encoder": self.env._buf.encoder.state_dict(), "optimizer": self.optimizer.state_dict(),
            "obs_mean": self.env._buf.obs_mean, "obs_std": self.env._buf.obs_std,
        }, path)
        _log(0, f"Saved checkpoint -> {path}")

    def load(self, path: str):
        _log(_R(), f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.episode     = ckpt.get("episode",     0)
        self.global_step = ckpt.get("global_step", 0)
        self.manager.load_state_dict(ckpt["manager"])
        self.sub_policies.load_state_dict(ckpt["sub_policies"])
        self.env._buf.encoder.load_state_dict(ckpt["encoder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.env._buf.obs_mean          = ckpt["obs_mean"]
        self.env._buf.obs_std           = ckpt["obs_std"]
        self.env._normalizer_fitted     = True
        self._sync_params()
        _log(_R(), "Checkpoint loaded")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    cfg     = PPOConfig()
    trainer = HRSSATrainer(cfg)
    obs     = trainer.env.reset()
    h       = trainer.manager.init_hidden(1).to(trainer.device)
    while trainer.global_step < cfg.total_timesteps:
        obs, h, _ = trainer.collect_rollout(obs, h)
        trainer.update()
        if is_global_rank0() and trainer.episode % cfg.save_interval == 0:
            trainer.save()


if __name__ == "__main__":
    main()