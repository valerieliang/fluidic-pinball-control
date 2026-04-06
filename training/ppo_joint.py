# training/ppo_joint.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from typing import Optional
import time

from envs.pinball_env import PinballEnv
from models.manager import Manager
from models.sub_policies import HRSSASubPolicies


@dataclass
class PPOConfig:
    # Environment
    Re:              float = 100.0
    mesh:            str   = "medium"
    dt:              float = 1e-2
    num_substeps:    int   = 2
    warmup_steps:    int   = 200
    buffer_len:      int   = 50
    embed_dim:       int   = 16

    # Architecture
    goal_dim:        int   = 8
    manager_hidden:  int   = 64
    policy_hidden:   int   = 64

    # Rollout
    n_steps:         int   = 200   # steps per rollout (1 episode)
    n_epochs:        int   = 4     # PPO update epochs per rollout
    batch_size:      int   = 32

    # PPO hyperparams
    lr:              float = 3e-4
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    clip_eps:        float = 0.2
    vf_coef:         float = 0.5
    ent_coef:        float = 0.01
    aux_freq_coef:   float = 0.1   # weight on manager aux frequency loss
    max_grad_norm:   float = 0.5

    # Training
    total_rollouts:  int   = 500
    log_interval:    int   = 10
    save_interval:   int   = 50
    save_path:       str   = "checkpoints"

    device:          str   = "cpu"


class RolloutBuffer:
    """Stores one rollout's worth of transitions."""

    def __init__(self, n_steps, obs_dim, embed_dim, goal_dim, device):
        self.n      = n_steps
        self.device = device
        self.obs         = torch.zeros(n_steps, obs_dim)
        self.embeds      = torch.zeros(n_steps, embed_dim)
        self.actions     = torch.zeros(n_steps, 3)
        self.log_probs   = torch.zeros(n_steps)
        self.rewards     = torch.zeros(n_steps)
        self.dones       = torch.zeros(n_steps)
        self.v_manager   = torch.zeros(n_steps)
        self.v_stab      = torch.zeros(n_steps)
        self.v_symm      = torch.zeros(n_steps)
        self.h_states    = torch.zeros(n_steps, 64)   # manager hidden
        self.ptr         = 0

    def push(self, obs, embed, action, log_prob, reward, done,
             v_mgr, v_stab, v_symm, h):
        i = self.ptr
        self.obs[i]       = obs
        self.embeds[i]    = embed
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.dones[i]     = done
        self.v_manager[i] = v_mgr.squeeze()
        self.v_stab[i]    = v_stab.squeeze()
        self.v_symm[i]    = v_symm.squeeze()
        self.h_states[i]  = h.squeeze()
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_v_mgr, last_v_stab, last_v_symm,
                    gamma, gae_lambda):
        """Returns advantages and returns for manager, stab, symm."""
        def _gae(values, last_v):
            adv     = torch.zeros(self.n)
            last_gae = 0.0
            for t in reversed(range(self.n)):
                next_v    = last_v if t == self.n - 1 else values[t + 1]
                next_non_term = 1.0 - self.dones[t]
                delta     = (self.rewards[t]
                             + gamma * next_v * next_non_term
                             - values[t])
                last_gae  = delta + gamma * gae_lambda * next_non_term * last_gae
                adv[t]    = last_gae
            returns = adv + values
            return adv, returns

        adv_mgr,  ret_mgr  = _gae(self.v_manager, last_v_mgr)
        adv_stab, ret_stab = _gae(self.v_stab,    last_v_stab)
        adv_symm, ret_symm = _gae(self.v_symm,    last_v_symm)
        return (adv_mgr,  ret_mgr,
                adv_stab, ret_stab,
                adv_symm, ret_symm)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-8)


def collect_rollout(env, manager, sub_policies, buf, cfg, h, obs):
    """
    Run n_steps in the environment, storing transitions in buf.
    Returns the final obs, embed, and manager hidden state.
    """
    manager.eval()
    sub_policies.eval()
    buf.reset()

    obs_t = torch.from_numpy(obs).float().unsqueeze(0)   # (1, obs_dim)

    with torch.no_grad():
        for _ in range(cfg.n_steps):
            # Spectral embedding lives inside the env's buffer
            embed = env._buf.embed().unsqueeze(0)         # (1, embed_dim)

            # Manager step
            goal_stab, goal_symm, v_mgr, h, pred_freq = manager(embed, h)

            # Sub-policy actions
            action, log_prob, v_stab, v_symm = sub_policies.get_actions(
                obs_t, goal_stab, goal_symm
            )

            action_np = action.squeeze(0).numpy()
            # Clip to action space
            action_np = np.clip(
                action_np,
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)

            next_obs, reward, done, _ = env.step(action_np)
            next_obs_t = torch.from_numpy(
                np.array(next_obs, dtype=np.float32)
            ).unsqueeze(0)

            buf.push(
                obs_t.squeeze(0), embed.squeeze(0),
                action.squeeze(0), log_prob.squeeze(),
                reward, float(done),
                v_mgr, v_stab, v_symm,
                h.squeeze(0),
            )

            obs_t = next_obs_t
            if done:
                obs_np = np.array(env.reset(), dtype=np.float32)
                obs_t  = torch.from_numpy(obs_np).unsqueeze(0)
                h      = manager.init_hidden(1)

        # Bootstrap values at end of rollout
        embed_last     = env._buf.embed().unsqueeze(0)
        _, _, last_v_mgr, _, _ = manager(embed_last, h)
        _, _, last_v_stab, last_v_symm = sub_policies.get_actions(
            obs_t,
            *manager(embed_last, h)[:2]
        )

    return obs_t.squeeze(0).numpy(), h


def update(manager, sub_policies, optimizer, buf, cfg):
    """PPO update over the collected rollout."""
    manager.train()
    sub_policies.train()

    (adv_mgr,  ret_mgr,
     adv_stab, ret_stab,
     adv_symm, ret_symm) = buf.compute_gae(
        buf.v_manager[-1], buf.v_stab[-1], buf.v_symm[-1],
        cfg.gamma, cfg.gae_lambda,
    )

    adv_mgr  = _normalize(adv_mgr)
    adv_stab = _normalize(adv_stab)
    adv_symm = _normalize(adv_symm)

    dataset = TensorDataset(
        buf.obs, buf.embeds, buf.actions, buf.log_probs,
        adv_mgr, ret_mgr, adv_stab, ret_stab, adv_symm, ret_symm,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    metrics = {"loss": [], "pg_loss": [], "vf_loss": [], "ent": [],
               "aux_loss": []}

    for _ in range(cfg.n_epochs):
        for batch in loader:
            (obs_b, emb_b, act_b, old_lp_b,
             adv_mgr_b, ret_mgr_b,
             adv_stab_b, ret_stab_b,
             adv_symm_b, ret_symm_b) = batch

            h_b = manager.init_hidden(obs_b.shape[0])

            # --- Manager forward ---
            goal_stab, goal_symm, v_mgr_b, _, pred_freq_b = manager(emb_b, h_b)

            # --- Sub-policy forward (recompute log probs) ---
            dist_stab, v_stab_b = sub_policies.stabilization(obs_b, goal_stab)
            dist_symm, v_symm_b = sub_policies.symmetry(obs_b, goal_symm)

            new_lp_stab = dist_stab.log_prob(act_b[:, 1:]).sum(-1)
            new_lp_symm = dist_symm.log_prob(act_b[:, :1]).sum(-1)
            new_lp      = new_lp_stab + new_lp_symm

            entropy = (dist_stab.entropy().sum(-1) +
                       dist_symm.entropy().sum(-1)).mean()

            # --- PPO clip loss (shared advantage = mean of three) ---
            adv_b   = (adv_mgr_b + adv_stab_b + adv_symm_b) / 3.0
            ratio   = (new_lp - old_lp_b).exp()
            pg1     = ratio * adv_b
            pg2     = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
            pg_loss = -torch.min(pg1, pg2).mean()

            # --- Value losses (one per critic) ---
            vf_loss = (
                nn.functional.mse_loss(v_mgr_b.squeeze(),  ret_mgr_b)
              + nn.functional.mse_loss(v_stab_b.squeeze(), ret_stab_b)
              + nn.functional.mse_loss(v_symm_b.squeeze(), ret_symm_b)
            )

            # --- Aux frequency loss ---
            # Target: dominant frequency of probe 0 over the buffer window.
            # Approximated here as the mean absolute probe reading (cheap
            # proxy during early training; replace with FFT peak later).
            freq_target = emb_b.abs().mean(dim=-1, keepdim=True).detach()
            aux_loss    = nn.functional.mse_loss(pred_freq_b, freq_target)

            loss = (pg_loss
                    + cfg.vf_coef   * vf_loss
                    - cfg.ent_coef  * entropy
                    + cfg.aux_freq_coef * aux_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(manager.parameters()) + list(sub_policies.parameters()),
                cfg.max_grad_norm,
            )
            optimizer.step()

            metrics["loss"].append(loss.item())
            metrics["pg_loss"].append(pg_loss.item())
            metrics["vf_loss"].append(vf_loss.item())
            metrics["ent"].append(entropy.item())
            metrics["aux_loss"].append(aux_loss.item())

    return {k: np.mean(v) for k, v in metrics.items()}


def train(cfg: PPOConfig):
    import os
    os.makedirs(cfg.save_path, exist_ok=True)

    device = torch.device(cfg.device)

    env = PinballEnv({
        "Re":          cfg.Re,
        "mesh":        cfg.mesh,
        "dt":          cfg.dt,
        "num_substeps": cfg.num_substeps,
        "warmup_steps": cfg.warmup_steps,
        "buffer_len":   cfg.buffer_len,
        "embed_dim":    cfg.embed_dim,
    })

    obs_dim = env.observation_space.shape[0]   # 6 + embed_dim

    manager      = Manager(embed_dim=cfg.embed_dim,
                           hidden=cfg.manager_hidden,
                           goal_dim=cfg.goal_dim).to(device)
    sub_policies = HRSSASubPolicies(obs_dim=obs_dim,
                                    goal_dim=cfg.goal_dim,
                                    hidden=cfg.policy_hidden).to(device)

    optimizer = optim.Adam(
        list(manager.parameters()) + list(sub_policies.parameters()),
        lr=cfg.lr,
    )

    buf = RolloutBuffer(
        n_steps=cfg.n_steps,
        obs_dim=obs_dim,
        embed_dim=cfg.embed_dim,
        goal_dim=cfg.goal_dim,
        device=device,
    )

    obs = np.array(env.reset(), dtype=np.float32)
    h   = manager.init_hidden(1)

    print(f"Starting training — {cfg.total_rollouts} rollouts × "
          f"{cfg.n_steps} steps = "
          f"{cfg.total_rollouts * cfg.n_steps} total env steps")

    episode_rewards = []
    t0 = time.time()

    for rollout in range(cfg.total_rollouts):
        obs, h = collect_rollout(env, manager, sub_policies, buf, cfg, h, obs)

        ep_reward = buf.rewards.sum().item()
        episode_rewards.append(ep_reward)

        metrics = update(manager, sub_policies, optimizer, buf, cfg)

        if (rollout + 1) % cfg.log_interval == 0:
            elapsed = time.time() - t0
            mean_r  = np.mean(episode_rewards[-cfg.log_interval:])
            print(
                f"rollout {rollout+1:4d}/{cfg.total_rollouts}"
                f"  mean_ep_reward={mean_r:8.3f}"
                f"  loss={metrics['loss']:.4f}"
                f"  pg={metrics['pg_loss']:.4f}"
                f"  vf={metrics['vf_loss']:.4f}"
                f"  ent={metrics['ent']:.4f}"
                f"  aux={metrics['aux_loss']:.4f}"
                f"  elapsed={elapsed:.0f}s"
            )

        if (rollout + 1) % cfg.save_interval == 0:
            torch.save({
                "rollout":       rollout + 1,
                "manager":       manager.state_dict(),
                "sub_policies":  sub_policies.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "cfg":           cfg,
            }, f"{cfg.save_path}/ckpt_{rollout+1:04d}.pt")
            print(f"  -> saved checkpoint at rollout {rollout+1}")

    env.close()
    return manager, sub_policies


if __name__ == "__main__":
    cfg = PPOConfig(
        Re=100,
        total_rollouts=500,
        n_steps=200,
        log_interval=10,
        save_interval=50,
    )
    train(cfg)