import hydrogym.firedrake as hgym
import firedrake as fd
from firedrake.pyplot import tricontourf
import matplotlib.pyplot as plt
import numpy as np

dt = 1e-2
num_steps = 200

env_config = {
    "flow": hgym.Pinball,
    "flow_config": {},
    "solver": hgym.SemiImplicitBDF,
    "solver_config": {"dt": dt},
}

env = hgym.FlowEnv(env_config)

history = {
    "time": [],
    "reward": [],
    "lift_total": [],
    "lift_2": [],
    "lift_3": [],
    "drag_1": [],
    "drag_2": [],
    "drag_3": [],
}

print(f"Running for {num_steps} steps (t={num_steps * dt:.1f}s)...")
for i in range(num_steps):
    action = [0.0, 0.0, 0.0]
    obs, reward, done, info = env.step(action)

    history["time"].append(i * dt)
    history["reward"].append(reward)
    history["lift_total"].append(obs[0])
    history["lift_2"].append(obs[1])
    history["lift_3"].append(obs[2])
    history["drag_1"].append(obs[3])
    history["drag_2"].append(obs[4])
    history["drag_3"].append(obs[5])

    if i % 50 == 0:
        print(f"  t={i * dt:5.1f}s | reward={reward:.4f} | drag_1={obs[3]:.4f}")

# --- Vorticity ---
u, p = env.flow.q.subfunctions
vorticity = fd.project(fd.curl(u), fd.FunctionSpace(env.flow.mesh, "CG", 1))

# --- Time series ---
times = history["time"]
fig1, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig1.suptitle("Fluidic Pinball - Uncontrolled Flow", fontsize=14)

axes[0].plot(times, history["reward"], color="black", linewidth=1.0)
axes[0].set_ylabel("Reward")
axes[0].set_title("Reward over time")
axes[0].grid(True, alpha=0.3)

axes[1].plot(times, history["drag_1"], label="Cylinder 1 (front)", linewidth=1.0)
axes[1].plot(times, history["drag_2"], label="Cylinder 2", linewidth=1.0)
axes[1].plot(times, history["drag_3"], label="Cylinder 3", linewidth=1.0)
axes[1].set_ylabel("Drag")
axes[1].set_title("Drag per cylinder")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(times, history["lift_total"], label="Net lift", linewidth=1.0)
axes[2].plot(times, history["lift_2"], label="Cylinder 2", linewidth=1.0, linestyle="--")
axes[2].plot(times, history["lift_3"], label="Cylinder 3", linewidth=1.0, linestyle="--")
axes[2].set_ylabel("Lift")
axes[2].set_xlabel("Time (s)")
axes[2].set_title("Lift per cylinder")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pinball_timeseries.png", dpi=150)

# --- Vorticity field ---
fig2, ax = plt.subplots(figsize=(14, 5))
levels = np.linspace(-3, 3, 50)
contours = tricontourf(vorticity, levels=levels, axes=ax, cmap="RdBu_r", extend="both")
fig2.colorbar(contours, ax=ax, label="Vorticity")
ax.set_aspect("equal")
ax.set_title(f"Vorticity field at t={num_steps * dt:.1f}s")
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("pinball_vorticity.png", dpi=150)

plt.show()
print("Plots saved!")