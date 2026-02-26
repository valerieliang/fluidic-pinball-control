import hydrogym.firedrake as hgym
import firedrake as fd
from firedrake.pyplot import tricontourf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import h5py
import os

matplotlib.use("Agg")

dt = 1e-2
num_steps = 8000
Re = 100

OUTPUT_FILE = "pinball_data.h5"

env_config = {
    "flow": hgym.Pinball,
    "flow_config": {"Re": Re},
    "solver": hgym.SemiImplicitBDF,
    "solver_config": {"dt": dt},
}

env = hgym.FlowEnv(env_config)

history = {
    "time":     [],
    "reward":   [],
    "drag":     [],   # total drag (used as reward signal)
    "CD":       [],   # drag coefficient cylinder 1 (front)
    "CL2":      [],   # lift coefficient cylinder 2
    "CL3":      [],   # lift coefficient cylinder 3
    "drag_2":   [],
    "drag_3":   [],
}

print(f"Running Pinball Re={Re} for {num_steps} steps (t={num_steps * dt:.1f}s)...")

for i in range(num_steps):
    
    action = [0.0, 0.0, 0.0]
    obs, reward, done, info = env.step(action)

    if i == 0:
        print(f"  obs[0]={obs[0]:.3f} obs[1]={obs[1]:.3f} obs[2]={obs[2]:.3f} obs[3]={obs[3]:.3f} obs[4]={obs[4]:.3f} obs[5]={obs[5]:.3f}")


    history["time"].append(i * dt)
    history["reward"].append(reward)
    history["CD"].append(obs[3])                       # drag cylinder 1 (front)
    history["drag_2"].append(obs[4])                   # drag cylinder 2
    history["drag_3"].append(obs[5])                   # drag cylinder 3
    history["CL2"].append(obs[1])                      # lift cylinder 2
    history["CL3"].append(obs[2])                      # lift cylinder 3
    history["drag"].append(obs[3] + obs[4] + obs[5])   # total drag

    if i % 100 == 0:
        print(f"  t={i * dt:6.2f}s | reward={reward:.5f} | CD={obs[3]:.4f} | CL2={obs[1]:.4f} | CL3={obs[2]:.4f}")

# --- Save to HDF5 ---
print(f"\nSaving data to {OUTPUT_FILE}...")
with h5py.File(OUTPUT_FILE, "w") as f:
    # Metadata
    meta = f.create_group("metadata")
    meta.attrs["Re"] = Re
    meta.attrs["dt"] = dt
    meta.attrs["num_steps"] = num_steps
    meta.attrs["description"] = "Fluidic Pinball uncontrolled flow data"

    # Time series
    ts = f.create_group("timeseries")
    ts.create_dataset("time",   data=np.array(history["time"]))
    ts.create_dataset("reward", data=np.array(history["reward"]))
    ts.create_dataset("drag",   data=np.array(history["drag"]))
    ts.create_dataset("CD",     data=np.array(history["CD"]))
    ts.create_dataset("CL2",    data=np.array(history["CL2"]))
    ts.create_dataset("CL3",    data=np.array(history["CL3"]))
    ts.create_dataset("drag_2", data=np.array(history["drag_2"]))
    ts.create_dataset("drag_3", data=np.array(history["drag_3"]))

print(f"Data saved to {OUTPUT_FILE}")

# --- Vorticity field ---
u, p = env.flow.q.subfunctions
vorticity = fd.project(fd.curl(u), fd.FunctionSpace(env.flow.mesh, "CG", 1))

# --- Time series plot ---
times = history["time"]
fig1, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig1.suptitle(f"Fluidic Pinball Re={Re} - Uncontrolled Flow", fontsize=14)

axes[0].plot(times, history["reward"], color="black", linewidth=1.0)
axes[0].set_ylabel("Reward")
axes[0].set_title("Reward over time")
axes[0].grid(True, alpha=0.3)

axes[1].plot(times, history["CD"],     label="CD (Cylinder 1 front)", linewidth=1.0)
axes[1].plot(times, history["drag_2"], label="Drag Cylinder 2", linewidth=1.0)
axes[1].plot(times, history["drag_3"], label="Drag Cylinder 3", linewidth=1.0)
axes[1].set_ylabel("Drag")
axes[1].set_title("Drag per cylinder")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(times, history["CL2"], label="CL,2 (Cylinder 2)", linewidth=1.0, linestyle="--")
axes[2].plot(times, history["CL3"], label="CL,3 (Cylinder 3)", linewidth=1.0, linestyle="--")
axes[2].set_ylabel("Lift")
axes[2].set_xlabel("Time (s)")
axes[2].set_title("Lift coefficients")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pinball_timeseries.png", dpi=150)
print("Saved pinball_timeseries.png")

# --- Vorticity plot ---
fig2, ax = plt.subplots(figsize=(14, 5))
levels = np.linspace(-3, 3, 50)
contours = tricontourf(vorticity, levels=levels, axes=ax, cmap="RdBu_r", extend="both")
fig2.colorbar(contours, ax=ax, label="Vorticity")
ax.set_aspect("equal")
ax.set_xlim(-2, 20)
ax.set_title(f"Vorticity field at t={num_steps * dt:.1f}s  (Re={Re})")
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("pinball_vorticity.png", dpi=150)
print("Saved pinball_vorticity.png")