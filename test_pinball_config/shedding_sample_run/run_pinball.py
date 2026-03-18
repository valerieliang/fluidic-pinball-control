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
Re = 100
OUTPUT_FILE = "pinball_data.h5"

# Run in chunks - save every SAVE_EVERY steps
# Total budget: 30000 steps = t=300s (should be more than enough for Re=100)
TOTAL_STEPS = 30000
SAVE_EVERY  = 1000   # save HDF5 snapshot every 1000 steps
PRINT_EVERY = 500

env_config = {
    "flow": hgym.Pinball,
    "flow_config": {"Re": Re},
    "solver": hgym.SemiImplicitBDF,
    "solver_config": {"dt": dt},
}

env = hgym.FlowEnv(env_config)

# --- Break symmetry to trigger shedding ---
# Without this, Re=100 finds the stable symmetric fixed point and never sheds
print("Applying symmetry-breaking perturbation...")
u, p = env.flow.q.subfunctions
noise = fd.Function(u.function_space())
np.random.seed(42)
noise.dat.data[:] = 0.05 * np.random.randn(*noise.dat.data.shape)
u.assign(u + noise)
print("Perturbation applied.")

history = {
    "time":     [],
    "reward":   [],
    "drag":     [],
    "CD":       [],   # drag cyl 1 (front) = obs[3]
    "drag_2":   [],   # drag cyl 2        = obs[4]
    "drag_3":   [],   # drag cyl 3        = obs[5]
    "CL1":      [],   # lift cyl 1        = obs[0]
    "CL2":      [],   # lift cyl 2        = obs[1]
    "CL3":      [],   # lift cyl 3        = obs[2]
}

def save_hdf5(history, step, shedding=False):
    tag = "SHEDDING" if shedding else f"step{step}"
    fname = f"pinball_data_{tag}.h5"
    with h5py.File(fname, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["Re"] = Re
        meta.attrs["dt"] = dt
        meta.attrs["num_steps"] = step
        meta.attrs["description"] = f"Fluidic Pinball Re={Re} uncontrolled"
        ts = f.create_group("timeseries")
        for key, val in history.items():
            ts.create_dataset(key, data=np.array(val))
    print(f"  --> Saved {fname}")
    return fname

def is_shedding(history, window=500, threshold=0.05):
    """Returns True if CL2 has been oscillating steadily for `window` steps."""
    if len(history["CL2"]) < window:
        return False
    recent = np.array(history["CL2"][-window:])
    std = recent.std()
    # Also check it's periodic, not just noisy: compare first and second half std
    half = window // 2
    std1 = recent[:half].std()
    std2 = recent[half:].std()
    # Both halves oscillating similarly = sustained shedding
    return std > threshold and abs(std1 - std2) / (std + 1e-10) < 0.3

print(f"Running Pinball Re={Re} for up to {TOTAL_STEPS} steps (t={TOTAL_STEPS*dt:.0f}s)...")
print(f"Will save every {SAVE_EVERY} steps and stop early if shedding is detected.\n")

shedding_detected = False
last_save = OUTPUT_FILE

for i in range(TOTAL_STEPS):
    action = [0.0, 0.0, 0.0]
    obs, reward, done, info = env.step(action)

    if i == 0:
        print(f"obs layout: [CL1={obs[0]:.3f}, CL2={obs[1]:.3f}, CL3={obs[2]:.3f}, CD1={obs[3]:.3f}, CD2={obs[4]:.3f}, CD3={obs[5]:.3f}]")

    history["time"].append(i * dt)
    history["reward"].append(reward)
    history["CL1"].append(obs[0])
    history["CL2"].append(obs[1])
    history["CL3"].append(obs[2])
    history["CD"].append(obs[3])
    history["drag_2"].append(obs[4])
    history["drag_3"].append(obs[5])
    history["drag"].append(obs[3] + obs[4] + obs[5])

    if i % PRINT_EVERY == 0:
        cl2_std = np.std(history["CL2"][-500:]) if len(history["CL2"]) >= 500 else 0
        print(f"  t={i*dt:7.1f}s | reward={reward:.5f} | CD={obs[3]:.4f} | CL2={obs[1]:.4f} | CL2_std(500)={cl2_std:.4f}")

    # Periodic save
    if (i + 1) % SAVE_EVERY == 0:
        last_save = save_hdf5(history, i + 1)

    # Check for shedding
    if not shedding_detected and is_shedding(history):
        t_shed = i * dt
        print(f"\n*** SHEDDING DETECTED at t={t_shed:.1f}s! ***")
        shedding_detected = True
        last_save = save_hdf5(history, i + 1, shedding=True)
        # Don't stop - keep running to build up a good shedding dataset

print(f"\nSimulation complete. Total time: t={TOTAL_STEPS * dt:.0f}s")
if shedding_detected:
    print("Shedding was detected - check pinball_data_SHEDDING.h5")
else:
    print("WARNING: No shedding detected. May need longer run or higher Re.")

# --- Final save ---
save_hdf5(history, TOTAL_STEPS)

# --- Vorticity plot at end ---
u, p = env.flow.q.subfunctions
vorticity = fd.project(fd.curl(u), fd.FunctionSpace(env.flow.mesh, "CG", 1))

fig2, ax = plt.subplots(figsize=(14, 5))
levels = np.linspace(-3, 3, 50)
contours = tricontourf(vorticity, levels=levels, axes=ax, cmap="RdBu_r", extend="both")
fig2.colorbar(contours, ax=ax, label="Vorticity")
ax.set_aspect("equal")
ax.set_xlim(-2, 20)
ax.set_title(f"Vorticity at t={TOTAL_STEPS*dt:.0f}s  Re={Re}")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()
plt.savefig("pinball_vorticity_final.png", dpi=150)
print("Saved pinball_vorticity_final.png")

# --- Final time series plot ---
times = history["time"]
fig1, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig1.suptitle(f"Fluidic Pinball Re={Re} - Uncontrolled (with perturbation)", fontsize=14)

axes[0].plot(times, history["reward"], color="black", linewidth=0.7)
axes[0].set_ylabel("Reward")
axes[0].set_title("Reward (= -total drag)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(times, history["CD"],     label="CD cyl 1 (front)", linewidth=0.7)
axes[1].plot(times, history["drag_2"], label="CD cyl 2", linewidth=0.7)
axes[1].plot(times, history["drag_3"], label="CD cyl 3", linewidth=0.7)
axes[1].set_ylabel("Drag")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(times, history["CL2"], label="CL2", linewidth=0.7)
axes[2].plot(times, history["CL3"], label="CL3", linewidth=0.7)
axes[2].set_ylabel("Lift")
axes[2].set_xlabel("Time (s)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pinball_timeseries_final.png", dpi=150)
print("Saved pinball_timeseries_final.png")