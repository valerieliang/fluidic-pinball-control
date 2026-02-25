import hydrogym.firedrake as hgym
import firedrake as fd
from firedrake.pyplot import tricontourf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import subprocess

# ─── Knobs ────────────────────────────────────────────────────────────────────
Re            = 100          # higher Re = richer shedding 
dt            = 1e-2
TOTAL_STEPS   = 50_000       # t = 500s at dt=0.01
PNG_EVERY     = 10           # render a vorticity PNG every N steps
PRINT_EVERY   = 200

HDF5_FILE     = "pinball_timeseries.h5"
PNG_DIR       = "snapshots"

NOISE_AMP     = 0.002        # small symmetry-breaking perturbation
VIDEO_FPS     = 30           # for final ffmpeg video
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(PNG_DIR, exist_ok=True)

# ─── Environment ──────────────────────────────────────────────────────────────
env_config = {
    "flow":          hgym.Pinball,
    "flow_config":   {"Re": Re},
    "solver":        hgym.SemiImplicitBDF,
    "solver_config": {"dt": dt},
}
env = hgym.FlowEnv(env_config)

# ─── Small symmetry-breaking perturbation ─────────────────────────────────────
print(f"Applying symmetry-breaking perturbation (amplitude={NOISE_AMP})...")
u, p = env.flow.q.subfunctions
noise = fd.Function(u.function_space())
np.random.seed(42)
noise.dat.data[:] = NOISE_AMP * np.random.randn(*noise.dat.data.shape)
u.assign(u + noise)
print("Done.\n")

# ─── Vorticity function space (reuse across steps) ────────────────────────────
V_cg1     = fd.FunctionSpace(env.flow.mesh, "CG", 1)
vorticity = fd.Function(V_cg1, name="vorticity")

# ─── Pre-allocate HDF5 timeseries ─────────────────────────────────────────────
scalar_keys = ["time", "reward", "CL1", "CL2", "CL3", "CD1", "CD2", "CD3", "drag_total"]

h5 = h5py.File(HDF5_FILE, "w")
meta = h5.create_group("metadata")
meta.attrs["Re"]          = Re
meta.attrs["dt"]          = dt
meta.attrs["total_steps"] = TOTAL_STEPS
ts  = h5.create_group("timeseries")
ds  = {}
for key in scalar_keys:
    ds[key] = ts.create_dataset(key, shape=(TOTAL_STEPS,), dtype="f4",
                                chunks=(1000,), compression="gzip",
                                compression_opts=4)
write_ptr = 0

# ─── Plotting setup ───────────────────────────────────────────────────────────
VORT_LEVELS = np.linspace(-3, 3, 60)

def save_png(step, t, u):
    vorticity.assign(fd.project(fd.curl(u), V_cg1))
    fig, ax = plt.subplots(figsize=(12, 4))
    tricontourf(vorticity, levels=VORT_LEVELS, axes=ax, cmap="RdBu_r", extend="both")
    ax.set_aspect("equal")
    ax.set_xlim(-2, 18)
    ax.set_title(f"Vorticity  Re={Re}  t={t:.2f}s  step={step}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    fpath = os.path.join(PNG_DIR, f"vort_{step:06d}.png")
    plt.savefig(fpath, dpi=100)
    plt.close(fig)   # critical — releases memory immediately

# ─── Main loop ────────────────────────────────────────────────────────────────
print(f"Running Pinball Re={Re} for {TOTAL_STEPS} steps (t={TOTAL_STEPS*dt:.0f}s)")
print(f"  PNGs    -> {PNG_DIR}/ every {PNG_EVERY} steps")
print(f"  Scalars -> {HDF5_FILE}\n")

for i in range(TOTAL_STEPS):
    obs, reward, done, info = env.step([0.0, 0.0, 0.0])
    t = (i + 1) * dt

    # Write scalars directly into pre-allocated HDF5 (no list growth)
    ds["time"][write_ptr]       = t
    ds["reward"][write_ptr]     = reward
    ds["CL1"][write_ptr]        = obs[0]
    ds["CL2"][write_ptr]        = obs[1]
    ds["CL3"][write_ptr]        = obs[2]
    ds["CD1"][write_ptr]        = obs[3]
    ds["CD2"][write_ptr]        = obs[4]
    ds["CD3"][write_ptr]        = obs[5]
    ds["drag_total"][write_ptr] = obs[3] + obs[4] + obs[5]
    write_ptr += 1

    # Render PNG and close figure immediately
    if i % PNG_EVERY == 0:
        save_png(i, t, u)

    if i % PRINT_EVERY == 0:
        cl2_std = float(np.std(ds["CL2"][max(0, write_ptr-500):write_ptr])) if write_ptr > 10 else 0.0
        print(f"  t={t:7.1f}s | CD1={obs[3]:.4f} | CL2={obs[1]:.5f} | "
              f"CL2_std(500)={cl2_std:.4f} | reward={reward:.5f}")

# ─── Close HDF5 ───────────────────────────────────────────────────────────────
h5.flush()
h5.close()
print(f"\nDone. {write_ptr} steps written to {HDF5_FILE}")

# ─── Summary timeseries plot ──────────────────────────────────────────────────
with h5py.File(HDF5_FILE, "r") as f:
    time  = f["timeseries/time"][:]
    CL2   = f["timeseries/CL2"][:]
    CL3   = f["timeseries/CL3"][:]
    CD1   = f["timeseries/CD1"][:]
    CD2   = f["timeseries/CD2"][:]
    CD3   = f["timeseries/CD3"][:]
    drag  = f["timeseries/drag_total"][:]

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
fig.suptitle(f"Fluidic Pinball Re={Re} -- Uncontrolled", fontsize=13)

axes[0].plot(time, drag, color="black", lw=0.7, label="Total drag")
axes[0].set_ylabel("Total CD"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(time, CD1, lw=0.7, label="CD1 (front)")
axes[1].plot(time, CD2, lw=0.7, label="CD2")
axes[1].plot(time, CD3, lw=0.7, label="CD3")
axes[1].set_ylabel("Drag / cylinder"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(time, CL2, lw=0.7, label="CL2")
axes[2].plot(time, CL3, lw=0.7, label="CL3")
axes[2].set_ylabel("Lift"); axes[2].set_xlabel("Time (s)")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pinball_summary.png", dpi=150)
plt.close()
print("Saved pinball_summary.png")