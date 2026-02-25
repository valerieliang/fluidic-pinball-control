"""
run_pinball_v2.py
=================
Fluidic Pinball simulation with efficient on-disk output.

Memory strategy:
  - Vorticity fields  → written incrementally to XDMF (Paraview-compatible)
                        never accumulated in RAM
  - Scalar timeseries → written to HDF5 in chunks, pre-allocated
  - PNG snapshots     → rendered and saved immediately, figure closed

Tune the knobs at the top of the file.
"""

import hydrogym.firedrake as hgym
import firedrake as fd
from firedrake.pyplot import tricontourf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

# ─── Knobs ────────────────────────────────────────────────────────────────────
Re            = 100          
dt            = 1e-2
TOTAL_STEPS   = 50_000       # t = 500 s at dt=0.01
FIELD_EVERY   = 10           # save vorticity field every N steps
PNG_EVERY     = 500          # render a PNG every N steps
PRINT_EVERY   = 200

XDMF_FILE     = "pinball_vorticity.xdmf"   # Paraview-compatible field output
HDF5_FILE     = "pinball_timeseries.h5"    # scalar timeseries
PNG_DIR       = "snapshots"                # directory for PNG frames

NOISE_AMP     = 0.002        # small symmetry-breaking perturbation
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
print("Applying symmetry-breaking perturbation (amplitude={NOISE_AMP})...")
u, p = env.flow.q.subfunctions
noise = fd.Function(u.function_space())
np.random.seed(42)
noise.dat.data[:] = NOISE_AMP * np.random.randn(*noise.dat.data.shape)
u.assign(u + noise)
print("Done.\n")

# ─── Pre-compute vorticity function space ─────────────────────────────────────
V_cg1     = fd.FunctionSpace(env.flow.mesh, "CG", 1)
vorticity = fd.Function(V_cg1, name="vorticity")

# ─── XDMF writer (incremental, no RAM buildup) ────────────────────────────────
xdmf = fd.File(XDMF_FILE)   # Firedrake appends on each .write() call

# ─── Pre-allocate HDF5 timeseries (avoids repeated resize) ────────────────────
n_steps = TOTAL_STEPS
scalar_keys = ["time", "reward", "CL1", "CL2", "CL3", "CD1", "CD2", "CD3", "drag_total"]

h5 = h5py.File(HDF5_FILE, "w")
meta = h5.create_group("metadata")
meta.attrs["Re"]          = Re
meta.attrs["dt"]          = dt
meta.attrs["total_steps"] = TOTAL_STEPS
meta.attrs["field_every"] = FIELD_EVERY
ts  = h5.create_group("timeseries")
ds  = {}
for key in scalar_keys:
    # chunk by 1000 rows; gzip level 4 gives ~3× compression with low CPU cost
    ds[key] = ts.create_dataset(key, shape=(n_steps,), dtype="f4",
                                chunks=(1000,), compression="gzip",
                                compression_opts=4)
write_ptr = 0   # index into HDF5 datasets

# ─── Plotting helper ──────────────────────────────────────────────────────────
VORT_LEVELS = np.linspace(-3, 3, 60)

def save_png(step, t):
    vorticity.assign(fd.project(fd.curl(u), V_cg1))
    fig, ax = plt.subplots(figsize=(12, 4))
    tricontourf(vorticity, levels=VORT_LEVELS, axes=ax, cmap="RdBu_r", extend="both")
    ax.set_aspect("equal"); ax.set_xlim(-2, 18)
    ax.set_title(f"Vorticity  Re={Re}  t={t:.1f}s  step={step}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    fpath = os.path.join(PNG_DIR, f"vort_{step:06d}.png")
    plt.savefig(fpath, dpi=120)
    plt.close(fig)   # ← critical: releases memory immediately
    return fpath

# ─── Main loop ────────────────────────────────────────────────────────────────
print(f"Running Pinball Re={Re} for {TOTAL_STEPS} steps "
      f"(t={TOTAL_STEPS*dt:.0f}s)\n"
      f"  Fields → {XDMF_FILE}  (every {FIELD_EVERY} steps)\n"
      f"  PNGs   → {PNG_DIR}/   (every {PNG_EVERY} steps)\n"
      f"  Scalars→ {HDF5_FILE}\n")

for i in range(TOTAL_STEPS):
    obs, reward, done, info = env.step([0.0, 0.0, 0.0])
    t = (i + 1) * dt

    # ── Write scalars ──
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

    # ── Write vorticity field to XDMF ──
    if i % FIELD_EVERY == 0:
        vorticity.assign(fd.project(fd.curl(u), V_cg1))
        xdmf.write(vorticity, time=t)

    # ── Render PNG snapshot ──
    if i % PNG_EVERY == 0:
        fpath = save_png(i, t)
        print(f"  [PNG] saved {fpath}")

    # ── Console progress ──
    if i % PRINT_EVERY == 0:
        cl2_std = np.std(ds["CL2"][max(0, write_ptr-500):write_ptr]) if write_ptr > 10 else 0
        print(f"  t={t:7.1f}s | CD1={obs[3]:.4f} | CL2={obs[1]:.5f} | "
              f"CL2_std(500)={cl2_std:.4f} | reward={reward:.5f}")

# ─── Flush and close HDF5 ─────────────────────────────────────────────────────
h5.flush()
h5.close()
print(f"\nDone. Wrote {write_ptr} timesteps.")
print(f"  XDMF:    {XDMF_FILE}  (open in Paraview)")
print(f"  HDF5:    {HDF5_FILE}")
print(f"  PNGs:    {PNG_DIR}/ ({TOTAL_STEPS // PNG_EVERY + 1} frames)")

# ─── Final summary timeseries plot ────────────────────────────────────────────
with h5py.File(HDF5_FILE, "r") as f:
    ts_r = f["timeseries"]
    time  = ts_r["time"][:]
    CL2   = ts_r["CL2"][:]
    CL3   = ts_r["CL3"][:]
    CD1   = ts_r["CD1"][:]
    CD2   = ts_r["CD2"][:]
    CD3   = ts_r["CD3"][:]
    drag  = ts_r["drag_total"][:]

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
fig.suptitle(f"Fluidic Pinball Re={Re} — Uncontrolled", fontsize=13)

axes[0].plot(time, drag,  color="black", lw=0.7, label="Total drag")
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