#!/usr/bin/env python3
"""
Pinball Flow - Steady State Solver

Solve for steady-state flow around three cylinders in triangular arrangement.
Uses Newton iteration with Reynolds ramping for convergence.

Usage:
    python solve-steady.py

Physical setup:
    - Three cylinders in equilateral triangle configuration
    - Uniform inflow (U∞ = 1.0)
    - Re = 100 (default)
    - Complex wake structure with three-body interaction

Note: Pinball flow exhibits rich dynamics even at moderate Reynolds numbers.
      Steady state may be unstable, useful for stability analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import firedrake as fd
from petsc4py import PETSc

import hydrogym.firedrake as hgym

# -------------------------
# Setup output directories
# -------------------------
output_dir = "output"
frames_dir = os.path.join(output_dir, "frames")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

# -------------------------
# Parameters
# -------------------------
mesh_resolution = "medium"
Re_target = 80

# Smooth continuation 
n_frames = 80
Re_values = np.linspace(40, Re_target, n_frames)

solver_parameters = {"snes_monitor": None}

# -------------------------
# Initialize flow
# -------------------------
flow = hgym.Pinball(
    Re=Re_values[0],
    mesh=mesh_resolution,
    velocity_order=2,
    use_HF_data_manager=False,
)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof / fd.COMM_WORLD.size)}")

# -------------------------
# Solver
# -------------------------
solver = hgym.NewtonSolver(
    flow,
    stabilization="none",
    solver_parameters=solver_parameters,
)

# -------------------------
# Visualization helper
# -------------------------
def save_frame(flow, step, Re_val):
    vort = flow.vorticity()

    fig, ax = plt.subplots(figsize=(6, 4))

    c = fd.tripcolor(vort, axes=ax)
    plt.colorbar(c, ax=ax)

    ax.set_title(f"Vorticity | Re={Re_val:.1f}")
    ax.set_aspect("equal")

    filename = os.path.join(frames_dir, f"frame_{step:04d}.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)

# -------------------------
# Continuation solve + frames
# -------------------------
for i, Re_val in enumerate(Re_values):
    flow.Re.assign(Re_val)
    hgym.print(f"Solving steady state at Re={Re_val:.2f}")

    solver.solve()

    save_frame(flow, i, Re_val)

# -------------------------
# Save final solution (NumPy)
# -------------------------
npz_path = f"{output_dir}/pinball_Re{Re_target}_steady.npz"

with flow.q.dat.vec_ro as vec:
    q_array = vec.getArray().copy()

np.savez(npz_path, q=q_array)

# -------------------------
# Save fields
# -------------------------
u_array = flow.u.dat.data_ro.copy()
p_array = flow.p.dat.data_ro.copy()

np.savez(f"{output_dir}/pinball_Re{Re_target}_fields.npz", u=u_array, p=p_array)

# -------------------------
# Forces
# -------------------------
CL, CD = flow.compute_forces()

hgym.print("Final steady state forces:")
for i in range(3):
    hgym.print(f"Cylinder {i+1}: CL={CL[i]:.6f}, CD={CD[i]:.6f}")

# -------------------------
# ParaView output
# -------------------------
vort = flow.vorticity()
vtk_path = f"{output_dir}/pinball_Re{Re_target}_steady.pvd"

try:
    pvd = fd.VTKFile(vtk_path)
    pvd.write(flow.u, flow.p, vort)
except AttributeError:
    pvd = fd.File(vtk_path)
    pvd.write(flow.u, flow.p, vort)

hgym.print(f"Done. Frames saved in {frames_dir}/")