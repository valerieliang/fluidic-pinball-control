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

import firedrake as fd
import hydrogym.firedrake as hgym

# -------------------------
# Output directory
# -------------------------
output_dir = "output_original"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Parameters
# -------------------------
mesh_resolution = "medium"
Re = 80

solver_parameters = {"snes_monitor": None}

# Reynolds ramping
Re_init = [40, 60, Re]

# -------------------------
# Flow setup
# -------------------------
flow = hgym.Pinball(
    Re=Re_init[0],
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
# Reynolds ramping solve
# -------------------------
for i, Re_val in enumerate(Re_init):
    flow.Re.assign(Re_val)
    hgym.print(f"Steady solve at Re={Re_val}")
    solver.solve()

# -------------------------
# Save solution (NumPy instead of HDF5)
# -------------------------
npz_path = f"{output_dir}/pinball_Re{Re}_steady.npz"

with flow.q.dat.vec_ro as vec:
    q_array = vec.getArray().copy()

np.savez(npz_path, q=q_array)

hgym.print(f"Checkpoint saved (NumPy) -> {npz_path}")

# -------------------------
# Save fields (u, p)
# -------------------------
u_array = flow.u.dat.data_ro.copy()
p_array = flow.p.dat.data_ro.copy()

fields_path = f"{output_dir}/pinball_Re{Re}_fields.npz"
np.savez(fields_path, u=u_array, p=p_array)

hgym.print(f"Field data saved -> {fields_path}")

# -------------------------
# Compute forces
# -------------------------
CL, CD = flow.compute_forces()

hgym.print("Steady state forces:")
hgym.print(f"  Cylinder 1: CL={CL[0]:.6f}, CD={CD[0]:.6f}")
hgym.print(f"  Cylinder 2: CL={CL[1]:.6f}, CD={CD[1]:.6f}")
hgym.print(f"  Cylinder 3: CL={CL[2]:.6f}, CD={CD[2]:.6f}")

# -------------------------
# Save visualization (ParaView)
# -------------------------
vort = flow.vorticity()

vtk_path = f"{output_dir}/pinball_Re{Re}_steady.pvd"

try:
    pvd = fd.VTKFile(vtk_path)
    pvd.write(flow.u, flow.p, vort)
except AttributeError:
    pvd = fd.File(vtk_path)
    pvd.write(flow.u, flow.p, vort)

hgym.print(f"Visualization saved -> {vtk_path}")