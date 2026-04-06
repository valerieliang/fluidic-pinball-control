#!/usr/bin/env python3
"""
Pinball Flow - Steady State Solver
"""

import os

import firedrake as fd
from petsc4py import PETSc

import hydrogym.firedrake as hgym

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mesh_resolution = "medium"
Re = 80

solver_parameters = {"snes_monitor": None}

Re_init = [40, 60, Re]

flow = hgym.Pinball(
    Re=Re_init[0],
    mesh=mesh_resolution,
    velocity_order=2,
    use_HF_data_manager=False,
)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof / fd.COMM_WORLD.size)}")

solver = hgym.NewtonSolver(
    flow,
    stabilization="none",
    solver_parameters=solver_parameters,
)

for i, Re_val in enumerate(Re_init):
    flow.Re.assign(Re_val)
    hgym.print(f"Steady solve at Re={Re_init[i]}")
    qB = solver.solve()

# Both CheckpointFile and DumbCheckpoint internally call h5i.get_h5py_file,
# which fails because PETSc (/home/valval/petsc/arch-firedrake-default) and
# h5py (1.10.10) are linked against different HDF5 shared libraries.
# PETSc's binary viewer has zero h5py involvement — completely sidesteps it.
checkpoint_path = f"{output_dir}/pinball_Re{Re}_steady.dat"
viewer = PETSc.Viewer().createBinary(checkpoint_path, mode=PETSc.Viewer.Mode.WRITE)
with flow.q.dat.vec_ro as vec:
    vec.view(viewer)
viewer.destroy()
hgym.print(f"Checkpoint saved (PETSc binary) -> {checkpoint_path}")

# Compute and save force coefficients
CL, CD = flow.compute_forces()
hgym.print("Steady state forces:")
hgym.print(f"  Cylinder 1: CL={CL[0]:.6f}, CD={CD[0]:.6f}")
hgym.print(f"  Cylinder 2: CL={CL[1]:.6f}, CD={CD[1]:.6f}")
hgym.print(f"  Cylinder 3: CL={CL[2]:.6f}, CD={CD[2]:.6f}")

# Save visualization
vort = flow.vorticity()
try:
    pvd = fd.VTKFile(f"{output_dir}/pinball_Re{Re}_steady.pvd")
    pvd.write(flow.u, flow.p, vort)
except AttributeError:
    pvd = fd.File(f"{output_dir}/pinball_Re{Re}_steady.pvd")
    pvd.write(flow.u, flow.p, vort)

hgym.print(f"Steady state saved to {output_dir}/pinball_Re{Re}_steady.*")