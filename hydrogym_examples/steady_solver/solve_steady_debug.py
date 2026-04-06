#!/usr/bin/env python3
"""
Pinball Flow - Steady State Solver with working HDF5
"""

import os
import firedrake as fd
import hydrogym.firedrake as hgym
import h5py
import numpy as np

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mesh_resolution = "medium"
Re = 80

solver_parameters = {"snes_monitor": None}
Re_init = [40, 60, Re]

# Create flow configuration
flow = hgym.Pinball(
    Re=Re_init[0],
    mesh=mesh_resolution,
    velocity_order=2,
    use_HF_data_manager=False,
)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof / fd.COMM_WORLD.size)}")

# Set up Newton solver
solver = hgym.NewtonSolver(
    flow,
    stabilization="none",
    solver_parameters=solver_parameters,
)

# Reynolds ramping
for i, Re_val in enumerate(Re_init):
    flow.Re.assign(Re_val)
    hgym.print(f"Steady solve at Re={Re_init[i]}")
    qB = solver.solve()

# Compute and save forces
CL, CD = flow.compute_forces()
hgym.print("Steady state forces:")
hgym.print(f"  Cylinder 1: CL={CL[0]:.6f}, CD={CD[0]:.6f}")
hgym.print(f"  Cylinder 2: CL={CL[1]:.6f}, CD={CD[1]:.6f}")
hgym.print(f"  Cylinder 3: CL={CL[2]:.6f}, CD={CD[2]:.6f}")

# Save visualization
vort = flow.vorticity()
pvd = fd.VTKFile(f"{output_dir}/pinball_Re{Re}_steady.pvd")
pvd.write(flow.u, flow.p, vort)

# --- WORKING HDF5 SAVE (like the timeseries code) ---
h5_file = f"{output_dir}/pinball_Re{Re}_steady.h5"
with h5py.File(h5_file, 'w') as f:
    # Save metadata
    f.attrs['Re'] = Re
    f.attrs['mesh_resolution'] = mesh_resolution
    f.attrs['velocity_order'] = 2
    
    # Save force coefficients
    f.create_dataset('CL', data=CL)
    f.create_dataset('CD', data=CD)
    
    # Save solution data as arrays
    # Extract the actual solution vectors
    u_array = flow.u.dat.data[:]
    p_array = flow.p.dat.data[:]
    
    f.create_dataset('velocity', data=u_array, compression='gzip')
    f.create_dataset('pressure', data=p_array, compression='gzip')
    
    # Save mesh coordinates if needed for restart
    coords = flow.mesh.coordinates.dat.data[:]
    f.create_dataset('mesh_coordinates', data=coords, compression='gzip')
    
    # Save vorticity field
    vort_array = vort.dat.data[:]
    f.create_dataset('vorticity', data=vort_array, compression='gzip')

hgym.print(f"Checkpoint saved to {h5_file} using h5py directly")

# Verify file was created
if os.path.exists(h5_file):
    size = os.path.getsize(h5_file)
    hgym.print(f"File created successfully ({size/1024:.1f} KB)")

hgym.print("Done!")