import firedrake as fd
import firedrake_adjoint as fda
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ufl import dx, inner

import hydrogym.firedrake as hgym

mpl.rc("text", usetex=False)
mpl.rc("font", family="serif")
mpl.rc("xtick", labelsize=14)
mpl.rc("ytick", labelsize=14)
mpl.rc("axes", labelsize=20)
mpl.rc("axes", titlesize=20)
mpl.rc("figure", figsize=(6, 4))

precomputed_data = ""

"""
# High-level interface

The highest-level interface is an implementation of the OpenAI gym `Env` called `FlowEnv`, where you have to do almost nothing to run the simulation.  We can optionally load from a previous checkpoint, which we'll do here to save time.
"""

env_config = {
    "flow": hgym.Cylinder,
    "flow_config": {
        "restart": f"{precomputed_data}/checkpoint.h5",
    },
    "solver": hgym.IPCS,
}
env = hgym.FlowEnv(env_config)

"""
Firedrake has a [full set of plotting tools](https://www.firedrakeproject.org/_modules/firedrake/plot.html) built on matplotlib, but as an easy way to see what's going on, the `FlowEnv` can also render the flow and plot vorticity.

For a much more powerful set of visualization tools, the fields can also be written out to Paraview.  We'll come back to that later.
"""

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
env.render(axes=ax)
plt.show()

"""
Keyword arguments are passed on to firedrake and then matplotlib, so the behavior is similar to what you'd expect for `matplotlib.contourf`
"""

fig, ax = plt.subplots(1, 1, figsize=(7.5, 3))
env.render(axes=ax, cmap=sns.color_palette("icefire", as_cmap=True))
ax.set_xlim([-2, 9])
ax.set_ylim([-2, 2])
plt.show()

"""
We can easily step the simulation forward in time by calling the `step` method.  This takes as an optional input the control "action" (more on this in a minute).  If not supplied the flow will just evolve naturally.

`step` returns a tuple of `(obs, reward, done, info)`.  For the cylinder the observations are the lift and drag coefficients.  Since the objective is drag minimization, `reward` is negative drag (maximizing negative drag = minimizing drag).  As of now `done` and `info` are not implemented.

The vortex shedding period is about 5.6, so let's run it for about half of that and compare to the previous state.
"""

# Runtime: ~2m
Tf = 2.8
num_steps = int(Tf // env.solver.dt)
for i in range(num_steps):
    (CL, CD), _, _, _ = env.step()  # obs, reward, done, info
    print(f"Step: {i+1}/{num_steps},\tLift: {CL:0.3f},\tDrag: {CD:0.3f}")

"""
The vortex shedding has advanced by about half a period compared to the initial state.
"""

fig, ax = plt.subplots(1, 1, figsize=(7.5, 3))
env.render(axes=ax, cmap=sns.color_palette("icefire", as_cmap=True))
ax.set_xlim([-2, 9])
ax.set_ylim([-2, 2])
plt.show()

"""
As usual with `Env` objects, we can `reset` to the initial state if we want.
"""
CL, CD = env.reset()
fig, ax = plt.subplots(1, 1, figsize=(7.5, 3))
env.render(axes=ax, cmap=sns.color_palette("icefire", as_cmap=True))
ax.set_xlim([-2, 9])
ax.set_ylim([-2, 2])