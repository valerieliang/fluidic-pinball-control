# Fluidic Pinball Dataset - Data Documentation

## Dataset Overview

This dataset contains simulation data from the **Fluidic Pinball** flow control environment, a benchmark problem in fluid dynamics and reinforcement learning. The environment simulates turbulent flow around three cylinders arranged in an equilateral triangle formation, where the cylinders can be rotated to control the wake dynamics and minimize drag.

**Dataset Purpose:** Train reinforcement learning agents to learn optimal flow control policies for drag reduction and wake stabilization.

**Environment:** HydroGym Fluidic Pinball (Firedrake-based CFD simulation)

**Physics:** 2D incompressible Navier-Stokes equations at Reynolds number Re = 30

---

## File Structure

### HDF5 Storage Format
```
pinball_dataset_h5/
└── fluidic_pinball_data.h5
    ├── episode_0/
    │   ├── obs_flat            [1D array: concatenated observations]
    │   ├── obs_lengths         [200 timesteps: length of each observation]
    │   ├── next_obs_flat       [1D array: concatenated next observations]
    │   ├── next_obs_lengths    [200 timesteps: length of each next observation]
    │   ├── actions_flat        [1D array: concatenated actions]
    │   ├── actions_lengths     [200 timesteps: length of each action]
    │   ├── rewards             [200 timesteps × 1 feature]
    │   ├── Cd                  [200 timesteps × 1 feature]
    │   ├── Cl                  [200 timesteps × 1 feature]
    │   ├── Re                  [200 timesteps × 1 feature]
    │   ├── done                [200 timesteps × 1 feature]
    │   ├── vorticity           [200 timesteps × 1 feature]
    │   ├── pressure            [200 timesteps × 1 feature]
    │   ├── kinetic_energy      [200 timesteps × 1 feature]
    │   ├── timestep            [200 timesteps × 1 feature]
    │   ├── cumulative_reward   [200 timesteps × 1 feature]
    │   └── attrs/
    │       ├── num_steps       [200]
    │       ├── obs_dim_range   ["6-6"]
    │       └── act_dim_range   ["3-3"]
    ├── episode_1/
    │   └── ... (same structure)
    └── episode_N/
        └── ... (same structure)
```

### Variable-Length Array Storage

The dataset uses a **flattened storage format** to handle variable-length arrays efficiently:

- **Flat arrays** contain all data concatenated sequentially
- **Length arrays** specify how to split the flat array back into individual timesteps

**Reconstruction Example:**
```python
import numpy as np
import h5py

# Load episode data
with h5py.File('fluidic_pinball_data.h5', 'r') as f:
    ep = f['episode_0']
    obs_flat = ep['obs_flat'][:]
    obs_lengths = ep['obs_lengths'][:]

# Reconstruct individual observations
offsets = np.concatenate([[0], np.cumsum(obs_lengths[:-1])])
observations = [obs_flat[offset:offset+length] 
                for offset, length in zip(offsets, obs_lengths)]

# observations[0] is the first observation (typically length 6)
# observations[1] is the second observation (typically length 6)
# etc.
```

**Typical Dimensions** (for this dataset):
- Observations: Length 6 per timestep (constant in this case)
- Actions: Length 3 per timestep (constant in this case)

**Total Size:**
- Episodes: 2 (configurable, can be set to 100 or more)
- Steps per episode: 200
- Total transitions: 400 (or 20,000 for 100 episodes)

---

## Data Fields

### 1. State Information

#### **obs_flat** + **obs_lengths** (Observations)
- **Type:** Float arrays
- **Flat Shape:** (total_obs_elements,) - all observations concatenated
- **Lengths Shape:** (timesteps,) - number of elements in each observation
- **Typical Observation Size:** 6 features per timestep
- **Range:** Environment-dependent
- **Description:** Current observation of the flow state from the environment's sensor/observation function. Includes measurements such as:
  - Drag coefficient (Cd)
  - Lift coefficient (Cl)
  - Possibly: pressure sensors, velocity probes, or other flow measurements
- **Source:** `flow.get_observations()`
- **Use Case:** Input to RL policy network
- **Note:** Variable-length storage allows flexibility if observations change size during simulation

#### **next_obs_flat** + **next_obs_lengths** (Next Observations)
- **Type:** Float arrays
- **Flat Shape:** (total_obs_elements,)
- **Lengths Shape:** (timesteps,)
- **Typical Observation Size:** 6 features per timestep
- **Range:** Environment-dependent
- **Description:** Observation at the next timestep (t+1). Essential for temporal difference (TD) learning algorithms like Q-learning, SARSA, and actor-critic methods.
- **Source:** `flow.get_observations()` after `solver.step()`
- **Use Case:** Target for value function estimation in TD methods

---

### 2. Action Information

#### **actions_flat** + **actions_lengths** (Control Actions)
- **Type:** Float arrays
- **Flat Shape:** (total_action_elements,)
- **Lengths Shape:** (timesteps,)
- **Typical Action Size:** 3 features per timestep
- **Range:** [-flow.MAX_CONTROL, flow.MAX_CONTROL] (typically around ±1.5708, or ±π/2 radians)
- **Description:** Control actions applied to the three cylinders. Each value represents the rotation angle or angular velocity of one cylinder. The three cylinders form an equilateral triangle in the flow, and their rotation affects the wake dynamics.
  - `actions[0]`: Control for cylinder 1 (front cylinder)
  - `actions[1]`: Control for cylinder 2 (rear-left cylinder)
  - `actions[2]`: Control for cylinder 3 (rear-right cylinder)
- **Source:** Mixed policy (see Data Collection Strategy)
- **Physical Meaning:** Cylinder rotation angles that modify the boundary conditions and alter flow separation
- **Use Case:** Action taken by the agent; input to dynamics model

---

### 3. Reward Signal

#### **rewards** (Instantaneous Reward)
- **Type:** Float array
- **Shape:** (timesteps,)
- **Range:** Problem-dependent (typically negative for drag-based objectives)
- **Description:** Instantaneous reward signal computed by the environment's objective function. Typically designed to encourage drag reduction and/or flow stabilization.
- **Source:** `flow.evaluate_objective()`
- **Typical Formulation:** Often `-Cd` (negative drag) or a combination of drag and control cost
- **Use Case:** Training signal for RL algorithms; defines the optimization objective

#### **cumulative_reward** (Return)
- **Type:** Float array
- **Shape:** (timesteps,)
- **Range:** Monotonically increasing/decreasing within episode
- **Description:** Sum of all rewards from the start of the episode up to the current timestep. Represents the total return accumulated so far.
- **Formula:** `cumulative_reward[t] = sum(rewards[0:t+1])`
- **Use Case:** Episode performance tracking; can be used for reward normalization or trajectory filtering

---

### 4. Aerodynamic Forces

#### **Cd** (Drag Coefficient)
- **Type:** Float array
- **Shape:** (timesteps,)
- **Range:** Typically 0.1 - 2.0 for this geometry at Re=30
- **Description:** Dimensionless drag coefficient representing the resistance force in the flow direction. Lower values indicate better aerodynamic performance.
- **Formula:** `Cd = Fd / (0.5 × ρ × U² × A)` where Fd is drag force, ρ is density, U is freestream velocity, A is reference area
- **Source:** `flow.compute_forces()['Cd']` or extracted from observations
- **Physical Meaning:** Primary performance metric for drag reduction
- **Use Case:** Additional supervision signal; can be used as auxiliary reward or for analysis

#### **Cl** (Lift Coefficient)
- **Type:** Float array
- **Shape:** (timesteps,)
- **Range:** Typically -1.0 to 1.0 for this symmetric geometry
- **Description:** Dimensionless lift coefficient representing the force perpendicular to the flow direction. For symmetric configurations, this oscillates around zero due to vortex shedding.
- **Formula:** `Cl = Fl / (0.5 × ρ × U² × A)` where Fl is lift force
- **Source:** `flow.compute_forces()['Cl']` or extracted from observations
- **Physical Meaning:** Indicates flow asymmetry and oscillations; large variations suggest strong vortex shedding
- **Use Case:** Flow stability analysis; detecting bifurcations or transitions

---

### 5. Flow Physics Features

#### **vorticity** (Vorticity Magnitude)
- **Type:** Float array
- **Shape:** (timesteps,)
- **Range:** 0 to ~100+ (problem-dependent)
- **Description:** Measure of local rotation in the flow field. High vorticity indicates strong rotational motion, typically found in shear layers and wakes.
- **Formula:** `ω = ∇ × u` (curl of velocity field), then `||ω||` (L2 norm)
- **Source:** `np.linalg.norm(flow.vorticity())`
- **Physical Meaning:** Indicator of turbulence intensity and flow separation
- **Use Case:** Feature for flow regime classification; proxy for turbulence; auxiliary reward signal

#### **pressure** (Mean Pressure)
- **Type:** Float array
- **Shape:** (timesteps,)
- **Range:** Problem-dependent; often centered around 0
- **Description:** Spatial average of the pressure field in the computational domain. Pressure drives the flow and is coupled with velocity through the Navier-Stokes equations.
- **Source:** `np.mean(flow.p.dat.data)`
- **Physical Meaning:** Overall pressure level in the domain; related to flow acceleration
- **Use Case:** Flow characterization; can indicate stagnation or acceleration regions

#### **kinetic_energy** (Mean Kinetic Energy)
- **Type:** Float array
- **Shape:** (timesteps,)
- **Range:** > 0
- **Description:** Average kinetic energy per unit volume in the flow field. Indicates the overall energy content of the flow.
- **Formula:** `KE = 0.5 × mean(u²)` where u is the velocity magnitude
- **Source:** `0.5 * np.mean(flow.u.dat.data**2)`
- **Physical Meaning:** Energy available for turbulence and mixing; higher values indicate more vigorous flow
- **Use Case:** Energy budget analysis; detecting flow transitions

---

### 6. Episode Metadata

#### **Re** (Reynolds Number)
- **Type:** Float array (constant within dataset)
- **Shape:** (timesteps,)
- **Value:** 30.0
- **Description:** Reynolds number characterizing the flow regime. Represents the ratio of inertial forces to viscous forces.
- **Formula:** `Re = U × L / ν` where U is velocity scale, L is length scale, ν is kinematic viscosity
- **Physical Meaning:** Re = 30 is in the laminar-to-transitional regime; vortex shedding is expected
- **Use Case:** Flow regime identification; useful if dataset is extended to include multiple Reynolds numbers

#### **done** (Episode Termination Flag)
- **Type:** Boolean array
- **Shape:** (timesteps,)
- **Values:** False for steps 0-198, True for step 199
- **Description:** Indicates whether the episode has terminated. True only at the last timestep of each episode.
- **Use Case:** Essential for episodic RL algorithms; marks boundaries for return calculation and episode segmentation

#### **timestep** (Step Index)
- **Type:** Integer array
- **Shape:** (timesteps,)
- **Range:** 0 to 199
- **Description:** Index of the current timestep within the episode. Allows temporal awareness in the policy.
- **Use Case:** Time-dependent policies; understanding transient vs. steady-state behavior

#### **Episode Attributes** (HDF5 metadata)
- **num_steps:** Number of timesteps in the episode (200)
- **obs_dim_range:** Range of observation dimensions (e.g., "6-6" means all observations have length 6)
- **act_dim_range:** Range of action dimensions (e.g., "3-3" means all actions have length 3)

---

## Physical Setup

### Geometry
- **Configuration:** Three circular cylinders in equilateral triangle arrangement
- **Cylinder Diameter:** D (reference length)
- **Triangle Side Length:** ~3D
- **Domain:** Rectangular channel with inlet, outlet, and cylinder obstacles

### Flow Conditions
- **Reynolds Number:** Re = 30
- **Flow Regime:** Laminar with vortex shedding
- **Freestream Velocity:** U_∞ (normalized to 1)
- **Fluid:** Incompressible Newtonian fluid (air or water analog)

### Control Mechanism
- **Actuators:** Rotation of the three cylinders
- **Control Authority:** ±flow.MAX_CONTROL radians (typically ±π/2)
- **Control Frequency:** Every timestep (dt ≈ 0.01)

### Numerical Scheme
- **Solver:** IPCS (Incremental Pressure Correction Scheme)
- **Time Discretization:** Implicit
- **Timestep:** dt = flow.DEFAULT_DT (typically 0.01)
- **Spatial Discretization:** Finite elements (Firedrake)

---

## Data Collection Strategy

### Episode Structure
- **Episode Length:** 200 timesteps (~2.0 time units)
- **Total Episodes:** Configurable (2 for testing, 100+ for full dataset)
- **Episode Initialization:** Flow reset to base state with `flow.reset()`

### Action Policy
The dataset contains a mix of control strategies to ensure diversity:
- **30% Random:** Uniform random actions in [-flow.MAX_CONTROL, flow.MAX_CONTROL]³
- **70% Sinusoidal:** Structured periodic control with random frequency
  - Frequency: uniform random in [0.5, 1.0] Hz
  - Amplitude: 0.5 × flow.MAX_CONTROL
  - All three cylinders receive the same sinusoidal signal

This mixed policy ensures:
1. Exploration of state space
2. Structured temporal patterns (useful for learning dynamics)
3. Diversity for off-policy RL training
4. Coverage of both steady and oscillatory control regimes

### Temporal Resolution
- **Physical Time per Step:** ~0.01 time units (flow.DEFAULT_DT)
- **Episode Duration:** ~2.0 time units
- **Total Simulation Time:** 2.0 × num_episodes time units

---

## Loading the Data

### Using the Provided Loader Script

```python
from load_pinball_data import load_episode, get_dataset_info

# Get dataset summary
info = get_dataset_info('pinball_dataset_h5/fluidic_pinball_data.h5')
print(info)

# Load a single episode
episode_data = load_episode('pinball_dataset_h5/fluidic_pinball_data.h5', episode_num=0)

# Access reconstructed arrays
observations = episode_data['obs']          # List of arrays, each shape (6,)
actions = episode_data['actions']           # List of arrays, each shape (3,)
rewards = episode_data['rewards']           # Array shape (200,)
Cd = episode_data['Cd']                     # Array shape (200,)
```