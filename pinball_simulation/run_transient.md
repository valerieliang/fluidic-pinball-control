# 2D Pinball Flow Simulation Documentation

## Overview
This script simulates fluid flow around a pinball configuration (cylinders in a triangular arrangement) at Reynolds number 30 using the HydroGym library with Firedrake backend. The simulation includes real-time monitoring of lift coefficients and system memory usage.

## Dependencies
```python
import numpy as np           # Numerical operations
import psutil               # System monitoring
import hydrogym.firedrake as hgym  # Fluid dynamics simulation framework
```

## Key Components

### 1. Flow Configuration
- **Reynolds Number**: Re = 30 (laminar flow regime)
- **Mesh**: "fine" resolution mesh
- **Restart Capability**: Can restart from previous simulation state
- **Output Directory**: Current directory (`.`)

### 2. Monitoring and Logging
#### Memory Usage Tracking
The `log_postprocess` function extracts:
- Three lift coefficients (CL[0], CL[1], CL[2]) from flow observations
- System memory usage percentage via `psutil`

#### Output Format
```
t: 0.00,     CL[0]: 0.000,     CL[1]: 0.000,     CL[2]: 0.000     Mem: 45.3
```

#### Data Output
- **Console**: Real-time formatted output at 1-second intervals
- **File**: `coeffs.dat` - raw coefficient and memory data

### 3. Simulation Parameters
```python
Tf = 1.0           # Total simulation time (seconds)
dt = 0.1           # Time step size
method = "BDF"     # Backward Differentiation Formula time integration
stabilization = "gls"  # Galerkin Least Squares stabilization
```

### 4. Control System (Currently Disabled)
A placeholder controller function is defined but commented out. When enabled, it would apply constant control inputs to actuators:
```python
# Returns: [actuator_1, actuator_2, actuator_3] = [0.0, 1.0, 1.0]
```

## File Structure
```
./
├── run-transient.py            # Main simulation script
├── coeffs.dat                  # Time series data (CL values + memory)
└── checkpoint.h5               # Simulation restart file (generated periodically)
```

## Usage Notes

### Running the Simulation
Execute the script to run a 1-second simulation:
```bash
python run-transient.py
```

### Restarting Simulations
To restart from a checkpoint:
1. Set `restart = "checkpoint.h5"` in the `hgym.Pinball` constructor
2. Ensure the checkpoint file exists in the working directory

### Modifying Parameters
- **Simulation Time**: Adjust `Tf` for longer/shorter runs
- **Resolution**: Change `mesh` parameter ("coarse", "medium", "fine")
- **Reynolds Number**: Modify `Re` in `hgym.Pinball` constructor
- **Time Step**: Adjust `dt` (smaller for stability, larger for speed)

### Adding Control
Uncomment the `controller` argument in `hgym.integrate()` to enable active flow control.

## Output Interpretation
- **CL[0], CL[1], CL[2]**: Lift coefficients on each cylinder
- **Mem**: System memory usage percentage
- The coefficients typically oscillate due to vortex shedding at Re=30

## Troubleshooting
1. **Memory Issues**: Reduce mesh resolution or time step if memory usage approaches 100%
2. **Stability Problems**: Decrease `dt` or try different stabilization methods
3. **Checkpoint Errors**: Ensure write permissions in output directory

## Computational Requirements
- **Memory**: Depends on mesh resolution ("fine" mesh requires more memory)
- **Time**: Simulation time scales with `Tf/dt` and mesh complexity
- **Storage**: `coeffs.dat` grows with simulation duration

This simulation is particularly useful for studying:
- Wake interactions in multiple cylinder configurations
- Active flow control strategies
- Validation of numerical methods in bluff body flows