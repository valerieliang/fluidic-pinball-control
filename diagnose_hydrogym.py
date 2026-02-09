"""
Diagnostic script to verify HydroGym setup and validate against paper specifications
"""
import numpy as np
import h5py
import sys

def check_hydrogym_import():
    """Check if HydroGym can be imported"""
    try:
        from hydrogym.firedrake import Pinball, IPCS
        print("OK: HydroGym imported successfully")
        return True
    except ImportError as e:
        print(f"FAIL: Failed to import HydroGym: {e}")
        return False

def check_flow_initialization():
    """Check if flow can be initialized"""
    try:
        from hydrogym.firedrake import Pinball, IPCS
        flow = Pinball()
        solver = IPCS(flow, dt=flow.DEFAULT_DT)
        print("OK: Flow and solver initialized successfully")

        # Check attributes
        print(f"  - Flow has ACT_DIM: {hasattr(flow, 'ACT_DIM')}")
        print(f"  - Flow has OBS_DIM: {hasattr(flow, 'OBS_DIM')}")
        print(f"  - Flow has MAX_CONTROL: {hasattr(flow, 'MAX_CONTROL')}")
        print(f"  - Flow has Re: {hasattr(flow, 'Re')}")

        if hasattr(flow, 'ACT_DIM'):
            print(f"  - Action dimension: {flow.ACT_DIM}")
        if hasattr(flow, 'Re'):
            print(f"  - Reynolds number: {flow.Re}")

        return True, flow, solver
    except Exception as e:
        print(f"FAIL: Failed to initialize flow: {e}")
        return False, None, None

def test_force_computation(flow, solver):
    """Test if force computation works"""
    try:
        # Reset and initialize
        flow.reset()

        # Run a few timesteps
        print("\n  Running initialization timesteps...")
        for i in range(20):
            solver.step(i)

        # Try to compute forces
        forces = flow.compute_forces()
        print("OK: Force computation successful")
        print(f"  - Forces type: {type(forces)}")
        print(f"  - Forces value: {forces}")

        # Extract CD and CL
        cd, cl = None, None
        if isinstance(forces, dict):
            cd = forces.get("Cd") or forces.get("cd")
            cl = forces.get("Cl") or forces.get("cl")
        elif isinstance(forces, (tuple, list)) and len(forces) >= 2:
            cd = forces[0]
            cl = forces[1]

        print(f"  - CD: {cd}")
        print(f"  - CL: {cl}")

        # Handle arrays (pinball has 3 cylinders)
        if isinstance(cd, (list, np.ndarray)):
            cd = np.array(cd)
            print(f"  - CD is array with {len(cd)} elements (one per cylinder)")
            print(f"    CD values: {cd}")
            print(f"    Total CD: {np.sum(cd):.3f}")
            if np.all(np.isfinite(cd)):
                print("OK: All CD values are valid")
            else:
                print("FAIL: Some CD values are NaN")
        elif cd is not None and np.isfinite(cd):
            print(f"OK: CD is valid scalar: {cd}")
        else:
            print("FAIL: CD is NaN or None")

        if isinstance(cl, (list, np.ndarray)):
            cl = np.array(cl)
            print(f"  - CL is array with {len(cl)} elements (one per cylinder)")
            print(f"    CL values: {cl}")
            if np.all(np.isfinite(cl)):
                print("OK: All CL values are valid")
            else:
                print("FAIL: Some CL values are NaN")
        elif cl is not None and np.isfinite(cl):
            print(f"OK: CL is valid scalar: {cl}")
        else:
            print("FAIL: CL is NaN or None")

        return cd, cl
    except Exception as e:
        print(f"FAIL: Force computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def check_dataset(filepath):
    """Check if dataset exists and inspect contents"""
    try:
        with h5py.File(filepath, "r") as f:
            print(f"\nOK: Dataset file opened: {filepath}")
            print(f"  Episodes: {len([k for k in f.keys() if k.startswith('episode')])}")

            for ep_name in sorted([k for k in f.keys() if k.startswith("episode")])[:1]:
                grp = f[ep_name]
                print(f"\n  {ep_name}:")
                print(f"    Datasets: {list(grp.keys())}")
                print(f"    Attributes: {dict(grp.attrs)}")

                # Check CD and CL
                if "Cd" in grp:
                    cd_data = grp["Cd"][:]
                    cd_valid = cd_data[np.isfinite(cd_data)]
                    print(f"    CD: {len(cd_valid)}/{len(cd_data)} valid samples")
                    if len(cd_valid) > 0:
                        print(f"      Mean: {np.mean(cd_valid):.3f}, Std: {np.std(cd_valid):.3f}")
                        print(f"      Range: [{np.min(cd_valid):.3f}, {np.max(cd_valid):.3f}]")

                if "Cl" in grp:
                    cl_data = grp["Cl"][:]
                    cl_valid = cl_data[np.isfinite(cl_data)]
                    print(f"    CL: {len(cl_valid)}/{len(cl_data)} valid samples")
                    if len(cl_valid) > 0:
                        print(f"      Mean: {np.mean(cl_valid):.3f}, Std: {np.std(cl_valid):.3f}")
                        print(f"      Range: [{np.min(cl_valid):.3f}, {np.max(cl_valid):.3f}]")

                # Check f0
                if "f0" in grp.attrs:
                    print(f"    f0 (frequency): {grp.attrs['f0']:.4f}")

        return True
    except Exception as e:
        print(f"FAIL: Failed to check dataset: {e}")
        return False

def print_paper_reference():
    """Print expected values from Table SI 4"""
    print("\n" + "=" * 60)
    print("Expected values from Table SI 4 (Re=100, 2D Pinball):")
    print("=" * 60)
    print("  f0 (frequency):  0.088")
    print("  CD (drag):       2.904")
    print("  CL,2 (lift):    -0.079")
    print("  CL,3 (lift):     0.110")
    print("\nNote: These are time-averaged values for the uncontrolled case")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("HydroGym Diagnostic Tool")
    print("=" * 60)

    # Step 1: Check import
    print("\n1. Checking HydroGym import...")
    if not check_hydrogym_import():
        sys.exit(1)

    # Step 2: Check flow initialization
    print("\n2. Checking flow initialization...")
    success, flow, solver = check_flow_initialization()
    if not success:
        sys.exit(1)

    # Step 3: Test force computation
    print("\n3. Testing force computation...")
    cd, cl = test_force_computation(flow, solver)

    # Step 4: Print reference values
    print_paper_reference()

    # Step 5: Check existing dataset if provided
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        print("\n4. Checking existing dataset...")
        check_dataset(dataset_path)

    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("=" * 60)
