# Firedrake Installation Guide

## firedrake-configure

To simplify the installation process, Firedrake provides a utility script called `firedrake-configure`. This script can be downloaded by executing:

```bash
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/main/scripts/firedrake-configure
```

> **Note:** `firedrake-configure` does not install Firedrake for you. It is simply a helper script that emits the configuration options that Firedrake needs for the various steps needed during installation.

To improve robustness, `firedrake-configure` is intentionally kept extremely minimal and simple. This means that if you want to install Firedrake in a non-standard way (for instance with a custom installation of PETSc, HDF5 or MPI) then it is your responsibility to modify the output from `firedrake-configure` as necessary.

---

## Installing System Dependencies

If on Ubuntu or macOS, system dependencies can be installed with `firedrake-configure`. On Ubuntu run:

```bash
sudo apt install $(python3.10 firedrake-configure --show-system-packages)
```

This will install the following packages:

```
build-essential flex gfortran git ninja-build pkg-config python3-dev python3-pip
bison cmake libopenblas-dev libopenmpi-dev libfftw3-dev libfftw3-mpi-dev
libhwloc-dev libhdf5-mpi-dev libmumps-ptscotch-dev libmetis-dev libnetcdf-dev
libpnetcdf-dev libptscotch-dev libscalapack-openmpi-dev libsuitesparse-dev
libsuperlu-dev libsuperlu-dist-dev
```

---

## Installing PETSc

For Firedrake to work as expected, a specific version of PETSc must be installed with a specific set of external packages.

**1. Clone the PETSc repository at the required version:**

```bash
git clone --branch $(python3.10 firedrake-configure --show-petsc-version) https://gitlab.com/petsc/petsc.git
cd petsc
```

**2. Run PETSc configure with flags from `firedrake-configure`:**

```bash
python3.10 ../firedrake-configure --no-package-manager --show-petsc-configure-options | xargs -L1 ./configure
```

**3. Compile PETSc using the `make` command prompted by configure:**

```bash
make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-firedrake-default all
```

**4. Test the installation (optional) and return to the parent directory:**

```bash
make check
cd ..
```

---

## Installing Firedrake

Now that the right system packages are installed and PETSc is built, you can install Firedrake.

**1. Create a virtual environment** *(optional but strongly recommended)*:

```bash
python3 -m venv venv-firedrake
. venv-firedrake/bin/activate
```

**2. Purge the pip cache** *(optional but strongly recommended)*:

```bash
pip cache purge
```

Some cached pip packages may be linked against old or missing libraries. For a lighter-weight alternative, you can remove specific packages:

```bash
pip cache remove mpi4py
pip cache remove petsc4py
pip cache remove h5py
pip cache remove slepc4py
pip cache remove libsupermesh
pip cache remove firedrake
```

> **Note:** This list may not be exhaustive.

**3. Set necessary environment variables:**

```bash
export $(python3.10 firedrake-configure --show-env)
```

This will at a minimum set the following variables:

```
PETSC_DIR=/path/to/petsc
PETSC_ARCH=arch-firedrake-{default,complex}
HDF5_MPI=ON
```

**4. Set `PIP_CONSTRAINT` to work around a `setuptools` issue:**

```bash
echo 'setuptools<81' > constraints.txt
export PIP_CONSTRAINT=constraints.txt
```

**5. Ensure MPI is installed:**

```bash
sudo apt install libopenmpi-dev openmpi-bin
```

Locate `mpi.h`:

```bash
find /usr -name "mpi.h" 2>/dev/null
```

Example output:
```
/usr/include/mumps_seq/mpi.h
/usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-15/openmpi/mpi.h
/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h
```

Add the relevant paths:

```bash
export C_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/openmpi/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/openmpi/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:$LIBRARY_PATH
```

**6. Install Firedrake:**

```bash
pip install --no-binary h5py 'firedrake[check]'
```

---

## Checking the Installation

Run some simple tests after installation to check that Firedrake is fully functional:

```bash
firedrake-check
```

This will run a few unit tests that exercise a good chunk of the library's functionality. These tests should take a minute or less.

> **Note:** You need to have installed Firedrake with its optional test dependencies by specifying the `[check]` dependency group as shown above.

---

## Installing Hydrogym

```bash
# Make sure you're home and venv is active
cd ~

sudo apt install git-lfs
git clone https://github.com/dynamicslab/hydrogym.git
cd hydrogym
git lfs install && git lfs fetch --all

# Disable the setuptools constraint
unset PIP_CONSTRAINT
pip install .
```

**Run a sanity check:**

```bash
python -c "import hydrogym.firedrake as hgym; print('All good!')"
```