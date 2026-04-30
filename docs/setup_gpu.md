# GPU Setup Guide: Firedrake + Hydrogym on WSL2 with RTX 5070 Ti (sm_120 Blackwell)

## System

| Component | Version |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| GPU | NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120) |
| Windows Driver | 596.21 |
| CUDA Version (host) | 13.2 |
| CUDA Toolkit (WSL2) | 13.1 |
| Python | 3.10.19 |
| PETSc | 3.24.0 |
| Firedrake | 2025.10.2 |
| Hydrogym | 0.1.2.3 |

---

## Key Decisions & Why

### Do not install the CUDA driver inside WSL2
The Windows host driver exposes CUDA to WSL2 via a stub library. Only the toolkit
(compiler + headers) needs to be installed inside WSL2, not the driver itself.

### SuperLU_DIST GPU support disabled
SuperLU_DIST's CUDA code triggers a bug in CUDA 13.1's CCCL headers (`fma.h`):
`invalid combination of type specifiers`. The fix is to use the system-installed
pre-built `libsuperlu-dist-dev` instead of letting PETSc download and compile
SuperLU_DIST with CUDA. This has no impact on Firedrake's GPU capability — PETSc's
own CUDA kernels (cuSPARSE, cuBLAS, cuSolver) are what matter.

### `-use_gpu_aware_mpi 0` required
WSL2's OpenMPI is not GPU-aware. Without this flag PETSc aborts on any GPU vector
operation. Set it permanently via `PETSC_OPTIONS`.

### C++17 dialect required
CUDA 13.1's Thrust/CUB/libcu++ headers require C++17 minimum. Must be passed
explicitly to both the C++ and CUDA compilers via `--with-cxx-dialect=C++17` and
`--with-cuda-dialect=C++17`.

---

## Step 0: Install CUDA Toolkit in WSL2

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-1

echo 'export PATH=/usr/local/cuda-13.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version   # should show 13.1
nvidia-smi       # should show RTX 5070 Ti
```

---

## Step 1: Download firedrake-configure

```bash
cd ~
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/main/scripts/firedrake-configure
```

---

## Step 2: Install System Dependencies

```bash
sudo apt install $(python3.10 firedrake-configure --show-system-packages)
```

---

## Step 3: Build PETSc with CUDA

### Clone PETSc at the version Firedrake requires

```bash
cd ~
git clone --branch $(python3.10 firedrake-configure --show-petsc-version) https://gitlab.com/petsc/petsc.git
cd petsc
```

### Configure

The configure command deviates from the standard Firedrake guide in four ways:
1. `--with-cuda` flags added
2. `--with-cxx-dialect=C++17` and `--with-cuda-dialect=C++17` added (required for CUDA 13.1)
3. `--download-superlu_dist` removed (broken with CUDA 13.1 CCCL headers)
4. `--with-superlu_dist-include` / `--with-superlu_dist-lib` added to use system pre-built version

```bash
./configure \
  --with-c2html=0 \
  --with-debugging=0 \
  --with-fortran-bindings=0 \
  --with-shared-libraries=1 \
  --with-strict-petscerrorcode \
  PETSC_ARCH=arch-firedrake-default \
  --COPTFLAGS='-O3 -march=native -mtune=native' \
  --CXXOPTFLAGS='-O3 -march=native -mtune=native' \
  --FOPTFLAGS='-O3 -march=native -mtune=native' \
  --download-bison \
  --download-fftw \
  --download-hdf5 \
  --download-hwloc \
  --download-metis \
  --download-mumps \
  --download-netcdf \
  --download-pnetcdf \
  --download-ptscotch \
  --download-scalapack \
  --download-suitesparse \
  --download-zlib \
  --download-hypre \
  --with-superlu_dist=1 \
  --with-superlu_dist-include=/usr/include/superlu-dist \
  --with-superlu_dist-lib=/usr/lib/x86_64-linux-gnu/libsuperlu_dist.so \
  --with-cuda=1 \
  --with-cuda-dir=/usr/local/cuda-13.1 \
  --with-cxx-dialect=C++17 \
  --with-cuda-dialect=C++17 \
  CUDAOPTFLAGS="-O3 --use_fast_math -gencode arch=compute_120,code=sm_120"
```

Configure should complete with a summary showing:
```
CUDA:
  Version:    13.1
  CUDA SM 120
```

### Compile and verify

```bash
make PETSC_DIR=/home/$USER/petsc PETSC_ARCH=arch-firedrake-default all -j$(nproc)
make PETSC_DIR=/home/$USER/petsc PETSC_ARCH=arch-firedrake-default check
cd ~
```

Expected check output includes:
```
C/C++ example src/snes/tutorials/ex19 run successfully with HYPRE/CUDA
C/C++ example src/snes/tutorials/ex19 run successfully with CUDA
```

---

## Step 4: Install Firedrake

```bash
python3 -m venv venv-firedrake
source ~/venv-firedrake/bin/activate

pip cache purge

export PETSC_DIR=~/petsc
export PETSC_ARCH=arch-firedrake-default
export HDF5_MPI=ON

echo 'setuptools<81' > ~/constraints.txt
export PIP_CONSTRAINT=~/constraints.txt

export C_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/openmpi/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/openmpi/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:$LIBRARY_PATH

pip install --no-binary h5py 'firedrake[check]'
firedrake-check
```

---

## Step 5: Set Permanent Environment Variables

```bash
echo 'export PETSC_OPTIONS="-use_gpu_aware_mpi 0"' >> ~/.bashrc
source ~/.bashrc
```

### Verify CUDA works from Python

```bash
PETSC_OPTIONS="-use_gpu_aware_mpi 0" python3 -c "
from firedrake.petsc import PETSc
v = PETSc.Vec().create()
v.setSizes(100)
v.setType('cuda')
v.set(1.0)
print('SUCCESS: PETSc CUDA vector type:', v.getType())
v.destroy()
"
# Expected output: SUCCESS: PETSc CUDA vector type: seqcuda
```

---

## Step 6: Install Hydrogym

```bash
cd ~
sudo apt install -y git-lfs
git clone https://github.com/dynamicslab/hydrogym.git
cd hydrogym
git lfs install && git lfs fetch --all
unset PIP_CONSTRAINT
pip install .

python -c "import hydrogym.firedrake as hgym; print('All good')"
# Expected output: All good
```

---

## Step 7: Activating the Environment in Future Sessions

Add to `~/.bashrc` or run manually each session:

```bash
source ~/venv-firedrake/bin/activate
export PETSC_DIR=~/petsc
export PETSC_ARCH=arch-firedrake-default
export HDF5_MPI=ON
export PETSC_OPTIONS="-use_gpu_aware_mpi 0"
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
```

---

## Using GPU Acceleration in Simulations

Pass these solver parameters to activate GPU-accelerated linear solves:

```python
solver_parameters = {
    "mat_type": "aijcusparse",   # GPU sparse matrix (cuSPARSE)
    "vec_type": "cuda",           # GPU vectors
    "ksp_type": "cg",
    "pc_type": "jacobi",
    "ksp_rtol": 1e-6,
}
```

To enable GPU globally for an entire script:

```python
import os
os.environ["PETSC_OPTIONS"] = "-vec_type cuda -mat_type aijcusparse -use_gpu_aware_mpi 0"
```

---

## Known Issues & Notes

| Issue | Cause | Fix |
|---|---|---|
| `MPI_ABORT` on GPU vector creation | WSL2 OpenMPI not GPU-aware | Set `PETSC_OPTIONS="-use_gpu_aware_mpi 0"` |
| SuperLU_DIST CUDA build fails | CUDA 13.1 CCCL `fma.h` bug (`invalid combination of type specifiers`) | Use system `libsuperlu-dist-dev` instead of `--download-superlu_dist` |
| `Thrust requires at least C++17` | PETSc defaulting to C++14 | Add `--with-cxx-dialect=C++17 --with-cuda-dialect=C++17` |
| `Option left: name:-c` warning | PETSc parsing Python's `-c` flag | Harmless, ignore |
| Gym deprecation warning on Hydrogym import | Hydrogym depends on unmaintained `gym` | Harmless, does not affect functionality |