# ---- Base Firedrake image ----
FROM firedrakeproject/firedrake:latest

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-psutil \
    gmsh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade pip ----
RUN python3 -m pip install --upgrade pip

# ---- Python dependencies (from your conda env) ----
RUN python3 -m pip install \
    numpy \
    scipy \
    matplotlib \
    control>=0.9.2 \
    dmsuite>=0.1.1 \
    gym>=0.26.2 \
    modred>=2.1.0 \
    evotorch==0.3.0 \
    hydrogym==0.1.2.1

# ---- Sanity check at build time (optional but recommended) ----
RUN python3 - <<EOF
import firedrake
import hydrogym
import psutil
import gym
print("All core imports OK")
EOF

# ---- Default working directory ----
WORKDIR /workspace
