FROM firedrakeproject/firedrake:2023-11

ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-psutil \
    gmsh \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==1.13.0+cpu \
    functorch==1.13.0 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    control>=0.9.2 \
    dmsuite>=0.1.1 \
    gym>=0.26.2 \
    modred>=2.1.0 \
    hydrogym==0.1.2.1 \
    evotorch==0.3.0
