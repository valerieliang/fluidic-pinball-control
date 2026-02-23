# HydroGym First Time Install Notes

## Prerequisites

Open a WSL window from VSCode using `Ctrl+Shift+P` and select `~` as the current directory.

```bash
cd ~
```

## Install Python 3.10

If on Ubuntu 24.04 or later, add the deadsnakes PPA which has older Python versions:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

## Create Virtual Environment

```bash
python3.10 -m venv hydrogym-venv
```

## Clone Repository

```bash
git clone --recursive https://github.com/dynamicslab/hydrogym.git
```

## Install Firedrake

```bash
cd hydrogym/third_party/firedrake/scripts
~/hydrogym-venv/bin/python firedrake-install
```

## Install HydroGym

```bash
source ~/hydrogym-venv/bin/activate
cd ../../.. && pip install .
```