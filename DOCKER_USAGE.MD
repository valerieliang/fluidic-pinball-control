# DOCKER_USAGE.MD

This project uses a custom Docker image based on Firedrake.

## Build and run:

Initial build:
```
docker compose up --build
```

If already built:
```
docker compose up
```

## Start Docker Container:

In a new terminal, enter:
```
docker exec -it hydrogym /bin/bash
source /home/firedrake/firedrake/bin/activate
```

## Using Jupyter Notebook:

Install dependencies:
```
pip install jupyterlab notebook ipykernel
```

If first time: create kernel:
```
python -m ipykernel install --user --name firedrake --display-name "Firedrake"
```

### Start Jupyter in Docker:

Run:
```
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
Follow the output in the terminal to use the notebook.

## Notes:

- Exit container with `exit`
- All dependencies are installed inside the Docker image
