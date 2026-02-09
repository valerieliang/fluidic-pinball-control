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

## Notes:

- Exit container with `exit`
- All dependencies are installed inside the Docker image

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

You’ll see output like:
```
http://127.0.0.1:8888/?token=abc123...
```
Copy the token.

### Using the notebook:

In your web browser, go to: 
```
http://localhost:8888
```
and open your desired notebook.