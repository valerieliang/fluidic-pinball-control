# DOCKER_USAGE.MD

This project uses a custom Docker image based on Firedrake.

## Build and run:

```
docker compose up --build
```

## If already built:

```
docker compose up
```

## In a new terminal, enter:

```
docker exec -it hydrogym /bin/bash
source /home/firedrake/firedrake/bin/activate
```

## Notes:

- Exit container with `exit`
- All dependencies are installed inside the Docker image

# Docker

This project uses a custom Docker image based on Firedrake.

## Build and run:

```
docker compose up --build
```

## If already built:

```
docker compose up
```

## In a new terminal, enter:

```
docker exec -it hydrogym /bin/bash
source /home/firedrake/firedrake/bin/activate
```

### Test: view hydrogym demo:

```
cd hydrogym/examples/demo
python3 run-transient.py
```

## Notes:

- Exit container with `exit`
- All dependencies are installed inside the Docker image
