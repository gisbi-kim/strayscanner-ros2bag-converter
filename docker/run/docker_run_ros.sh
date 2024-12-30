# Set working directory
WORKDIR="$(pwd)/../../"
DATADIR="/home/gskim/Documents/data/iphone/stray"

# Set Docker image
IMAGE="ros:rolling-ros-core"

# Set container name
CONTAINER_NAME="ros-container"

# Run Docker container (with GPU support and custom name)
xhost +local:root
docker run --rm -it \
    --gpus all \
    --name ros-container \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v "$WORKDIR":/ws \
    -v "$DATADIR":/data \
    -w /ws \
    -p 8888:8888 \
    ros:rolling-ros-core \
    /bin/bash
