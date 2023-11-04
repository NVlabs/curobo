#!/bin/bash
##
## Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
##
## NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
## property and proprietary rights in and to this material, related
## documentation and any modifications thereto. Any use, reproduction,
## disclosure or distribution of this material and related documentation
## without an express license agreement from NVIDIA CORPORATION or
## its affiliates is strictly prohibited.
##

if [ $1 == "x86" ]; then

    docker run --rm -it \
    --privileged \
    -e NVIDIA_DISABLE_REQUIRE=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all  --device /dev/dri \
    --hostname ros1-docker \
    --add-host ros1-docker:127.0.0.1 \
    --gpus all \
    --network host \
    --env DISPLAY=unix$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume /dev:/dev \
    curobo_docker:$1

elif [ $1 == "aarch64" ]; then

    docker run --rm -it \
    --runtime nvidia \
    --hostname ros1-docker \
    --add-host ros1-docker:127.0.0.1 \
    --network host \
    --gpus all \
    --env ROS_HOSTNAME=localhost \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume /dev/input:/dev/input \
    curobo_docker:$1

elif [[ $1 == *isaac_sim* ]] ; then

    docker run --name container_$1 --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
        --privileged \
        -e "PRIVACY_CONSENT=Y" \
        -v $HOME/.Xauthority:/root/.Xauthority \
        -e DISPLAY \
        -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
        -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
        -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
        -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
        -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
        -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
        -v ~/docker/isaac-sim/documents:/root/Documents:rw \
        --volume /dev:/dev \
        curobo_docker:$1

else
    echo "Unknown docker"
fi