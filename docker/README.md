<!--
Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# Docker Instructions

## Running docker from NGC (Recommended)
1. `sh build_user_docker.sh $UID`
2. `sh start_docker_x86.sh` will start the docker

## Building your own docker image with CuRobo
1. Add default nvidia runtime to enable cuda compilation during docker build:
    ```
    Edit/create the /etc/docker/daemon.json with content:
    {
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" # ADD this line (the above lines will already exist in your json file)
    }
    ```
2. `sh pull_repos.sh`
3. `bash build_dev_docker.sh`
4. Change the docker image name in `user.dockerfile`
5. `sh build_user_docker.sh`
6. `sh start_docker_x86.sh` will start the docker

