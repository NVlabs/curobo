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


# This script will create a dev docker. Run this script by calling `bash build_dev_docker.sh`
# If you want to build a isaac sim docker, run this script with `bash build_dev_docker.sh isaac`

# Check architecture to build:

image_tag="x86"
isaac_sim_version=""
input_arg="$1"

if [ -z "$input_arg" ]; then
    arch=$(uname -m)
    echo "Argument empty, trying to build based on architecture"
    if [ "$arch" == "x86_64" ]; then
        input_arg="x86"
    elif [ "$arch" == "arm64" ]; then
        input_arg="aarch64"
    elif [ "$arch" == "aarch64" ]; then
        input_arg="aarch64"
    fi
fi

if [ "$input_arg" == "isaac_sim_4.0.0" ]; then
    echo "Building Isaac Sim headless docker"
    dockerfile="isaac_sim.dockerfile"
    image_tag="isaac_sim_4.0.0"
    isaac_sim_version="4.0.0"
elif [ "$input_arg" == "x86" ]; then
    echo "Building for X86 Architecture"
    dockerfile="x86.dockerfile"
    image_tag="x86"
elif [ "$input_arg" = "aarch64" ]; then
    echo "Building for ARM Architecture"
    dockerfile="aarch64.dockerfile"
    image_tag="aarch64"
else
    echo "Unknown Argument. Please pass one of [x86, aarch64, isaac_sim_2022.2.1, isaac_sim_2023.1.0]"
    exit
fi

# build docker file:
# Make sure you enable nvidia runtime by:
# Edit/create the /etc/docker/daemon.json with content:
# {
#    "runtimes": {
#        "nvidia": {
#            "path": "/usr/bin/nvidia-container-runtime",
#            "runtimeArgs": []
#         }
#    },
#    "default-runtime": "nvidia" # ADD this line (the above lines will already exist in your json file)
# }
#
echo "${dockerfile}"

docker build --build-arg ISAAC_SIM_VERSION=${isaac_sim_version} -t curobo_docker:${image_tag} -f ${dockerfile} .
