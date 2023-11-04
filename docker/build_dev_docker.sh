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
arch=`uname -m`

image_tag="x86"
isaac_sim_version=""

if [ $1 == "isaac_sim_2022.2.1" ]; then
    echo "Building Isaac Sim docker"
    dockerfile="isaac_sim.dockerfile"
    image_tag="isaac_sim_2022.2.1"
    isaac_sim_version="2022.2.1"
elif [ $1 == "isaac_sim_2023.1.0" ]; then
    echo "Building Isaac Sim headless docker"
    dockerfile="isaac_sim.dockerfile"
    image_tag="isaac_sim_2023.1.0"
    isaac_sim_version="2023.1.0"
elif [ ${arch} == "x86" ]; then
    echo "Building for X86 Architecture"
    dockerfile="x86.dockerfile"
    image_tag="x86"
elif [ ${arch} == "x86_64" ]; then
    echo "Building for x86_64 Architecture"
    dockerfile="x86.dockerfile"
    image_tag="x86_64"
elif [ ${arch} = "aarch64" ]; then
    echo "Building for ARM Architecture"
    dockerfile="aarch64.dockerfile"
    image_tag="aarch64"
else
    echo "Unknown Architecture"
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
