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

# Make sure you have pulled all required repos into pkgs folder (see pull_repos.sh script)

# Check architecture to build:
arch=`uname -m`

if [ ${arch} == "x86_64" ]; then
    echo "Building for X86 Architecture"
    dockerfile="x86.dockerfile"
elif [ ${arch} = "aarch64" ]; then
    echo "Building for ARM Architecture"
    dockerfile="arm64.dockerfile"
else
    echo "Unknown Architecture, defaulting to " + ${arch}
    dockerfile="x86.dockerfile"
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
docker build -t curobo_docker:latest -f ${dockerfile} .
