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

echo $1
echo $2

docker build --build-arg USERNAME=$USER --no-cache --build-arg USER_ID=$1 --build-arg IMAGE_TAG=$2 -f user.dockerfile --tag curobo_docker:user_$2 . 