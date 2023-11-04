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
docker run --rm -it \
--privileged --mount type=bind,src=/home/$USER/code,target=/home/$USER/code \
-e NVIDIA_DISABLE_REQUIRE=1 \
-e NVIDIA_DRIVER_CAPABILITIES=all  --device /dev/dri \
--hostname ros1-docker \
--add-host ros1-docker:127.0.0.1 \
--gpus all \
--network host \
--env ROS_MASTER_URI=http://127.0.0.1:11311 \
--env ROS_IP=127.0.0.1 \
--env DISPLAY=unix$DISPLAY \
--volume /tmp/.X11-unix:/tmp/.X11-unix \
--volume /dev/input:/dev/input \
curobo_user_docker:latest