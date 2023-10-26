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
FROM nvcr.io/nvidia/pytorch:23.02-py3


RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && apt-get install -y \
  tzdata \
  && rm -rf /var/lib/apt/lists/* \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
  && echo "America/Los_Angeles" > /etc/timezone \
  && dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update &&\
    apt-get install -y sudo git bash software-properties-common graphviz &&\
    rm -rf /var/lib/apt/lists/*



RUN python -m pip install --upgrade pip && python3 -m pip install graphviz

# Install ROS noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' \
&& apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
&& apt-get update && apt-get install -y \
  ros-noetic-desktop-full git build-essential python3-rosdep \
  && rm -rf /var/lib/apt/lists/*


# install realsense and azure kinect
# Install the RealSense library (https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages)
#RUN sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
#RUN sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
#RUN apt-get update && apt-get install -y \
#  librealsense2-dkms \
#  software-properties-common \
#  librealsense2-utils \
#  && rm -rf /var/lib/apt/lists/*


# install moveit from source for all algos:
ARG ROS_DISTRO=noetic
RUN apt-get update && apt-get install -y \
  ros-$ROS_DISTRO-apriltag-ros \
  ros-$ROS_DISTRO-realsense2-camera \
  ros-$ROS_DISTRO-ros-numpy \
  ros-$ROS_DISTRO-vision-msgs \
  ros-$ROS_DISTRO-franka-ros \
  ros-$ROS_DISTRO-moveit-resources \
  ros-$ROS_DISTRO-rosparam-shortcuts \
  libglfw3-dev \
  ros-$ROS_DISTRO-collada-urdf \
  ros-$ROS_DISTRO-ur-msgs \
  swig \
  && rm -rf /var/lib/apt/lists/*


RUN apt-get update && rosdep init && rosdep update && apt-get install -y ros-noetic-moveit-ros-visualization && rm -rf /var/lib/apt/lists/*
RUN pip3 install netifaces

