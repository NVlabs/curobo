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
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3 AS l4t_pytorch

# Install ros components:
RUN apt-get update &&\
    apt-get install -y sudo git bash unattended-upgrades glmark2 &&\
    rm -rf /var/lib/apt/lists/*


# Deal with getting tons of debconf messages
# See: https://github.com/phusion/baseimage-docker/issues/58
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# TODO: Don't hardcode timezone setting to Los_Angeles, pull from host computer
# Set timezone info
RUN apt-get update && apt-get install -y \
  tzdata \
  && rm -rf /var/lib/apt/lists/* \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
  && echo "America/Los_Angeles" > /etc/timezone \
  && dpkg-reconfigure -f noninteractive tzdata

# Install apt-get packages necessary for building, downloading, etc
# NOTE: Dockerfile best practices recommends having apt-get update
# and install commands in one line to avoid apt-get caching issues.
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run
RUN apt-get update && apt-get install -y \
  curl \
  lsb-core \
  software-properties-common \
  wget \
  && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:git-core/ppa

RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  git \
  git-lfs \
  iputils-ping \
  make \
  openssh-server \
  openssh-client \
  libeigen3-dev \
  libssl-dev \
  python3-pip \
  python3-ipdb \
  python3-tk \
  python3-wstool \
  sudo git bash unattended-upgrades \
  apt-utils \
  terminator \
  && rm -rf /var/lib/apt/lists/*

ARG ROS_PKG=ros_base # desktop does not work
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace


#
# add the ROS deb repo to the apt sources list
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          git \
		cmake \
		build-essential \
		curl \
		wget \
		gnupg2 \
		lsb-release \
		ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -


#
# install bootstrap dependencies
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          libpython3-dev \
          python3-rosdep \
          python3-rosinstall-generator \
          python3-vcstool \
          build-essential && \
    rosdep init && \
    rosdep update && \
    rm -rf /var/lib/apt/lists/*


#
# download/build the ROS source
#
RUN mkdir ros_catkin_ws && \
    cd ros_catkin_ws && \
    rosinstall_generator ${ROS_PKG} vision_msgs --rosdistro ${ROS_DISTRO} --deps --tar > ${ROS_DISTRO}-${ROS_PKG}.rosinstall && \
    mkdir src && \
    vcs import --input ${ROS_DISTRO}-${ROS_PKG}.rosinstall ./src && \
    apt-get update && \
    rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro ${ROS_DISTRO} --skip-keys python3-pykdl -y && \
    python3 ./src/catkin/bin/catkin_make_isolated --install --install-space ${ROS_ROOT} -DCMAKE_BUILD_TYPE=Release && \
    rm -rf /var/lib/apt/lists/*


RUN pip3 install trimesh \
  numpy-quaternion \
  networkx \
  pyyaml \
  rospkg \ 
  rosdep \
  empy
# copy pkgs directory:
COPY pkgs /pkgs


# install warp:
RUN cd /pkgs/warp && python3 build_lib.py && pip3 install .

# install curobolib-extras/warp_torch:
RUN cd /pkgs/warp_torch && pip3 install .

# install nvblox:
RUN apt-get update && \
    apt-get install -y libgoogle-glog-dev libgtest-dev curl libsqlite3-dev && \
    cd /usr/src/googletest && cmake . && cmake --build . --target install && \
    rm -rf /var/lib/apt/lists/*
RUN cd /pkgs/nvblox/nvblox && mkdir build && cd build && cmake .. && \
    make -j8 && make install

# install curobolib-extras/nvblox_torch:
RUN cd /pkgs/nvblox_torch && sh install.sh 

# install curobolib and curobo:
RUN cd /pkgs/curobolib && pip3 install .

RUN cd /pkgs/curobo && pip3 install .

# sudo apt-get install lcov libbullet-extras-dev python3-catkin-tools swig ros-noetic-ifopt libyaml-cpp-dev libjsoncpp-dev
# pip3 install tqdm
# sudo apt-get install liborocos-kdl-dev ros-noetic-fcl libpcl-dev libompl-dev  libnlopt-cxx-dev liburdf-dev libkdlparser-dev
# clone taskflow and compile


# libfranka:
RUN apt-get update && apt-get install -y build-essential cmake git libpoco-dev libeigen3-dev && cd /pkgs && git clone --recursive https://github.com/frankaemika/libfranka && cd libfranka && git checkout 0.7.1 && git submodule update && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. && \
  cmake --build . && cpack -G DEB && dpkg -i libfranka*.deb &&  rm -rf /var/lib/apt/lists/*
# franka_ros and franka_motion_control:
RUN mkdir -p /curobo_ws/src && mv /pkgs/curobo_ros /curobo_ws/src/ && mv /pkgs/franka_motion_control /curobo_ws/src/

RUN cd /curobo_ws/src && git clone --branch 0.7.1 https://github.com/frankaemika/franka_ros.git && \
    git clone https://github.com/justagist/franka_panda_description.git 


RUN apt-get update && apt-get install -y ros-noetic-robot-state-publisher ros-noetic-rviz \
  ros-noetic-moveit-visualization ros-noetic-moveit-msgs \
  ros-noetic-controller-interface ros-noetic-combined-robot-hw ros-noetic-joint-limits-interface \
  ros-noetic-control-msgs ros-noetic-controller-manager ros-noetic-realtime-tools ros-noetic-eigen-conversions ros-noetic-tf-conversions ros-noetic-moveit-ros-visualization && rm -rf /var/lib/apt/lists/*

RUN cd /curobo_ws && \
  /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make -DCMAKE_BUILD_TYPE=Release"

#RUN apt-get update && apt-get install -y ros-noetic-robot-state-publisher ros-noetic-rviz ros-noetic-moveit-msgs\
#  ros-noetic-moveit-ros ros-noetic-controller-interface ros-noetic-combined-robot-hw ros-noetic-joint-limits-interface ros-noetic-control-msgs && rm -rf /var/lib/apt/lists/*