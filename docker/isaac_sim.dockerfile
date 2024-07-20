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
ARG DEBIAN_FRONTEND=noninteractive
ARG BASE_DIST=ubuntu20.04
ARG CUDA_VERSION=11.4.2
ARG ISAAC_SIM_VERSION=4.0.0


FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION} AS isaac-sim

FROM nvcr.io/nvidia/cudagl:${CUDA_VERSION}-devel-${BASE_DIST}


# this does not work for 2022.2.1
#$FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-${BASE_DIST}

LABEL maintainer "User Name"

ARG VULKAN_SDK_VERSION=1.3.224.1



# Deal with getting tons of debconf messages
# See: https://github.com/phusion/baseimage-docker/issues/58
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


# add GL if using a cuda docker instead of cudagl:
#RUN apt-get update && apt-get install -y --no-install-recommends \
#        pkg-config \
#        libglvnd-dev \
#        libgl1-mesa-dev \
#        libegl1-mesa-dev \
#        libgles2-mesa-dev && \
#    rm -rf /var/lib/apt/lists/*

# Set timezone info
RUN apt-get update && apt-get install -y \
  tzdata \
  software-properties-common \
  && rm -rf /var/lib/apt/lists/* \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
  && echo "America/Los_Angeles" > /etc/timezone \
  && dpkg-reconfigure -f noninteractive tzdata \
  && add-apt-repository -y ppa:git-core/ppa \
  && apt-get update && apt-get install -y \
  curl \
  lsb-core \
  wget \
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


# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cudagl

RUN apt-get update && apt-get install -y --no-install-recommends \
    libatomic1 \
    libegl1 \
    libglu1-mesa \
    libgomp1 \
    libsm6 \
    libxi6 \
    libxrandr2 \
    libxt6 \
    libfreetype-dev \
    libfontconfig1 \
    openssl \
    libssl1.1 \
    wget \
    vulkan-utils \
&& apt-get -y autoremove \
&& apt-get clean autoclean \
&& rm -rf /var/lib/apt/lists/*



# Download the Vulkan SDK and extract the headers, loaders, layers and binary utilities
RUN wget -q --show-progress \
    --progress=bar:force:noscroll \
    https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz \
    -O /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz \
    && echo "Installing Vulkan SDK ${VULKAN_SDK_VERSION}" \
    && mkdir -p /opt/vulkan \
    && tar -xf /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz -C /opt/vulkan \
    && mkdir -p /usr/local/include/ && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/include/* /usr/local/include/ \
    && mkdir -p /usr/local/lib && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/* /usr/local/lib/ \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/libVkLayer_*.so /usr/local/lib \
    && mkdir -p /usr/local/share/vulkan/explicit_layer.d \
    && cp /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/etc/vulkan/explicit_layer.d/VkLayer_*.json /usr/local/share/vulkan/explicit_layer.d \
    && mkdir -p /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/share/vulkan/registry/* /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/bin/* /usr/local/bin \
    && ldconfig \
    && rm /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz && rm -rf /opt/vulkan


# Setup the required capabilities for the container runtime
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# Open ports for live streaming
EXPOSE 47995-48012/udp \
       47995-48012/tcp \
       49000-49007/udp \
       49000-49007/tcp \
       49100/tcp \
       8011/tcp \
       8012/tcp \
       8211/tcp \
       8899/tcp \
       8891/tcp

ENV OMNI_SERVER http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/${ISAAC_SIM_VERSION}
# ENV OMNI_SERVER omniverse://localhost/NVIDIA/Assets/Isaac/2022.1
# ENV OMNI_USER admin
# ENV OMNI_PASS admin
ENV MIN_DRIVER_VERSION 525.60.11

# Copy Isaac Sim files
COPY --from=isaac-sim /isaac-sim /isaac-sim
RUN mkdir -p /root/.nvidia-omniverse/config
COPY --from=isaac-sim /root/.nvidia-omniverse/config /root/.nvidia-omniverse/config
COPY --from=isaac-sim /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
COPY --from=isaac-sim /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

WORKDIR /isaac-sim


ENV TORCH_CUDA_ARCH_LIST="7.0+PTX"




# create an alias for omniverse python
ENV omni_python='/isaac-sim/python.sh'

RUN echo "alias omni_python='/isaac-sim/python.sh'" >> ~/.bashrc


# Add cache date to avoid using cached layers older than this
ARG CACHE_DATE=2024-04-11


# if you want to use a different version of curobo, create folder as docker/pkgs and put your
# version of curobo there. Then uncomment below line and comment the next line that clones from
# github

# COPY pkgs /pkgs

RUN mkdir /pkgs && cd /pkgs && git clone https://github.com/NVlabs/curobo.git

RUN $omni_python -m pip install ninja wheel tomli


RUN cd /pkgs/curobo && $omni_python -m pip install .[dev] --no-build-isolation

WORKDIR /pkgs/curobo

# Optionally install nvblox:

RUN apt-get update && \
    apt-get install -y curl tcl && \
    rm -rf /var/lib/apt/lists/*



# install gflags and glog statically, instructions from: https://github.com/nvidia-isaac/nvblox/blob/public/docs/redistributable.md


RUN cd /pkgs && wget https://cmake.org/files/v3.27/cmake-3.27.1.tar.gz && \
    tar -xvzf cmake-3.27.1.tar.gz && \
    apt update &&  apt install -y build-essential checkinstall zlib1g-dev libssl-dev && \
    cd cmake-3.27.1 && ./bootstrap && \
    make -j8 && \
    make install &&  rm -rf /var/lib/apt/lists/*


ENV USE_CX11_ABI=0
ENV PRE_CX11_ABI=ON



RUN cd /pkgs && git clone https://github.com/sqlite/sqlite.git -b version-3.39.4 && \
    cd /pkgs/sqlite && CFLAGS=-fPIC ./configure --prefix=/pkgs/sqlite/install/ && \
    make && make install



RUN cd /pkgs && git clone https://github.com/google/glog.git -b v0.6.0 && \
    cd glog && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=/pkgs/glog/install/ \
    -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    && make -j8 && make install


RUN cd /pkgs && git clone https://github.com/gflags/gflags.git -b v2.2.2 && \
    cd gflags &&  \
    mkdir build && cd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=/pkgs/gflags/install/ \
    -DGFLAGS_BUILD_STATIC_LIBS=ON -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    && make -j8 && make install

RUN cd /pkgs &&  git clone https://github.com/valtsblukis/nvblox.git && cd /pkgs/nvblox/nvblox && \
    mkdir build && cd build && \
    cmake ..  -DBUILD_REDISTRIBUTABLE=ON \
    -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI}  -DPRE_CXX11_ABI_LINKABLE=${PRE_CX11_ABI} \
    -DSQLITE3_BASE_PATH="/pkgs/sqlite/install/" -DGLOG_BASE_PATH="/pkgs/glog/install/" \
    -DGFLAGS_BASE_PATH="/pkgs/gflags/install/" -DCMAKE_CUDA_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    -DBUILD_TESTING=OFF && \
    make -j32 && \
    make install

# we also need libglog for pytorch:
RUN cd /pkgs/glog && \
    mkdir build_isaac && cd build_isaac && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    && make -j8 && make install

RUN cd /pkgs && git clone https://github.com/nvlabs/nvblox_torch.git && \
    cd /pkgs/nvblox_torch && \
    sh install_isaac_sim.sh $($omni_python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)') && \
    $omni_python -m pip install -e .

# install realsense for nvblox demos:
RUN $omni_python -m pip install pyrealsense2 opencv-python transforms3d

RUN $omni_python -m pip install "robometrics[evaluator] @ git+https://github.com/fishbotics/robometrics.git"

