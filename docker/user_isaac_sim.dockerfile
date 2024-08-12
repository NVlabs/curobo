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

# Check architecture and load:
ARG IMAGE_TAG
FROM curobo_docker:${IMAGE_TAG}
# Set variables
ARG USERNAME
ARG USER_ID
ARG CACHE_DATE=2024-07-19


# Set environment variables

# Set up sudo user
#RUN /sbin/adduser --disabled-password --gecos '' --uid $USER_ID $USERNAME
RUN useradd -l -u $USER_ID -g users $USERNAME

RUN /sbin/adduser $USERNAME sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN usermod -aG root $USERNAME


# change ownership of isaac sim folder if it exists:
RUN mkdir /isaac-sim/kit/cache && chown -R $USERNAME:users /isaac-sim/kit/cache
RUN chown $USERNAME:users /root && chown $USERNAME:users /isaac-sim
RUN mkdir /root/.nv && chown -R $USERNAME:users /root/.nv
RUN chown -R $USERNAME:users /root/.cache

# change permission for some exts:
RUN mkdir -p /isaac-sim/kit/logs/Kit/Isaac-Sim && chown -R $USERNAME:users /isaac-sim/kit/logs/Kit/Isaac-Sim

#RUN chown -R $USERNAME:users /root/.cache/pip
#RUN chown -R $USERNAME:users /root/.cache/nvidia/GLCache
#RUN chown -R $USERNAME:users /root/.local/share/ov
RUN mkdir /root/.nvidia-omniverse/logs && mkdir -p /home/$USERNAME/.nvidia-omniverse && cp -r /root/.nvidia-omniverse/* /home/$USERNAME/.nvidia-omniverse && chown -R $USERNAME:users /home/$USERNAME/.nvidia-omniverse
RUN chown -R $USERNAME:users /isaac-sim/exts/omni.isaac.synthetic_recorder/
RUN chown -R $USERNAME:users /isaac-sim/kit/exts/omni.gpu_foundation
RUN mkdir -p /home/$USERNAME/.cache && cp -r /root/.cache/* /home/$USERNAME/.cache && chown -R $USERNAME:users /home/$USERNAME/.cache
RUN mkdir -p /isaac-sim/kit/data/documents/Kit && mkdir -p /isaac-sim/kit/data/documents/Kit/apps/Isaac-Sim/scripts/ &&chown -R $USERNAME:users /isaac-sim/kit/data/documents/Kit /isaac-sim/kit/data/documents/Kit/apps/Isaac-Sim/scripts/
RUN mkdir -p /home/$USERNAME/.local


RUN echo "alias omni_python='/isaac-sim/python.sh'" >> /home/$USERNAME/.bashrc
RUN echo "alias python='/isaac-sim/python.sh'" >> /home/$USERNAME/.bashrc

RUN chown -R $USERNAME:users /home/$USERNAME
# /isaac-sim/kit/data
# /isaac-sim/kit/logs/Kit

# Set user
USER $USERNAME
WORKDIR /home/$USERNAME
ENV USER=$USERNAME
ENV PATH="${PATH}:/home/${USER}/.local/bin"
ENV SHELL /bin/bash
ENV OMNI_USER=admin
ENV OMNI_PASS=admin


RUN mkdir /root/Documents && chown -R $USERNAME:users /root/Documents

RUN echo 'completed'

