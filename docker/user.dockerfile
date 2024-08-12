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
# RUN useradd -l -u 1000 $USERNAME
RUN useradd -l -u $USER_ID -g users $USERNAME

RUN /sbin/adduser $USERNAME sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


# Set user
# RUN mkdir /home/$USERNAME && chown $USERNAME:$USERNAME /home/$USERNAME
USER $USERNAME
WORKDIR /home/$USERNAME
ENV USER=$USERNAME
ENV PATH="${PATH}:/home/${USER}/.local/bin"


RUN echo 'completed'

