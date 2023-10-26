#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
from abc import ABC, abstractmethod


class DynamicsModelBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, start_state, act_seq, *args):
        pass

    @abstractmethod
    def get_next_state(self, currend_state, act, dt):
        pass

    @abstractmethod
    def filter_robot_state(self, current_state):
        pass

    @abstractmethod
    def get_robot_command(self, current_state, act_seq):
        pass
