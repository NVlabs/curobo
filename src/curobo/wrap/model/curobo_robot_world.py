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
"""This module has differentiable layers built from CuRobo's core features for use in Pytorch."""

# Standard Library
from dataclasses import dataclass

# CuRobo
from curobo.util.logger import log_warn
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


@dataclass
class CuroboRobotWorldConfig(RobotWorldConfig):
    def __post_init__(self):
        log_warn("CuroboRobotWorldConfig is deprecated, use RobotWorldConfig instead")
        return super().__post_init__()


class CuroboRobotWorld(RobotWorld):
    def __init__(self, config: CuroboRobotWorldConfig):
        log_warn("CuroboRobotWorld is deprecated, use RobotWorld instead")
        return super().__init__(config)
