# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Robot type definitions."""

from curobo._src.robot.types.collision_geometry import RobotCollisionGeometry
from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.robot.types.joint_types import JointType
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.robot.types.link_params import LinkParams
from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg

__all__ = [
    "JointType",
    "JointLimits",
    "LinkParams",
    "CSpaceParams",
    "KinematicsParams",
    "SelfCollisionKinematicsCfg",
    "RobotCollisionGeometry",
]

