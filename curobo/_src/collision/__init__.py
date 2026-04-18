# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Collision checking module for robot-scene collision detection.

This module provides differentiable collision checking between robot and scene
for use in optimization-based motion planning.
"""

from curobo._src.collision.collision_robot_scene import RobotSceneCollision
from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg

__all__ = [
    "RobotSceneCollision",
    "RobotSceneCollisionCfg",
]

