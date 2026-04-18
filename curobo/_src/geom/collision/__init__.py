# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Collision checking module for sphere-obstacle queries.

This module provides GPU-accelerated collision checking using Warp kernels.

Components:
- CollisionBuffer: Unified buffer for collision cost and gradient storage
- CollisionChecker: Low-level collision checking interface
- SceneCollision: High-level interface bundling data and operations
- SceneCollisionCfg: Configuration for SceneCollision
- sphere_obstacle_collision_kernel: Generic kernel for single-timestep collision
- swept_sphere_obstacle_collision_kernel: Generic kernel for trajectory collision
- SphereObstacleCollision: PyTorch autograd wrapper for collision
- SweptSphereObstacleCollision: PyTorch autograd wrapper for swept collision
"""

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.checker_collision import CollisionChecker
from curobo._src.geom.collision.collision_scene import (
    SceneCollision,
    SceneCollisionCfg,
    create_scene_collision,
)

# Backwards compatibility alias
create_collision_checker = create_scene_collision
from curobo._src.geom.collision.wp_autograd import (
    SphereObstacleCollision,
    SweptSphereObstacleCollision,
)

#from curobo._src.geom.collision.wp_collision_kernels import (
#    sphere_obstacle_collision_kernel,
#    swept_sphere_obstacle_collision_kernel,
#)
#from curobo._src.geom.collision.wp_speed_metric import apply_speed_metric

__all__ = [
    # Buffer
    "CollisionBuffer",
    # Checker
    "CollisionChecker",
    # Scene
    "SceneCollision",
    "SceneCollisionCfg",
    "create_scene_collision",
    "create_collision_checker",  # Backwards compatibility alias
    # Kernels
    #"apply_speed_metric",
    #"sphere_obstacle_collision_kernel",
    #"swept_sphere_obstacle_collision_kernel",
    # Autograd
    "SphereObstacleCollision",
    "SweptSphereObstacleCollision",
]
