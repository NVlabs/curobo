# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot-scene collision checking module.

This module provides robot-scene collision checking for custom pipelines.
Most users should use InverseKinematics or TrajectoryOptimizer which handle
collision checking internally.

Use this module when building custom collision-aware pipelines outside the
main solvers.

Example:
    ```python
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo.scene import Scene
    import torch

    # Create collision checker for custom pipeline
    config = RobotCollisionCheckerCfg.load_from_config(
        robot_config="franka.yml",
        scene_model="scene.yml",
        collision_activation_distance=0.01,
    )
    checker = RobotCollisionChecker(config)

    # Query collision at joint configurations
    joint_positions = torch.rand((10, 7))  # 10 configurations
    scene_distance, self_distance = checker.get_scene_self_collision_distance_from_joints(
        joint_positions
    )

    # Sample collision-free configurations
    valid_configs = checker.sample(num_samples=100, mask_valid=True)

    # Validate configurations
    is_valid = checker.validate(joint_positions)

    # Use in custom optimization (differentiable)
    joint_positions.requires_grad = True
    scene_dist, self_dist = checker.get_scene_self_collision_distance_from_joints(
        joint_positions
    )
    cost = scene_dist.sum() + self_dist.sum()
    cost.backward()  # Gradients flow through
    ```
"""

from curobo._src.collision.collision_robot_scene import (
    RobotSceneCollision as RobotCollisionChecker,
)
from curobo._src.collision.collision_robot_scene_cfg import (
    RobotSceneCollisionCfg as RobotCollisionCheckerCfg,
)

__all__ = [
    "RobotCollisionChecker",
    "RobotCollisionCheckerCfg",
]
