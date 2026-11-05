# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor helpers for cached obstacle bounds."""

# Third Party
import torch

# CuRobo
from curobo._src.types.pose import Pose


def update_world_aabb(
    output: torch.Tensor,
    local_lower: torch.Tensor,
    local_upper: torch.Tensor,
    world_obstacle_pose: Pose,
) -> None:
    """Write world-AABB center and extent from local bounds and obstacle poses.

    Args:
        output: Destination tensor with shape ``(..., 8)``.
        local_lower: Local AABB lower corners with shape ``(..., 3)``.
        local_upper: Local AABB upper corners with shape ``(..., 3)``.
        world_obstacle_pose: Obstacle poses in the world frame.
    """
    local_center = 0.5 * (local_lower + local_upper)
    local_extent = 0.5 * (local_upper - local_lower)
    rotation = world_obstacle_pose.get_rotation()
    world_center = (
        torch.matmul(rotation, local_center.unsqueeze(-1)).squeeze(-1)
        + world_obstacle_pose.position
    )
    world_extent = torch.matmul(torch.abs(rotation), local_extent.unsqueeze(-1)).squeeze(-1)
    target_shape = output.shape[:-1] + (3,)
    output[..., :3].copy_(world_center.reshape(target_shape))
    output[..., 3] = 0.0
    output[..., 4:7].copy_(world_extent.reshape(target_shape))
    output[..., 7] = 0.0
