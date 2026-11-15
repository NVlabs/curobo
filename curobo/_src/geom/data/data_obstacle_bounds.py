# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor helpers for cached obstacle bounds."""

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.types.pose import Pose


@wp.func
def is_swept_sphere_outside_box(
    current: wp.vec3,
    previous: wp.vec3,
    following: wp.vec3,
    center: wp.vec3,
    half_extents: wp.vec3,
    radius_adjusted: wp.float32,
) -> wp.bool:
    """Reject both sphere half-sweeps when separated from a local-frame box."""
    previous_midpoint = 0.5 * (current + previous)
    following_midpoint = 0.5 * (current + following)
    lower = wp.min(current, wp.min(previous_midpoint, following_midpoint)) - center
    upper = wp.max(current, wp.max(previous_midpoint, following_midpoint)) - center
    bound_scale = wp.max(
        1.0,
        wp.max(
            wp.max(wp.abs(center[0]), wp.max(wp.abs(center[1]), wp.abs(center[2]))),
            wp.max(
                wp.abs(half_extents[0]), wp.max(wp.abs(half_extents[1]), wp.abs(half_extents[2]))
            ),
        ),
    )
    padding = radius_adjusted + 1.0e-6 * bound_scale
    return (
        lower[0] > half_extents[0] + padding
        or upper[0] < -half_extents[0] - padding
        or lower[1] > half_extents[1] + padding
        or upper[1] < -half_extents[1] - padding
        or lower[2] > half_extents[2] + padding
        or upper[2] < -half_extents[2] - padding
    )


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
