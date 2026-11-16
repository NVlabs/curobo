# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cartesian reference construction outside the captured robot optimization."""

from __future__ import annotations

import math
from typing import Tuple

import torch

from curobo._src.types.spline_layout import reference_sample_indices
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise


def validate_pose_tolerance(tolerance: Tuple[float, float], *, context: str = "pose") -> None:
    """Require positive position meters and an orientation angle in ``(0, pi]``."""
    if len(tolerance) != 2:
        log_and_raise(f"{context} tolerance must contain position meters and orientation radians")
    if not math.isfinite(tolerance[0]) or tolerance[0] <= 0:
        log_and_raise(f"{context} position tolerance must be finite and positive")
    if not math.isfinite(tolerance[1]) or not 0 < tolerance[1] <= math.pi:
        log_and_raise(f"{context} orientation tolerance must be finite and in (0, pi]")


def pose_errors(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    target_position: torch.Tensor,
    target_quaternion: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return unweighted Euclidean meters and shortest rotation radians, broadcasting batches."""
    current = quaternion / quaternion.norm(dim=-1, keepdim=True)
    target = target_quaternion / target_quaternion.norm(dim=-1, keepdim=True)
    current_vector, target_vector = torch.broadcast_tensors(current[..., 1:], target[..., 1:])
    vector = (
        target[..., :1] * current[..., 1:]
        - current[..., :1] * target[..., 1:]
        - torch.linalg.cross(current_vector, target_vector, dim=-1)
    )
    scalar = (current * target).sum(dim=-1).abs()
    return (position - target_position).norm(dim=-1), 2 * torch.atan2(vector.norm(dim=-1), scalar)


def _slerp(first: torch.Tensor, second: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
    """Interpolate normalized wxyz rotations along the shortest arc."""
    dot = (first * second).sum(dim=-1, keepdim=True)
    second = torch.where(dot < 0, -second, second)
    dot = dot.abs().clamp(max=1.0)
    angle = torch.acos(dot.clamp(max=1.0 - 1e-7))
    spherical = (
        torch.sin((1 - fraction) * angle) * first + torch.sin(fraction * angle) * second
    ) / torch.sin(angle)
    linear = (1 - fraction) * first + fraction * second
    result = torch.where(dot > 1.0 - 1e-7, linear, spherical)
    return result / result.norm(dim=-1, keepdim=True)


@torch.no_grad()
def interpolate_pose_waypoints(
    waypoints: GoalToolPose, start: ToolPose, *, horizon: int, padding: int
) -> GoalToolPose:
    """Resample 1..256 targets to ``[B,horizon,L,1,3/4]``, retaining every target.

    The start is supplied separately. Padding and both active endpoints are static.
    Translation is piecewise linear; orientation uses sign-consistent SLERP.
    The target count must fit the active native reference intervals.
    """
    if not 1 <= waypoints.horizon <= 256 or waypoints.num_goalset != 1:
        log_and_raise("waypoints require 1 to 256 ordered targets and goalset size one")
    if start.horizon != 1 or start.tool_frames != waypoints.tool_frames:
        log_and_raise("waypoint frames must match the single current FK pose")
    if (
        waypoints.position.shape[-1] != 3
        or waypoints.quaternion.device != waypoints.position.device
    ):
        log_and_raise("waypoint positions must end in xyz and quaternion devices must match")
    if start.batch_size != waypoints.batch_size or start.position.device != waypoints.device:
        log_and_raise("waypoints and current FK must have matching batch and device")
    if waypoints.position.dtype != torch.float32 or waypoints.quaternion.dtype != torch.float32:
        log_and_raise("Cartesian waypoints require float32 position and quaternion tensors")
    if waypoints.quaternion.shape != (*waypoints.position.shape[:-1], 4):
        log_and_raise("waypoint position and quaternion shapes must match")
    if not bool(torch.isfinite(waypoints.position).all()):
        log_and_raise("waypoint positions must be finite")
    quaternion = torch.cat((start.quaternion.unsqueeze(3), waypoints.quaternion), dim=1)
    norm = quaternion.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(quaternion).all() & (norm > 0).all()):
        log_and_raise("waypoint quaternions must be finite and nonzero")
    quaternion = quaternion / norm
    signs = torch.where(
        (quaternion[:, 1:] * quaternion[:, :-1]).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0
    ).cumprod(dim=1)
    quaternion = torch.cat((quaternion[:, :1], quaternion[:, 1:] * signs), dim=1)
    position = torch.cat((start.position.unsqueeze(3), waypoints.position), dim=1)
    if not bool(torch.isfinite(position).all()):
        log_and_raise("current and waypoint positions must be finite")
    active = horizon - 2 * padding
    indices = torch.tensor(
        [0]
        + [
            index - padding
            for index in reference_sample_indices(horizon, padding, waypoints.horizon)
        ],
        device=waypoints.device,
        dtype=torch.long,
    )
    x = indices.to(dtype=position.dtype)
    query = torch.arange(active, device=waypoints.device, dtype=position.dtype)
    segment = torch.searchsorted(x, query).sub(1).clamp(0, x.numel() - 2)
    width = x[segment + 1] - x[segment]
    t = ((query - x[segment]) / width).view(1, active, 1, 1, 1)
    interpolated = (1 - t) * position[:, segment] + t * position[:, segment + 1]
    rotation = _slerp(quaternion[:, segment], quaternion[:, segment + 1], t)
    # Exact copies avoid accumulating interpolation roundoff at requested targets.
    interpolated[:, indices] = position
    rotation[:, indices] = quaternion
    return GoalToolPose(
        tool_frames=waypoints.tool_frames.copy(),
        position=torch.cat(
            (
                position[:, :1].expand(-1, padding, -1, -1, -1),
                interpolated,
                position[:, -1:].expand(-1, padding, -1, -1, -1),
            ),
            dim=1,
        ).contiguous(),
        quaternion=torch.cat(
            (
                quaternion[:, :1].expand(-1, padding, -1, -1, -1),
                rotation,
                quaternion[:, -1:].expand(-1, padding, -1, -1, -1),
            ),
            dim=1,
        ).contiguous(),
    )
