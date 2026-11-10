# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cartesian reference interpolation and waypoint preservation."""

from __future__ import annotations

import pytest
import torch

from curobo._src.trajectory.trajectory_pose_reference import (
    interpolate_pose_waypoints,
    validate_pose_tolerance,
)
from curobo._src.trajectory.wp_pose_reference import sample_pose_reference
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.warp import init_warp


@pytest.mark.parametrize("count", range(1, 257))
def test_interpolation_preserves_every_waypoint_and_static_padding(count: int) -> None:
    """Retain every user target and keep each position segment straight."""
    torch.manual_seed(count)
    position = 0.1 * torch.randn((2, count, 2, 1, 3), dtype=torch.float32)
    quaternion = torch.randn((2, count, 2, 1, 4), dtype=torch.float32)
    quaternion /= quaternion.norm(dim=-1, keepdim=True)
    start = ToolPose(
        ["a", "b"],
        torch.zeros((2, 1, 2, 3), dtype=torch.float32),
        torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 1, 4).repeat(2, 1, 2, 1),
    )
    source = GoalToolPose(["a", "b"], position, quaternion)
    horizon = 81 if count <= 72 else 273
    intervals = horizon - 9
    reference = interpolate_pose_waypoints(source, start, horizon=horizon, padding=4)
    indices = torch.tensor([4 + round(intervals * i / count) for i in range(1, count + 1)])

    assert reference.position.shape == (2, horizon, 2, 1, 3)
    torch.testing.assert_close(reference.position[:, indices], position)
    similarity = (reference.quaternion[:, indices] * quaternion).sum(dim=-1).abs()
    torch.testing.assert_close(similarity, torch.ones_like(similarity))
    torch.testing.assert_close(reference.position[:, :5], torch.zeros((2, 5, 2, 1, 3)))
    torch.testing.assert_close(
        reference.position[:, -5:], position[:, -1:].expand(-1, 5, -1, -1, -1)
    )
    knots = torch.cat((torch.zeros(1, dtype=torch.long), indices - 4))
    source_position = torch.cat((start.position.unsqueeze(3), position), dim=1)
    for segment in range(count):
        left, right = int(knots[segment]), int(knots[segment + 1])
        fraction = torch.linspace(0, 1, right - left + 1).view(1, -1, 1, 1, 1)
        expected = torch.lerp(
            source_position[:, segment : segment + 1],
            source_position[:, segment + 1 : segment + 2],
            fraction,
        )
        torch.testing.assert_close(reference.position[:, 4 + left : 5 + right], expected)


def test_invalid_reference_quaternion_is_rejected() -> None:
    """A zero quaternion cannot silently become a reference rotation."""
    start = ToolPose(
        ["a"], torch.zeros((1, 1, 1, 3)), torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 1, 4)
    )
    waypoints = GoalToolPose(["a"], torch.zeros((1, 1, 1, 1, 3)), torch.zeros((1, 1, 1, 1, 4)))
    with pytest.raises(ValueError, match="quaternion"):
        interpolate_pose_waypoints(waypoints, start, horizon=81, padding=4)


@pytest.mark.parametrize("count", [0, 257])
def test_waypoint_count_outside_supported_range_is_rejected(count: int) -> None:
    """The fixed-control API cannot silently drop excess targets."""
    start = ToolPose(
        ["a"], torch.zeros((1, 1, 1, 3)), torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 1, 4)
    )
    waypoints = GoalToolPose(
        ["a"], torch.zeros((1, count, 1, 1, 3)), torch.zeros((1, count, 1, 1, 4))
    )
    with pytest.raises(ValueError, match="1 to 256"):
        interpolate_pose_waypoints(waypoints, start, horizon=273, padding=4)


def test_waypoint_count_exceeding_native_grid_is_rejected() -> None:
    """A larger API limit cannot silently discard targets on the default grid."""
    start = ToolPose(
        ["a"], torch.zeros((1, 1, 1, 3)), torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 1, 4)
    )
    waypoints = GoalToolPose(
        ["a"],
        torch.zeros((1, 73, 1, 1, 3)),
        torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 1, 1, 4).repeat(1, 73, 1, 1, 1),
    )
    with pytest.raises(ValueError, match="supports 1 to 72"):
        interpolate_pose_waypoints(waypoints, start, horizon=81, padding=4)


@pytest.mark.parametrize(
    "tolerance",
    [
        (0.0, 0.1),
        (-0.01, 0.1),
        (0.02, 0.0),
        (0.02, -0.1),
        (float("nan"), 0.1),
        (0.02, float("inf")),
        (0.02, 3.2),
    ],
)
def test_invalid_pose_tolerances_are_rejected(tolerance: tuple[float, float]) -> None:
    """Tolerances are finite physical units with an unambiguous angular range."""
    with pytest.raises(ValueError, match="tolerance"):
        validate_pose_tolerance(tolerance)


@pytest.mark.parametrize("count", [1, 7, 32, 256])
def test_dense_reference_preserves_linear_segments_and_rotation_arcs(
    count: int,
    cuda_device_cfg: DeviceCfg,
) -> None:
    """Dense GPU sampling follows the original position segments and SLERP rotations."""
    init_warp()
    kwargs = cuda_device_cfg.as_torch_dict()
    horizon = 81 if count <= 72 else 273
    intervals = horizon - 9
    knots = torch.tensor(
        [round(intervals * i / count) / intervals for i in range(count + 1)], **kwargs
    )
    direction = torch.tensor(
        [[[0.1, 0.05, 0.0], [0.0, 0.1, -0.02]], [[0.05, 0.0, 0.1], [0.02, -0.05, 0.0]]],
        **kwargs,
    )
    angle_scale = torch.tensor([[0.6, 1.0], [1.8, 2.5]], **kwargs)
    angle = knots.square()[None, :, None] * angle_scale[:, None]
    quaternion = torch.zeros((2, count + 1, 2, 1, 4), **kwargs)
    quaternion[..., 0] = torch.cos(angle.unsqueeze(-1) / 2)
    quaternion[..., 3] = torch.sin(angle.unsqueeze(-1) / 2)
    quaternion[:, 1::2] *= -1
    position = (knots.square() * (3 - 2 * knots))[None, :, None, None] * direction[:, None]
    start = ToolPose(["a", "b"], position[:, :1], quaternion[:, :1, :, 0])
    waypoints = GoalToolPose(
        ["a", "b"],
        position[:, 1:].unsqueeze(3),
        quaternion[:, 1:],
    )
    reference = interpolate_pose_waypoints(waypoints, start, horizon=horizon, padding=4)
    progress = torch.linspace(-0.1, 1.1, 257, **kwargs)[None].repeat(2, 1)
    output_position = torch.empty((2, 257, 2, 1, 3), **kwargs)
    output_quaternion = torch.empty((2, 257, 2, 1, 4), **kwargs)
    routes = torch.tensor([[1], [0]], device=cuda_device_cfg.device, dtype=torch.int32)
    sample_pose_reference(
        reference.position,
        reference.quaternion,
        progress,
        routes,
        output_position,
        output_quaternion,
        4,
    )
    query = progress[0].clamp(0, 1)
    segment = torch.searchsorted(knots, query).sub(1).clamp(0, count - 1)
    fraction = (query - knots[segment]) / (knots[segment + 1] - knots[segment])
    expected_position = torch.lerp(
        position.flip(0)[:, segment],
        position.flip(0)[:, segment + 1],
        fraction[None, :, None, None],
    )
    torch.testing.assert_close(
        output_position.squeeze(3),
        expected_position,
        atol=1e-6,
        rtol=1e-5,
    )
    expected_angle = (
        (1 - fraction) * knots[segment].square() + fraction * knots[segment + 1].square()
    )[None, :, None] * angle_scale.flip(0)[:, None]
    similarity = (
        output_quaternion[..., 0, 0] * torch.cos(expected_angle / 2)
        + output_quaternion[..., 0, 3] * torch.sin(expected_angle / 2)
    ).abs()
    torch.testing.assert_close(similarity, torch.ones_like(similarity), atol=1e-6, rtol=0)


def test_dense_reference_nonfinite_progress_stays_in_bounds(cuda_device_cfg: DeviceCfg) -> None:
    """Invalid sample progress cannot turn into an out-of-bounds reference read."""
    init_warp()
    kwargs = cuda_device_cfg.as_torch_dict()
    position = torch.zeros((1, 81, 1, 1, 3), **kwargs)
    quaternion = torch.zeros((1, 81, 1, 1, 4), **kwargs)
    quaternion[..., 0] = 1
    progress = torch.tensor([[float("nan"), float("inf"), -float("inf")]], **kwargs)
    routes = torch.zeros((1, 1), device=cuda_device_cfg.device, dtype=torch.int32)
    output_position = torch.empty((1, 3, 1, 1, 3), **kwargs)
    output_quaternion = torch.empty((1, 3, 1, 1, 4), **kwargs)
    sample_pose_reference(
        position,
        quaternion,
        progress,
        routes,
        output_position,
        output_quaternion,
        4,
    )
    assert torch.isfinite(output_position).all() and torch.isfinite(output_quaternion).all()
    assert not torch.isfinite(progress).any()
