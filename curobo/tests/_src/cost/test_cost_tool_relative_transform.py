# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scalar and endpoint-gradient tests for relative tool-transform cost."""

# Standard Library
import math

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_tool_relative_transform import ToolRelativeTransformCost
from curobo._src.cost.cost_tool_relative_transform_cfg import ToolRelativeTransformCostCfg
from curobo._src.cost.criteria_tool_relative_transform import ToolRelativeTransformCriteria
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util.warp import init_warp


@pytest.fixture(scope="module", autouse=True)
def setup_warp() -> None:
    """Initialize Warp before compiling the relative-cost CPU kernel."""
    init_warp(quiet=True)


def _axis_angle_quaternion_wxyz(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis)
    half_angle = 0.5 * angle
    return torch.cat(
        (
            torch.tensor([math.cos(half_angle)], dtype=torch.float32),
            axis * math.sin(half_angle),
        )
    )


def _quaternion_multiply_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    scalar = lhs[:1] * rhs[:1] - torch.dot(lhs[1:], rhs[1:])
    vector = lhs[:1] * rhs[1:] + rhs[:1] * lhs[1:] + torch.linalg.cross(lhs[1:], rhs[1:])
    return torch.cat((scalar, vector))


def _left_perturb_quaternion(quaternion: torch.Tensor, axis: int, step: float) -> torch.Tensor:
    perturbation_axis = torch.zeros(3, dtype=quaternion.dtype)
    perturbation_axis[axis] = 1.0
    delta_quaternion = _axis_angle_quaternion_wxyz(perturbation_axis, step)
    return _quaternion_multiply_wxyz(delta_quaternion, quaternion)


def _make_cost(
    criteria: ToolRelativeTransformCriteria,
    *,
    tool_frames: list[str] | None = None,
) -> ToolRelativeTransformCost:
    frames = ["reference", "tool"] if tool_frames is None else tool_frames
    device_cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
    cost = ToolRelativeTransformCost(
        ToolRelativeTransformCostCfg(
            weight=[2.0, 3.0],
            reference_tool_frame="reference",
            tool_frames=frames,
            tool_relative_transform_criteria={"tool": criteria},
            device_cfg=device_cfg,
        )
    )
    cost.setup_batch_tensors(batch_size=1, horizon=1)
    return cost


def _evaluate_sum(
    cost: ToolRelativeTransformCost,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    target: ToolRelativeTransformTarget,
) -> torch.Tensor:
    values, _, _ = cost.forward(
        current_tool_poses=ToolPose(
            tool_frames=cost.tool_frames,
            position=position,
            quaternion=quaternion,
        ),
        tool_relative_transform_target=target,
        idxs_tool_relative_transform_target=torch.zeros((1, 1), dtype=torch.int32),
    )
    return values.sum()


def test_relative_cost_matches_simple_scalar_value_and_distances() -> None:
    """The cost reports one relation and leaves the reference slot at zero."""
    device_cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
    criteria = ToolRelativeTransformCriteria(
        position_axis_scale=[2.0, 1.0, 1.0],
        rotation_axis_scale=[1.0, 1.0, 2.0],
        device_cfg=device_cfg,
    )
    cost = _make_cost(criteria)
    position = torch.tensor([[[[1.0, 0.0, 0.0], [3.0, 1.0, 0.0]]]])
    quaternion = torch.stack(
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _axis_angle_quaternion_wxyz(torch.tensor([0.0, 0.0, 1.0]), 0.2),
        )
    ).reshape(1, 1, 2, 4)
    target = ToolRelativeTransformTarget(
        reference_tool_frame="reference",
        tool_frames=["reference", "tool"],
        position=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]),
        quaternion=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
    )

    values, position_distance, rotation_distance = cost.forward(
        current_tool_poses=ToolPose(
            tool_frames=["reference", "tool"],
            position=position,
            quaternion=quaternion,
        ),
        tool_relative_transform_target=target,
        idxs_tool_relative_transform_target=torch.zeros((1, 1), dtype=torch.int32),
    )

    expected_rotation_distance = 4.0 * math.sin(0.1)
    expected_cost = 4.0 + 3.0 * expected_rotation_distance**2
    torch.testing.assert_close(
        values[0, 0], torch.tensor([0.0, expected_cost]), atol=2.0e-5, rtol=0.0
    )
    torch.testing.assert_close(position_distance[0, 0], torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(
        rotation_distance[0, 0],
        torch.tensor([0.0, expected_rotation_distance]),
        atol=2.0e-6,
        rtol=0.0,
    )


def test_relative_cost_routes_target_groups() -> None:
    """Each batch lane selects its complete aligned relative-target group."""
    device_cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
    cost = _make_cost(ToolRelativeTransformCriteria.track_position(device_cfg=device_cfg))
    cost.setup_batch_tensors(batch_size=2, horizon=1)
    current_position = torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        ]
    )
    current_quaternion = torch.tensor(
        [
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        ]
    )
    target = ToolRelativeTransformTarget(
        reference_tool_frame="reference",
        tool_frames=["reference", "tool"],
        position=torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ]
        ),
        quaternion=torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            ]
        ),
    )

    values, _, _ = cost.forward(
        current_tool_poses=ToolPose(
            tool_frames=["reference", "tool"],
            position=current_position,
            quaternion=current_quaternion,
        ),
        tool_relative_transform_target=target,
        idxs_tool_relative_transform_target=torch.tensor([[0], [1]], dtype=torch.int32),
    )

    torch.testing.assert_close(values[:, 0, 1], torch.tensor([0.0, 4.0]))


def test_relative_endpoint_gradients_match_independent_finite_differences() -> None:
    """Both endpoint translations and world-left rotations match finite differences."""
    device_cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
    direction_position = torch.tensor([1.0, -2.0, 0.7])
    direction_position /= torch.linalg.vector_norm(direction_position)
    direction_rotation = torch.tensor([-0.3, 1.1, 0.8])
    direction_rotation /= torch.linalg.vector_norm(direction_rotation)
    criteria = ToolRelativeTransformCriteria(
        position_projector=torch.eye(3) - torch.outer(direction_position, direction_position),
        rotation_projector=torch.eye(3) - torch.outer(direction_rotation, direction_rotation),
        position_axis_scale=[1.7, 0.6, 1.2],
        rotation_axis_scale=[0.7, 1.4, 1.1],
        device_cfg=device_cfg,
    )
    cost = _make_cost(criteria)
    position = torch.tensor([[[[0.3, -0.4, 0.2], [1.0, 0.7, -0.5]]]], requires_grad=True)
    quaternion = torch.stack(
        (
            _axis_angle_quaternion_wxyz(torch.tensor([0.4, -0.2, 0.9]), 0.8),
            _axis_angle_quaternion_wxyz(torch.tensor([-0.5, 0.7, 0.3]), -0.6),
        )
    ).reshape(1, 1, 2, 4)
    quaternion.requires_grad_(True)
    target = ToolRelativeTransformTarget(
        reference_tool_frame="reference",
        tool_frames=["reference", "tool"],
        position=torch.tensor([[[0.0, 0.0, 0.0], [0.4, -0.1, 0.8]]]),
        quaternion=torch.stack(
            (
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                _axis_angle_quaternion_wxyz(torch.tensor([0.2, 0.8, -0.4]), 0.5),
            )
        ).reshape(1, 2, 4),
    )

    value = _evaluate_sum(cost, position, quaternion, target)
    value.backward()
    analytic_position = position.grad.detach().clone()[0, 0]
    analytic_quaternion = quaternion.grad.detach().clone()[0, 0]

    position_step = 2.0e-3
    for link_index in range(2):
        for axis in range(3):
            plus = position.detach().clone()
            minus = position.detach().clone()
            plus[0, 0, link_index, axis] += position_step
            minus[0, 0, link_index, axis] -= position_step
            finite_difference = (
                _evaluate_sum(cost, plus, quaternion.detach(), target)
                - _evaluate_sum(cost, minus, quaternion.detach(), target)
            ) / (2.0 * position_step)
            torch.testing.assert_close(
                analytic_position[link_index, axis],
                finite_difference,
                atol=2.0e-3,
                rtol=2.0e-3,
            )

    rotation_step = 2.0e-3
    for link_index in range(2):
        for axis in range(3):
            plus = quaternion.detach().clone()
            minus = quaternion.detach().clone()
            plus[0, 0, link_index] = _left_perturb_quaternion(
                plus[0, 0, link_index], axis, rotation_step
            )
            minus[0, 0, link_index] = _left_perturb_quaternion(
                minus[0, 0, link_index], axis, -rotation_step
            )
            finite_difference = (
                _evaluate_sum(cost, position.detach(), plus, target)
                - _evaluate_sum(cost, position.detach(), minus, target)
            ) / (2.0 * rotation_step)

            axis_vector = torch.zeros(3)
            axis_vector[axis] = 1.0
            delta_quaternion = torch.cat(
                (
                    -0.5 * torch.dot(axis_vector, quaternion.detach()[0, 0, link_index, 1:])[None],
                    0.5
                    * (
                        quaternion.detach()[0, 0, link_index, 0] * axis_vector
                        + torch.linalg.cross(
                            axis_vector, quaternion.detach()[0, 0, link_index, 1:]
                        )
                    ),
                )
            )
            analytic_directional = torch.dot(analytic_quaternion[link_index], delta_quaternion)
            torch.testing.assert_close(
                analytic_directional,
                finite_difference,
                atol=3.0e-3,
                rtol=3.0e-3,
            )


def test_relative_cost_rejects_mismatched_target_metadata() -> None:
    """Runtime target metadata must match the cost's fixed topology."""
    device_cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
    cost = _make_cost(ToolRelativeTransformCriteria.rigid(device_cfg=device_cfg))
    pose = ToolPose(
        tool_frames=["reference", "tool"],
        position=torch.zeros((1, 1, 2, 3)),
        quaternion=torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]]),
    )
    target = ToolRelativeTransformTarget(
        reference_tool_frame="tool",
        tool_frames=["reference", "tool"],
        position=torch.zeros((1, 2, 3)),
        quaternion=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
    )

    with pytest.raises(ValueError, match="reference_tool_frame"):
        cost.forward(
            current_tool_poses=pose,
            tool_relative_transform_target=target,
            idxs_tool_relative_transform_target=torch.zeros((1, 1), dtype=torch.int32),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_relative_cost_cuda_kernel_smoke() -> None:
    """Compile and evaluate the conventional relation kernel on CUDA."""
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    cost = ToolRelativeTransformCost(
        ToolRelativeTransformCostCfg(
            weight=[2.0, 0.0],
            reference_tool_frame="reference",
            tool_frames=["reference", "tool"],
            tool_relative_transform_criteria={
                "tool": ToolRelativeTransformCriteria.track_position(device_cfg=device_cfg)
            },
            device_cfg=device_cfg,
        )
    )
    cost.setup_batch_tensors(batch_size=1, horizon=1)
    pose = ToolPose(
        tool_frames=["reference", "tool"],
        position=torch.tensor([[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]], device=device_cfg.device),
        quaternion=torch.tensor(
            [[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]],
            device=device_cfg.device,
        ),
    )
    target = ToolRelativeTransformTarget(
        reference_tool_frame="reference",
        tool_frames=["reference", "tool"],
        position=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], device=device_cfg.device),
        quaternion=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            device=device_cfg.device,
        ),
    )

    values, position_distance, rotation_distance = cost.forward(
        current_tool_poses=pose,
        tool_relative_transform_target=target,
        idxs_tool_relative_transform_target=torch.zeros(
            (1, 1), dtype=torch.int32, device=device_cfg.device
        ),
    )

    torch.testing.assert_close(values.cpu(), torch.tensor([[[0.0, 1.0]]]))
    torch.testing.assert_close(position_distance.cpu(), torch.tensor([[[0.0, 1.0]]]))
    torch.testing.assert_close(rotation_distance.cpu(), torch.zeros((1, 1, 2)))
