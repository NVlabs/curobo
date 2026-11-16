# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end scalar tests for the projected absolute tool-pose cost kernel."""

# Standard Library
import math

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.cost.wp_tool_pose import (
    ToolPoseDistance,
    create_goalset_pose_distance_kernel_with_constants,
)
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.warp import init_warp
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module", autouse=True)
def setup_warp():
    """Initialize Warp before compiling the CPU test kernels."""
    init_warp(quiet=True)


def _axis_angle_quaternion_wxyz(axis, angle):
    axis_tensor = torch.as_tensor(axis, dtype=torch.float32)
    axis_tensor = axis_tensor / torch.linalg.vector_norm(axis_tensor)
    half_angle = 0.5 * angle
    return torch.cat(
        (
            torch.tensor([math.cos(half_angle)], dtype=torch.float32),
            axis_tensor * math.sin(half_angle),
        )
    )


def _quaternion_multiply_wxyz(lhs, rhs):
    scalar = lhs[..., :1] * rhs[..., :1] - torch.sum(
        lhs[..., 1:] * rhs[..., 1:], dim=-1, keepdim=True
    )
    vector = (
        lhs[..., :1] * rhs[..., 1:]
        + rhs[..., :1] * lhs[..., 1:]
        + torch.linalg.cross(lhs[..., 1:], rhs[..., 1:], dim=-1)
    )
    return torch.cat((scalar, vector), dim=-1)


def _quaternion_inverse_wxyz(quaternion):
    conjugate = torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)
    return conjugate / torch.sum(quaternion * quaternion, dim=-1, keepdim=True)


def _rotation_error(current_quat, target_quat):
    error_quat = _quaternion_multiply_wxyz(current_quat, _quaternion_inverse_wxyz(target_quat))
    error_quat = error_quat / torch.linalg.vector_norm(error_quat, dim=-1, keepdim=True)
    error_quat = torch.where(error_quat[..., :1] < 0.0, -error_quat, error_quat)
    return 2.0 * error_quat[..., 1:]


def _rotation_matrix_wxyz(quaternion):
    quaternion = quaternion / torch.linalg.vector_norm(quaternion)
    w, x, y, z = quaternion.unbind()
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        )
    ).reshape(3, 3)


def _evaluate_pose_distance(
    current_position,
    current_quat,
    goal_position,
    goal_quat,
    weight,
    terminal_position_projector,
    terminal_rotation_projector,
    path_position_projector,
    path_rotation_projector,
    terminal_position_scale,
    terminal_rotation_scale,
    path_position_scale,
    path_rotation_scale,
    terminal_tolerance,
    path_tolerance,
    error_frame,
):
    batch_size, horizon, num_links, _ = current_position.shape
    num_goalset = goal_position.shape[-2]
    output_cost = torch.zeros((batch_size, horizon, 2 * num_links), dtype=torch.float32)
    output_position_distance = torch.zeros((batch_size, horizon, num_links), dtype=torch.float32)
    output_rotation_distance = torch.zeros((batch_size, horizon, num_links), dtype=torch.float32)
    output_position_gradient = torch.zeros(
        (batch_size, horizon, num_links, 3), dtype=torch.float32
    )
    output_rotation_gradient = torch.zeros(
        (batch_size, horizon, num_links, 4), dtype=torch.float32
    )
    output_goalset_index = torch.zeros((batch_size, horizon, num_links), dtype=torch.int32)
    position_projector = torch.cat(
        (
            terminal_position_projector,
            path_position_projector,
        )
    )
    rotation_projector = torch.cat(
        (
            terminal_rotation_projector,
            path_rotation_projector,
        )
    )
    position_scale = torch.cat((terminal_position_scale, path_position_scale))
    rotation_scale = torch.cat((terminal_rotation_scale, path_rotation_scale))
    scaled_position_projector = position_scale.unsqueeze(-1) * position_projector
    scaled_rotation_projector = rotation_scale.unsqueeze(-1) * rotation_projector
    position_metric = (
        scaled_position_projector.transpose(-1, -2) @ scaled_position_projector
    )
    rotation_metric = (
        scaled_rotation_projector.transpose(-1, -2) @ scaled_rotation_projector
    )
    pose_tolerance = torch.cat((terminal_tolerance, path_tolerance))
    return ToolPoseDistance.apply(
        current_position,
        current_quat,
        goal_position,
        goal_quat,
        torch.zeros((batch_size, 1), dtype=torch.int32),
        weight,
        position_metric,
        rotation_metric,
        pose_tolerance,
        error_frame,
        output_cost,
        output_position_distance,
        output_rotation_distance,
        output_position_gradient,
        output_rotation_gradient,
        output_goalset_index,
        True,
        create_goalset_pose_distance_kernel_with_constants(num_goalset),
    )


@pytest.mark.parametrize("error_frame", [0, 1])
def test_projected_pose_cost_matches_scalar_reference_and_gradients(error_frame):
    current_position = torch.tensor([[[[0.7, -0.2, 0.5]]]], requires_grad=True)
    # Deliberately scale the quaternion to verify that the raw-input gradient
    # matches the normalization performed by the metric.
    current_quat = 1.7 * _axis_angle_quaternion_wxyz([0.3, -0.7, 0.4], 1.1).reshape(1, 1, 1, 4)
    current_quat.requires_grad_(True)
    goal_position = torch.tensor([[[[-0.1, 0.4, 0.2]]]])
    goal_quat = _axis_angle_quaternion_wxyz([-0.5, 0.2, 0.8], -0.6).reshape(1, 1, 1, 4)
    direction = torch.tensor([1.0, -2.0, 0.6])
    direction = direction / torch.linalg.vector_norm(direction)
    position_projector = torch.eye(3) - torch.outer(direction, direction)
    rotation_projector = torch.eye(3) - torch.outer(direction.flip(0), direction.flip(0))
    position_scale = torch.tensor([1.8, 0.6, 1.3])
    rotation_scale = torch.tensor([0.7, 1.5, 1.1])
    weight = torch.tensor([1.4, 0.9])

    cost, position_distance, rotation_distance, _ = _evaluate_pose_distance(
        current_position,
        current_quat,
        goal_position,
        goal_quat,
        weight,
        position_projector[None],
        rotation_projector[None],
        position_projector[None],
        rotation_projector[None],
        position_scale[None],
        rotation_scale[None],
        position_scale[None],
        rotation_scale[None],
        torch.zeros((1, 2)),
        torch.zeros((1, 2)),
        torch.tensor([error_frame], dtype=torch.int32),
    )
    cost.sum().backward()

    reference_position = current_position.detach().clone().reshape(3).requires_grad_(True)
    reference_quat = current_quat.detach().clone().reshape(4).requires_grad_(True)
    goal_position_vector = goal_position.reshape(3)
    goal_quat_vector = goal_quat.reshape(4)
    if error_frame == 0:
        position_delta = reference_position - goal_position_vector
        rotation_error = _rotation_error(reference_quat, goal_quat_vector)
    else:
        goal_rotation = _rotation_matrix_wxyz(goal_quat_vector)
        position_delta = goal_rotation.T @ (reference_position - goal_position_vector)
        rotation_error = _rotation_error(
            _quaternion_multiply_wxyz(_quaternion_inverse_wxyz(goal_quat_vector), reference_quat),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )

    scaled_position = position_scale * (position_projector @ position_delta)
    scaled_rotation = rotation_scale * (rotation_projector @ rotation_error)
    reference_position_cost = 0.5 * weight[0] * torch.dot(scaled_position, scaled_position)
    reference_rotation_cost = weight[1] * torch.dot(scaled_rotation, scaled_rotation)
    (reference_position_cost + reference_rotation_cost).backward()

    torch.testing.assert_close(
        cost[0, 0], torch.stack((reference_position_cost, reference_rotation_cost))
    )
    torch.testing.assert_close(
        position_distance[0, 0, 0], torch.linalg.vector_norm(scaled_position)
    )
    torch.testing.assert_close(
        rotation_distance[0, 0, 0], torch.linalg.vector_norm(scaled_rotation)
    )
    torch.testing.assert_close(
        current_position.grad.reshape(3), reference_position.grad, atol=2.0e-5, rtol=2.0e-5
    )
    torch.testing.assert_close(
        current_quat.grad.reshape(4), reference_quat.grad, atol=3.0e-5, rtol=3.0e-5
    )


def test_path_terminal_selection_and_goalset_reduction():
    current_position = torch.tensor([[[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]])
    current_quat = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]])
    goal_position = torch.tensor([[[[2.0, 0.0, 0.0], [0.9, 0.0, 0.0]]]])
    goal_quat = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]])
    identity = torch.eye(3)[None]
    zero = torch.zeros((1, 3, 3))

    cost, position_distance, rotation_distance, goalset_index = _evaluate_pose_distance(
        current_position,
        current_quat,
        goal_position,
        goal_quat,
        torch.tensor([2.0, 0.0]),
        identity,
        zero,
        zero,
        zero,
        torch.ones((1, 3)),
        torch.ones((1, 3)),
        torch.ones((1, 3)),
        torch.ones((1, 3)),
        torch.zeros((1, 2)),
        torch.zeros((1, 2)),
        torch.zeros(1, dtype=torch.int32),
    )

    torch.testing.assert_close(cost[0, 0], torch.zeros(2))
    torch.testing.assert_close(cost[0, 1], torch.tensor([0.01, 0.0]))
    torch.testing.assert_close(position_distance[0, :, 0], torch.tensor([0.0, 0.1]))
    torch.testing.assert_close(rotation_distance, torch.zeros_like(rotation_distance))
    torch.testing.assert_close(goalset_index[0, :, 0], torch.tensor([0, 1], dtype=torch.int32))


def test_tool_pose_cost_passes_canonical_stacked_criteria_to_kernel():
    device_cfg = DeviceCfg(device=torch.device("cpu"))
    config = ToolPoseCostCfg(
        weight=[2.0, 0.0],
        tool_frames=["tool"],
        device_cfg=device_cfg,
    )
    config.tool_pose_criteria = {
        "tool": ToolPoseCriteria(
            terminal_position_projector=torch.eye(3),
            terminal_rotation_projector=torch.zeros((3, 3)),
            path_position_projector=torch.zeros((3, 3)),
            path_rotation_projector=torch.zeros((3, 3)),
            device_cfg=device_cfg,
        )
    }
    pose_cost = ToolPoseCost(config)
    pose_cost.setup_batch_tensors(batch_size=1, horizon=2)

    current_pose = ToolPose(
        tool_frames=["tool"],
        position=torch.tensor([[[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]]),
        quaternion=torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]]),
    )
    goal_pose = GoalToolPose(
        tool_frames=["tool"],
        position=torch.zeros((1, 1, 1, 1, 3)),
        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]]),
    )
    cost, position_distance, rotation_distance, goalset_index = pose_cost.forward(
        current_tool_poses=current_pose,
        goal_tool_poses=goal_pose,
        idxs_goal=torch.zeros((1, 1), dtype=torch.int32),
    )

    torch.testing.assert_close(cost[0, 0], torch.zeros(2))
    torch.testing.assert_close(cost[0, 1], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(position_distance[0, :, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(rotation_distance, torch.zeros_like(rotation_distance))
    torch.testing.assert_close(goalset_index, torch.zeros_like(goalset_index))


def test_projected_rotation_gradient_matches_joint_space_finite_difference():
    """Check the pose metric and FK quaternion pullback as one gradient chain."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for robot kinematics")

    device_cfg = DeviceCfg(device=torch.device("cuda:0"))
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    kinematics = Kinematics(
        KinematicsCfg.from_robot_yaml_file(robot_data, device_cfg=device_cfg),
        compute_jacobian=False,
    )
    joint_names = kinematics.joint_names
    start = device_cfg.to_device([[0.31, -0.82, 0.27, -1.91, 0.44, 1.16, -0.38]])
    goal = device_cfg.to_device([[-0.16, -0.41, 0.63, -1.42, -0.22, 1.53, 0.35]])
    goal_tool_poses = (
        kinematics.compute_kinematics(JointState.from_position(goal, joint_names=joint_names))
        .tool_poses.as_goal()
        .clone()
    )

    free_axis = device_cfg.to_device([0.3, -0.7, 0.4])
    free_axis = free_axis / torch.linalg.vector_norm(free_axis)
    rotation_projector = torch.eye(3, **device_cfg.as_torch_dict()) - torch.outer(
        free_axis, free_axis
    )
    zero_projector = torch.zeros((3, 3), **device_cfg.as_torch_dict())
    criteria = {
        frame: ToolPoseCriteria(
            terminal_position_projector=zero_projector,
            terminal_rotation_projector=rotation_projector,
            terminal_rotation_axis_scale=[0.7, 1.4, 2.0],
            device_cfg=device_cfg,
        )
        for frame in kinematics.tool_frames
    }
    pose_cost = ToolPoseCost(
        ToolPoseCostCfg(
            weight=[0.0, 1.3],
            tool_frames=kinematics.tool_frames,
            tool_pose_criteria=criteria,
            device_cfg=device_cfg,
        )
    )
    pose_cost.setup_batch_tensors(batch_size=1, horizon=1)
    goal_index = torch.zeros((1, 1), device=device_cfg.device, dtype=torch.int32)

    def evaluate(joint_position: torch.Tensor) -> torch.Tensor:
        current_tool_poses = kinematics.compute_kinematics(
            JointState.from_position(joint_position, joint_names=joint_names)
        ).tool_poses
        cost, _, _, _ = pose_cost.forward(
            current_tool_poses=current_tool_poses,
            goal_tool_poses=goal_tool_poses,
            idxs_goal=goal_index,
        )
        return cost.sum()

    differentiable_start = start.clone().requires_grad_(True)
    evaluate(differentiable_start).backward()
    analytical_gradient = differentiable_start.grad.detach().clone()

    epsilon = 5.0e-4
    finite_difference_gradient = torch.zeros_like(start)
    for joint_index in range(kinematics.get_dof()):
        plus = start.clone()
        minus = start.clone()
        plus[0, joint_index] += epsilon
        minus[0, joint_index] -= epsilon
        plus_cost = evaluate(plus).detach().clone()
        minus_cost = evaluate(minus).detach().clone()
        finite_difference_gradient[0, joint_index] = (plus_cost - minus_cost) / (2.0 * epsilon)

    torch.testing.assert_close(
        analytical_gradient,
        finite_difference_gradient,
        atol=3.0e-3,
        rtol=3.0e-2,
    )
