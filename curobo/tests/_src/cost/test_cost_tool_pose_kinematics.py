# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for tool-pose gradients through kinematics."""

import torch

from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose


def test_orientation_gradient_matches_finite_difference(cuda_device_cfg) -> None:
    """Check the complete orientation-cost pullback to Franka joint space."""
    config = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=cuda_device_cfg)
    kinematics = Kinematics(config, compute_spheres=False, compute_com=False)
    default = kinematics.default_joint_position.unsqueeze(0)
    target = default + cuda_device_cfg.to_device([[0.10, -0.12, 0.08, -0.15, 0.07, 0.11, -0.09]])
    position = target + cuda_device_cfg.to_device([[-0.20, 0.18, -0.14, 0.21, -0.17, 0.13, 0.16]])

    target_state = kinematics.compute_kinematics(JointState.from_position(target))
    goal = GoalToolPose(
        tool_frames=kinematics.tool_frames,
        position=target_state.tool_poses.position.detach().clone().unsqueeze(3),
        quaternion=target_state.tool_poses.quaternion.detach().clone().unsqueeze(3),
    )
    goal_indices = torch.zeros((1, 1), device=cuda_device_cfg.device, dtype=torch.int32)
    pose_cost = ToolPoseCost(
        ToolPoseCostCfg(
            weight=[0.0, 1.0],
            tool_frames=kinematics.tool_frames,
            device_cfg=cuda_device_cfg,
        )
    )
    pose_cost.setup_batch_tensors(batch_size=1, horizon=1)

    def orientation_cost(joint_position: torch.Tensor) -> torch.Tensor:
        state = kinematics.compute_kinematics(JointState.from_position(joint_position))
        cost, _, _, _ = pose_cost.forward(state.tool_poses, goal, goal_indices)
        return cost[..., 1::2].sum()

    differentiable_position = position.detach().clone().requires_grad_(True)
    orientation_cost(differentiable_position).backward()
    gradient = differentiable_position.grad.detach().clone()

    epsilon = 1.0e-3
    finite_difference = torch.empty_like(position)
    with torch.no_grad():
        for joint_index in range(position.shape[-1]):
            plus = position.clone()
            minus = position.clone()
            plus[:, joint_index] += epsilon
            minus[:, joint_index] -= epsilon
            plus_cost = orientation_cost(plus).clone()
            minus_cost = orientation_cost(minus).clone()
            finite_difference[:, joint_index] = (plus_cost - minus_cost) / (2.0 * epsilon)

    torch.testing.assert_close(gradient, finite_difference, rtol=3.0e-3, atol=5.0e-4)
