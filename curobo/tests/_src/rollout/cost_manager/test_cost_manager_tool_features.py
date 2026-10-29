# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for relative-transform and tool-velocity cost dispatch."""

# Standard Library
from types import SimpleNamespace

# Third Party
import pytest
import torch
import warp as wp

# CuRobo
from curobo._src.cost.cost_tool_relative_transform_cfg import ToolRelativeTransformCostCfg
from curobo._src.cost.cost_tool_velocity_limit_cfg import ToolVelocityLimitCostCfg
from curobo._src.cost.criteria_tool_relative_transform import ToolRelativeTransformCriteria
from curobo._src.cost.criteria_tool_velocity_limit import ToolVelocityLimitCriteria
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


def _transition(tool_frames: list[str], *, compute_jacobian: bool = False):
    robot_model = SimpleNamespace(
        tool_frames=tool_frames,
        total_spheres=0,
        compute_jacobian=compute_jacobian,
        get_self_collision_config=lambda: None,
    )
    return SimpleNamespace(robot_model=robot_model, interpolation_steps=1)


def _device_cfg() -> DeviceCfg:
    if not torch.cuda.is_available():
        pytest.skip("RobotCostManager uses CUDA streams")
    wp.init()
    return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)


def test_relative_transform_dispatch_and_convergence() -> None:
    device_cfg = _device_cfg()
    criteria = {
        "tool": ToolRelativeTransformCriteria.track_position(device_cfg=device_cfg),
    }
    manager = RobotCostManager(device_cfg)
    manager.initialize_from_config(
        RobotCostManagerCfg(
            tool_relative_transform_cfg=ToolRelativeTransformCostCfg(
                weight=[2.0, 1.0],
                reference_tool_frame="reference",
                tool_relative_transform_criteria=criteria,
                device_cfg=device_cfg,
            )
        ),
        _transition(["reference", "tool"]),
    )

    position = torch.zeros((1, 2, 2, 3), **device_cfg.as_torch_dict())
    position[:, :, 1, 0] = 0.3
    quaternion = torch.zeros((1, 2, 2, 4), **device_cfg.as_torch_dict())
    quaternion[..., 0] = 1.0
    state = RobotState(
        joint_state=JointState.from_position(
            torch.zeros((1, 2, 1), **device_cfg.as_torch_dict())
        ),
        cuda_robot_model_state=KinematicsState(
            tool_poses=ToolPose(
                tool_frames=["reference", "tool"],
                position=position,
                quaternion=quaternion,
            )
        ),
    )
    target_position = torch.zeros((1, 2, 3), **device_cfg.as_torch_dict())
    target_position[:, 1, 0] = 0.2
    target_quaternion = torch.zeros((1, 2, 4), **device_cfg.as_torch_dict())
    target_quaternion[..., 0] = 1.0
    goal = GoalRegistry(
        batch_size=1,
        tool_relative_transform_target=ToolRelativeTransformTarget(
            reference_tool_frame="reference",
            tool_frames=["reference", "tool"],
            position=target_position,
            quaternion=target_quaternion,
        ),
    )

    costs = manager.compute_costs(state, goal=goal)
    convergence = manager.compute_convergence(state, goal=goal)

    assert costs.names == ["tool_relative_transform"]
    assert torch.all(costs.values[0][..., 1] > 0.0)
    assert "tool_relative_transform_position_tolerance" in convergence.names
    assert "tool_relative_transform_orientation_tolerance" in convergence.names


def test_tool_velocity_limit_dispatch_and_gradients() -> None:
    device_cfg = _device_cfg()
    manager = RobotCostManager(device_cfg)
    manager.initialize_from_config(
        RobotCostManagerCfg(
            tool_velocity_limit_cfg=ToolVelocityLimitCostCfg(
                weight=[2.0, 1.0],
                tool_velocity_limit_criteria={
                    "tool": ToolVelocityLimitCriteria.limit(
                        max_linear_speed=0.1,
                        max_angular_speed=10.0,
                        device_cfg=device_cfg,
                    )
                },
                device_cfg=device_cfg,
                use_grad_input=True,
            )
        ),
        _transition(["tool"], compute_jacobian=True),
    )

    velocity = torch.full(
        (1, 2, 1),
        0.5,
        **device_cfg.as_torch_dict(),
        requires_grad=True,
    )
    jacobian = torch.zeros(
        (1, 2, 1, 6, 1),
        **device_cfg.as_torch_dict(),
    )
    jacobian[..., 0, 0] = 1.0
    jacobian.requires_grad_(True)
    state = RobotState(
        joint_state=JointState(
            position=torch.zeros((1, 2, 1), **device_cfg.as_torch_dict()),
            velocity=velocity,
            device_cfg=device_cfg,
        ),
        cuda_robot_model_state=KinematicsState(tool_jacobians=jacobian),
    )

    costs = manager.compute_costs(state)
    costs.values[0].sum().backward()

    assert costs.names == ["tool_velocity_limit"]
    assert torch.all(costs.values[0][..., 0] > 0.0)
    assert velocity.grad is not None
    assert jacobian.grad is not None


def test_tool_velocity_limit_rollout_uses_full_horizon_and_backpropagates() -> None:
    device_cfg = _device_cfg()
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    rollout_cfg = RobotRolloutCfg.create_with_component_types(
        data_dict={
            "transition_model_cfg": {
                "dt_traj_params": {
                    "base_dt": 0.05,
                    "base_ratio": 1.0,
                    "max_dt": 0.05,
                },
                "control_space": "POSITION",
                "teleport_mode": False,
                "batch_size": 1,
                "horizon": 8,
                "state_filter_cfg": {"filter_coeff": {}, "enable": False},
            },
            "constraint_cfg": {
                "tool_velocity_limit_cfg": {
                    "weight": [1.0, 1.0],
                    "tool_velocity_limit_criteria": {
                        "panda_hand": {
                            "linear_projector": [
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "angular_projector": [
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "max_linear_speed": 1.0e-6,
                            "max_angular_speed": 1.0e-6,
                        }
                    },
                }
            },
        },
        robot_cfg=robot_cfg,
        device_cfg=device_cfg,
    )
    rollout = RobotRollout(rollout_cfg)
    start = JointState.from_position(
        rollout.default_joint_position.unsqueeze(0),
        joint_names=rollout.transition_model.joint_names,
    )
    seed_goal = start.clone().unsqueeze(1)
    seed_goal.dt = torch.full((1, 1), 0.05, **device_cfg.as_torch_dict())
    rollout.update_params(GoalRegistry(current_js=start, seed_goal_js=seed_goal))
    actions = rollout.default_joint_position.view(1, 1, -1).repeat(
        1, rollout.action_horizon, 1
    )
    actions = actions + torch.linspace(
        0.002,
        0.008,
        rollout.action_horizon,
        device=device_cfg.device,
        dtype=device_cfg.dtype,
    ).view(1, rollout.action_horizon, 1)
    actions.requires_grad_(True)

    result = rollout.evaluate_action(actions)
    constraint = result.costs_and_constraints.get_sum_constraint(sum_horizon=True)
    constraint.sum().backward()

    assert rollout.transition_model.robot_model.compute_jacobian
    assert result.state.tool_jacobians is not None
    assert result.costs_and_constraints.constraints.names == ["tool_velocity_limit"]
    assert result.costs_and_constraints.constraints.values[0].shape[1] == rollout.horizon
    assert torch.all(constraint > 0.0)
    assert actions.grad is not None
    assert torch.count_nonzero(actions.grad) > 0
