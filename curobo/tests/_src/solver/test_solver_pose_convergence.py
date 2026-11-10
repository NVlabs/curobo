# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for absolute and relative pose convergence parsing."""

# Standard Library
from types import SimpleNamespace

# Third Party
import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_mpc import MPCSolver
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.tool_pose import ToolPose


def _make_convergence() -> CostCollection:
    """Create intentionally interleaved absolute/relative metrics for two seeds."""
    convergence = CostCollection()
    convergence.add(
        torch.tensor([[[0.01, 0.02]], [[0.05, 0.06]]]),
        "tool_pose_position_tolerance",
    )
    convergence.add(
        torch.tensor([[[1.0, 1.0, 1.0]], [[0.01, 0.02, 0.03]]]),
        "tool_relative_transform_position_tolerance",
    )
    convergence.add(
        torch.tensor([[[0.03, 0.04]], [[0.07, 0.08]]]),
        "tool_pose_orientation_tolerance",
    )
    convergence.add(
        torch.tensor([[[1.0, 1.0, 1.0]], [[0.01, 0.02, 0.03]]]),
        "tool_relative_transform_orientation_tolerance",
    )
    convergence.add(
        torch.tensor([[[3, 4]], [[7, 8]]], dtype=torch.int32),
        "tool_pose_goalset_index",
    )
    return convergence


def _make_metrics() -> RolloutMetrics:
    costs_and_constraints = CostsAndConstraints()
    costs_and_constraints.constraints.add(torch.zeros((2, 1, 1)), "feasible")
    return RolloutMetrics(
        costs_and_constraints=costs_and_constraints,
        convergence=_make_convergence(),
    )


def test_trajopt_result_keeps_absolute_pose_errors_and_goalset_width_independent():
    result = TrajOptSolverResult(
        success=None,
        metrics=_make_metrics(),
        batch_size=1,
        num_seeds=2,
        position_tolerance=0.1,
        orientation_tolerance=0.1,
    )

    result._process_metrics()

    torch.testing.assert_close(result.position_error, torch.tensor([[0.02, 0.06]]))
    torch.testing.assert_close(result.rotation_error, torch.tensor([[0.04, 0.08]]))
    torch.testing.assert_close(
        result.goalset_index,
        torch.tensor([[[3, 4], [7, 8]]], dtype=torch.int32),
    )
    torch.testing.assert_close(result.success, torch.tensor([[False, True]]))
    assert result.seed_cost.shape == (1, 2)


def test_ik_result_selects_by_all_constraints_but_reports_absolute_pose_only():
    solver = object.__new__(IKSolver)
    transition_model = SimpleNamespace(get_full_dof_from_solution=lambda state: state)
    solver.core = SimpleNamespace(
        action_dim=1,
        joint_names=["joint"],
        auxiliary_rollout=SimpleNamespace(transition_model=transition_model),
        goal_registry_manager=SimpleNamespace(batch_helper=torch.tensor([[0]])),
    )
    solver.config = SimpleNamespace(
        position_tolerance=0.1,
        orientation_tolerance=0.1,
        success_requires_convergence=True,
    )

    metrics = _make_metrics()
    tool_position = torch.zeros((2, 1, 2, 3))
    tool_quaternion = torch.zeros((2, 1, 2, 4))
    tool_quaternion[..., 0] = 1.0
    metrics.state = RobotState(
        joint_state=JointState.from_position(torch.tensor([[[0.0]], [[1.0]]])),
        cuda_robot_model_state=KinematicsState(
            tool_poses=ToolPose(
                tool_frames=["tool_0", "tool_1"],
                position=tool_position,
                quaternion=tool_quaternion,
            ),
            robot_spheres=torch.zeros((2, 1, 1, 4)),
            robot_com=torch.zeros((2, 1, 4)),
        ),
    )

    result = solver._get_result(
        opt_result=torch.tensor([[[0.0]], [[1.0]]]),
        metrics_result=metrics,
        num_seeds=2,
        return_seeds=1,
    )

    torch.testing.assert_close(result.success, torch.tensor([[True]]))
    torch.testing.assert_close(result.solution, torch.tensor([[[1.0]]]))
    torch.testing.assert_close(result.position_error, torch.tensor([[0.06]]))
    torch.testing.assert_close(result.rotation_error, torch.tensor([[0.08]]))
    torch.testing.assert_close(
        result.goalset_index,
        torch.tensor([[[7, 8]]], dtype=torch.int32),
    )


def test_mpc_result_reports_absolute_pose_and_keeps_all_convergence_terms():
    solver = object.__new__(MPCSolver)
    solver.core = SimpleNamespace(
        solve_state=SimpleNamespace(get_trajopt_batch_size=lambda: 2),
        optimizer=SimpleNamespace(get_recorded_trace=lambda: None),
    )
    solver.config = SimpleNamespace(
        num_seeds=1,
        position_tolerance=0.1,
        orientation_tolerance=0.1,
    )
    metrics = _make_metrics()
    metrics.convergence.add(
        torch.tensor([[[0.02]], [[0.005]]]),
        "goal_cspace_tolerance",
    )

    result = solver._get_result(metrics)

    torch.testing.assert_close(result.success, torch.tensor([True, True]))
    torch.testing.assert_close(result.position_error, torch.tensor([0.02, 0.06]))
    torch.testing.assert_close(result.rotation_error, torch.tensor([0.04, 0.08]))
    torch.testing.assert_close(
        result.goalset_index,
        torch.tensor([[[3, 4]], [[7, 8]]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        result.debug_info["converged"],
        torch.tensor([False, True]),
    )
