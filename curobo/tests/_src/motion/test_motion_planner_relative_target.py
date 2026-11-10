# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Propagation tests for relative tool-transform pose-planning targets."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import curobo._src.motion.motion_planner as motion_planner_module
import curobo._src.motion.motion_planner_batch as motion_planner_batch_module
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.state.state_joint import JointState
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget


def _relative_target(batch_size: int = 1) -> ToolRelativeTransformTarget:
    return ToolRelativeTransformTarget(
        reference_tool_frame="reference",
        tool_frames=["reference", "follower"],
        position=torch.zeros(batch_size, 2, 3),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).repeat(batch_size, 2, 1),
    )


def _joint_state(batch_size: int = 1) -> JointState:
    return JointState.from_position(torch.zeros(batch_size, 2))


def _ik_result(batch_size: int, num_seeds: int) -> SimpleNamespace:
    return SimpleNamespace(
        success=torch.ones(batch_size, num_seeds, dtype=torch.bool),
        solution=torch.zeros(batch_size, num_seeds, 2),
        total_time=0.1,
        solve_time=0.1,
    )


def _traj_result(batch_size: int, success: bool) -> SimpleNamespace:
    return SimpleNamespace(
        success=torch.full((batch_size, 1), success, dtype=torch.bool),
        total_time=0.1,
        solve_time=0.1,
        debug_info={},
        trajectory_source=None,
        copy_at_batch_indices=Mock(),
    )


@pytest.mark.parametrize(
    ("num_goalset", "expected_method"),
    [(1, "_plan_pose_single"), (2, "_plan_pose_goalset")],
)
def test_motion_planner_public_pose_api_forwards_relative_target(
    num_goalset: int,
    expected_method: str,
) -> None:
    planner = object.__new__(MotionPlanner)
    planner._destroyed = True
    planner._plan_pose_single = Mock(return_value=object())
    planner._plan_pose_goalset = Mock(return_value=object())
    target = _relative_target()

    planner.plan_pose(
        SimpleNamespace(num_goalset=num_goalset),
        _joint_state(),
        tool_relative_transform_target=target,
    )

    called_method = getattr(planner, expected_method)
    assert called_method.call_args.kwargs["tool_relative_transform_target"] is target


def test_motion_planner_goalset_warmup_fallback_forwards_relative_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = object.__new__(MotionPlanner)
    planner._destroyed = True
    target = _relative_target()
    current_state = _joint_state()
    goal = SimpleNamespace()
    plan_pose = Mock(return_value=None)
    trajopt_solve = Mock(return_value=_traj_result(1, True))

    planner.config = SimpleNamespace(ik_solver_config=SimpleNamespace(max_goalset=2))
    planner.ik_solver = SimpleNamespace(config=SimpleNamespace(exit_early=True))
    planner.trajopt_solver = SimpleNamespace(
        config=SimpleNamespace(num_seeds=2),
        default_joint_state=current_state,
        solve_pose=trajopt_solve,
    )
    planner.graph_planner = None
    planner.plan_pose = plan_pose
    planner._make_warmup_goalset = Mock(return_value={})
    planner.reset_seed = Mock()
    planner._reset_graph_telemetry = Mock()
    monkeypatch.setattr(
        motion_planner_module.GoalToolPose,
        "from_poses",
        Mock(return_value=goal),
    )

    planner.warmup(
        enable_graph=False,
        num_warmup_iterations=1,
        tool_relative_transform_target=target,
    )

    assert plan_pose.call_args.kwargs["tool_relative_transform_target"] is target
    assert trajopt_solve.call_args.kwargs["tool_relative_transform_target"] is target


def test_motion_planner_single_graph_seed_forwards_relative_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = object.__new__(MotionPlanner)
    planner._destroyed = True
    target = _relative_target()
    graph_seed = torch.zeros(1, 2, 3, 2)
    ik_solve = Mock(return_value=_ik_result(1, 2))
    evaluate_seed = Mock(return_value=_traj_result(1, False))
    trajopt_solve = Mock(return_value=_traj_result(1, True))

    planner.config = SimpleNamespace(
        enable_trajopt=True,
        always_run_trajopt=False,
        run_trajopt_on_graph_seed=False,
    )
    planner._warming_up = False
    planner._reset_graph_telemetry = Mock()
    planner.graph_planner = object()
    planner.graph_seed_fitter = object()
    planner._last_graph_seed_success = torch.ones(1, 2, dtype=torch.bool)
    planner._get_graph_seed_trajectories = Mock(return_value=graph_seed)
    planner.ik_solver = SimpleNamespace(solve_pose=ik_solve)
    planner.trajopt_solver = SimpleNamespace(
        config=SimpleNamespace(num_seeds=2),
        action_horizon=3,
        evaluate_pose_seed=evaluate_seed,
        solve_pose=trajopt_solve,
    )
    monkeypatch.setattr(
        motion_planner_module,
        "_fit_linear_seed_trajectories",
        lambda *args, **kwargs: graph_seed,
    )

    result = planner._plan_pose_single(
        SimpleNamespace(),
        _joint_state(),
        use_implicit_goal=True,
        max_attempts=1,
        enable_graph_attempt=0,
        tool_relative_transform_target=target,
    )

    assert result is not None
    assert ik_solve.call_args.kwargs["tool_relative_transform_target"] is target
    assert evaluate_seed.call_args.kwargs["tool_relative_transform_target"] is target
    assert trajopt_solve.call_args.kwargs["tool_relative_transform_target"] is target
    assert evaluate_seed.call_args.kwargs["seed_valid_mask"] is planner._last_graph_seed_success


def test_motion_planner_goalset_forwards_relative_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = object.__new__(MotionPlanner)
    planner._destroyed = True
    target = _relative_target()
    seed_traj = torch.zeros(1, 2, 3, 2)
    ik_solve = Mock(return_value=_ik_result(1, 2))
    evaluate_seed = Mock(return_value=_traj_result(1, False))
    trajopt_solve = Mock(return_value=_traj_result(1, True))

    planner.config = SimpleNamespace(
        enable_trajopt=True,
        always_run_trajopt=False,
        run_trajopt_on_graph_seed=False,
    )
    planner._warming_up = False
    planner._reset_graph_telemetry = Mock()
    planner.graph_seed_fitter = object()
    planner.ik_solver = SimpleNamespace(solve_pose=ik_solve)
    planner.trajopt_solver = SimpleNamespace(
        config=SimpleNamespace(num_seeds=2),
        action_horizon=3,
        evaluate_pose_seed=evaluate_seed,
        solve_pose=trajopt_solve,
    )
    monkeypatch.setattr(
        motion_planner_module,
        "_fit_linear_seed_trajectories",
        lambda *args, **kwargs: seed_traj,
    )

    result = planner._plan_pose_goalset(
        SimpleNamespace(),
        _joint_state(),
        max_attempts=1,
        tool_relative_transform_target=target,
    )

    assert result is not None
    assert ik_solve.call_args.kwargs["tool_relative_transform_target"] is target
    assert evaluate_seed.call_args.kwargs["tool_relative_transform_target"] is target
    assert trajopt_solve.call_args.kwargs["tool_relative_transform_target"] is target


def test_batch_motion_planner_graph_seed_forwards_relative_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = object.__new__(BatchMotionPlanner)
    planner._destroyed = True
    batch_size = 2
    target = _relative_target(batch_size)
    seed_traj = torch.zeros(batch_size, 2, 3, 2)
    ik_solve = Mock(return_value=_ik_result(batch_size, 2))
    evaluate_seed = Mock(return_value=_traj_result(batch_size, False))
    trajopt_solve = Mock(return_value=_traj_result(batch_size, True))

    planner.config = SimpleNamespace(
        ik_solver_config=SimpleNamespace(max_batch_size=batch_size),
        enable_trajopt=True,
        always_run_trajopt=False,
        run_trajopt_on_graph_seed=False,
    )
    planner._warming_up = False
    planner.graph_planner = object()
    planner.graph_seed_fitter = object()
    planner._last_graph_seed_success = torch.ones(batch_size, 2, dtype=torch.bool)
    planner._get_graph_seed_trajectories = Mock(return_value=seed_traj)
    planner.ik_solver = SimpleNamespace(solve_pose=ik_solve)
    planner.trajopt_solver = SimpleNamespace(
        config=SimpleNamespace(num_seeds=2),
        action_horizon=3,
        evaluate_pose_seed=evaluate_seed,
        solve_pose=trajopt_solve,
    )
    monkeypatch.setattr(
        motion_planner_batch_module,
        "_fit_linear_seed_trajectories",
        lambda *args, **kwargs: seed_traj,
    )

    result = planner.plan_pose(
        SimpleNamespace(),
        _joint_state(batch_size),
        max_attempts=1,
        tool_relative_transform_target=target,
    )

    assert result is not None
    assert ik_solve.call_args.kwargs["tool_relative_transform_target"] is target
    assert evaluate_seed.call_args.kwargs["tool_relative_transform_target"] is target
    assert trajopt_solve.call_args.kwargs["tool_relative_transform_target"] is target
    assert evaluate_seed.call_args.kwargs["seed_valid_mask"] is planner._last_graph_seed_success


@pytest.mark.parametrize("planner_type", [MotionPlanner, BatchMotionPlanner])
def test_motion_planner_updates_relative_and_velocity_criteria(planner_type) -> None:
    """Public planners route geometry to both solvers and speed only to TrajOpt."""
    planner = object.__new__(planner_type)
    planner._destroyed = True
    planner.ik_solver = SimpleNamespace(
        update_tool_relative_transform_criteria=Mock(),
    )
    planner.trajopt_solver = SimpleNamespace(
        update_tool_relative_transform_criteria=Mock(),
        update_tool_velocity_limit_criteria=Mock(),
    )
    relative_criteria = {"follower": object()}
    velocity_criteria = {"follower": object()}

    planner.update_tool_relative_transform_criteria(relative_criteria)
    planner.update_tool_velocity_limit_criteria(velocity_criteria)

    planner.ik_solver.update_tool_relative_transform_criteria.assert_called_once_with(
        relative_criteria
    )
    planner.trajopt_solver.update_tool_relative_transform_criteria.assert_called_once_with(
        relative_criteria
    )
    planner.trajopt_solver.update_tool_velocity_limit_criteria.assert_called_once_with(
        velocity_criteria
    )
