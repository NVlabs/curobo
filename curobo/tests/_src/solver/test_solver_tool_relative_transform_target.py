# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for relative-transform target routing through solver APIs."""

# Standard Library
from types import SimpleNamespace

# Third Party
import pytest
import torch

# CuRobo
import curobo._src.runtime as curobo_runtime
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.solver.solver_core import SolverCore
from curobo._src.solver.solver_ik import IKSolver, _pad_tool_relative_transform_target
from curobo._src.solver.solver_mpc import MPCSolver
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.state.state_joint import JointState
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import GoalToolPose


def _make_target(num_target_groups: int) -> ToolRelativeTransformTarget:
    position = torch.arange(num_target_groups * 3, dtype=torch.float32).view(
        num_target_groups, 1, 3
    )
    quaternion = torch.zeros((num_target_groups, 1, 4), dtype=torch.float32)
    quaternion[..., 0] = 1.0
    return ToolRelativeTransformTarget(
        reference_tool_frame="tool",
        tool_frames=["tool"],
        position=position,
        quaternion=quaternion,
    )


def _make_goal(batch_size: int = 2) -> GoalToolPose:
    position = torch.zeros((batch_size, 1, 1, 1, 3), dtype=torch.float32)
    quaternion = torch.zeros((batch_size, 1, 1, 1, 4), dtype=torch.float32)
    quaternion[..., 0] = 1.0
    return GoalToolPose(
        tool_frames=["tool"],
        position=position,
        quaternion=quaternion,
    )


def _make_solve_state(batch_size: int = 2) -> SolveState:
    return SolveState(
        solve_type=SolveMode.BATCH,
        num_ik_seeds=1,
        num_trajopt_seeds=1,
        batch_size=batch_size,
        num_envs=1,
        num_goalset=1,
        tool_frames=["tool"],
    )


class _CapturingCore:
    def __init__(self):
        self.kinematics = SimpleNamespace(tool_frames=["tool"])
        self.kwargs = None

    def prepare_goal_buffer(self, **kwargs):
        self.kwargs = kwargs
        return "goal_buffer", False


class _CapturingRollout:
    def __init__(self):
        self.goals = []

    def update_params(self, goal):
        self.goals.append(goal)


class _CapturingOptimizer:
    def __init__(self):
        self.goals = []

    def update_rollout_params(self, goal):
        self.goals.append(goal)


def test_shared_relative_target_is_not_padded():
    target = _make_target(num_target_groups=1)

    padded = _pad_tool_relative_transform_target(target, batch_size=2, max_batch=4)

    assert padded is target


def test_per_problem_relative_target_is_padded_without_mutating_input():
    target = _make_target(num_target_groups=2)

    padded = _pad_tool_relative_transform_target(target, batch_size=2, max_batch=4)

    assert padded is not target
    assert padded.num_target_groups == 4
    torch.testing.assert_close(padded.position[:2], target.position)
    torch.testing.assert_close(padded.position[2:], target.position[:1].expand(2, -1, -1))
    torch.testing.assert_close(padded.quaternion[:2], target.quaternion)
    torch.testing.assert_close(padded.quaternion[2:], target.quaternion[:1].expand(2, -1, -1))
    assert target.num_target_groups == 2


def test_relative_target_group_count_must_match_problem_batch():
    target = _make_target(num_target_groups=3)

    with pytest.raises(ValueError, match="one shared target group or one group per problem"):
        _pad_tool_relative_transform_target(target, batch_size=2, max_batch=4)


def test_core_refreshes_interpolated_absolute_and_relative_targets():
    core = object.__new__(SolverCore)
    core.metrics_rollout = _CapturingRollout()
    core.auxiliary_rollout = _CapturingRollout()
    interpolated_rollout = _CapturingRollout()
    core.additional_metrics_rollouts = {"interpolated_rollout": interpolated_rollout}
    core.optimizer = _CapturingOptimizer()

    first_goal = GoalRegistry(
        link_goal_poses=_make_goal(batch_size=1),
        tool_relative_transform_target=_make_target(num_target_groups=1),
    )
    second_goal_pose = _make_goal(batch_size=1)
    second_goal_pose.position[..., 0] = 0.25
    second_relative_target = _make_target(num_target_groups=1)
    second_relative_target.position[..., 1] = 0.5
    second_goal = GoalRegistry(
        link_goal_poses=second_goal_pose,
        tool_relative_transform_target=second_relative_target,
    )

    core.update_rollout_params(first_goal)
    core.update_rollout_params(second_goal)

    assert interpolated_rollout.goals == [first_goal, second_goal]
    dense_goal = interpolated_rollout.goals[-1]
    assert dense_goal.link_goal_poses is second_goal_pose
    assert dense_goal.tool_relative_transform_target is second_relative_target
    torch.testing.assert_close(
        dense_goal.link_goal_poses.position[..., 0],
        torch.full_like(dense_goal.link_goal_poses.position[..., 0], 0.25),
    )
    torch.testing.assert_close(
        dense_goal.tool_relative_transform_target.position[..., 1],
        torch.full_like(dense_goal.tool_relative_transform_target.position[..., 1], 0.5),
    )


@pytest.mark.parametrize("solver_cls", [IKSolver, TrajOptSolver, MPCSolver])
def test_prepare_goal_buffer_forwards_relative_target(solver_cls):
    solver = object.__new__(solver_cls)
    solver.core = _CapturingCore()
    target = _make_target(num_target_groups=2)
    current_state = JointState.from_position(torch.zeros((2, 1), dtype=torch.float32))

    goal_buffer, update_reference = solver._prepare_goal_buffer(
        solve_state=_make_solve_state(),
        goal_tool_poses=_make_goal(),
        current_state=current_state,
        tool_relative_transform_target=target,
    )

    assert goal_buffer == "goal_buffer"
    assert update_reference is False
    assert solver.core.kwargs["tool_relative_transform_target"] is target


def test_ik_solve_pose_forwards_relative_target(monkeypatch):
    monkeypatch.setattr(curobo_runtime, "cuda_event_timers", False)
    solver = object.__new__(IKSolver)
    solver.core = _CapturingCore()
    solver.config = SimpleNamespace(
        max_batch_size=2,
        num_seeds=1,
        max_goalset=1,
        multi_env=False,
        use_lm_seed=False,
        optimization_dt=None,
    )
    captured = {}
    expected_result = SimpleNamespace(solve_time=0.0, total_time=0.0)

    def solve_impl(**kwargs):
        captured.update(kwargs)
        return expected_result

    solver._solve_impl = solve_impl
    target = _make_target(num_target_groups=2)

    result = solver.solve_pose(
        goal_tool_poses=_make_goal(),
        tool_relative_transform_target=target,
    )

    assert result is expected_result
    assert captured["tool_relative_transform_target"] is target


def test_trajopt_solve_pose_forwards_relative_target():
    solver = object.__new__(TrajOptSolver)
    solver.config = SimpleNamespace(
        max_batch_size=2,
        num_seeds=1,
        max_goalset=1,
        multi_env=False,
    )
    captured = {}
    expected_result = object()

    def solve_impl(**kwargs):
        captured.update(kwargs)
        return expected_result

    solver._solve_impl = solve_impl
    target = _make_target(num_target_groups=2)

    result = solver.solve_pose(
        goal_tool_poses=_make_goal(),
        current_state=JointState.from_position(torch.zeros((2, 1), dtype=torch.float32)),
        tool_relative_transform_target=target,
    )

    assert result is expected_result
    assert captured["tool_relative_transform_target"] is target


def test_mpc_updates_relative_target_and_rollouts():
    solver = object.__new__(MPCSolver)
    captured = {}

    class GoalManager:
        def update_tool_relative_transform_target(self, target):
            captured["target"] = target
            return "updated_goal_buffer"

    solver.core = SimpleNamespace(goal_registry_manager=GoalManager())
    solver._update_rollout_params = lambda goal: captured.update(goal=goal)
    target = _make_target(num_target_groups=1)

    solver.update_tool_relative_transform_target(target)

    assert captured == {
        "target": target,
        "goal": "updated_goal_buffer",
    }


def test_mpc_internal_ik_uses_installed_relative_target():
    solver = object.__new__(MPCSolver)
    target = _make_target(num_target_groups=2)
    current_state = JointState.from_position(torch.zeros((2, 1), dtype=torch.float32))
    solver.core = SimpleNamespace(
        action_dim=1,
        _goal_buffer=GoalRegistry(
            current_js=current_state,
            tool_relative_transform_target=target,
        ),
    )
    captured = {}

    class IKSolverStub:
        def solve_pose(self, **kwargs):
            captured.update(kwargs)
            return "ik_result"

    solver.ik_solver = IKSolverStub()

    result = solver._solve_ik_for_goal(_make_goal(batch_size=2))

    assert result == "ik_result"
    assert captured["tool_relative_transform_target"] is target
