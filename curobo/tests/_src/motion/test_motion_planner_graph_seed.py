# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for graph-seeded motion planning control-space fallbacks."""

# Standard Library
from types import SimpleNamespace
from typing import Any

# Third Party
import torch

# CuRobo
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.state.state_joint import JointState


class _FakeIkSolver:
    """Minimal IK solver used by graph-seed fallback tests."""

    def solve_pose(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        """Return two valid IK seeds."""
        return SimpleNamespace(
            success=torch.ones((1, 2), dtype=torch.bool),
            solution=torch.tensor(
                [[[0.5, -0.2], [0.7, 0.1]]],
                dtype=torch.float32,
            ),
            total_time=0.1,
            solve_time=0.05,
        )


class _FakeTrajOptSolver:
    """Minimal trajectory optimizer that declines waypoint conversion."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(num_seeds=2)
        self.joint_names = ["joint_1", "joint_2"]
        self.solve_kwargs: dict[str, Any] = {}

    def prepare_seed_trajectory(self, *args: Any, **kwargs: Any) -> None:
        """Represent a control space without graph-waypoint conversion."""
        return None

    def solve_pose(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        """Capture solver inputs and return a successful result."""
        self.solve_kwargs = kwargs
        return SimpleNamespace(
            success=torch.ones((1, 1), dtype=torch.bool),
            total_time=0.2,
            solve_time=0.1,
        )


def test_plan_pose_handles_unsupported_graph_seed_conversion(monkeypatch: Any) -> None:
    """A ``None`` converted seed should fall back without dereferencing its shape."""
    planner = object.__new__(MotionPlanner)
    planner.ik_solver = _FakeIkSolver()
    planner.trajopt_solver = _FakeTrajOptSolver()
    planner.graph_planner = object()
    graph_waypoints = torch.zeros((1, 2, 20, 2), dtype=torch.float32)
    monkeypatch.setattr(
        planner,
        "_get_graph_seed_trajectories",
        lambda *args, **kwargs: graph_waypoints,
    )
    current_state = JointState.from_position(
        torch.zeros((1, 2), dtype=torch.float32),
        joint_names=planner.trajopt_solver.joint_names,
    )

    result = planner._plan_pose_single(
        goal_tool_poses=object(),
        current_state=current_state,
        max_attempts=1,
        enable_graph_attempt=0,
    )

    assert result is not None
    assert planner.trajopt_solver.solve_kwargs["seed_traj"] is None
    assert planner.trajopt_solver.solve_kwargs["seed_implicit_goal_state"] is None
