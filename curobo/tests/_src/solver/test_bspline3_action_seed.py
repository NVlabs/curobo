# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for cubic B-spline action seed fitting."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.trajopt_seed.bspline3_action_seed import (
    BSpline3ActionSeedGenerator,
)
from curobo._src.state.state_joint import JointState


def _generator() -> BSpline3ActionSeedGenerator:
    return BSpline3ActionSeedGenerator(
        n_knots=6,
        interpolation_steps=4,
        interpolated_dt=0.05,
    )


def _linear_waypoints(
    start: torch.Tensor,
    goal: torch.Tensor,
    n_waypoints: int = 20,
) -> torch.Tensor:
    alpha = torch.linspace(0.0, 1.0, n_waypoints, dtype=start.dtype)
    return start[:, None, None, :] + alpha[None, None, :, None] * (
        goal[:, :, None, :] - start[:, None, None, :]
    )


def test_fit_action_seed_preserves_shape_dtype_and_terminal_position() -> None:
    """Fitted seeds should preserve layout and end at each waypoint goal."""
    generator = _generator()
    start = torch.tensor([[0.0, 0.2], [0.3, -0.1]], dtype=torch.float32)
    goal = torch.tensor(
        [
            [[1.0, 0.5], [0.7, -0.4], [0.2, 0.8]],
            [[-0.2, 0.6], [0.9, 0.1], [0.4, -0.7]],
        ],
        dtype=torch.float32,
    )
    waypoints = _linear_waypoints(start, goal)

    action_seed = generator.fit_action_seed(
        waypoints,
        current_state=JointState.from_position(start),
        goal_state=None,
        use_implicit_goal=True,
    )

    assert action_seed.shape == (2, 3, generator.n_knots, 2)
    assert action_seed.dtype == torch.float32
    assert torch.isfinite(action_seed).all()
    torch.testing.assert_close(action_seed[:, :, -1], goal)


def test_fit_action_seed_matches_implicit_goal_boundary_knot() -> None:
    """The overlapping final action knot should match the first fixed goal knot."""
    generator = _generator()
    start = torch.zeros((1, 2), dtype=torch.float32)
    goal_position = torch.tensor([[[1.0, -0.5], [0.5, 0.75]]], dtype=torch.float32)
    goal_velocity = torch.tensor([[[0.2, -0.1], [-0.3, 0.4]]], dtype=torch.float32)
    goal_acceleration = torch.tensor([[[0.05, 0.02], [0.01, -0.04]]], dtype=torch.float32)
    waypoints = _linear_waypoints(start, goal_position)
    goal_state = JointState(
        position=goal_position,
        velocity=goal_velocity,
        acceleration=goal_acceleration,
    )

    action_seed = generator.fit_action_seed(
        waypoints,
        current_state=JointState.from_position(start),
        goal_state=goal_state,
        use_implicit_goal=True,
    )

    knot_dt = torch.tensor(generator.knot_dt, dtype=torch.float32)
    for seed_idx in range(goal_position.shape[1]):
        fixed_goal_knots = generator._compute_fixed_knots(
            goal_position[0, seed_idx],
            -goal_velocity[0, seed_idx],
            goal_acceleration[0, seed_idx],
            knot_dt,
        ).flip(0)
        torch.testing.assert_close(action_seed[0, seed_idx, -1], fixed_goal_knots[0])


def test_fit_action_seed_handles_stationary_waypoints() -> None:
    """Zero-length waypoint paths should produce finite constant seeds."""
    generator = _generator()
    position = torch.tensor([[0.4, -0.2]], dtype=torch.float32)
    waypoints = position[:, None, None, :].expand(1, 2, 20, 2).clone()

    action_seed = generator.fit_action_seed(
        waypoints,
        current_state=JointState.from_position(position),
        goal_state=None,
        use_implicit_goal=True,
    )

    assert torch.isfinite(action_seed).all()
    torch.testing.assert_close(action_seed, position[:, None, None, :].expand_as(action_seed))


@pytest.mark.parametrize(
    ("waypoint_shape", "message"),
    [
        ((2, 20, 3), "waypoints must have shape"),
        ((1, 2, 12, 3), "Need at least 13 waypoints"),
    ],
)
def test_fit_action_seed_rejects_invalid_waypoints(
    waypoint_shape: tuple[int, ...],
    message: str,
) -> None:
    """Malformed or undersampled waypoint tensors should fail clearly."""
    generator = _generator()
    waypoints = torch.zeros(waypoint_shape, dtype=torch.float32)
    current_state = JointState.from_position(torch.zeros((1, 3), dtype=torch.float32))

    with pytest.raises(ValueError, match=message):
        generator.fit_action_seed(
            waypoints,
            current_state=current_state,
            goal_state=None,
            use_implicit_goal=True,
        )


def test_fit_action_seed_rejects_current_state_batch_mismatch() -> None:
    """Current state batch and waypoint batch must agree."""
    generator = _generator()
    waypoints = torch.zeros((2, 1, 20, 3), dtype=torch.float32)
    current_state = JointState.from_position(torch.zeros((1, 3), dtype=torch.float32))

    with pytest.raises(ValueError, match="current_state.position must have shape"):
        generator.fit_action_seed(
            waypoints,
            current_state=current_state,
            goal_state=None,
            use_implicit_goal=True,
        )


def test_fit_action_seed_rejects_explicit_goal_mode() -> None:
    """The generator should reject its unsupported explicit-goal mode."""
    generator = _generator()
    waypoints = torch.zeros((1, 1, 20, 2), dtype=torch.float32)
    current_state = JointState.from_position(torch.zeros((1, 2), dtype=torch.float32))

    with pytest.raises(ValueError, match="requires use_implicit_goal=True"):
        generator.fit_action_seed(
            waypoints,
            current_state=current_state,
            goal_state=None,
            use_implicit_goal=False,
        )


def test_design_matrix_is_partition_of_unity_at_boundaries() -> None:
    """Cubic basis rows should sum to one across the full parameter range."""
    generator = _generator()
    parameter = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)

    design = generator._build_design_matrix(parameter, n_control=13)

    torch.testing.assert_close(design.sum(dim=1), torch.ones(5, dtype=torch.float32))
    assert torch.count_nonzero(design[0]) == 3
    assert torch.count_nonzero(design[-1]) == 3
