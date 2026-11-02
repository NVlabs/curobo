# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configured spline grids, retained targets, and control-count contracts."""

from __future__ import annotations

import pytest
import torch

from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.trajectory.trajectory_pose_reference import interpolate_pose_waypoints
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.spline_layout import SplineLayout
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


@pytest.mark.parametrize("num_knots", range(12, 1025))
def test_native_layout_retains_32_targets(num_knots: int) -> None:
    """Every supported integer count keeps its native endpoint holds and all targets."""
    layout = SplineLayout(num_knots)
    assert layout.native_horizon == 4 * (num_knots + 4) + 1
    assert layout.reference_intervals == 4 * (num_knots + 2)
    targets = torch.linspace(0.01, 0.32, 32, dtype=torch.float32)
    position = torch.zeros((1, 32, 1, 1, 3), dtype=torch.float32)
    position[0, :, 0, 0, 0] = targets
    q = torch.tensor([1, 0, 0, 0], dtype=torch.float32).reshape(1, 1, 1, 1, 4)
    start = ToolPose(["tool"], torch.zeros((1, 1, 1, 3)), q[:, :, :, 0])
    reference = interpolate_pose_waypoints(
        GoalToolPose(["tool"], position, q.expand(1, 32, 1, 1, 4)),
        start,
        horizon=layout.native_horizon,
        padding=layout.padding,
    )
    assert reference.horizon == layout.native_horizon
    torch.testing.assert_close(reference.position[:, layout.waypoint_indices(32)], position)
    torch.testing.assert_close(reference.position[:, :5], torch.zeros((1, 5, 1, 1, 3)))
    torch.testing.assert_close(
        reference.position[:, -5:], position[:, -1:].expand(-1, 5, -1, -1, -1)
    )


@pytest.mark.parametrize(
    "num_knots", [12, 13, 16, 17, 32, 63, 64, 65, 128, 256, 512, 513, 1023, 1024]
)
def test_configured_control_count_reaches_all_rollouts(num_knots: int) -> None:
    """Planner and standalone TrajOpt share one layout across optimization and validation."""
    planner = MotionPlannerCfg.create(robot="franka.yml", num_knots=num_knots)
    standalone = TrajOptSolverCfg.create(robot="franka.yml", num_knots=num_knots)
    expected = SplineLayout(num_knots)
    for config in (planner.trajopt_solver_config, standalone):
        assert config.spline_layout == expected
        assert config.num_knots == num_knots
        assert config.native_horizon == expected.native_horizon
        assert config.interpolation_buffer_size >= config.minimum_interpolation_buffer_size
        for rollout in [config.metrics_rollout_config, *config.optimizer_rollout_configs]:
            assert rollout.transition_model_cfg.native_horizon == expected.native_horizon
            assert rollout.transition_model_cfg.horizon == expected.native_horizon
    assert planner.native_horizon == expected.native_horizon
    assert planner.spline_layout == standalone.spline_layout


@pytest.mark.parametrize("num_knots", [0, 11, 1025, 2048, 12.5, True, "32"])
def test_invalid_control_counts_are_rejected(num_knots: object) -> None:
    """Reject unsupported counts before allocating solver resources."""
    with pytest.raises(ValueError, match="num_knots must be an integer"):
        TrajOptSolverCfg.create(robot="franka.yml", num_knots=num_knots)


def test_spline_native_horizon_agrees_with_existing_common_api() -> None:
    """Other spline degrees and sampling densities still use the shared calculation."""
    for degree in (ControlSpace.BSPLINE_3, ControlSpace.BSPLINE_4, ControlSpace.BSPLINE_5):
        for steps in (1, 4, 8):
            layout = SplineLayout(17, degree, steps)
            assert layout.native_horizon == ControlSpace.spline_total_interpolation_steps(
                degree, 17, steps
            )


@pytest.mark.parametrize("num_knots", range(12, 1025))
def test_native_spline_forward_and_backward_all_counts(num_knots: int) -> None:
    """Irregular control counts preserve endpoints and correct native CUDA gradients."""
    from curobo._src.state.state_joint import JointState
    from curobo._src.transition.fns_state_transition import StateFromBSplineKnot
    from curobo._src.types.device_cfg import DeviceCfg

    layout = SplineLayout(num_knots)
    device = DeviceCfg()
    operator = StateFromBSplineKnot(
        device,
        7,
        batch_size=1,
        horizon=layout.native_horizon,
        n_knots=num_knots,
        interpolation_steps=layout.interpolation_steps,
        control_space=layout.control_space,
    )
    start = JointState.zeros((1, 7), device)
    start.dt = torch.full((1,), 0.02, **device.as_torch_dict())
    index = torch.zeros((1,), device=device.device, dtype=torch.int32)
    implicit = torch.zeros((1,), device=device.device, dtype=torch.uint8)
    controls = torch.linspace(0, 0.2, num_knots, **device.as_torch_dict())[None, :, None].repeat(
        1, 1, 7
    )
    direction = torch.cos(torch.arange(controls.numel(), **device.as_torch_dict())).reshape_as(
        controls
    )
    weights = torch.sin(torch.arange(layout.native_horizon * 7, **device.as_torch_dict())).reshape(
        1, -1, 7
    )

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        output = JointState.zeros((1, layout.native_horizon, 7), device)
        output.dt = torch.full((1,), 0.02, **device.as_torch_dict())
        return operator.forward(
            start, value, output, index, use_implicit_goal_state=implicit
        ).position

    controls.requires_grad_(True)
    position = evaluate(controls)
    assert position.shape[1] == layout.native_horizon
    torch.testing.assert_close(position[:, 0], start.position)
    torch.testing.assert_close(position[:, -1], controls[:, -1])
    derivative = torch.autograd.grad((position * weights).sum(), controls)[0]
    assert torch.isfinite(derivative).all()
    # Position is affine in controls. A unit central difference is exact in real
    # arithmetic and avoids subtracting nearly equal FP32 sums on large grids.
    epsilon = 1.0
    with torch.no_grad():
        plus = (evaluate(controls + epsilon * direction) * weights).sum().clone()
        minus = (evaluate(controls - epsilon * direction) * weights).sum().clone()
        finite_difference = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(
        (derivative * direction).sum(), finite_difference, atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize("num_knots", [12, 13, 16, 17, 32, 63, 64, 128, 256, 512, 513, 1024])
def test_standalone_trajopt_uses_configured_native_grid(num_knots: int) -> None:
    """Standalone joint-goal optimization and seed fitting share the configured grid."""
    from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
    from curobo.types import JointState

    config = TrajectoryOptimizerCfg.create(robot="franka.yml", num_knots=num_knots, num_seeds=2)
    solver = TrajectoryOptimizer(config)
    try:
        start = JointState.from_position(
            solver.default_joint_state.position.reshape(1, -1).clone(),
            joint_names=solver.joint_names,
        )
        goal = start.clone()
        goal.position[:, 0] += 0.08
        result = solver.solve_cspace(goal, start, finetune_attempts=0)
        assert result is not None and result.success.all()
        assert (
            result.js_solution.position.shape[-2] == config.native_horizon == solver.native_horizon
        )
        assert result.solution.shape[-2] == num_knots
        derivative_buffer = solver.core._goal_buffer.current_js.velocity.data_ptr()
        metrics_graph = solver.metrics_rollout._compute_metrics_from_action_executor._graph
        start.velocity = torch.zeros_like(start.position)
        start.velocity[:, 0] = 0.01
        moving = solver.solve_cspace(goal, start, finetune_attempts=0)
        assert moving is not None and moving.success.all()
        torch.testing.assert_close(
            solver.get_active_js(moving.interpolated_trajectory).velocity[:, :, 0],
            start.velocity[:, None],
            atol=1e-4,
            rtol=1e-4,
        )
        assert solver.core._goal_buffer.current_js.velocity.data_ptr() == derivative_buffer
        assert solver.metrics_rollout._compute_metrics_from_action_executor._graph is metrics_graph
        start.velocity = None
        stopped = solver.solve_cspace(goal, start, finetune_attempts=0)
        assert stopped is not None and stopped.success.all()
        torch.testing.assert_close(
            solver.get_active_js(stopped.interpolated_trajectory).velocity[:, :, 0],
            torch.zeros_like(start.position[:, None]),
            atol=1e-4,
            rtol=1e-4,
        )
        assert solver.core._goal_buffer.current_js.velocity.data_ptr() == derivative_buffer
        assert solver.metrics_rollout._compute_metrics_from_action_executor._graph is metrics_graph
    finally:
        solver.destroy()
