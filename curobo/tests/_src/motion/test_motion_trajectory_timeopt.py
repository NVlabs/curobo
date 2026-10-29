# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone timeopt integration: continuous states, failures, and graph replay."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pytest
import torch

from curobo._src.curobolib.backends import trajectory as trajectory_cu
from curobo._src.state.state_joint import JointState
from curobo._src.trajectory.timeopt.timeopt_solver import _StateOffset
from curobo.kinematics import KinematicsCfg
from curobo.trajectory_timeopt import TrajectoryTimeOpt, TrajectoryTimeOptCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def full_precision_reference() -> Iterator[None]:
    """Keep FP32 equation references independent of IK's process-wide TF32 setting."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("intervals", [127, 512, 514])
def test_finish_matches_equations(robot: KinematicsCfg, intervals: int) -> None:
    """Check analytic states, quadrature, and failure masks on identical controls."""
    timeopt = TrajectoryTimeOpt(
        TrajectoryTimeOptCfg(
            robot,
            batch_size=4,
            num_intervals=intervals,
            acceleration_limits=torch.tensor([3.0], device="cuda"),
            jerk_limits=torch.tensor([50.0], device="cuda"),
        )
    )
    controls, start, goal = inputs(4)
    controls[1, 3] = torch.nan
    controls[2] = start[2] = goal[2] = 1.5
    controls[3] = start[3] = goal[3] = 0.0
    actual = timeopt.timeopt(controls, start, goal)
    assert actual.success.tolist() == [True, False, False, True]
    assert actual.status.tolist() == [0, 3, 1, 0]
    timing = actual.timing_controls
    samples = timeopt._basis.at(timeopt._evaluation.grid_cpu)
    x, first, second = (
        (timing @ samples.reshape(-1, timeopt.config.timing_controls).T)
        .reshape(4, 3, -1)
        .unbind(1)
    )
    q1, q2, q3 = timeopt._path[1:]
    speed = x.clamp_min(0).sqrt()[..., None]
    velocity = q1 * speed
    acceleration = q2 * x[..., None] + 0.5 * q1 * first[..., None]
    jerk = speed * (q3 * x[..., None] + 1.5 * q2 * first[..., None] + 0.5 * q1 * second[..., None])
    torque = (
        0.5 * timeopt._coefficients[:, 0] * first[..., None]
        + timeopt._coefficients[:, 1] * x[..., None]
        + timeopt._coefficients[:, 2]
    )
    count = intervals + 1
    for field, expected in zip(
        ("position", "velocity", "acceleration", "jerk"),
        (timeopt._path[0], velocity, acceleration, jerk),
    ):
        expected = torch.where(actual.success[:, None, None], expected[:, :count], torch.nan)
        torch.testing.assert_close(
            getattr(actual.joint_state, field), expected, rtol=2e-4, atol=3e-4, equal_nan=True
        )
    expected_torque = torch.where(actual.success[:, None, None], torque[:, :count], torch.nan)
    torch.testing.assert_close(
        actual.torque, expected_torque, rtol=3e-5, atol=3e-6, equal_nan=True
    )
    quadrature = (timing @ timeopt._time_basis.T).reshape(4, intervals, -1)
    dt = (quadrature.clamp_min(1e-20).rsqrt() * timeopt._weights).sum(-1) * torch.diff(
        timeopt._grid
    )
    times = torch.cat((torch.zeros_like(dt[:, :1]), dt.cumsum(-1)), -1)
    times = torch.where(actual.stationary[:, None], 0.0, times)
    times = torch.where(actual.success[:, None], times, torch.nan)
    torch.testing.assert_close(actual.time, times, rtol=3e-5, atol=3e-6, equal_nan=True)


@pytest.mark.parametrize("intervals", [0, 1.5])
def test_invalid_duration_intervals_rejected(robot: KinematicsCfg, intervals: int | float) -> None:
    """The uniform grid requires a positive integer number of intervals."""
    with pytest.raises(ValueError, match="num_intervals must be an integer >= 1"):
        TrajectoryTimeOpt(
            TrajectoryTimeOptCfg(
                robot,
                num_intervals=intervals,
                acceleration_limits=torch.tensor([3.0], device="cuda"),
                jerk_limits=torch.tensor([50.0], device="cuda"),
            )
        )


@pytest.fixture
def robot(tmp_path: Path) -> KinematicsCfg:
    """One rigid pendulum, including gravity and an explicit effort limit."""
    path = tmp_path / "pendulum.urdf"
    path.write_text("""<robot name="pendulum"><link name="base"/>
<link name="body"><inertial><origin xyz="0 0 0.5"/><mass value="1"/>
<inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
</inertial></link><joint name="hinge" type="revolute"><parent link="base"/>
<child link="body"/><axis xyz="0 1 0"/>
<limit lower="-3" upper="3" velocity="2" effort="2"/></joint></robot>""")
    return KinematicsCfg.from_basic_urdf(str(path), "base", ["body"])


def make_timeopt(
    robot: KinematicsCfg,
    batch: int,
    graph: bool,
    iterations: int = 10,
) -> TrajectoryTimeOpt:
    """Use a small solver budget while exercising the complete fixed-budget solve."""
    return TrajectoryTimeOpt(
        TrajectoryTimeOptCfg(
            robot,
            batch_size=batch,
            num_intervals=128,
            max_iterations=iterations,
            acceleration_limits=torch.tensor([3.0], device="cuda"),
            jerk_limits=torch.tensor([50.0], device="cuda"),
            use_cuda_graph=graph,
        )
    )


def inputs(batch: int) -> tuple[torch.Tensor, ...]:
    """Distinct paths with the native 16-action static endpoint convention."""
    start = torch.zeros((batch, 1), device="cuda")
    goal = torch.linspace(0.08, 0.2, batch, device="cuda")[:, None]
    controls = torch.linspace(0, 1, 16, device="cuda")[None, :, None] * goal[:, None]
    return controls.contiguous(), start, goal


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("allow_tf32", [False, True])
def test_batched_timeopt_replay_and_time_sampling(
    robot: KinematicsCfg,
    graph: bool,
    allow_tf32: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare independent lanes and evaluate derivatives using analytic pendulum dynamics."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", allow_tf32)
    timeopt = make_timeopt(robot, 2, graph)
    single = make_timeopt(robot, 1, False)
    controls, start, goal = inputs(2)
    for factor in (1.0, 0.8):
        result = timeopt.timeopt(controls * factor, start, goal * factor)
        assert result.success.all(), (result.status, result.optimization_ok)
        assert result.optimization_ok.all()
        assert (torch.diff(result.time) > 0).all()
        assert (result.duration <= result.uniform_duration * 1.0001).all()
        assert (result.joint_state.velocity.abs() <= 2.0).all()
        assert (result.joint_state.acceleration.abs() <= 3.0).all()
        assert (result.joint_state.jerk.abs() <= 50.0).all()
        torch.testing.assert_close(result.joint_state.position[:, 0], start)
        torch.testing.assert_close(result.joint_state.position[:, -1], goal * factor)
        for lane in range(2):
            expected = single.timeopt(
                (controls[lane : lane + 1] * factor).contiguous(),
                start[lane : lane + 1],
                (goal[lane : lane + 1] * factor).contiguous(),
            )
            torch.testing.assert_close(
                result.duration[lane : lane + 1], expected.duration, rtol=3e-3, atol=1e-4
            )
        times = (result.duration[:, None] * torch.linspace(0, 1, 257, device="cuda")).contiguous()
        sampled = timeopt.sample(times)
        assert torch.isfinite(sampled.jerk).all()
        torch.testing.assert_close(sampled.position[:, 0], start)
        torch.testing.assert_close(sampled.position[:, -1], goal * factor)
        assert sampled.velocity[:, [0, -1]].abs().max() < 1e-5
        assert sampled.acceleration[:, [0, -1]].abs().max() < 1e-4
        # Inertia about hinge = .1 + 1*.5**2; gravity torque = -m*g*.5*sin(q).
        torque = 0.35 * sampled.acceleration - 4.905 * sampled.position.sin()
        assert torque.abs().max() < 2.0
    retained = result.clone()
    time = retained.time.clone()
    timeopt.timeopt(controls * 0.5, start, goal * 0.5)
    torch.testing.assert_close(retained.time, time)
    timeopt.reset_cuda_graph()
    assert timeopt.timeopt(controls, start, goal).success.all()


def test_mixed_invalid_and_stationary_lanes(robot: KinematicsCfg) -> None:
    """Invalid or torque-infeasible lanes do not poison other paths or replay."""
    timeopt = make_timeopt(robot, 4, True)
    controls, start, goal = inputs(4)
    controls[1, 3] = torch.nan
    controls[2] = 1.5
    start[2] = goal[2] = 1.5
    controls[3] = start[3] = goal[3] = 0.0
    result = timeopt.timeopt(controls, start, goal)
    assert result.success.tolist() == [True, False, False, True]
    assert result.stationary.tolist() == [False, False, True, True]
    assert result.duration[3] == 0
    assert torch.isnan(result.joint_state.position[1:3]).all()
    assert torch.all(result.joint_state.velocity[3] == 0)
    recovered = timeopt.timeopt(*inputs(4))
    assert recovered.success.all()


def test_twenty_iteration_trace_retains_feasibility(
    robot: KinematicsCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Longer fixed-budget solves retain feasible outputs and a nonincreasing best cost."""
    timeopt = make_timeopt(robot, 2, False)
    timeopt._prepare(*inputs(2))
    trace = []
    accept = timeopt._solver._search_and_accept

    def record_iteration() -> None:
        accept()
        state = timeopt._solver.state
        trace.append(
            {
                "best_duration": (
                    state[:, _StateOffset.BEST_COST] * state[:, _StateOffset.SPEED_SCALE].rsqrt()
                ).clone(),
                "gap": state[:, _StateOffset.GAP].clone(),
            }
        )

    monkeypatch.setattr(timeopt._solver, "_search_and_accept", record_iteration)
    solved = timeopt._solver.solve(20)
    expected_timing = solved[0].clone()
    timeopt._finish(*solved)
    assert timeopt._result.success.all()
    assert timeopt._result.optimization_ok.all()
    assert len(trace) == 20
    assert torch.isfinite(torch.stack([row["gap"] for row in trace])).all()
    best = torch.stack([row["best_duration"] for row in trace])
    assert (torch.diff(best, dim=0) <= 0).all()
    # Finishing must preserve the prepared seed bounds for another solve.
    torch.testing.assert_close(timeopt._solver.solve(20)[0], expected_timing, rtol=0, atol=0)


def test_ten_iteration_primal_dual_replay(
    robot: KinematicsCfg,
) -> None:
    """Exercise the normal ten-update solver, including graph replay and mixed failures."""
    timeopt = TrajectoryTimeOpt(
        TrajectoryTimeOptCfg(
            robot,
            batch_size=3,
            num_intervals=128,
            max_iterations=10,
            acceleration_limits=torch.tensor([3.0], device="cuda"),
            jerk_limits=torch.tensor([50.0], device="cuda"),
        )
    )
    assert timeopt.config.timing_controls == 32
    controls, start, goal = inputs(3)
    for factor in (1.0, 0.7):
        result = timeopt.timeopt(controls * factor, start, goal * factor)
        assert result.timing_controls.shape == (3, 32)
        assert result.success.all()
        assert result.optimization_ok.all()
        assert (result.duration <= result.uniform_duration * 1.0001).all()
        assert (result.joint_state.velocity.abs() <= 2.0).all()
        assert (result.joint_state.acceleration.abs() <= 3.0).all()
        assert (result.joint_state.jerk.abs() <= 50.0).all()
    controls[1, 3] = torch.nan
    controls[2] = start[2] = goal[2] = 0.0
    result = timeopt.timeopt(controls, start, goal)
    assert result.success.tolist() == [True, False, True]
    assert result.duration[2] == 0
    assert timeopt.timeopt(*inputs(3)).success.all()


@pytest.mark.parametrize("timing_controls", [4, 8, 15, 16, 17, 31, 33, 64, 128])
def test_unsupported_timing_controls(
    robot: KinematicsCfg,
    timing_controls: int,
) -> None:
    """Reject unsupported public shapes before allocating or launching solver kernels."""
    with pytest.raises(ValueError, match="timing_controls must be 32"):
        TrajectoryTimeOpt(
            TrajectoryTimeOptCfg(
                robot,
                timing_controls=timing_controls,
            )
        )


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("iterations", [10, 20])
def test_moving_start_replay(robot: KinematicsCfg, graph: bool, iterations: int) -> None:
    """Preserve moving boundaries, improve valid timing, and replay changed states/dt."""
    timeopt = make_timeopt(robot, 3, graph, iterations)
    controls, position, goal = inputs(3)
    controls[2] = goal[2] = 0
    velocity = torch.tensor([[-0.025], [0.05], [0.0]], device="cuda")
    acceleration = torch.tensor([[0.02], [-0.04], [0.0]], device="cuda")
    knot_dt = torch.tensor([0.15, 0.18, 0.2], device="cuda")
    # Exercise mode changes on the same object as well as normal graph replay.
    static = timeopt.timeopt(controls, position, goal).clone()
    for factor in (1.0, 0.8):
        start = JointState(
            position=position, velocity=velocity * factor, acceleration=acceleration * factor
        )
        dt = knot_dt * factor
        result = timeopt.timeopt(controls, start, goal, knot_dt=dt)
        assert result.success.all(), (result.status, result.optimization_ok)
        assert result.optimization_ok.all()
        assert result.stationary.tolist() == [False, False, True]
        assert result.duration[2] == 0
        assert (result.duration[:2] < result.uniform_duration[:2] * 0.99).all()
        torch.testing.assert_close(result.uniform_duration[:2], 19 * dt[:2])
        expected = (19 * dt).square().reciprocal()[:, None].expand(-1, 2)
        torch.testing.assert_close(result.timing_controls[:2, :2], expected[:2], rtol=2e-6, atol=0)
        for field in ("position", "velocity", "acceleration"):
            torch.testing.assert_close(
                getattr(result.joint_state, field)[:, 0],
                getattr(start, field),
                rtol=1e-4,
                atol=2e-5,
            )
        times = (result.duration[:, None] * torch.linspace(0, 1, 2049, device="cuda")).contiguous()
        sampled = timeopt.sample(times)
        for field in ("position", "velocity", "acceleration"):
            torch.testing.assert_close(
                getattr(sampled, field)[:, 0], getattr(start, field), rtol=1e-4, atol=2e-5
            )
        torch.testing.assert_close(sampled.position[:, -1], goal)
        assert sampled.velocity[:, -1].abs().max() < 1e-5
        assert sampled.acceleration[:, -1].abs().max() < 1e-4
        assert sampled.velocity.abs().max() < 2.0
        assert sampled.acceleration.abs().max() < 3.0
        assert sampled.jerk.abs().max() < 50.0
        torque = 0.35 * sampled.acceleration - 4.905 * sampled.position.sin()
        assert torque.abs().max() < 2.0
    restored = timeopt.timeopt(controls, position, goal)
    torch.testing.assert_close(restored.timing_controls, static.timing_controls, rtol=0, atol=0)


@pytest.mark.parametrize("graph", [False, True])
def test_static_reference_initialization(robot: KinematicsCfg, graph: bool) -> None:
    """Clamp an overfast static reference to feasible timing without changing its path."""
    timeopt = make_timeopt(robot, 2, graph)
    controls, start, goal = inputs(2)
    for intervals in ([0.001, 0.2], [0.2, 0.001]):
        dt = torch.tensor(intervals, device="cuda")
        result = timeopt.timeopt(controls, start, goal, knot_dt=dt)
        assert result.success.all(), (result.status, result.optimization_ok)
        assert result.optimization_ok.all()
        assert (result.duration < result.uniform_duration).all()
        fast = dt.argmin()
        assert result.uniform_duration[fast] > 19 * dt[fast]
        torch.testing.assert_close(result.joint_state.position[:, 0], start)
        torch.testing.assert_close(result.joint_state.position[:, -1], goal)
        assert result.joint_state.velocity[:, 0].abs().max() < 1e-6
        assert result.joint_state.acceleration[:, 0].abs().max() < 1e-5
        assert result.joint_state.velocity.abs().max() < 2.0 * 0.95
        assert result.joint_state.acceleration.abs().max() < 3.0 * 0.95
        assert result.joint_state.jerk.abs().max() < 50.0 * 0.95
        assert result.torque.abs().max() < 2.0 * 0.95


def test_reference_path_matches_trajopt(robot: KinematicsCfg) -> None:
    """Retain TrajOpt's first span and boundary geometry at its original knot interval."""
    timeopt = TrajectoryTimeOpt(
        TrajectoryTimeOptCfg(
            robot,
            batch_size=2,
            num_intervals=19 * 8,
            use_cuda_graph=False,
            acceleration_limits=torch.tensor([3.0], device="cuda"),
            jerk_limits=torch.tensor([50.0], device="cuda"),
        )
    )
    controls, position, goal = inputs(2)
    start = JointState(
        position=position,
        velocity=torch.tensor([[-0.025], [0.05]], device="cuda"),
        acceleration=torch.tensor([[0.02], [-0.04]], device="cuda"),
        jerk=torch.tensor([[0.01], [-0.03]], device="cuda"),
    )
    dt = torch.tensor([0.15, 0.18], device="cuda")
    timeopt.timeopt(controls, start, goal, knot_dt=dt)
    states = [torch.empty((2, 20 * 8 + 1, 1), device="cuda") for _ in range(4)]
    zero = torch.zeros_like(position)
    lanes = torch.arange(2, device="cuda", dtype=torch.int32)
    implicit = torch.ones(2, device="cuda", dtype=torch.uint8)
    trajectory_cu.launch_bspline_interpolation_forward_kernel(
        *states,
        torch.empty_like(dt),
        controls,
        start.position,
        start.velocity,
        start.acceleration,
        start.jerk,
        goal,
        zero,
        zero,
        zero,
        lanes,
        lanes,
        dt / 8,
        implicit,
        2,
        states[0].shape[1],
        1,
        16,
        3,
    )
    for order, expected in enumerate(states):
        actual = timeopt._path[order, :, :153] / (19 * dt[:, None, None]).pow(order)
        expected = expected[:, :153]
        if order == 3:
            # Cubic jerk jumps at knots; native rollout chooses the right side,
            # whereas endpoint/grid rounding may select the left in the timeopt.
            interior = torch.arange(153, device="cuda") % 8 != 0
            actual, expected = actual[:, interior], expected[:, interior]
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=5e-5)


def test_reference_input_failures(robot: KinematicsCfg) -> None:
    """Bad reference intervals are isolated, while infeasible seeds fail the final check."""
    timeopt = make_timeopt(robot, 5, True)
    controls, position, goal = inputs(5)
    start = JointState(position=position, velocity=torch.full_like(position, 0.01))
    with pytest.raises(ValueError, match="requires the original knot_dt"):
        timeopt.timeopt(controls, start, goal)
    with pytest.raises(ValueError, match="knot_dt must be"):
        timeopt.timeopt(controls, start, goal, knot_dt=torch.ones((5, 1), device="cuda"))
    dt = torch.tensor([0.2, 0.0, -0.1, float("nan"), 0.001], device="cuda")
    result = timeopt.timeopt(controls, start, goal, knot_dt=dt)
    assert result.success.tolist() == [True, False, False, False, False]
    assert result.status.tolist() == [0, 3, 3, 3, 2]
    assert torch.isnan(result.joint_state.position[1:]).all()
    assert timeopt.timeopt(controls, start, goal, knot_dt=torch.full_like(dt, 0.2)).success.all()


def test_sampling_input_contract(robot: KinematicsCfg) -> None:
    """Reject missing solutions and empty samples; mask nonfinite query times per sample."""
    timeopt = make_timeopt(robot, 1, True)
    times = torch.zeros((1, 5), device="cuda")
    with pytest.raises(ValueError, match="timeopt.*before.*sample"):
        timeopt.sample(times)
    controls, start, goal = inputs(1)
    result = timeopt.timeopt(controls, start, goal)
    with pytest.raises(ValueError, match="N > 0"):
        timeopt.sample(times[:, :0].contiguous())
    times[0] = torch.tensor([-1, float("nan"), float("inf"), -float("inf"), 10], device="cuda")
    sampled = timeopt.sample(times)
    for field in ("position", "velocity", "acceleration", "jerk"):
        assert torch.isnan(getattr(sampled, field)[:, 1:4]).all()
    torch.testing.assert_close(sampled.position[:, 0], start)
    torch.testing.assert_close(sampled.position[:, -1], goal)
    assert sampled.velocity[:, [0, -1]].count_nonzero() == 0
    # A reset permits a new sample count without discarding the solved trajectory.
    timeopt.reset_cuda_graph()
    sampled = timeopt.sample(torch.stack((result.duration * 0, result.duration), dim=1))
    torch.testing.assert_close(sampled.position[:, 0], start)
    torch.testing.assert_close(sampled.position[:, -1], goal)


def test_timeopt_input_contract(robot: KinematicsCfg) -> None:
    """Reject missing, misordered, incorrectly typed, and noncontiguous input buffers."""
    timeopt = make_timeopt(robot, 1, True)
    controls, start, goal = inputs(1)
    for args, name in (
        ((None, start, goal), "controls"),
        ((controls, None, goal), "start"),
        ((controls, start, None), "goal"),
        ((controls.half(), start, goal), "controls"),
        ((torch.stack((controls, controls), dim=-1)[..., 0], start, goal), "controls"),
        ((controls, start.cpu(), goal), "start"),
    ):
        with pytest.raises(ValueError, match=name):
            timeopt.timeopt(*args)
    dt = torch.full((1,), 0.2, device="cuda")
    state = JointState(position=start, joint_names=["wrong_joint"])
    with pytest.raises(ValueError, match="joint_names"):
        timeopt.timeopt(controls, state, goal, knot_dt=dt)
    state.joint_names = list(robot.kinematics_config.joint_names)
    assert timeopt.timeopt(controls, state, goal, knot_dt=dt).success.all()


def test_float32_buffers_ignore_default_dtype(robot: KinematicsCfg) -> None:
    """The float32 CUDA ABI must hold even when the caller defaults to another dtype."""
    controls, start, goal = inputs(1)
    config = TrajectoryTimeOptCfg(
        robot,
        num_intervals=128,
        use_cuda_graph=False,
        acceleration_limits=torch.tensor([3.0], device="cuda"),
        jerk_limits=torch.tensor([50.0], device="cuda"),
    )
    dt = torch.full((1,), 0.2, device="cuda")
    moving = JointState(position=start, velocity=torch.full_like(start, 0.01))
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        timeopt = TrajectoryTimeOpt(config)
        # Check storage before launching kernels, so a regression cannot corrupt memory.
        for owner in (timeopt, timeopt._solver):
            for value in vars(owner).values():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    assert value.dtype == torch.float32
        result = timeopt.timeopt(controls, moving, goal, knot_dt=dt)
        assert result.success.all()
        sampled = timeopt.sample(result.time.contiguous())
        assert sampled.position.dtype == torch.float32
        torch.testing.assert_close(sampled.velocity[:, 0], moving.velocity, atol=1e-5, rtol=1e-5)
    finally:
        torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("moving", [False, True])
def test_outer_graph_replay(robot: KinematicsCfg, moving: bool) -> None:
    """TimeOpt and sample together in a caller-owned graph with changed input storage values."""
    timeopt = make_timeopt(robot, 1, True)
    controls, position, goal = inputs(1)
    start = (
        JointState(position=position, velocity=torch.full_like(position, 0.01))
        if moving
        else position
    )
    dt = torch.full((1,), 0.2, device="cuda") if moving else None
    times = torch.zeros((1, 7), device="cuda")
    timeopt.timeopt(controls, start, goal, knot_dt=dt)
    timeopt.sample(times)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = timeopt.timeopt(controls, start, goal, knot_dt=dt)
        sampled = timeopt.sample(times)
    for factor in (1.0, 0.8):
        controls.mul_(factor)
        goal.mul_(factor)
        if moving:
            start.velocity.mul_(factor)
            dt.mul_(factor)
        graph.replay()
        assert result.success.all()
        torch.testing.assert_close(sampled.position[:, 0], position, atol=1e-5, rtol=1e-5)
        expected = start.velocity if moving else torch.zeros_like(position)
        torch.testing.assert_close(sampled.velocity[:, 0], expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_torque_margin_preserves_one_sided_bounds(robot: KinematicsCfg, sign: float) -> None:
    """A planning margin must never admit effort outside the requested physical interval."""
    bounds = (1.0, 2.0) if sign > 0 else (-2.0, -1.0)
    timeopt = TrajectoryTimeOpt(
        TrajectoryTimeOptCfg(
            robot,
            num_intervals=128,
            acceleration_limits=torch.tensor([3.0], device="cuda"),
            jerk_limits=torch.tensor([50.0], device="cuda"),
            torque_lower=torch.tensor([bounds[0]], device="cuda"),
            torque_upper=torch.tensor([bounds[1]], device="cuda"),
        )
    )
    # Gravity effort is approximately +/-0.9745: inside the old scaled bounds,
    # but outside the physical interval in either direction.
    position = torch.full((1, 1), -0.2 * sign, device="cuda")
    controls = position[:, None].expand(-1, 16, -1).contiguous()
    result = timeopt.timeopt(controls, position, position)
    assert not result.success.any()
    assert result.status.tolist() == [1]
    tight = (1.0, 1.01) if sign > 0 else (-1.01, -1.0)
    with pytest.raises(ValueError, match="empty torque interval"):
        TrajectoryTimeOpt(
            replace(
                timeopt.config,
                torque_lower=torch.tensor([tight[0]], device="cuda"),
                torque_upper=torch.tensor([tight[1]], device="cuda"),
            )
        )
