# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytic physical-time and ownership contracts for complete splines."""

import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from curobo.types import SplineTrajectory, TimeOptTiming, UniformTiming


def linear_spline(device: str = "cuda") -> SplineTrajectory:
    """Four spans with q(s)=1+4s."""
    points = torch.arange(7, device=device, dtype=torch.float32).reshape(1, 7, 1)
    return SplineTrajectory(points, ["joint"], UniformTiming(points.new_tensor([0.1])))


def test_uniform_analytic_and_exact_endpoint() -> None:
    """Return uniform physical derivatives and both endpoints at actual dt."""
    spline = linear_spline()
    states = spline.interpolate(0.03)
    assert states.position.shape == (1, 15, 1)
    time = torch.arange(15, device="cuda", dtype=torch.float32) * states.dt
    torch.testing.assert_close(states.position[0, :, 0], 1 + 10 * time)
    torch.testing.assert_close(states.velocity, torch.full_like(states.velocity, 10))
    torch.testing.assert_close(
        states.acceleration, torch.zeros_like(states.acceleration), atol=1e-3, rtol=0
    )
    torch.testing.assert_close(states.jerk, torch.zeros_like(states.jerk), atol=1e-3, rtol=0)
    assert float(states.dt.item()) <= 0.03
    assert states.joint_names == ["joint"]


def test_timeopt_nonconstant_analytic() -> None:
    """For x(s)=1+s, recover analytic progress, velocity, and acceleration."""
    spline = linear_spline()
    device = spline.control_points.device
    breaks = torch.linspace(0, 1, 30, device=device, dtype=torch.float32)
    knots = torch.cat((breaks.new_zeros(3), breaks, breaks.new_ones(3)))
    greville = torch.stack([knots[i + 1 : i + 4].mean() for i in range(32)])
    s = torch.linspace(0, 1, 512, device=device, dtype=torch.float32)
    spline.timing = TimeOptTiming(
        (1 + greville).unsqueeze(0), (2 * (torch.sqrt(1 + s) - 1)).unsqueeze(0)
    )
    states = spline.interpolate(0.01)
    time = torch.arange(states.position.shape[1], device=device, dtype=torch.float32) * states.dt
    progress = (1 + time / 2).square() - 1
    torch.testing.assert_close(states.position[0, :, 0], 1 + 4 * progress, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(states.velocity[0, :, 0], 4 * (1 + time / 2), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        states.acceleration, torch.full_like(states.acceleration, 2), atol=2e-4, rtol=2e-4
    )
    torch.testing.assert_close(states.jerk, torch.zeros_like(states.jerk), atol=2e-3, rtol=0)


@pytest.mark.parametrize("retimed", [False, True])
def test_clone_transfer_and_names(retimed: bool) -> None:
    """Copy every payload independently while same-device to may share storage."""
    spline = linear_spline()
    if retimed:
        spline.timing = TimeOptTiming(
            torch.ones(1, 32, device="cuda"), torch.linspace(0, 1, 33, device="cuda")[None]
        )
    clone = spline.clone()
    host = spline.to("cpu")
    roundtrip = host.to("cuda")
    assert spline.to("cuda") is spline
    spline.control_points.zero_()
    spline.joint_names[0] = "changed"
    torch.testing.assert_close(clone.control_points.cpu(), host.control_points)
    torch.testing.assert_close(roundtrip.control_points.cpu(), host.control_points)
    assert clone.joint_names == host.joint_names == ["joint"]
    if retimed:
        spline.timing.controls.zero_()
        spline.timing.time.zero_()
        assert bool((clone.timing.controls == 1).all())
        assert float(clone.duration.item()) == 1
    else:
        spline.timing.knot_dt.zero_()
        assert float(clone.duration.item()) > 0


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf")])
def test_invalid_dt(dt: float) -> None:
    """Reject invalid intervals before allocation."""
    with pytest.raises(ValueError, match="dt must"):
        linear_spline().interpolate(dt)


def test_stationary_zero_duration() -> None:
    """A zero-duration TimeOpt path has one state and no sample interval."""
    points = torch.ones(1, 7, 1, device="cuda", dtype=torch.float32)
    spline = SplineTrajectory(
        points, ["joint"], TimeOptTiming(points.new_zeros(1, 32), points.new_zeros(1, 33))
    )
    state = spline.interpolate(0.01)
    assert state.position.shape == (1, 1, 1) and state.dt is None
    assert float(state.position.item()) == 1
    assert (
        float(state.velocity.item())
        == float(state.acceleration.item())
        == float(state.jerk.item())
        == 0
    )


def test_cpu_storage_and_batch_rejection() -> None:
    """CPU supports transport; interpolation requires a selected CUDA lane."""
    with pytest.raises(ValueError, match="requires CUDA"):
        linear_spline("cpu").interpolate(0.01)
    points = torch.zeros(2, 7, 1, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="single trajectory"):
        SplineTrajectory(points, ["joint"], UniformTiming(points.new_ones(2))).interpolate(0.01)


@pytest.mark.parametrize("retimed", [False, True])
def test_planner_export_matches_existing_sampler(retimed: bool) -> None:
    """A complete selected plan reproduces existing sampled motion and survives reuse."""
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState

    planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot="franka.yml",
            enable_timeopt=retimed,
            num_trajopt_seeds=2,
            use_cuda_graph=True,
        )
    )
    try:
        start = JointState.from_position(
            planner.default_joint_state.position[None], joint_names=planner.joint_names
        )
        goal = start.clone()
        goal.position[:, 0] += 0.1
        result = planner.plan_cspace(goal, start, max_attempts=1)
        assert result is not None and bool(result.success.all())
        spline = result.get_trajectory()
        reference = (result.get_interpolated_plan() if retimed else result.js_solution).reorder(
            spline.joint_names
        )
        dt = float(reference.dt.reshape(-1)[0].item())
        state = spline.interpolate(dt * 1.00001)
        assert state.position.shape[-2] == reference.position.shape[-2]
        expected = reference.position.reshape(1, -1, len(spline.joint_names))
        torch.testing.assert_close(state.position, expected, atol=2e-4, rtol=2e-4)
        saved = spline.clone()
        goal.position[:, 0] += 0.1
        planner.plan_cspace(goal, start, max_attempts=1)
        torch.testing.assert_close(saved.control_points, spline.control_points)
    finally:
        planner.destroy()


def test_result_selection_clone_and_batch_replacement() -> None:
    """Keep seed identity, control points, and timing together through result copies."""
    from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult

    points = (
        torch.arange(4, device="cuda", dtype=torch.float32)
        .view(4, 1, 1)
        .expand(4, 7, 1)
        .contiguous()
    )
    timing = UniformTiming(torch.arange(1, 5, device="cuda", dtype=torch.float32))
    result = TrajOptSolverResult(
        success=torch.ones(2, 2, device="cuda", dtype=torch.bool),
        spline_trajectory=SplineTrajectory(points, ["joint"], timing),
    )
    with pytest.raises(ValueError, match="batch_index"):
        result.get_trajectory()
    selected = result.get_trajectory(1, 1)
    assert float(selected.control_points[0, 0, 0]) == 3
    assert float(selected.timing.knot_dt.item()) == 4
    other = result.clone()
    other.spline_trajectory.control_points.add_(10)
    other.spline_trajectory.timing.knot_dt.add_(10)
    result.copy_at_batch_indices(other, torch.tensor([False, True], device="cuda"))
    assert float(result.get_trajectory(0, 0).control_points[0, 0, 0]) == 0
    assert float(result.get_trajectory(1, 1).control_points[0, 0, 0]) == 13
    assert float(result.get_trajectory(1, 1).timing.knot_dt.item()) == 14
    result.success[1, 1] = False
    with pytest.raises(ValueError, match="unsuccessful"):
        result.get_trajectory(1, 1)


def test_grasp_phase_availability_and_retention() -> None:
    """Omitted and failed phases are distinct, and earlier phases own their data."""
    from curobo.motion_planner import GraspPlanResult

    result = GraspPlanResult()
    assert result.get_grasp_trajectory() is None
    result._grasp_attempted = True
    with pytest.raises(ValueError, match="did not succeed"):
        result.get_grasp_trajectory()
    source = linear_spline()
    result.approach_spline = source.clone()
    source.control_points.zero_()
    assert float(result.get_approach_trajectory().control_points[0, -1, 0]) == 6


@pytest.mark.parametrize("retimed", [False, True])
def test_ranked_selection_and_success_replacement(retimed: bool) -> None:
    """Ranking and successful-lane replacement preserve the corresponding time law."""
    from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult

    points = (
        torch.arange(4, device="cuda", dtype=torch.float32)[:, None, None]
        .expand(4, 7, 1)
        .contiguous()
    )
    duration = torch.arange(1, 5, device="cuda", dtype=torch.float32)
    timing = (
        TimeOptTiming(
            torch.ones(4, 32, device="cuda"),
            duration[:, None] * torch.linspace(0, 1, 33, device="cuda"),
        )
        if retimed
        else UniformTiming(duration)
    )
    solution = points.reshape(2, 2, 7, 1).clone()
    result = TrajOptSolverResult(
        success=torch.ones(2, 2, device="cuda", dtype=torch.bool),
        solution=solution,
        optimized_seeds=solution.clone(),
        position_error=torch.zeros(2, 2, device="cuda"),
        rotation_error=torch.zeros(2, 2, device="cuda"),
        seed_cost=duration.reshape(2, 2),
        total_cost_reshaped=duration.reshape(2, 2),
        seed_rank=torch.tensor([[1, 0], [0, 1]], device="cuda"),
        batch_size=2,
        num_seeds=2,
        spline_trajectory=SplineTrajectory(points, ["joint"], timing),
    )
    ranked = result.get_topk_seeds(1)
    assert float(ranked.get_trajectory(0).control_points[0, 0, 0]) == 1
    assert float(ranked.get_trajectory(1).control_points[0, 0, 0]) == 2
    assert float(ranked.get_trajectory(0).duration.item()) == (2 if retimed else 8)
    replacement = result.clone()
    replacement.success[:] = False
    replacement.success[0, 1] = True
    replacement.spline_trajectory.control_points.add_(10)
    result.copy_successful_solutions(replacement)
    assert float(result.get_trajectory(0, 1).control_points[0, 0, 0]) == 11
    assert float(result.get_trajectory(1, 0).control_points[0, 0, 0]) == 2


def test_native_python_interpolation_parity(tmp_path: Path) -> None:
    """Native and Python complete values evaluate the same two physical time laws."""
    executable = os.environ.get("CUROBO_SPLINE_TEST_EXECUTABLE")
    if executable is None:
        pytest.skip("set CUROBO_SPLINE_TEST_EXECUTABLE to the native spline test")
    output = tmp_path / "spline.bin"
    subprocess.run([executable, str(output)], check=True, capture_output=True)
    spline = linear_spline()
    device = spline.control_points.device
    # Match native float32 table construction exactly.
    knots = np.clip((np.arange(36, dtype=np.float32) - 3) / np.float32(29), 0, 1)
    controls = np.array(
        [1 + (knots[i + 1] + knots[i + 2] + knots[i + 3]) / np.float32(3) for i in range(32)],
        dtype=np.float32,
    )
    times = 2 * (np.sqrt(1 + np.arange(512, dtype=np.float32) / np.float32(511)) - 1)
    with output.open("rb") as source:
        for retimed, dt in ((False, 0.03), (True, 0.01)):
            if retimed:
                spline.timing = TimeOptTiming(
                    torch.as_tensor(controls, device=device)[None],
                    torch.as_tensor(times, device=device)[None],
                )
            count, actual_dt = struct.unpack("if", source.read(8))
            expected = torch.from_numpy(
                np.frombuffer(source.read(4 * count * 4), dtype=np.float32).copy()
            ).reshape(4, count)
            state = spline.interpolate(float(np.float32(dt)))
            assert state.position.shape[1] == count
            assert float(state.dt.item()) == pytest.approx(actual_dt)
            for derivative, name in enumerate(("position", "velocity", "acceleration", "jerk")):
                torch.testing.assert_close(
                    getattr(state, name)[0, :, 0].cpu(),
                    expected[derivative],
                    atol=3e-3 if derivative == 3 else 3e-4,
                    rtol=2e-5,
                )
        assert source.read() == b""
