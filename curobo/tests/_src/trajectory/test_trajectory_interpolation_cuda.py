# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend import trajectory as trajectory_cuda
from curobo._src.state.state_joint import JointState
from curobo._src.trajectory.trajectory_interpolation import (
    TrajInterpolationType,
    get_batch_interpolated_trajectory,
)
from curobo._src.types.device_cfg import DeviceCfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("implicit_goal", [False, True])
def test_cubic_moving_start_preserves_small_derivatives(implicit_goal: bool) -> None:
    """Preserve the first cubic span at nonzero positions and small knot intervals."""
    options = {"device": "cuda", "dtype": torch.float32}
    start = torch.tensor([[2.5, -2.5]], **options).repeat(3, 1)
    velocity = torch.full_like(start, -0.01)
    acceleration = torch.full_like(start, -0.02)
    jerk = torch.full_like(start, 0.03)
    goal = start + 0.5
    zero = torch.zeros_like(start)
    knot_dt = torch.tensor([0.015, 0.04, 0.12], **options)
    controls = start[:, None] + torch.linspace(0, 1, 16, **options)[None, :, None] * 0.5
    states = [torch.empty((3, 81, 2), **options) for _ in range(4)]
    output_dt = torch.empty_like(knot_dt)
    dt = knot_dt / 4
    indices = torch.arange(3, device="cuda", dtype=torch.int32)
    implicit = torch.full((3,), implicit_goal, device="cuda", dtype=torch.uint8)

    def interpolate() -> None:
        trajectory_cuda.launch_bspline_interpolation_forward_kernel(
            *states,
            output_dt,
            controls,
            start,
            velocity,
            acceleration,
            jerk,
            goal,
            zero,
            zero,
            zero,
            indices,
            indices,
            dt,
            implicit,
            3,
            81,
            2,
            16,
            3,
        )

    time = dt[:, None, None] * torch.arange(4, **options)[None, :, None]
    expected = [
        start[:, None]
        + velocity[:, None] * time
        + acceleration[:, None] * time.square() / 2
        + jerk[:, None] * time.pow(3) / 6,
        velocity[:, None] + acceleration[:, None] * time + jerk[:, None] * time.square() / 2,
        acceleration[:, None] + jerk[:, None] * time,
        jerk[:, None].expand(-1, 4, -1),
    ]
    interpolate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        interpolate()
    for replay in (False, True):
        if replay:
            graph.replay()
        for actual, reference, tolerance in zip(states, expected, (5e-7, 2e-6, 1e-5, 1e-4)):
            torch.testing.assert_close(actual[:, :4], reference, rtol=0, atol=tolerance)
        torch.testing.assert_close(states[0][:, -1], goal, rtol=0, atol=5e-7)
        for state in states[1:]:
            torch.testing.assert_close(state[:, -1], zero, rtol=0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("degree", [3, 4, 5])
@pytest.mark.parametrize("implicit_goal", [False, True])
@pytest.mark.parametrize("moving_goal", [False, True])
def test_bspline_goal_boundary_derivatives(
    degree: int, implicit_goal: bool, moving_goal: bool
) -> None:
    """Recover endpoint derivatives in eager execution and CUDA graph replay."""
    options = {"device": "cuda", "dtype": torch.float32}
    batch_size, num_knots, dof = 3, 16, 2
    horizon = (num_knots + degree + 1) * 4 + 1
    start = torch.tensor([[0.2, -0.3]], **options).repeat(batch_size, 1)
    goal = start + 0.5
    start_derivatives = [torch.full_like(start, value) for value in (0.15, -0.2, 0.35)]
    goal_derivatives = [
        torch.full_like(goal, value if moving_goal else 0.0)
        for value in (0.3, -0.4, 0.5)
    ]
    controls = start[:, None] + torch.linspace(0, 1, num_knots, **options)[None, :, None] * 0.5
    # Replicated goals use the final control rather than the supplied goal state.
    # Zero also isolates this regression from constant-basis cancellation errors.
    controls[:, -1] = 0.0
    states = [torch.empty((batch_size, horizon, dof), **options) for _ in range(4)]
    dt = torch.tensor([0.02, 0.05, 0.1], **options)
    output_dt = torch.empty_like(dt)
    indices = torch.arange(batch_size, device="cuda", dtype=torch.int32)
    implicit = torch.full((batch_size,), implicit_goal, device="cuda", dtype=torch.uint8)

    def interpolate() -> None:
        trajectory_cuda.launch_bspline_interpolation_forward_kernel(
            *states,
            output_dt,
            controls,
            start,
            *start_derivatives,
            goal,
            *goal_derivatives,
            indices,
            indices,
            dt,
            implicit,
            batch_size,
            horizon,
            dof,
            num_knots,
            degree,
        )

    def check_boundaries() -> None:
        expected_start = [start, *start_derivatives]
        expected_goal = [goal, *goal_derivatives] if implicit_goal else [
            controls[:, -1], *[torch.zeros_like(goal) for _ in range(3)]
        ]
        for actual, initial, final, tolerance in zip(
            states, expected_start, expected_goal, (5e-7, 2e-6, 1e-5, 1e-4)
        ):
            torch.testing.assert_close(actual[:, 0], initial, rtol=0, atol=tolerance)
            torch.testing.assert_close(actual[:, -1], final, rtol=0, atol=tolerance)

    interpolate()
    check_boundaries()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        interpolate()
    for state in states:
        state.fill_(float("nan"))
    graph.replay()
    check_boundaries()


def test_linear_interpolation():
    device_cfg = DeviceCfg()

    b, h, dof = 1, 12, 1
    raw_dt = device_cfg.to_device([0.4])
    int_dt = device_cfg.to_device([0.1])
    # initialize raw trajectory:
    in_traj = JointState.zeros((b, h, dof), device_cfg)
    in_traj.position = torch.zeros((b, h, dof), device=device_cfg.device)
    # in_traj.position[:, 1, :] = 0.1

    in_traj.position[:, -2:, :] = 1.0
    # in_traj.position[:, 10, :] = -0.01

    in_traj.position[:, -1, :] = 0.01

    in_traj.dt = raw_dt

    # create max_velocity buffer:
    out_traj_gpu, _ = get_batch_interpolated_trajectory(
        raw_traj=in_traj,
        interpolation_dt=int_dt,
        kind=TrajInterpolationType.LINEAR_CUDA,
    )
    #
    out_traj_gpu = out_traj_gpu.clone()

    out_traj_cpu, _ = get_batch_interpolated_trajectory(
        raw_traj=in_traj,
        interpolation_dt=int_dt,
        kind=TrajInterpolationType.LINEAR,
    )

    assert (
        torch.max(
            torch.abs(out_traj_gpu.position[:, -5:, :] - out_traj_cpu.position[:, -5:, :])
        ).item()
        < 0.001
    )
