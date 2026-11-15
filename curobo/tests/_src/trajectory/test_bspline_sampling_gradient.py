# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check spline VJPs for warp-dividing and non-warp-dividing sample counts."""

import pytest
import torch

from curobo._src.curobolib.backends.cuda_core_backend.trajectory import (
    launch_bspline_interpolation_backward_kernel,
    launch_bspline_interpolation_forward_kernel,
)


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5, 6, 7, 8, 16, 32])
def test_sampling_gradient(steps: int) -> None:
    """Match a forward finite difference, including partial interpolation warps."""
    torch.manual_seed(33)
    batch, dof, knots = 2, 2, 16
    horizon = 20 * steps + 1
    controls = torch.randn(batch, knots, dof, device="cuda", dtype=torch.float32) * 0.1
    direction = torch.randn_like(controls)
    boundaries = torch.zeros(batch, dof, device="cuda", dtype=torch.float32)
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    implicit = torch.zeros(batch, device="cuda", dtype=torch.uint8)
    dt = torch.full((batch,), 0.6 / steps, device="cuda", dtype=torch.float32)
    states = [
        torch.empty(batch, horizon, dof, device="cuda", dtype=torch.float32) for _ in range(4)
    ]
    state_dt = torch.empty(batch, horizon, device="cuda", dtype=torch.float32)
    weights = torch.randn_like(states[0])
    zero = torch.zeros_like(weights)
    gradient = torch.full_like(controls, float("nan"))

    def loss(value: torch.Tensor) -> torch.Tensor:
        launch_bspline_interpolation_forward_kernel(
            *states,
            state_dt,
            value,
            *([boundaries] * 8),
            indices,
            indices,
            dt,
            implicit,
            batch,
            horizon,
            dof,
            knots,
            3,
        )
        return (states[0] * weights).sum()

    loss(controls)
    launch_bspline_interpolation_backward_kernel(
        gradient,
        weights,
        zero,
        zero,
        zero,
        dt,
        indices,
        implicit,
        batch,
        horizon,
        dof,
        knots,
        3,
        False,
    )
    assert bool(torch.isfinite(gradient).all())
    analytic = (gradient * direction).sum()
    epsilon = 1e-3
    numeric = (loss(controls + epsilon * direction) - loss(controls - epsilon * direction)) / (
        2 * epsilon
    )
    torch.testing.assert_close(analytic, numeric, rtol=2e-3, atol=5e-4)
