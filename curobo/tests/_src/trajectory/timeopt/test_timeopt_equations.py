# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent checks of the timing shape's exact scalar feasibility reduction."""

from __future__ import annotations

import pytest
import torch

from curobo.tests._src.trajectory.timeopt.reference_equations import feasible_scale


def test_scale_obeys_jerk_power_and_asymmetric_torque() -> None:
    """Jerk scales as alpha**1.5; gravity can impose a nonzero lower bound."""
    # Three independent candidates, one sample and one joint each.
    v = torch.tensor([[[0.1]], [[0.1]], [[0.1]]])
    a = torch.zeros_like(v)
    j = torch.tensor([[[8.0]], [[0.0]], [[0.0]]])
    dynamic = torch.tensor([[[0.0]], [[-2.0]], [[0.0]]])
    gravity = torch.tensor([[[0.0]], [[3.0]], [[3.0]]])
    one = torch.ones(1)
    alpha, lower, feasible = feasible_scale(v, a, j, dynamic, gravity, one, one, one, -one, one)
    torch.testing.assert_close(alpha[:2], torch.tensor([0.25, 2.0]))
    torch.testing.assert_close(lower[:2], torch.tensor([0.0, 1.0]))
    assert feasible.tolist() == [True, True, False]
    assert torch.all(j[:2].abs() * alpha[:2, None, None].pow(1.5) <= 1.0)
    torque = gravity[:2] + alpha[:2, None, None] * dynamic[:2]
    assert torch.all((torque >= -1) & (torque <= 1))


def test_scale_rejects_empty_intersection_and_nonfinite_data() -> None:
    """Slowing below the torque lower bound is infeasible; NaNs fail their own lane."""
    v = torch.tensor([[[2.0]], [[float("nan")]], [[0.5]]])
    zero = torch.zeros_like(v)
    dynamic = torch.tensor([[[-2.0]], [[0.0]], [[0.0]]])
    gravity = torch.tensor([[[3.0]], [[0.0]], [[0.0]]])
    one = torch.ones(1)
    alpha, lower, feasible = feasible_scale(
        v, zero, zero, dynamic, gravity, one, one, one, -one, one
    )
    assert feasible.tolist() == [False, False, True]
    assert alpha[0] < lower[0]
    torch.testing.assert_close(alpha[2], torch.tensor(4.0))


@pytest.mark.parametrize("kind", ["velocity", "acceleration", "jerk", "torque"])
def test_scale_gradient_matches_finite_difference(kind: str) -> None:
    """Differentiate the active bound away from ties, using only float32."""

    def value(parameter: torch.Tensor) -> torch.Tensor:
        z = parameter.reshape(1, 1, 1)
        zero, one = torch.zeros_like(z), torch.ones(1)
        fields = [zero, zero, zero, zero]
        fields[["velocity", "acceleration", "jerk", "torque"].index(kind)] = z
        return feasible_scale(*fields, zero, one, one, one, -one, one)[0].sum()

    p = torch.tensor(2.0, requires_grad=True)
    analytic = torch.autograd.grad(value(p), p)[0]
    h = 0.002
    numerical = (value(p.detach() + h) - value(p.detach() - h)) / (2 * h)
    torch.testing.assert_close(analytic, numerical, rtol=2e-3, atol=1e-4)
