# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate native LM rejection and retained objectives under CUDA graph replay."""

from __future__ import annotations

import pytest
import torch
import warp as wp

from cpp_runtime.robot.kernels.ik import create_seed_ik_kernels
from curobo._src.util.warp import init_warp


@pytest.mark.parametrize("dof", [7, 35])
def test_lm_retains_rejected_state(dof: int) -> None:
    """Worsening or invalid predictions retain every field, including the objective."""
    init_warp()
    kernels = create_seed_ik_kernels(dof, 1, 4)
    batch = 7
    residuals = kernels["num_residuals"]
    position = torch.zeros((batch, dof), device="cuda", dtype=torch.float32)
    candidate = torch.ones_like(position)
    jacobian = torch.zeros((batch, residuals, dof), device="cuda", dtype=torch.float32)
    candidate_jacobian = torch.ones_like(jacobian)
    gradient = torch.zeros_like(position)
    candidate_gradient = torch.ones_like(position)
    error = torch.ones(batch, device="cuda", dtype=torch.float32)
    candidate_error = torch.tensor(
        [0.5, 2.0, 2.0, 2.0, 2.0, 2.0, float("nan")], device="cuda", dtype=torch.float32
    )
    predicted = torch.tensor(
        [0.5, 0.5, -1.0, 0.0, float("inf"), float("nan"), 0.5],
        device="cuda", dtype=torch.float32,
    )
    damping = torch.full_like(error, 0.1)
    arrays = [position, candidate, jacobian, candidate_jacobian, gradient, candidate_gradient,
              error, candidate_error, predicted, damping]
    arguments = [wp.from_torch(value) for value in arrays] + [1e-3, 2.0, 1e-5, 1e10]
    error_arguments = [wp.from_torch(value) for value in (error, candidate_error, predicted)]
    error_arguments.append(1e-3)
    def launch() -> None:
        stream = wp.stream_from_torch(torch.cuda.current_stream())
        wp.launch(kernels["update"], dim=(batch, dof), inputs=arguments, stream=stream)
        wp.launch(kernels["update_error"], dim=batch, inputs=error_arguments, stream=stream)

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(3):
        position.zero_()
        jacobian.zero_()
        gradient.zero_()
        error.fill_(1.0)
        damping.fill_(0.1)
        graph.replay()
        for value in (position, gradient, jacobian):
            torch.testing.assert_close(value[0], torch.ones_like(value[0]))
            torch.testing.assert_close(value[1:], torch.zeros_like(value[1:]))
        torch.testing.assert_close(error, torch.tensor(
            [0.5, 1, 1, 1, 1, 1, 1], device="cuda", dtype=torch.float32
        ))
        torch.testing.assert_close(damping, torch.tensor(
            [0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], device="cuda", dtype=torch.float32
        ))


@pytest.mark.parametrize("weight", [0.0, 1.0, 2.5])
@pytest.mark.parametrize("include_limits", [False, True])
def test_native_joint_limit_cost_matches_derivatives(weight: float, include_limits: bool) -> None:
    """Native LM evaluates the same least-squares objective as its gradient and Jacobian."""
    init_warp()
    dof, batch = 7, 5
    kernels = create_seed_ik_kernels(dof, 1, 4)
    position = torch.tensor([-1.25, -1.0, 0.0, 1.0, 1.5], device="cuda", dtype=torch.float32)
    position = position[:, None].expand(batch, dof).contiguous()
    lower, upper = torch.full_like(position[0], -1.0), torch.full_like(position[0], 1.0)
    current_position = torch.zeros((batch, 3), device="cuda", dtype=torch.float32)
    quaternion = torch.zeros((batch, 4), device="cuda", dtype=torch.float32)
    quaternion[:, 0] = 1
    selected = torch.zeros(batch, device="cuda", dtype=torch.int32)
    tracking = torch.ones(1, device="cuda", dtype=torch.int32)
    raw_jacobian = torch.zeros((batch, 6, dof), device="cuda", dtype=torch.float32)
    jacobian = torch.zeros(
        (batch, kernels["num_residuals"], dof), device="cuda", dtype=torch.float32
    )
    gradient = torch.zeros_like(position)
    cost, pe, qe = (torch.zeros(batch, device="cuda", dtype=torch.float32) for _ in range(3))
    arguments = [
        wp.from_torch(current_position, dtype=wp.vec3),
        wp.from_torch(quaternion, dtype=wp.vec4),
        wp.from_torch(current_position, dtype=wp.vec3),
        wp.from_torch(quaternion, dtype=wp.vec4),
        *[wp.from_torch(value) for value in
          (selected, raw_jacobian, position, lower, upper, jacobian, gradient, cost, pe, qe)],
        1, 1, 0.0, 0.0, weight, wp.uint8(include_limits), wp.from_torch(tracking),
    ]

    def launch() -> None:
        wp.launch(
            kernels["error"], dim=(batch, 128), block_dim=128, inputs=arguments,
            stream=wp.stream_from_torch(torch.cuda.current_stream()),
        )

    launch()
    reference_position = position.detach().clone().requires_grad_(True)
    residual = weight * (
        torch.clamp(lower - reference_position, min=0)
        + torch.clamp(reference_position - upper, min=0)
    )
    reference_cost = 0.5 * residual.square().sum(dim=-1) * int(include_limits)
    reference_gradient = torch.autograd.grad(reference_cost.sum(), reference_position)[0]
    torch.testing.assert_close(cost, reference_cost)
    torch.testing.assert_close(gradient, reference_gradient)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    cost.fill_(-1)
    gradient.fill_(-1)
    graph.replay()
    torch.testing.assert_close(cost, reference_cost)
    torch.testing.assert_close(gradient, reference_gradient)
