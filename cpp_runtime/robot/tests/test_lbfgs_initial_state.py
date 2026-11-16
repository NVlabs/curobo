# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate native optimizer initialization and its graph-replay contract."""

from __future__ import annotations

import pytest
import torch
import warp as wp

from cpp_runtime.robot.kernels.lbfgs import create_lbfgs_runtime_kernels
from curobo._src.util.warp import init_warp


@pytest.mark.parametrize("dimension", [7, 31, 32, 33, 112, 448, 2049])
@pytest.mark.parametrize("clear_history", [False, True])
def test_initial_state(dimension: int, clear_history: bool) -> None:
    """Copy every state field, limit the first direction, and preserve optional history."""
    init_warp()
    torch.manual_seed(19)
    batch = 4
    action = torch.randn((batch, dimension), device="cuda", dtype=torch.float32)
    gradient = torch.randn_like(action) * 24.0
    gradient[0].zero_()
    cost = torch.arange(batch, device="cuda", dtype=torch.float32)
    limits = torch.linspace(0.05, 1.0, dimension, device="cuda", dtype=torch.float32)
    limits[::3] = 0.0
    limits[1::5] = -1.0
    vector_names = (
        "action",
        "gradient",
        "exploration_action",
        "exploration_gradient",
        "best_action",
        "step_direction",
        "reference_action",
        "reference_gradient",
    )
    outputs = {name: torch.full_like(action, -99.0) for name in vector_names}
    for name in ("cost", "exploration_cost", "best_cost", "beta"):
        outputs[name] = torch.full_like(cost, -99.0)
    for name in ("best_iteration", "current_iteration", "history_head", "history_count"):
        outputs[name] = torch.full((batch,), 3, device="cuda", dtype=torch.int32)
    outputs["converged"] = torch.ones(batch, device="cuda", dtype=torch.uint8)
    outputs["curvature_update_valid"] = torch.ones(batch, device="cuda", dtype=torch.bool)
    names = (
        "action",
        "gradient",
        "cost",
        "exploration_action",
        "exploration_gradient",
        "exploration_cost",
        "best_action",
        "best_cost",
        "best_iteration",
        "current_iteration",
        "converged",
        "step_direction",
        "reference_action",
        "reference_gradient",
        "history_head",
        "history_count",
        "curvature_update_valid",
    )
    kernel = create_lbfgs_runtime_kernels(dimension, 3, "test")["prepare_initial_state"]
    arguments = [wp.from_torch(value) for value in [action, cost, gradient]]
    arguments += [wp.from_torch(outputs[name]) for name in names]
    arguments += [
        wp.from_torch(limits),
        0.125,
        wp.uint8(clear_history),
        wp.from_torch(outputs["beta"]),
    ]

    def initialize() -> None:
        wp.launch(
            kernel,
            dim=(batch, 32),
            block_dim=32,
            inputs=arguments,
            stream=wp.stream_from_torch(torch.cuda.current_stream()),
        )

    initialize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        initialize()
    for replay in range(3):
        action.add_(0.25)
        gradient.mul_(-0.5)
        cost.add_(1.0)
        outputs["history_head"].fill_(3 + replay)
        outputs["history_count"].fill_(5 + replay)
        graph.replay()
        direction = -0.125 * gradient
        ratios = torch.where(limits > 0, direction.abs() / limits, 0.0)
        scale = ratios.amax(dim=1, keepdim=True).clamp_min(1.0)
        for name in vector_names:
            expected = gradient if "gradient" in name else action
            if name == "step_direction":
                torch.testing.assert_close(
                    outputs[name], direction / scale, atol=1.0e-6, rtol=1.0e-6
                )
            else:
                torch.testing.assert_close(outputs[name], expected, atol=0, rtol=0)
        for name in ("cost", "exploration_cost", "best_cost"):
            torch.testing.assert_close(outputs[name], cost, atol=0, rtol=0)
        for name in ("beta", "best_iteration", "current_iteration"):
            assert torch.all(outputs[name] == 1)
        assert torch.all(outputs["converged"] == 0)
        assert not torch.any(outputs["curvature_update_valid"])
        assert torch.all(outputs["history_head"] == (0 if clear_history else 3 + replay))
        assert torch.all(outputs["history_count"] == (0 if clear_history else 5 + replay))
