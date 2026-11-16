# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native validation with an MPC-owned sample count."""

import pytest
import torch
import warp as wp

from cpp_runtime.robot.kernels.trajopt import create_trajopt_runtime_kernels
from curobo._src.util.warp import init_warp


@pytest.mark.parametrize("horizon", [7, 41])
def test_validation_uses_runtime_sample_count(horizon: int) -> None:
    """Attribute violations to the correct seed when capacity differs from export."""
    init_warp()
    batch, dof = 3, 2
    states = [torch.zeros((batch, horizon, dof), device="cuda") for _ in range(5)]
    states[0][1, 0, 0] = 2.0
    states[1][2, -1, 1] = 2.0
    costs = [torch.zeros(batch * horizon, device="cuda") for _ in range(2)]
    lower = torch.full((dof,), -1.0, device="cuda")
    upper = torch.ones(dof, device="cuda")
    violation = torch.zeros(batch, dtype=torch.int32, device="cuda")
    kernel = create_trajopt_runtime_kernels(dof, 16, 81, 32, batch, 3)[
        "validate_interpolated"
    ]

    def evaluate() -> None:
        violation.zero_()
        wp.launch(
            kernel,
            dim=batch * horizon * dof,
            inputs=[wp.from_torch(value.reshape(-1)) for value in states + costs]
            + [wp.from_torch(value) for value in [lower, upper] * 5]
            + [wp.from_torch(violation), wp.float32(0.0), wp.uint8(0)],
            stream=wp.stream_from_torch(torch.cuda.current_stream()),
        )

    evaluate()
    expected = torch.tensor([0, 1, 1], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(violation, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        evaluate()
    graph.replay()
    torch.testing.assert_close(violation, expected)
