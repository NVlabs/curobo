# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity of C++ fused gradients with explicit assembly and shared B-spline backward."""

from pathlib import Path
from typing import Any

import pytest
import torch
from cuda.core import ObjectCode, launch

from cpp_runtime.robot.tools.export_robot_aot import compile_trajopt_gradient_kernels
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.trajectory import (
    launch_bspline_interpolation_backward_kernel,
)
from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import BSplineLaunchCfg


@pytest.fixture(scope="module")
def gradient_kernels(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Compile the same AOT specializations used by the native exporter."""
    directory: Path = tmp_path_factory.mktemp("trajectory_gradient")
    major, minor = torch.cuda.get_device_capability()
    metadata = compile_trajopt_gradient_kernels(directory, f"sm_{major}{minor}")
    return {
        name: ObjectCode.from_cubin(str(directory / item["cubin"])).get_kernel(item["symbol"])
        for name, item in metadata.items()
    }


@pytest.mark.parametrize("use_effort", [False, True])
@pytest.mark.parametrize(
    "batch,dof,knots,interpolation", [(1, 2, 12, 1), (4, 7, 16, 4), (12, 7, 16, 8), (3, 12, 17, 4)]
)
def test_fused_trajectory_gradient(
    gradient_kernels: dict[str, Any],
    use_effort: bool,
    batch: int,
    dof: int,
    knots: int,
    interpolation: int,
) -> None:
    """Preserve assembly order, endpoint replication, goal masking, and dt indirection."""
    torch.manual_seed(17)
    horizon = (knots + 4) * interpolation
    shape = (batch, horizon + 1, dof)
    base = [torch.randn(shape, device="cuda", dtype=torch.float32) for _ in range(4)]
    collision = torch.randn_like(base[0])
    effort = [torch.randn_like(base[0]) for _ in range(3)]
    if not use_effort:
        for value in effort:
            value.fill_(float("nan"))
    assembled = [base[0] + collision, base[1].clone(), base[2].clone(), base[3]]
    if use_effort:
        assembled[:3] = [value + term for value, term in zip(assembled[:3], effort)]
    dt = torch.linspace(0.03, 0.13, batch, device="cuda", dtype=torch.float32)
    indices = torch.arange(batch - 1, -1, -1, device="cuda", dtype=torch.int32)
    implicit = (torch.arange(batch, device="cuda") % 2).to(torch.uint8)
    result = torch.empty((batch, knots, dof), device="cuda", dtype=torch.float32)
    reference = torch.empty_like(result)
    kernel = gradient_kernels[
        "bspline_backward_effort" if use_effort else "bspline_backward"
    ]
    cache = get_runtime().get_cuda_core_cache()
    stream = cache.get_stream_wrapper(torch.cuda.current_stream())
    config = BSplineLaunchCfg.calculate_backward_config(batch, dof, knots, horizon, 3)

    def evaluate() -> None:
        launch_bspline_interpolation_backward_kernel(
            reference,
            *assembled,
            dt,
            indices,
            implicit,
            batch,
            horizon + 1,
            dof,
            knots,
            3,
            False,
        )
        launch(
            stream,
            config,
            kernel,
            result.data_ptr(),
            *(value.data_ptr() for value in base),
            dt.data_ptr(),
            indices.data_ptr(),
            implicit.data_ptr(),
            batch,
            horizon,
            dof,
            knots,
            collision.data_ptr(),
            *(value.data_ptr() for value in effort),
        )

    evaluate()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        evaluate()
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("use_pose", [False, True])
def test_cost_reduction(use_pose: bool) -> None:
    """Reduce all cost terms without modifying inputs, including graph replay."""
    import warp as wp

    from cpp_runtime.robot.kernels.trajopt import create_trajopt_runtime_kernels
    from curobo._src.util.warp import init_warp

    init_warp()
    torch.manual_seed(23)
    batch, dof, horizon, tools = 4, 7, 81, 2
    inputs = [
        torch.randn(batch * dof * horizon, device="cuda", dtype=torch.float32),
        torch.randn(batch * horizon, device="cuda", dtype=torch.float32),
        torch.randn(batch * horizon, device="cuda", dtype=torch.float32),
        torch.randn(batch * horizon * tools * 2, device="cuda", dtype=torch.float32),
    ]
    original = [value.clone() for value in inputs]
    reference = inputs[0].view(batch, horizon, dof).sum(dim=(1, 2))
    reference += (inputs[1] + inputs[2]).view(batch, horizon).sum(dim=1)
    if use_pose:
        reference += inputs[3].view(batch, -1).sum(dim=1)
    costs = torch.empty(batch, device="cuda", dtype=torch.float32)
    kernel = create_trajopt_runtime_kernels(dof, 16, horizon, 1000, batch, 3)[
        "reduce_rollout_cost"
    ]

    def evaluate() -> None:
        wp.launch(
            kernel,
            dim=(batch, 128),
            block_dim=128,
            inputs=[wp.from_torch(value) for value in inputs]
            + [tools, wp.uint8(use_pose), wp.from_torch(costs)],
            stream=wp.stream_from_torch(torch.cuda.current_stream()),
        )

    evaluate()
    torch.testing.assert_close(costs, reference, rtol=1e-5, atol=1e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        evaluate()
    for _ in range(2):
        costs.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(costs, reference, rtol=1e-5, atol=1e-5)
    for value, expected in zip(inputs, original):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
