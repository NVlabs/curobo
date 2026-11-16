# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check exported CUDA cost sums, fixed layouts, and graph replay."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from cuda.core import LaunchConfig, ObjectCode, launch

from cpp_runtime.robot.tools.export_robot_aot import compile_cost_reduction_kernels
from curobo._src.context import get_runtime

LAYOUTS = {
    "single": ((1, 1),),
    "broadcast": ((17, 23), (65, 0)),
    "empty": ((0, 0), (0, 0)),
    "without_pose": ((567, 567), (81, 81), (81, 81)),
    "with_pose": ((567, 567), (81, 81), (81, 81), (324, 324)),
    "many": ((8, 11), (81, 81), (129, 133), (9, 0), (257, 260), (1, 2), (0, 0)),
    "long": ((28791, 28791), (4113, 4113), (4113, 4113), (8226, 8226)),
}


@pytest.fixture(scope="module")
def cost_kernels(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Load the same cubin and argument ABI used by the native runtime."""
    directory: Path = tmp_path_factory.mktemp("cost_reduction")
    major, minor = torch.cuda.get_device_capability()
    named_layouts = {
        name: {f"term_{i}": layout for i, layout in enumerate(terms)}
        for name, terms in LAYOUTS.items()
    }
    metadata = compile_cost_reduction_kernels(directory, f"sm_{major}{minor}", named_layouts)
    assert all(item["cost_terms"] == list(named_layouts[name]) for name, item in metadata.items())
    return {
        name: ObjectCode.from_cubin(str(directory / item["cubin"])).get_kernel(item["symbol"])
        for name, item in metadata.items()
    }


@pytest.mark.parametrize("name", LAYOUTS)
@pytest.mark.parametrize("batch", [1, 4, 12])
def test_cost_reduction(cost_kernels: dict[str, Any], name: str, batch: int) -> None:
    """Preserve strided/broadcast trials, empty terms, inputs, and graph replay."""
    torch.manual_seed(23)
    inputs = []
    reference = torch.zeros(batch, device="cuda", dtype=torch.float32)
    for length, stride in LAYOUTS[name]:
        value = torch.full(((batch - 1) * stride + length,), float("nan"), device="cuda")
        for row in range(batch):
            sample = (
                torch.randint(-8, 9, (length,), device="cuda").to(torch.float32)
                if name == "with_pose"
                else torch.rand(length, device="cuda")
            )
            value[row * stride : row * stride + length] = sample
        reference += value.as_strided((batch, length), (stride, 1)).sum(1)
        inputs.append(value)
    original = [value.clone() for value in inputs]
    output = torch.full((batch + 2,), -123.0, device="cuda", dtype=torch.float32)
    cache = get_runtime().get_cuda_core_cache()

    def evaluate() -> None:
        launch(
            cache.get_stream_wrapper(torch.cuda.current_stream()),
            LaunchConfig(grid=batch, block=256),
            cost_kernels[name],
            output.data_ptr(),
            np.int32(batch),
            *(value.data_ptr() for value in inputs),
        )

    evaluate()
    torch.testing.assert_close(output[:batch], reference, rtol=1e-5, atol=1e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        evaluate()
    for _ in range(3):
        output[:batch].fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output[:batch], reference, rtol=1e-5, atol=1e-5)
    assert bool((output[batch:] == -123.0).all())
    for value, expected in zip(inputs, original):
        torch.testing.assert_close(value, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("layouts", [{}, {"empty": {}}, {"negative": {"term": (-1, 1)}}])
def test_invalid_cost_layout(
    layouts: dict[str, dict[str, tuple[int, int]]], tmp_path: Path
) -> None:
    """Reject invalid specializations before invoking the compiler."""
    with pytest.raises(ValueError, match="cost layouts"):
        compile_cost_reduction_kernels(tmp_path, "sm_120", layouts)
