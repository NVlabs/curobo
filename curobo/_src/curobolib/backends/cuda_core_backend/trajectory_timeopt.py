# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sample the retained path at requested timeopt coordinates."""

from __future__ import annotations

import torch
from cuda.core import LaunchConfig

from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg
from curobo._src.curobolib.backends.cuda_core_backend.util import ceil_div


def _launch(name: str, output: torch.Tensor, count: int, *args: object) -> None:
    """Compile once and launch on the current PyTorch stream."""
    cache = get_runtime().get_cuda_core_cache()
    cfg = TrajectoryKernelCfg()
    qualified = "curobo::trajectory::timeopt::" + name
    kernel = cache.get_or_compile_kernel(
        source_files=[cfg.kernel_dir / "timeopt" / "timeopt_path_kernel.cuh"],
        kernel_name=qualified,
        include_dirs=cfg.get_include_dirs(),
        compile_flags=cfg.get_compile_flags(),
    )
    stream = cache.get_stream_wrapper(torch.cuda.current_stream(output.device))
    launch_kernel(
        qualified, stream, LaunchConfig(grid=ceil_div(count, 256), block=256), kernel, *args
    )


def sample(
    controls: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    timing: torch.Tensor,
    polynomial: torch.Tensor,
    nodes: torch.Tensor,
    weights: torch.Tensor,
    time: torch.Tensor,
    requested: torch.Tensor,
    success: torch.Tensor,
    stationary: torch.Tensor,
    output: torch.Tensor,
    start_offset: int,
) -> None:
    """Invert time and evaluate joint states [4,B,N,D] with the native CUDA sampler."""
    batch, knots, dof = controls.shape
    _launch(
        "sample_kernel",
        output,
        batch * requested.shape[1],
        *[
            x.data_ptr()
            for x in (
                controls, start, goal, timing, polynomial, nodes, weights, time,
                requested, success, stationary, output,
            )
        ],
        batch,
        requested.shape[1],
        dof,
        knots,
        time.shape[1] - 1,
        start_offset,
    )
