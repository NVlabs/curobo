# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sample the retained path at requested timeopt coordinates."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from cuda.core import LaunchConfig

from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg
from curobo._src.curobolib.backends.cuda_core_backend.util import ceil_div

if TYPE_CHECKING:
    from curobo._src.trajectory.trajectory_spline import SplineTrajectory


def _launch(name: str, output: torch.Tensor, count: int, *args: object) -> None:
    """Compile once and launch on the current PyTorch stream."""
    cache = get_runtime().get_cuda_core_cache()
    cfg = TrajectoryKernelCfg()
    qualified = "curobo::trajectory::timeopt::" + name
    kernel = cache.get_or_compile_kernel(
        source_files=[
            cfg.kernel_dir
            / "timeopt"
            / (
                "trajectory_spline_kernel.cuh"
                if name == "interpolate_spline_kernel"
                else "timeopt_path_kernel.cuh"
            )
        ],
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
    progress: torch.Tensor,
    start_offset: int,
) -> None:
    """Write joint states ``[4,B,N,D]`` and active reference progress ``[B,N]``."""
    batch, knots, dof = controls.shape
    _launch(
        "sample_with_progress_kernel",
        output,
        batch * requested.shape[1],
        *[
            x.data_ptr()
            for x in (
                controls,
                start,
                goal,
                timing,
                polynomial,
                nodes,
                weights,
                time,
                requested,
                success,
                stationary,
                output,
                progress,
            )
        ],
        batch,
        requested.shape[1],
        dof,
        knots,
        time.shape[1] - 1,
        start_offset,
    )


@lru_cache(maxsize=None)
def _spline_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retain static timing-basis and quadrature tables per CUDA device."""
    import numpy as np

    from curobo._src.trajectory.timeopt.timeopt_basis import CubicTimingBasis

    basis = CubicTimingBasis(32, device)
    nodes, weights = np.polynomial.legendre.leggauss(6)
    return basis.polynomial, basis.tensor((nodes + 1) / 2), basis.tensor(weights / 2)


def interpolate_spline(trajectory: SplineTrajectory, samples: int, dt: float) -> torch.Tensor:
    """Allocate and evaluate one complete spline into [4,1,N,D]."""
    import numpy as np

    from curobo._src.trajectory.trajectory_spline import TimeOptTiming

    points = trajectory.control_points
    timing = trajectory.timing
    output = points.new_empty((4, 1, samples, points.shape[-1]))
    retimed = isinstance(timing, TimeOptTiming)
    polynomial, nodes, weights = _spline_tables(points.device)
    _launch(
        "interpolate_spline_kernel",
        output,
        samples,
        points.data_ptr(),
        0 if retimed else timing.knot_dt.data_ptr(),
        timing.controls.data_ptr() if retimed else 0,
        timing.time.data_ptr() if retimed else 0,
        polynomial.data_ptr(),
        nodes.data_ptr(),
        weights.data_ptr(),
        output.data_ptr(),
        points.shape[1],
        points.shape[2],
        samples,
        timing.time.shape[1] if retimed else 0,
        np.float32(dt),
    )
    return output
