# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime compilation and launch helpers for trajectory seed fitting."""

from __future__ import annotations

# Standard Library
from typing import Any

# Third Party
import numpy as np
import torch
from cuda.bindings import runtime as cudart
from cuda.core import LaunchConfig

# CuRobo
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import (
    TrajectoryKernelCfg,
)
from curobo._src.curobolib.backends.cuda_core_backend.util import ceil_div
from curobo._src.util.logging import log_and_raise

_THREADS_PER_BLOCK = 256
_KERNEL_PREFIX = "curobo::trajectory::seed_fitter::"


def _kernel(kernel_name: str) -> tuple[Any, Any]:
    """Return one prepared seed-fitter kernel from the process cache."""
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = TrajectoryKernelCfg()
    kernel_files = [kernel_config.kernel_dir / "seed_fitter" / "seed_fitter_kernel.cuh"]
    return cache, cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=_KERNEL_PREFIX + kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )


def _launch(
    kernel_name: str,
    output: torch.Tensor,
    grid: int,
    *arguments: object,
    shared_memory: int = 0,
) -> None:
    """Launch a cached seed-fitter kernel on the active PyTorch stream."""
    cache, kernel = _kernel(kernel_name)
    if shared_memory > 48 * 1024:
        # Larger fits need the device's opt-in dynamic shared-memory capacity.
        error = cudart.cudaFuncSetAttribute(
            int(kernel._handle),
            cudart.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_memory,
        )
        if error[0] != cudart.cudaError_t.cudaSuccess:
            log_and_raise(
                f"Cannot configure {shared_memory} shared-memory bytes for "
                f"seed-fitter kernel {kernel_name}: {error}"
            )
    stream = cache.get_stream_wrapper(torch.cuda.current_stream(output.device))
    config = LaunchConfig(grid=grid, block=_THREADS_PER_BLOCK, shmem_size=shared_memory)
    launch_kernel(_KERNEL_PREFIX + kernel_name, stream, config, kernel, *arguments)


def launch_resample_dense_kernel(
    target_path: torch.Tensor,
    arc_progress: torch.Tensor,
    arc_target: torch.Tensor,
) -> None:
    """Arc-length resample paths from ``[seeds, steps, dof]`` into ``arc_target``."""
    seeds, steps, dof = target_path.shape
    _launch(
        "resample_dense_kernel",
        arc_target,
        seeds,
        target_path.data_ptr(),
        arc_progress.data_ptr(),
        arc_target.data_ptr(),
        seeds,
        steps,
        dof,
        shared_memory=2 * steps * 4,
    )


def launch_initialize_fit_kernel(
    arc_target: torch.Tensor,
    boundary_position: torch.Tensor,
    start_position: torch.Tensor,
    start_velocity: torch.Tensor,
    start_acceleration: torch.Tensor,
    start_jerk: torch.Tensor,
    goal_position: torch.Tensor,
    fit_matrix: torch.Tensor,
    start_position_coefficient: torch.Tensor,
    start_velocity_coefficient: torch.Tensor,
    start_acceleration_coefficient: torch.Tensor,
    start_jerk_coefficient: torch.Tensor,
    goal_position_coefficient: torch.Tensor,
    position_residual: torch.Tensor,
    knots: torch.Tensor,
) -> None:
    """Initialize fitted knots and the fixed position residual in-place."""
    seeds, steps, dof = arc_target.shape
    knot_count = knots.shape[1]
    _launch(
        "initialize_fit_kernel",
        knots,
        ceil_div(seeds * knot_count * dof, _THREADS_PER_BLOCK),
        arc_target.data_ptr(),
        boundary_position.data_ptr(),
        start_position.data_ptr(),
        start_velocity.data_ptr(),
        start_acceleration.data_ptr(),
        start_jerk.data_ptr(),
        goal_position.data_ptr(),
        fit_matrix.data_ptr(),
        start_position_coefficient.data_ptr(),
        start_velocity_coefficient.data_ptr(),
        start_acceleration_coefficient.data_ptr(),
        start_jerk_coefficient.data_ptr(),
        goal_position_coefficient.data_ptr(),
        position_residual.data_ptr(),
        knots.data_ptr(),
        seeds,
        steps,
        dof,
        knot_count,
    )


def launch_required_dt_kernel(
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    jerk: torch.Tensor,
    max_velocity: torch.Tensor,
    max_acceleration: torch.Tensor,
    max_jerk: torch.Tensor,
    trajectory_dt: float,
    dt_epsilon: float,
    required_dt: torch.Tensor,
) -> None:
    """Reduce derivative-limit ratios into one required timestep per seed."""
    seeds, steps, dof = velocity.shape
    _launch(
        "required_dt_kernel",
        required_dt,
        seeds,
        velocity.data_ptr(),
        acceleration.data_ptr(),
        jerk.data_ptr(),
        max_velocity.data_ptr(),
        max_acceleration.data_ptr(),
        max_jerk.data_ptr(),
        steps,
        dof,
        np.float32(trajectory_dt),
        np.float32(dt_epsilon),
        required_dt.data_ptr(),
    )


def launch_derivative_scale_kernel(
    current_velocity: torch.Tensor,
    current_acceleration: torch.Tensor,
    max_velocity: torch.Tensor,
    max_acceleration: torch.Tensor,
    max_jerk: torch.Tensor,
    basis_velocity: torch.Tensor,
    basis_acceleration: torch.Tensor,
    basis_jerk: torch.Tensor,
    path_normal_trace: float,
    timing_regularization: float,
    irls_epsilon: float,
    velocity_trace_workspace: torch.Tensor,
    acceleration_trace_workspace: torch.Tensor,
    jerk_trace_workspace: torch.Tensor,
    derivative_scale: torch.Tensor,
) -> None:
    """Compute one derivative objective scale per path into ``[seeds]`` storage."""
    seeds, steps, dof = current_velocity.shape
    free_knots = basis_velocity.shape[1]
    _launch(
        "derivative_scale_kernel",
        derivative_scale,
        seeds,
        current_velocity.data_ptr(),
        current_acceleration.data_ptr(),
        max_velocity.data_ptr(),
        max_acceleration.data_ptr(),
        max_jerk.data_ptr(),
        basis_velocity.data_ptr(),
        basis_acceleration.data_ptr(),
        basis_jerk.data_ptr(),
        seeds,
        steps,
        dof,
        free_knots,
        np.float32(path_normal_trace),
        np.float32(timing_regularization),
        np.float32(irls_epsilon),
        velocity_trace_workspace.data_ptr(),
        acceleration_trace_workspace.data_ptr(),
        jerk_trace_workspace.data_ptr(),
        derivative_scale.data_ptr(),
    )


def launch_solve_irls_kernel(
    position_residual: torch.Tensor,
    boundary_velocity: torch.Tensor,
    boundary_acceleration: torch.Tensor,
    boundary_jerk: torch.Tensor,
    current_velocity: torch.Tensor,
    current_acceleration: torch.Tensor,
    current_knots: torch.Tensor,
    goal_position: torch.Tensor,
    max_velocity: torch.Tensor,
    max_acceleration: torch.Tensor,
    max_jerk: torch.Tensor,
    basis_position: torch.Tensor,
    basis_velocity: torch.Tensor,
    basis_acceleration: torch.Tensor,
    basis_jerk: torch.Tensor,
    path_normal: torch.Tensor,
    derivative_scale: torch.Tensor,
    regularization: float,
    trust_region_regularization: float,
    path_diagonal_scale: float,
    irls_epsilon: float,
    candidate_knots: torch.Tensor,
) -> None:
    """Solve one small Cholesky IRLS system per seed and joint."""
    seeds, steps, dof = current_velocity.shape
    knot_count = current_knots.shape[1]
    free_knots = knot_count - 1
    shared_values = 2 * steps + free_knots * free_knots + free_knots
    _launch(
        "solve_irls_kernel",
        candidate_knots,
        seeds * dof,
        position_residual.data_ptr(),
        boundary_velocity.data_ptr(),
        boundary_acceleration.data_ptr(),
        boundary_jerk.data_ptr(),
        current_velocity.data_ptr(),
        current_acceleration.data_ptr(),
        current_knots.data_ptr(),
        goal_position.data_ptr(),
        max_velocity.data_ptr(),
        max_acceleration.data_ptr(),
        max_jerk.data_ptr(),
        basis_position.data_ptr(),
        basis_velocity.data_ptr(),
        basis_acceleration.data_ptr(),
        basis_jerk.data_ptr(),
        path_normal.data_ptr(),
        derivative_scale.data_ptr(),
        steps,
        dof,
        knot_count,
        np.float32(regularization),
        np.float32(trust_region_regularization),
        np.float32(path_diagonal_scale),
        np.float32(irls_epsilon),
        candidate_knots.data_ptr(),
        shared_memory=shared_values * 4,
    )


def launch_accept_candidate_kernel(
    candidate_knots: torch.Tensor,
    candidate_velocity: torch.Tensor,
    candidate_acceleration: torch.Tensor,
    candidate_jerk: torch.Tensor,
    candidate_dt: torch.Tensor,
    current_knots: torch.Tensor,
    current_velocity: torch.Tensor,
    current_acceleration: torch.Tensor,
    current_jerk: torch.Tensor,
    current_dt: torch.Tensor,
) -> None:
    """Accept only candidates with a lower required timestep."""
    seeds, steps, dof = current_velocity.shape
    knot_count = current_knots.shape[1]
    _launch(
        "accept_candidate_kernel",
        current_knots,
        ceil_div(seeds * steps * dof, _THREADS_PER_BLOCK),
        candidate_knots.data_ptr(),
        candidate_velocity.data_ptr(),
        candidate_acceleration.data_ptr(),
        candidate_jerk.data_ptr(),
        candidate_dt.data_ptr(),
        current_knots.data_ptr(),
        current_velocity.data_ptr(),
        current_acceleration.data_ptr(),
        current_jerk.data_ptr(),
        current_dt.data_ptr(),
        seeds,
        steps,
        dof,
        knot_count,
    )
    _launch(
        "accept_candidate_dt_kernel",
        current_dt,
        ceil_div(seeds, _THREADS_PER_BLOCK),
        candidate_dt.data_ptr(),
        current_dt.data_ptr(),
        seeds,
    )
