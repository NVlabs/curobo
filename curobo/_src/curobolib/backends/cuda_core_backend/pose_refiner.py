# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime compilation and launch helpers for camera-pose LM kernels."""

from __future__ import annotations

# Standard Library
from typing import Any

# Third Party
import numpy as np
import torch
from cuda.core import LaunchConfig

# CuRobo
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.optimization_config import (
    OptimizationKernelCfg,
)
from curobo._src.curobolib.backends.cuda_core_backend.util import ceil_div

_THREADS_PER_BLOCK = 256
_KERNEL_PREFIX = "curobo::optimization::pose_refiner::"


def _kernel(kernel_name: str) -> tuple[Any, Any]:
    """Return one prepared pose-refiner kernel from the process cache."""
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = OptimizationKernelCfg()
    source_files = [
        kernel_config.kernel_dir / filename
        for filename in kernel_config.get_kernel_files("pose_refiner_lm")
    ]
    return cache, cache.get_or_compile_kernel(
        source_files=source_files,
        kernel_name=_KERNEL_PREFIX + kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )


def _launch(
    kernel_name: str,
    anchor: torch.Tensor,
    config: LaunchConfig,
    *arguments: object,
) -> None:
    """Launch one cached kernel on the active PyTorch stream."""
    cache, kernel = _kernel(kernel_name)
    stream = cache.get_stream_wrapper(torch.cuda.current_stream(anchor.device))
    launch_kernel(_KERNEL_PREFIX + kernel_name, stream, config, kernel, *arguments)


def launch_pose_refiner_count_valid_depth(
    depth: torch.Tensor,
    depth_minimum: float,
    depth_maximum: float,
    valid_count: torch.Tensor,
) -> None:
    """Count finite in-range values in ``depth`` into scalar ``valid_count``."""
    pixel_count = depth.numel()
    _launch(
        "count_valid_depth_kernel",
        depth,
        LaunchConfig(grid=ceil_div(pixel_count, _THREADS_PER_BLOCK), block=_THREADS_PER_BLOCK),
        depth.data_ptr(),
        np.int32(pixel_count),
        np.float32(depth_minimum),
        np.float32(depth_maximum),
        valid_count.data_ptr(),
    )


def launch_pose_refiner_initialize_state(
    initial_position: torch.Tensor,
    initial_quaternion: torch.Tensor,
    initial_jtj: torch.Tensor,
    initial_jtr: torch.Tensor,
    initial_sum_sq: torch.Tensor,
    initial_valid_count: torch.Tensor,
    valid_depth_count: torch.Tensor,
    minimum_valid_depth_pixels: int,
    minimum_valid_correspondences: int,
    lambda_initial: float,
    best_position: torch.Tensor,
    best_quaternion: torch.Tensor,
    best_jtj: torch.Tensor,
    best_jtr: torch.Tensor,
    best_sum_sq: torch.Tensor,
    best_error: torch.Tensor,
    best_valid_count: torch.Tensor,
    lambda_damping: torch.Tensor,
    result_float: torch.Tensor,
    result_int: torch.Tensor,
) -> None:
    """Initialize persistent LM state and result buffers from one alignment."""
    _launch(
        "initialize_state_kernel",
        best_position,
        LaunchConfig(grid=1, block=1),
        initial_position.data_ptr(),
        initial_quaternion.data_ptr(),
        initial_jtj.data_ptr(),
        initial_jtr.data_ptr(),
        initial_sum_sq.data_ptr(),
        initial_valid_count.data_ptr(),
        valid_depth_count.data_ptr(),
        np.int32(minimum_valid_depth_pixels),
        np.int32(minimum_valid_correspondences),
        np.float32(lambda_initial),
        best_position.data_ptr(),
        best_quaternion.data_ptr(),
        best_jtj.data_ptr(),
        best_jtr.data_ptr(),
        best_sum_sq.data_ptr(),
        best_error.data_ptr(),
        best_valid_count.data_ptr(),
        lambda_damping.data_ptr(),
        result_float.data_ptr(),
        result_int.data_ptr(),
    )


def launch_pose_refiner_solve_candidate(
    best_position: torch.Tensor,
    best_quaternion: torch.Tensor,
    best_jtj: torch.Tensor,
    best_jtr: torch.Tensor,
    lambda_damping: torch.Tensor,
    candidate_position: torch.Tensor,
    candidate_quaternion: torch.Tensor,
    predicted_reduction: torch.Tensor,
    result_int: torch.Tensor,
) -> None:
    """Solve one 6-DOF LM system and compose its candidate camera pose."""
    _launch(
        "solve_candidate_kernel",
        candidate_position,
        LaunchConfig(grid=1, block=1),
        best_position.data_ptr(),
        best_quaternion.data_ptr(),
        best_jtj.data_ptr(),
        best_jtr.data_ptr(),
        lambda_damping.data_ptr(),
        candidate_position.data_ptr(),
        candidate_quaternion.data_ptr(),
        predicted_reduction.data_ptr(),
        result_int.data_ptr(),
    )


def launch_pose_refiner_accept_candidate(
    candidate_position: torch.Tensor,
    candidate_quaternion: torch.Tensor,
    candidate_jtj: torch.Tensor,
    candidate_jtr: torch.Tensor,
    candidate_sum_sq: torch.Tensor,
    candidate_valid_count: torch.Tensor,
    best_position: torch.Tensor,
    best_quaternion: torch.Tensor,
    best_jtj: torch.Tensor,
    best_jtr: torch.Tensor,
    best_sum_sq: torch.Tensor,
    best_error: torch.Tensor,
    best_valid_count: torch.Tensor,
    lambda_damping: torch.Tensor,
    predicted_reduction: torch.Tensor,
    minimum_valid_correspondences: int,
    lambda_factor: float,
    lambda_minimum: float,
    lambda_maximum: float,
    result_float: torch.Tensor,
    result_int: torch.Tensor,
) -> None:
    """Accept or reject one candidate and update persistent LM state in-place."""
    _launch(
        "accept_candidate_kernel",
        best_position,
        LaunchConfig(grid=1, block=1),
        candidate_position.data_ptr(),
        candidate_quaternion.data_ptr(),
        candidate_jtj.data_ptr(),
        candidate_jtr.data_ptr(),
        candidate_sum_sq.data_ptr(),
        candidate_valid_count.data_ptr(),
        best_position.data_ptr(),
        best_quaternion.data_ptr(),
        best_jtj.data_ptr(),
        best_jtr.data_ptr(),
        best_sum_sq.data_ptr(),
        best_error.data_ptr(),
        best_valid_count.data_ptr(),
        lambda_damping.data_ptr(),
        predicted_reduction.data_ptr(),
        np.int32(minimum_valid_correspondences),
        np.float32(lambda_factor),
        np.float32(lambda_minimum),
        np.float32(lambda_maximum),
        result_float.data_ptr(),
        result_int.data_ptr(),
    )


def launch_pose_refiner_copy_best_pose(
    best_position: torch.Tensor,
    best_quaternion: torch.Tensor,
    output_position: torch.Tensor,
    output_quaternion: torch.Tensor,
) -> None:
    """Copy the accepted position and quaternion into output tensors."""
    _launch(
        "copy_best_pose_kernel",
        output_position,
        LaunchConfig(grid=1, block=4),
        best_position.data_ptr(),
        best_quaternion.data_ptr(),
        output_position.data_ptr(),
        output_quaternion.data_ptr(),
    )
