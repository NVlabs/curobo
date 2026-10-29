# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Runtime compilation and launch helpers for optimization CUDA kernels."""

# Standard Library
from typing import List

# Third Party
import numpy as np
import torch
from cuda import pathfinder
from cuda.core import LaunchConfig

# CuRobo
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.optimization_config import (
    LBFGSCompactLaunchCfg,
    LBFGSLaunchCfg,
    LineSearchLaunchCfg,
    OptimizationKernelCfg,
)
from curobo._src.util.logging import log_and_raise, log_warn


def launch_line_search(
    best_cost: torch.Tensor,
    best_action: torch.Tensor,
    best_iteration: torch.Tensor,
    current_iteration: torch.Tensor,
    converged_global: torch.Tensor,
    convergence_iteration: int,
    cost_delta_threshold: float,
    cost_relative_threshold: float,
    exploration_cost: torch.Tensor,
    exploration_action: torch.Tensor,
    exploration_gradient: torch.Tensor,
    exploration_idx: torch.Tensor,
    selected_cost: torch.Tensor,
    selected_action: torch.Tensor,
    selected_gradient: torch.Tensor,
    selected_idx: torch.Tensor,
    search_cost: torch.Tensor,
    search_action: torch.Tensor,
    search_gradient: torch.Tensor,
    step_direction: torch.Tensor,
    search_magnitudes: torch.Tensor,
    line_search_index: torch.Tensor,
    line_search_status: torch.Tensor,
    curvature_update_valid: torch.Tensor,
    armijo_threshold_c_1: float,
    curvature_threshold_c_2: float,
    strong_wolfe: bool,
    approx_wolfe: bool,
    n_linesearch: int,
    opt_dim: int,
    batchsize: int,
    backtracking: bool,
    beta: torch.Tensor,
):
    """Launch line search kernel using cuda.core runtime compilation.

    NVRTC specializes the candidate count, Wolfe mode, and backtracking policy.

    Args:
        All tensor arguments and parameters as expected by line search kernel

    Note:
        Modifies output tensors in-place
    """
    runtime = get_runtime()

    # Get cuda.core kernel cache
    cache = runtime.get_cuda_core_cache()

    # Get kernel configuration
    kernel_config = OptimizationKernelCfg()

    if n_linesearch < 1 or n_linesearch > 32:
        log_and_raise("CUDA-core line search requires between 1 and 32 probes")
    if opt_dim < 1 or batchsize < 1:
        log_and_raise("CUDA-core line search requires positive opt_dim and batchsize")
    search_elements = batchsize * n_linesearch * opt_dim
    if search_elements > np.iinfo(np.int32).max:
        log_and_raise(
            "CUDA-core line-search tensors must contain at most INT32_MAX elements: "
            + f"got batchsize * n_linesearch * opt_dim = {search_elements}"
        )

    # Keep these integer values synchronized with line_search_kernel.cuh. The
    # mode and exact probe count are part of the NVRTC kernel specialization.
    if strong_wolfe:
        line_search_mode = 1
    elif approx_wolfe:
        line_search_mode = 2
    else:
        line_search_mode = 0
    single_variable_per_thread = opt_dim <= LineSearchLaunchCfg.MAX_THREADS_PER_BLOCK
    kernel_name = (
        "curobo::optimization::line_search::kernel_line_search"
        + f"<float, {n_linesearch}, {line_search_mode}, "
        + f"{str(single_variable_per_thread).lower()}, {str(backtracking).lower()}>"
    )

    # Get or compile kernel
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("line_search")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Calculate launch configuration
    config = LineSearchLaunchCfg.calculate_config(opt_dim, batchsize)

    # Get stream wrapper
    pt_stream = torch.cuda.current_stream()
    stream = cache.get_stream_wrapper(pt_stream)

    # Prepare kernel arguments as data pointers
    # Candidate count and Wolfe mode are compile-time kernel parameters.
    kernel_args = (
        best_cost.data_ptr(),
        best_action.data_ptr(),
        best_iteration.data_ptr(),
        current_iteration.data_ptr(),
        converged_global.data_ptr(),
        convergence_iteration,
        np.float32(cost_delta_threshold),
        np.float32(cost_relative_threshold),
        exploration_cost.data_ptr(),
        exploration_action.data_ptr(),
        exploration_gradient.data_ptr(),
        exploration_idx.data_ptr(),
        selected_cost.data_ptr(),
        selected_action.data_ptr(),
        selected_gradient.data_ptr(),
        selected_idx.data_ptr(),
        search_cost.data_ptr(),
        search_action.data_ptr(),
        search_gradient.data_ptr(),
        step_direction.data_ptr(),
        search_magnitudes.data_ptr(),
        line_search_index.data_ptr(),
        line_search_status.data_ptr(),
        curvature_update_valid.data_ptr(),
        np.float32(armijo_threshold_c_1),
        np.float32(curvature_threshold_c_2),
        opt_dim,
        batchsize,
        beta.data_ptr(),
    )

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def launch_lbfgs_step(
    step_vec: torch.Tensor,
    rho_buffer: torch.Tensor,
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    q: torch.Tensor,
    grad_q: torch.Tensor,
    x_0: torch.Tensor,
    grad_0: torch.Tensor,
    history_head: torch.Tensor,
    history_count: torch.Tensor,
    epsilon: float,
    batch_size: int,
    history_m: int,
    v_dim: int,
    stable_mode: bool,
    use_shared_buffers: bool,
    use_geometric_scaling: bool,
    optimization_step_max: torch.Tensor,
    scale_step: bool,
    curvature_update_valid: torch.Tensor,
    line_search_status: torch.Tensor,
) -> List[torch.Tensor]:
    """Launch LBFGS step kernel using cuda.core runtime compilation.

    Args:
        All tensor arguments and parameters as expected by LBFGS step kernel

    Returns:
        List of tensors: [step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0]

    Note:
        Modifies output tensors in-place
    """
    # print("LBFGS: ")
    # print(batch_size, v_dim, history_m)
    runtime = get_runtime()

    # Get cuda.core kernel cache
    cache = runtime.get_cuda_core_cache()

    # Get kernel configuration
    kernel_config = OptimizationKernelCfg()

    # Use compile-time history through 31 and the runtime-history kernel above
    # that range to avoid excessive unrolling and compilation cost.
    if history_m > 128:
        log_and_raise("History_m greater than 128 is not supported")
    if history_m < 1:
        log_and_raise("History_m must be greater than 0")
    if v_dim < 1:
        log_and_raise("v_dim must be greater than 0")
    if batch_size * history_m * v_dim > np.iinfo(np.int32).max:
        log_and_raise("L-BFGS history must contain at most INT_MAX elements")

    properties = torch.cuda.get_device_properties(step_vec.device)
    device_max_shared_memory = int(
        getattr(properties, "shared_memory_per_block_optin", 0)
        or properties.shared_memory_per_block
    )
    config, use_shared_buffers_actual, dynamic_shared_memory = LBFGSLaunchCfg.calculate_config(
        batch_size,
        v_dim,
        history_m,
        use_shared_buffers,
        device_max_shared_memory,
    )

    items_per_thread, threads_per_block = LBFGSLaunchCfg.calculate_items_and_threads(v_dim)
    shared_specialization = str(use_shared_buffers_actual).lower()
    geometric_specialization = str(use_geometric_scaling).lower()
    fixed_history = history_m if history_m <= 31 else -1
    kernel_name = (
        "curobo::optimization::kernel_lbfgs_step_circular"
        f"<float, false, {fixed_history}, {shared_specialization}, {items_per_thread}, "
        f"{geometric_specialization}>"
    )

    # Get or compile kernel
    kernel_files = [kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("lbfgs")]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Configure extended shared memory if needed (Volta+)
    # This is equivalent to cudaFuncSetAttribute in the C++ version
    static_shared_memory = (2 * LBFGSLaunchCfg.SCRATCH_HISTORY + 1) * 4
    if use_shared_buffers_actual and dynamic_shared_memory + static_shared_memory > 48000:
        # Use cuda-python to configure max dynamic shared memory size
        import cuda.bindings.runtime as cudart

        kernel_ptr = int(kernel._handle)  # This is a Kernel object
        err = cudart.cudaFuncSetAttribute(
            kernel_ptr,
            cudart.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory,
        )

        if err[0] != cudart.cudaError_t.cudaSuccess:
            log_warn(
                f"Failed to configure {dynamic_shared_memory} bytes of dynamic shared memory "
                f"for {kernel_name}; using global history instead. Error: {err}"
            )
            reduction_shared_memory = (
                0
                if threads_per_block == 32
                else (2 * history_m + 3) * (threads_per_block // 32) * 4
            )
            config = LaunchConfig(
                grid=batch_size,
                block=config.block,
                shmem_size=reduction_shared_memory,
            )
            use_shared_buffers_actual = False
            kernel_name = (
                "curobo::optimization::kernel_lbfgs_step_circular"
                f"<float, false, {fixed_history}, false, {items_per_thread}, "
                f"{geometric_specialization}>"
            )
            kernel = cache.get_or_compile_kernel(
                source_files=kernel_files,
                kernel_name=kernel_name,
                include_dirs=kernel_config.get_include_dirs(),
                compile_flags=kernel_config.get_compile_flags(),
            )

    # Get stream wrapper
    pt_stream = torch.cuda.current_stream()
    stream = cache.get_stream_wrapper(pt_stream)

    # Prepare kernel arguments as data pointers
    # Match the C++ kernel signature from lines 266-287 in lbfgs_step_kernel_launch.cu
    kernel_args = (
        step_vec.data_ptr(),
        rho_buffer.data_ptr(),
        y_buffer.data_ptr(),
        s_buffer.data_ptr(),
        q.data_ptr(),
        x_0.data_ptr(),
        grad_0.data_ptr(),
        grad_q.data_ptr(),
        optimization_step_max.data_ptr(),
        np.float32(epsilon),
        batch_size,
        history_m,
        v_dim,
        stable_mode,
        history_head.data_ptr(),
        history_count.data_ptr(),
        scale_step,
        curvature_update_valid.data_ptr(),
        line_search_status.data_ptr(),
    )

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)

    return [step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0]


def launch_lbfgs_compact_step(
    step_vec: torch.Tensor,
    rho_buffer: torch.Tensor,
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    inverse_r_buffer: torch.Tensor,
    matrix_workspace: torch.Tensor,
    q: torch.Tensor,
    grad_q: torch.Tensor,
    x_0: torch.Tensor,
    grad_0: torch.Tensor,
    history_head: torch.Tensor,
    history_count: torch.Tensor,
    optimization_step_max: torch.Tensor,
    epsilon: float,
    batch_size: int,
    history_m: int,
    v_dim: int,
    shared_matrix_count: int,
    scale_step: bool,
    curvature_update_valid: torch.Tensor,
    use_geometric_scaling: bool,
    line_search_status: torch.Tensor,
) -> List[torch.Tensor]:
    """Launch the compact tensor-core L-BFGS step kernel.

    Args:
        step_vec: Output direction buffer with shape ``(batch_size, v_dim)``.
        rho_buffer: Reciprocal curvature history with shape
            ``(history_m, batch_size, 1, 1)``.
        y_buffer: Gradient-difference history with shape
            ``(history_m, batch_size, v_dim, 1)``.
        s_buffer: Position-difference history with shape
            ``(history_m, batch_size, v_dim, 1)``.
        inverse_r_buffer: Persistent compact inverse with shape
            ``(batch_size, history_m, history_m)``.
        matrix_workspace: Persistent global matrix workspace.
        q: Current positions with shape ``(batch_size, v_dim)``.
        grad_q: Current gradients with shape ``(batch_size, 1, v_dim)``.
        x_0: Previous positions with shape ``(batch_size, v_dim, 1)``.
        grad_0: Previous gradients with shape ``(batch_size, v_dim, 1)``.
        history_head: Circular history write indices with shape ``(batch_size,)``.
        history_count: Active history counts with shape ``(batch_size,)``.
        optimization_step_max: Per-variable maximum direction magnitude.
        epsilon: Fallback initial inverse-Hessian scale.
        batch_size: Number of independent optimization problems.
        history_m: L-BFGS history capacity.
        v_dim: Number of optimization variables.
        shared_matrix_count: Number of compact matrices staged in shared memory.
        scale_step: Whether to limit the direction using ``optimization_step_max``.
        curvature_update_valid: Per-problem mask controlling curvature-pair insertion.
        use_geometric_scaling: Whether to initialize the inverse-Hessian scale
            with ``sqrt(s^T s / y^T y)`` instead of ``s^T y / y^T y``.
        line_search_status: Per-problem line-search outcome code used to select
            the stable recurrence and guard abrupt fallback curvature scales.

    Returns:
        Mutated L-BFGS tensors, matching :func:`launch_lbfgs_step`.
    """
    properties = torch.cuda.get_device_properties(step_vec.device)
    device_max_shared_memory = int(
        getattr(properties, "shared_memory_per_block_optin", 0)
        or properties.shared_memory_per_block
    )
    compact_config = LBFGSCompactLaunchCfg.calculate_config(
        batch_size,
        v_dim,
        history_m,
        device_max_shared_memory,
    )
    if compact_config is None:
        log_and_raise(
            f"Compact L-BFGS does not support history={history_m}, v_dim={v_dim} "
            f"with {device_max_shared_memory} bytes of shared memory"
        )
    config, expected_shared_matrix_count, workspace_elements = compact_config
    if shared_matrix_count != expected_shared_matrix_count:
        log_and_raise(
            "Compact L-BFGS shared-matrix layout changed after buffer allocation: "
            f"expected {expected_shared_matrix_count}, got {shared_matrix_count}"
        )
    if inverse_r_buffer.numel() != batch_size * history_m * history_m:
        log_and_raise("Compact L-BFGS inverse-R buffer has an invalid shape")
    required_workspace_elements = batch_size * workspace_elements
    if required_workspace_elements > 0 and matrix_workspace.numel() < required_workspace_elements:
        log_and_raise(
            "Compact L-BFGS matrix workspace is too small: "
            f"expected {required_workspace_elements}, got {matrix_workspace.numel()}"
        )

    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = OptimizationKernelCfg()
    scale_specialization = str(scale_step).lower()
    geometric_specialization = str(use_geometric_scaling).lower()
    kernel_name = (
        "curobo::optimization::kernel_lbfgs_step_compact"
        f"<{history_m}, {v_dim}, {shared_matrix_count}, {scale_specialization}, "
        f"{geometric_specialization}>"
    )
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("lbfgs_compact")
    ]
    include_dirs = kernel_config.get_include_dirs()
    compiler_include = pathfinder.find_nvidia_header_directory("nvcc")
    if compiler_include is not None:
        include_dirs.append(compiler_include)
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=include_dirs,
        compile_flags=kernel_config.get_compile_flags(),
    )
    if config.shmem_size > 48000:
        import cuda.bindings.runtime as cudart

        result = cudart.cudaFuncSetAttribute(
            int(kernel._handle),
            cudart.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize,
            config.shmem_size,
        )
        if result[0] != cudart.cudaError_t.cudaSuccess:
            log_and_raise(
                f"Unable to configure {config.shmem_size} bytes of shared memory "
                f"for compact L-BFGS: {result}"
            )

    stream = cache.get_stream_wrapper(torch.cuda.current_stream(step_vec.device))
    kernel_args = (
        step_vec.data_ptr(),
        rho_buffer.data_ptr(),
        y_buffer.data_ptr(),
        s_buffer.data_ptr(),
        inverse_r_buffer.data_ptr(),
        matrix_workspace.data_ptr(),
        q.data_ptr(),
        x_0.data_ptr(),
        grad_0.data_ptr(),
        grad_q.data_ptr(),
        optimization_step_max.data_ptr(),
        np.float32(epsilon),
        batch_size,
        history_head.data_ptr(),
        history_count.data_ptr(),
        curvature_update_valid.data_ptr(),
        line_search_status.data_ptr(),
    )
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)
    return [step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0]
