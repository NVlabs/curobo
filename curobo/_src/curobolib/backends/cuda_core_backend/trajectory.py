# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""cuda.core backend for trajectory kernels.
Provides runtime compilation of CUDA kernels with the same interface as PyBind11 extensions.
"""

# Standard Library

# Third Party
import torch

# CuRobo
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import (
    BSplineLaunchCfg,
    LegacyTrajectoryLaunchCfg,
    TrajectoryKernelCfg,
)

# ============================================================================
# B-SPLINE TRAJECTORY KERNELS
# ============================================================================


def launch_bspline_interpolation_forward_kernel(
    out_position: torch.Tensor,
    out_velocity: torch.Tensor,
    out_acceleration: torch.Tensor,
    out_jerk: torch.Tensor,
    out_dt: torch.Tensor,
    u_position: torch.Tensor,
    start_position: torch.Tensor,
    start_velocity: torch.Tensor,
    start_acceleration: torch.Tensor,
    start_jerk: torch.Tensor,
    goal_position: torch.Tensor,
    goal_velocity: torch.Tensor,
    goal_acceleration: torch.Tensor,
    goal_jerk: torch.Tensor,
    start_idx: torch.Tensor,
    goal_idx: torch.Tensor,
    traj_dt: torch.Tensor,
    use_implicit_goal_state: torch.Tensor,
    batch_size: int,
    horizon: int,
    dof: int,
    n_knots: int,
    bspline_degree: int,
):
    """Launch B-spline interpolation forward kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Args:
        All tensor arguments and parameters as expected by B-spline forward kernel

    Note:
        Modifies output tensors in-place
    """
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = TrajectoryKernelCfg()

    # Select kernel based on bspline_degree (3, 4, or 5)
    # Default BasisBackend is MATRIX (value 0)
    # kernel signature: interpolate_bspline_kernel<float, DEGREE, BasisBackend>
    kernel_name = f"curobo::trajectory::bspline::interpolate_bspline_kernel<float, {bspline_degree}, (curobo::trajectory::bspline::BasisBackend)2>"

    # Get or compile kernel
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("bspline_forward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Calculate launch configuration
    config = BSplineLaunchCfg.calculate_forward_config(batch_size, dof, horizon)

    # Get stream wrapper
    pt_stream = torch.cuda.current_stream(out_position.device)
    stream = cache.get_stream_wrapper(pt_stream)

    # Prepare kernel arguments
    kernel_args = (
        out_position.data_ptr(),
        out_velocity.data_ptr(),
        out_acceleration.data_ptr(),
        out_jerk.data_ptr(),
        out_dt.data_ptr(),
        u_position.data_ptr(),
        start_position.data_ptr(),
        start_velocity.data_ptr(),
        start_acceleration.data_ptr(),
        start_jerk.data_ptr(),
        goal_position.data_ptr(),
        goal_velocity.data_ptr(),
        goal_acceleration.data_ptr(),
        goal_jerk.data_ptr(),
        start_idx.data_ptr(),
        goal_idx.data_ptr(),
        traj_dt.data_ptr(),
        use_implicit_goal_state.data_ptr(),
        batch_size,
        horizon,
        dof,
        n_knots,
    )

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def launch_bspline_interpolation_backward_kernel(
    out_grad_position: torch.Tensor,
    grad_position: torch.Tensor,
    grad_velocity: torch.Tensor,
    grad_acceleration: torch.Tensor,
    grad_jerk: torch.Tensor,
    traj_dt: torch.Tensor,
    dt_idx: torch.Tensor,
    use_implicit_goal_state: torch.Tensor,
    batch_size: int,
    padded_horizon: int,
    dof: int,
    n_knots: int,
    bspline_degree: int,
    use_direct_polynomial: bool,
):
    """Launch B-spline interpolation backward kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Args:
        All tensor arguments and parameters as expected by B-spline backward kernel
        padded_horizon: The padded horizon value (horizon will be computed as padded_horizon - 1)

    Note:
        Modifies out_grad_position tensor in-place
    """
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = TrajectoryKernelCfg()

    # Convert padded_horizon to horizon (matching C++ implementation line 600)
    horizon = padded_horizon - 1

    # Validate horizon
    if horizon < 5:
        raise RuntimeError("horizon must be greater than 5")

    # Select kernel based on bspline_degree (3, 4, or 5)
    # Default BasisBackend is MATRIX (value 2)
    # kernel signature: bspline_backward_kernel<DEGREE, float, BasisBackend>
    kernel_name = f"curobo::trajectory::bspline::bspline_backward_kernel<{bspline_degree}, float, (curobo::trajectory::bspline::BasisBackend)2>"

    # Get or compile kernel
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("bspline_backward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Calculate launch configuration using compute_bspline_backward_layout
    # This matches the C++ implementation which computes layout and uses threads_for_n_knots
    config = BSplineLaunchCfg.calculate_backward_config(
        batch_size, dof, n_knots, horizon, bspline_degree
    )

    # Get stream wrapper
    pt_stream = torch.cuda.current_stream(out_grad_position.device)
    stream = cache.get_stream_wrapper(pt_stream)

    # Prepare kernel arguments
    # Pass horizon (not padded_horizon) to match the kernel signature
    kernel_args = (
        out_grad_position.data_ptr(),
        grad_position.data_ptr(),
        grad_velocity.data_ptr(),
        grad_acceleration.data_ptr(),
        grad_jerk.data_ptr(),
        traj_dt.data_ptr(),
        dt_idx.data_ptr(),
        use_implicit_goal_state.data_ptr(),
        batch_size,
        horizon,  # Pass horizon directly (C++ kernel expects horizon, not padded_horizon)
        dof,
        n_knots,
    )

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def launch_bspline_interpolation_single_dt_kernel(
    out_position: torch.Tensor,
    out_velocity: torch.Tensor,
    out_acceleration: torch.Tensor,
    out_jerk: torch.Tensor,
    out_dt: torch.Tensor,
    knots: torch.Tensor,
    knot_dt: torch.Tensor,
    start_position: torch.Tensor,
    start_velocity: torch.Tensor,
    start_acceleration: torch.Tensor,
    start_jerk: torch.Tensor,
    goal_position: torch.Tensor,
    goal_velocity: torch.Tensor,
    goal_acceleration: torch.Tensor,
    goal_jerk: torch.Tensor,
    start_idx: torch.Tensor,
    goal_idx: torch.Tensor,
    interpolation_dt: torch.Tensor,
    use_implicit_goal_state: torch.Tensor,
    interpolation_horizon: torch.Tensor,
    batch_size: int,
    max_out_tsteps: int,
    dof: int,
    n_knots: int,
    bspline_degree: int,
):
    """Launch B-spline single dt interpolation kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Args:
        All tensor arguments and parameters as expected by B-spline single dt kernel

    Note:
        Modifies output tensors in-place
    """
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = TrajectoryKernelCfg()

    # Select kernel based on bspline_degree (3, 4, or 5)
    # Default BasisBackend is MATRIX (value 0)
    # kernel signature: interpolate_bspline_single_dt_kernel<float, DEGREE, BasisBackend>
    kernel_name = f"curobo::trajectory::bspline::interpolate_bspline_single_dt_kernel<float, {bspline_degree}, (curobo::trajectory::bspline::BasisBackend)2>"

    # Get or compile kernel
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("bspline_single_dt")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Calculate launch configuration
    config = BSplineLaunchCfg.calculate_single_dt_config(batch_size, dof, max_out_tsteps)

    # Get stream wrapper
    pt_stream = torch.cuda.current_stream(out_position.device)
    stream = cache.get_stream_wrapper(pt_stream)

    # Prepare kernel arguments
    kernel_args = (
        out_position.data_ptr(),
        out_velocity.data_ptr(),
        out_acceleration.data_ptr(),
        out_jerk.data_ptr(),
        out_dt.data_ptr(),
        knots.data_ptr(),
        knot_dt.data_ptr(),
        start_position.data_ptr(),
        start_velocity.data_ptr(),
        start_acceleration.data_ptr(),
        start_jerk.data_ptr(),
        goal_position.data_ptr(),
        goal_velocity.data_ptr(),
        goal_acceleration.data_ptr(),
        goal_jerk.data_ptr(),
        start_idx.data_ptr(),
        goal_idx.data_ptr(),
        interpolation_dt.data_ptr(),
        use_implicit_goal_state.data_ptr(),
        interpolation_horizon.data_ptr(),
        batch_size,
        max_out_tsteps,
        dof,
        n_knots,
    )

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


# ============================================================================
# LEGACY TRAJECTORY KERNELS
# ============================================================================


def launch_differentiation_position_forward_kernel(
    out_position: torch.Tensor,
    out_velocity: torch.Tensor,
    out_acceleration: torch.Tensor,
    out_jerk: torch.Tensor,
    out_dt: torch.Tensor,
    u_position: torch.Tensor,
    start_position: torch.Tensor,
    start_velocity: torch.Tensor,
    start_acceleration: torch.Tensor,
    goal_position: torch.Tensor,
    goal_velocity: torch.Tensor,
    goal_acceleration: torch.Tensor,
    start_idx: torch.Tensor,
    goal_idx: torch.Tensor,
    traj_dt: torch.Tensor,
    use_implicit_goal_state: torch.Tensor,
    batch_size: int,
    horizon: int,
    dof: int,
):
    """Launch differentiation position forward kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Args:
        All tensor arguments and parameters as expected by differentiation forward kernel

    Note:
        Modifies output tensors in-place
    """
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = TrajectoryKernelCfg()
    use_stencil = True
    # kernel signature: position_clique_loop_idx_fwd_kernel<float>
    kernel_name = "curobo::trajectory::legacy::position_clique_loop_idx_fwd_kernel<float," + f"{str(use_stencil).lower()}" + ">"

    # Get or compile kernel
    kernel_files = [
        kernel_config.kernel_dir / f
        for f in kernel_config.get_kernel_files("differentiation_forward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Calculate launch configuration
    config = LegacyTrajectoryLaunchCfg.calculate_differentiation_forward_config(
        batch_size, dof, horizon
    )

    # Get stream wrapper
    pt_stream = torch.cuda.current_stream()
    stream = cache.get_stream_wrapper(pt_stream)

    # Prepare kernel arguments
    kernel_args = (
        out_position.data_ptr(),
        out_velocity.data_ptr(),
        out_acceleration.data_ptr(),
        out_jerk.data_ptr(),
        out_dt.data_ptr(),
        u_position.data_ptr(),
        start_position.data_ptr(),
        start_velocity.data_ptr(),
        start_acceleration.data_ptr(),
        goal_position.data_ptr(),
        goal_velocity.data_ptr(),
        goal_acceleration.data_ptr(),
        start_idx.data_ptr(),
        goal_idx.data_ptr(),
        traj_dt.data_ptr(),
        use_implicit_goal_state.data_ptr(),
        batch_size,
        horizon,
        dof,
    )

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def launch_differentiation_position_backward_kernel(
    out_grad_position: torch.Tensor,
    grad_position: torch.Tensor,
    grad_velocity: torch.Tensor,
    grad_acceleration: torch.Tensor,
    grad_jerk: torch.Tensor,
    traj_dt: torch.Tensor,
    dt_idx: torch.Tensor,
    use_implicit_goal_state: torch.Tensor,
    batch_size: int,
    horizon: int,
    dof: int,
):
    """Launch differentiation position backward kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Args:
        All tensor arguments and parameters as expected by differentiation backward kernel

    Note:
        Modifies out_grad_position tensor in-place
    """
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = TrajectoryKernelCfg()
    use_stencil = True

    # kernel signature: position_clique_loop_idx_bwd_kernel<float>
    kernel_name = "curobo::trajectory::legacy::position_clique_loop_idx_bwd_kernel<float," + f"{str(use_stencil).lower()}" + ">"

    # Get or compile kernel
    kernel_files = [
        kernel_config.kernel_dir / f
        for f in kernel_config.get_kernel_files("differentiation_backward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Calculate launch configuration
    config = LegacyTrajectoryLaunchCfg.calculate_differentiation_backward_config(
        batch_size, dof, horizon
    )

    # Get stream wrapper
    pt_stream = torch.cuda.current_stream()
    stream = cache.get_stream_wrapper(pt_stream)

    # Prepare kernel arguments
    kernel_args = (
        out_grad_position.data_ptr(),
        grad_position.data_ptr(),
        grad_velocity.data_ptr(),
        grad_acceleration.data_ptr(),
        grad_jerk.data_ptr(),
        traj_dt.data_ptr(),
        dt_idx.data_ptr(),
        use_implicit_goal_state.data_ptr(),
        batch_size,
        horizon,
        dof,
    )

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def launch_integration_acceleration_kernel(
    out_position: torch.Tensor,
    out_velocity: torch.Tensor,
    out_acceleration: torch.Tensor,
    out_jerk: torch.Tensor,
    u_acc: torch.Tensor,
    start_position: torch.Tensor,
    start_velocity: torch.Tensor,
    start_acceleration: torch.Tensor,
    start_idx: torch.Tensor,
    traj_dt: torch.Tensor,
    batch_size: int,
    horizon: int,
    dof: int,
    use_rk2: bool = True,
):
    """Launch integration acceleration kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Args:
        All tensor arguments and parameters as expected by integration kernel

    Note:
        Modifies output tensors in-place
    """
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = TrajectoryKernelCfg()
    max_horizon = horizon

    # Select kernel based on use_rk2
    if use_rk2:
        # kernel signature: acceleration_loop_idx_rk2_kernel<float>
        kernel_name = "curobo::trajectory::legacy::acceleration_loop_idx_rk2_kernel<float," + f"{max_horizon}" + ">"
    else:
        # kernel signature: acceleration_loop_idx_kernel<float>
        kernel_name = "curobo::trajectory::legacy::acceleration_loop_idx_kernel<float," + f"{max_horizon}" + ">"

    # Get or compile kernel
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("integration")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Calculate launch configuration
    config = LegacyTrajectoryLaunchCfg.calculate_integration_config(batch_size, dof)

    # Get stream wrapper
    pt_stream = torch.cuda.current_stream()
    stream = cache.get_stream_wrapper(pt_stream)

    # Prepare kernel arguments
    kernel_args = (
        out_position.data_ptr(),
        out_velocity.data_ptr(),
        out_acceleration.data_ptr(),
        out_jerk.data_ptr(),
        u_acc.data_ptr(),
        start_position.data_ptr(),
        start_velocity.data_ptr(),
        start_acceleration.data_ptr(),
        start_idx.data_ptr(),
        traj_dt.data_ptr(),
        batch_size,
        horizon,
        dof,
    )

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)
