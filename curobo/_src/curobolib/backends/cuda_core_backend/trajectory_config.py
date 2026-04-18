# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for trajectory kernel compilation."""

# Standard Library
from pathlib import Path
from typing import List

from cuda.core import LaunchConfig

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend.kernel_config import CudaCoreKernelCfg
from curobo._src.curobolib.backends.cuda_core_backend.util import ceil_div


class TrajectoryKernelCfg(CudaCoreKernelCfg):
    """Configuration for trajectory kernel compilation"""

    def __init__(self):
        super().__init__("trajectory")

    def get_kernel_files(self, kernel_type: str) -> List[str]:
        """Get kernel source files for a given kernel type.

        Args:
            kernel_type: Type of kernel ("bspline_forward", "bspline_backward", "bspline_single_dt",
                                        "differentiation_forward", "differentiation_backward", "integration")

        Returns:
            List of kernel filenames
        """
        kernel_files = {
            "bspline_forward": ["bspline/bspline_kernel.cuh"],
            "bspline_backward": ["bspline/bspline_kernel.cuh"],
            "bspline_single_dt": ["bspline/bspline_kernel.cuh"],
            "differentiation_forward": ["legacy/differentiation_position_kernel.cuh"],
            "differentiation_backward": ["legacy/differentiation_position_kernel.cuh"],
            "integration": ["legacy/integration_acceleration_kernel.cuh"],
        }
        return kernel_files.get(kernel_type, [])

    def get_include_dirs(self) -> List[Path]:
        """Get include directories for kernel compilation"""
        # Get base include dirs and add trajectory-specific ones
        base_dirs = self.get_base_include_dirs()
        trajectory_dirs = [
            self.kernel_dir,  # kernels/trajectory/
            self.kernel_dir / "bspline",
            self.kernel_dir / "bspline" / "basis",
            self.kernel_dir / "legacy",
        ]
        return base_dirs + trajectory_dirs


class BSplineBackwardLayout:
    """Layout information for B-spline backward kernel launch configuration"""

    def __init__(self):
        self.interpolation_steps: int = 0
        self.knots_per_warp: int = 0
        self.warps_for_n_knots: int = 0
        self.threads_for_n_knots: int = 0
        self.padded_horizon: int = 0
        self.n_knots: int = 0
        self.padded_n_knots: int = 0
        self.horizon: int = 0
        self.dof: int = 0


def get_spline_support_size(degree: int) -> int:
    """Get the support size for a B-spline of given degree"""
    return degree + 1


def get_total_knots(n_knots: int, degree: int) -> int:
    """Get total number of knots including support"""
    return n_knots + get_spline_support_size(degree)


def compute_bspline_backward_layout(
    horizon: int, dof: int, n_knots: int, bspline_degree: int
) -> BSplineBackwardLayout:
    """Compute layout information for B-spline backward kernel.

    This function ports the C++ compute_bspline_backward_layout function to Python.
    It calculates thread organization for warp-level parallelism in gradient computation.

    Args:
        horizon: Trajectory horizon (note: this is padded_horizon - 1 in C++)
        dof: Degrees of freedom
        n_knots: Number of B-spline control points
        bspline_degree: B-spline degree (3, 4, or 5)

    Returns:
        BSplineBackwardLayout with computed thread organization
    """
    WARP_SIZE = 32  # CUDA warp size

    layout = BSplineBackwardLayout()

    # Degree-specific calculations
    padded_n_knots = get_total_knots(n_knots, bspline_degree)
    layout.interpolation_steps = horizon // padded_n_knots

    layout.knots_per_warp = WARP_SIZE // layout.interpolation_steps
    layout.warps_for_n_knots = ceil_div(n_knots, layout.knots_per_warp)
    layout.threads_for_n_knots = layout.warps_for_n_knots * WARP_SIZE
    layout.padded_horizon = horizon + 1

    layout.n_knots = n_knots
    layout.padded_n_knots = padded_n_knots
    layout.horizon = horizon
    layout.dof = dof

    return layout


class BSplineLaunchCfg:
    """Helper class for calculating launch configurations for B-spline trajectory kernels"""

    @staticmethod
    def calculate_forward_config(batch_size: int, dof: int, horizon: int) -> LaunchConfig:
        """Calculate launch configuration for B-spline interpolation forward kernel.

        Args:
            batch_size: Number of batches
            dof: Degrees of freedom
            horizon: Trajectory horizon

        Returns:
            LaunchConfig for kernel launch
        """
        k_size = batch_size * dof * horizon
        max_threads = 128
        threads_per_block = min(k_size, max_threads)
        blocks_per_grid = ceil_div(k_size, threads_per_block)

        return LaunchConfig(grid=blocks_per_grid, block=threads_per_block, shmem_size=0)

    @staticmethod
    def calculate_backward_config(
        batch_size: int, dof: int, n_knots: int, horizon: int, bspline_degree: int
    ) -> LaunchConfig:
        """Calculate launch configuration for B-spline interpolation backward kernel.

        Args:
            batch_size: Number of batches
            dof: Degrees of freedom
            n_knots: Number of B-spline knots
            horizon: Trajectory horizon (after padded_horizon - 1 conversion)
            bspline_degree: B-spline degree (3, 4, or 5)

        Returns:
            LaunchConfig for kernel launch
        """
        # Compute layout to get correct thread organization
        layout = compute_bspline_backward_layout(horizon, dof, n_knots, bspline_degree)

        # Validate interpolation_steps doesn't exceed warp size
        WARP_SIZE = 32
        if layout.interpolation_steps > WARP_SIZE:
            raise RuntimeError(
                f"interpolation_steps ({layout.interpolation_steps}) > {WARP_SIZE} is not supported"
            )

        # Use threads_for_n_knots instead of n_knots directly
        k_size = batch_size * dof * layout.threads_for_n_knots
        max_threads = 128
        threads_per_block = min(k_size, max_threads)
        blocks_per_grid = ceil_div(k_size, threads_per_block)

        return LaunchConfig(grid=blocks_per_grid, block=threads_per_block, shmem_size=0)

    @staticmethod
    def calculate_single_dt_config(batch_size: int, dof: int, max_out_tsteps: int) -> LaunchConfig:
        """Calculate launch configuration for B-spline single dt kernel.

        Args:
            batch_size: Number of batches
            dof: Degrees of freedom
            max_out_tsteps: Maximum output timesteps

        Returns:
            LaunchConfig for kernel launch
        """
        k_size = batch_size * dof * max_out_tsteps
        max_threads = 256
        threads_per_block = min(k_size, max_threads)
        blocks_per_grid = ceil_div(k_size, threads_per_block)

        return LaunchConfig(grid=blocks_per_grid, block=threads_per_block, shmem_size=0)


class LegacyTrajectoryLaunchCfg:
    """Helper class for calculating launch configurations for legacy trajectory kernels"""

    @staticmethod
    def calculate_differentiation_forward_config(
        batch_size: int, dof: int, horizon: int
    ) -> LaunchConfig:
        """Calculate launch configuration for differentiation position forward kernel.

        Args:
            batch_size: Number of batches
            dof: Degrees of freedom
            horizon: Trajectory horizon

        Returns:
            LaunchConfig for kernel launch
        """
        k_size = batch_size * dof * horizon
        threads_per_block = min(k_size, 128)
        blocks_per_grid = ceil_div(k_size, threads_per_block)

        return LaunchConfig(grid=blocks_per_grid, block=threads_per_block, shmem_size=0)

    @staticmethod
    def calculate_differentiation_backward_config(
        batch_size: int, dof: int, horizon: int
    ) -> LaunchConfig:
        """Calculate launch configuration for differentiation position backward kernel.

        Args:
            batch_size: Number of batches
            dof: Degrees of freedom
            horizon: Trajectory horizon

        Returns:
            LaunchConfig for kernel launch
        """
        k_size = batch_size * dof * (horizon - 4)
        threads_per_block = min(k_size, 128)
        blocks_per_grid = ceil_div(k_size, threads_per_block)

        return LaunchConfig(grid=blocks_per_grid, block=threads_per_block, shmem_size=0)

    @staticmethod
    def calculate_integration_config(batch_size: int, dof: int) -> LaunchConfig:
        """Calculate launch configuration for integration acceleration kernel.

        Args:
            batch_size: Number of batches
            dof: Degrees of freedom

        Returns:
            LaunchConfig for kernel launch
        """
        k_size = batch_size * dof
        threads_per_block = min(k_size, 512)
        blocks_per_grid = ceil_div(k_size, threads_per_block)

        return LaunchConfig(grid=blocks_per_grid, block=threads_per_block, shmem_size=0)
