# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for optimization kernel compilation."""

# Standard Library
from pathlib import Path
from typing import List, Tuple

from cuda.core import LaunchConfig

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend.kernel_config import CudaCoreKernelCfg


class OptimizationKernelCfg(CudaCoreKernelCfg):
    """Configuration for optimization kernel compilation"""

    def __init__(self):
        super().__init__("optimization")

    def get_kernel_files(self, kernel_type: str) -> List[str]:
        """Get kernel source files for a given kernel type.

        Args:
            kernel_type: Type of kernel ("line_search", "lbfgs")

        Returns:
            List of kernel filenames
        """
        kernel_files = {
            "line_search": ["line_search/line_search_kernel.cuh"],
            "lbfgs": ["lbfgs/lbfgs_step_kernel.cuh"],
        }
        return kernel_files.get(kernel_type, [])

    def get_include_dirs(self) -> List[Path]:
        """Get include directories for kernel compilation"""
        # Get base include dirs and add optimization-specific ones
        base_dirs = self.get_base_include_dirs()
        optimization_dirs = [
            self.kernel_dir,  # kernels/optimization/
            self.kernel_dir / "line_search",
            self.kernel_dir / "lbfgs",
        ]
        return base_dirs + optimization_dirs


class LineSearchLaunchCfg:
    """Helper class for calculating launch configurations for line search kernels"""

    @staticmethod
    def calculate_config(opt_dim: int, batchsize: int) -> LaunchConfig:
        """Calculate launch configuration for line search kernel.

        Ported from launch_line_search in line_search_kernel_launch.cu (lines 94-98)

        Args:
            opt_dim: Optimization dimension (number of parameters)
            batchsize: Number of parallel searches

        Returns:
            LaunchConfig for kernel launch
        """
        threads_per_block = opt_dim
        blocks_per_grid = batchsize

        return LaunchConfig(grid=blocks_per_grid, block=threads_per_block, shmem_size=0)


class LBFGSLaunchCfg:
    """Helper class for calculating launch configurations for LBFGS kernels"""

    @staticmethod
    def calculate_config(
        batch_size: int, v_dim: int, history_m: int, use_shared_buffers: bool
    ) -> Tuple[LaunchConfig, bool, int]:
        """Calculate launch configuration for LBFGS step kernel.

        Ported from calculate_lbfgs_launch_config in lbfgs_step_kernel_launch.cu (lines 93-133)

        Args:
            batch_size: Number of batches
            v_dim: Variable dimension
            history_m: History size
            use_shared_buffers: Whether to use shared memory buffers

        Returns:
            Tuple of (LaunchConfig, use_shared_buffers_actual, max_shared_memory_needed)
        """
        threads_per_block = v_dim
        blocks_per_grid = batch_size

        # Calculate shared memory requirements
        basic_smem_size = history_m * 4  # sizeof(float) = 4
        shared_buffer_smem_size = (((2 * v_dim) + 2) * history_m + 32 + 1) * 4

        # Shared memory limits
        max_shared_base = 48000
        max_shared_allowed = 65536  # Turing/Volta+ limit

        # Determine if we can use shared buffers
        use_shared_buffers_actual = False
        shared_mem_size = basic_smem_size
        max_shared_memory_needed = max_shared_base

        if use_shared_buffers:
            shared_mem_size = shared_buffer_smem_size

            # Check if we can fit in base shared memory
            if shared_buffer_smem_size <= max_shared_base:
                use_shared_buffers_actual = True
                max_shared_memory_needed = shared_buffer_smem_size
            # Check if we need extended shared memory (Volta+)
            elif shared_buffer_smem_size <= max_shared_allowed:
                use_shared_buffers_actual = True
                max_shared_memory_needed = shared_buffer_smem_size
                # Note: Caller must configure cudaFuncAttributeMaxDynamicSharedMemorySize
        else:
            shared_mem_size = basic_smem_size

        config = LaunchConfig(
            grid=blocks_per_grid, block=threads_per_block, shmem_size=shared_mem_size
        )

        return config, use_shared_buffers_actual, max_shared_memory_needed
