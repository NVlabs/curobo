# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for optimization kernel compilation."""

# Standard Library
from pathlib import Path
from typing import List, Optional, Tuple

from cuda.core import LaunchConfig

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend.kernel_config import CudaCoreKernelCfg


class OptimizationKernelCfg(CudaCoreKernelCfg):
    """Configuration for optimization kernel compilation."""

    def __init__(self):
        """Initialize optimization kernel paths and compiler settings."""
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
            "lbfgs_compact": ["lbfgs/lbfgs_compact_step_kernel.cuh"],
            "pose_refiner_lm": ["pose_refiner/pose_refiner_lm_kernel.cuh"],
        }
        return kernel_files.get(kernel_type, [])

    def get_include_dirs(self) -> List[Path]:
        """Get include directories for kernel compilation."""
        # Get base include dirs and add optimization-specific ones
        base_dirs = self.get_base_include_dirs()
        optimization_dirs = [
            self.kernel_dir,  # kernels/optimization/
            self.kernel_dir / "line_search",
            self.kernel_dir / "lbfgs",
            self.kernel_dir / "pose_refiner",
        ]
        return base_dirs + optimization_dirs


class LineSearchLaunchCfg:
    """Calculate launch configurations for line-search kernels."""

    MAX_THREADS_PER_BLOCK = 1024

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
        rounded_threads = ((opt_dim + 31) // 32) * 32
        threads_per_block = min(rounded_threads, LineSearchLaunchCfg.MAX_THREADS_PER_BLOCK)
        blocks_per_grid = batchsize

        return LaunchConfig(grid=blocks_per_grid, block=threads_per_block, shmem_size=0)


class LBFGSLaunchCfg:
    """Calculate launch configurations for L-BFGS kernels."""

    SCRATCH_HISTORY = 128

    @staticmethod
    def calculate_items_and_threads(v_dim: int) -> Tuple[int, int]:
        """Return register items per thread and the matching block width."""
        if v_dim <= 512:
            items_per_thread = 1
        elif v_dim <= 1024:
            items_per_thread = 2
        elif v_dim <= 2048:
            items_per_thread = 4
        elif v_dim <= 4096:
            items_per_thread = 8
        elif v_dim <= 8192:
            return 16, 512
        else:
            return 0, 512

        threads_per_block = ((v_dim + 32 * items_per_thread - 1) // (32 * items_per_thread)) * 32
        return items_per_thread, threads_per_block

    @staticmethod
    def calculate_config(
        batch_size: int,
        v_dim: int,
        history_m: int,
        use_shared_buffers: bool,
        device_max_shared_memory: int,
    ) -> Tuple[LaunchConfig, bool, int]:
        """Calculate launch configuration for LBFGS step kernel.

        Ported from calculate_lbfgs_launch_config in lbfgs_step_kernel_launch.cu (lines 93-133)

        Args:
            batch_size: Number of batches
            v_dim: Variable dimension
            history_m: History size
            use_shared_buffers: Whether to stage history in shared memory when it fits.
            device_max_shared_memory: Device opt-in shared-memory limit in bytes.

        Returns:
            Tuple of (LaunchConfig, use_shared_buffers_actual, dynamic_shared_memory)
        """
        _, threads_per_block = LBFGSLaunchCfg.calculate_items_and_threads(v_dim)
        blocks_per_grid = batch_size

        # Each reduction uses one slot per active warp. Distinct slots avoid a
        # second barrier without reserving scratch for inactive warps.
        reduction_shared_memory = (
            0 if threads_per_block == 32 else (2 * history_m + 3) * (threads_per_block // 32) * 4
        )
        static_shared_memory = (2 * LBFGSLaunchCfg.SCRATCH_HISTORY + 1) * 4
        history_shared_memory = 2 * v_dim * history_m * 4
        use_shared_buffers_actual = use_shared_buffers and (
            history_shared_memory + reduction_shared_memory + static_shared_memory
            <= device_max_shared_memory
        )
        shared_mem_size = reduction_shared_memory + (
            history_shared_memory if use_shared_buffers_actual else 0
        )

        config = LaunchConfig(
            grid=blocks_per_grid, block=threads_per_block, shmem_size=shared_mem_size
        )

        return config, use_shared_buffers_actual, shared_mem_size


class LBFGSCompactLaunchCfg:
    """Calculate launch and workspace configuration for compact L-BFGS."""

    THREADS_PER_BLOCK = 256
    MIN_HISTORY = 1
    MAX_HISTORY = 128
    DIMENSION_ALIGNMENT = 8
    STATIC_SHARED_MEMORY_MARGIN = 64

    @staticmethod
    def calculate_config(
        batch_size: int,
        v_dim: int,
        history_m: int,
        device_max_shared_memory: int,
    ) -> Optional[Tuple[LaunchConfig, int, int]]:
        """Return launch config, shared-matrix count, and global workspace size.

        Compact input panels reside in shared memory. A value of ``None``
        requests the existing two-loop implementation.

        Args:
            batch_size: Number of independent optimization problems.
            v_dim: Number of optimization variables per problem.
            history_m: Number of L-BFGS history entries.
            device_max_shared_memory: Per-block shared-memory limit in bytes.

        Returns:
            A tuple containing the launch configuration, number of compact
            matrices stored in shared memory, and global workspace elements
            per problem, or ``None`` when the shape is unsupported.
        """
        if (
            history_m < LBFGSCompactLaunchCfg.MIN_HISTORY
            or history_m > LBFGSCompactLaunchCfg.MAX_HISTORY
            or v_dim < 1
        ):
            return None

        padded_history = ((history_m + 15) // 16) * 16
        padded_v_dim = (
            (v_dim + LBFGSCompactLaunchCfg.DIMENSION_ALIGNMENT - 1)
            // LBFGSCompactLaunchCfg.DIMENSION_ALIGNMENT
        ) * LBFGSCompactLaunchCfg.DIMENSION_ALIGNMENT
        matrix_elements = padded_history * padded_history
        panel_elements = 2 * padded_history * padded_v_dim
        base_shared_elements = (
            panel_elements
            + 5 * padded_history
            + 3 * (LBFGSCompactLaunchCfg.THREADS_PER_BLOCK // 32)
        )

        for shared_matrix_count in (3, 2, 1, 0):
            extra_shared_elements = shared_matrix_count * matrix_elements
            shared_memory = 4 * (base_shared_elements + extra_shared_elements)
            if (
                shared_memory + LBFGSCompactLaunchCfg.STATIC_SHARED_MEMORY_MARGIN
                <= device_max_shared_memory
            ):
                global_workspace_elements = (3 - shared_matrix_count) * matrix_elements
                return (
                    LaunchConfig(
                        grid=batch_size,
                        block=LBFGSCompactLaunchCfg.THREADS_PER_BLOCK,
                        shmem_size=shared_memory,
                    ),
                    shared_matrix_count,
                    global_workspace_elements,
                )

        return None
