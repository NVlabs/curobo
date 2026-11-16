# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import List

from cuda.core import LaunchConfig

from curobo._src.curobolib.backends.cuda_core_backend.kernel_config import CudaCoreKernelCfg


class KinematicsKernelCfg(CudaCoreKernelCfg):
    """Configuration for kinematics kernel compilation and launching"""

    def __init__(self):
        super().__init__("kinematics")

    def get_kernel_files(self, kernel_type: str) -> List[str]:
        """Get kernel source files for a given kernel type.

        Args:
            kernel_type: Type of kernel ("forward", "backward")

        Returns:
            List of kernel filenames
        """
        kernel_files = {
            # Only include main kernel files - they have #include directives for helpers.
            "forward": ["kinematics_forward_kernel.cuh"],
            "backward": ["kinematics_backward_kernel.cuh"],
        }
        return kernel_files.get(kernel_type, [])

    def get_include_dirs(self) -> List[Path]:
        """Get include directories for kernel compilation"""
        # Get base include dirs and add kinematics-specific ones
        base_dirs = self.get_base_include_dirs()
        kinematics_dirs = [
            self.kernel_dir,  # kernels/kinematics/
        ]
        return base_dirs + kinematics_dirs


class KinematicsLaunchCfg:
    """Helper class for calculating launch configurations for kinematics kernels"""

    MAX_FW_BATCH_PER_BLOCK = 8
    MAX_JACOBIAN_VJP_BATCH_PER_BLOCK = 4
    DEFAULT_MAX_THREADS = 128
    DEFAULT_MAX_SHARED_MEM = 48 * 1024  # 48 KB
    MAX_FORWARD_LINKS = 500

    @staticmethod
    def calculate_forward_config(
        batch_size: int,
        num_links: int,
        max_threads: int = None,
        max_shared_mem: int = None,
    ) -> LaunchConfig:
        """Calculate launch configuration for the forward kinematics kernels."""
        threads_per_batch = 32
        max_threads = max_threads or KinematicsLaunchCfg.DEFAULT_MAX_THREADS
        max_shared_mem = max_shared_mem or KinematicsLaunchCfg.DEFAULT_MAX_SHARED_MEM

        if num_links > KinematicsLaunchCfg.MAX_FORWARD_LINKS:
            raise RuntimeError(
                "Forward kinematics kernel supports at most "
                f"{KinematicsLaunchCfg.MAX_FORWARD_LINKS} links"
            )

        shared_mem_per_batch = num_links * 12 * 4 * 2

        if shared_mem_per_batch > max_shared_mem:
            raise RuntimeError("Single batch shared memory requirement exceeds limit")

        max_batches_per_block = KinematicsLaunchCfg.MAX_FW_BATCH_PER_BLOCK
        max_batches_from_shared_mem = max_shared_mem // shared_mem_per_batch
        max_batches_per_block = min(max_batches_per_block, max_batches_from_shared_mem)

        if max_batches_per_block * threads_per_batch > max_threads:
            max_batches_per_block = max_threads // threads_per_batch

        max_batches_per_block = max(1, max_batches_per_block)
        batches_per_block = min(batch_size, max_batches_per_block)

        threads_per_block = batches_per_block * threads_per_batch
        blocks_per_grid = (batch_size + batches_per_block - 1) // batches_per_block
        shared_mem_size = batches_per_block * num_links * 12 * 4 * 2

        return LaunchConfig(
            grid=blocks_per_grid,
            block=threads_per_block,
            shmem_size=shared_mem_size,
        )

    @staticmethod
    def calculate_jacobian_vjp_config(
        batch_size: int,
        num_links: int,
        max_threads: int = None,
        max_shared_mem: int = None,
    ) -> LaunchConfig:
        """Calculate the launch configuration for the 32-lane Jacobian VJP."""
        threads_per_batch = 32
        max_threads = max_threads or KinematicsLaunchCfg.DEFAULT_MAX_THREADS
        max_shared_mem = max_shared_mem or KinematicsLaunchCfg.DEFAULT_MAX_SHARED_MEM
        shared_mem_per_batch = num_links * 12 * 4
        if shared_mem_per_batch > max_shared_mem:
            raise RuntimeError("Single batch shared memory requirement exceeds limit")

        max_batches_per_block = min(
            KinematicsLaunchCfg.MAX_JACOBIAN_VJP_BATCH_PER_BLOCK,
            max_shared_mem // shared_mem_per_batch,
            max_threads // threads_per_batch,
        )
        max_batches_per_block = max(1, max_batches_per_block)
        batches_per_block = min(batch_size, max_batches_per_block)
        return LaunchConfig(
            grid=(batch_size + batches_per_block - 1) // batches_per_block,
            block=batches_per_block * threads_per_batch,
            shmem_size=batches_per_block * shared_mem_per_batch,
        )
