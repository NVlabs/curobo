# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import List, Tuple

from cuda.core import LaunchConfig

from curobo._src.curobolib.backends.cuda_core_backend.kernel_config import CudaCoreKernelCfg


class KinematicsKernelCfg(CudaCoreKernelCfg):
    """Configuration for kinematics kernel compilation and launching"""

    def __init__(self):
        super().__init__("kinematics")

    def get_kernel_files(self, kernel_type: str) -> List[str]:
        """Get kernel source files for a given kernel type.

        Args:
            kernel_type: Type of kernel ("forward", "backward", "jacobian_backward")

        Returns:
            List of kernel filenames
        """
        kernel_files = {
            # Only include main kernel file - it has #include directives for the others
            "forward": ["kinematics_forward_kernel.cuh"],
            "backward": ["kinematics_backward_kernel.cuh"],
            "jacobian_backward": ["kinematics_jacobian_backward_kernel.cuh"],
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
    MAX_BW_BATCH_PER_BLOCK = 32  # From C++ code
    DEFAULT_MAX_THREADS = 128
    DEFAULT_MAX_SHARED_MEM = 48 * 1024  # 48 KB

    @staticmethod
    def calculate_forward_config(
        batch_size: int,
        num_links: int,
        num_spheres: int,
        n_tool_frames: int,
        compute_jacobian: bool,
        max_threads: int = None,
        max_shared_mem: int = None,
    ) -> Tuple[LaunchConfig, LaunchConfig]:
        """Calculate launch configuration for kinematics forward kernel.

        Ported from calculate_kinematics_forward_launch_config in
        kinematics_forward_kernel_launch.cu (lines 32-131)

        Args:
            batch_size: Number of batches
            num_links: Number of links
            num_spheres: Number of spheres
            n_tool_frames: Number of links to store
            compute_jacobian: Whether computing jacobian
            max_threads: Maximum threads per block (default 128)
            max_shared_mem: Maximum shared memory (default 48KB)

        Returns:
            Tuple of LaunchConfig for single or separate kernels
        """
        max_threads = max_threads or KinematicsLaunchCfg.DEFAULT_MAX_THREADS
        max_shared_mem = max_shared_mem or KinematicsLaunchCfg.DEFAULT_MAX_SHARED_MEM

        # Determine execution strategy based on workload size
        use_single_kernel = num_spheres < 100

        if use_single_kernel:
            # Single fused kernel configuration
            max_batches_per_block = KinematicsLaunchCfg.MAX_FW_BATCH_PER_BLOCK
            threads_per_batch = 4

            # Check if even 1 batch per block would fit in shared memory
            # Include space for cumul_mat + center of mass data
            shared_mem_per_batch = num_links * 12 * 4 + threads_per_batch * 16  # sizeof(float4) = 16

            if shared_mem_per_batch > max_shared_mem:
                raise RuntimeError("Single batch shared memory requirement exceeds limit")

            # Limit batches per block based on shared memory constraints
            max_batches_from_shared_mem = max_shared_mem // shared_mem_per_batch
            max_batches_per_block = min(max_batches_per_block, max_batches_from_shared_mem)

            # Limit threads per block based on input parameter for occupancy
            if max_batches_per_block * threads_per_batch > max_threads:
                max_batches_per_block = max_threads // threads_per_batch

            # Ensure at least 1 batch per block
            max_batches_per_block = max(1, max_batches_per_block)

            # Actual batches per block (limited by batch_size)
            batches_per_block = min(batch_size, max_batches_per_block)

            threads_per_block = batches_per_block * threads_per_batch
            blocks_per_grid = (
                batch_size * threads_per_batch + threads_per_block - 1
            ) // threads_per_block
            shared_mem_size = (
                batches_per_block * num_links * 12 * 4 + batches_per_block * threads_per_batch * 16
            )
            config = LaunchConfig(
                grid=blocks_per_grid, block=threads_per_block, shmem_size=shared_mem_size
            )

            return config, None

        else:
            # Separate kernels configuration
            # First kernel (cumulative transform)
            max_batches_per_block = KinematicsLaunchCfg.MAX_FW_BATCH_PER_BLOCK
            threads_per_batch = 4

            shared_mem_per_batch = num_links * 12 * 4 + threads_per_batch * 16

            if shared_mem_per_batch > max_shared_mem:
                raise RuntimeError("Single batch shared memory requirement exceeds limit")

            max_batches_from_shared_mem = max_shared_mem // shared_mem_per_batch
            max_batches_per_block = min(max_batches_per_block, max_batches_from_shared_mem)

            if max_batches_per_block * threads_per_batch > max_threads:
                max_batches_per_block = max_threads // threads_per_batch

            max_batches_per_block = max(1, max_batches_per_block)
            batches_per_block = min(batch_size, max_batches_per_block)

            threads_per_block = batches_per_block * threads_per_batch
            blocks_per_grid = (
                batch_size * threads_per_batch + threads_per_block - 1
            ) // threads_per_block
            shared_mem_size = (
                batches_per_block * num_links * 12 * 4 + batches_per_block * threads_per_batch * 16
            )

            config_cumul = LaunchConfig(
                grid=blocks_per_grid, block=threads_per_block, shmem_size=shared_mem_size
            )

            # Second kernel (spheres and links) configuration
            estimated_threads_per_batch = min(128, max(num_spheres, n_tool_frames))
            separate_shared_mem_per_batch = num_links * 12 * 4 + estimated_threads_per_batch * 16

            separate_max_batches_per_block = max_shared_mem // separate_shared_mem_per_batch
            separate_max_batches_per_block = min(separate_max_batches_per_block, 4)
            separate_max_batches_per_block = max(1, separate_max_batches_per_block)

            separate_batches_per_block = separate_max_batches_per_block
            separate_threads_per_batch = min(128, max(num_spheres, n_tool_frames))
            separate_shared_mem_size = (
                separate_batches_per_block * num_links * 12 * 4
                + separate_batches_per_block * separate_threads_per_batch * 16
            )
            separate_threads_per_block = separate_batches_per_block * separate_threads_per_batch
            separate_blocks_per_grid = (
                batch_size + separate_batches_per_block - 1
            ) // separate_batches_per_block

            config_spheres_links = LaunchConfig(
                grid=separate_blocks_per_grid,
                block=separate_threads_per_block,
                shmem_size=separate_shared_mem_size,
            )

            return config_cumul, config_spheres_links

    @staticmethod
    def calculate_backward_config(
        batch_size: int,
        num_links: int,
        num_spheres: int,
        n_tool_frames: int,
        n_joints: int,
        max_threads: int = None,
        max_shared_mem: int = None,
    ) -> Tuple[LaunchConfig, int, bool, int]:
        """Calculate launch configuration for kinematics backward kernel.

        Ported from calculate_kinematics_backward_launch_config in
        kinematics_backward_kernel_launch.cu (lines 20-74)

        Args:
            batch_size: Number of batches
            num_links: Number of links
            num_spheres: Number of spheres
            n_tool_frames: Number of links to store
            n_joints: Number of joints
            max_threads: Maximum threads per block (default 128)
            max_shared_mem: Maximum shared memory (default 48KB)

        Returns:
            Tuple of (LaunchConfig, threads_per_batch, use_warp_reduce, max_joints_template)
        """
        max_threads = max_threads or KinematicsLaunchCfg.DEFAULT_MAX_THREADS
        max_shared_mem = max_shared_mem or KinematicsLaunchCfg.DEFAULT_MAX_SHARED_MEM

        use_warp_reduce = num_spheres < 5000

        # Select max_joints template parameter based on n_joints
        if n_joints < 16:
            max_joints_template = 16
        elif n_joints < 64:
            max_joints_template = 64
        else:
            max_joints_template = 128

        if use_warp_reduce:
            # Warp reduction configuration
            max_batches_per_block = KinematicsLaunchCfg.MAX_BW_BATCH_PER_BLOCK
            max_threads_per_batch = 32

            # Check if even 1 batch per block would fit in shared memory
            shared_mem_per_batch = num_links * 12 * 4  # sizeof(float) = 4
            if shared_mem_per_batch > max_shared_mem:
                raise RuntimeError("Single batch shared memory requirement exceeds limit")

            # Limit batches per block based on shared memory constraints
            max_batches_from_shared_mem = max_shared_mem // shared_mem_per_batch
            max_batches_per_block = min(max_batches_per_block, max_batches_from_shared_mem)

            # Limit threads per block based on input parameter for occupancy
            if max_batches_per_block * max_threads_per_batch > max_threads:
                max_batches_per_block = max_threads // max_threads_per_batch

            # Ensure at least 1 batch per block
            max_batches_per_block = max(1, max_batches_per_block)

            # Actual batches per block (limited by batch_size)
            batches_per_block = min(batch_size, max_batches_per_block)

            threads_per_batch = max_threads_per_batch
            threads_per_block = batches_per_block * max_threads_per_batch
            blocks_per_grid = (
                batch_size * max_threads_per_batch + threads_per_block - 1
            ) // threads_per_block
            shared_mem_size = batches_per_block * num_links * 12 * 4
        else:
            # Block reduction configuration
            required_shared_mem = num_links * 12 * 4
            if required_shared_mem > max_shared_mem:
                raise RuntimeError("Single batch shared memory requirement exceeds limit")

            threads_per_block = min(max_threads, max(32, max(num_spheres, n_tool_frames)))
            blocks_per_grid = batch_size
            shared_mem_size = required_shared_mem
            threads_per_batch = threads_per_block

        config = LaunchConfig(
            grid=blocks_per_grid, block=threads_per_block, shmem_size=shared_mem_size
        )

        return config, threads_per_batch, use_warp_reduce, max_joints_template

    @staticmethod
    def calculate_jacobian_backward_config(
        batch_size: int,
        num_links: int,
        n_tool_frames: int,
        n_joints: int,
        max_threads: int = None,
        max_shared_mem: int = None,
    ) -> Tuple[LaunchConfig, int, bool, int]:
        """Calculate launch configuration for kinematics jacobian backward kernel.

        Ported from calculate_kinematics_jacobian_backward_launch_config in
        kinematics_backward_jacobian_kernel_launch.cu (lines 33-86)

        Args:
            batch_size: Number of batches
            num_links: Number of links
            n_tool_frames: Number of links to store
            n_joints: Number of joints
            max_threads: Maximum threads per block (default 128)
            max_shared_mem: Maximum shared memory (default 48KB)

        Returns:
            Tuple of (LaunchConfig, threads_per_batch, use_warp_reduce, max_joints_template)
        """
        max_threads = max_threads or KinematicsLaunchCfg.DEFAULT_MAX_THREADS
        max_shared_mem = max_shared_mem or KinematicsLaunchCfg.DEFAULT_MAX_SHARED_MEM

        use_warp_reduce = n_tool_frames < 5000

        # Select max_joints template parameter based on n_joints
        if n_joints < 16:
            max_joints_template = 16
        elif n_joints < 64:
            max_joints_template = 64
        else:
            max_joints_template = 128

        if use_warp_reduce:
            # Warp reduction configuration
            max_batches_per_block = KinematicsLaunchCfg.MAX_BW_BATCH_PER_BLOCK
            max_threads_per_batch = 32

            # Check if even 1 batch per block would fit in shared memory
            shared_mem_per_batch = num_links * 12 * 4  # sizeof(float) = 4
            if shared_mem_per_batch > max_shared_mem:
                raise RuntimeError("Single batch shared memory requirement exceeds limit")

            # Limit batches per block based on shared memory constraints
            max_batches_from_shared_mem = max_shared_mem // shared_mem_per_batch
            max_batches_per_block = min(max_batches_per_block, max_batches_from_shared_mem)

            # Limit threads per block based on input parameter for occupancy
            if max_batches_per_block * max_threads_per_batch > max_threads:
                max_batches_per_block = max_threads // max_threads_per_batch

            # Ensure at least 1 batch per block
            max_batches_per_block = max(1, max_batches_per_block)

            # Actual batches per block (limited by batch_size)
            batches_per_block = min(batch_size, max_batches_per_block)

            threads_per_batch = max_threads_per_batch
            threads_per_block = batches_per_block * max_threads_per_batch
            blocks_per_grid = (
                batch_size * max_threads_per_batch + threads_per_block - 1
            ) // threads_per_block
            shared_mem_size = batches_per_block * num_links * 12 * 4
        else:
            # Block reduction configuration
            required_shared_mem = num_links * 12 * 4
            if required_shared_mem > max_shared_mem:
                raise RuntimeError("Single batch shared memory requirement exceeds limit")

            threads_per_block = min(max_threads, max(32, n_tool_frames))
            blocks_per_grid = batch_size
            shared_mem_size = required_shared_mem
            threads_per_batch = threads_per_block

        config = LaunchConfig(
            grid=blocks_per_grid, block=threads_per_block, shmem_size=shared_mem_size
        )

        return config, threads_per_batch, use_warp_reduce, max_joints_template
