# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for CUDA rigid-body dynamics kernel compilation and launching."""

# Standard Library
from pathlib import Path
from typing import List

# Third Party
from cuda.core import LaunchConfig

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend.kernel_config import CudaCoreKernelCfg


class DynamicsKernelCfg(CudaCoreKernelCfg):
    """Configuration for rigid-body dynamics kernel compilation."""

    def __init__(self) -> None:
        """Initialize the dynamics kernel source configuration."""
        super().__init__("dynamics")

    def get_kernel_files(self, kernel_type: str) -> List[str]:
        """Get kernel source files for a given kernel type.

        Args:
            kernel_type: Type of dynamics kernel.

        Returns:
            List of kernel filenames.
        """
        kernel_files = {
            "forward": ["rnea_forward_kernel.cuh"],
            "backward": ["rnea_backward_kernel.cuh"],
            "mass_matrix": ["crba_forward_kernel.cuh"],
            "mass_matrix_backward": ["crba_backward_kernel.cuh"],
        }
        return kernel_files.get(kernel_type, [])

    def get_include_dirs(self) -> List[Path]:
        """Get include directories for kernel compilation.

        Includes base dirs (common/, third_party/) plus the dynamics and
        kinematics directories (for kinematics_constants.h).
        """
        base_dirs = self.get_base_include_dirs()
        extra_dirs = [
            self.kernel_dir,  # kernels/dynamics/
            self.kernel_dir.parent / "kinematics",  # kernels/kinematics/
        ]
        return base_dirs + extra_dirs


class DynamicsLaunchCfg:
    """Helper class for calculating launch configurations for RNEA kernels.

    RNEA defaults select state packing from the runtime cooperative traversal
    width, batch size, and effective SM count. Explicit experimental caps still
    use the occupancy helper below.

    The helper :meth:`_warp_align_batches` picks the warp-aligned batch count that
    maximises the number of resident blocks on an SM (given the shared-memory budget).
    """

    DEFAULT_SERIAL_FORWARD_BATCHES_PER_BLOCK = None
    DEFAULT_COOPERATIVE_FORWARD_BATCHES_PER_BLOCK = None
    DEFAULT_MASS_MATRIX_BATCHES_PER_BLOCK = 2
    DEFAULT_MASS_MATRIX_BACKWARD_BATCHES_PER_BLOCK = 1
    DEFAULT_SERIAL_BACKWARD_BATCHES_PER_BLOCK = None
    DEFAULT_COOPERATIVE_BACKWARD_BATCHES_PER_BLOCK = None

    DEFAULT_MAX_SHARED_MEM = 48 * 1024  # 48 KB
    # Total configurable L1/shared-memory per SM (Ampere=128 KB, Hopper=228 KB).
    # Used to check how many blocks can co-reside on one SM.
    DEFAULT_SM_SHARED_MEM_CAPACITY = 100 * 1024  # conservative for Ampere

    WARP_SIZE = 32

    @staticmethod
    def _default_forward_batches_per_block(
        batch_size: int,
        threads_per_batch: int,
        policy_sm_count: int = None,
    ) -> int:
        """Return the measured forward state-packing cap."""
        if threads_per_batch >= 32:
            return 2
        if threads_per_batch >= 16:
            return 4 if batch_size <= 4096 else 8
        if threads_per_batch >= 4:
            return 4
        if threads_per_batch == 2:
            return 8
        if policy_sm_count is not None and policy_sm_count <= 32:
            return 8
        return 16

    @staticmethod
    def _default_backward_batches_per_block(
        batch_size: int,
        threads_per_batch: int,
        policy_sm_count: int = None,
    ) -> int:
        """Return the measured backward state-packing cap."""
        if threads_per_batch >= 32:
            return 2
        if threads_per_batch >= 16:
            return 2 if batch_size <= 4096 else 8
        if threads_per_batch >= 4:
            if threads_per_batch == 4 and 2048 < batch_size <= 4096:
                return 8
            return 4
        if threads_per_batch == 2:
            return 8
        if policy_sm_count is not None and policy_sm_count <= 32:
            return 8
        return 16

    @staticmethod
    def calculate_mass_matrix_config(
        batch_size: int,
        num_links: int,
        threads_per_batch: int,
        max_batches_per_block: int = None,
        max_shared_mem: int = None,
    ) -> LaunchConfig:
        """Calculate the launch configuration for the CRBA kernel.

        Shared memory stores one 6x6 composite inertia and one 12-float local
        transform per retained body and batch element.

        Args:
            batch_size: Number of robot states.
            num_links: Number of retained rigid bodies.
            threads_per_batch: Cooperative threads assigned to each robot state.
            max_batches_per_block: Optional maximum states assigned to one block.
            max_shared_mem: Optional shared-memory limit in bytes.

        Returns:
            CUDA launch configuration for CRBA.
        """
        max_bpb = (
            DynamicsLaunchCfg.DEFAULT_MASS_MATRIX_BATCHES_PER_BLOCK
            if max_batches_per_block is None
            else max_batches_per_block
        )
        max_shared_mem = max_shared_mem or DynamicsLaunchCfg.DEFAULT_MAX_SHARED_MEM
        shared_memory_per_batch = num_links * 48 * 4
        if shared_memory_per_batch > max_shared_mem:
            raise RuntimeError(
                f"Single batch requires {shared_memory_per_batch} bytes shared memory, "
                f"exceeds limit of {max_shared_mem} bytes"
            )

        batches_per_block = min(
            max_shared_mem // shared_memory_per_batch,
            1024 // threads_per_batch,
            max_bpb,
            batch_size,
        )
        batches_per_block = max(1, batches_per_block)
        blocks_per_grid = (batch_size + batches_per_block - 1) // batches_per_block
        return LaunchConfig(
            grid=blocks_per_grid,
            block=batches_per_block * threads_per_batch,
            shmem_size=batches_per_block * shared_memory_per_batch,
        )

    @staticmethod
    def calculate_mass_matrix_backward_config(
        batch_size: int,
        num_links: int,
        threads_per_batch: int,
        max_batches_per_block: int = None,
        max_shared_mem: int = None,
    ) -> LaunchConfig:
        """Calculate the launch configuration for the analytical CRBA VJP.

        Shared memory stores the composite inertia, its adjoint, and the local
        transform for every retained body.

        Args:
            batch_size: Number of robot states.
            num_links: Number of retained rigid bodies.
            threads_per_batch: Cooperative threads assigned to each robot state.
            max_batches_per_block: Optional maximum states assigned to one block.
            max_shared_mem: Optional shared-memory limit in bytes.

        Returns:
            CUDA launch configuration for the CRBA backward kernel.
        """
        max_bpb = (
            DynamicsLaunchCfg.DEFAULT_MASS_MATRIX_BACKWARD_BATCHES_PER_BLOCK
            if max_batches_per_block is None
            else max_batches_per_block
        )
        max_shared_mem = max_shared_mem or DynamicsLaunchCfg.DEFAULT_MAX_SHARED_MEM
        shared_memory_per_batch = num_links * 84 * 4
        if shared_memory_per_batch > max_shared_mem:
            raise RuntimeError(
                f"Single batch requires {shared_memory_per_batch} bytes shared memory, "
                f"exceeds limit of {max_shared_mem} bytes"
            )

        batches_per_block = min(
            max_shared_mem // shared_memory_per_batch,
            1024 // threads_per_batch,
            max_bpb,
            batch_size,
        )
        batches_per_block = max(1, batches_per_block)
        blocks_per_grid = (batch_size + batches_per_block - 1) // batches_per_block
        return LaunchConfig(
            grid=blocks_per_grid,
            block=batches_per_block * threads_per_batch,
            shmem_size=batches_per_block * shared_memory_per_batch,
        )

    @staticmethod
    def _warp_align_batches(
        batches_per_block: int,
        threads_per_batch: int,
        smem_per_block_fn,
        sm_shared_mem_capacity: int,
    ) -> int:
        """Round batches per block to maximize estimated SM occupancy.

        We try the warp-aligned candidate and the next-lower one and keep whichever
        yields more total active threads across co-resident blocks.

        Args:
            batches_per_block: Raw (un-aligned) batch count.
            threads_per_batch: Threads cooperating on one batch element.
            smem_per_block_fn: ``f(batches) -> bytes`` returning shared memory for
                a block of *batches* batch elements.
            sm_shared_mem_capacity: Total shared memory available on one SM.

        Returns:
            Warp-aligned batch count (≥ 1).
        """
        warp_batch_granularity = max(1, DynamicsLaunchCfg.WARP_SIZE // threads_per_batch)

        # Candidate: round down to nearest warp-aligned batch count
        candidate = (batches_per_block // warp_batch_granularity) * warp_batch_granularity
        if candidate < warp_batch_granularity:
            # Fewer batches than one warp; keep original (partial warp unavoidable)
            return batches_per_block

        # Also consider one step lower (fewer batches ⇒ smaller smem ⇒ more blocks/SM)
        candidate_lower = candidate - warp_batch_granularity
        if candidate_lower < warp_batch_granularity:
            candidate_lower = 0  # skip this option

        best_batches = candidate
        best_score = 0  # total threads across co-resident blocks

        for cand in (candidate, candidate_lower):
            if cand < 1:
                continue
            smem = smem_per_block_fn(cand)
            blocks_per_sm = sm_shared_mem_capacity // smem if smem > 0 else 1
            blocks_per_sm = max(1, blocks_per_sm)
            threads = cand * threads_per_batch
            score = blocks_per_sm * threads
            if score > best_score:
                best_score = score
                best_batches = cand

        return best_batches

    @staticmethod
    def calculate_forward_config(
        batch_size: int,
        num_links: int,
        threads_per_batch: int = 1,
        max_batches_per_block: int = None,
        max_shared_mem: int = None,
        policy_sm_count: int = None,
    ) -> LaunchConfig:
        """Calculate launch configuration for RNEA forward kernel.

        Shared memory per batch = num_links × 24 × 4 bytes (v + a + R,p).

        Args:
            batch_size: Number of batch elements.
            num_links: Number of links in the robot.
            threads_per_batch: Threads per batch element (1 for serial, >1 for tree-parallel).
            max_batches_per_block: Optional override for the topology-specific
                batches-per-block default.
            max_shared_mem: Maximum shared memory per block (default 48 KB).
            policy_sm_count: Effective SM count selected by the runtime policy.

        Returns:
            LaunchConfig for the kernel.
        """
        if max_batches_per_block is None:
            override = (
                DynamicsLaunchCfg.DEFAULT_SERIAL_FORWARD_BATCHES_PER_BLOCK
                if threads_per_batch == 1
                else DynamicsLaunchCfg.DEFAULT_COOPERATIVE_FORWARD_BATCHES_PER_BLOCK
            )
            max_bpb = (
                DynamicsLaunchCfg._default_forward_batches_per_block(
                    batch_size,
                    threads_per_batch,
                    policy_sm_count,
                )
                if override is None
                else override
            )
        else:
            max_bpb = max_batches_per_block
        max_shared_mem = max_shared_mem or DynamicsLaunchCfg.DEFAULT_MAX_SHARED_MEM

        # 24 = velocity(6) + acceleration/force(6) + transform R,p(12)
        smem_per_batch = num_links * 24 * 4

        if smem_per_batch > max_shared_mem:
            raise RuntimeError(
                f"Single batch requires {smem_per_batch} bytes shared memory, "
                f"exceeds limit of {max_shared_mem} bytes"
            )

        # Constraints: shared memory, max batches, hardware thread limit (1024)
        max_batches_from_smem = max_shared_mem // smem_per_batch
        max_batches_from_threads = 1024 // threads_per_batch
        batches_per_block = min(
            max_batches_from_smem, max_batches_from_threads, max_bpb, batch_size
        )
        batches_per_block = max(1, batches_per_block)

        # Round down to warp-aligned count, picking the option that maximises
        # total threads across co-resident blocks on one SM.
        batches_per_block = DynamicsLaunchCfg._warp_align_batches(
            batches_per_block,
            threads_per_batch,
            smem_per_block_fn=lambda b: b * smem_per_batch,
            sm_shared_mem_capacity=DynamicsLaunchCfg.DEFAULT_SM_SHARED_MEM_CAPACITY,
        )

        threads_per_block = batches_per_block * threads_per_batch
        blocks_per_grid = (batch_size + batches_per_block - 1) // batches_per_block
        shared_mem_size = batches_per_block * smem_per_batch

        return LaunchConfig(
            grid=blocks_per_grid,
            block=threads_per_block,
            shmem_size=shared_mem_size,
        )

    @staticmethod
    def calculate_backward_config(
        batch_size: int,
        num_links: int,
        threads_per_batch: int = 1,
        max_batches_per_block: int = None,
        max_shared_mem: int = None,
        policy_sm_count: int = None,
    ) -> LaunchConfig:
        """Calculate launch configuration for RNEA backward (VJP) kernel.

        Shared memory layout:
          - Block-shared: num_links × 12 × 4 bytes (mc + inertia, one copy per block)
          - Per-batch serial: num_links × 30 × 4 bytes
          - Per-batch cooperative: num_links × 44 × 4 bytes, adding aligned R,p

        Args:
            batch_size: Number of batch elements.
            num_links: Number of links in the robot.
            threads_per_batch: Threads per batch element (1 for serial, >1 for tree-parallel).
            max_batches_per_block: Optional override for the topology-specific
                batches-per-block default.
            max_shared_mem: Maximum shared memory per block (default 48 KB).
            policy_sm_count: Effective SM count selected by the runtime policy.

        Returns:
            LaunchConfig for the backward kernel.
        """
        if max_batches_per_block is None:
            override = (
                DynamicsLaunchCfg.DEFAULT_SERIAL_BACKWARD_BATCHES_PER_BLOCK
                if threads_per_batch == 1
                else DynamicsLaunchCfg.DEFAULT_COOPERATIVE_BACKWARD_BATCHES_PER_BLOCK
            )
            max_bpb = (
                DynamicsLaunchCfg._default_backward_batches_per_block(
                    batch_size,
                    threads_per_batch,
                    policy_sm_count,
                )
                if override is None
                else override
            )
        else:
            max_bpb = max_batches_per_block
        max_shared_mem = max_shared_mem or DynamicsLaunchCfg.DEFAULT_MAX_SHARED_MEM

        # Block-shared: mc(4) + inertia(8) = 12 floats per link
        smem_block_shared = num_links * 12 * 4

        # Cooperative kernels stage persisted transforms at an aligned offset.
        smem_floats_per_link = 44 if threads_per_batch > 1 else 30
        smem_per_batch = num_links * smem_floats_per_link * 4

        if smem_block_shared + smem_per_batch > max_shared_mem:
            raise RuntimeError(
                f"Single batch requires {smem_block_shared + smem_per_batch} bytes "
                f"shared memory, exceeds limit of {max_shared_mem} bytes"
            )

        # Available smem for per-batch data after block-shared allocation
        available_smem = max_shared_mem - smem_block_shared
        max_batches_from_smem = available_smem // smem_per_batch
        max_batches_from_threads = 1024 // threads_per_batch
        batches_per_block = min(
            max_batches_from_smem, max_batches_from_threads, max_bpb, batch_size
        )
        batches_per_block = max(1, batches_per_block)

        # Round down to warp-aligned count, picking the option that maximises
        # total threads across co-resident blocks on one SM.
        batches_per_block = DynamicsLaunchCfg._warp_align_batches(
            batches_per_block,
            threads_per_batch,
            smem_per_block_fn=lambda b: smem_block_shared + b * smem_per_batch,
            sm_shared_mem_capacity=DynamicsLaunchCfg.DEFAULT_SM_SHARED_MEM_CAPACITY,
        )

        threads_per_block = batches_per_block * threads_per_batch
        blocks_per_grid = (batch_size + batches_per_block - 1) // batches_per_block
        shared_mem_size = smem_block_shared + batches_per_block * smem_per_batch

        return LaunchConfig(
            grid=blocks_per_grid,
            block=threads_per_block,
            shmem_size=shared_mem_size,
        )
