# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for dynamics (RNEA) CUDA kernel compilation and launching."""

# Standard Library
from pathlib import Path
from typing import List

# Third Party
from cuda.core import LaunchConfig

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend.kernel_config import CudaCoreKernelCfg


class DynamicsKernelCfg(CudaCoreKernelCfg):
    """Configuration for RNEA dynamics kernel compilation."""

    def __init__(self):
        super().__init__("dynamics")

    def get_kernel_files(self, kernel_type: str) -> List[str]:
        """Get kernel source files for a given kernel type.

        Args:
            kernel_type: Type of kernel ("forward", "backward").

        Returns:
            List of kernel filenames.
        """
        kernel_files = {
            "forward": ["rnea_forward_kernel.cuh"],
            "backward": ["rnea_backward_kernel.cuh"],
        }
        return kernel_files.get(kernel_type, [])

    def get_include_dirs(self) -> List[Path]:
        """Get include directories for kernel compilation.

        Includes base dirs (common/, third_party/) plus the dynamics and
        kinematics directories (for kinematics_constants.h).
        """
        base_dirs = self.get_base_include_dirs()
        extra_dirs = [
            self.kernel_dir,                          # kernels/dynamics/
            self.kernel_dir.parent / "kinematics",    # kernels/kinematics/
        ]
        return base_dirs + extra_dirs


class DynamicsLaunchCfg:
    """Helper class for calculating launch configurations for RNEA kernels.

    Block sizes are rounded down to warp boundaries (multiples of 32 threads) to
    avoid partial warps where some threads are idle.  A smaller warp-aligned block
    often uses less shared memory per block, allowing the SM to schedule multiple
    blocks concurrently and thus *increase* overall occupancy despite fewer threads
    per block.

    The helper :meth:`_warp_align_batches` picks the warp-aligned batch count that
    maximises the number of resident blocks on an SM (given the shared-memory budget).
    """

    DEFAULT_MAX_BATCHES_PER_BLOCK = 256
    DEFAULT_MAX_BW_BATCHES_PER_BLOCK = 256

    DEFAULT_MAX_SHARED_MEM = 48 * 1024  # 48 KB
    # Total configurable L1/shared-memory per SM (Ampere=128 KB, Hopper=228 KB).
    # Used to check how many blocks can co-reside on one SM.
    DEFAULT_SM_SHARED_MEM_CAPACITY = 100 * 1024  # conservative for Ampere

    WARP_SIZE = 32

    @staticmethod
    def _warp_align_batches(
        batches_per_block: int,
        threads_per_batch: int,
        smem_per_block_fn,
        sm_shared_mem_capacity: int,
    ) -> int:
        """Round *batches_per_block* down to a warp-aligned thread count that
        maximises SM occupancy (resident blocks × threads).

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
    ) -> LaunchConfig:
        """Calculate launch configuration for RNEA forward kernel.

        Shared memory per batch = num_links × 12 × 4 bytes (v + a).

        Args:
            batch_size: Number of batch elements.
            num_links: Number of links in the robot.
            threads_per_batch: Threads per batch element (1 for serial, >1 for tree-parallel).
            max_batches_per_block: Maximum batches per block (default 256).
            max_shared_mem: Maximum shared memory per block (default 48 KB).

        Returns:
            LaunchConfig for the kernel.
        """
        max_bpb = max_batches_per_block or DynamicsLaunchCfg.DEFAULT_MAX_BATCHES_PER_BLOCK
        max_shared_mem = max_shared_mem or DynamicsLaunchCfg.DEFAULT_MAX_SHARED_MEM

        # 12 = 2 spatial vectors × 6 floats × 4 bytes
        smem_per_batch = num_links * 12 * 4

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
    ) -> LaunchConfig:
        """Calculate launch configuration for RNEA backward (VJP) kernel.

        Shared memory layout:
          - Block-shared: num_links × 12 × 4 bytes (mc + inertia, one copy per block)
          - Per-batch: num_links × 30 × 4 bytes (v + a + f_bar + a_bar + v_bar)

        Args:
            batch_size: Number of batch elements.
            num_links: Number of links in the robot.
            threads_per_batch: Threads per batch element (1 for serial, >1 for tree-parallel).
            max_batches_per_block: Maximum batches per block (default 256).
            max_shared_mem: Maximum shared memory per block (default 48 KB).

        Returns:
            LaunchConfig for the backward kernel.
        """
        max_bpb = max_batches_per_block or DynamicsLaunchCfg.DEFAULT_MAX_BATCHES_PER_BLOCK
        max_shared_mem = max_shared_mem or DynamicsLaunchCfg.DEFAULT_MAX_SHARED_MEM

        # Block-shared: mc(4) + inertia(8) = 12 floats per link
        smem_block_shared = num_links * 12 * 4

        # Per-batch: 30 = 5 spatial vectors (6 each)
        smem_per_batch = num_links * 30 * 4

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
