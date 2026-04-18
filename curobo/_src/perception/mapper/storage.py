# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse TSDF storage - main class and data structures.

This module provides:
- BlockSparseTSDFCfg: Configuration dataclass
- BlockSparseTSDFData: Dataclass holding all tensor references
- BlockSparseTSDF: Main class managing block-sparse TSDF storage

Memory Layout:
    Block size: 8³ = 512 voxels per block
    Per voxel: 2 × float16 (sdf_weight, weight) + 3 × uint8 (RGB)

Memory Budget (100K blocks):
    - hash_table:         1.6 MB (200K × 8 bytes, packed key+value)
    - block_data:       200.0 MB (100K × 512 × 2 × 2 bytes)
    - block_rgb:        150.0 MB (100K × 512 × 3 bytes)
    - block_coords:       1.2 MB (100K × 3 × 4 bytes)
    - block_to_hash_slot: 0.4 MB (100K × 4 bytes)
    - free_list:          0.4 MB (100K × 4 bytes)
    - block_sums:         0.4 MB (100K × 4 bytes)
    - new_blocks:         0.4 MB (100K × 4 bytes)
    ─────────────────────────────────
    Total:              ~354 MB (vs 4+ GB dense = 11× savings)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import (
    BLOCK_SIZE,
    PY_HASH_EMPTY,
    PY_HASH_PRIME_X,
    PY_HASH_PRIME_Y,
    PY_HASH_PRIME_Z,
    PY_HASH_TOMBSTONE,
    PY_POSITIVE_MASK,
    PY_VALUE_MASK,
    BlockSparseTSDFWarp,
)
from curobo._src.util.warp import init_warp

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BlockSparseTSDFCfg:
    """Configuration for BlockSparseTSDF.

    Attributes:
        max_blocks: Maximum number of allocatable blocks. Memory scales with this.
            100K blocks = ~355 MB, covers most indoor scenes.
        hash_capacity: Hash table size. Should be 2× max_blocks for load factor 0.5.
        voxel_size: Size of each voxel in meters.
        origin: World coordinate of grid origin (3,).
        truncation_distance: TSDF truncation distance in meters.
        device: CUDA device (default: cuda:0).
        grid_shape: Optional virtual grid dimensions (nz, ny, nx) for center-origin convention.
            Z is slowest-varying, X is fastest-varying (row-major).
            If provided, origin is treated as center of this grid for mesh extraction.
            If None, uses corner-origin convention.
        enable_dynamic: Enable dynamic (depth) integration channel.
        enable_static: Enable static (primitive) integration channel.
        static_obstacle_color: RGB color for static obstacles (0-1 range).

    Note:
        block_size is a module constant (BLOCK_SIZE=8 from warp_types.py).
        It cannot be changed at runtime - kernels use hardcoded optimizations.
    """

    max_blocks: int = 100_000
    hash_capacity: int = 200_000
    voxel_size: float = 0.002
    origin: torch.Tensor = None  # Set in __post_init__
    truncation_distance: float = 0.04
    device: str = "cuda:0"
    grid_shape: Optional[Tuple[int, int, int]] = None
    enable_dynamic: bool = True
    enable_static: bool = False
    static_obstacle_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    @property
    def block_size(self) -> int:
        """Block size (voxels per edge). Fixed at BLOCK_SIZE=8."""
        return BLOCK_SIZE

    def __post_init__(self):
        if self.origin is None:
            self.origin = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.float32)
        if not isinstance(self.origin, torch.Tensor):
            self.origin = torch.tensor(self.origin, dtype=torch.float32)
        # Ensure float32 dtype and move to device
        self.origin = self.origin.to(dtype=torch.float32, device=self.device)
        # Validate at least one channel is enabled
        if not self.enable_dynamic and not self.enable_static:
            raise ValueError("At least one of enable_dynamic or enable_static must be True")


# =============================================================================
# Data Container
# =============================================================================


@dataclass
class BlockSparseTSDFData:
    """Dataclass holding all block-sparse TSDF tensor references.

    This is the primary Python interface for data access. Use to_warp()
    to convert to a Warp struct for kernel launches.

    All tensors are pre-allocated at construction time for CUDA graph
    compatibility. Tensor shapes are static.

    Hash Table Entry Layout (64 bits):
        | X (14 bits) | Y (14 bits) | Z (14 bits) | pool_idx (22 bits) |
        - Coordinates: ±2,048 per axis
        - Pool index: 0 to 4,194,303 (4M blocks max)
        - Empty: -1, Tombstone: -2
    """

    # Hash table (packed key+value in single int64)
    hash_table: torch.Tensor  # (capacity,) int64, packed entries

    # Block pool - packed voxel data (3D for Warp INT32_MAX compatibility)
    block_data: torch.Tensor  # (max_blocks, 512, 2) float16 or (1, 1, 2) dummy
    block_rgb: torch.Tensor  # (max_blocks, 4) float32 - per-block [R×w, G×w, B×w, W]

    # Static SDF channel (separate tensor for primitives)
    static_block_data: torch.Tensor  # (max_blocks, 512) float16 or (1, 1) dummy

    # Block metadata
    block_coords: torch.Tensor  # (max_blocks * 3,) int32, block world coords
    block_to_hash_slot: torch.Tensor  # (max_blocks,) int32, reverse mapping

    # Free list (stack)
    free_list: torch.Tensor  # (max_blocks,) int32
    free_count: torch.Tensor  # (1,) int32

    # Counters
    num_allocated: torch.Tensor  # (1,) int32
    allocation_failures: torch.Tensor  # (1,) int32

    # Decay bookkeeping
    block_sums: torch.Tensor  # (max_blocks,) float32 - dynamic channel weights
    static_block_sums: torch.Tensor  # (max_blocks,) int32 - static channel voxel count
    frustum_flags: torch.Tensor  # (max_blocks,) int32 - 1 = block in frustum, 0 = outside
    decay_factor: torch.Tensor  # (max_blocks,) float16 - per-block decay multiplier

    # Per-frame new block tracking
    new_blocks: torch.Tensor  # (max_blocks,) int32
    new_block_count: torch.Tensor  # (1,) int32

    # Recycle counter (pre-allocated for CUDA graph safety)
    recycle_count: torch.Tensor  # (1,) int32

    # Grid parameters (non-tensor)
    origin: torch.Tensor  # (3,) float32
    voxel_size: float
    block_size: int
    hash_capacity: int
    max_blocks: int
    truncation_distance: float
    grid_shape: Optional[Tuple[int, int, int]] = None  # (nz, ny, nx) for center-origin

    # Feature flags
    has_dynamic: bool = True
    has_static: bool = False

    def to_warp(self) -> BlockSparseTSDFWarp:
        """Convert to Warp struct for kernel launches.

        This method creates a new Warp struct with wp.from_torch() wrappers.
        The result should be cached for CUDA graph compatibility.

        Returns:
            BlockSparseTSDFWarp struct for use in Warp kernels.
        """
        s = BlockSparseTSDFWarp()

        # Hash table (packed key+value)
        s.hash_table = wp.from_torch(self.hash_table, dtype=wp.int64)

        # Block pool - dynamic channel
        s.block_data = wp.from_torch(self.block_data, dtype=wp.float16)
        s.block_rgb = wp.from_torch(self.block_rgb, dtype=wp.float32)  # Per-block weighted sums

        # Block pool - static channel
        s.static_block_data = wp.from_torch(self.static_block_data, dtype=wp.float16)

        # Feature flags
        s.has_dynamic = self.has_dynamic
        s.has_static = self.has_static

        # Block metadata
        s.block_coords = wp.from_torch(self.block_coords, dtype=wp.int32)
        s.block_to_hash_slot = wp.from_torch(self.block_to_hash_slot, dtype=wp.int32)

        # Free list
        s.free_list = wp.from_torch(self.free_list, dtype=wp.int32)
        s.free_count = wp.from_torch(self.free_count, dtype=wp.int32)

        # Counters
        s.num_allocated = wp.from_torch(self.num_allocated, dtype=wp.int32)
        s.allocation_failures = wp.from_torch(
            self.allocation_failures, dtype=wp.int32
        )

        # Block sums
        s.block_sums = wp.from_torch(self.block_sums, dtype=wp.float32)
        s.static_block_sums = wp.from_torch(self.static_block_sums, dtype=wp.int32)

        # Per-frame tracking
        s.new_blocks = wp.from_torch(self.new_blocks, dtype=wp.int32)
        s.new_block_count = wp.from_torch(self.new_block_count, dtype=wp.int32)

        # Recycle counter
        s.recycle_count = wp.from_torch(self.recycle_count, dtype=wp.int32)

        # Scalars
        s.origin = wp.vec3(
            self.origin[0].item(),
            self.origin[1].item(),
            self.origin[2].item(),
        )
        s.voxel_size = self.voxel_size
        s.hash_capacity = self.hash_capacity
        s.max_blocks = self.max_blocks
        s.truncation_distance = self.truncation_distance
        s.block_size = self.block_size
        # Grid shape for center-origin convention (0 = corner-origin)
        if self.grid_shape is not None:
            s.grid_D = self.grid_shape[0]  # nz
            s.grid_H = self.grid_shape[1]  # ny
            s.grid_W = self.grid_shape[2]  # nx
        else:
            s.grid_D = 0
            s.grid_H = 0
            s.grid_W = 0

        return s


# =============================================================================
# Main Class
# =============================================================================


class BlockSparseTSDF:
    """Block-sparse TSDF storage for memory-efficient volumetric mapping.

    This class manages a block-sparse TSDF grid where only blocks containing
    observed data are allocated. Provides 20× memory reduction compared to
    dense grids while maintaining CUDA graph compatibility.

    Architecture:
        - Hash table: Maps block coordinates → pool index
        - Block pool: Pre-allocated contiguous memory for block data
        - Free list: Stack of available pool indices for recycling

    Example:
        config = BlockSparseTSDFCfg(
            max_blocks=100_000,
            voxel_size=0.002,
            origin=torch.tensor([-1.0, -1.0, 0.0]),
        )
        tsdf = BlockSparseTSDF(config)

        # Get Warp struct for kernel launches
        warp_data = tsdf.get_warp_data()

        # Monitor usage
        stats = tsdf.get_stats()
        print(f"Pool usage: {stats['pool_usage_pct']:.1f}%")
    """

    def __init__(self, config: BlockSparseTSDFCfg):
        """Initialize block-sparse TSDF storage.

        Args:
            config: Configuration dataclass with grid parameters.
        """
        init_warp()

        self.config = config
        self.device = config.device

        # Conditional allocation for dynamic channel
        if config.enable_dynamic:
            block_data = torch.zeros(
                (config.max_blocks, 512, 2),
                dtype=torch.float16,
                device=self.device,
            )
        else:
            # Dummy tensor for Warp compatibility
            block_data = torch.zeros(
                (1, 1, 2),
                dtype=torch.float16,
                device=self.device,
            )

        # Conditional allocation for static channel
        if config.enable_static:
            static_block_data = torch.full(
                (config.max_blocks, 512),
                float("inf"),  # Initialize to +inf (no obstacle)
                dtype=torch.float16,
                device=self.device,
            )
        else:
            # Dummy tensor for Warp compatibility
            static_block_data = torch.full(
                (1, 1),
                float("inf"),
                dtype=torch.float16,
                device=self.device,
            )

        # Pre-allocate all tensors
        self._data = BlockSparseTSDFData(
            # Hash table (packed key+value, -1 = empty)
            hash_table=torch.full(
                (config.hash_capacity,),
                -1,  # ENTRY_EMPTY
                dtype=torch.int64,
                device=self.device,
            ),
            # Block pool - dynamic channel (conditional)
            block_data=block_data,
            # Per-block RGBW: [R×w, G×w, B×w, weight_sum]
            # One color per block (not per voxel) - 512× memory savings
            # Divide by channel 3 (weight_sum) at read time for averaging
            block_rgb=torch.zeros(
                (config.max_blocks, 4),
                dtype=torch.float32,
                device=self.device,
            ),
            # Block pool - static channel (conditional)
            static_block_data=static_block_data,
            # Block metadata
            block_coords=torch.zeros(
                config.max_blocks * 3,
                dtype=torch.int32,
                device=self.device,
            ),
            block_to_hash_slot=torch.full(
                (config.max_blocks,),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            # Free list - starts empty (no recycled blocks yet)
            # Blocks are allocated from num_allocated, recycled blocks go to free_list
            free_list=torch.zeros(
                config.max_blocks,
                dtype=torch.int32,
                device=self.device,
            ),
            free_count=torch.zeros(1, dtype=torch.int32, device=self.device),
            # Counters
            num_allocated=torch.zeros(1, dtype=torch.int32, device=self.device),
            allocation_failures=torch.zeros(1, dtype=torch.int32, device=self.device),
            # Decay bookkeeping
            block_sums=torch.zeros(
                config.max_blocks,
                dtype=torch.float32,
                device=self.device,
            ),
            static_block_sums=torch.zeros(
                config.max_blocks,
                dtype=torch.int32,
                device=self.device,
            ),
            frustum_flags=torch.zeros(
                config.max_blocks,
                dtype=torch.int32,
                device=self.device,
            ),
            decay_factor=torch.ones(
                config.max_blocks,
                dtype=torch.float16,
                device=self.device,
            ),
            # Per-frame tracking
            new_blocks=torch.zeros(
                config.max_blocks,
                dtype=torch.int32,
                device=self.device,
            ),
            new_block_count=torch.zeros(1, dtype=torch.int32, device=self.device),
            # Recycle counter (pre-allocated for CUDA graph safety)
            recycle_count=torch.zeros(1, dtype=torch.int32, device=self.device),
            # Grid parameters
            origin=config.origin.clone(),
            voxel_size=config.voxel_size,
            block_size=config.block_size,
            hash_capacity=config.hash_capacity,
            max_blocks=config.max_blocks,
            truncation_distance=config.truncation_distance,
            grid_shape=config.grid_shape,
            # Feature flags
            has_dynamic=config.enable_dynamic,
            has_static=config.enable_static,
        )

        # Cached Warp struct (for CUDA graph compatibility)
        self._warp_cache: Optional[BlockSparseTSDFWarp] = None

        # Tombstone counter (updated during rehash)
        self._tombstone_count: int = 0

    @property
    def data(self) -> BlockSparseTSDFData:
        """Get dataclass for Python manipulation."""
        return self._data

    def get_warp_data(self) -> BlockSparseTSDFWarp:
        """Get Warp struct for kernel launches.

        The struct is cached and reused across calls for efficiency.
        This is critical for CUDA graph compatibility - the same struct
        (with same pointers) must be used for graph capture and replay.

        Returns:
            BlockSparseTSDFWarp struct for use in Warp kernels.
        """
        if self._warp_cache is None:
            self._warp_cache = self._data.to_warp()
        return self._warp_cache

    def invalidate_cache(self):
        """Invalidate cached Warp struct.

        Call this if tensors are reallocated (rare). Note that tensor
        reallocation breaks CUDA graphs anyway.
        """
        self._warp_cache = None

    def reset(self):
        """Reset all data to initial state.

        This clears all blocks and resets the hash table. Call this
        when starting a new mapping session.

        Note: This modifies tensor contents but not tensor allocations,
        so it's safe to call within a CUDA graph context (though the
        graph would need to be recaptured).
        """
        # Reset hash table
        self._data.hash_table.fill_(PY_HASH_EMPTY)

        # Reset block metadata
        self._data.block_to_hash_slot.fill_(PY_HASH_EMPTY)

        # Reset free list to empty (no recycled blocks)
        self._data.free_list.zero_()
        self._data.free_count.zero_()

        # Reset counters
        self._data.num_allocated.zero_()
        self._data.allocation_failures.zero_()

        # Reset per-frame tracking
        self._data.new_block_count.zero_()

        # Reset block sums
        self._data.block_sums.zero_()
        self._data.static_block_sums.zero_()

        # Reset recycle counter
        self._data.recycle_count.zero_()

        # Reset tombstone count
        self._tombstone_count = 0

        # Reset static channel to +inf (no obstacles)
        if self._data.has_static:
            self._data.static_block_data.fill_(float("inf"))

        # Note: block_data and block_rgb are cleared lazily when blocks
        # are allocated (via clear_new_blocks_kernel)

    def get_stats(self) -> Dict[str, float]:
        """Get monitoring statistics.

        Call this OUTSIDE of CUDA graph execution for accurate readings.

        Returns:
            Dictionary with:
                - num_allocated: High-water mark of pool usage
                - free_count: Current number of free slots
                - active_blocks: Currently active blocks
                - pool_usage_pct: Pool utilization percentage
                - fragmentation_pct: Fragmentation percentage
                - allocation_failures: Total allocation failures
                - tombstone_count: Number of tombstones in hash table
                - hash_load_pct: Hash table load percentage
        """
        num_alloc = int(self._data.num_allocated.item())
        free_cnt = int(self._data.free_count.item())
        failures = int(self._data.allocation_failures.item())

        active = num_alloc - free_cnt

        return {
            "num_allocated": num_alloc,
            "free_count": free_cnt,
            "active_blocks": active,
            "pool_usage_pct": num_alloc / self.config.max_blocks * 100,
            "fragmentation_pct": free_cnt / max(num_alloc, 1) * 100,
            "allocation_failures": failures,
            "tombstone_count": self._tombstone_count,
            "hash_load_pct": active / self.config.hash_capacity * 100,
        }

    def reset_failure_counter(self):
        """Reset allocation failure counter.

        Call OUTSIDE of CUDA graph execution.
        """
        self._data.allocation_failures.zero_()

    def compact_hash_table(self):
        """Rebuild hash table to remove tombstones.

        Call OUTSIDE of CUDA graph when tombstone ratio exceeds threshold.
        This operation is expensive and should be done infrequently
        (e.g., every few hundred frames if needed).

        Note: This breaks CUDA graph compatibility - the graph must be
        recaptured after calling this method.
        """
        # Count active entries (not empty, not tombstone)
        table = self._data.hash_table
        active_mask = (table != PY_HASH_EMPTY) & (table != PY_HASH_TOMBSTONE)

        # Extract active entries
        active_entries = table[active_mask]

        if len(active_entries) == 0:
            # Nothing to compact
            self._data.hash_table.fill_(PY_HASH_EMPTY)
            self._tombstone_count = 0
            self.invalidate_cache()
            return

        # Clear hash table
        self._data.hash_table.fill_(PY_HASH_EMPTY)

        # Re-insert all active entries (CPU-side for simplicity)
        active_entries_cpu = active_entries.cpu().numpy()

        for entry in active_entries_cpu:
            # Extract pool_idx from lower 22 bits
            pool_idx = int(entry & PY_VALUE_MASK)

            # Hash using block_coords for this pool_idx
            bx = self._data.block_coords[pool_idx * 3].item()
            by = self._data.block_coords[pool_idx * 3 + 1].item()
            bz = self._data.block_coords[pool_idx * 3 + 2].item()

            # Compute hash matching GPU spatial_hash exactly
            h = (bx * PY_HASH_PRIME_X) ^ (by * PY_HASH_PRIME_Y) ^ (bz * PY_HASH_PRIME_Z)
            h = h & PY_POSITIVE_MASK  # Clear sign bit (match GPU)
            slot = h % self.config.hash_capacity

            for _ in range(64):  # Linear probing
                if self._data.hash_table[slot].item() == PY_HASH_EMPTY:
                    self._data.hash_table[slot] = int(entry)

                    # Update reverse mapping
                    self._data.block_to_hash_slot[pool_idx] = slot
                    break
                slot = (slot + 1) % self.config.hash_capacity

        self._tombstone_count = 0
        self.invalidate_cache()

    def memory_usage_bytes(self) -> int:
        """Calculate total GPU memory usage in bytes."""
        total = 0
        total += self._data.hash_table.numel() * 8  # int64 (packed key+value)
        total += self._data.block_data.numel() * 2  # float16
        total += self._data.block_rgb.numel() * 4  # float32
        total += self._data.static_block_data.numel() * 2  # float16
        total += self._data.block_coords.numel() * 4  # int32
        total += self._data.block_to_hash_slot.numel() * 4  # int32
        total += self._data.free_list.numel() * 4  # int32
        total += self._data.free_count.numel() * 4  # int32
        total += self._data.num_allocated.numel() * 4  # int32
        total += self._data.allocation_failures.numel() * 4  # int32
        total += self._data.block_sums.numel() * 4  # float32
        total += self._data.new_blocks.numel() * 4  # int32
        total += self._data.new_block_count.numel() * 4  # int32
        total += self._data.recycle_count.numel() * 4  # int32
        total += self._data.origin.numel() * 4  # float32
        return total

    def memory_usage_mb(self) -> float:
        """Calculate total GPU memory usage in megabytes."""
        return self.memory_usage_bytes() / (1024 * 1024)

    def prepare_frame(self):
        """Prepare for a new frame of integration.

        Call this at the start of each frame to reset per-frame counters.
        """
        self._data.new_block_count.zero_()

