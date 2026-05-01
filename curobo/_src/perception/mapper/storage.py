# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse TSDF storage - main class and data structures.

This module provides:
- BlockSparseTSDFCfg: Configuration dataclass
- BlockSparseTSDFData: Dataclass holding all tensor references
- BlockSparseTSDF: Main class managing block-sparse TSDF storage

Memory Layout:
    Block size: ``block_size**3`` voxels per block (default 8**3 = 512)
    Per voxel: 2 × float16 (sdf_weight, weight)
    Per block: 4 × float16 RGBW + feature_dim × float16 features + 1 × float16 feature_weight

Memory Budget example (block_size=8, 100K blocks):
    - hash_table:         1.6 MB (200K × 8 bytes, packed key+value)
    - block_data:       200.0 MB (100K × 512 × 2 × 2 bytes)
    - block_rgb:          0.8 MB (100K × 4 × 2 bytes)
    - block_coords:       1.2 MB (100K × 3 × 4 bytes)
    - block_to_hash_slot: 0.4 MB (100K × 4 bytes)
    - free_list:          0.4 MB (100K × 4 bytes)
    - block_sums:         0.4 MB (100K × 4 bytes)
    - new_blocks:         0.4 MB (100K × 4 bytes)
    ─────────────────────────────────
    Total:              ~205 MB (vs 4+ GB dense = 20× savings)

All per-block weighted-sum accumulators (``block_rgb``, ``block_features``,
``block_feature_weight``) are stored as fp16. The integration kernels
pre-normalize RGB inputs by 255 and feature inputs to unit magnitude so
per-pixel atomic-add increments stay ``O(1)``. A post-frame rescale
kernel caps each per-block weight at :attr:`BlockSparseTSDFCfg.accumulator_w_max`
and scales the weighted-sum channels proportionally, preserving the
weighted mean while keeping magnitudes inside fp16's finite range.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.constants import (
    DEFAULT_HASH_LAYOUT,
    PY_HASH_EMPTY,
    PY_HASH_PRIME_X,
    PY_HASH_PRIME_Y,
    PY_HASH_PRIME_Z,
    PY_HASH_TOMBSTONE,
    PY_POSITIVE_MASK,
    PY_VALUE_MASK,
    _validate_feature_channels_per_thread,
    _validate_feature_grid_shape,
    _validate_block_size,
    validate_grid_shape_for_hash_layout,
)
from curobo._src.perception.mapper.kernel.builder.builder_block_sparse_kernel import (
    BlockSparseKernels,
    make_block_sparse_kernels,
)
from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
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
        grid_shape: Virtual grid dimensions (nz, ny, nx) for center-origin convention.
            Z is slowest-varying, X is fastest-varying (row-major).
            Origin is treated as the center of this bounded grid.
        enable_dynamic: Enable dynamic (depth) integration channel.
        enable_static: Enable static (primitive) integration channel.
        static_obstacle_color: RGB color for static obstacles (0-1 range).

    Note:
        block_size specializes the Warp kernels compiled for this TSDF
        (see :class:`~curobo._src.perception.mapper.kernel.builder.builder_block_sparse_kernel.BlockSparseKernels`).
        Supported values are 1 or powers of 2 in [2, 32]. Warp owns compiled-kernel reuse,
        so two TSDFs with different values can coexist in the same process.
    """

    max_blocks: int = 100_000
    hash_capacity: int = 200_000
    voxel_size: float = 0.002
    origin: torch.Tensor = None  # Set in __post_init__
    truncation_distance: float = 0.04
    device: str = "cuda:0"
    grid_shape: Tuple[int, int, int] = None
    enable_dynamic: bool = True
    enable_static: bool = False
    static_obstacle_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    #: Voxels per block edge. Supported values are 1 or powers of 2 in [2, 32].
    block_size: int = 8
    #: Per-block feature channel dimensionality. 0 disables the channel
    #: (dummy tensors are allocated for Warp compatibility).
    feature_dim: int = 0
    #: Compile-time feature-grid height used when this storage builds its
    #: own kernel bundle. Required when ``feature_dim > 0``; must be
    #: ``None`` when features are disabled.
    feature_grid_height: Optional[int] = None
    #: Compile-time feature-grid width used when this storage builds its
    #: own kernel bundle. Required when ``feature_dim > 0``; must be
    #: ``None`` when features are disabled.
    feature_grid_width: Optional[int] = None
    #: Number of adjacent feature channels accumulated per feature-kernel
    #: thread. Must match the voxel-project integrator launch grouping.
    feature_channels_per_thread: int = 4
    #: Compile-time cap for feature channels accumulated by one tiled
    #: feature-kernel CTA.
    max_feature_tile_channels: int = 4096
    #: Compile-time support-pixel capacity specialized into RGB and feature
    #: integration kernels. Must match the voxel-project scratch depth.
    max_support_pixels_per_block_camera: int = 32
    #: Upper bound on per-block accumulator weight (``block_rgb[:, 3]`` and
    #: ``block_feature_weight``) after each integration step. Caps the
    #: magnitude of the fp16 weighted-sum accumulators so they stay inside
    #: fp16's finite range and bounds ulp-loss on subsequent atomic adds.
    #: Setting this finite also gives the mapper EMA semantics: old
    #: observations decay at a rate set by
    #: ``W_max / mean_per_frame_weight``, which is desirable for dynamic
    #: scenes but means callers should pick ``W_max`` deliberately.
    accumulator_w_max: float = 1000.0

    def __post_init__(self):
        if self.origin is None:
            self.origin = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.float32)
        if not isinstance(self.origin, torch.Tensor):
            self.origin = torch.tensor(self.origin, dtype=torch.float32)
        # Ensure float32 dtype and move to device
        self.origin = self.origin.to(dtype=torch.float32, device=self.device)
        _validate_block_size(self.block_size)
        self.grid_shape = validate_grid_shape_for_hash_layout(
            self.grid_shape,
            self.block_size,
            field_name="BlockSparseTSDFCfg.grid_shape",
        )
        _validate_feature_channels_per_thread(self.feature_channels_per_thread)
        _validate_feature_grid_shape(
            self.feature_dim,
            self.feature_grid_height,
            self.feature_grid_width,
        )
        if self.max_feature_tile_channels <= 0:
            raise ValueError(
                "max_feature_tile_channels must be positive, got "
                f"{self.max_feature_tile_channels}."
            )
        if self.max_support_pixels_per_block_camera <= 0:
            raise ValueError(
                "max_support_pixels_per_block_camera must be positive, got "
                f"{self.max_support_pixels_per_block_camera}."
            )
        if self.max_blocks > DEFAULT_HASH_LAYOUT.max_pool_idx:
            raise ValueError(
                f"max_blocks={self.max_blocks:,} exceeds the "
                f"{DEFAULT_HASH_LAYOUT.name} pool limit of "
                f"{DEFAULT_HASH_LAYOUT.max_pool_idx:,}."
            )
        # Validate at least one channel is enabled
        if not self.enable_dynamic and not self.enable_static:
            raise ValueError("At least one of enable_dynamic or enable_static must be True")


# =============================================================================
# Data Container
# =============================================================================


@dataclass
class BlockDataView:
    """References (not copies) to per-block tensors.

    Index any field with ``block_idx_per_voxel`` (global ``pool_idx``) returned
    by :func:`extract_occupied_voxels`. Normalize RGB by dividing the first
    3 columns by the weight column::

        rgb_unweighted = view.rgb[:, :3] / view.rgb[:, 3:4]

    The tensors reference the underlying
    :class:`BlockSparseTSDFData` storage; callers must treat them as
    read-only (mutation affects storage).

    When the feature channel is disabled (``feature_dim == 0``),
    ``features`` / ``feature_weight`` are dummy ``(1, 1)`` / ``(1,)``
    tensors kept only for Warp compatibility.
    """

    rgb: torch.Tensor  # (max_blocks, 4) fp16 weighted sums [R*w, G*w, B*w, W]
    coords: torch.Tensor  # (max_blocks * 3,) int32 signed centered block keys
    num_allocated: int
    voxel_size: float
    block_size: int
    features: torch.Tensor  # (max_blocks, feature_dim) fp16 weighted sums, or (1, 1) dummy
    feature_weight: torch.Tensor  # (max_blocks,) fp16 weight sums, or (1,) dummy
    feature_dim: int  # 0 when the feature channel is disabled

    def features_normalized(self, eps: float = 1e-6) -> torch.Tensor:
        """Return active-block features normalized by per-block weight.

        Shape: ``(num_allocated, feature_dim)`` float32. Accumulators
        are stored fp16; the divide is done in fp32 to avoid compounding
        ulp loss on top of the post-frame rescale. Raises if the feature
        channel is disabled so callers fail loudly rather than silently
        operating on dummy tensors.
        """
        if self.feature_dim == 0:
            raise RuntimeError(
                "features_normalized() called on a BlockDataView with "
                "feature_dim == 0. Enable features via "
                "BlockSparseTSDFCfg.feature_dim (or MapperCfg.feature_dim)."
            )
        n = self.num_allocated
        weight = self.feature_weight[:n].float().clamp(min=eps).unsqueeze(-1)
        return self.features[:n].float() / weight


@dataclass
class OccupiedVoxels:
    """Result of :func:`extract_occupied_voxels`.

    Attributes:
        centers: ``(N, 3)`` float32 voxel world positions.
        block_idx_per_voxel: ``(N,)`` int32 global ``pool_idx`` — index into
            :class:`BlockDataView` fields.
        block_data: Reference view of per-block storage tensors.
    """

    centers: torch.Tensor
    block_idx_per_voxel: torch.Tensor
    block_data: BlockDataView

    def __len__(self) -> int:
        return self.centers.shape[0]

    def colors_uint8(self, eps: float = 1e-6) -> torch.Tensor:
        """Gather per-voxel RGB colors as ``(N, 3)`` uint8.

        Weighted sums (stored fp16, RGB normalized to ``[0, 1]`` at the
        integration site) are divided in fp32 and rescaled back to
        ``[0, 255]`` before the uint8 cast. Clamp protects degenerate
        blocks with zero weight from division by zero.
        """
        rgb = self.block_data.rgb[self.block_idx_per_voxel].float()
        normalized = rgb[:, :3] / rgb[:, 3:4].clamp(min=eps)
        return (normalized * 255.0).clamp(0.0, 255.0).to(torch.uint8)

    def features(self, eps: float = 1e-6) -> torch.Tensor:
        """Gather per-voxel normalized features as ``(N, feature_dim)`` float32.

        Weighted sums (stored fp16) are divided in fp32 to avoid
        compounding ulp loss. The per-block feature weight is clamped to
        avoid division by zero for blocks that never received a feature
        observation.
        """
        if self.block_data.feature_dim == 0:
            raise RuntimeError(
                "OccupiedVoxels.features() called with feature_dim == 0. "
                "Enable features via BlockSparseTSDFCfg.feature_dim."
            )
        gathered = self.block_data.features[self.block_idx_per_voxel].float()
        weight = self.block_data.feature_weight[self.block_idx_per_voxel].float()
        return gathered / weight.clamp(min=eps).unsqueeze(-1)


@dataclass
class MatchedVoxels:
    """Result of a feature-similarity query on a BlockSparseTSDF.

    Voxels are extracted from the top-K matched blocks. ``block_pool_idx``
    and ``block_scores`` are parallel ``(K,)`` tensors in descending score
    order — index ``i`` in one corresponds to index ``i`` in the other.

    Scores are cosine similarity in the same feature space as the query
    vector (range ``[-1, 1]``). Callers querying through a projection
    (e.g., RADIO -> teacher space) must interpret thresholds in the
    projected space.

    ``block_pool_idx`` is suitable for direct use with
    :meth:`Mapper.clear_blocks` to drop the matched blocks from the map.

    Attributes:
        voxels: Extracted voxels from the matched blocks. ``len(voxels)``
            may be 0 even when ``len(block_pool_idx) > 0`` if surface
            filtering rejected every voxel inside the matched blocks.
        block_pool_idx: ``(K,)`` int32 global ``pool_idx`` of matched
            blocks, sorted by ``block_scores`` descending.
        block_scores: ``(K,)`` float32 cosine scores, parallel to
            ``block_pool_idx`` (same descending order).
    """

    voxels: OccupiedVoxels
    block_pool_idx: torch.Tensor
    block_scores: torch.Tensor

    def __len__(self) -> int:
        return len(self.voxels)

    def scores_per_voxel(self, fill_value: float = float("nan")) -> torch.Tensor:
        """Gather per-voxel score from per-block scores.

        Convenience for visualization (e.g., color voxels by similarity).
        Granularity is fundamentally per-block — every voxel in a matched
        block shares its block's score. Prefer :attr:`block_scores` for
        ranking and filtering.

        Args:
            fill_value: Value written for voxels whose ``pool_idx`` is
                not in ``block_pool_idx``. Defaults to NaN so an unset
                entry is loud rather than silently masquerading as a
                low-similarity match. Voxels in :attr:`voxels` should
                always come from matched blocks; ``fill_value`` is a
                safety net.

        Returns:
            ``(N,)`` float32 tensor parallel to ``voxels.centers``.
        """
        num_alloc = self.voxels.block_data.num_allocated
        lookup = self.block_scores.new_full((num_alloc,), fill_value)
        lookup[self.block_pool_idx.long()] = self.block_scores
        return lookup[self.voxels.block_idx_per_voxel.long()]


@dataclass
class BlockSparseTSDFData:
    """Dataclass holding all block-sparse TSDF tensor references.

    This is the primary Python interface for data access. Use to_warp()
    to convert to a Warp struct for kernel launches.

    All tensors are pre-allocated at construction time for CUDA graph
    compatibility. Tensor shapes are static.

    Hash Table Entry Layout (64 bits):
        | X (13 bits) | Y (13 bits) | Z (13 bits) | pool_idx (25 bits) |
        - Coordinates: [-4096, 4095] per axis
        - Pool index: 0 to 33,554,430
        - Empty: -1, Tombstone: -2
    """

    # Hash table (packed key+value in single int64)
    hash_table: torch.Tensor  # (capacity,) int64, packed entries

    # Block pool - packed voxel data (3D for Warp INT32_MAX compatibility)
    block_data: torch.Tensor  # (max_blocks, block_size**3, 2) float16 or (1, 1, 2) dummy
    block_rgb: torch.Tensor  # (max_blocks, 4) float16 - per-block [R×w, G×w, B×w, W]

    # Per-block feature channel (weighted-sum accumulator + dedicated weight)
    block_features: torch.Tensor  # (max_blocks, feature_dim) float16 or (1, 1) dummy
    block_feature_weight: torch.Tensor  # (max_blocks,) float16 or (1,) dummy

    # Static SDF channel (separate tensor for primitives)
    static_block_data: torch.Tensor  # (max_blocks, block_size**3) float16 or (1, 1) dummy

    # Block metadata
    block_coords: torch.Tensor  # (max_blocks * 3,) int32 signed centered block keys
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
    grid_shape: Tuple[int, int, int] = None  # (nz, ny, nx) for center-origin

    # Feature flags
    has_dynamic: bool = True
    has_static: bool = False
    has_features: bool = False
    feature_dim: int = 0

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
        # Per-block weighted sums (fp16)
        s.block_rgb = wp.from_torch(self.block_rgb, dtype=wp.float16)

        # Per-block feature channel (fp16 weighted sums, post-frame cap bounds magnitude)
        s.block_features = wp.from_torch(self.block_features, dtype=wp.float16)
        s.block_feature_weight = wp.from_torch(self.block_feature_weight, dtype=wp.float16)

        # Block pool - static channel
        s.static_block_data = wp.from_torch(self.static_block_data, dtype=wp.float16)

        # Feature flags
        s.has_dynamic = self.has_dynamic
        s.has_static = self.has_static
        s.has_features = self.has_features
        s.feature_dim = self.feature_dim

        # Block metadata
        s.block_coords = wp.from_torch(self.block_coords, dtype=wp.int32)
        s.block_to_hash_slot = wp.from_torch(self.block_to_hash_slot, dtype=wp.int32)

        # Free list
        s.free_list = wp.from_torch(self.free_list, dtype=wp.int32)
        s.free_count = wp.from_torch(self.free_count, dtype=wp.int32)

        # Counters
        s.num_allocated = wp.from_torch(self.num_allocated, dtype=wp.int32)
        s.allocation_failures = wp.from_torch(self.allocation_failures, dtype=wp.int32)

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
        # Grid shape for bounded center-origin convention.
        s.grid_D = self.grid_shape[0]  # nz
        s.grid_H = self.grid_shape[1]  # ny
        s.grid_W = self.grid_shape[2]  # nx

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
        - Hash table: Maps signed centered block keys → pool index
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

    def __init__(
        self,
        config: BlockSparseTSDFCfg,
        kernels: Optional[BlockSparseKernels] = None,
    ):
        """Initialize block-sparse TSDF storage.

        Args:
            config: Configuration dataclass with grid parameters.
            kernels: Pre-resolved kernel bundle. If ``None``, builds a
                fresh bundle from ``config.block_size``.
        """
        init_warp()

        self.config = config
        self.device = config.device

        if kernels is None:
            kernels = make_block_sparse_kernels(config)
        assert kernels.block_size == config.block_size, (
            f"kernels.block_size={kernels.block_size} does not match "
            f"config.block_size={config.block_size}"
        )
        assert kernels.feature_dim == config.feature_dim, (
            f"kernels.feature_dim={kernels.feature_dim} does not match "
            f"config.feature_dim={config.feature_dim}"
        )
        expected_feature_grid_shape = (
            (config.feature_grid_height, config.feature_grid_width)
            if config.feature_grid_height is not None
            else None
        )
        assert kernels.feature_grid_shape == expected_feature_grid_shape, (
            f"kernels.feature_grid_shape={kernels.feature_grid_shape} does not match "
            f"config feature grid shape={expected_feature_grid_shape}"
        )
        assert kernels.grid_shape == config.grid_shape, (
            f"kernels.grid_shape={kernels.grid_shape} does not match "
            f"config.grid_shape={config.grid_shape}"
        )
        config_origin = tuple(
            float(v) for v in config.origin.detach().to(device="cpu").flatten().tolist()
        )
        assert all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-6)
            for a, b in zip(kernels.origin_xyz, config_origin)
        ), (
            f"kernels.origin_xyz={kernels.origin_xyz} does not match "
            f"config.origin={config_origin}"
        )
        assert math.isclose(
            kernels.voxel_size, config.voxel_size, rel_tol=0.0, abs_tol=1.0e-12
        ), (
            f"kernels.voxel_size={kernels.voxel_size} does not match "
            f"config.voxel_size={config.voxel_size}"
        )
        assert math.isclose(
            kernels.truncation_distance,
            config.truncation_distance,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ), (
            "kernels.truncation_distance="
            f"{kernels.truncation_distance} does not match "
            f"config.truncation_distance={config.truncation_distance}"
        )
        assert kernels.feature_channels_per_thread == config.feature_channels_per_thread, (
            "kernels.feature_channels_per_thread="
            f"{kernels.feature_channels_per_thread} does not match "
            "config.feature_channels_per_thread="
            f"{config.feature_channels_per_thread}"
        )
        assert kernels.max_feature_tile_channels == config.max_feature_tile_channels, (
            "kernels.max_feature_tile_channels="
            f"{kernels.max_feature_tile_channels} does not match "
            "config.max_feature_tile_channels="
            f"{config.max_feature_tile_channels}"
        )
        assert (
            kernels.max_support_pixels_per_block_camera
            == config.max_support_pixels_per_block_camera
        ), (
            "kernels.max_support_pixels_per_block_camera="
            f"{kernels.max_support_pixels_per_block_camera} does not match "
            "config.max_support_pixels_per_block_camera="
            f"{config.max_support_pixels_per_block_camera}"
        )
        assert kernels.hash_layout == DEFAULT_HASH_LAYOUT, (
            f"kernels.hash_layout={kernels.hash_layout} does not match "
            f"DEFAULT_HASH_LAYOUT={DEFAULT_HASH_LAYOUT}"
        )
        self.kernels = kernels

        block_voxels = config.block_size**3

        # Conditional allocation for dynamic channel
        if config.enable_dynamic:
            block_data = torch.zeros(
                (config.max_blocks, block_voxels, 2),
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
                (config.max_blocks, block_voxels),
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

        # Conditional allocation for per-block feature channel
        has_features = config.feature_dim > 0
        if has_features:
            block_features = torch.zeros(
                (config.max_blocks, config.feature_dim),
                dtype=torch.float16,
                device=self.device,
            )
            block_feature_weight = torch.zeros(
                config.max_blocks,
                dtype=torch.float16,
                device=self.device,
            )
        else:
            # Dummy tensors for Warp compatibility when disabled
            block_features = torch.zeros(
                (1, 1),
                dtype=torch.float16,
                device=self.device,
            )
            block_feature_weight = torch.zeros(
                1,
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
            # One color per block (not per voxel) - block_size**3 memory savings
            # Divide by channel 3 (weight_sum) at read time for averaging.
            # fp16 storage: integration pre-normalizes RGB to [0, 1] and a
            # post-frame rescale kernel caps the weight at
            # config.accumulator_w_max so accumulators stay in fp16 range.
            block_rgb=torch.zeros(
                (config.max_blocks, 4),
                dtype=torch.float16,
                device=self.device,
            ),
            # Per-block feature channel (conditional). feature_dim==0 gives
            # (1, 1) / (1,) dummies so the Warp struct can always be built.
            block_features=block_features,
            block_feature_weight=block_feature_weight,
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
            has_features=has_features,
            feature_dim=config.feature_dim,
        )

        # Cached Warp struct (for CUDA graph compatibility)
        self._warp_cache: Optional[BlockSparseTSDFWarp] = None

        # Tombstone counter (updated during rehash)
        self._tombstone_count: int = 0

    @property
    def data(self) -> BlockSparseTSDFData:
        """Get dataclass for Python manipulation."""
        return self._data

    @property
    def block_size(self) -> int:
        """Voxels per block edge. Single source of truth, sourced
        from the kernel specialization (which must match
        ``config.block_size``; the ctor asserts this)."""
        return self.kernels.block_size

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

    def get_stats(
        self,
        scan_pool: bool = True,
        scan_hash: bool = False,
    ) -> Dict[str, float]:
        """Get monitoring statistics.

        Call this OUTSIDE of CUDA graph execution for accurate readings.

        Args:
            scan_pool: When True (default) reduce ``block_to_hash_slot``
                to compute the ground-truth ``active_blocks`` and the
                ``holes`` invariant. Cost is O(num_allocated) on the GPU
                plus a single D2H sync. When False, ``active_blocks``
                falls back to the cheap ``num_allocated - free_count``
                approximation and ``holes`` is omitted; this is correct
                only if no pool_idx leak ever occurred.
            scan_hash: When True, reduce the entire hash table to count
                empty / tombstone / occupied slots. Cost is
                O(hash_capacity); enable only for periodic diagnostics.

        Returns:
            Dictionary with:
                - num_allocated: High-water mark of pool usage
                - free_count: Current number of free slots
                - active_blocks: Currently active blocks
                  (ground-truth scan when scan_pool=True; cheap
                  ``num_allocated - free_count`` approximation otherwise)
                - holes: ``num_allocated - active_blocks - free_count``
                  (only when scan_pool=True). Should always be 0;
                  non-zero indicates a pool_idx leak.
                - recycled_last: Blocks recycled in the most recent
                  decay/recycle pass.
                - pool_usage_pct: Pool utilization percentage
                - fragmentation_pct: Fragmentation percentage
                - allocation_failures: Lifetime allocation failures
                - tombstone_count: Tombstones tracked by compact()
                  (legacy field; not updated by recycling)
                - hash_load_pct: Active fraction of hash table
                - hash_occ / hash_tomb / hash_empty: Hash slot counts
                  (only when scan_hash=True)

        Note:
            Prior to the leak fix, ``active_blocks`` was always defined
            as ``num_allocated - free_count``. That value over-counted
            by the number of orphaned pool indices. The default
            ``scan_pool=True`` returns the correct value, which may
            differ from historical readings on buggy code paths.
        """
        num_alloc = int(self._data.num_allocated.item())
        free_cnt = int(self._data.free_count.item())
        failures = int(self._data.allocation_failures.item())
        recycled_last = int(self._data.recycle_count.item())

        stats: Dict[str, float] = {
            "num_allocated": num_alloc,
            "free_count": free_cnt,
            "allocation_failures": failures,
            "recycled_last": recycled_last,
            "tombstone_count": self._tombstone_count,
            "pool_usage_pct": num_alloc / self.config.max_blocks * 100,
            "fragmentation_pct": free_cnt / max(num_alloc, 1) * 100,
        }

        if scan_pool and num_alloc > 0:
            active = int((self._data.block_to_hash_slot[:num_alloc] >= 0).sum().item())
            stats["active_blocks"] = active
            stats["holes"] = num_alloc - active - free_cnt
        else:
            stats["active_blocks"] = num_alloc - free_cnt

        if scan_hash:
            hash_table = self._data.hash_table
            hash_empty = int((hash_table == PY_HASH_EMPTY).sum().item())
            hash_tomb = int((hash_table == PY_HASH_TOMBSTONE).sum().item())
            hash_occ = self.config.hash_capacity - hash_empty - hash_tomb
            stats["hash_empty"] = hash_empty
            stats["hash_tomb"] = hash_tomb
            stats["hash_occ"] = hash_occ
            stats["hash_load_pct"] = hash_occ / self.config.hash_capacity * 100
        else:
            stats["hash_load_pct"] = stats["active_blocks"] / self.config.hash_capacity * 100

        return stats

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
            # Extract pool_idx from the active layout's value field.
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
        total += self._data.block_rgb.numel() * 2  # float16
        total += self._data.block_features.numel() * 2  # float16
        total += self._data.block_feature_weight.numel() * 2  # float16
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
