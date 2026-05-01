# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""MapperCfg - Unified configuration for volumetric mapping.

This module provides the MapperCfg dataclass for configuring the Mapper
class with intuitive physical parameters.

Example:
    config = MapperCfg(
        extent_meters_xyz=(2.0, 2.0, 2.0),
        voxel_size=0.005,
    )
    mapper = Mapper(config)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from curobo._src.perception.mapper.block_allocation import calculate_tsdf_max_blocks
from curobo._src.perception.mapper.constants import (
    _validate_feature_channels_per_thread,
    _validate_feature_grid_shape,
    _validate_feature_integration_kernel,
)
from curobo.logging import log_and_raise


@dataclass
class MapperCfg:
    """Configuration for volumetric mapping.

    Coordinate Convention:
        - extent_meters_xyz: (x, y, z) physical extent of voxel bounds (not centers)
        - grid_shape: (nz, ny, nx) voxel counts - Z is slowest, X is fastest
        - Memory layout: row-major with X contiguous (optimal for image projection)
        - grid_center: world coordinate at center of grid (default: origin)

    Note: Actual extent may be slightly larger than requested due to ceil() rounding.
    Use get_actual_extent() to get the true physical dimensions.

    Index-to-World Transform (voxel center):
        world_x = grid_center[0] + (ix - (nx-1)/2) * voxel_size
        world_y = grid_center[1] + (iy - (ny-1)/2) * voxel_size
        world_z = grid_center[2] + (iz - (nz-1)/2) * voxel_size

    Attributes:
        extent_meters_xyz: Physical extent (x, y, z) of voxel bounds in meters.
        voxel_size: Voxel edge length in meters.
        grid_center: World coordinate at grid center (default: origin).
        truncation_distance: TSDF truncation distance in meters.
        minimum_tsdf_weight: Minimum weight to consider a voxel as observed.
        depth_minimum_distance: Minimum valid depth in meters.
        depth_maximum_distance: Maximum valid depth in meters.
        decay_factor: Weight decay applied to ALL voxels each frame (1.0 = no decay).
        frustum_decay_factor: Additional decay for in-view voxels.
        block_size: Voxels per block edge. Supported values are 1 or
            powers of 2 in [2, 32]
            (default 8). Specializes the Warp kernel builder; two
            Mappers with different block sizes can coexist.
        hash_load_factor: Target hash table load factor.
        device: CUDA device string.
    """

    # === Grid (REQUIRED) ===
    extent_meters_xyz: Tuple[float, float, float]
    voxel_size: float = 0.005
    esdf_voxel_size: float = 0.05
    extent_esdf_meters_xyz: Tuple[float, float, float] = None
    grid_center: Optional[torch.Tensor] = None

    # === TSDF ===
    truncation_distance: float = 0.04
    minimum_tsdf_weight: float = 0.1

    # === Depth Sensor ===
    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 10.0

    # === Decay ===
    decay_factor: float = 1.0
    frustum_decay_factor: float = 1.0

    # === Block Storage ===
    #: Voxels per block edge. Supported values are 1 or powers of 2 in [2, 32]. See
    #: :class:`~curobo._src.perception.mapper.kernel.builder.builder_block_sparse_kernel.BlockSparseKernels`.
    block_size: int = 4
    # Target hash table active-entry fraction. 0.5 keeps probe chains short
    # enough that the average lookup cost stays low even after tombstone
    # accumulation from aggressive recycling. Lookup now probes the full
    # capacity rather than a fixed 64-slot cap, so raising this further is
    # safe from a correctness standpoint but costs probe time per lookup.
    hash_load_factor: float = 0.5
    roughness: float = 3.0

    # === Integration ===
    seeding_method: str = "gather"
    """ESDF surface seeding strategy: ``"scatter"`` or ``"gather"``.

    - ``"scatter"``: iterates every allocated TSDF voxel and maps each
      surface voxel to the single ESDF cell that contains its center.
      The resulting seed band is exactly one ESDF voxel thick at the
      surface, faithfully matching the TSDF. Launch dimension depends on
      the number of allocated TSDF blocks, so it is **not** CUDA-graph
      safe.
    - ``"gather"``: iterates every ESDF voxel and probes 7 TSDF
      positions (cell center + 6 face centers at ``±esdf_vs/2``).
      Because the face-center samples lie on the ESDF cell boundary,
      they can detect surface voxels that belong to neighboring ESDF
      cells, producing a **dilated** seed band (~1.5 voxels thick).
      This thicker band gives PBA more seed sites near the surface,
      which typically yields slightly higher collision recall at the
      cost of not matching the TSDF surface exactly. Fixed launch
      dimension (``D×H×W``), **CUDA-graph safe**.
    """
    edt_solver: str = "pba"  # "jfa" (Jump Flooding) or "pba" (Parallel Banding, exact)

    # === Static Obstacles ===
    enable_static: bool = False
    static_obstacle_color: Tuple[int, int, int] = (20, 20, 20)

    # === Cameras ===
    num_cameras: int = 1
    #: Camera image height in pixels. Required; used to pre-allocate the
    #: voxel-project scratch buffer at Mapper construction (buffer size
    #: ``num_cameras * image_height * image_width * num_samples``) so no
    #: per-frame reallocation or D2H syncs occur.
    image_height: Optional[int] = None
    #: Camera image width in pixels. Required; see :attr:`image_height`.
    image_width: Optional[int] = None

    # === Per-block Features ===
    #: Per-block feature channel dimensionality. 0 disables features.
    feature_dim: int = 0
    #: Compile-time feature-grid height. Required when ``feature_dim > 0``;
    #: must be ``None`` when features are disabled.
    feature_grid_height: Optional[int] = None
    #: Compile-time feature-grid width. Required when ``feature_dim > 0``;
    #: must be ``None`` when features are disabled.
    feature_grid_width: Optional[int] = None
    #: Maximum visible blocks one integration frame may process. Defaults
    #: to :attr:`max_blocks` after that value is computed.
    max_visible_blocks_per_integration: Optional[int] = None
    #: Maximum support pixels stored per visible block per camera for RGB
    #: and feature integration. Larger values use more scratch memory and
    #: preserve more per-block color/feature evidence.
    max_support_pixels_per_block_camera: int = 8
    #: Number of adjacent feature channels accumulated by one feature-kernel
    #: thread. Kept explicit so Python launch grouping and Warp kernel
    #: decoding cannot drift apart.
    feature_channels_per_thread: int = 8
    #: Compile-time cap for feature channels accumulated by one tiled
    #: feature-kernel CTA. The generated tile width is
    #: ``min(feature_dim, max_feature_tile_channels)``.
    max_feature_tile_channels: int = 4096
    #: Feature integration launch policy: ``"auto"``, ``"grouped"``, or
    #: ``"tiled"``. ``"auto"`` resolves to the tiled kernel only for feature
    #: dimensions and support capacities where benchmarks showed a win.
    feature_integration_kernel: str = "auto"
    #: Record per-kernel integration timings into
    #: ``Mapper.get_stats()["last_integration_kernel_timings_ms"]``.
    #: Disabled by default to avoid profiling synchronizations in normal
    #: integration.
    profile_integration_kernel_timings: bool = False
    #: Upper bound on per-block accumulator weight. Caps fp16
    #: ``block_rgb`` / ``block_features`` magnitudes each frame and sets
    #: the EMA decay rate for old observations (effective window
    #: ``~W_max / mean_per_frame_weight``). Raise for longer memory,
    #: lower for faster adaptation to dynamic scenes.
    accumulator_w_max: float = 1000.0

    # === Device ===
    device: str = "cuda:0"

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate extent
        if not all(e > 0 for e in self.extent_meters_xyz):
            raise ValueError(f"extent_meters_xyz must be positive: {self.extent_meters_xyz}")

        # Validate voxel_size
        if self.voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive: {self.voxel_size}")

        # Validate truncation_distance
        if self.truncation_distance <= 0:
            raise ValueError(f"truncation_distance must be positive: {self.truncation_distance}")

        # Validate depth range
        if self.depth_minimum_distance >= self.depth_maximum_distance:
            raise ValueError(
                f"depth_minimum_distance ({self.depth_minimum_distance}) must be < "
                f"depth_maximum_distance ({self.depth_maximum_distance})"
            )

        # Validate decay factors
        if not (0.0 <= self.decay_factor <= 1.0):
            raise ValueError(f"decay_factor must be in (0, 1]: {self.decay_factor}")
        if not (0.0 <= self.frustum_decay_factor <= 1.0):
            raise ValueError(f"frustum_decay_factor must be in (0, 1]: {self.frustum_decay_factor}")

        # Validate hash_load_factor
        if not (0.0 < self.hash_load_factor <= 1.0):
            raise ValueError(f"hash_load_factor must be in (0, 1]: {self.hash_load_factor}")

        if self.image_height is None or self.image_width is None:
            log_and_raise(
                "MapperCfg requires image_height and image_width to pre-allocate "
                "the voxel-project scratch buffer. Got image_height="
                f"{self.image_height}, image_width={self.image_width}."
            )
        if self.image_height <= 0 or self.image_width <= 0:
            log_and_raise(
                f"image_height and image_width must be positive, got "
                f"image_height={self.image_height}, image_width={self.image_width}."
            )
        _validate_feature_channels_per_thread(self.feature_channels_per_thread)
        _validate_feature_grid_shape(
            self.feature_dim,
            self.feature_grid_height,
            self.feature_grid_width,
        )
        if self.max_feature_tile_channels <= 0:
            log_and_raise(
                "max_feature_tile_channels must be positive, got "
                f"{self.max_feature_tile_channels}."
            )
        if self.max_support_pixels_per_block_camera <= 0:
            log_and_raise(
                "max_support_pixels_per_block_camera must be positive: "
                f"{self.max_support_pixels_per_block_camera}"
            )
        _validate_feature_integration_kernel(self.feature_integration_kernel)
        if not isinstance(self.profile_integration_kernel_timings, bool):
            log_and_raise(
                "profile_integration_kernel_timings must be bool, got "
                f"{type(self.profile_integration_kernel_timings).__name__}."
            )
        max_blocks = self.max_blocks
        if self.max_visible_blocks_per_integration is None:
            self.max_visible_blocks_per_integration = max_blocks
        if (
            self.max_visible_blocks_per_integration <= 0
            or self.max_visible_blocks_per_integration > max_blocks
        ):
            log_and_raise(
                "max_visible_blocks_per_integration must satisfy "
                f"0 < C <= max_blocks ({max_blocks}), got "
                f"{self.max_visible_blocks_per_integration}."
            )

        # Set default grid_center
        if self.grid_center is None:
            self.grid_center = torch.zeros(3, dtype=torch.float32)
        elif not isinstance(self.grid_center, torch.Tensor):
            self.grid_center = torch.tensor(self.grid_center, dtype=torch.float32)

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Returns (nz, ny, nx) voxel counts."""
        x, y, z = self.extent_meters_xyz
        nx = math.ceil(x / self.voxel_size)
        ny = math.ceil(y / self.voxel_size)
        nz = math.ceil(z / self.voxel_size)
        return (nz, ny, nx)

    @property
    def max_blocks(self) -> int:
        """Compute max allocated blocks using surface area × thickness heuristic.

        TSDF only allocates blocks near surfaces, not throughout the volume.
        The surface is a volumetric band with thickness = 2 × truncation_distance.

        Formula:
            max_blocks = surface_area_blocks × thickness_factor × roughness

        This scales O(N²) with grid size, not O(N³), matching actual TSDF usage.
        """
        return calculate_tsdf_max_blocks(
            grid_shape=self.grid_shape,
            voxel_size=self.voxel_size,
            block_size=self.block_size,
            truncation_dist=self.truncation_distance,
            roughness=self.roughness,  # roughness scaled by fill ratio
        )

    @property
    def hash_capacity(self) -> int:
        """Hash table capacity based on target load factor."""
        return int(math.ceil(self.max_blocks / self.hash_load_factor))

    def get_actual_extent(self) -> Tuple[float, float, float]:
        """Returns actual physical extent (x, y, z) after ceil() rounding."""
        nz, ny, nx = self.grid_shape
        return (nx * self.voxel_size, ny * self.voxel_size, nz * self.voxel_size)

    def voxel_to_world(
        self, iz: int, iy: int, ix: int
    ) -> Tuple[float, float, float]:
        """Convert voxel index to world coordinate at voxel center.

        Args:
            iz: Z index (0 to nz-1)
            iy: Y index (0 to ny-1)
            ix: X index (0 to nx-1)

        Returns:
            (world_x, world_y, world_z) coordinate at voxel center.
        """
        nz, ny, nx = self.grid_shape
        s = self.voxel_size
        cx, cy, cz = self.grid_center.tolist()
        world_x = cx + (ix - (nx - 1) / 2.0) * s
        world_y = cy + (iy - (ny - 1) / 2.0) * s
        world_z = cz + (iz - (nz - 1) / 2.0) * s
        return (world_x, world_y, world_z)

    def world_to_voxel(
        self, world_x: float, world_y: float, world_z: float
    ) -> Tuple[int, int, int]:
        """Convert world coordinate to voxel index.

        Args:
            world_x: World X coordinate.
            world_y: World Y coordinate.
            world_z: World Z coordinate.

        Returns:
            (iz, iy, ix) voxel indices, or (-1, -1, -1) if out of bounds.
        """
        nz, ny, nx = self.grid_shape
        s = self.voxel_size
        cx, cy, cz = self.grid_center.tolist()
        # Continuous indices
        fx = (world_x - cx) / s + (nx - 1) / 2.0
        fy = (world_y - cy) / s + (ny - 1) / 2.0
        fz = (world_z - cz) / s + (nz - 1) / 2.0
        # Round to nearest voxel
        ix = int(round(fx))
        iy = int(round(fy))
        iz = int(round(fz))
        # Bounds check
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            return (iz, iy, ix)
        return (-1, -1, -1)

    def get_grid_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get world coordinates of grid corners.

        Returns:
            (min_corner, max_corner) as ((x, y, z), (x, y, z)).
        """
        nz, ny, nx = self.grid_shape
        s = self.voxel_size
        cx, cy, cz = self.grid_center.tolist()
        half_extent_x = (nx - 1) / 2.0 * s + s / 2.0
        half_extent_y = (ny - 1) / 2.0 * s + s / 2.0
        half_extent_z = (nz - 1) / 2.0 * s + s / 2.0
        min_corner = (cx - half_extent_x, cy - half_extent_y, cz - half_extent_z)
        max_corner = (cx + half_extent_x, cy + half_extent_y, cz + half_extent_z)
        return (min_corner, max_corner)
