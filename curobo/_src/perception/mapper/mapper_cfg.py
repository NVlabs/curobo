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
from curobo._src.perception.mapper.kernel.warp_types import BLOCK_SIZE


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
        rgb_scale: Downscale factor for RGB storage relative to TSDF.
        block_size: Voxels per block edge (default 8).
        block_fill_ratio: Expected fraction of blocks to be allocated.
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

    # === RGB ===
    rgb_scale: int = 1

    # === Block Storage ===
    # Note: block_size is a module constant (BLOCK_SIZE=8 from warp_types.py)
    block_fill_ratio: float = 1.0
    hash_load_factor: float = 0.5
    roughness: float = 4.0

    # === Integration ===
    integration_method: str = "voxel_project"  # "voxel_project" or "sort_filter"
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
        if not (0.0 < self.decay_factor <= 1.0):
            raise ValueError(f"decay_factor must be in (0, 1]: {self.decay_factor}")
        if not (0.0 < self.frustum_decay_factor <= 1.0):
            raise ValueError(f"frustum_decay_factor must be in (0, 1]: {self.frustum_decay_factor}")

        # Validate block_fill_ratio
        if not (0.0 < self.block_fill_ratio <= 1.0):
            raise ValueError(f"block_fill_ratio must be in (0, 1]: {self.block_fill_ratio}")

        # Validate hash_load_factor
        if not (0.0 < self.hash_load_factor <= 1.0):
            raise ValueError(f"hash_load_factor must be in (0, 1]: {self.hash_load_factor}")

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
    def block_size(self) -> int:
        """Block size (voxels per edge). Fixed at BLOCK_SIZE=8."""
        return BLOCK_SIZE

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
            block_dim=BLOCK_SIZE,
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

