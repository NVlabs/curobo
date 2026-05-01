# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse TSDF Integrator - High-level interface for volumetric mapping.

This module provides a user-friendly interface for block-sparse TSDF integration,
similar to the dense TSDFIntegrator but with 20× memory savings.

Example:
    integrator = BlockSparseTSDFIntegrator(
        voxel_size=0.005,
        origin=torch.tensor([-1.0, -1.0, 0.0]),
        grid_shape=(400, 400, 400),
        max_blocks=100_000,
    )

    for observation in observations:
        integrator.integrate(observation)

    vertices, triangles, colors = integrator.extract_mesh()
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Mesh
from curobo._src.perception.mapper.block_allocation import (
    calculate_tsdf_max_blocks,
)
from curobo._src.perception.mapper.constants import (
    DEFAULT_HASH_LAYOUT,
    _validate_feature_channels_per_thread,
    _validate_feature_grid_shape,
    _validate_feature_integration_kernel,
    _validate_block_size,
    resolve_feature_integration_kernel,
    validate_grid_shape_for_hash_layout,
)
from curobo._src.perception.mapper.kernel.builder.builder_block_sparse_kernel import (
    BlockSparseKernels,
    make_block_sparse_kernels,
)
from curobo._src.perception.mapper.kernel.wp_decay import (
    decay_and_recycle,
    decay_frustum_aware_multi_camera,
)
from curobo._src.perception.mapper.kernel.wp_integrate_voxel_project import (
    VoxelProjectIntegrator,
)
from curobo._src.perception.mapper.kernel.wp_stamp_obstacles import (
    clear_static_channel,
    stamp_scene_obstacles,
)
from curobo._src.perception.mapper.kernel.wp_voxel_extraction import (
    extract_matching_voxels_block_sparse,
    extract_occupied_voxels_block_sparse,
    extract_surface_voxels_block_sparse,
)
from curobo._src.perception.mapper.mesh_extractor import (
    extract_mesh_block_sparse,
)
from curobo._src.perception.mapper.storage import (
    BlockSparseTSDF,
    BlockSparseTSDFCfg,
    MatchedVoxels,
    OccupiedVoxels,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.util.logging import log_info
from curobo._src.util.torch_util import profile_class_methods
from curobo.logging import log_and_raise


@dataclass
class BlockSparseTSDFIntegratorCfg:
    """Configuration for BlockSparseTSDFIntegrator.

    Attributes:
        voxel_size: Size of each voxel in meters.
        origin: World coordinate of grid origin (3,).
        truncation_distance: TSDF truncation distance in meters.
        max_blocks: Maximum number of allocatable blocks.
            If None and grid_shape is provided, auto-calculated using surface area heuristic.
        hash_capacity: Hash table size (should be ~2× max_blocks). Auto-calculated if None.
        block_size: Voxels per block edge. Supported values are 1 or
            powers of 2 in [2, 32]
            (default 8). Specializes the Warp kernels used by this
            integrator.
        depth_minimum_distance: Minimum valid depth in meters.
        depth_maximum_distance: Maximum valid depth in meters.
        frustum_decay: Decay for in-view voxels (0.5=quick adapt, 1.0=no extra decay).
        time_decay: Decay for all voxels (1.0=persist, 0.99=slow fade).
        minimum_tsdf_weight: Minimum weight to consider a voxel as observed.
            Since weight = 1/depth² (clamped to [0.001, 2.0]), this can be interpreted
            as the number of observations at 1m depth. E.g., 0.5 = half an observation
            at 1m, or one observation at ~1.4m, or two observations at 2m.
        grid_shape: Required (nz, ny, nx) voxel grid dimensions. The map
            uses center-origin convention, and voxels outside the bounded
            grid are skipped.
        roughness: Geometric complexity multiplier for max_blocks calculation.
            1.0 = simple walls, 3.0 = cluttered room (default).
        image_height: Image height for buffer pre-allocation.
        image_width: Image width for buffer pre-allocation.
        device: CUDA device.
    """

    voxel_size: float = 0.005
    origin: torch.Tensor = None
    truncation_distance: float = 0.04
    max_blocks: Optional[int] = None  # Auto-calculate if None and grid_shape provided
    hash_capacity: Optional[int] = None  # Auto-calculate as 2× max_blocks
    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 5.0
    frustum_decay: float = 1.0  # 1.0 = no extra decay for in-view voxels
    time_decay: float = 1.0  # 1.0 = no time decay
    minimum_tsdf_weight: float = 0.1
    grid_shape: Tuple[int, int, int] = None
    roughness: float = 3.0  # Geometric complexity multiplier
    image_height: Optional[int] = None  # For buffer pre-allocation
    image_width: Optional[int] = None  # For buffer pre-allocation
    enable_static: bool = False  # Enable static obstacle integration
    static_obstacle_color: Tuple[int, int, int] = (20, 20, 20)  # RGB for static obstacles
    seeding_method: str = "gather"  # "gather" or "scatter"; selects ESDF seed kernel variant
    num_cameras: int = 1
    device: str = "cuda:0"
    #: Voxels per block edge. Supported values are 1 or powers of 2 in [2, 32].
    block_size: int = 8
    #: Per-block feature channel dimensionality. 0 disables features.
    feature_dim: int = 0
    #: Compile-time feature-grid height. Required when ``feature_dim > 0``;
    #: must be ``None`` when features are disabled.
    feature_grid_height: Optional[int] = None
    #: Compile-time feature-grid width. Required when ``feature_dim > 0``;
    #: must be ``None`` when features are disabled.
    feature_grid_width: Optional[int] = None
    #: Maximum visible blocks one integration frame may process. Defaults
    #: to ``max_blocks`` after auto-calculation.
    max_visible_blocks_per_integration: Optional[int] = None
    #: Maximum support pixels stored per visible block per camera for RGB
    #: and feature integration. ``32`` matches the memory budget in the
    #: visible-capacity refactor note and keeps scratch bounded.
    max_support_pixels_per_block_camera: int = 32
    #: Number of adjacent feature channels accumulated by one feature-kernel
    #: thread. Must match the compiled Warp kernel's thread decoding.
    feature_channels_per_thread: int = 4
    #: Compile-time cap for feature channels accumulated by one tiled
    #: feature-kernel CTA.
    max_feature_tile_channels: int = 4096
    #: Feature integration launch policy: ``"auto"``, ``"grouped"``, or
    #: ``"tiled"``. Resolved to a low-level tiled bool at construction time.
    feature_integration_kernel: str = "auto"
    #: Record per-kernel integration timings into
    #: ``get_stats()["last_integration_kernel_timings_ms"]``.
    profile_integration_kernel_timings: bool = False
    #: Upper bound on per-block accumulator weight after each frame. Caps
    #: the fp16 weighted-sum magnitudes in ``block_rgb`` and
    #: ``block_features``; also sets EMA decay rate
    #: ``W_max / mean_per_frame_weight`` for old observations. See
    #: :attr:`BlockSparseTSDFCfg.accumulator_w_max`.
    accumulator_w_max: float = 10000.0

    def __post_init__(self):
        if self.origin is None:
            self.origin = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.float32)
        if not isinstance(self.origin, torch.Tensor):
            self.origin = torch.tensor(self.origin, dtype=torch.float32)
        _validate_block_size(self.block_size)
        self.grid_shape = validate_grid_shape_for_hash_layout(
            self.grid_shape,
            self.block_size,
            field_name="BlockSparseTSDFIntegratorCfg.grid_shape",
        )

        # Auto-calculate max_blocks if not provided
        if self.max_blocks is None:
            self.max_blocks = calculate_tsdf_max_blocks(
                grid_shape=self.grid_shape,
                voxel_size=self.voxel_size,
                block_size=self.block_size,
                truncation_dist=self.truncation_distance,
                roughness=self.roughness,
            )
        if self.max_blocks > DEFAULT_HASH_LAYOUT.max_pool_idx:
            log_and_raise(
                f"max_blocks={self.max_blocks:,} exceeds the "
                f"{DEFAULT_HASH_LAYOUT.name} pool limit of "
                f"{DEFAULT_HASH_LAYOUT.max_pool_idx:,}."
            )

        # Auto-calculate hash_capacity as 2× max_blocks (active load factor
        # ~0.5). ``hash_lookup`` now probes the full capacity (not a fixed
        # 64-slot cap), so probe-chain saturation from recycle-induced
        # tombstones no longer produces silent allocation failures; the
        # tighter 2× packing is safe.
        if self.hash_capacity is None:
            self.hash_capacity = int(math.ceil(self.max_blocks * 2.0))
        if self.max_visible_blocks_per_integration is None:
            self.max_visible_blocks_per_integration = self.max_blocks
        if (
            self.max_visible_blocks_per_integration <= 0
            or self.max_visible_blocks_per_integration > self.max_blocks
        ):
            log_and_raise(
                "max_visible_blocks_per_integration must satisfy "
                f"0 < C <= max_blocks ({self.max_blocks}), got "
                f"{self.max_visible_blocks_per_integration}."
            )

        if self.image_height is None or self.image_width is None:
            log_and_raise(
                "BlockSparseTSDFIntegratorCfg requires image_height and "
                "image_width to pre-allocate the voxel-project scratch "
                "buffer. Got image_height="
                f"{self.image_height}, image_width={self.image_width}."
            )
        if self.image_height <= 0 or self.image_width <= 0:
            log_and_raise(
                f"image_height and image_width must be positive, got "
                f"image_height={self.image_height}, image_width={self.image_width}."
            )
        if self.num_cameras <= 0:
            log_and_raise(f"num_cameras must be positive, got num_cameras={self.num_cameras}.")
        _validate_feature_channels_per_thread(self.feature_channels_per_thread)
        _validate_feature_grid_shape(
            self.feature_dim,
            self.feature_grid_height,
            self.feature_grid_width,
        )
        if self.max_feature_tile_channels <= 0:
            log_and_raise(
                "max_feature_tile_channels must be positive, got "
                f"max_feature_tile_channels={self.max_feature_tile_channels}."
            )
        if self.max_support_pixels_per_block_camera <= 0:
            log_and_raise(
                "max_support_pixels_per_block_camera must be positive, got "
                f"max_support_pixels_per_block_camera="
                f"{self.max_support_pixels_per_block_camera}."
            )
        _validate_feature_integration_kernel(self.feature_integration_kernel)
        if not isinstance(self.profile_integration_kernel_timings, bool):
            log_and_raise(
                "profile_integration_kernel_timings must be bool, got "
                f"{type(self.profile_integration_kernel_timings).__name__}."
            )


@profile_class_methods
class BlockSparseTSDFIntegrator:
    """Block-sparse TSDF Integrator for memory-efficient volumetric mapping.

    This class provides a high-level interface for integrating depth images
    into a block-sparse TSDF representation. It uses ~20× less memory than
    the dense TSDFIntegrator while maintaining CUDA graph compatibility.

    Features:
        - On-demand block allocation (only allocates observed regions)
        - Optional weight decay for dynamic scene tracking
        - Block recycling to bound memory usage
        - Mesh extraction via marching cubes

    Example:
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.005,
            origin=torch.tensor([-1.0, -1.0, 0.0]),
        )
        integrator = BlockSparseTSDFIntegrator(config)

        # Integrate depth frames
        for obs in observations:
            integrator.integrate(obs)

        # Extract mesh
        mesh = integrator.extract_mesh()
    """

    def __init__(
        self,
        config: BlockSparseTSDFIntegratorCfg,
        kernels: Optional[BlockSparseKernels] = None,
    ):
        """Initialize the integrator.

        Args:
            config: Configuration dataclass.
            kernels: Optional prebuilt kernel bundle. ESDF integrators pass
                a bundle built from the richer ESDF config so ESDF-specific
                compile-time constants stay owned by the ESDF layer.
        """
        self.config = config

        use_tiled_feature_kernel = resolve_feature_integration_kernel(
            config.feature_integration_kernel,
            config.feature_dim,
            config.max_support_pixels_per_block_camera,
        )

        tsdf_config = BlockSparseTSDFCfg(
            max_blocks=config.max_blocks,
            hash_capacity=config.hash_capacity,
            voxel_size=config.voxel_size,
            origin=config.origin,
            truncation_distance=config.truncation_distance,
            device=config.device,
            grid_shape=config.grid_shape,
            enable_static=config.enable_static,
            static_obstacle_color=config.static_obstacle_color,
            block_size=config.block_size,
            feature_dim=config.feature_dim,
            feature_grid_height=config.feature_grid_height,
            feature_grid_width=config.feature_grid_width,
            feature_channels_per_thread=config.feature_channels_per_thread,
            max_feature_tile_channels=config.max_feature_tile_channels,
            max_support_pixels_per_block_camera=config.max_support_pixels_per_block_camera,
            accumulator_w_max=config.accumulator_w_max,
        )
        if kernels is None:
            kernels = make_block_sparse_kernels(config)
        self._tsdf = BlockSparseTSDF(tsdf_config, kernels=kernels)

        # Frame counter for periodic operations
        self._frame_count = 0

        self._integrator = VoxelProjectIntegrator(
            num_cameras=config.num_cameras,
            image_height=config.image_height,
            image_width=config.image_width,
            voxel_size=config.voxel_size,
            block_size=config.block_size,
            truncation_distance=config.truncation_distance,
            max_blocks=self._tsdf.config.max_blocks,
            max_visible_blocks_per_integration=config.max_visible_blocks_per_integration,
            max_support_pixels_per_block_camera=config.max_support_pixels_per_block_camera,
            device=config.device,
            feature_channels_per_thread=config.feature_channels_per_thread,
            use_tiled_feature_kernel=use_tiled_feature_kernel,
            feature_grid_shape=(
                (config.feature_grid_height, config.feature_grid_width)
                if config.feature_grid_height is not None
                else None
            ),
            profile_kernel_timings=config.profile_integration_kernel_timings,
        )

    @property
    def tsdf(self) -> BlockSparseTSDF:
        """Get the underlying BlockSparseTSDF instance."""
        return self._tsdf

    def reset(self):
        """Reset the integrator to initial state."""
        self._tsdf.reset()
        self._frame_count = 0

    def integrate(
        self,
        observation: CameraObservation,
    ):
        """Integrate a batched camera observation into the TSDF.

        The observation must have a leading camera dimension matching
        ``config.num_cameras``:
        - ``depth_image``: ``(num_cameras, H, W)``
        - ``rgb_image``: ``(num_cameras, H, W, 3)``
        - ``intrinsics``: ``(num_cameras, 3, 3)``
        - ``pose.position``: ``(num_cameras, 3)``
        - ``pose.quaternion``: ``(num_cameras, 4)``
        - ``feature_grid``: optional ``(num_cameras, feature_H, feature_W, feature_dim)``
          float16 tensor when features are enabled

        Args:
            observation: Batched camera observation.

        Raises:
            ValueError: If the leading dimension does not match ``num_cameras``.
        """
        depth_images = observation.depth_image
        if depth_images.ndim != 3:
            raise ValueError(
                f"depth_image must be (num_cameras, H, W), got shape {depth_images.shape}"
            )
        n_cameras = depth_images.shape[0]
        if n_cameras != self.config.num_cameras:
            raise ValueError(
                f"Expected num_cameras={self.config.num_cameras}, "
                f"got depth_image batch dim {n_cameras}"
            )

        rgb_images = observation.rgb_image
        positions = observation.pose.position.view(n_cameras, 3)
        quaternions = observation.pose.quaternion.view(n_cameras, 4)
        intrinsics = observation.intrinsics

        feature_grid = observation.feature_grid
        if feature_grid is not None and not self._tsdf.data.has_features:
            log_and_raise(
                "feature_grid was provided but feature_dim == 0; enable features via "
                "MapperCfg.feature_dim or BlockSparseTSDFIntegratorCfg.feature_dim."
            )
        if self._tsdf.data.has_features and feature_grid is not None:
            if feature_grid.ndim != 4:
                log_and_raise(
                    "feature_grid must be (num_cameras, feature_H, feature_W, feature_dim), "
                    f"got shape {tuple(feature_grid.shape)}."
                )
            if feature_grid.shape[0] != n_cameras:
                log_and_raise(
                    f"feature_grid num_cameras={feature_grid.shape[0]} "
                    f"does not match depth_image num_cameras={n_cameras}."
                )
            if feature_grid.shape[1] <= 0 or feature_grid.shape[2] <= 0:
                log_and_raise(
                    "feature_grid feature_H and feature_W must be positive, got "
                    f"shape {tuple(feature_grid.shape)}."
                )
            if feature_grid.shape[-1] != self._tsdf.data.feature_dim:
                log_and_raise(
                    f"feature_grid feature_dim={feature_grid.shape[-1]} does not match "
                    f"configured feature_dim={self._tsdf.data.feature_dim}."
                )
            if self._tsdf.kernels.feature_grid_shape is not None:
                expected_feature_height, expected_feature_width = (
                    self._tsdf.kernels.feature_grid_shape
                )
                if (
                    int(feature_grid.shape[1]) != expected_feature_height
                    or int(feature_grid.shape[2]) != expected_feature_width
                ):
                    log_and_raise(
                        "feature_grid shape mismatch: expected "
                        f"feature_H={expected_feature_height}, "
                        f"feature_W={expected_feature_width}, got "
                        f"{tuple(feature_grid.shape)}."
                    )
            if feature_grid.dtype != torch.float16:
                log_and_raise(
                    f"feature_grid dtype must be torch.float16, got {feature_grid.dtype}."
                )
            if feature_grid.device != depth_images.device:
                log_and_raise(
                    f"feature_grid device {feature_grid.device} does not match "
                    f"depth_image device {depth_images.device}."
                )
            if feature_grid.stride(-1) != 1:
                log_and_raise(
                    "feature_grid must be channels-last with stride 1 on the channel dim."
                )

        if self._frame_count > 0 and (self.config.time_decay < 1.0 or self.config.frustum_decay < 1.0):
            img_shape = (depth_images.shape[1], depth_images.shape[2])
            num_blocks = int(self._tsdf.data.num_allocated.item())
            decay_frustum_aware_multi_camera(
                self._tsdf,
                intrinsics=intrinsics,
                cam_positions=positions,
                cam_quaternions=quaternions,
                img_shape=img_shape,
                depth_minimum_distance=self.config.depth_minimum_distance,
                depth_maximum_distance=self.config.depth_maximum_distance,
                time_decay=self.config.time_decay,
                frustum_decay=self.config.frustum_decay,
                num_blocks=num_blocks,
            )

        self._integrator.integrate(
            self._tsdf,
            depth_images,
            rgb_images,
            positions,
            quaternions,
            intrinsics,
            depth_min=self.config.depth_minimum_distance,
            depth_max=self.config.depth_maximum_distance,
            grid_size=self.config.grid_shape,
            feature_grid=feature_grid,
        )

        self._frame_count += 1

    def recycle_empty_blocks(self) -> int:
        """Recycle blocks that have decayed to empty.

        Call this periodically (e.g., every 100 frames) to reclaim memory
        from blocks that are no longer observed.

        Returns:
            Number of blocks recycled.
        """
        return decay_and_recycle(self._tsdf, 1.0)  # No additional decay

    def clear_region(self, bounds_min, bounds_max) -> int:
        """Clear dynamic map contents for allocated blocks intersecting an AABB.

        Args:
            bounds_min: World-space lower AABB corner, shape ``(3,)``.
            bounds_max: World-space upper AABB corner, shape ``(3,)``.

        Returns:
            Number of allocated blocks cleared. Blocks remain allocated; only
            dynamic TSDF/RGB, block sums, and per-block features are reset.
        """
        return self._integrator.clear_region(self._tsdf, bounds_min, bounds_max)

    def clear_blocks(self, pool_indices) -> int:
        """Clear dynamic map contents for explicit block-pool indices.

        Args:
            pool_indices: Tensor/list of allocated ``pool_idx`` values.

        Returns:
            Number of blocks cleared. Blocks remain allocated; only dynamic
            TSDF/RGB, block sums, and per-block features are reset.
        """
        return self._integrator.clear_blocks(self._tsdf, pool_indices)

    def extract_mesh(
        self,
        refine_iterations: int = 2,
        surface_only: bool = False,
        level: float = 0.0,
    ) -> Mesh:
        """Extract mesh from the TSDF using marching cubes.

        Args:
            refine_iterations: Number of Newton-Raphson iterations for vertex refinement.
                0 = no refinement (linear interpolation only). Higher values (2-5)
                produce smoother meshes at the cost of more computation.
            surface_only: If True, only extract mesh near the surface (|sdf| < truncation).
                Excludes triangles from regions deep inside the object where TSDF
                is clamped to -truncation_distance.
            level: Isosurface level (typically 0.0).


        Returns:
            Mesh object with vertices, triangles, normals, and colors.
        """
        vertices, triangles, normals, colors = extract_mesh_block_sparse(
            self._tsdf,
            level=level,
            surface_only=surface_only,
            refine_iterations=refine_iterations,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
        )

        # Convert tensors to lists for Mesh class
        # faces should be (N, 3) for trimesh compatibility
        return Mesh(
            name="block_sparse_tsdf_mesh",
            vertices=vertices,  # .cpu().tolist(),
            faces=triangles,  # .cpu().tolist(),  # Keep as (N, 3), not flattened
            vertex_colors=(colors.float() / 255.0),  # .cpu().tolist(),
            vertex_normals=normals,  # .cpu().tolist(),
        )

    def extract_mesh_tensors(
        self,
        level: float = 0.0,
        surface_only: bool = False,
        refine_iterations: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract mesh as raw tensors.

        Args:
            level: Isosurface level (typically 0.0).
            surface_only: If True, only extract mesh near the surface (|sdf| < truncation).
                Excludes triangles from regions deep inside the object where TSDF
                is clamped to -truncation_distance.
            refine_iterations: Number of Newton-Raphson iterations for vertex refinement.
                0 = no refinement (linear interpolation only). Higher values (2-5)
                produce smoother meshes at the cost of more computation.

        Returns:
            Tuple of (vertices, triangles, normals, colors):
                - vertices: (N, 3) float32
                - triangles: (M, 3) int32
                - normals: (N, 3) float32
                - colors: (N, 3) uint8
        """
        return extract_mesh_block_sparse(
            self._tsdf,
            level=level,
            surface_only=surface_only,
            refine_iterations=refine_iterations,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
        )

    def get_stats(
        self,
        scan_pool: bool = True,
        scan_hash: bool = False,
    ) -> Dict[str, Any]:
        """Get integrator statistics.

        Args:
            scan_pool: Forwarded to ``BlockSparseTSDF.get_stats``.
                When True (default) returns the ground-truth
                ``active_blocks`` and the ``holes`` invariant via an
                O(num_allocated) GPU reduction.
            scan_hash: Forwarded to ``BlockSparseTSDF.get_stats``.
                When True, adds hash table occupancy stats via an
                O(hash_capacity) reduction. Periodic use only.

        Returns:
            Dictionary with block usage, allocation stats, and the last
            integration's kernel timing dict. See
            :meth:`BlockSparseTSDF.get_stats` for the full block-pool key list.
        """
        stats = self._tsdf.get_stats(scan_pool=scan_pool, scan_hash=scan_hash)
        stats["frame_count"] = self._frame_count
        stats["memory_mb"] = self._tsdf.memory_usage_mb()
        last_integration = dict(self._integrator.last_integration_stats)
        last_integration["support_overflow_count"] = int(
            self._integrator.support_overflow_count.item()
        )
        last_integration["profile_kernel_timings"] = self._integrator.profile_kernel_timings
        last_integration["use_tiled_feature_kernel"] = self._integrator.use_tiled_feature_kernel
        stats["last_integration"] = last_integration
        stats["last_integration_kernel_timings_ms"] = dict(
            self._integrator.last_kernel_timings_ms
        )
        return stats

    def memory_usage_mb(self) -> float:
        """Get current GPU memory usage in megabytes."""
        return self._tsdf.memory_usage_mb()

    def extract_surface_voxels(
        self,
        sdf_threshold: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract surface voxel centers, colors, and SDF values.

        This extracts raw voxel data for debugging (bypasses marching cubes).
        Surface voxels are those with |SDF| < sdf_threshold and weight > minimum_tsdf_weight.

        Args:
            sdf_threshold: Maximum |SDF| for surface voxels. Defaults to truncation_distance.
            minimum_tsdf_weight: Minimum weight for valid voxels.

        Returns:
            Tuple of (centers, colors, sdf_values):
                - centers: (N, 3) float32 voxel world positions
                - colors: (N, 3) uint8 RGB colors
                - sdf_values: (N,) float32 SDF values
        """
        if sdf_threshold is None:
            sdf_threshold = self.config.truncation_distance

        return extract_surface_voxels_block_sparse(
            self._tsdf,
            sdf_threshold=sdf_threshold,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            grid_shape=self.config.grid_shape,
        )

    def extract_occupied_voxels(
        self,
        surface_only: bool = False,
        sdf_threshold: float = None,
    ) -> OccupiedVoxels:
        """Extract occupied voxel centers and per-voxel block indices.

        Matches dense TSDFIntegrator.extract_occupied_voxels behavior:
        - surface_only=True: extract voxels near zero-crossing (|SDF| < sdf_threshold)
        - surface_only=False: extract surface + inside voxels (SDF <= 0)

        Args:
            surface_only: If True, extract only surface voxels (|SDF| < sdf_threshold).
                          If False (default), include inside voxels too (SDF <= 0).
            sdf_threshold: Threshold for surface_only=True. Defaults to voxel_size.

        Returns:
            :class:`OccupiedVoxels` — ``.centers`` for voxel positions,
            ``.block_idx_per_voxel`` for the per-voxel ``pool_idx``, and
            ``.block_data`` as a read-only view over per-block storage.
            Use ``.colors_uint8()`` to gather per-voxel RGB.
        """
        # Default threshold to voxel_size (matches dense behavior)
        if sdf_threshold is None:
            sdf_threshold = self.config.voxel_size

        return extract_occupied_voxels_block_sparse(
            self._tsdf,
            surface_only=surface_only,
            sdf_threshold=sdf_threshold,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            grid_shape=self.config.grid_shape,
        )

    def extract_matching_feature_voxels(
        self,
        feature_vector: torch.Tensor,
        top_k: int,
        surface_only: bool = False,
        sdf_threshold: Optional[float] = None,
        minimum_score: Optional[float] = None,
        feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> MatchedVoxels:
        """Extract voxels from the top-k blocks whose per-block feature vector
        is most similar (cosine) to ``feature_vector``.

        Args:
            feature_vector: ``(D,)`` float32 query vector. ``D`` must
                match ``feature_dim`` unless ``feature_projector`` maps
                stored block features to another query space.
            top_k: Number of most-similar blocks to extract voxels from.
            surface_only: If True, keep only surface voxels.
            sdf_threshold: Surface threshold; defaults to voxel_size.
            minimum_score: If set, drop blocks with cosine score strictly
                below this threshold *after* the top-k selection. Lets
                callers pair a generous ``top_k`` with a quality cut
                without paying voxel-extraction cost on rejected blocks.
                Score is cosine in the same feature space as
                ``feature_vector`` (range ``[-1, 1]``); pick a threshold
                appropriate for that space (raw RADIO and teacher-projected
                features have different score distributions). ``None``
                disables the cut.
            feature_projector: Optional callable applied to normalized
                per-block features before scoring. It receives a
                ``(num_allocated, feature_dim)`` float32 tensor and must
                return ``(num_allocated, D)`` features in the same space
                as ``feature_vector``.

        Returns:
            :class:`MatchedVoxels` containing the extracted voxels plus
            parallel ``(K,)`` arrays of pool indices and cosine scores
            sorted by score descending. ``K <= min(top_k, num_allocated)``;
            ``K`` may be smaller (including zero) when ``minimum_score``
            filters blocks.
        """
        if not self._tsdf.data.has_features:
            raise RuntimeError("extract_matching_feature_voxels() requires feature_dim > 0")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        feature_vector = feature_vector.to(
            device=self._tsdf.data.block_features.device, dtype=torch.float32
        )
        if feature_vector.ndim != 1:
            raise ValueError(
                f"feature_vector must be shape (D,), got {tuple(feature_vector.shape)}"
            )
        if feature_projector is None and feature_vector.shape[0] != self._tsdf.data.feature_dim:
            raise ValueError(
                "feature_vector must be shape "
                f"({self._tsdf.data.feature_dim},), got {tuple(feature_vector.shape)}"
            )

        num_alloc = int(self._tsdf.data.num_allocated.item())
        max_blocks = self._tsdf.config.max_blocks
        device = self._tsdf.device

        block_mask = torch.zeros(max_blocks, dtype=torch.uint8, device=device)

        def _empty_result() -> MatchedVoxels:
            voxels = extract_matching_voxels_block_sparse(
                self._tsdf,
                block_mask=block_mask,
                surface_only=surface_only,
                sdf_threshold=sdf_threshold,
                minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            )
            return MatchedVoxels(
                voxels=voxels,
                block_pool_idx=torch.empty(0, dtype=torch.int32, device=device),
                block_scores=torch.empty(0, dtype=torch.float32, device=device),
            )

        if num_alloc == 0:
            return _empty_result()

        active_pool_idx = torch.nonzero(
            self._tsdf.data.block_to_hash_slot[:num_alloc] >= 0,
            as_tuple=False,
        ).flatten()
        if active_pool_idx.numel() == 0:
            return _empty_result()

        # Accumulators are fp16; divide in fp32 to avoid compounding
        # ulp loss on top of the post-frame rescale, and to match the
        # fp32 query vector for the dot product. Recycled pool slots may
        # retain stale storage, so score only active pool indices.
        features = self._tsdf.data.block_features[active_pool_idx].float()
        weight = self._tsdf.data.block_feature_weight[active_pool_idx].float().clamp(min=1e-6)
        normalized = features / weight.unsqueeze(-1)

        if feature_projector is not None:
            with torch.inference_mode():
                normalized = feature_projector(normalized)
            if normalized.ndim != 2 or normalized.shape[0] != active_pool_idx.numel():
                raise ValueError(
                    "feature_projector must return shape "
                    f"({active_pool_idx.numel()}, D), got {tuple(normalized.shape)}"
                )
            normalized = normalized.to(device=device, dtype=torch.float32)

        if feature_vector.shape[0] != normalized.shape[1]:
            raise ValueError(
                "feature_vector must be shape "
                f"({normalized.shape[1]},), got {tuple(feature_vector.shape)}"
            )

        query = torch.nn.functional.normalize(feature_vector, dim=0, eps=1e-6)
        normalized = torch.nn.functional.normalize(normalized, dim=1, eps=1e-6)
        scores = torch.nan_to_num(normalized @ query, nan=-float("inf"))

        k_effective = min(top_k, active_pool_idx.numel())
        # topk(..., sorted=True) is the default and gives descending order;
        # both block_pool_idx and block_scores inherit it by construction,
        # which preserves the parallel-arrays invariant for downstream
        # consumers (slicing prefixes, scatter-gather in scores_per_voxel).
        topk = torch.topk(scores, k_effective)
        top = active_pool_idx[topk.indices].to(torch.int64)
        top_scores = topk.values

        if minimum_score is not None:
            # Apply the threshold AFTER topk so the kept indices stay in
            # descending-score order without a re-sort, and so we never
            # waste topk on more blocks than necessary. The Warp
            # extraction kernel below sees a strictly smaller mask, so
            # the perf win is proportional to (k_effective - K_kept).
            keep = top_scores >= minimum_score
            if not bool(keep.any()):
                return _empty_result()
            top = top[keep]
            top_scores = top_scores[keep]

        block_mask.index_fill_(0, top, 1)

        voxels = extract_matching_voxels_block_sparse(
            self._tsdf,
            block_mask=block_mask,
            surface_only=surface_only,
            sdf_threshold=sdf_threshold,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
        )
        return MatchedVoxels(
            voxels=voxels,
            block_pool_idx=top.to(torch.int32),
            block_scores=top_scores.float(),
        )

    def get_matching_feature_voxels(
        self,
        feature_vector: torch.Tensor,
        top_k: int,
        surface_only: bool = False,
        sdf_threshold: Optional[float] = None,
        minimum_score: Optional[float] = None,
        feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> MatchedVoxels:
        """Compatibility wrapper for :meth:`extract_matching_feature_voxels`."""
        return self.extract_matching_feature_voxels(
            feature_vector=feature_vector,
            top_k=top_k,
            surface_only=surface_only,
            sdf_threshold=sdf_threshold,
            minimum_score=minimum_score,
            feature_projector=feature_projector,
        )

    # =========================================================================
    # Static Obstacle Integration
    # =========================================================================

    def update_static_obstacles(
        self,
        scene: SceneData,
        env_idx: int = 0,
        debug: bool = False,
    ) -> None:
        """Update static obstacles in the TSDF from scene collision tensors.

        This method clears the static SDF channel and re-stamps all primitives
        from the provided scene. The static channel does not decay and is
        combined with the dynamic (depth) channel using min() for collision.

        Note: This is not intended to be called every frame. Use it when static
        obstacle poses change (e.g., after robot reconfiguration).

        Args:
            scene: Scene collision tensors containing cuboids and meshes.
            env_idx: Environment index for multi-environment scenes.
            debug: If True, log diagnostic information.
        """
        tsdf = self._tsdf

        # Check if static channel is enabled
        if not tsdf.data.has_static:
            if debug:
                log_info("[update_static_obstacles] SKIP: has_static=False")
            return

        if debug:
            log_info(f"[update_static_obstacles] num_allocated={tsdf.data.num_allocated.item()}")

        # Step 1: Clear static channel to +inf
        clear_static_channel(tsdf.data)

        # Step 2: Recycle blocks that are now empty (no dynamic AND no static data)
        decay_and_recycle(tsdf, decay_factor=1.0)

        # Step 3: Stamp all obstacle types (cuboids, meshes, voxels) generically
        stamp_scene_obstacles(tsdf, scene, env_idx, self.config.static_obstacle_color, debug=debug)
