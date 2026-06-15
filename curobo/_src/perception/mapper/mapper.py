# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Mapper - Unified volumetric mapping interface.

This module provides the Mapper class, a high-level facade for volumetric
mapping using block-sparse TSDF storage with ESDF generation.

Example:
    from curobo._src.perception.mapper.mapper import Mapper
    from curobo._src.perception.mapper.mapper_cfg import MapperCfg

    config = MapperCfg(
        extent_meters_xyz=(2.0, 2.0, 2.0),
        voxel_size=0.005,
    )
    mapper = Mapper(config)

    for obs in observations:
        mapper.integrate(obs)

    voxel_grid = mapper.compute_esdf()
    mesh = mapper.extract_mesh()
"""

from __future__ import annotations

import math
from os import PathLike
from typing import Callable, Optional, Tuple, Union

import torch

from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Mesh, VoxelGrid
from curobo._src.perception.mapper.checkpoint_blocks import (
    build_block_metadata,
    load_block_checkpoint,
    prepare_blocks_for_import,
    save_block_checkpoint,
    validate_block_metadata_for_target,
)
from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegrator,
    BlockSparseESDFIntegratorCfg,
)
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.perception.mapper.storage import (
    BlockSparseTSDF,
    MatchedVoxels,
    OccupiedVoxels,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.lidar import LidarObservation
from curobo._src.types.pose import Pose
from curobo._src.util.logging import deprecated, log_and_raise


# Per-element magnitude limit for CameraObservation.feature_grid. The
# integration kernel atomic-adds into fp16 ``block_features``; inputs
# well outside ``O(1)`` can overflow the per-thread footprint sum
# (``~support_capacity · max_per_pixel_value``) within a single frame
# before the post-integration rescale kicks in.
_FEATURE_MAGNITUDE_LIMIT: float = 10.0


@deprecated(
    "Use Mapper.integrate(camera_observation=...) or Mapper.integrate(lidar_observation=...) "
    "instead."
)
def _deprecated_mapper_integrate_observation_alias() -> None:
    pass


class Mapper:
    """Volumetric mapper using block-sparse storage.

    Encapsulates block-sparse TSDF integrator with ESDF generation.
    Provides a simplified API compared to BlockSparseESDFIntegrator.

    Args:
        config: MapperCfg with grid and integration parameters.

    Example:
        config = MapperCfg(
            extent_meters_xyz=(2.0, 2.0, 2.0),
            voxel_size=0.005,
        )
        mapper = Mapper(config)

        for obs in observations:
            mapper.integrate(obs)

        voxel_grid = mapper.compute_esdf()
        mesh = mapper.extract_mesh()
    """

    def __init__(self, config: MapperCfg):
        """Initialize Mapper with configuration."""
        self.config = config
        self._device = torch.device(config.device)
        if config.extent_esdf_meters_xyz is None:
            esdf_grid_shape = (128, 128, 128)
        else:
            esdf_grid_shape = (
                math.ceil(config.extent_esdf_meters_xyz[0] / config.esdf_voxel_size),
                math.ceil(config.extent_esdf_meters_xyz[1] / config.esdf_voxel_size),
                math.ceil(config.extent_esdf_meters_xyz[2] / config.esdf_voxel_size),
            )

        # Convert MapperCfg to BlockSparseESDFIntegratorCfg
        integrator_config = BlockSparseESDFIntegratorCfg(
            voxel_size=config.voxel_size,
            origin=config.grid_center.to(self._device),
            esdf_grid_shape=esdf_grid_shape,
            esdf_voxel_size=config.esdf_voxel_size,
            truncation_distance=config.truncation_distance,
            max_blocks=config.max_blocks,
            hash_capacity=config.hash_capacity,
            depth_minimum_distance=config.depth_minimum_distance,
            depth_maximum_distance=config.depth_maximum_distance,
            frustum_decay=config.frustum_decay_factor,
            time_decay=config.decay_factor,
            minimum_tsdf_weight=config.minimum_tsdf_weight,
            grid_shape=config.grid_shape,
            enable_static=config.enable_static,
            static_obstacle_color=config.static_obstacle_color,
            seeding_method=config.seeding_method,
            edt_solver=config.edt_solver,
            num_cameras=config.num_cameras,
            image_height=config.image_height,
            image_width=config.image_width,
            lidar_num_sensors=config.lidar_num_sensors,
            lidar_image_height=config.lidar_image_height,
            lidar_image_width=config.lidar_image_width,
            lidar_feature_grid_height=config.lidar_feature_grid_height,
            lidar_feature_grid_width=config.lidar_feature_grid_width,
            lidar_linear_interpolation_max_allowable_difference_vox=(
                config.lidar_linear_interpolation_max_allowable_difference_vox
            ),
            lidar_nearest_interpolation_max_allowable_dist_to_ray_vox=(
                config.lidar_nearest_interpolation_max_allowable_dist_to_ray_vox
            ),
            max_visible_blocks_per_lidar_integration=(
                config.max_visible_blocks_per_lidar_integration
            ),
            max_support_pixels_per_block_lidar=config.max_support_pixels_per_block_lidar,
            device=config.device,
            block_size=config.block_size,
            roughness=config.roughness,
            feature_dim=config.feature_dim,
            feature_grid_height=config.feature_grid_height,
            feature_grid_width=config.feature_grid_width,
            max_visible_blocks_per_integration=config.max_visible_blocks_per_integration,
            max_support_pixels_per_block_camera=config.max_support_pixels_per_block_camera,
            feature_channels_per_thread=config.feature_channels_per_thread,
            max_feature_tile_channels=config.max_feature_tile_channels,
            feature_integration_kernel=config.feature_integration_kernel,
            profile_integration_kernel_timings=config.profile_integration_kernel_timings,
            accumulator_w_max=config.accumulator_w_max,
        )

        self._integrator = BlockSparseESDFIntegrator(integrator_config)
        self._last_voxel_grid: Optional[VoxelGrid] = None

    @property
    def integrator(self) -> BlockSparseESDFIntegrator:
        """Access underlying integrator for advanced operations."""
        return self._integrator

    @property
    def tsdf(self) -> BlockSparseTSDF:
        """Access the BlockSparseTSDF storage."""
        return self._integrator.tsdf

    def integrate(
        self,
        *args,
        observation: Optional[Union[CameraObservation, LidarObservation]] = None,
        camera_observation: Optional[CameraObservation] = None,
        lidar_observation: Optional[LidarObservation] = None,
    ) -> None:
        """Integrate camera and/or LiDAR observations into the TSDF.

        Positional ``mapper.integrate(obs)`` calls remain supported. The
        ``observation=`` keyword is retained as a deprecated alias; prefer
        ``camera_observation=`` and/or ``lidar_observation=`` for new code.

        Args:
            observation: Deprecated keyword alias for one camera or LiDAR observation.
            camera_observation: Optional batched camera observation.
            lidar_observation: Optional batched LiDAR observation.
        """
        camera_observation, lidar_observation = self._normalize_integrate_observations(
            args,
            observation=observation,
            camera_observation=camera_observation,
            lidar_observation=lidar_observation,
        )
        self._integrator.integrate(
            camera_observation=camera_observation,
            lidar_observation=lidar_observation,
        )
        self._last_voxel_grid = None

    def _normalize_integrate_observations(
        self,
        args: tuple,
        *,
        observation: Optional[Union[CameraObservation, LidarObservation]],
        camera_observation: Optional[CameraObservation],
        lidar_observation: Optional[LidarObservation],
    ) -> tuple[Optional[CameraObservation], Optional[LidarObservation]]:
        if len(args) > 1:
            log_and_raise(
                f"integrate() takes at most one positional observation, got {len(args)}.",
                exception_type=TypeError,
            )
        if args:
            if observation is not None or camera_observation is not None or lidar_observation is not None:
                log_and_raise(
                    "Positional observation cannot be combined with observation=, "
                    "camera_observation=, or lidar_observation=.",
                    exception_type=TypeError,
                )
            observation = args[0]
        elif observation is not None:
            _deprecated_mapper_integrate_observation_alias()
            if camera_observation is not None or lidar_observation is not None:
                log_and_raise(
                    "observation= cannot be combined with camera_observation= "
                    "or lidar_observation=.",
                    exception_type=TypeError,
                )

        if observation is not None:
            if isinstance(observation, CameraObservation):
                camera_observation = observation
            elif isinstance(observation, LidarObservation):
                lidar_observation = observation
            else:
                log_and_raise(
                    "observation must be CameraObservation or LidarObservation, got "
                    f"{type(observation).__name__}.",
                    exception_type=TypeError,
                )

        if camera_observation is None and lidar_observation is None:
            log_and_raise(
                "integrate() requires a camera_observation, lidar_observation, "
                "or one positional observation.",
                exception_type=TypeError,
            )
        return camera_observation, lidar_observation

    def clear_region(self, bounds_min, bounds_max) -> int:
        """Clear dynamic map contents inside a conservative world-space AABB.

        Blocks remain allocated. The cached ESDF voxel grid is invalidated and
        must be recomputed with :meth:`compute_esdf`.
        """
        n_clear = self._integrator.clear_region(bounds_min, bounds_max)
        if n_clear > 0:
            self._last_voxel_grid = None
        return n_clear

    def clear_blocks(self, pool_indices) -> int:
        """Clear dynamic map contents for explicit block-pool indices."""
        n_clear = self._integrator.clear_blocks(pool_indices)
        if n_clear > 0:
            self._last_voxel_grid = None
        return n_clear

    def compute_esdf(
        self,
        esdf_origin: Optional[torch.Tensor] = None,
        esdf_voxel_size: Optional[float] = None,
    ) -> VoxelGrid:
        """Compute ESDF from current TSDF and return as VoxelGrid.

        Args:
            esdf_origin: Override ESDF origin (for sliding window).
            esdf_voxel_size: Override ESDF voxel size (meters).

        Returns:
            VoxelGrid containing the signed distance field for collision checking.
        """
        esdf_voxel_size_tensor = None
        if esdf_voxel_size is not None:
            esdf_voxel_size_tensor = torch.tensor(
                [esdf_voxel_size], device=self._device, dtype=torch.float32
            )
        self._integrator.compute_esdf(
            esdf_origin=esdf_origin, esdf_voxel_size=esdf_voxel_size_tensor
        )
        self._last_voxel_grid = self._integrator.get_voxel_grid()
        return self._last_voxel_grid

    def extract_mesh(
        self,
        refine_iterations: int = 2,
        surface_only: bool = True,
    ) -> Mesh:
        """Extract mesh using GPU marching cubes.

        Args:
            refine_iterations: Newton-Raphson iterations for vertex refinement.
            surface_only: Only extract mesh near surface.

        Returns:
            Mesh object with vertices, faces, and colors.
        """
        return self._integrator.extract_mesh(
            refine_iterations=refine_iterations,
            surface_only=surface_only,
        )

    def extract_occupied_voxels(
        self,
        surface_only: bool = False,
        sdf_threshold: Optional[float] = None,
    ) -> OccupiedVoxels:
        """Extract occupied voxel centers and per-voxel block indices.

        Args:
            surface_only: If True, only extract voxels near the TSDF
                surface. If False, include inside voxels too.
            sdf_threshold: Surface threshold. Defaults to the underlying
                TSDF integrator's voxel size.

        Returns:
            :class:`OccupiedVoxels` with voxel centers, per-voxel block
            pool indices, and a view over per-block storage.
        """
        return self._integrator.extract_occupied_voxels(
            surface_only=surface_only,
            sdf_threshold=sdf_threshold,
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
        """Extract voxels from the top-k blocks most similar (cosine) to
        ``feature_vector``. Requires :attr:`MapperCfg.feature_dim` > 0.

        ``minimum_score`` (cosine in the query feature space) drops
        matched blocks below the cut *after* the top-k selection, so the
        voxel-extraction kernel skips work proportional to filtered
        blocks. Pair a generous ``top_k`` with ``minimum_score`` to get
        all matches above a quality threshold.

        Returns a :class:`MatchedVoxels` carrying the extracted voxels
        plus parallel ``(K,)`` ``block_pool_idx`` / ``block_scores``
        tensors in descending score order. ``block_pool_idx`` can be
        passed directly to :meth:`clear_blocks` to drop the matched
        blocks from the map.

        ``feature_projector`` optionally maps stored block features into
        the query space before scoring. This keeps model-specific heads
        (e.g. RADIO -> SigLIP) outside the mapper while still using the
        mapper's all-allocated-block scoring and extraction path.
        """
        return self._integrator.extract_matching_feature_voxels(
            feature_vector=feature_vector,
            top_k=top_k,
            surface_only=surface_only,
            sdf_threshold=sdf_threshold,
            minimum_score=minimum_score,
            feature_projector=feature_projector,
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

    def reset(self) -> None:
        """Reset mapper for new scene."""
        self._integrator.reset()
        self._last_voxel_grid = None

    def save_blocks(self, file_path: Union[str, PathLike[str]]) -> None:
        """Save active mapper TSDF blocks to a compact checkpoint."""
        save_block_checkpoint(
            file_path,
            block_metadata=build_block_metadata(self.tsdf),
            blocks=self.tsdf.export_blocks(),
        )

    @classmethod
    def load_blocks(
        cls,
        file_path: Union[str, PathLike[str]],
        target_cfg: MapperCfg,
        import_weight: Optional[float] = None,
    ) -> "Mapper":
        """Construct a mapper from ``target_cfg`` and import block checkpoint data."""
        mapper = cls(target_cfg)
        mapper.import_blocks(file_path, import_weight=import_weight)
        return mapper

    def import_blocks(
        self,
        file_path: Union[str, PathLike[str]],
        import_weight: Optional[float] = None,
    ) -> int:
        """Import active TSDF blocks from disk into this empty mapper.

        Args:
            file_path: Checkpoint written by :meth:`save_blocks`.
            import_weight: Optional constant confidence to assign to imported
                dynamic TSDF/RGB/feature accumulators. ``None`` preserves saved
                weights.

        Returns:
            Number of imported active blocks.
        """
        checkpoint = load_block_checkpoint(file_path)
        block_metadata = checkpoint["block_metadata"]
        validate_block_metadata_for_target(block_metadata, self.tsdf)
        blocks = prepare_blocks_for_import(
            checkpoint["blocks"],
            block_metadata,
            import_weight=import_weight,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            block_empty_threshold=float(self.tsdf.kernels.block_empty_threshold),
        )
        num_blocks = self._integrator.import_blocks(blocks)
        self._last_voxel_grid = None
        return num_blocks

    def get_stats(
        self,
        scan_pool: bool = True,
        scan_hash: bool = False,
    ) -> dict:
        """Get mapper statistics.

        Args:
            scan_pool: Forwarded to the integrator's stats call. When
                True (default) returns the ground-truth ``active_blocks``
                and the ``holes`` invariant via an O(num_allocated) GPU
                reduction. When False, falls back to the cheap
                ``num_allocated - free_count`` approximation.
            scan_hash: Forwarded to the integrator's stats call. When
                True, adds hash table occupancy stats via an
                O(hash_capacity) reduction. Periodic use only.

        Returns:
            Dictionary with block usage, memory stats, frame count, and
            ``last_integration_kernel_timings_ms``. See
            :meth:`BlockSparseTSDF.get_stats` for the full block-pool key list.
        """
        return self._integrator.get_stats(
            scan_pool=scan_pool, scan_hash=scan_hash,
        )

    def memory_usage_mb(self) -> float:
        """Get total GPU memory usage in megabytes."""
        return self._integrator.memory_usage_mb()

    def update_static_obstacles(
        self,
        scene: SceneData,
        env_idx: int = 0,
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
        """
        self._integrator.update_static_obstacles(scene, env_idx=env_idx)

    # === Rendering (lazy-initialized) ===

    def _get_renderer(self):
        """Get or create the renderer instance."""
        from curobo._src.perception.mapper.renderer import BlockSparseTSDFRenderer

        if not hasattr(self, "_renderer"):
            self._renderer = BlockSparseTSDFRenderer(self._integrator)
        return self._renderer

    def render(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render depth and normals from current TSDF.

        Args:
            intrinsics: Camera intrinsics (3, 3) or (4,) as [fx, fy, cx, cy].
            pose: Camera-to-world transform as Pose.
            image_shape: (height, width) of output images.

        Returns:
            Tuple of (depth, normals, valid_mask).
        """
        return self._get_renderer().render(intrinsics, pose, image_shape)

    def render_color(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render depth, normals, and color from current TSDF.

        Args:
            intrinsics: Camera intrinsics (3, 3) or (4,) as [fx, fy, cx, cy].
            pose: Camera-to-world transform as Pose.
            image_shape: (height, width) of output images.

        Returns:
            Tuple of (depth, normals, color, valid_mask).
            - depth: (H, W) in meters
            - normals: (H, W, 3) world-frame normals
            - color: (H, W, 3) uint8 RGB
            - valid_mask: (H, W) bool
        """
        return self._get_renderer().render_color(intrinsics, pose, image_shape)

    def render_depth(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Render only depth image.

        Args:
            intrinsics: Camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (height, width) of output images.

        Returns:
            depth: (H, W) in meters.
        """
        return self._get_renderer().render_depth(intrinsics, pose, image_shape)

    def render_color_only(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Render only color image.

        Args:
            intrinsics: Camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (height, width) of output images.

        Returns:
            color: (H, W, 3) uint8 RGB.
        """
        return self._get_renderer().render_color_only(intrinsics, pose, image_shape)

    def render_shaded(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
        light_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        ambient: float = 1.0,
        use_color: bool = True,
    ) -> torch.Tensor:
        """Render Lambertian-shaded image.

        Args:
            intrinsics: Camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (height, width) of output images.
            light_direction: Light direction in camera frame.
            ambient: Ambient lighting factor (0-1).
            use_color: If True, use TSDF color. If False, use gray.

        Returns:
            shaded: (H, W, 3) uint8 RGB shaded image.
        """
        return self._get_renderer().render_shaded(
            intrinsics, pose, image_shape, light_direction, ambient, use_color
        )

    def render_depth_colormap(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Render depth as colormap for visualization.

        Args:
            intrinsics: Camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (height, width) of output images.

        Returns:
            colormap: (H, W, 3) uint8 RGB colormap.
        """
        return self._get_renderer().render_depth_colormap(
            intrinsics, pose, image_shape
        )

    def render_normal_colormap(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Render normals as colormap for visualization.

        Args:
            intrinsics: Camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (height, width) of output images.

        Returns:
            colormap: (H, W, 3) uint8 RGB colormap.
        """
        return self._get_renderer().render_normal_colormap(intrinsics, pose, image_shape)
