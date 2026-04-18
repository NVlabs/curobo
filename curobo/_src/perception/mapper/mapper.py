# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Mapper - Unified volumetric mapping interface.

This module provides the Mapper class, a high-level facade for volumetric
mapping using block-sparse TSDF storage with ESDF generation.

Example:
    from curobo._src.perception.mapper import Mapper, MapperCfg

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
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from curobo._src.geom.data import SceneData
from curobo._src.geom.types import Mesh, VoxelGrid
from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegrator,
    BlockSparseESDFIntegratorCfg,
)
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.perception.mapper.pose_refiner import (
    BlockSparseRaycastPoseRefiner,
    BlockSparseRaycastRefinerCfg,
)
from curobo._src.perception.mapper.storage import (
    BlockSparseTSDF,
)
from curobo._src.types.pose import Pose

if TYPE_CHECKING:
    from curobo._src.types.camera import CameraObservation


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
            block_size=config.block_size,
            depth_minimum_distance=config.depth_minimum_distance,
            depth_maximum_distance=config.depth_maximum_distance,
            frustum_decay=config.frustum_decay_factor,
            time_decay=config.decay_factor,
            minimum_tsdf_weight=config.minimum_tsdf_weight,
            grid_shape=config.grid_shape,
            enable_static=config.enable_static,
            static_obstacle_color=config.static_obstacle_color,
            integration_method=config.integration_method,
            seeding_method=config.seeding_method,
            edt_solver=config.edt_solver,
            num_cameras=config.num_cameras,
            device=config.device,
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
        observation: CameraObservation,
    ) -> None:
        """Integrate batched depth observation into TSDF.

        The observation must have a leading camera dimension matching
        ``config.num_cameras``. See ``BlockSparseTSDFIntegrator.integrate``
        for the expected tensor shapes.

        Args:
            observation: Batched camera observation.
        """
        self._integrator.integrate(observation)

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
        refine_iterations: int = 5,
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

    def reset(self) -> None:
        """Reset mapper for new scene."""
        self._integrator.reset()
        self._last_voxel_grid = None

    def get_stats(self) -> dict:
        """Get mapper statistics.

        Returns:
            Dictionary with block usage, memory stats, and frame count.
        """
        return self._integrator.get_stats()

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

    # === Pose Refinement (lazy-initialized) ===

    def refine_pose(
        self,
        depth_image: torch.Tensor,
        intrinsics: torch.Tensor,
        initial_pose: "Pose",
        max_iterations: int = 100,
        minimum_tsdf_weight: float = 0.01,
        n_points: int = 10000,
    ) -> Tuple["Pose", float, int]:
        """Refine camera pose using ICP with TSDF raycasting.

        Args:
            depth_image: Input depth image (H, W).
            intrinsics: Camera intrinsics (3, 3).
            initial_pose: Initial pose estimate.
            max_iterations: Maximum ICP iterations.

        Returns:
            Tuple of (refined_pose, final_error, iterations_used).
        """
        if not hasattr(self, "_pose_refiner"):
            refiner_config = BlockSparseRaycastRefinerCfg(
                n_points=n_points,
                minimum_tsdf_weight=minimum_tsdf_weight,
                depth_minimum_distance=self._integrator._tsdf_integrator.config.depth_minimum_distance,
                depth_maximum_distance=self._integrator._tsdf_integrator.config.depth_maximum_distance,
            )
            self._pose_refiner = BlockSparseRaycastPoseRefiner(
                self._integrator,
                config=refiner_config,
            )

        return self._pose_refiner.refine_pose(
            depth=depth_image,
            intrinsics=intrinsics,
            estimated_pose=initial_pose,
        )

