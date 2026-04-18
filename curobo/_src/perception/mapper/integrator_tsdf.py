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
        max_blocks=100_000,
    )

    for observation in observations:
        integrator.integrate(observation)

    vertices, triangles, colors = integrator.extract_mesh()
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from curobo._src.geom.data import SceneData
from curobo._src.geom.types import Mesh
from curobo._src.perception.mapper.block_allocation import calculate_tsdf_max_blocks
from curobo._src.perception.mapper.kernel.warp_types import BLOCK_SIZE
from curobo._src.perception.mapper.kernel.wp_decay import (
    decay_and_recycle,
    decay_frustum_aware_multi_camera,
)
from curobo._src.perception.mapper.kernel.wp_integrate_sort_filter import (
    SortFilterIntegrator,
)
from curobo._src.perception.mapper.kernel.wp_integrate_voxel_project import (
    VoxelProjectIntegrator,
)
from curobo._src.perception.mapper.kernel.wp_stamp_obstacles import (
    clear_static_channel,
    stamp_scene_obstacles,
)
from curobo._src.perception.mapper.kernel.wp_voxel_extraction import (
    extract_occupied_voxels_block_sparse,
    extract_surface_voxels_block_sparse,
)
from curobo._src.perception.mapper.mesh_extractor import (
    extract_mesh_block_sparse,
)
from curobo._src.perception.mapper.storage import (
    BlockSparseTSDF,
    BlockSparseTSDFCfg,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.torch_util import profile_class_methods


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
        block_size: Voxels per block edge (default 8).
        depth_minimum_distance: Minimum valid depth in meters.
        depth_maximum_distance: Maximum valid depth in meters.
        frustum_decay: Decay for in-view voxels (0.5=quick adapt, 1.0=no extra decay).
        time_decay: Decay for all voxels (1.0=persist, 0.99=slow fade).
        minimum_tsdf_weight: Minimum weight to consider a voxel as observed.
            Since weight = 1/depth² (clamped to [0.001, 2.0]), this can be interpreted
            as the number of observations at 1m depth. E.g., 0.5 = half an observation
            at 1m, or one observation at ~1.4m, or two observations at 2m.
        grid_shape: Optional (W, H, D) voxel grid dimensions for bounds checking.
            If provided, uses center-origin convention (origin at center of grid)
            matching the dense TSDFIntegrator. Voxels outside [0, W)×[0, H)×[0, D) are skipped.
            If None (default), no bounds checking is performed (unbounded, corner-origin).
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
    grid_shape: Optional[Tuple[int, int, int]] = None  # Optional bounds checking
    roughness: float = 3.0  # Geometric complexity multiplier
    image_height: Optional[int] = None  # For buffer pre-allocation
    image_width: Optional[int] = None  # For buffer pre-allocation
    enable_static: bool = False  # Enable static obstacle integration
    static_obstacle_color: Tuple[int, int, int] = (20, 20, 20)  # RGB for static obstacles
    integration_method: str = "voxel_project"  # "voxel_project" or "sort_filter"
    num_cameras: int = 1
    device: str = "cuda:0"

    @property
    def block_size(self) -> int:
        """Block size (voxels per edge). Fixed at BLOCK_SIZE=8."""
        return BLOCK_SIZE

    def __post_init__(self):
        if self.origin is None:
            self.origin = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.float32)
        if not isinstance(self.origin, torch.Tensor):
            self.origin = torch.tensor(self.origin, dtype=torch.float32)

        # Auto-calculate max_blocks if not provided
        if self.max_blocks is None:
            if self.grid_shape is not None:
                self.max_blocks = calculate_tsdf_max_blocks(
                    grid_shape=self.grid_shape,
                    voxel_size=self.voxel_size,
                    block_dim=BLOCK_SIZE,
                    truncation_dist=self.truncation_distance,
                    roughness=self.roughness,
                )
            else:
                # Default fallback for unbounded mode
                self.max_blocks = 100_000

        # Auto-calculate hash_capacity
        if self.hash_capacity is None:
            self.hash_capacity = int(math.ceil(self.max_blocks * 2.0))


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

    def __init__(self, config: BlockSparseTSDFIntegratorCfg):
        """Initialize the integrator.

        Args:
            config: Configuration dataclass.
        """
        self.config = config

        # Create block-sparse TSDF storage
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
        )
        self._tsdf = BlockSparseTSDF(tsdf_config)

        # Frame counter for periodic operations
        self._frame_count = 0

        # Create integration backend
        if config.integration_method == "voxel_project":
            self._integrator = VoxelProjectIntegrator(
                device=config.device,
            )
        elif config.integration_method == "sort_filter":
            self._integrator = SortFilterIntegrator(
                device=config.device,
            )
        else:
            log_and_raise(
                ValueError,
                f"Unknown integration_method: {config.integration_method}. "
                "Must be 'voxel_project' or 'sort_filter'.",
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
        )

        if self.config.time_decay < 1.0 or self.config.frustum_decay < 1.0:
            img_shape = (depth_images.shape[1], depth_images.shape[2])
            num_blocks = None
            if self.config.integration_method == "voxel_project":
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

        self._frame_count += 1

    def recycle_empty_blocks(self) -> int:
        """Recycle blocks that have decayed to empty.

        Call this periodically (e.g., every 100 frames) to reclaim memory
        from blocks that are no longer observed.

        Returns:
            Number of blocks recycled.
        """
        return decay_and_recycle(self._tsdf, 1.0)  # No additional decay

    def extract_mesh(
        self,
        level: float = 0.0,
        surface_only: bool = False,
        refine_iterations: int = 0,
    ) -> Mesh:
        """Extract mesh from the TSDF using marching cubes.

        Args:
            level: Isosurface level (typically 0.0).
            surface_only: If True, only extract mesh near the surface (|sdf| < truncation).
                Excludes triangles from regions deep inside the object where TSDF
                is clamped to -truncation_distance.
            refine_iterations: Number of Newton-Raphson iterations for vertex refinement.
                0 = no refinement (linear interpolation only). Higher values (2-5)
                produce smoother meshes at the cost of more computation.

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
            vertices=vertices, #.cpu().tolist(),
            faces=triangles, #.cpu().tolist(),  # Keep as (N, 3), not flattened
            vertex_colors=(colors.float() / 255.0),#.cpu().tolist(),
            vertex_normals=normals,#.cpu().tolist(),
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

    def get_stats(self) -> Dict[str, float]:
        """Get integrator statistics.

        Returns:
            Dictionary with block usage and allocation stats.
        """
        stats = self._tsdf.get_stats()
        stats["frame_count"] = self._frame_count
        stats["memory_mb"] = self._tsdf.memory_usage_mb()
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract occupied voxel centers and colors.

        Matches dense TSDFIntegrator.extract_occupied_voxels behavior:
        - surface_only=True: extract voxels near zero-crossing (|SDF| < sdf_threshold)
        - surface_only=False: extract surface + inside voxels (SDF <= 0)

        Args:
            surface_only: If True, extract only surface voxels (|SDF| < sdf_threshold).
                          If False (default), include inside voxels too (SDF <= 0).
            sdf_threshold: Threshold for surface_only=True. Defaults to voxel_size.

        Returns:
            Tuple of (centers, colors):
                - centers: (N, 3) float32 voxel world positions
                - colors: (N, 3) uint8 RGB colors
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
            log_info(
                f"[update_static_obstacles] num_allocated={tsdf.data.num_allocated.item()}"
            )

        # Step 1: Clear static channel to +inf
        clear_static_channel(tsdf.data)

        # Step 2: Recycle blocks that are now empty (no dynamic AND no static data)
        decay_and_recycle(tsdf, decay_factor=1.0)

        # Step 3: Stamp all obstacle types (cuboids, meshes, voxels) generically
        stamp_scene_obstacles(tsdf, scene, env_idx, self.config.static_obstacle_color, debug=debug)
