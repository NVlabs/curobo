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
from collections.abc import Sequence
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
    _validate_block_size,
    _validate_color_grid_size,
    _validate_feature_block_grid_size,
    _validate_feature_channels_per_thread,
    _validate_feature_grid_shape,
    _validate_feature_integration_kernel,
    _validate_instance_grid_size,
    _validate_lidar_config,
    resolve_feature_integration_kernel,
    validate_grid_shape_for_hash_layout,
)
from curobo._src.perception.mapper.extractor_points import (
    OccupiedPoints,
    SurfacePointWorkspace,
    validate_point_query,
)
from curobo._src.perception.mapper.filter_semantic import (
    SemanticFilter,
    _SemanticFilterWorkspace,
)
from curobo._src.perception.mapper.kernel.builder.builder_block_sparse_kernel import (
    BlockSparseKernels,
    make_block_sparse_kernels,
)
from curobo._src.perception.mapper.kernel.warp_types import PreparedSemanticFilterWarp
from curobo._src.perception.mapper.kernel.wp_decay import (
    decay_and_recycle,
    decay_frustum_aware_multi_sensor,
)
from curobo._src.perception.mapper.kernel.wp_integrate_camera_project import (
    CameraProjectIntegrator,
)
from curobo._src.perception.mapper.kernel.wp_integrate_lidar_project import (
    LidarProjectIntegrator,
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
from curobo._src.perception.mapper.mesh_extractor import extract_mesh_block_sparse
from curobo._src.perception.mapper.projector_texture import (
    ProjectiveTextureProjector,
    ProjectiveTextureProjectorCfg,
)
from curobo._src.perception.mapper.renderer import BlockSparseTSDFRenderer
from curobo._src.perception.mapper.storage import (
    BlockSparseTSDF,
    BlockSparseTSDFCfg,
    MatchedVoxels,
    OccupiedVoxels,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.lidar import LidarObservation
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
    #: Number of RGB cameras used by projective texture export. Defaults to
    #: ``num_cameras`` when unset.
    texture_num_cameras: Optional[int] = None
    #: RGB texture camera image height in pixels. Defaults to ``image_height``.
    texture_camera_image_height: Optional[int] = None
    #: RGB texture camera image width in pixels. Defaults to ``image_width``.
    texture_camera_image_width: Optional[int] = None
    lidar_num_sensors: int = 0
    lidar_image_height: Optional[int] = None
    lidar_image_width: Optional[int] = None
    lidar_feature_grid_height: Optional[int] = None
    lidar_feature_grid_width: Optional[int] = None
    lidar_linear_interpolation_max_allowable_difference_vox: float = 2.0
    lidar_nearest_interpolation_max_allowable_dist_to_ray_vox: float = 0.5
    max_visible_blocks_per_lidar_integration: Optional[int] = None
    max_support_pixels_per_block_lidar: int = 32
    enable_static: bool = False  # Enable static obstacle integration
    static_obstacle_color: Tuple[int, int, int] = (20, 20, 20)  # RGB for static obstacles
    seeding_method: str = "gather"  # "gather" or "scatter"; selects ESDF seed kernel variant
    num_cameras: int = 1
    device: str = "cuda:0"
    #: Voxels per block edge. Supported values are 1 or powers of 2 in [2, 32].
    block_size: int = 8
    #: Per-block feature channel dimensionality. 0 disables features.
    feature_dim: int = 0
    #: Number of feature control points per block edge. Independent from
    #: ``feature_grid_height`` and ``feature_grid_width``, which describe the
    #: incoming 2D camera feature image.
    feature_block_grid_size: int = 1
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
    #: Number of RGBW control points per block edge. This is a construction-time
    #: kernel specialization value. Total controls per block are
    #: ``color_grid_size ** 3``.
    color_grid_size: int = 1
    #: Number of categorical instance cells per block edge. ``0`` disables
    #: packed instance ownership storage.
    instance_grid_size: int = 0
    #: Feature integration launch policy: ``"auto"``, ``"grouped"``, or
    #: ``"tiled"``. Resolved to a low-level tiled bool at construction time.
    feature_integration_kernel: str = "auto"
    #: Record per-kernel integration timings into
    #: ``get_stats()["last_integration_kernel_timings_ms"]``.
    profile_integration_kernel_timings: bool = False
    #: Upper bound on each RGB or feature-grid node's weight after each frame.
    #: Caps the fp16 weighted-sum magnitudes in ``block_grid_rgb`` and
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
        _validate_lidar_config(
            self.lidar_num_sensors,
            self.lidar_image_height,
            self.lidar_image_width,
            self.lidar_feature_grid_height,
            self.lidar_feature_grid_width,
            self.feature_dim,
            self.lidar_linear_interpolation_max_allowable_difference_vox,
            self.lidar_nearest_interpolation_max_allowable_dist_to_ray_vox,
            self.max_support_pixels_per_block_lidar,
        )
        if self.lidar_num_sensors > 0:
            if self.max_visible_blocks_per_lidar_integration is None:
                self.max_visible_blocks_per_lidar_integration = self.max_blocks
            if (
                self.max_visible_blocks_per_lidar_integration <= 0
                or self.max_visible_blocks_per_lidar_integration > self.max_blocks
            ):
                log_and_raise(
                    "max_visible_blocks_per_lidar_integration must satisfy "
                    f"0 < C <= max_blocks ({self.max_blocks}), got "
                    f"{self.max_visible_blocks_per_lidar_integration}."
                )
        elif self.max_visible_blocks_per_lidar_integration is not None:
            log_and_raise(
                "max_visible_blocks_per_lidar_integration requires lidar_num_sensors > 0."
            )

        if self.image_height is None or self.image_width is None:
            log_and_raise(
                "BlockSparseTSDFIntegratorCfg requires image_height and "
                "image_width to pre-allocate the camera projective scratch "
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
        if (self.texture_camera_image_height is None) != (self.texture_camera_image_width is None):
            log_and_raise(
                "texture_camera_image_height and texture_camera_image_width must be "
                "specified together."
            )
        if self.texture_num_cameras is None:
            self.texture_num_cameras = self.num_cameras
        if self.texture_camera_image_height is None:
            self.texture_camera_image_height = self.image_height
            self.texture_camera_image_width = self.image_width
        if self.texture_num_cameras <= 0:
            log_and_raise(
                "texture_num_cameras must be positive, got "
                f"texture_num_cameras={self.texture_num_cameras}."
            )
        if self.texture_camera_image_height <= 0 or self.texture_camera_image_width <= 0:
            log_and_raise(
                "texture_camera_image_height and texture_camera_image_width must be "
                "positive, got "
                f"{self.texture_camera_image_height}x{self.texture_camera_image_width}."
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
                f"max_feature_tile_channels={self.max_feature_tile_channels}."
            )
        if self.max_support_pixels_per_block_camera <= 0:
            log_and_raise(
                "max_support_pixels_per_block_camera must be positive, got "
                f"max_support_pixels_per_block_camera="
                f"{self.max_support_pixels_per_block_camera}."
            )
        _validate_color_grid_size(self.color_grid_size, self.block_size)
        _validate_feature_block_grid_size(
            self.feature_block_grid_size,
            self.block_size,
        )
        _validate_instance_grid_size(self.instance_grid_size, self.block_size)
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
        use_tiled_lidar_feature_kernel = resolve_feature_integration_kernel(
            config.feature_integration_kernel,
            config.feature_dim,
            config.max_support_pixels_per_block_lidar,
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
            feature_block_grid_size=config.feature_block_grid_size,
            feature_grid_height=config.feature_grid_height,
            feature_grid_width=config.feature_grid_width,
            feature_channels_per_thread=config.feature_channels_per_thread,
            max_feature_tile_channels=config.max_feature_tile_channels,
            max_support_pixels_per_block_camera=config.max_support_pixels_per_block_camera,
            color_grid_size=config.color_grid_size,
            instance_grid_size=config.instance_grid_size,
            accumulator_w_max=config.accumulator_w_max,
        )
        if kernels is None:
            kernels = make_block_sparse_kernels(config)
        self._tsdf = BlockSparseTSDF(tsdf_config, kernels=kernels)
        texture_renderer = BlockSparseTSDFRenderer(self)
        self._texture_projector = ProjectiveTextureProjector(
            tsdf=self._tsdf,
            renderer=texture_renderer,
            config=ProjectiveTextureProjectorCfg(
                texture_num_cameras=int(config.texture_num_cameras),
                image_height=int(config.texture_camera_image_height),
                image_width=int(config.texture_camera_image_width),
                depth_minimum_distance=float(config.depth_minimum_distance),
                depth_maximum_distance=float(config.depth_maximum_distance),
                voxel_size=float(config.voxel_size),
            ),
        )

        # Frame counter for periodic operations
        self._frame_count = 0

        self._camera_integrator = CameraProjectIntegrator(
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
        if config.lidar_num_sensors > 0:
            self._lidar_integrator = LidarProjectIntegrator(
                lidar_num_sensors=config.lidar_num_sensors,
                lidar_image_height=config.lidar_image_height,
                lidar_image_width=config.lidar_image_width,
                voxel_size=config.voxel_size,
                block_size=config.block_size,
                truncation_distance=config.truncation_distance,
                max_blocks=self._tsdf.config.max_blocks,
                max_visible_blocks_per_lidar_integration=(
                    config.max_visible_blocks_per_lidar_integration
                ),
                max_support_pixels_per_block_lidar=(config.max_support_pixels_per_block_lidar),
                device=config.device,
                feature_channels_per_thread=config.feature_channels_per_thread,
                use_tiled_feature_kernel=use_tiled_lidar_feature_kernel,
                lidar_feature_grid_shape=(
                    (config.lidar_feature_grid_height, config.lidar_feature_grid_width)
                    if config.lidar_feature_grid_height is not None
                    else None
                ),
                linear_interpolation_max_allowable_difference_vox=(
                    config.lidar_linear_interpolation_max_allowable_difference_vox
                ),
                nearest_interpolation_max_allowable_dist_to_ray_vox=(
                    config.lidar_nearest_interpolation_max_allowable_dist_to_ray_vox
                ),
                profile_kernel_timings=config.profile_integration_kernel_timings,
            )
        else:
            self._lidar_integrator = None
        self._semantic_revision = 0
        self._semantic_filter_workspace = _SemanticFilterWorkspace()
        self._point_workspace: Optional[SurfacePointWorkspace] = None

    @property
    def tsdf(self) -> BlockSparseTSDF:
        """Get the underlying BlockSparseTSDF instance."""
        return self._tsdf

    def _mark_semantic_data_changed(self) -> None:
        """Invalidate prepared semantic masks after a successful map mutation."""
        self._semantic_revision += 1

    def _prepare_semantic_filter(
        self,
        semantic_filter: SemanticFilter,
    ) -> PreparedSemanticFilterWarp:
        """Compile ``semantic_filter`` against the current fused feature grid."""
        return self._semantic_filter_workspace.prepare(
            semantic_filter,
            self._tsdf,
            self._semantic_revision,
        )

    def reset(self):
        """Reset the integrator to initial state."""
        self._tsdf.reset()
        self._frame_count = 0
        self._mark_semantic_data_changed()

    def clear_instance_slot(self, instance_slot: int) -> None:
        """Clear one retired compact instance slot from all allocated blocks."""
        self._tsdf.clear_instance_slot(instance_slot)

    def import_blocks(self, blocks: Dict[str, torch.Tensor]) -> int:
        """Import compact block payloads into the underlying TSDF storage.

        Returns:
            Number of imported active blocks.
        """
        self._tsdf.import_blocks(blocks)
        num_blocks = int(self._tsdf.data.num_allocated.item())
        self._frame_count = 1 if num_blocks > 0 else 0
        self._mark_semantic_data_changed()
        return num_blocks

    def integrate(
        self,
        observation: Optional[CameraObservation | LidarObservation] = None,
        *,
        camera_observation: Optional[CameraObservation] = None,
        lidar_observation: Optional[LidarObservation] = None,
    ):
        """Integrate camera and/or LiDAR observations into the TSDF."""
        if observation is not None:
            if camera_observation is not None or lidar_observation is not None:
                log_and_raise(
                    "observation= cannot be combined with camera_observation= "
                    "or lidar_observation=."
                )
            if isinstance(observation, CameraObservation):
                camera_observation = observation
            elif isinstance(observation, LidarObservation):
                lidar_observation = observation
            else:
                log_and_raise(
                    "observation must be CameraObservation or LidarObservation, got "
                    f"{type(observation).__name__}."
                )

        if camera_observation is None and lidar_observation is None:
            log_and_raise(
                "integrate() requires camera_observation, lidar_observation, or observation."
            )

        has_lidar = lidar_observation is not None
        if has_lidar:
            self._validate_lidar_observation(lidar_observation)
            self._apply_frame_decay(camera_observation, lidar_observation)
        if camera_observation is not None:
            self._integrate_camera_frame(
                camera_observation,
                advance_frame=False,
                apply_decay=not has_lidar,
            )
        if has_lidar:
            self._integrate_lidar_frame(lidar_observation, advance_frame=False)
        self._frame_count += 1

    def _integrate_camera_frame(
        self,
        observation: CameraObservation,
        *,
        advance_frame: bool = True,
        apply_decay: bool = True,
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
        - ``instance_image``: optional ``(num_cameras, H, W)`` uint8 compact slots

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

        if apply_decay:
            self._apply_frame_decay(observation, None)

        self._camera_integrator.integrate(
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
            instance_image=observation.instance_image,
        )
        self._mark_semantic_data_changed()

        if advance_frame:
            self._frame_count += 1

    def _apply_frame_decay(
        self,
        camera_observation: Optional[CameraObservation],
        lidar_observation: Optional[LidarObservation],
    ) -> None:
        """Decay once for the union of all frusta in the current frame."""
        if self._frame_count <= 0:
            return
        if self.config.time_decay >= 1.0 and self.config.frustum_decay >= 1.0:
            return

        camera_intrinsics = None
        camera_positions = None
        camera_quaternions = None
        camera_img_shape = None
        if camera_observation is not None:
            depth_images = camera_observation.depth_image
            if depth_images.ndim != 3:
                log_and_raise(
                    "depth_image must be (num_cameras, H, W), got "
                    f"shape {tuple(depth_images.shape)}."
                )
            n_cameras = int(depth_images.shape[0])
            if n_cameras != self.config.num_cameras:
                log_and_raise(
                    f"Expected num_cameras={self.config.num_cameras}, "
                    f"got depth_image batch dim {n_cameras}."
                )
            camera_intrinsics = camera_observation.intrinsics
            camera_positions = camera_observation.pose.position.view(n_cameras, 3)
            camera_quaternions = camera_observation.pose.quaternion.view(n_cameras, 4)
            camera_img_shape = (int(depth_images.shape[1]), int(depth_images.shape[2]))

        lidar_positions = None
        lidar_quaternions = None
        lidar_valid_range_m = None
        lidar_elevation_range_rad = None
        lidar_img_shape = None
        if lidar_observation is not None:
            range_images = lidar_observation.range_image
            n_lidars = int(range_images.shape[0])
            lidar_positions = lidar_observation.pose.position.view(n_lidars, 3)
            lidar_quaternions = lidar_observation.pose.quaternion.view(n_lidars, 4)
            lidar_valid_range_m = lidar_observation.valid_range_m
            lidar_elevation_range_rad = lidar_observation.elevation_range_rad
            lidar_img_shape = (int(range_images.shape[1]), int(range_images.shape[2]))

        num_blocks = int(self._tsdf.data.num_allocated.item())
        decay_frustum_aware_multi_sensor(
            self._tsdf,
            camera_intrinsics=camera_intrinsics,
            camera_positions=camera_positions,
            camera_quaternions=camera_quaternions,
            camera_img_shape=camera_img_shape,
            camera_depth_minimum_distance=self.config.depth_minimum_distance,
            camera_depth_maximum_distance=self.config.depth_maximum_distance,
            lidar_positions=lidar_positions,
            lidar_quaternions=lidar_quaternions,
            lidar_valid_range_m=lidar_valid_range_m,
            lidar_elevation_range_rad=lidar_elevation_range_rad,
            lidar_img_shape=lidar_img_shape,
            time_decay=self.config.time_decay,
            frustum_decay=self.config.frustum_decay,
            num_blocks=num_blocks,
        )

    def _integrate_lidar_frame(
        self,
        observation: LidarObservation,
        *,
        advance_frame: bool = True,
    ):
        """Integrate a batched LiDAR observation into the TSDF."""
        if self.config.lidar_num_sensors <= 0:
            log_and_raise(
                "LidarObservation was provided but lidar_num_sensors == 0. "
                "Enable LiDAR integration in the mapper config."
            )
        if self._lidar_integrator is None:
            log_and_raise("LiDAR integrator is not initialized.")
        self._validate_lidar_observation(observation)
        n_lidars = int(observation.range_image.shape[0])
        positions = observation.pose.position.view(n_lidars, 3)
        quaternions = observation.pose.quaternion.view(n_lidars, 4)

        self._lidar_integrator.integrate(
            self._tsdf,
            observation.range_image,
            observation.rgb_image,
            positions,
            quaternions,
            observation.valid_range_m,
            observation.elevation_range_rad,
            feature_grid=observation.feature_grid,
        )
        self._mark_semantic_data_changed()
        if advance_frame:
            self._frame_count += 1

    def _validate_lidar_observation(self, observation: LidarObservation) -> None:
        range_image = observation.range_image
        if range_image is None:
            log_and_raise("LidarObservation.range_image is required.")
        if range_image.ndim != 3:
            log_and_raise(
                f"range_image must be (num_lidars, H, W), got shape {tuple(range_image.shape)}."
            )
        n_lidars = int(range_image.shape[0])
        if n_lidars != self.config.lidar_num_sensors:
            log_and_raise(
                f"Expected lidar_num_sensors={self.config.lidar_num_sensors}, "
                f"got range_image batch dim {n_lidars}."
            )
        expected_shape = (
            self.config.lidar_num_sensors,
            self.config.lidar_image_height,
            self.config.lidar_image_width,
        )
        if tuple(range_image.shape) != expected_shape:
            log_and_raise(
                f"range_image shape mismatch: expected {expected_shape}, "
                f"got {tuple(range_image.shape)}."
            )
        if range_image.dtype != torch.float32:
            log_and_raise(f"range_image dtype must be torch.float32, got {range_image.dtype}.")

        rgb_image = observation.rgb_image
        if rgb_image is None:
            log_and_raise("LidarObservation.rgb_image is required.")
        expected_rgb_shape = expected_shape + (3,)
        if tuple(rgb_image.shape) != expected_rgb_shape:
            log_and_raise(
                f"rgb_image shape mismatch: expected {expected_rgb_shape}, "
                f"got {tuple(rgb_image.shape)}."
            )
        if rgb_image.dtype != torch.uint8:
            log_and_raise(f"rgb_image dtype must be torch.uint8, got {rgb_image.dtype}.")
        if rgb_image.device != range_image.device:
            log_and_raise(
                f"rgb_image device {rgb_image.device} does not match "
                f"range_image device {range_image.device}."
            )

        if observation.pose is None:
            log_and_raise("LidarObservation.pose is required.")
        positions = observation.pose.position.view(n_lidars, 3)
        quaternions = observation.pose.quaternion.view(n_lidars, 4)
        if positions.device != range_image.device or quaternions.device != range_image.device:
            log_and_raise("LidarObservation.pose tensors must be on the range_image device.")

        valid_range_m = observation.valid_range_m
        if valid_range_m is None:
            log_and_raise("LidarObservation.valid_range_m is required.")
        if tuple(valid_range_m.shape) != (n_lidars, 2):
            log_and_raise(
                f"valid_range_m must be (num_lidars, 2), got shape {tuple(valid_range_m.shape)}."
            )
        if valid_range_m.dtype != torch.float32:
            log_and_raise(f"valid_range_m dtype must be torch.float32, got {valid_range_m.dtype}.")
        if valid_range_m.device != range_image.device:
            log_and_raise(
                f"valid_range_m device {valid_range_m.device} does not match "
                f"range_image device {range_image.device}."
            )

        elevation_range_rad = observation.elevation_range_rad
        if elevation_range_rad is None:
            log_and_raise("LidarObservation.elevation_range_rad is required.")
        if tuple(elevation_range_rad.shape) != (n_lidars, 2):
            log_and_raise(
                "elevation_range_rad must be (num_lidars, 2), got "
                f"shape {tuple(elevation_range_rad.shape)}."
            )
        if elevation_range_rad.dtype != torch.float32:
            log_and_raise(
                "elevation_range_rad dtype must be torch.float32, got "
                f"{elevation_range_rad.dtype}."
            )
        if elevation_range_rad.device != range_image.device:
            log_and_raise(
                f"elevation_range_rad device {elevation_range_rad.device} does not "
                f"match range_image device {range_image.device}."
            )
        if self.config.lidar_image_height == 1 and not torch.allclose(
            elevation_range_rad[:, 0], elevation_range_rad[:, 1]
        ):
            log_and_raise(
                "Planar LiDAR with lidar_image_height == 1 requires "
                "elevation_range_rad[:, 0] == elevation_range_rad[:, 1]."
            )

        feature_grid = observation.feature_grid
        if feature_grid is not None and not self._tsdf.data.has_features:
            log_and_raise(
                "feature_grid was provided but feature_dim == 0; enable features via "
                "MapperCfg.feature_dim or BlockSparseTSDFIntegratorCfg.feature_dim."
            )
        if self._tsdf.data.has_features and feature_grid is not None:
            expected_feature_shape = (
                n_lidars,
                self.config.lidar_feature_grid_height,
                self.config.lidar_feature_grid_width,
                self._tsdf.data.feature_dim,
            )
            if tuple(feature_grid.shape) != expected_feature_shape:
                log_and_raise(
                    f"feature_grid shape mismatch: expected {expected_feature_shape}, "
                    f"got {tuple(feature_grid.shape)}."
                )
            if feature_grid.dtype != torch.float16:
                log_and_raise(
                    f"feature_grid dtype must be torch.float16, got {feature_grid.dtype}."
                )
            if feature_grid.device != range_image.device:
                log_and_raise(
                    f"feature_grid device {feature_grid.device} does not match "
                    f"range_image device {range_image.device}."
                )
            if feature_grid.stride(-1) != 1:
                log_and_raise(
                    "feature_grid must be channels-last with stride 1 on the channel dim."
                )

    def recycle_empty_blocks(self) -> int:
        """Recycle blocks that have decayed to empty.

        Call this periodically (e.g., every 100 frames) to reclaim memory
        from blocks that are no longer observed.

        Returns:
            Number of blocks recycled.
        """
        recycled = decay_and_recycle(self._tsdf, 1.0)  # No additional decay
        self._mark_semantic_data_changed()
        return recycled

    def clear_region(self, bounds_min, bounds_max) -> int:
        """Clear dynamic map contents for allocated blocks intersecting an AABB.

        Args:
            bounds_min: World-space lower AABB corner, shape ``(3,)``.
            bounds_max: World-space upper AABB corner, shape ``(3,)``.

        Returns:
            Number of allocated blocks cleared. Blocks remain allocated; only
            dynamic TSDF/RGB, block sums, and per-block features are reset.
        """
        cleared = self._camera_integrator.clear_region(self._tsdf, bounds_min, bounds_max)
        self._mark_semantic_data_changed()
        return cleared

    def clear_blocks(self, pool_indices) -> int:
        """Clear dynamic map contents for explicit block-pool indices.

        Args:
            pool_indices: Tensor/list of allocated ``pool_idx`` values.

        Returns:
            Number of blocks cleared. Blocks remain allocated; only dynamic
            TSDF/RGB, block sums, and per-block features are reset.
        """
        cleared = self._camera_integrator.clear_blocks(self._tsdf, pool_indices)
        self._mark_semantic_data_changed()
        return cleared

    def extract_mesh(
        self,
        refine_iterations: int = 0,
        surface_only: bool = False,
        level: float = 0.0,
        semantic_filter: Optional[SemanticFilter] = None,
    ) -> Mesh:
        """Extract a triangle-soup mesh from the TSDF using marching cubes.

        Args:
            refine_iterations: Number of Newton-Raphson iterations for vertex refinement.
                0 = no refinement (linear interpolation only). Higher values (2-5)
                produce smoother meshes at the cost of more computation.
            surface_only: If True, only extract mesh near the surface (|sdf| < truncation).
                Excludes triangles from regions deep inside the object where TSDF
                is clamped to -truncation_distance.
            level: Isosurface level (typically 0.0).
            semantic_filter: Optional fused-feature filter applied before mesh generation.

        Returns:
            Mesh object with one unique vertex for each triangle corner.
        """
        prepared_semantic_filter = (
            self._prepare_semantic_filter(semantic_filter) if semantic_filter is not None else None
        )
        vertices, normals, colors = extract_mesh_block_sparse(
            self._tsdf,
            level=level,
            surface_only=surface_only,
            refine_iterations=refine_iterations,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            semantic_filter=prepared_semantic_filter,
        )
        triangles = torch.arange(
            vertices.shape[0], dtype=torch.int32, device=vertices.device
        ).view(-1, 3)

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
        semantic_filter: Optional[SemanticFilter] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a triangle-soup mesh as raw tensors.

        Args:
            level: Isosurface level (typically 0.0).
            surface_only: If True, only extract mesh near the surface (|sdf| < truncation).
                Excludes triangles from regions deep inside the object where TSDF
                is clamped to -truncation_distance.
            refine_iterations: Number of Newton-Raphson iterations for vertex refinement.
                0 = no refinement (linear interpolation only). Higher values (2-5)
                produce smoother meshes at the cost of more computation.
            semantic_filter: Optional fused-feature filter applied before mesh generation.

        Returns:
            Tuple of (vertices, triangles, normals, colors):
                - vertices: (3 * M, 3) float32
                - triangles: (M, 3) int32 identity indices into ``vertices``
                - normals: (N, 3) float32
                - colors: (N, 3) uint8
        """
        prepared_semantic_filter = (
            self._prepare_semantic_filter(semantic_filter) if semantic_filter is not None else None
        )
        vertices, normals, colors = extract_mesh_block_sparse(
            self._tsdf,
            level=level,
            surface_only=surface_only,
            refine_iterations=refine_iterations,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            semantic_filter=prepared_semantic_filter,
        )
        triangles = torch.arange(
            vertices.shape[0], dtype=torch.int32, device=vertices.device
        ).view(-1, 3)
        return vertices, triangles, normals, colors

    def extract_textured_mesh(
        self,
        texture_observations: CameraObservation | Sequence[CameraObservation],
        refine_iterations: int = 0,
        surface_only: bool = True,
        level: float = 0.0,
        camera_min_distance: Optional[float] = None,
        camera_max_distance: Optional[float] = None,
        texture_depth_tolerance_m: Optional[float] = None,
        semantic_filter: Optional[SemanticFilter] = None,
    ) -> Mesh:
        """Extract a mesh with visibility-tested RGB texture.

        The TSDF geometry can come from camera or LiDAR integration. Texture
        projection consumes RGB images and aligned visibility depth. When an
        observation omits depth, visibility depth is rendered from this TSDF.

        Args:
            texture_observations: Camera observation or sequence of camera
                observations containing ``rgb_image``, ``intrinsics``, and
                ``pose``. RGB tensors must be ``uint8``. Optional depth tensors
                must be float32 camera-frame z in meters.
            refine_iterations: Newton-Raphson iterations for vertex refinement.
            surface_only: Only extract mesh near surface.
            level: Isosurface level.
            camera_min_distance: Minimum positive camera-frame Z distance used
                for projection. Defaults to the mapper depth minimum.
            camera_max_distance: Maximum camera-frame Z distance used for
                projection. Defaults to the mapper depth maximum.
            texture_depth_tolerance_m: Absolute visibility-depth tolerance. If
                None, each sample uses ``max(2 * voxel_size, 0.01 * projected_z)``.
            semantic_filter: Optional fused-feature filter applied to mesh geometry.

        Returns:
            Mesh with triangle-soup geometry, vertex colors, per-vertex UVs,
            and an RGB texture atlas. Triangles that cannot be projected into
            any RGB observation receive fallback UV patches filled from voxel
            colors.
        """
        projection = self._texture_projector.prepare_mesh_projection(
            texture_observations,
            texture_depth_tolerance_m=texture_depth_tolerance_m,
        )
        vertices, triangles, normals, colors = self.extract_mesh_tensors(
            level=level,
            surface_only=surface_only,
            refine_iterations=refine_iterations,
            semantic_filter=semantic_filter,
        )
        return self._texture_projector.project_mesh(
            vertices,
            triangles,
            normals,
            colors,
            projection,
            camera_min_distance=camera_min_distance,
            camera_max_distance=camera_max_distance,
        )

    @staticmethod
    def _validate_subvoxel_factor(subvoxel_factor: int) -> int:
        """Return a positive integer subvoxel factor or raise."""
        try:
            factor = int(subvoxel_factor)
        except (TypeError, ValueError):
            log_and_raise(f"subvoxel_factor must be a positive integer, got {subvoxel_factor}.")
        if isinstance(subvoxel_factor, bool) or factor != subvoxel_factor:
            log_and_raise(f"subvoxel_factor must be a positive integer, got {subvoxel_factor}.")
        if factor <= 0:
            log_and_raise(f"subvoxel_factor must be positive, got {subvoxel_factor}.")
        return factor

    @staticmethod
    def _validate_max_points(max_points: Optional[int]) -> Optional[int]:
        """Return a positive integer point cap or ``None``."""
        if max_points is None:
            return None
        try:
            point_cap = int(max_points)
        except (TypeError, ValueError):
            log_and_raise(f"max_points must be a positive integer or None, got {max_points}.")
        if isinstance(max_points, bool) or point_cap != max_points:
            log_and_raise(f"max_points must be a positive integer or None, got {max_points}.")
        if point_cap <= 0:
            log_and_raise(f"max_points must be positive, got {max_points}.")
        return point_cap

    def _limit_voxels_for_point_cap(
        self,
        voxels: OccupiedVoxels,
        *,
        subvoxel_factor: int,
        max_points: Optional[int],
    ) -> OccupiedVoxels:
        """Deterministically downsample source voxels before subvoxel expansion."""
        point_cap = BlockSparseTSDFIntegrator._validate_max_points(max_points)
        samples_per_voxel = subvoxel_factor**3
        if point_cap is None or len(voxels) * samples_per_voxel <= point_cap:
            return voxels

        source_count = point_cap // samples_per_voxel
        if source_count == 0:
            empty_centers = voxels.centers[:0]
            empty_blocks = voxels.block_idx_per_voxel[:0]
            return OccupiedVoxels(
                centers=empty_centers,
                block_idx_per_voxel=empty_blocks,
                block_data=voxels.block_data,
            )

        indices = (
            torch.linspace(
                0,
                len(voxels) - 1,
                steps=source_count,
                device=voxels.centers.device,
                dtype=torch.float32,
            )
            .round()
            .to(torch.long)
        )
        return OccupiedVoxels(
            centers=voxels.centers[indices].contiguous(),
            block_idx_per_voxel=voxels.block_idx_per_voxel[indices].contiguous(),
            block_data=voxels.block_data,
        )

    def _expand_subvoxels(
        self,
        voxels: OccupiedVoxels,
        *,
        subvoxel_factor: int,
    ) -> OccupiedVoxels:
        """Return voxel centers or regular subvoxel sample centers."""
        if subvoxel_factor == 1 or len(voxels) == 0:
            return OccupiedVoxels(
                centers=voxels.centers,
                block_idx_per_voxel=voxels.block_idx_per_voxel,
                block_data=voxels.block_data,
                subvoxel_factor=subvoxel_factor,
            )

        device = voxels.centers.device
        dtype = voxels.centers.dtype
        offsets_1d = (
            (torch.arange(subvoxel_factor, device=device, dtype=dtype) + 0.5)
            / float(subvoxel_factor)
            - 0.5
        ) * float(self.config.voxel_size)
        offset_grid = torch.stack(
            torch.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)
        centers = (voxels.centers[:, None, :] + offset_grid[None, :, :]).reshape(-1, 3)
        block_idx_per_voxel = voxels.block_idx_per_voxel.repeat_interleave(subvoxel_factor**3)
        return OccupiedVoxels(
            centers=centers.contiguous(),
            block_idx_per_voxel=block_idx_per_voxel.contiguous(),
            block_data=voxels.block_data,
            subvoxel_factor=subvoxel_factor,
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
        last_camera_integration = dict(self._camera_integrator.last_integration_stats)
        last_camera_integration["support_overflow_count"] = int(
            self._camera_integrator.support_overflow_count.item()
        )
        last_camera_integration["profile_kernel_timings"] = (
            self._camera_integrator.profile_kernel_timings
        )
        last_camera_integration["use_tiled_feature_kernel"] = (
            self._camera_integrator.use_tiled_feature_kernel
        )
        stats["last_camera_integration"] = last_camera_integration
        stats["last_camera_integration_kernel_timings_ms"] = dict(
            self._camera_integrator.last_kernel_timings_ms
        )
        stats["last_integration"] = last_camera_integration
        stats["last_integration_kernel_timings_ms"] = dict(
            self._camera_integrator.last_kernel_timings_ms
        )
        if self._lidar_integrator is not None:
            last_lidar_integration = dict(self._lidar_integrator.last_integration_stats)
            last_lidar_integration["support_overflow_count"] = int(
                self._lidar_integrator.support_overflow_count.item()
            )
            last_lidar_integration["profile_kernel_timings"] = (
                self._lidar_integrator.profile_kernel_timings
            )
            last_lidar_integration["use_tiled_feature_kernel"] = (
                self._lidar_integrator.use_tiled_feature_kernel
            )
            stats["last_lidar_integration"] = last_lidar_integration
            stats["last_lidar_integration_kernel_timings_ms"] = dict(
                self._lidar_integrator.last_kernel_timings_ms
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

    def extract_occupied_points(
        self,
        sample_budget: int = 2000,
        *,
        instance_slot: Optional[int] = None,
        semantic_filter: Optional[SemanticFilter] = None,
        voxel_selection: Optional[torch.Tensor] = None,
        seed: int = 0,
    ) -> OccupiedPoints:
        """Sample observed TSDF zero-crossing faces, pruning abs(SDF) >= 0.5 voxel.

        Returns fixed-capacity CUDA buffers and a GPU valid count. Unknown
        interpolation corners are omitted and weights renormalized. Crossing
        endpoints must both be observed. Filters select the nearest endpoint;
        interpolation still uses the full field. No refill or CPU count reads.
        Warm up once for each budget before graph capture. Calls sharing this
        mapper must be serialized; results are reused by the next same-budget call.
        """
        validate_point_query(sample_budget, instance_slot, seed)
        if self._point_workspace is None:
            self._point_workspace = SurfacePointWorkspace(self._tsdf)
        return self._point_workspace.extract(
            self._tsdf,
            sample_budget,
            self.config.minimum_tsdf_weight,
            instance_slot,
            semantic_filter,
            voxel_selection,
            seed,
        )

    def extract_occupied_voxels(
        self,
        surface_only: bool = False,
        sdf_threshold: float = None,
        *,
        subvoxel_factor: int = 1,
        max_points: Optional[int] = None,
        texture_observations: CameraObservation | Sequence[CameraObservation] | None = None,
        camera_min_distance: Optional[float] = None,
        camera_max_distance: Optional[float] = None,
        texture_depth_tolerance_m: Optional[float] = None,
        semantic_filter: Optional[SemanticFilter] = None,
    ) -> OccupiedVoxels:
        """Extract occupied voxel centers and per-voxel block indices.

        Matches dense TSDFIntegrator.extract_occupied_voxels behavior:
        - surface_only=True: extract voxels near zero-crossing (|SDF| < sdf_threshold)
        - surface_only=False: extract surface + inside voxels (SDF <= 0)

        Args:
            surface_only: If True, extract only surface voxels (|SDF| < sdf_threshold).
                          If False (default), include inside voxels too (SDF <= 0).
            sdf_threshold: Threshold for surface_only=True. Defaults to voxel_size.
            subvoxel_factor: Number of point samples per extracted voxel axis.
            max_points: Optional cap applied after subvoxel expansion by
                downsampling source voxels before expansion.
            texture_observations: Optional RGB or RGB-D texture observations used
                to color output samples after visibility-depth checks. Missing
                depth is rendered from the current TSDF.
            camera_min_distance: Minimum camera-frame z used for texture projection.
            camera_max_distance: Maximum camera-frame z used for texture projection.
            texture_depth_tolerance_m: Absolute visibility-depth tolerance. If
                None, each sample uses ``max(2 * voxel_size, 0.01 * projected_z)``.
            semantic_filter: Optional fused-feature filter applied before extraction.

        Returns:
            :class:`OccupiedVoxels` — ``.centers`` for voxel positions,
            ``.block_idx_per_voxel`` for the per-voxel ``pool_idx``, and
            ``.block_data`` as a read-only view over per-block storage.
            Use ``.colors_uint8()`` to gather per-voxel RGB.
        """
        subvoxel_factor = BlockSparseTSDFIntegrator._validate_subvoxel_factor(subvoxel_factor)
        # Default threshold to voxel_size (matches dense behavior)
        if sdf_threshold is None:
            sdf_threshold = self.config.voxel_size

        prepared_semantic_filter = (
            self._prepare_semantic_filter(semantic_filter) if semantic_filter is not None else None
        )
        voxels = extract_occupied_voxels_block_sparse(
            self._tsdf,
            surface_only=surface_only,
            sdf_threshold=sdf_threshold,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            grid_shape=self.config.grid_shape,
            semantic_filter=prepared_semantic_filter,
        )
        voxels = self._limit_voxels_for_point_cap(
            voxels,
            subvoxel_factor=subvoxel_factor,
            max_points=max_points,
        )
        voxels = self._expand_subvoxels(voxels, subvoxel_factor=subvoxel_factor)
        if texture_observations is None:
            return voxels
        return self._texture_projector.texture_occupied_voxels(
            voxels,
            texture_observations,
            camera_min_distance=camera_min_distance,
            camera_max_distance=camera_max_distance,
            texture_depth_tolerance_m=texture_depth_tolerance_m,
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
        features = self._tsdf.data.block_features[active_pool_idx].float().sum(dim=1)
        weight = (
            self._tsdf.data.block_feature_weight[active_pool_idx]
            .float()
            .sum(dim=1)
            .clamp(min=1e-6)
        )
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
        self._mark_semantic_data_changed()
