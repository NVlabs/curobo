# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-Sparse Fused ESDF Integrator - Memory-efficient TSDF+ESDF pipeline.

This module provides a fused integrator that combines block-sparse TSDF
integration with dense ESDF computation. It provides the same API as
ESDFIntegrator but with ~20× memory savings for TSDF storage.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  BlockSparseESDFIntegrator                                 │
    │  ├── BlockSparseTSDF (sparse, ~200MB for 100K blocks)           │
    │  ├── EDT (dense, for PBA propagation on site_index)             │
    │  ├── site_index (dense, ≤1024³)                                 │
    │  ├── min_tsdf (dense, conservative sign buffer)                 │
    │  └── dist_field (dense ESDF output)                             │
    └─────────────────────────────────────────────────────────────────┘

ESDF Pipeline:
    1. Seed: Iterate allocated blocks, mark surface voxels in dense site_index
       - Also writes atomic_min to min_tsdf for ALL observed voxels
       - Guarantees conservative collision detection even with coarse ESDF
    2. Propagate: Run PBA on dense site_index
    3. Distance: Compute signed distances using pre-aggregated min_tsdf
       - No hash lookups required (faster than sampling TSDF directly)
       - Conservative: if ANY TSDF voxel in ESDF cell is inside, cell is inside

Usage:
    integrator = BlockSparseESDFIntegrator(
        voxel_size=0.005,
        origin=torch.tensor([0.0, 0.0, 0.0]),
        grid_shape=(400, 400, 400),
        max_blocks=100_000,
    )

    for obs in observations:
        integrator.integrate(obs)

    esdf = integrator.compute_esdf()
    voxel_grid = integrator.get_voxel_grid()
    mesh = integrator.extract_mesh()
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Mesh, VoxelGrid
from curobo._src.perception.mapper.block_allocation import (
    calculate_tsdf_max_blocks,
)
from curobo._src.perception.mapper.constants import (
    DEFAULT_HASH_LAYOUT,
    _validate_feature_channels_per_thread,
    _validate_feature_grid_shape,
    _validate_feature_integration_kernel,
    _validate_block_size,
    validate_grid_shape_for_hash_layout,
)
from curobo._src.perception.mapper.esdf.edt_jump_flooding import JumpFloodingEDT
from curobo._src.perception.mapper.esdf.edt_parallel_banding import ParallelBandingEDT
from curobo._src.perception.mapper.esdf.kernel.wp_esdf_distance import (
    compute_esdf_from_min_tsdf_warp,
)
from curobo._src.perception.mapper.esdf.kernel.wp_esdf_seed import (
    seed_esdf_sites_from_block_sparse_warp,
    seed_esdf_sites_gather_warp,
)
from curobo._src.perception.mapper.kernel.builder.builder_block_sparse_kernel import (
    make_block_sparse_kernels,
)
from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.storage import (
    BlockSparseTSDF,
    MatchedVoxels,
    OccupiedVoxels,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.util.cuda_graph_util import GraphExecutor
from curobo._src.util.logging import log_info
from curobo._src.util.torch_util import profile_class_methods
from curobo.logging import log_and_raise


@dataclass
class BlockSparseESDFIntegratorCfg:
    """Configuration for BlockSparseESDFIntegrator.

    Attributes:
        voxel_size: TSDF voxel size in meters.
        origin: World coordinate of grid origin (3,).
        esdf_voxel_size: ESDF voxel size. If None, uses same as TSDF.
        esdf_grid_shape: ESDF grid dimensions (nx, ny, nz). X slowest, Z fastest.
            This matches collision kernel memory layout for direct tensor sharing.
        truncation_distance: TSDF truncation distance in meters.
        max_blocks: Maximum allocatable blocks for TSDF.
        hash_capacity: Hash table size (should be ~2× max_blocks).
        depth_minimum_distance: Minimum valid depth in meters.
        depth_maximum_distance: Maximum valid depth in meters.
        frustum_decay: Decay for in-view voxels (0.5=quick adapt, 1.0=no extra decay).
        time_decay: Decay for all voxels (1.0=persist, 0.99=slow fade).
        minimum_tsdf_weight: Minimum weight to consider a voxel as observed.
            Since weight = 1/depth² (clamped to [0.001, 2.0]), this can be interpreted
            as the number of observations at 1m depth. E.g., 0.5 = half an observation
            at 1m, or one observation at ~1.4m, or two observations at 2m.
        blend_esdf: Blend TSDF with EDT for smoother ESDF near surfaces.
            Uses truncation_distance as the blend boundary since TSDF is only
            accurate within truncation distance.
        use_cuda_graph: Use CUDA graphs for acceleration. The integrate
            step is never graph-captured (voxel-project Phase 4 has a
            data-dependent 2D launch dim); only the ESDF computation pipeline
            is captured, gated on ``seeding_method`` (``"gather"`` is
            CUDA-graph safe, ``"scatter"`` is not).
        grid_shape: Required grid dimensions (nz, ny, nx) for bounded TSDF mapping.
        image_height: Image height for buffer pre-allocation.
        image_width: Image width for buffer pre-allocation.
        device: CUDA device.
        dtype: Data type for ESDF distance field.
    """

    voxel_size: float = 0.005
    origin: torch.Tensor = None
    esdf_voxel_size: Optional[float] = None
    esdf_grid_shape: Tuple[int, int, int] = (128, 128, 128)
    truncation_distance: float = 0.04
    # Auto-computed in __post_init__ when left as None so the scratch pool
    # scales with block_size. Hardcoding 100_000 (the historical default at
    # block_size=8) silently under-allocates at smaller block_size and causes
    # whole regions of the scene to drop out once the pool saturates.
    max_blocks: Optional[int] = None
    hash_capacity: Optional[int] = None
    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 5.0
    frustum_decay: float = 0.5
    time_decay: float = 1.0
    minimum_tsdf_weight: float = 0.1
    blend_esdf: bool = False
    use_cuda_graph: bool = True
    grid_shape: Tuple[int, int, int] = None
    image_height: Optional[int] = None  # For buffer pre-allocation
    image_width: Optional[int] = None  # For buffer pre-allocation
    enable_static: bool = False  # Enable static obstacle integration
    static_obstacle_color: Tuple[int, int, int] = (20, 20, 20)  # RGB for static obstacles
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    # Adjacent skip steps for ESDF sign determination.
    # Looks N voxels from site towards query for sign. Set to 0.0 to disable.
    adjacent_skip_steps: float = 1.0
    seeding_method: str = "gather"  # "gather" (CUDA graph safe) or "scatter" (accurate, no graph)
    edt_solver: str = "pba"  # "jfa" (Jump Flooding) or "pba" (Parallel Banding, exact)
    roughness: float = 3.0
    num_cameras: int = 1
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
    #: and feature integration.
    max_support_pixels_per_block_camera: int = 32
    #: Number of adjacent feature channels accumulated per feature-kernel
    #: thread (see BlockSparseTSDFIntegratorCfg).
    feature_channels_per_thread: int = 4
    #: Compile-time cap for feature channels accumulated by one tiled
    #: feature-kernel CTA.
    max_feature_tile_channels: int = 4096
    #: Feature integration launch policy: ``"auto"``, ``"grouped"``, or
    #: ``"tiled"``. Resolved by the TSDF integrator to the low-level launch bool.
    feature_integration_kernel: str = "auto"
    #: Record per-kernel integration timings into
    #: ``get_stats()["last_integration_kernel_timings_ms"]``.
    profile_integration_kernel_timings: bool = False
    #: Per-block accumulator weight cap (see BlockSparseTSDFIntegratorCfg).
    accumulator_w_max: float = 1000.0

    def __post_init__(self):
        if self.origin is None:
            self.origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        if not isinstance(self.origin, torch.Tensor):
            self.origin = torch.tensor(self.origin, dtype=torch.float32)
        if self.esdf_voxel_size is None:
            self.esdf_voxel_size = self.voxel_size
        _validate_block_size(self.block_size)
        self.grid_shape = validate_grid_shape_for_hash_layout(
            self.grid_shape,
            self.block_size,
            field_name="BlockSparseESDFIntegratorCfg.grid_shape",
        )

        # Auto-scale max_blocks / hash_capacity by block_size so that reducing
        # block_size below 8 does not silently shrink the allocation pool.
        if self.max_blocks is None:
            self.max_blocks = calculate_tsdf_max_blocks(
                grid_shape=self.grid_shape,
                voxel_size=self.voxel_size,
                block_size=self.block_size,
                truncation_dist=self.truncation_distance,
                roughness=self.roughness,
            )
        if self.max_blocks > DEFAULT_HASH_LAYOUT.max_pool_idx:
            raise ValueError(
                f"max_blocks={self.max_blocks:,} exceeds the "
                f"{DEFAULT_HASH_LAYOUT.name} pool limit of "
                f"{DEFAULT_HASH_LAYOUT.max_pool_idx:,}."
            )
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
        _validate_feature_channels_per_thread(self.feature_channels_per_thread)
        _validate_feature_grid_shape(
            self.feature_dim,
            self.feature_grid_height,
            self.feature_grid_width,
        )
        if self.max_feature_tile_channels <= 0:
            log_and_raise(
                "max_feature_tile_channels must be positive: "
                f"{self.max_feature_tile_channels}"
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


@profile_class_methods
class BlockSparseESDFIntegrator:
    """Block-sparse fused TSDF+ESDF integrator for memory-efficient mapping.

    Provides the same API as ESDFIntegrator but uses block-sparse TSDF
    storage for ~20× memory savings. ESDF is computed as a dense grid.

    For pose refinement, use BlockSparseRaycastPoseRefiner.

    Args:
        config: Configuration dataclass.

    Example:
        config = BlockSparseESDFIntegratorCfg(
            voxel_size=0.005,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            esdf_grid_shape=(256, 256, 256),
        )
        integrator = BlockSparseESDFIntegrator(config)

        for obs in observations:
            integrator.integrate(obs)

        esdf = integrator.compute_esdf()
        voxel_grid = integrator.get_voxel_grid()
    """

    def __init__(self, config: BlockSparseESDFIntegratorCfg):
        """Initialize BlockSparseESDFIntegrator."""
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = config.dtype
        self.use_cuda_graph = config.use_cuda_graph

        # Validate ESDF grid dimensions (nx, ny, nz) - X slowest, Z fastest
        nx, ny, nz = config.esdf_grid_shape
        if nx > 1024 or ny > 1024 or nz > 1024:
            raise ValueError(
                f"ESDF grid dimensions must be <= 1024 per axis for packed site coordinates. "
                f"Got esdf_grid_shape={config.esdf_grid_shape}. Reduce grid size."
            )

        # Store ESDF configuration
        self._esdf_grid_shape = config.esdf_grid_shape
        self._esdf_voxel_size = torch.tensor(
            [config.esdf_voxel_size], device=self.device, dtype=torch.float32
        )
        self._origin = config.origin.to(device=self.device, dtype=torch.float32)

        # --- Create Block-Sparse TSDF Integrator (handles decay internally) ---
        tsdf_integrator_config = BlockSparseTSDFIntegratorCfg(
            voxel_size=config.voxel_size,
            origin=config.origin,
            truncation_distance=config.truncation_distance,
            max_blocks=config.max_blocks,
            hash_capacity=config.hash_capacity,
            depth_minimum_distance=config.depth_minimum_distance,
            depth_maximum_distance=config.depth_maximum_distance,
            frustum_decay=config.frustum_decay,
            time_decay=config.time_decay,
            minimum_tsdf_weight=config.minimum_tsdf_weight,
            grid_shape=config.grid_shape,
            image_height=config.image_height,
            image_width=config.image_width,
            enable_static=config.enable_static,
            static_obstacle_color=config.static_obstacle_color,
            seeding_method=config.seeding_method,
            num_cameras=config.num_cameras,
            device=config.device,
            block_size=config.block_size,
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
        self._tsdf_integrator = BlockSparseTSDFIntegrator(
            tsdf_integrator_config,
            kernels=make_block_sparse_kernels(config),
        )
        self._tsdf = self._tsdf_integrator.tsdf

        # --- Create Dense ESDF Buffers ---
        self._site_index = torch.full(
            config.esdf_grid_shape, -1, dtype=torch.int32, device=self.device
        )
        self._dist_field = torch.zeros(
            config.esdf_grid_shape, dtype=config.dtype, device=self.device
        )

        # --- Create EDT solver for distance propagation ---
        if config.edt_solver == "pba":
            self._edt = ParallelBandingEDT(
                grid_shape=config.esdf_grid_shape,
                voxel_size=config.esdf_voxel_size,
                device=self.device,
            )
        else:
            self._edt = JumpFloodingEDT(
                grid_shape=config.esdf_grid_shape,
                voxel_size=config.esdf_voxel_size,
                device=self.device,
            )

        # Blending configuration
        self._blend_esdf = config.blend_esdf

        # Ray parity test for inside inference: voxels with no TSDF observation
        # but close to surface are tested via 6 axis-aligned rays.
        # Set to 0 to disable. Default: 0 (disabled).
        self._ray_parity_max_distance_voxels = 0

        # Plane test for inside inference: uses TSDF gradient at nearest site
        # to determine inside/outside. More accurate than ray parity for convex regions.
        self._use_plane_test = False

        # Track last-used ESDF origin for get_voxel_grid()
        self._last_esdf_origin = self._origin.clone()

        # Frame counter
        self._frame_count = 0

        # --- CUDA Graph Executor for the ESDF compute pipeline ---
        # The integrate step is never graph-captured: voxel-project Phase 4
        # has a data-dependent 2D launch dim (`n_unique, block_size**3`)
        # and requires a D2H sync on `n_unique`. Scatter seeding is
        # similarly launch-dim-variable, so ESDF graph capture is gated
        # on ``seeding_method != "scatter"``.
        self._compute_esdf_graph: Optional[GraphExecutor] = None

        if self.use_cuda_graph and config.seeding_method != "scatter":
            self._compute_esdf_graph = GraphExecutor(
                capture_fn=self._compute_esdf_impl,
                device=self.device,
                use_cuda_graph=True,
                clone_outputs=False,
            )

        # Print summary
        # self._print_config()

    def _print_config(self):
        """Log integrator configuration."""
        log_info("BlockSparseESDFIntegrator initialized:")
        log_info(f"  TSDF: block-sparse @ {self.config.voxel_size:.4f}m")
        log_info(f"  TSDF memory: ~{self._tsdf_integrator.tsdf.memory_usage_mb():.1f} MB")
        log_info(f"  ESDF: {self._esdf_grid_shape} @ {self._esdf_voxel_size.item():.4f}m")
        frustum_info = (
            "disabled" if self.config.frustum_decay >= 1.0 else f"{self.config.frustum_decay:.2f}"
        )
        time_info = (
            "disabled" if self.config.time_decay >= 1.0 else f"{self.config.time_decay:.2f}"
        )
        log_info(f"  Frustum decay: {frustum_info} (in-view voxels)")
        log_info(f"  Time decay: {time_info} (all voxels)")
        log_info(f"  CUDA graph: {self.use_cuda_graph}")

    # =========================================================================
    # Component Access
    # =========================================================================

    @property
    def tsdf(self) -> BlockSparseTSDF:
        """Access the BlockSparseTSDF component."""
        return self._tsdf_integrator.tsdf

    @property
    def esdf_grid_shape(self) -> Tuple[int, int, int]:
        """ESDF grid shape."""
        return self._esdf_grid_shape

    @property
    def esdf_voxel_size(self) -> float:
        """ESDF voxel size in meters."""
        return self._esdf_voxel_size.item()

    @property
    def origin(self) -> torch.Tensor:
        """Grid origin in world coordinates."""
        return self._origin

    @property
    def voxel_size(self) -> float:
        """TSDF voxel size in meters."""
        return self.config.voxel_size

    @property
    def truncation_distance(self) -> float:
        """TSDF truncation distance in meters."""
        return self.config.truncation_distance

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Bounded TSDF grid shape."""
        return self.config.grid_shape

    @property
    def dist_field(self) -> torch.Tensor:
        """ESDF distance field tensor."""
        return self._dist_field

    # =========================================================================
    # Public API
    # =========================================================================

    def reset(self) -> None:
        """Reset for new scene. Call OUTSIDE of CUDA graph."""
        self._tsdf_integrator.reset()
        self._site_index.fill_(-1)
        self._dist_field.zero_()
        self._last_esdf_origin.copy_(self._origin)
        self._frame_count = 0

    def integrate(
        self,
        observation: CameraObservation,
    ) -> None:
        """Integrate batched depth observation into block-sparse TSDF.

        The observation must have a leading camera dimension matching
        ``config.num_cameras``. See ``BlockSparseTSDFIntegrator.integrate``
        for the expected tensor shapes. The integrate step is not
        graph-captured (see ``use_cuda_graph`` in the cfg docstring).

        Args:
            observation: Batched camera observation.
        """
        self._tsdf_integrator.integrate(observation)
        self._frame_count += 1

    def clear_region(self, bounds_min, bounds_max) -> int:
        """Clear dynamic TSDF contents for blocks intersecting a world AABB.

        Cached ESDF buffers are invalidated because the TSDF source data has
        changed. Call :meth:`compute_esdf` again before using the ESDF.
        """
        n_clear = self._tsdf_integrator.clear_region(bounds_min, bounds_max)
        if n_clear > 0:
            self._site_index.fill_(-1)
            self._dist_field.zero_()
        return n_clear

    def clear_blocks(self, pool_indices) -> int:
        """Clear dynamic TSDF contents for explicit block-pool indices."""
        n_clear = self._tsdf_integrator.clear_blocks(pool_indices)
        if n_clear > 0:
            self._site_index.fill_(-1)
            self._dist_field.zero_()
        return n_clear

    def compute_esdf(
        self,
        esdf_origin: Optional[torch.Tensor] = None,
        esdf_voxel_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ESDF from current block-sparse TSDF.

        Seeds surface voxels from block-sparse TSDF, runs EDT propagation,
        then computes signed distances.

        Args:
            esdf_origin: Override ESDF origin [m] (for sliding window).
            esdf_voxel_size: Override ESDF voxel size [m].

        Returns:
            Signed distance field (esdf_grid_shape) [m].
        """
        origin = esdf_origin if esdf_origin is not None else self._origin
        if esdf_voxel_size is not None:
            self._esdf_voxel_size.copy_(esdf_voxel_size)

        # Track last-used origin so get_voxel_grid() returns the correct pose
        self._last_esdf_origin.copy_(origin)

        # Full ESDF pipeline (seed + PBA + distance) in single graph
        if self._compute_esdf_graph is not None:
            return self._compute_esdf_graph(origin, self._esdf_voxel_size)
        return self._compute_esdf_impl(origin, self._esdf_voxel_size)

    def _seed_esdf_impl(
        self,
        esdf_origin: torch.Tensor,
        esdf_voxel_size: torch.Tensor,
    ) -> None:
        """Seed ESDF sites from block-sparse TSDF.

        Two methods are available (set via ``config.seeding_method``):

        - ``"scatter"``:
          One thread per TSDF voxel. Each surface voxel maps to the single
          ESDF cell whose grid boundaries contain its center, producing a
          seed band exactly one ESDF voxel thick. This faithfully matches
          the TSDF surface geometry. Launch size depends on allocated TSDF
          blocks (not CUDA-graph safe).
        - ``"gather"``:
          One thread per ESDF voxel. Probes 7 TSDF positions (cell center
          + 6 face centers at ``±esdf_vs/2``). The face-center samples lie
          on the ESDF cell boundary, so they can detect surface voxels from
          neighboring ESDF cells, producing a dilated seed band (~1.5
          voxels thick). The thicker band gives PBA more seed sites near
          surfaces and typically yields slightly higher collision recall.
          Fixed launch dimensions (CUDA-graph safe).

        Note:
            Scatter matches the TSDF surface exactly (1:1 voxel mapping)
            while gather over-seeds by detecting surfaces across cell
            boundaries. The dilated band from gather improves PBA distance
            propagation near surfaces, which is why gather can achieve
            higher recall despite being approximate.
        """
        if self.config.seeding_method == "scatter":
            seed_esdf_sites_from_block_sparse_warp(
                self._tsdf_integrator.tsdf,
                self._site_index,
                esdf_origin,
                esdf_voxel_size,
                self._esdf_grid_shape,
                minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            )
        else:
            # Clear once with a dedicated tensor fill; gather kernel writes only hits.
            seed_esdf_sites_gather_warp(
                self._tsdf_integrator.tsdf,
                self._site_index,
                esdf_origin,
                esdf_voxel_size,
                self._esdf_grid_shape,
                minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            )

    def _propagate_and_distance_impl(
        self,
        esdf_origin: torch.Tensor,
        esdf_voxel_size: torch.Tensor,
    ) -> torch.Tensor:
        """PBA propagation + signed distance computation.

        All launch dimensions are fixed (esdf grid shape).
        Safe for CUDA graph capture.
        """
        self._edt.propagate(self._site_index)

        compute_esdf_from_min_tsdf_warp(
            self._site_index,
            self._tsdf_integrator.tsdf,
            esdf_voxel_size,
            self._esdf_grid_shape,
            self._dist_field,
            esdf_origin,
            adjacent_skip_steps=self.config.adjacent_skip_steps,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
        )

        return self._dist_field

    def _compute_esdf_impl(
        self,
        esdf_origin: torch.Tensor,
        esdf_voxel_size: torch.Tensor,
    ) -> torch.Tensor:
        """Full ESDF computation (seed + propagate + distance).

        All steps use fixed launch dimensions. Safe for CUDA graph capture.
        """
        self._seed_esdf_impl(esdf_origin, esdf_voxel_size)
        return self._propagate_and_distance_impl(esdf_origin, esdf_voxel_size)

    # =========================================================================
    # Mesh and Voxel Extraction
    # =========================================================================
    def extract_mesh(
        self,
        refine_iterations: int = 2,
        surface_only: bool = True,
        level: float = 0.0,
    ) -> Mesh:
        """Extract mesh using GPU marching cubes.

        Args:
            refine_iterations: Newton-Raphson iterations for vertex refinement.
            surface_only: Only extract mesh near surface (|sdf| < truncation).
            level: Isosurface level (typically 0.0).

        Returns:
            Mesh object with vertices, faces, and colors.
        """
        mesh = self._tsdf_integrator.extract_mesh(
            level=level,
            refine_iterations=refine_iterations,
            surface_only=surface_only,
        )

        return mesh

    def extract_occupied_voxels(
        self,
        surface_only: bool = False,
        sdf_threshold: Optional[float] = None,
    ) -> OccupiedVoxels:
        """Extract occupied voxel centers and per-voxel block indices.

        Args:
            surface_only: If True, only extract surface voxels.
            sdf_threshold: Surface threshold. Defaults to the TSDF
                integrator's voxel size.

        Returns:
            :class:`OccupiedVoxels`; see
            :meth:`BlockSparseTSDFIntegrator.extract_occupied_voxels`.
        """
        return self._tsdf_integrator.extract_occupied_voxels(
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
        """Forward to :meth:`BlockSparseTSDFIntegrator.extract_matching_feature_voxels`.

        Returns a :class:`MatchedVoxels` with the extracted voxels plus
        per-block ``pool_idx`` and cosine scores in descending order.
        ``minimum_score`` (if set) prunes blocks below that cosine score
        before voxel extraction. ``feature_projector`` can map stored
        features to the query feature space before cosine scoring.
        """
        return self._tsdf_integrator.extract_matching_feature_voxels(
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

    def get_voxel_grid(self) -> VoxelGrid:
        """Get ESDF as a VoxelGrid for use in collision checking.

        The ESDF is stored in (nx, ny, nz) order (X slowest, Z fastest),
        which matches collision kernel memory layout. No transpose needed.

        Returns:
            VoxelGrid with ESDF distance field and metadata.
        """
        nx, ny, nz = self._esdf_grid_shape
        voxel_size = self._esdf_voxel_size.item()

        # Compute grid dimensions in meters (x, y, z)
        dims = [nx * voxel_size, ny * voxel_size, nz * voxel_size]

        # Pose at center (use last-used ESDF origin, not the initial default)
        origin = self._last_esdf_origin.cpu().tolist()
        pose = origin + [1.0, 0.0, 0.0, 0.0]  # identity quaternion

        # No transpose needed - ESDF is already in (nx, ny, nz) order
        # which matches collision kernel memory layout (X slowest, Z fastest)
        return VoxelGrid(
            name="block_sparse_esdf_grid",
            pose=pose,
            dims=dims,
            voxel_size=voxel_size,
            feature_tensor=self._dist_field,
            feature_dtype=self.dtype,
        )

    def get_stats(
        self,
        scan_pool: bool = True,
        scan_hash: bool = False,
    ) -> Dict[str, Any]:
        """Get integrator statistics.

        Args:
            scan_pool: Forwarded to ``BlockSparseTSDF.get_stats``.
            scan_hash: Forwarded to ``BlockSparseTSDF.get_stats``.

        Returns:
            Dictionary with block usage, allocation stats, and ESDF
            memory accounting. See :meth:`BlockSparseTSDF.get_stats`
            for the full block-pool key list.
        """
        stats = self._tsdf_integrator.get_stats(
            scan_pool=scan_pool,
            scan_hash=scan_hash,
        )
        tsdf_frame_count = stats.get("frame_count", self._frame_count)
        tsdf_memory_mb = stats.pop("memory_mb", self._tsdf_integrator.tsdf.memory_usage_mb())
        stats["frame_count"] = self._frame_count
        stats["tsdf_frame_count"] = tsdf_frame_count
        stats["tsdf_memory_mb"] = tsdf_memory_mb
        stats["esdf_memory_mb"] = (self._site_index.numel() * 4 + self._dist_field.numel() * 2) / (
            1024 * 1024
        )
        stats["total_memory_mb"] = stats["tsdf_memory_mb"] + stats["esdf_memory_mb"]
        return stats

    def memory_usage_mb(self) -> float:
        """Get total GPU memory usage in megabytes."""
        tsdf_mem = self._tsdf_integrator.tsdf.memory_usage_mb()
        esdf_mem = (self._site_index.numel() * 4 + self._dist_field.numel() * 2) / (1024 * 1024)
        return tsdf_mem + esdf_mem

    # =========================================================================
    # Static Obstacle Integration
    # =========================================================================

    def update_static_obstacles(
        self,
        scene: SceneData,
        env_idx: int = 0,
    ) -> None:
        """Update static obstacles in the TSDF from scene collision tensors.

        Delegates to the TSDF integrator's update_static_obstacles method.

        Args:
            scene: Scene collision tensors containing cuboids and meshes.
            env_idx: Environment index for multi-environment scenes.
        """
        self._tsdf_integrator.update_static_obstacles(scene, env_idx)
