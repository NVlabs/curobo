# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Voxel-Project TSDF integration launcher.

The Warp kernels themselves are built by
:func:`curobo._src.perception.mapper.kernel.builder.builder_integrate.make_integrate_kernels`
and are reached through ``tsdf.kernels`` at launch time. This module only hosts the
:class:`VoxelProjectIntegrator` launch wrapper used by
:class:`BlockSparseTSDFIntegrator`.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from curobo._src.perception.mapper.constants import _validate_feature_channels_per_thread
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.warp import get_warp_device_stream
from curobo.logging import log_and_raise


class VoxelProjectIntegrator:
    """Voxel-Project TSDF Integrator - Zero Atomics, Voxel-Centric.

    Phases 1-3 discover and allocate blocks. Phase 4 uses voxel-centric
    projection for contention-free TSDF updates. Phase 4 uses a
    variable 2D launch dimension (``n_visible, block_size ** 3``), which
    requires a small D2H sync but eliminates all atomic contention on
    ``block_data``.

    Scratch buffers are allocated once in :meth:`__init__`.
    ``block_keys`` is sized by pixel samples, persistent pool-indexed
    scratch is sized by ``max_blocks``, and visible-frame scratch is
    sized by ``max_visible_blocks_per_integration``. :meth:`integrate`
    never reallocates — incoming tensor shapes must match the values
    passed here or a :class:`ValueError` is raised.
    """

    def __init__(
        self,
        num_cameras: int,
        image_height: int,
        image_width: int,
        voxel_size: float,
        block_size: int,
        truncation_distance: float,
        max_blocks: int,
        max_visible_blocks_per_integration: int | None = None,
        max_support_pixels_per_block_camera: int = 32,
        device: str = "cuda:0",
        use_hash_dedup: bool = True,
        feature_channels_per_thread: int = 4,
        use_tiled_feature_kernel: bool = True,
        feature_grid_shape: tuple[int, int] | None = None,
        profile_kernel_timings: bool = False,
    ):
        """Allocate integration scratch buffers and record launch parameters."""
        if not use_hash_dedup:
            log_and_raise(
                "VoxelProjectIntegrator support-pixel integration requires "
                "use_hash_dedup=True; the legacy torch dedup path does not "
                "build support lists."
            )
        if max_visible_blocks_per_integration is None:
            max_visible_blocks_per_integration = max_blocks
        if (
            max_visible_blocks_per_integration <= 0
            or max_visible_blocks_per_integration > max_blocks
        ):
            log_and_raise(
                "max_visible_blocks_per_integration must satisfy "
                f"0 < C <= max_blocks ({max_blocks}), got "
                f"{max_visible_blocks_per_integration}."
            )
        if max_support_pixels_per_block_camera <= 0:
            log_and_raise(
                "max_support_pixels_per_block_camera must be positive, got "
                f"{max_support_pixels_per_block_camera}."
            )
        if not isinstance(use_tiled_feature_kernel, bool):
            log_and_raise(
                "use_tiled_feature_kernel must be bool, got "
                f"{type(use_tiled_feature_kernel).__name__}."
            )
        if not isinstance(profile_kernel_timings, bool):
            log_and_raise(
                "profile_kernel_timings must be bool, got "
                f"{type(profile_kernel_timings).__name__}."
            )
        self.device = device
        self.num_cameras = num_cameras
        self.image_height = image_height
        self.image_width = image_width
        self.max_blocks = max_blocks
        self.max_visible_blocks_per_integration = int(max_visible_blocks_per_integration)
        self.max_support_pixels_per_block_camera = int(max_support_pixels_per_block_camera)
        self.use_hash_dedup = True

        block_edge = block_size * voxel_size
        safe_step = block_edge / 1.42
        self.safe_step = safe_step
        self.num_samples = math.ceil(2.0 * truncation_distance / safe_step) + 1
        self.num_block_key_candidates = (
            num_cameras * image_height * image_width * self.num_samples
        )

        self.block_keys = torch.empty(
            self.num_block_key_candidates,
            dtype=torch.int64,
            device=device,
        )
        self.pool_indices = torch.empty(
            self.max_visible_blocks_per_integration,
            dtype=torch.int32,
            device=device,
        )
        self.clear_pool_indices = torch.empty(
            max_blocks,
            dtype=torch.int32,
            device=device,
        )
        self.clear_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.visible_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.visible_epoch = torch.zeros(max_blocks, dtype=torch.int32, device=device)
        self.pool_to_visible_slot = torch.empty(max_blocks, dtype=torch.int32, device=device)
        self.support_counts = torch.zeros(
            (self.max_visible_blocks_per_integration, num_cameras),
            dtype=torch.int32,
            device=device,
        )
        self.support_pixels = torch.empty(
            (
                self.max_visible_blocks_per_integration,
                num_cameras,
                self.max_support_pixels_per_block_camera,
            ),
            dtype=torch.int32,
            device=device,
        )
        self.support_overflow_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.frame_epoch = 0
        _validate_feature_channels_per_thread(feature_channels_per_thread)
        self.feature_channels_per_thread = int(feature_channels_per_thread)
        self.use_tiled_feature_kernel = use_tiled_feature_kernel
        self.feature_grid_shape = feature_grid_shape
        self.profile_kernel_timings = profile_kernel_timings
        self.last_kernel_timings_ms: dict[str, float] = {}
        self.last_integration_stats: dict[str, int] = {}

        self._timer = CudaEventTimer()

    def _timer_start(self) -> None:
        if self.profile_kernel_timings:
            self._timer.start()

    def _timer_stop(self, name: str) -> None:
        if self.profile_kernel_timings:
            self.last_kernel_timings_ms[name] = 1000.0 * self._timer.stop()

    def _next_frame_epoch(self) -> int:
        """Return a non-zero int32 epoch for per-frame visible-block marking."""
        self.frame_epoch += 1
        if self.frame_epoch >= 2_147_483_647:
            self.visible_epoch.zero_()
            self.frame_epoch = 1
        return self.frame_epoch

    def _prepare_frame_scratch(self):
        """Reset per-frame visible/support scratch."""
        self.visible_count.zero_()
        self.support_counts.zero_()
        self.support_overflow_count.zero_()

    def _dedup_visible_blocks_hash(
        self, tsdf, kernels, data, num_block_key_candidates, device, stream
    ) -> int:
        """Fused GPU lookup/insert plus visible-pool dedup path."""
        frame_epoch = self._next_frame_epoch()

        self._timer_start()
        wp.launch(
            kernels.allocate_visible_blocks_from_keys_kernel,
            dim=num_block_key_candidates,
            inputs=[
                wp.from_torch(self.block_keys[:num_block_key_candidates]),
                num_block_key_candidates,
                data.hash_table,
                tsdf.config.hash_capacity,
                data.block_coords,
                data.block_to_hash_slot,
                data.num_allocated,
                data.allocation_failures,
                tsdf.config.max_blocks,
                data.free_list,
                data.free_count,
                data.new_blocks,
                data.new_block_count,
                wp.from_torch(self.visible_epoch),
                wp.from_torch(self.visible_count),
                frame_epoch,
                wp.from_torch(self.pool_indices),
                wp.from_torch(self.pool_to_visible_slot),
                self.max_visible_blocks_per_integration,
            ],
            device=device,
            stream=stream,
        )
        self._timer_stop("allocate_visible_blocks_from_keys_kernel")
        num_visible_blocks = int(self.visible_count.item())
        self.last_integration_stats["num_visible_blocks"] = num_visible_blocks
        return num_visible_blocks

    def _build_support_pixels(
        self, tsdf, kernels, data, num_block_key_candidates, frame_epoch, device, stream
    ) -> None:
        """Build per-visible-block support pixels from published visible slots."""
        self._timer_start()
        # ``pool_to_visible_slot`` is intentionally not cleared. Every
        # support key comes from the current frame's ``block_keys`` and the
        # kernel checks ``visible_epoch == frame_epoch`` before reading the
        # slot, so stale entries for non-visible pool indices are ignored.
        wp.launch(
            kernels.build_support_pixels_from_keys_kernel,
            dim=num_block_key_candidates,
            inputs=[
                wp.from_torch(self.block_keys[:num_block_key_candidates]),
                num_block_key_candidates,
                data.hash_table,
                tsdf.config.hash_capacity,
                tsdf.config.max_blocks,
                wp.from_torch(self.visible_epoch),
                frame_epoch,
                wp.from_torch(self.pool_to_visible_slot),
                self.max_visible_blocks_per_integration,
                wp.from_torch(self.support_counts),
                wp.from_torch(self.support_pixels),
                wp.from_torch(self.support_overflow_count),
            ],
            device=device,
            stream=stream,
        )
        self._timer_stop("build_support_pixels_from_keys_kernel")

    def _clear_new_blocks(self, tsdf, kernels, data, num_visible_blocks, device, stream):
        """Clear storage for pool slots allocated during the current frame."""
        block_voxels = tsdf.block_size**3
        max_clearable = min(num_visible_blocks, tsdf.config.max_blocks)
        self._timer_start()
        wp.launch(
            kernels.clear_new_blocks_kernel,
            dim=(max_clearable, block_voxels),
            inputs=[
                data.block_data,
                data.block_rgb,
                data.new_blocks,
                data.new_block_count,
                tsdf.config.max_blocks,
            ],
            device=device,
            stream=stream,
        )
        self._timer_stop("clear_new_blocks_kernel")
        if tsdf.data.has_features:
            self._timer_start()
            feature_dim_cfg = tsdf.data.feature_dim
            wp.launch(
                kernels.clear_new_block_features_kernel,
                dim=(max_clearable, feature_dim_cfg),
                inputs=[
                    data.block_features,
                    data.block_feature_weight,
                data.new_blocks,
                data.new_block_count,
                tsdf.config.max_blocks,
            ],
            device=device,
            stream=stream,
            )
            self._timer_stop("clear_new_block_features_kernel")

    def _world_aabb_to_block_bounds(self, tsdf, bounds_min, bounds_max) -> tuple:
        """Convert a world-space AABB to conservative inclusive block bounds."""
        lo_in = (
            torch.as_tensor(
                bounds_min,
                dtype=torch.float32,
            )
            .flatten()
            .detach()
            .cpu()
        )
        hi_in = (
            torch.as_tensor(
                bounds_max,
                dtype=torch.float32,
            )
            .flatten()
            .detach()
            .cpu()
        )
        if lo_in.numel() != 3 or hi_in.numel() != 3:
            log_and_raise(
                "clear_region bounds must each contain 3 values, got "
                f"bounds_min={tuple(lo_in.shape)}, bounds_max={tuple(hi_in.shape)}."
            )

        lo = torch.minimum(lo_in, hi_in)
        hi = torch.maximum(lo_in, hi_in)
        if not torch.isfinite(lo).all() or not torch.isfinite(hi).all():
            log_and_raise("clear_region bounds must be finite.")

        origin = tsdf.data.origin.detach().to(device="cpu", dtype=torch.float32).flatten()
        voxel_size = float(tsdf.config.voxel_size)
        block_size = int(tsdf.block_size)

        grid_D, grid_H, grid_W = (int(v) for v in tsdf.config.grid_shape)

        center_offset = (
            torch.tensor(
                [grid_W, grid_H, grid_D],
                dtype=torch.float32,
            )
            * 0.5
        )
        v_lo = (lo - origin) / voxel_size + center_offset
        v_hi = (hi - origin) / voxel_size + center_offset

        # Include blocks touching exact AABB boundaries. This can over-clear
        # one adjacent block on boundary-aligned regions, but avoids misses.
        eps_voxels = 1.0e-6
        min_bx = math.floor((float(v_lo[0]) - eps_voxels) / block_size)
        min_by = math.floor((float(v_lo[1]) - eps_voxels) / block_size)
        min_bz = math.floor((float(v_lo[2]) - eps_voxels) / block_size)
        max_bx = math.floor((float(v_hi[0]) + eps_voxels) / block_size)
        max_by = math.floor((float(v_hi[1]) + eps_voxels) / block_size)
        max_bz = math.floor((float(v_hi[2]) + eps_voxels) / block_size)

        max_grid_bx = math.ceil(grid_W / block_size) - 1
        max_grid_by = math.ceil(grid_H / block_size) - 1
        max_grid_bz = math.ceil(grid_D / block_size) - 1
        if max_grid_bx < 0 or max_grid_by < 0 or max_grid_bz < 0:
            return 0, 0, 0, 0, 0, 0, grid_W, grid_H, grid_D
        if (
            max_bx < 0
            or max_by < 0
            or max_bz < 0
            or min_bx > max_grid_bx
            or min_by > max_grid_by
            or min_bz > max_grid_bz
        ):
            return 0, 0, 0, 0, 0, 0, grid_W, grid_H, grid_D
        min_bx = max(min_bx, 0)
        min_by = max(min_by, 0)
        min_bz = max(min_bz, 0)
        max_bx = min(max_bx, max_grid_bx)
        max_by = min(max_by, max_grid_by)
        max_bz = min(max_bz, max_grid_bz)

        offset_x = (max_grid_bx + 1) // 2
        offset_y = (max_grid_by + 1) // 2
        offset_z = (max_grid_bz + 1) // 2
        min_bx -= offset_x
        max_bx -= offset_x
        min_by -= offset_y
        max_by -= offset_y
        min_bz -= offset_z
        max_bz -= offset_z

        count_x = max_bx - min_bx + 1
        count_y = max_by - min_by + 1
        count_z = max_bz - min_bz + 1
        if count_x <= 0 or count_y <= 0 or count_z <= 0:
            return 0, 0, 0, 0, 0, 0, grid_W, grid_H, grid_D

        return (
            min_bx,
            min_by,
            min_bz,
            count_x,
            count_y,
            count_z,
            grid_W,
            grid_H,
            grid_D,
        )

    def clear_blocks(self, tsdf, pool_indices) -> int:
        """Clear dynamic block contents for an explicit pool-index list."""
        if not tsdf.data.has_dynamic and not tsdf.data.has_features:
            return 0

        if not isinstance(pool_indices, torch.Tensor):
            pool_indices = torch.as_tensor(
                pool_indices,
                dtype=torch.int32,
                device=tsdf.data.hash_table.device,
            )
        else:
            pool_indices = pool_indices.to(
                device=tsdf.data.hash_table.device,
                dtype=torch.int32,
            )
        pool_indices = pool_indices.flatten()

        n_clear = int(pool_indices.numel())
        if n_clear <= 0:
            return 0
        n_clear = min(n_clear, tsdf.config.max_blocks)
        self.clear_count.fill_(n_clear)

        if n_clear == 0:
            return 0

        kernels = tsdf.kernels
        data = tsdf.get_warp_data()
        device, stream = get_warp_device_stream(tsdf.data.hash_table)

        if tsdf.data.has_dynamic:
            block_voxels = tsdf.block_size**3
            wp.launch(
                kernels.clear_blocks_by_pool_kernel,
                dim=(n_clear, block_voxels),
                inputs=[
                    wp.from_torch(pool_indices),
                    wp.from_torch(self.clear_count),
                    data.block_data,
                    data.block_rgb,
                    data.block_sums,
                    tsdf.config.max_blocks,
                ],
                device=device,
                stream=stream,
            )

        if tsdf.data.has_features:
            feature_dim = tsdf.data.feature_dim
            wp.launch(
                kernels.clear_block_features_by_pool_kernel,
                dim=(n_clear, feature_dim),
                inputs=[
                    wp.from_torch(pool_indices),
                    wp.from_torch(self.clear_count),
                    data.block_features,
                    data.block_feature_weight,
                    tsdf.config.max_blocks,
                ],
                device=device,
                stream=stream,
            )

        return n_clear

    def clear_region(self, tsdf, bounds_min, bounds_max) -> int:
        """Clear dynamic block contents for allocated blocks intersecting an AABB.

        The AABB is specified in world coordinates. Blocks remain allocated in
        the hash table; this only clears dynamic TSDF/RGB and feature
        accumulators so subsequent integration can refill them in place.
        """
        (
            min_bx,
            min_by,
            min_bz,
            count_x,
            count_y,
            count_z,
            _grid_W_dim,
            _grid_H_dim,
            _grid_D,
        ) = self._world_aabb_to_block_bounds(tsdf, bounds_min, bounds_max)
        if count_x <= 0 or count_y <= 0 or count_z <= 0:
            return 0

        kernels = tsdf.kernels
        data = tsdf.get_warp_data()
        device, stream = get_warp_device_stream(tsdf.data.hash_table)
        self.clear_count.zero_()

        wp.launch(
            kernels.collect_blocks_in_aabb_kernel,
            dim=(count_x, count_y, count_z),
            inputs=[
                data.hash_table,
                tsdf.config.hash_capacity,
                min_bx,
                min_by,
                min_bz,
                count_x,
                count_y,
                count_z,
                wp.from_torch(self.clear_pool_indices),
                wp.from_torch(self.clear_count),
                tsdf.config.max_blocks,
            ],
            device=device,
            stream=stream,
        )

        n_clear = min(int(self.clear_count.item()), tsdf.config.max_blocks)
        return self.clear_blocks(tsdf, self.clear_pool_indices[:n_clear])

    def integrate(
        self,
        tsdf,
        depth_images: torch.Tensor,
        rgb_images: torch.Tensor,
        cam_positions: torch.Tensor,
        cam_quaternions: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_min: float = 0.1,
        depth_max: float = 5.0,
        grid_size: tuple = None,
        feature_grid: "torch.Tensor | None" = None,
    ):
        """Integrate depth from one or more cameras using batched kernels.

        Args:
            tsdf: BlockSparseTSDF instance.
            depth_images: (num_cameras, H, W) float32, meters.
            rgb_images: (num_cameras, H, W, 3) uint8.
            cam_positions: (num_cameras, 3) float32.
            cam_quaternions: (num_cameras, 4) float32, wxyz.
            intrinsics: (num_cameras, 3, 3) float32.
            depth_min: Minimum valid depth.
            depth_max: Maximum valid depth.
            grid_size: Optional (nz, ny, nx) for bounds.
            feature_grid: Optional (num_cameras, feature_H, feature_W,
                feature_dim) float16 feature grid. Dense per-pixel features
                use ``feature_H == H`` and ``feature_W == W``. Launched as the
                per-block feature integration kernel when
                ``tsdf.data.has_features`` and this tensor is provided.

        Raises:
            ValueError: If ``depth_images`` shape does not match the
                ``(num_cameras, image_height, image_width)`` values this
                integrator was constructed with, or if ``rgb_images`` /
                ``feature_grid`` has mismatched shape/layout/dtype.
        """
        tsdf.prepare_frame()
        self._prepare_frame_scratch()
        self.last_kernel_timings_ms = {}
        self.last_integration_stats = {}
        kernels = tsdf.kernels
        if kernels.num_cameras != self.num_cameras:
            log_and_raise(
                f"kernels.num_cameras={kernels.num_cameras} does not match "
                f"VoxelProjectIntegrator.num_cameras={self.num_cameras}."
            )
        if kernels.image_height != self.image_height or kernels.image_width != self.image_width:
            log_and_raise(
                "kernel image shape mismatch: expected "
                f"({self.image_height}, {self.image_width}), got "
                f"({kernels.image_height}, {kernels.image_width})."
            )
        if kernels.num_samples != self.num_samples:
            log_and_raise(
                f"kernels.num_samples={kernels.num_samples} does not match "
                f"VoxelProjectIntegrator.num_samples={self.num_samples}."
            )
        if kernels.grid_shape != tuple(int(v) for v in tsdf.config.grid_shape):
            log_and_raise(
                f"kernels.grid_shape={kernels.grid_shape} does not match "
                f"tsdf.config.grid_shape={tsdf.config.grid_shape}."
            )
        if not math.isclose(
            kernels.voxel_size,
            float(tsdf.config.voxel_size),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            log_and_raise(
                f"kernels.voxel_size={kernels.voxel_size} does not match "
                f"tsdf.config.voxel_size={tsdf.config.voxel_size}."
            )
        if not math.isclose(
            kernels.truncation_distance,
            float(tsdf.config.truncation_distance),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            log_and_raise(
                "kernels.truncation_distance="
                f"{kernels.truncation_distance} does not match "
                f"tsdf.config.truncation_distance={tsdf.config.truncation_distance}."
            )
        if kernels.feature_grid_shape != self.feature_grid_shape:
            log_and_raise(
                f"kernels.feature_grid_shape={kernels.feature_grid_shape} does not match "
                f"VoxelProjectIntegrator.feature_grid_shape={self.feature_grid_shape}."
            )
        if self.feature_channels_per_thread != tsdf.config.feature_channels_per_thread:
            log_and_raise(
                "VoxelProjectIntegrator.feature_channels_per_thread="
                f"{self.feature_channels_per_thread} does not match "
                "tsdf.config.feature_channels_per_thread="
                f"{tsdf.config.feature_channels_per_thread}."
            )
        if self.feature_channels_per_thread != kernels.feature_channels_per_thread:
            log_and_raise(
                "VoxelProjectIntegrator.feature_channels_per_thread="
                f"{self.feature_channels_per_thread} does not match "
                "kernels.feature_channels_per_thread="
                f"{kernels.feature_channels_per_thread}."
            )
        if (
            self.max_support_pixels_per_block_camera
            != tsdf.config.max_support_pixels_per_block_camera
        ):
            log_and_raise(
                "VoxelProjectIntegrator.max_support_pixels_per_block_camera="
                f"{self.max_support_pixels_per_block_camera} does not match "
                "tsdf.config.max_support_pixels_per_block_camera="
                f"{tsdf.config.max_support_pixels_per_block_camera}."
            )
        if (
            self.max_support_pixels_per_block_camera
            != kernels.max_support_pixels_per_block_camera
        ):
            log_and_raise(
                "VoxelProjectIntegrator.max_support_pixels_per_block_camera="
                f"{self.max_support_pixels_per_block_camera} does not match "
                "kernels.max_support_pixels_per_block_camera="
                f"{kernels.max_support_pixels_per_block_camera}."
            )
        if depth_images.shape != (self.num_cameras, self.image_height, self.image_width):
            log_and_raise(
                "depth_images shape mismatch: expected "
                f"({self.num_cameras}, {self.image_height}, {self.image_width}), "
                f"got {tuple(depth_images.shape)}."
            )
        expected_rgb_shape = (self.num_cameras, self.image_height, self.image_width, 3)
        if rgb_images.shape != expected_rgb_shape:
            log_and_raise(
                "rgb_images shape mismatch: expected "
                f"{expected_rgb_shape}, got {tuple(rgb_images.shape)}."
            )

        n_cameras = self.num_cameras
        image_height = self.image_height
        image_width = self.image_width
        num_block_key_candidates = self.num_block_key_candidates

        if grid_size is None:
            grid_size = tsdf.config.grid_shape
        grid_size = tuple(int(v) for v in grid_size)
        if grid_size != kernels.grid_shape:
            log_and_raise(
                f"grid_size={grid_size} does not match compiled "
                f"kernel grid_shape={kernels.grid_shape}."
            )

        data = tsdf.get_warp_data()
        device, stream = get_warp_device_stream(depth_images)
        self._timer_start()

        wp.launch(
            kernels.compute_block_keys_only_kernel,
            dim=num_block_key_candidates,
            inputs=[
                wp.from_torch(intrinsics, dtype=wp.float32),
                wp.from_torch(cam_positions, dtype=wp.float32),
                wp.from_torch(cam_quaternions, dtype=wp.float32),
                wp.from_torch(depth_images, dtype=wp.float32),
                depth_min,
                depth_max,
                wp.from_torch(self.block_keys[:num_block_key_candidates]),
            ],
            device=device,
            stream=stream,
        )
        self._timer_stop("compute_block_keys_only_kernel")

        num_visible_blocks = self._dedup_visible_blocks_hash(
            tsdf, kernels, data, num_block_key_candidates, device, stream
        )

        if num_visible_blocks == 0:
            self.last_integration_stats["num_visible_blocks"] = 0
            return
        if num_visible_blocks > self.max_visible_blocks_per_integration:
            # Allocation has already inserted any missing keys. Clear the
            # newly allocated storage before failing so a caller that retries
            # with a larger visible capacity does not inherit stale recycled
            # TSDF/RGB/feature accumulators.
            self._clear_new_blocks(tsdf, kernels, data, num_visible_blocks, device, stream)
            log_and_raise(
                f"num_visible_blocks={num_visible_blocks} exceeds "
                "max_visible_blocks_per_integration="
                f"{self.max_visible_blocks_per_integration}. Increase the config value."
            )

        self._build_support_pixels(
            tsdf,
            kernels,
            data,
            num_block_key_candidates,
            self.frame_epoch,
            device,
            stream,
        )

        self._clear_new_blocks(tsdf, kernels, data, num_visible_blocks, device, stream)
        block_voxels = tsdf.block_size**3
        rgb_flat = rgb_images.reshape(n_cameras * image_height, image_width, 3)
        self._timer_start()

        wp.launch(
            kernels.integrate_voxels_kernel,
            dim=(num_visible_blocks, block_voxels),
            inputs=[
                wp.from_torch(self.pool_indices),
                num_visible_blocks,
                wp.from_torch(intrinsics, dtype=wp.float32),
                wp.from_torch(cam_positions, dtype=wp.float32),
                wp.from_torch(cam_quaternions, dtype=wp.float32),
                wp.from_torch(depth_images, dtype=wp.float32),
                depth_min,
                depth_max,
                data.block_coords,
                data.block_data,
            ],
            device=device,
            stream=stream,
        )
        self._timer_stop("integrate_voxels_kernel")
        self._timer_start()

        wp.launch(
            kernels.integrate_block_rgb_from_support_kernel,
            dim=(num_visible_blocks, n_cameras),
            inputs=[
                wp.from_torch(self.pool_indices),
                num_visible_blocks,
                wp.from_torch(self.support_counts),
                wp.from_torch(self.support_pixels),
                wp.from_torch(rgb_flat, dtype=wp.uint8),
                data.block_rgb,
            ],
            device=device,
            stream=stream,
        )
        self._timer_stop("integrate_block_rgb_from_support_kernel")
        feature_dim_cfg = tsdf.data.feature_dim if tsdf.data.has_features else 0

        if feature_grid is not None and not tsdf.data.has_features:
            log_and_raise(
                "feature_grid was provided but feature_dim == 0; enable features via "
                "MapperCfg.feature_dim or BlockSparseTSDFIntegratorCfg.feature_dim."
            )

        if tsdf.data.has_features and feature_grid is not None:
            self._timer_start()
            if feature_grid.ndim != 4:
                log_and_raise(
                    "feature_grid must be (num_cameras, feature_H, feature_W, feature_dim), "
                    f"got shape {tuple(feature_grid.shape)}."
                )
            if feature_grid.shape[0] != n_cameras:
                log_and_raise(
                    f"feature_grid num_cameras={feature_grid.shape[0]} does not match "
                    f"configured num_cameras={n_cameras}."
                )
            feature_height = int(feature_grid.shape[1])
            feature_width = int(feature_grid.shape[2])
            if feature_height <= 0 or feature_width <= 0:
                log_and_raise(
                    "feature_grid feature_H and feature_W must be positive, got "
                    f"shape {tuple(feature_grid.shape)}."
                )
            if kernels.feature_grid_shape is not None:
                expected_feature_height, expected_feature_width = kernels.feature_grid_shape
                if (
                    feature_height != expected_feature_height
                    or feature_width != expected_feature_width
                ):
                    log_and_raise(
                        "feature_grid shape mismatch: expected "
                        f"feature_H={expected_feature_height}, "
                        f"feature_W={expected_feature_width}, got "
                        f"{tuple(feature_grid.shape)}."
                    )
            if feature_grid.shape[-1] != feature_dim_cfg:
                log_and_raise(
                    f"feature_grid feature_dim={feature_grid.shape[-1]} does not match "
                    f"configured feature_dim={feature_dim_cfg}."
                )
            if feature_grid.dtype != torch.float16:
                log_and_raise(
                    f"feature_grid dtype must be torch.float16, got {feature_grid.dtype}."
                )
            if feature_grid.device != depth_images.device:
                log_and_raise(
                    f"feature_grid device {feature_grid.device} does not match "
                    f"depth_images device {depth_images.device}."
                )
            if feature_grid.stride(-1) != 1:
                log_and_raise(
                    "feature_grid must be channels-last with stride 1 on the channel dim "
                    "(standard PyTorch layout)."
                )
            feature_inputs = [
                wp.from_torch(self.pool_indices),
                num_visible_blocks,
                wp.from_torch(self.support_counts),
                wp.from_torch(self.support_pixels),
                wp.from_torch(feature_grid, dtype=wp.float16),
                data.block_features,
                data.block_feature_weight,
            ]
            feature_channel_groups = (
                feature_dim_cfg + self.feature_channels_per_thread - 1
            ) // self.feature_channels_per_thread
            if self.use_tiled_feature_kernel:
                feature_tile_channels = max(
                    1,
                    min(feature_dim_cfg, kernels.max_feature_tile_channels),
                )
                feature_channel_tiles = (
                    feature_dim_cfg + feature_tile_channels - 1
                ) // feature_tile_channels
                wp.launch_tiled(
                    kernels.integrate_features_from_support_tiled_kernel,
                    dim=(num_visible_blocks, n_cameras, feature_channel_tiles),
                    block_dim=64,
                    inputs=feature_inputs,
                    device=device,
                    stream=stream,
                )
                feature_kernel_name = "integrate_features_from_support_tiled_kernel"
            else:
                wp.launch(
                    kernels.integrate_features_from_support_grouped_kernel,
                    dim=(num_visible_blocks, n_cameras, feature_channel_groups),
                    inputs=feature_inputs,
                    device=device,
                    stream=stream,
                )
                feature_kernel_name = "integrate_features_from_support_grouped_kernel"
            self._timer_stop(feature_kernel_name)

        # Cap per-block weights so fp16 accumulators stay in finite range.
        # Runs every frame on the blocks we just touched; features are
        # rescaled only if the compiled feature channel is enabled.
        # One thread per (block, channel): RGB uses ch < 3 (with ch == 0
        # also writing the capped weight), features use ch < feature_dim.
        # ``n_channels = max(3, feature_dim)`` keeps RGB live even when
        # features are disabled.
        if True:
            self._timer_start()
            n_channels = max(3, kernels.feature_dim)
            wp.launch(
                kernels.rescale_block_accumulators_kernel,
                dim=(num_visible_blocks, n_channels),
                inputs=[
                    wp.from_torch(self.pool_indices),
                    num_visible_blocks,
                    float(tsdf.config.accumulator_w_max),
                    data.block_features,
                    data.block_feature_weight,
                    data.block_rgb,
                ],
                device=device,
                stream=stream,
            )
            self._timer_stop("rescale_block_accumulators_kernel")
