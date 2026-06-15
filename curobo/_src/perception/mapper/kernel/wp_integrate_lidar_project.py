# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""LiDAR range-image TSDF integration launcher."""

from __future__ import annotations

import math

import torch
import warp as wp

from curobo._src.perception.mapper.constants import _validate_feature_channels_per_thread
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.warp import get_warp_device_stream
from curobo.logging import log_and_raise


class LidarProjectIntegrator:
    """Projective LiDAR range-image TSDF/RGB/feature integrator."""

    def __init__(
        self,
        lidar_num_sensors: int,
        lidar_image_height: int,
        lidar_image_width: int,
        voxel_size: float,
        block_size: int,
        truncation_distance: float,
        max_blocks: int,
        max_visible_blocks_per_lidar_integration: int | None = None,
        max_support_pixels_per_block_lidar: int = 32,
        device: str = "cuda:0",
        feature_channels_per_thread: int = 4,
        use_tiled_feature_kernel: bool = True,
        lidar_feature_grid_shape: tuple[int, int] | None = None,
        linear_interpolation_max_allowable_difference_vox: float = 2.0,
        nearest_interpolation_max_allowable_dist_to_ray_vox: float = 0.5,
        profile_kernel_timings: bool = False,
    ):
        if lidar_num_sensors <= 0:
            log_and_raise(
                f"lidar_num_sensors must be positive, got {lidar_num_sensors}."
            )
        if lidar_image_height <= 0 or lidar_image_width <= 0:
            log_and_raise(
                "lidar_image_height and lidar_image_width must be positive, got "
                f"{lidar_image_height}x{lidar_image_width}."
            )
        if max_visible_blocks_per_lidar_integration is None:
            max_visible_blocks_per_lidar_integration = max_blocks
        if (
            max_visible_blocks_per_lidar_integration <= 0
            or max_visible_blocks_per_lidar_integration > max_blocks
        ):
            log_and_raise(
                "max_visible_blocks_per_lidar_integration must satisfy "
                f"0 < C <= max_blocks ({max_blocks}), got "
                f"{max_visible_blocks_per_lidar_integration}."
            )
        if max_support_pixels_per_block_lidar <= 0:
            log_and_raise(
                "max_support_pixels_per_block_lidar must be positive, got "
                f"{max_support_pixels_per_block_lidar}."
            )
        if linear_interpolation_max_allowable_difference_vox <= 0.0:
            log_and_raise(
                "linear_interpolation_max_allowable_difference_vox must be positive, got "
                f"{linear_interpolation_max_allowable_difference_vox}."
            )
        if nearest_interpolation_max_allowable_dist_to_ray_vox <= 0.0:
            log_and_raise(
                "nearest_interpolation_max_allowable_dist_to_ray_vox must be positive, got "
                f"{nearest_interpolation_max_allowable_dist_to_ray_vox}."
            )

        self.device = device
        self.lidar_num_sensors = int(lidar_num_sensors)
        self.lidar_image_height = int(lidar_image_height)
        self.lidar_image_width = int(lidar_image_width)
        self.max_blocks = int(max_blocks)
        self.max_visible_blocks_per_lidar_integration = int(
            max_visible_blocks_per_lidar_integration
        )
        self.max_support_pixels_per_block_lidar = int(max_support_pixels_per_block_lidar)
        self.linear_interpolation_max_allowable_difference_m = (
            float(linear_interpolation_max_allowable_difference_vox) * float(voxel_size)
        )
        self.nearest_interpolation_max_allowable_dist_to_ray_m = (
            float(nearest_interpolation_max_allowable_dist_to_ray_vox) * float(voxel_size)
        )

        block_edge = block_size * voxel_size
        safe_step = block_edge / 1.42
        self.safe_step = safe_step
        self.num_samples = math.ceil(2.0 * truncation_distance / safe_step) + 1
        self.num_block_key_candidates = (
            self.lidar_num_sensors
            * self.lidar_image_height
            * self.lidar_image_width
            * self.num_samples
        )

        self.block_keys = torch.empty(
            self.num_block_key_candidates,
            dtype=torch.int64,
            device=device,
        )
        self.pool_indices = torch.empty(
            self.max_visible_blocks_per_lidar_integration,
            dtype=torch.int32,
            device=device,
        )
        self.visible_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.visible_epoch = torch.zeros(max_blocks, dtype=torch.int32, device=device)
        self.pool_to_visible_slot = torch.empty(max_blocks, dtype=torch.int32, device=device)
        self.support_counts = torch.zeros(
            (self.max_visible_blocks_per_lidar_integration, self.lidar_num_sensors),
            dtype=torch.int32,
            device=device,
        )
        self.support_pixels = torch.empty(
            (
                self.max_visible_blocks_per_lidar_integration,
                self.lidar_num_sensors,
                self.max_support_pixels_per_block_lidar,
            ),
            dtype=torch.int32,
            device=device,
        )
        self.support_overflow_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.frame_epoch = 0
        _validate_feature_channels_per_thread(feature_channels_per_thread)
        self.feature_channels_per_thread = int(feature_channels_per_thread)
        self.use_tiled_feature_kernel = bool(use_tiled_feature_kernel)
        self.lidar_feature_grid_shape = lidar_feature_grid_shape
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
        self.frame_epoch += 1
        if self.frame_epoch >= 2_147_483_647:
            self.visible_epoch.zero_()
            self.frame_epoch = 1
        return self.frame_epoch

    def _prepare_frame_scratch(self) -> None:
        self.visible_count.zero_()
        self.support_counts.zero_()
        self.support_overflow_count.zero_()

    def _dedup_visible_blocks_hash(self, tsdf, kernels, data, n_keys, device, stream) -> int:
        frame_epoch = self._next_frame_epoch()
        self._timer_start()
        wp.launch(
            kernels.allocate_visible_blocks_from_keys_kernel,
            dim=n_keys,
            inputs=[
                wp.from_torch(self.block_keys[:n_keys]),
                n_keys,
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
                self.max_visible_blocks_per_lidar_integration,
            ],
            device=device,
            stream=stream,
        )
        self._timer_stop("allocate_visible_blocks_from_keys_kernel")
        num_visible_blocks = int(self.visible_count.item())
        self.last_integration_stats["num_visible_blocks"] = num_visible_blocks
        return num_visible_blocks

    def _build_support_pixels(self, tsdf, kernels, data, n_keys, frame_epoch, device, stream):
        self._timer_start()
        wp.launch(
            kernels.lidar_build_support_pixels_from_keys_kernel,
            dim=n_keys,
            inputs=[
                wp.from_torch(self.block_keys[:n_keys]),
                n_keys,
                data.hash_table,
                tsdf.config.hash_capacity,
                tsdf.config.max_blocks,
                wp.from_torch(self.visible_epoch),
                frame_epoch,
                wp.from_torch(self.pool_to_visible_slot),
                self.max_visible_blocks_per_lidar_integration,
                wp.from_torch(self.support_counts),
                wp.from_torch(self.support_pixels),
                wp.from_torch(self.support_overflow_count),
            ],
            device=device,
            stream=stream,
        )
        self._timer_stop("lidar_build_support_pixels_from_keys_kernel")

    def _clear_new_blocks(self, tsdf, kernels, data, num_visible_blocks, device, stream):
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
            wp.launch(
                kernels.clear_new_block_features_kernel,
                dim=(max_clearable, tsdf.data.feature_dim),
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

    def integrate(
        self,
        tsdf,
        range_images: torch.Tensor,
        rgb_images: torch.Tensor,
        lidar_positions: torch.Tensor,
        lidar_quaternions: torch.Tensor,
        valid_range_m: torch.Tensor,
        elevation_range_rad: torch.Tensor,
        feature_grid: torch.Tensor | None = None,
    ) -> None:
        tsdf.prepare_frame()
        self._prepare_frame_scratch()
        self.last_kernel_timings_ms = {}
        self.last_integration_stats = {}
        kernels = tsdf.kernels
        if kernels.lidar_num_sensors != self.lidar_num_sensors:
            log_and_raise(
                f"kernels.lidar_num_sensors={kernels.lidar_num_sensors} does not match "
                f"LidarProjectIntegrator.lidar_num_sensors={self.lidar_num_sensors}."
            )
        if (
            kernels.lidar_image_height != self.lidar_image_height
            or kernels.lidar_image_width != self.lidar_image_width
        ):
            log_and_raise(
                "kernel LiDAR image shape mismatch: expected "
                f"({self.lidar_image_height}, {self.lidar_image_width}), got "
                f"({kernels.lidar_image_height}, {kernels.lidar_image_width})."
            )
        if kernels.num_samples != self.num_samples:
            log_and_raise(
                f"kernels.num_samples={kernels.num_samples} does not match "
                f"LidarProjectIntegrator.num_samples={self.num_samples}."
            )
        if kernels.lidar_feature_grid_shape != self.lidar_feature_grid_shape:
            log_and_raise(
                f"kernels.lidar_feature_grid_shape={kernels.lidar_feature_grid_shape} "
                f"does not match LidarProjectIntegrator.lidar_feature_grid_shape="
                f"{self.lidar_feature_grid_shape}."
            )
        if (
            kernels.max_support_pixels_per_block_lidar
            != self.max_support_pixels_per_block_lidar
        ):
            log_and_raise(
                "kernel max_support_pixels_per_block_lidar="
                f"{kernels.max_support_pixels_per_block_lidar} does not match "
                "LidarProjectIntegrator.max_support_pixels_per_block_lidar="
                f"{self.max_support_pixels_per_block_lidar}."
            )

        expected_range_shape = (
            self.lidar_num_sensors,
            self.lidar_image_height,
            self.lidar_image_width,
        )
        if tuple(range_images.shape) != expected_range_shape:
            log_and_raise(
                f"range_images shape mismatch: expected {expected_range_shape}, "
                f"got {tuple(range_images.shape)}."
            )
        expected_rgb_shape = expected_range_shape + (3,)
        if tuple(rgb_images.shape) != expected_rgb_shape:
            log_and_raise(
                f"rgb_images shape mismatch: expected {expected_rgb_shape}, "
                f"got {tuple(rgb_images.shape)}."
            )

        data = tsdf.get_warp_data()
        device, stream = get_warp_device_stream(range_images)
        n_keys = self.num_block_key_candidates

        self._timer_start()
        wp.launch(
            kernels.lidar_compute_block_keys_only_kernel,
            dim=n_keys,
            inputs=[
                wp.from_torch(lidar_positions, dtype=wp.float32),
                wp.from_torch(lidar_quaternions, dtype=wp.float32),
                wp.from_torch(range_images, dtype=wp.float32),
                wp.from_torch(valid_range_m, dtype=wp.float32),
                wp.from_torch(elevation_range_rad, dtype=wp.float32),
                wp.from_torch(self.block_keys[:n_keys]),
            ],
            device=device,
            stream=stream,
        )
        self._timer_stop("lidar_compute_block_keys_only_kernel")

        num_visible_blocks = self._dedup_visible_blocks_hash(
            tsdf, kernels, data, n_keys, device, stream
        )
        if num_visible_blocks == 0:
            self.last_integration_stats["num_visible_blocks"] = 0
            return
        if num_visible_blocks > self.max_visible_blocks_per_lidar_integration:
            self._clear_new_blocks(tsdf, kernels, data, num_visible_blocks, device, stream)
            log_and_raise(
                f"num_visible_blocks={num_visible_blocks} exceeds "
                "max_visible_blocks_per_lidar_integration="
                f"{self.max_visible_blocks_per_lidar_integration}. Increase the config value."
            )

        self._build_support_pixels(tsdf, kernels, data, n_keys, self.frame_epoch, device, stream)
        self._clear_new_blocks(tsdf, kernels, data, num_visible_blocks, device, stream)

        block_voxels = tsdf.block_size**3
        self._timer_start()
        wp.launch(
            kernels.lidar_integrate_voxels_kernel,
            dim=(num_visible_blocks, block_voxels),
            inputs=[
                wp.from_torch(self.pool_indices),
                num_visible_blocks,
                wp.from_torch(lidar_positions, dtype=wp.float32),
                wp.from_torch(lidar_quaternions, dtype=wp.float32),
                wp.from_torch(range_images, dtype=wp.float32),
                wp.from_torch(valid_range_m, dtype=wp.float32),
                wp.from_torch(elevation_range_rad, dtype=wp.float32),
                self.linear_interpolation_max_allowable_difference_m,
                self.nearest_interpolation_max_allowable_dist_to_ray_m,
                data.block_coords,
                data.block_data,
            ],
            device=device,
            stream=stream,
        )
        self._timer_stop("lidar_integrate_voxels_kernel")

        rgb_flat = rgb_images.reshape(
            self.lidar_num_sensors * self.lidar_image_height,
            self.lidar_image_width,
            3,
        )
        self._timer_start()
        wp.launch(
            kernels.lidar_integrate_block_rgb_from_support_kernel,
            dim=(num_visible_blocks, self.lidar_num_sensors),
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
        self._timer_stop("lidar_integrate_block_rgb_from_support_kernel")

        feature_dim_cfg = tsdf.data.feature_dim if tsdf.data.has_features else 0
        if feature_grid is not None and not tsdf.data.has_features:
            log_and_raise(
                "feature_grid was provided but feature_dim == 0; enable features via "
                "MapperCfg.feature_dim or BlockSparseTSDFIntegratorCfg.feature_dim."
            )
        if tsdf.data.has_features and feature_grid is not None:
            expected_feature_shape = (
                self.lidar_num_sensors,
                self.lidar_feature_grid_shape[0],
                self.lidar_feature_grid_shape[1],
                feature_dim_cfg,
            )
            if tuple(feature_grid.shape) != expected_feature_shape:
                log_and_raise(
                    f"feature_grid shape mismatch: expected {expected_feature_shape}, "
                    f"got {tuple(feature_grid.shape)}."
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
            self._timer_start()
            if self.use_tiled_feature_kernel:
                feature_tile_channels = max(1, min(feature_dim_cfg, kernels.max_feature_tile_channels))
                feature_channel_tiles = (
                    feature_dim_cfg + feature_tile_channels - 1
                ) // feature_tile_channels
                wp.launch_tiled(
                    kernels.lidar_integrate_features_from_support_tiled_kernel,
                    dim=(num_visible_blocks, self.lidar_num_sensors, feature_channel_tiles),
                    block_dim=64,
                    inputs=feature_inputs,
                    device=device,
                    stream=stream,
                )
                feature_kernel_name = "lidar_integrate_features_from_support_tiled_kernel"
            else:
                wp.launch(
                    kernels.lidar_integrate_features_from_support_grouped_kernel,
                    dim=(num_visible_blocks, self.lidar_num_sensors, feature_channel_groups),
                    inputs=feature_inputs,
                    device=device,
                    stream=stream,
                )
                feature_kernel_name = "lidar_integrate_features_from_support_grouped_kernel"
            self._timer_stop(feature_kernel_name)

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
