# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse TSDF decay + recycle launch wrappers.

The Warp kernels themselves are built by
:func:`curobo._src.perception.mapper.kernel.builder.builder_decay.make_decay_kernels`
and are reached through ``tsdf.kernels`` at launch time. This module
only hosts the Python-side launch wrappers used by
:class:`BlockSparseTSDFIntegrator`.
"""

from __future__ import annotations

import torch
import warp as wp

from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream

# =============================================================================
# Public API
# =============================================================================


def decay_and_recycle(
    tsdf,  # BlockSparseTSDF instance
    decay_factor: float = 0.95,
) -> int:
    """Decay weights and recycle empty blocks.

    NOT CUDA graph safe (returns count, requires sync).
    Call this periodically OUTSIDE of CUDA graphs to:
    1. Decay all voxel weights by decay_factor
    2. Recycle blocks whose total weight falls below threshold

    Args:
        tsdf: BlockSparseTSDF instance.
        decay_factor: Weight multiplier per call (0.95 = 5% decay).

    Returns:
        Number of blocks recycled.
    """
    num_blocks = min(int(tsdf.data.num_allocated.item()), tsdf.config.max_blocks)

    if decay_factor < 1.0:
        tsdf.data.block_data[:num_blocks].mul_(decay_factor)
        tsdf.data.block_rgb[:num_blocks].mul_(decay_factor)
        if tsdf.data.has_features:
            tsdf.data.block_features[:num_blocks].mul_(decay_factor)
            tsdf.data.block_feature_weight[:num_blocks].mul_(decay_factor)
    tsdf.data.block_sums[:num_blocks] = tsdf.data.block_data[:num_blocks, :, 1].sum(
        dim=1, dtype=torch.float32
    )

    launch_recycle(tsdf, num_blocks=num_blocks)
    return tsdf.data.recycle_count


def launch_recycle(tsdf, num_blocks: int | None = None):
    """Launch block recycling without recomputing ``block_sums``.

    Callers must populate ``tsdf.data.block_sums`` for the launch range before
    invoking this helper.

    Args:
        tsdf: BlockSparseTSDF instance.
        num_blocks: If provided, use as launch dim. Otherwise launch over the
            current ``num_allocated`` high-water mark.
    """
    max_blocks = tsdf.config.max_blocks
    launch_blocks = num_blocks
    if launch_blocks is None:
        launch_blocks = int(tsdf.data.num_allocated.item())
    launch_blocks = min(int(launch_blocks), max_blocks)

    tsdf.data.recycle_count.zero_()
    if launch_blocks <= 0:
        return

    data = tsdf.get_warp_data()

    device, stream = get_warp_device_stream(tsdf.data.block_data)
    kernels = tsdf.kernels
    wp.launch(
        kernels.recycle_empty_blocks_kernel,
        dim=launch_blocks,
        inputs=[
            data.block_sums,
            data.static_block_sums,
            data.block_to_hash_slot,
            data.hash_table,
            data.free_list,
            data.free_count,
            data.num_allocated,
            max_blocks,
            wp.from_torch(tsdf.data.recycle_count, dtype=wp.int32),
        ],
        device=device,
        stream=stream,
    )


# =============================================================================
# Frustum-Aware Decay API
# =============================================================================


def apply_decay_from_frustum_flags(
    tsdf,
    *,
    num_blocks: int,
    time_decay: float,
    frustum_decay: float,
):
    """Apply decay from precomputed ``tsdf.data.frustum_flags`` and recycle."""
    n = int(num_blocks)
    if n <= 0:
        launch_recycle(tsdf, num_blocks=0)
        return

    if frustum_decay >= 1.0:
        block_data = tsdf.data.block_data[:n]
        if time_decay < 1.0:
            block_data.mul_(time_decay)
            tsdf.data.block_rgb[:n].mul_(time_decay)
            if tsdf.data.has_features:
                tsdf.data.block_features[:n].mul_(time_decay)
                tsdf.data.block_feature_weight[:n].mul_(time_decay)
        tsdf.data.block_sums[:n] = block_data[:, :, 1].sum(dim=1, dtype=torch.float32)
        launch_recycle(tsdf, num_blocks=n)
        return

    frustum_flags = tsdf.data.frustum_flags
    factor = tsdf.data.decay_factor[:n]
    factor.fill_(time_decay)
    factor.masked_fill_(frustum_flags[:n] > 0, time_decay * frustum_decay)

    tsdf.data.block_data[:n].mul_(factor.view(n, 1, 1))
    tsdf.data.block_rgb[:n].mul_(factor.view(n, 1))
    if tsdf.data.has_features:
        tsdf.data.block_features[:n].mul_(factor.view(n, 1))
        tsdf.data.block_feature_weight[:n].mul_(factor)

    tsdf.data.block_sums[:n] = tsdf.data.block_data[:n, :, 1].sum(dim=1, dtype=torch.float32)

    launch_recycle(tsdf, num_blocks=n)


def decay_frustum_aware_multi_sensor(
    tsdf,
    *,
    camera_intrinsics: torch.Tensor | None = None,
    camera_positions: torch.Tensor | None = None,
    camera_quaternions: torch.Tensor | None = None,
    camera_img_shape: tuple[int, int] | None = None,
    camera_depth_minimum_distance: float = 0.1,
    camera_depth_maximum_distance: float = 10.0,
    lidar_positions: torch.Tensor | None = None,
    lidar_quaternions: torch.Tensor | None = None,
    lidar_valid_range_m: torch.Tensor | None = None,
    lidar_elevation_range_rad: torch.Tensor | None = None,
    lidar_img_shape: tuple[int, int] | None = None,
    time_decay: float = 1.0,
    frustum_decay: float = 0.5,
    num_blocks: int = None,
):
    """Frustum-aware decay for the union of provided camera/LiDAR frusta.

    The current depth/range images are not inspected. Sensor geometry marks
    blocks that are currently observable, then one decay/recycle pass is
    applied using the union flag.
    """
    max_blocks = tsdf.config.max_blocks
    n = num_blocks
    if n is None:
        n = int(tsdf.data.num_allocated.item())
    n = min(int(n), max_blocks)
    if n <= 0:
        launch_recycle(tsdf, num_blocks=0)
        return

    if frustum_decay >= 1.0:
        apply_decay_from_frustum_flags(
            tsdf,
            num_blocks=n,
            time_decay=time_decay,
            frustum_decay=frustum_decay,
        )
        return

    data = tsdf.get_warp_data()
    device, stream = get_warp_device_stream(tsdf.data.block_data)
    kernels = tsdf.kernels
    frustum_flags = tsdf.data.frustum_flags
    frustum_flags.zero_()

    has_camera = camera_intrinsics is not None
    if has_camera:
        if camera_positions is None or camera_quaternions is None or camera_img_shape is None:
            log_and_raise(
                "camera_intrinsics requires camera_positions, camera_quaternions, "
                "and camera_img_shape."
            )
        n_cameras = int(camera_intrinsics.shape[0])
        img_H, img_W = (int(camera_img_shape[0]), int(camera_img_shape[1]))
        if n_cameras != kernels.num_cameras:
            log_and_raise(
                f"camera_intrinsics num_cameras={n_cameras} does not match compiled "
                f"kernel num_cameras={kernels.num_cameras}."
            )
        if img_H != kernels.image_height or img_W != kernels.image_width:
            log_and_raise(
                f"camera_img_shape={(img_H, img_W)} does not match compiled kernel "
                f"image shape={(kernels.image_height, kernels.image_width)}."
            )
        check_float32_tensors(
            camera_intrinsics.device,
            camera_intrinsics=camera_intrinsics,
            camera_positions=camera_positions,
            camera_quaternions=camera_quaternions,
        )
        wp.launch(
            kernels.mark_blocks_in_frustum_kernel,
            dim=(n, n_cameras),
            inputs=[
                data.block_coords,
                data.block_to_hash_slot,
                data.num_allocated,
                wp.from_torch(camera_intrinsics, dtype=wp.float32),
                wp.from_torch(camera_positions, dtype=wp.float32),
                wp.from_torch(camera_quaternions, dtype=wp.float32),
                camera_depth_minimum_distance,
                camera_depth_maximum_distance,
                wp.from_torch(frustum_flags, dtype=wp.int32),
                max_blocks,
            ],
            device=device,
            stream=stream,
        )

    has_lidar = lidar_positions is not None
    if has_lidar:
        if (
            lidar_quaternions is None
            or lidar_valid_range_m is None
            or lidar_elevation_range_rad is None
            or lidar_img_shape is None
        ):
            log_and_raise(
                "lidar_positions requires lidar_quaternions, lidar_valid_range_m, "
                "lidar_elevation_range_rad, and lidar_img_shape."
            )
        n_lidars = int(lidar_positions.shape[0])
        lidar_H = int(lidar_img_shape[0])
        if lidar_H <= 0:
            log_and_raise(f"lidar_img_shape height must be positive, got {lidar_H}.")
        check_float32_tensors(
            lidar_positions.device,
            lidar_positions=lidar_positions,
            lidar_quaternions=lidar_quaternions,
            lidar_valid_range_m=lidar_valid_range_m,
            lidar_elevation_range_rad=lidar_elevation_range_rad,
        )
        wp.launch(
            kernels.mark_lidar_blocks_in_frustum_kernel,
            dim=(n, n_lidars),
            inputs=[
                data.block_coords,
                data.block_to_hash_slot,
                data.num_allocated,
                wp.from_torch(lidar_positions, dtype=wp.float32),
                wp.from_torch(lidar_quaternions, dtype=wp.float32),
                wp.from_torch(lidar_valid_range_m, dtype=wp.float32),
                wp.from_torch(lidar_elevation_range_rad, dtype=wp.float32),
                lidar_H,
                wp.from_torch(frustum_flags, dtype=wp.int32),
                max_blocks,
            ],
            device=device,
            stream=stream,
        )

    apply_decay_from_frustum_flags(
        tsdf,
        num_blocks=n,
        time_decay=time_decay,
        frustum_decay=frustum_decay,
    )


def decay_frustum_aware_multi_camera(
    tsdf,
    intrinsics: torch.Tensor,
    cam_positions: torch.Tensor,
    cam_quaternions: torch.Tensor,
    img_shape: tuple,
    depth_minimum_distance: float = 0.1,
    depth_maximum_distance: float = 10.0,
    time_decay: float = 1.0,
    frustum_decay: float = 0.5,
    num_blocks: int = None,
):
    """Frustum-aware decay for multiple cameras + block recycling.

    Args:
        tsdf: BlockSparseTSDF instance.
        intrinsics: Camera intrinsics ``(num_cameras, 3, 3)`` float32.
        cam_positions: Camera positions ``(num_cameras, 3)`` float32.
        cam_quaternions: Camera quaternions ``(num_cameras, 4)`` wxyz float32.
        img_shape: Image dimensions ``(H, W)`` (shared across cameras).
        depth_minimum_distance: Minimum observable depth [m].
        depth_maximum_distance: Maximum observable depth [m].
        time_decay: Decay for all voxels.
        frustum_decay: Extra decay for in-view blocks.
        num_blocks: If provided, use as slice size instead of ``max_blocks``.
    """
    decay_frustum_aware_multi_sensor(
        tsdf,
        camera_intrinsics=intrinsics,
        camera_positions=cam_positions,
        camera_quaternions=cam_quaternions,
        camera_img_shape=img_shape,
        camera_depth_minimum_distance=depth_minimum_distance,
        camera_depth_maximum_distance=depth_maximum_distance,
        time_decay=time_decay,
        frustum_decay=frustum_decay,
        num_blocks=num_blocks,
    )
