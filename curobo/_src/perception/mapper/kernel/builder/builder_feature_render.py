# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deferred rendered-feature gather kernels."""

from __future__ import annotations

import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.util.warp import warp_kernel


def make_feature_render_kernels(
    block_size: int,
    *,
    feature_dim: int,
) -> dict[str, object]:
    """Build fp16 and fp32 kernels for materializing rendered feature references."""
    feature_weight_epsilon = wp.constant(wp.float32(1.0e-6))

    @warp_kernel(
        f"gather_rendered_features_fp32_bs{block_size}_fd{feature_dim}"
    )
    def gather_rendered_features_fp32_kernel(
        tsdf: BlockSparseTSDFWarp,
        feature_block_idx: wp.array(dtype=wp.int32),
        feature_node_idx: wp.array(dtype=wp.int16),
        feature_valid: wp.array(dtype=wp.uint8),
        source_width: wp.int32,
        source_pixels_per_camera: wp.int32,
        patch_top: wp.int32,
        patch_left: wp.int32,
        patch_height: wp.int32,
        patch_width: wp.int32,
        channel_start: wp.int32,
        out_features: wp.array2d(dtype=wp.float32),
    ):
        output_pixel_idx, output_channel_idx = wp.tid()
        patch_pixels = patch_height * patch_width
        camera_idx = output_pixel_idx // patch_pixels
        patch_pixel_idx = output_pixel_idx - camera_idx * patch_pixels
        patch_x = patch_pixel_idx % patch_width
        patch_y = patch_pixel_idx // patch_width
        source_idx = (
            camera_idx * source_pixels_per_camera
            + (patch_top + patch_y) * source_width
            + patch_left
            + patch_x
        )

        value = wp.float32(0.0)
        if feature_valid[source_idx] != wp.uint8(0):
            block_idx = feature_block_idx[source_idx]
            node_idx = wp.int32(feature_node_idx[source_idx])
            feature_weight = wp.float32(
                tsdf.block_feature_weight[block_idx, node_idx]
            )
            if feature_weight > feature_weight_epsilon:
                channel_idx = channel_start + wp.int32(output_channel_idx)
                value = (
                    wp.float32(tsdf.block_features[block_idx, node_idx, channel_idx])
                    / feature_weight
                )
        out_features[output_pixel_idx, output_channel_idx] = value

    @warp_kernel(
        f"gather_rendered_features_fp16_bs{block_size}_fd{feature_dim}"
    )
    def gather_rendered_features_fp16_kernel(
        tsdf: BlockSparseTSDFWarp,
        feature_block_idx: wp.array(dtype=wp.int32),
        feature_node_idx: wp.array(dtype=wp.int16),
        feature_valid: wp.array(dtype=wp.uint8),
        source_width: wp.int32,
        source_pixels_per_camera: wp.int32,
        patch_top: wp.int32,
        patch_left: wp.int32,
        patch_height: wp.int32,
        patch_width: wp.int32,
        channel_start: wp.int32,
        out_features: wp.array2d(dtype=wp.float16),
    ):
        output_pixel_idx, output_channel_idx = wp.tid()
        patch_pixels = patch_height * patch_width
        camera_idx = output_pixel_idx // patch_pixels
        patch_pixel_idx = output_pixel_idx - camera_idx * patch_pixels
        patch_x = patch_pixel_idx % patch_width
        patch_y = patch_pixel_idx // patch_width
        source_idx = (
            camera_idx * source_pixels_per_camera
            + (patch_top + patch_y) * source_width
            + patch_left
            + patch_x
        )

        value = wp.float32(0.0)
        if feature_valid[source_idx] != wp.uint8(0):
            block_idx = feature_block_idx[source_idx]
            node_idx = wp.int32(feature_node_idx[source_idx])
            feature_weight = wp.float32(
                tsdf.block_feature_weight[block_idx, node_idx]
            )
            if feature_weight > feature_weight_epsilon:
                channel_idx = channel_start + wp.int32(output_channel_idx)
                value = (
                    wp.float32(tsdf.block_features[block_idx, node_idx, channel_idx])
                    / feature_weight
                )
        out_features[output_pixel_idx, output_channel_idx] = wp.float16(value)

    return {
        "gather_rendered_features_fp32_kernel": (
            gather_rendered_features_fp32_kernel
        ),
        "gather_rendered_features_fp16_kernel": (
            gather_rendered_features_fp16_kernel
        ),
    }
