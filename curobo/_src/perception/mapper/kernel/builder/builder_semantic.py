# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic-filter kernels and predicates, specialized by feature layout."""

from __future__ import annotations

import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import (
    BlockSparseTSDFWarp,
    PreparedSemanticFilterWarp,
)
from curobo._src.util.warp import warp_func, warp_kernel


def make_semantic_kernels(
    block_size: int,
    *,
    feature_dim: int,
    feature_block_grid_size: int,
) -> dict[str, object]:
    """Build semantic preparation kernels and spatial keep predicates."""
    block_size_constant = wp.constant(wp.int32(block_size))
    feature_dim_constant = wp.constant(wp.int32(feature_dim))
    feature_grid_size_constant = wp.constant(wp.int32(feature_block_grid_size))
    feature_grid_voxels_constant = wp.constant(
        wp.int32(feature_block_grid_size**3)
    )
    feature_weight_epsilon = wp.constant(wp.float32(1.0e-6))

    @warp_func(
        f"semantic_feature_axis_from_local_bs{block_size}_gs{feature_block_grid_size}"
    )
    def semantic_feature_axis_from_local(local_coordinate: wp.float32) -> wp.int32:
        if (
            feature_grid_size_constant <= wp.int32(1)
            or block_size_constant <= wp.int32(1)
        ):
            return wp.int32(0)
        grid_max = wp.float32(feature_grid_size_constant - wp.int32(1))
        scale = grid_max / wp.float32(block_size_constant - wp.int32(1))
        grid_coordinate = wp.clamp(
            (local_coordinate - wp.float32(0.5)) * scale,
            wp.float32(0.0),
            grid_max,
        )
        return wp.int32(wp.floor(grid_coordinate + wp.float32(0.5)))

    @warp_func(
        f"semantic_feature_node_from_local_bs{block_size}_gs{feature_block_grid_size}"
    )
    def semantic_feature_node_from_local(
        local_x: wp.float32,
        local_y: wp.float32,
        local_z: wp.float32,
    ) -> wp.int32:
        gx = semantic_feature_axis_from_local(local_x)
        gy = semantic_feature_axis_from_local(local_y)
        gz = semantic_feature_axis_from_local(local_z)
        return (
            gz * feature_grid_size_constant * feature_grid_size_constant
            + gy * feature_grid_size_constant
            + gx
        )

    @warp_func(
        f"semantic_feature_node_from_local_idx_bs{block_size}_gs{feature_block_grid_size}"
    )
    def semantic_feature_node_from_local_idx(local_idx: wp.int32) -> wp.int32:
        local_x = local_idx % block_size_constant
        local_y = (local_idx // block_size_constant) % block_size_constant
        local_z = local_idx // (block_size_constant * block_size_constant)
        return semantic_feature_node_from_local(
            wp.float32(local_x) + wp.float32(0.5),
            wp.float32(local_y) + wp.float32(0.5),
            wp.float32(local_z) + wp.float32(0.5),
        )

    @warp_func(
        f"semantic_node_keep_bs{block_size}_fd{feature_dim}_gs{feature_block_grid_size}"
    )
    def semantic_node_keep(
        semantic_filter: PreparedSemanticFilterWarp,
        pool_idx: wp.int32,
        feature_node_idx: wp.int32,
    ) -> wp.bool:
        if semantic_filter.block_has_keep[pool_idx] == wp.uint8(0):
            return False
        matches = semantic_filter.node_match[pool_idx, feature_node_idx] != wp.uint8(0)
        if semantic_filter.exclude:
            return not matches
        return matches

    @warp_func(
        f"semantic_local_keep_bs{block_size}_fd{feature_dim}_gs{feature_block_grid_size}"
    )
    def semantic_local_keep(
        semantic_filter: PreparedSemanticFilterWarp,
        pool_idx: wp.int32,
        local_x: wp.float32,
        local_y: wp.float32,
        local_z: wp.float32,
    ) -> wp.bool:
        node_idx = semantic_feature_node_from_local(local_x, local_y, local_z)
        return semantic_node_keep(semantic_filter, pool_idx, node_idx)

    @warp_func(
        f"semantic_local_idx_keep_bs{block_size}_fd{feature_dim}_gs{feature_block_grid_size}"
    )
    def semantic_local_idx_keep(
        semantic_filter: PreparedSemanticFilterWarp,
        pool_idx: wp.int32,
        local_idx: wp.int32,
    ) -> wp.bool:
        node_idx = semantic_feature_node_from_local_idx(local_idx)
        return semantic_node_keep(semantic_filter, pool_idx, node_idx)

    @warp_kernel(
        f"prepare_semantic_node_matches_bs{block_size}_fd{feature_dim}_"
        f"gs{feature_block_grid_size}"
    )
    def prepare_semantic_node_matches_kernel(
        tsdf: BlockSparseTSDFWarp,
        target: wp.array(dtype=wp.float32),
        min_similarity: wp.float32,
        target_argmax: wp.int32,
        use_cosine: wp.int32,
        use_argmax: wp.int32,
        node_match: wp.array2d(dtype=wp.uint8),
    ):
        block_idx, node_idx, lane = wp.tid()
        if block_idx >= tsdf.max_blocks or node_idx >= feature_grid_voxels_constant:
            return
        if (
            block_idx >= tsdf.num_allocated[0]
            or tsdf.block_to_hash_slot[block_idx] < wp.int32(0)
            or wp.float32(tsdf.block_feature_weight[block_idx, node_idx])
            <= feature_weight_epsilon
        ):
            if lane == wp.int32(0):
                node_match[block_idx, node_idx] = wp.uint8(0)
            return

        lane_count = wp.block_dim()
        channel_limit = (
            (feature_dim_constant + lane_count - wp.int32(1)) // lane_count
        ) * lane_count
        local_dot = wp.float32(0.0)
        local_norm_squared = wp.float32(0.0)
        local_maximum = wp.float32(-wp.inf)
        local_argmax = feature_dim_constant

        for channel_idx in range(lane, channel_limit, lane_count):
            if channel_idx < feature_dim_constant:
                feature_value = wp.float32(
                    tsdf.block_features[block_idx, node_idx, channel_idx]
                )
                if use_cosine != wp.int32(0):
                    local_dot += feature_value * target[channel_idx]
                    local_norm_squared += feature_value * feature_value
                if use_argmax != wp.int32(0) and feature_value > local_maximum:
                    local_maximum = feature_value
                    local_argmax = channel_idx

        dot = wp.tile_sum(wp.tile(local_dot))[0]
        norm_squared = wp.tile_sum(wp.tile(local_norm_squared))[0]
        maximum = wp.tile_max(wp.tile(local_maximum))[0]
        argmax_candidate = feature_dim_constant
        if local_maximum == maximum:
            argmax_candidate = local_argmax
        argmax = wp.tile_min(wp.tile(argmax_candidate))[0]

        if lane == wp.int32(0):
            matches = bool(True)
            if use_cosine != wp.int32(0):
                cosine_match = bool(False)
                if norm_squared > wp.float32(0.0):
                    cosine = dot / wp.sqrt(norm_squared)
                    cosine_match = cosine >= min_similarity
                matches = matches and cosine_match
            if use_argmax != wp.int32(0):
                matches = matches and argmax == target_argmax
            node_match[block_idx, node_idx] = wp.uint8(1) if matches else wp.uint8(0)

    @warp_kernel(
        f"prepare_semantic_argmax_node_matches_bs{block_size}_fd{feature_dim}_"
        f"gs{feature_block_grid_size}"
    )
    def prepare_semantic_argmax_node_matches_kernel(
        tsdf: BlockSparseTSDFWarp,
        target_argmax: wp.int32,
        node_match: wp.array2d(dtype=wp.uint8),
    ):
        """Specialized categorical matcher with one thread per feature node."""
        block_idx, node_idx = wp.tid()
        if block_idx >= tsdf.max_blocks or node_idx >= feature_grid_voxels_constant:
            return
        if (
            block_idx >= tsdf.num_allocated[0]
            or tsdf.block_to_hash_slot[block_idx] < wp.int32(0)
            or wp.float32(tsdf.block_feature_weight[block_idx, node_idx])
            <= feature_weight_epsilon
        ):
            node_match[block_idx, node_idx] = wp.uint8(0)
            return

        maximum = wp.float32(-wp.inf)
        argmax = wp.int32(0)
        for channel_idx in range(feature_dim_constant):
            feature_value = wp.float32(
                tsdf.block_features[block_idx, node_idx, channel_idx]
            )
            if feature_value > maximum:
                maximum = feature_value
                argmax = channel_idx
        node_match[block_idx, node_idx] = (
            wp.uint8(1) if argmax == target_argmax else wp.uint8(0)
        )

    @warp_kernel(
        f"prepare_semantic_block_summary_bs{block_size}_fd{feature_dim}_"
        f"gs{feature_block_grid_size}"
    )
    def prepare_semantic_block_summary_kernel(
        tsdf: BlockSparseTSDFWarp,
        node_match: wp.array2d(dtype=wp.uint8),
        exclude: wp.int32,
        block_has_keep: wp.array(dtype=wp.uint8),
    ):
        block_idx, lane = wp.tid()
        if block_idx >= tsdf.max_blocks:
            return
        if (
            block_idx >= tsdf.num_allocated[0]
            or tsdf.block_to_hash_slot[block_idx] < wp.int32(0)
        ):
            if lane == wp.int32(0):
                block_has_keep[block_idx] = wp.uint8(0)
            return

        lane_count = wp.block_dim()
        node_limit = (
            (feature_grid_voxels_constant + lane_count - wp.int32(1)) // lane_count
        ) * lane_count
        local_has_keep = wp.int32(0)
        for node_idx in range(lane, node_limit, lane_count):
            if node_idx < feature_grid_voxels_constant:
                matches = node_match[block_idx, node_idx] != wp.uint8(0)
                keep = matches
                if exclude != wp.int32(0):
                    keep = not matches
                if keep:
                    local_has_keep = wp.int32(1)

        has_keep = wp.tile_max(wp.tile(local_has_keep))[0]
        if lane == wp.int32(0):
            block_has_keep[block_idx] = wp.uint8(has_keep)

    return {
        "semantic_feature_axis_from_local": semantic_feature_axis_from_local,
        "semantic_feature_node_from_local": semantic_feature_node_from_local,
        "semantic_feature_node_from_local_idx": semantic_feature_node_from_local_idx,
        "semantic_node_keep": semantic_node_keep,
        "semantic_local_keep": semantic_local_keep,
        "semantic_local_idx_keep": semantic_local_idx_keep,
        "prepare_semantic_node_matches_kernel": prepare_semantic_node_matches_kernel,
        "prepare_semantic_argmax_node_matches_kernel": (
            prepare_semantic_argmax_node_matches_kernel
        ),
        "prepare_semantic_block_summary_kernel": prepare_semantic_block_summary_kernel,
    }
