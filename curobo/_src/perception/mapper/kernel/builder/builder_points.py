# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU surface sampling shared by Python and the native mapper AOT export."""

from __future__ import annotations

import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.util.warp import warp_func, warp_kernel

POINT_SCAN_GROUP = 64
POINT_SCAN_THREADS = 65536


def make_point_kernels(
    block_size: int,
    *,
    hash_lookup: object,
    sample_voxel: object,
    sample_tsdf_trilinear_observed: object,
    voxel_to_world: object,
    block_key_to_voxel_base: object,
    semantic_feature_node_from_local: object,
) -> dict[str, object]:
    """Build shifted-crossing-square sampling; no mesh or CPU geometry buffers."""
    BS = wp.constant(block_size)
    BV = wp.constant(block_size**3)
    GROUP = wp.constant(POINT_SCAN_GROUP)
    THREADS = wp.constant(POINT_SCAN_THREADS)

    @warp_kernel(f"set_point_query_bs{block_size}")
    def set_point_query_kernel(
        query: wp.array(dtype=wp.int32),
        slot: int,
        semantic: int,
        selection: int,
        seed: int,
        argmax: int,
        exclude: int,
        cosine: int,
    ) -> None:
        query[0] = slot
        query[1] = semantic
        query[2] = selection
        query[3] = seed
        query[4] = argmax
        query[5] = exclude
        query[6] = cosine

    @warp_kernel(f"set_point_target_class_bs{block_size}")
    def set_point_target_class_kernel(target: wp.array(dtype=wp.float32), label: int) -> None:
        i = wp.tid()
        target[i] = 1.0 if i == label else 0.0

    @warp_func(f"point_neighbor_bs{block_size}")
    def neighbor(tsdf: BlockSparseTSDFWarp, pool: int, local: int, axis: int) -> wp.vec2i:
        xyz = wp.vec3i(local % BS, local // BS % BS, local // (BS * BS))
        xyz[axis] += 1
        other = pool
        if xyz[axis] == BS:
            xyz[axis] = 0
            key = wp.vec3i(
                tsdf.block_coords[3 * pool],
                tsdf.block_coords[3 * pool + 1],
                tsdf.block_coords[3 * pool + 2],
            )
            key[axis] += 1
            other = hash_lookup(tsdf.hash_table, key[0], key[1], key[2], tsdf.hash_capacity)
        return wp.vec2i(other, xyz[2] * BS * BS + xyz[1] * BS + xyz[0])

    @warp_func(f"point_owner_keep_bs{block_size}")
    def owner_keep(
        tsdf: BlockSparseTSDFWarp,
        target: wp.array(dtype=wp.float32),
        selection: wp.array2d(dtype=wp.uint8),
        pool: int,
        local: int,
        query: wp.array(dtype=wp.int32),
        minimum_similarity: float,
    ) -> bool:
        if query[2] != 0 and selection[pool, local] == wp.uint8(0):
            return False
        slot = query[0]
        if slot >= 0:
            size = tsdf.instance_grid_size
            if not tsdf.has_instances or size <= 0:
                return False
            x = (local % BS) * size // BS
            y = (local // BS % BS) * size // BS
            z = (local // (BS * BS)) * size // BS
            packed = wp.int32(tsdf.block_instance_state[pool, z * size * size + y * size + x])
            if (packed & 255) != slot:
                return False
        if query[1] != 0:
            node = semantic_feature_node_from_local(
                float(local % BS) + 0.5,
                float(local // BS % BS) + 0.5,
                float(local // (BS * BS)) + 0.5,
            )
            matches = bool(False)
            if wp.float32(tsdf.block_feature_weight[pool, node]) > 1.0e-6:
                dot = float(0.0)
                norm = float(0.0)
                target_norm = float(0.0)
                maximum = float(-wp.inf)
                target_maximum = float(-wp.inf)
                argmax = int(0)
                target_argmax = int(0)
                for channel in range(tsdf.feature_dim):
                    value = wp.float32(tsdf.block_features[pool, node, channel])
                    wanted = target[channel]
                    dot += value * wanted
                    norm += value * value
                    target_norm += wanted * wanted
                    if value > maximum:
                        maximum = value
                        argmax = channel
                    if wanted > target_maximum:
                        target_maximum = wanted
                        target_argmax = channel
                matches = bool(True)
                if query[4] != 0:
                    matches = argmax == target_argmax
                if query[6] != 0:
                    matches = (
                        matches
                        and norm > 0.0
                        and target_norm > 0.0
                        and dot >= minimum_similarity * wp.sqrt(norm * target_norm)
                    )
            if query[5] != 0:
                matches = not matches
            if not matches:
                return False
        return True

    @warp_kernel(f"count_point_edges_bs{block_size}")
    def count_point_edges_kernel(
        tsdf: BlockSparseTSDFWarp,
        target: wp.array(dtype=wp.float32),
        selection: wp.array2d(dtype=wp.uint8),
        query: wp.array(dtype=wp.int32),
        minimum_weight: float,
        minimum_similarity: float,
        flags: wp.array2d(dtype=wp.uint8),
        counts: wp.array(dtype=wp.int32),
    ) -> None:
        # Persistent grid avoids launching max_blocks*block_voxels threads when
        # only a small part of the capacity is allocated. Counts are zeroed by caller.
        tid = wp.tid()
        for index in range(tid, tsdf.num_allocated[0] * BV, THREADS):
            pool = index // BV
            local = index % BV
            flags[pool, local] = wp.uint8(0)
            if tsdf.block_to_hash_slot[pool] < 0:
                continue
            a = sample_voxel(tsdf, pool, local, minimum_weight)
            if a[1] <= 0.0 or not wp.isfinite(a[0]):
                continue
            base = block_key_to_voxel_base(
                tsdf.block_coords[3 * pool],
                tsdf.block_coords[3 * pool + 1],
                tsdf.block_coords[3 * pool + 2],
            )
            grid = base + wp.vec3i(local % BS, local // BS % BS, local // (BS * BS))
            if (
                grid[0] < 0
                or grid[1] < 0
                or grid[2] < 0
                or grid[0] >= tsdf.grid_W
                or grid[1] >= tsdf.grid_H
                or grid[2] >= tsdf.grid_D
            ):
                continue
            bits = int(0)
            count = int(0)
            for axis in range(3):
                limit = wp.vec3i(tsdf.grid_W, tsdf.grid_H, tsdf.grid_D)
                if grid[axis] + 1 >= limit[axis]:
                    continue
                nb = neighbor(tsdf, pool, local, axis)
                if nb[0] < 0:
                    continue
                b = sample_voxel(tsdf, nb[0], nb[1], minimum_weight)
                if b[1] <= 0.0 or not wp.isfinite(b[0]) or ((a[0] <= 0.0) == (b[0] <= 0.0)):
                    continue
                t = a[0] / (a[0] - b[0])
                owner = wp.vec2i(pool, local)
                if t > 0.5:
                    owner = nb
                if not owner_keep(
                    tsdf, target, selection, owner[0], owner[1], query, minimum_similarity
                ):
                    continue
                bits |= 1 << axis
                count += 1
            flags[pool, local] = wp.uint8(bits)
            if count != 0:
                wp.atomic_add(counts, index // GROUP, count)

    @warp_kernel(f"sample_point_edges_bs{block_size}")
    def sample_point_edges_kernel(
        tsdf: BlockSparseTSDFWarp,
        flags: wp.array2d(dtype=wp.uint8),
        offsets: wp.array(dtype=wp.int32),
        query: wp.array(dtype=wp.int32),
        minimum_weight: float,
        proposals: wp.array2d(dtype=wp.float32),
        sources: wp.array2d(dtype=wp.int32),
        valid: wp.array(dtype=wp.int32),
    ) -> None:
        i = wp.tid()
        valid[i] = 0
        groups = offsets.shape[0] - 1
        total = offsets[groups]
        if total == 0:
            return
        rng = wp.rand_init(query[3], i)
        rank = wp.randi(rng, 0, total)
        lo = int(0)
        hi = groups
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if offsets[mid] <= rank:
                lo = mid
            else:
                hi = mid
        rank -= offsets[lo]
        edge_index = int(-1)
        axis = int(0)
        for j in range(GROUP):
            index = lo * GROUP + j
            if index >= tsdf.max_blocks * BV:
                break
            bits = wp.int32(flags[index // BV, index % BV])
            number = (bits & 1) + ((bits >> 1) & 1) + ((bits >> 2) & 1)
            if rank >= number:
                rank -= number
                continue
            edge_index = index
            for a in range(3):
                if (bits & (1 << a)) != 0:
                    if rank == 0:
                        axis = a
                        break
                    rank -= 1
            break
        if edge_index < 0:
            return
        pool = edge_index // BV
        local = edge_index % BV
        nb = neighbor(tsdf, pool, local, axis)
        if nb[0] < 0:
            return
        sdf_a = sample_voxel(tsdf, pool, local, minimum_weight)
        sdf_b = sample_voxel(tsdf, nb[0], nb[1], minimum_weight)
        t = sdf_a[0] / (sdf_a[0] - sdf_b[0])
        base = block_key_to_voxel_base(
            tsdf.block_coords[3 * pool],
            tsdf.block_coords[3 * pool + 1],
            tsdf.block_coords[3 * pool + 2],
        )
        point = voxel_to_world(base + wp.vec3i(local % BS, local // BS % BS, local // (BS * BS)))
        for a in range(3):
            if a == axis:
                point[a] += t * tsdf.voxel_size
            else:
                point[a] += (wp.randf(rng) - 0.5) * tsdf.voxel_size
        sdf = sample_tsdf_trilinear_observed(tsdf, point, minimum_weight)
        if sdf[1] <= 0.0 or not wp.isfinite(sdf[0]) or wp.abs(sdf[0]) >= 0.5 * tsdf.voxel_size:
            return
        valid[i] = 1
        for a in range(3):
            proposals[i, a] = point[a]
        owner = wp.vec2i(pool, local)
        if t > 0.5:
            owner = nb
        sources[i, 0] = owner[0]
        sources[i, 1] = owner[1]

    @warp_kernel(f"compact_surface_points_bs{block_size}")
    def compact_surface_points_kernel(
        proposals: wp.array2d(dtype=wp.float32),
        sources: wp.array2d(dtype=wp.int32),
        valid: wp.array(dtype=wp.int32),
        offsets: wp.array(dtype=wp.int32),
        positions: wp.array2d(dtype=wp.float32),
        source_voxels: wp.array2d(dtype=wp.int32),
        count: wp.array(dtype=wp.int32),
    ) -> None:
        i = wp.tid()
        if i == 0:
            count[0] = offsets[proposals.shape[0]]
        if valid[i] == 0:
            return
        dst = offsets[i]
        for a in range(3):
            positions[dst, a] = proposals[i, a]
        source_voxels[dst, 0] = sources[i, 0]
        source_voxels[dst, 1] = sources[i, 1]

    return {
        "set_point_query_kernel": set_point_query_kernel,
        "set_point_target_class_kernel": set_point_target_class_kernel,
        "count_point_edges_kernel": count_point_edges_kernel,
        "sample_point_edges_kernel": sample_point_edges_kernel,
        "compact_surface_points_kernel": compact_surface_points_kernel,
    }
