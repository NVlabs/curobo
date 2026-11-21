# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discrete sphere–cuboid collision with conservative world-space pruning."""

from functools import lru_cache
from typing import Any

import warp as wp

from curobo._src.geom.collision.wp_collision_common import (
    SphereQueryData,
    evaluate_collision_sample,
    load_sphere_query,
)
from curobo._src.geom.data.data_cuboid import cuboid_distance_gradient
from curobo._src.geom.data.data_obstacle_query import load_obstacle_transform
from curobo._src.util.warp import warp_constant_suffix, warp_kernel


@wp.func
def first_obstacle(bits: wp.uint64) -> wp.int32:
    # Portable integer bisection, valid for any nonzero 64-bit candidate mask.
    index = wp.int32(0)
    part = bits
    step = wp.int32((part & wp.uint64(4294967295)) == wp.uint64(0)) * 32
    index += step
    part = part >> wp.uint64(step)
    step = wp.int32((part & wp.uint64(65535)) == wp.uint64(0)) * 16
    index += step
    part = part >> wp.uint64(step)
    step = wp.int32((part & wp.uint64(255)) == wp.uint64(0)) * 8
    index += step
    part = part >> wp.uint64(step)
    step = wp.int32((part & wp.uint64(15)) == wp.uint64(0)) * 4
    index += step
    part = part >> wp.uint64(step)
    step = wp.int32((part & wp.uint64(3)) == wp.uint64(0)) * 2
    index += step
    part = part >> wp.uint64(step)
    index += wp.int32((part & wp.uint64(1)) == wp.uint64(0))
    return index


@wp.func
def popcount64(bits: wp.uint64) -> wp.int32:
    value = bits - ((bits >> wp.uint64(1)) & wp.uint64(6148914691236517205))
    value = (value & wp.uint64(3689348814741910323)) + (
        (value >> wp.uint64(2)) & wp.uint64(3689348814741910323)
    )
    value = (value + (value >> wp.uint64(4))) & wp.uint64(1085102592571150095)
    return wp.int32((value * wp.uint64(72340172838076673)) >> wp.uint64(56))


@wp.struct
class BroadObstacle:
    center_scale: wp.vec4
    extent: wp.vec3
    layer_bits: wp.uint32
    inv_t: wp.transform
    full_dims: wp.vec3


@wp.func
def union_min(a: wp.spatial_vector, b: wp.spatial_vector) -> wp.spatial_vector:
    return wp.spatial_vector(
        wp.min(a[0], b[0]),
        wp.min(a[1], b[1]),
        wp.min(a[2], b[2]),
        wp.min(a[3], b[3]),
        wp.min(a[4], b[4]),
        wp.min(a[5], b[5]),
    )


@lru_cache(maxsize=None)
def get_cuboid_collision_kernel(
    max_n_obs: int, horizon: int, num_spheres: int, block_dim: int
) -> Any:
    """Stage overlapping cuboids per state and traverse each sphere's survivors in input order."""
    max_obstacles = wp.constant(max_n_obs)
    fixed_horizon = wp.constant(horizon)
    fixed_num_spheres = wp.constant(num_spheres)
    fixed_block_dim = wp.constant(block_dim)
    # Bound shared memory by a few native-width chunks; larger capacities page
    # through the same algorithm instead of requiring capacity-sized storage.
    page_capacity = wp.constant(max(1, min(max_n_obs, 4 * block_dim, 256)))
    scale_storage = wp.constant(((page_capacity + block_dim - 1) // block_dim) * block_dim)
    list_storage = wp.constant(scale_storage)
    groups_per_cohort = wp.constant(max(1, min(4, (num_spheres + block_dim - 1) // block_dim)))
    cohort_width = wp.constant(groups_per_cohort * block_dim)
    accum_type = wp.types.matrix(shape=(groups_per_cohort, 4), dtype=wp.float32)
    prefetch_chunks = wp.constant(scale_storage // block_dim)
    prefetch_type = wp.types.matrix(shape=(prefetch_chunks, 7), dtype=wp.float32)
    flag_words = wp.constant((block_dim + 63) // 64)
    mask_type = wp.types.vector(length=flag_words, dtype=wp.uint64)
    suffix = warp_constant_suffix(max_n_obs, horizon, num_spheres, block_dim)

    @warp_kernel(
        f"tiled_cuboid_{suffix}",
        enable_backward=False,
        launch_bounds=(block_dim, max(1, 512 // block_dim)),
        module_options={"fast_math": True, "block_dim": block_dim, "max_unroll": 4},
    )
    def tiled_collision_kernel(
        obs_set: Any,
        spheres: wp.array(dtype=wp.vec4),
        weight: wp.array(dtype=wp.float32),
        activation_distance: wp.array(dtype=wp.float32),
        env_query_idx: wp.array(dtype=wp.int32),
        distance: wp.array(dtype=wp.float32),
        gradient: wp.array(dtype=wp.float32),
        use_multi_env: wp.uint8,
        overwrite_output: wp.uint8,
        sum_distance: wp.uint8,
        obstacle_layers: wp.array(dtype=wp.uint32),
        sphere_collision_mask: wp.array2d(dtype=wp.uint32),
    ):
        state_idx, lane = wp.tid()
        b_idx = state_idx / fixed_horizon
        env_idx = wp.int32(0)
        if use_multi_env == wp.uint8(1):
            env_idx = env_query_idx[b_idx]
        mask_env_idx = wp.int32(0)
        if sphere_collision_mask.shape[0] > 1:
            mask_env_idx = env_idx
        valid_env = (
            env_idx >= 0
            and env_idx < obs_set.num_envs
            and (mask_env_idx >= 0)
            and (mask_env_idx < sphere_collision_mask.shape[0])
        )
        eta = activation_distance[0]
        w = weight[0]
        # Sphere and scene reads are independent. Keep the first sphere and
        # first-page bounds in registers while the sphere-union reduction runs.
        first_sphere = wp.vec4(0.0, 0.0, 0.0, 0.0)
        if lane < fixed_num_spheres:
            first_sphere = spheres[state_idx * fixed_num_spheres + lane]
        scene_count = wp.int32(0)
        if valid_env:
            scene_count = wp.min(max_obstacles, obs_set.n_per_env[env_idx])
        bounds_env = wp.int32(0)
        if valid_env:
            bounds_env = env_idx
        bounds4 = wp.array(
            ptr=obs_set.world_bounds.ptr, shape=(obs_set.world_bounds.shape[0] * 2,), dtype=wp.vec4
        )
        first_page = prefetch_type()
        for prefetch_chunk in range(prefetch_chunks):
            prefetch_obstacle = prefetch_chunk * fixed_block_dim + lane
            if prefetch_obstacle < scene_count:
                prefetch_flat = bounds_env * obs_set.max_n + prefetch_obstacle
                if obs_set.enable[prefetch_flat] == wp.uint8(1):
                    center = bounds4[prefetch_flat * 2]
                    extent = bounds4[prefetch_flat * 2 + 1]
                    first_page[prefetch_chunk, 0] = center[0]
                    first_page[prefetch_chunk, 1] = center[1]
                    first_page[prefetch_chunk, 2] = center[2]
                    first_page[prefetch_chunk, 3] = extent[0]
                    first_page[prefetch_chunk, 4] = extent[1]
                    first_page[prefetch_chunk, 5] = extent[2]
                    first_page[prefetch_chunk, 6] = 1.0
        # A conservative state-wide union amortizes rejection across all spheres.
        lo_x = wp.float32(wp.inf)
        lo_y = wp.float32(wp.inf)
        lo_z = wp.float32(wp.inf)
        hi_x = wp.float32(-wp.inf)
        hi_y = wp.float32(-wp.inf)
        hi_z = wp.float32(-wp.inf)
        bound_sphere = lane
        while bound_sphere < fixed_num_spheres:
            s = first_sphere
            if bound_sphere != lane:
                s = spheres[state_idx * fixed_num_spheres + bound_sphere]
            if s[3] >= 0.0:
                p = wp.max(s[3] + eta, 0.0) + 2.0e-5 * (
                    1.0 + wp.abs(s[0]) + wp.abs(s[1]) + wp.abs(s[2])
                )
                lo_x = wp.min(lo_x, s[0] - p)
                lo_y = wp.min(lo_y, s[1] - p)
                lo_z = wp.min(lo_z, s[2] - p)
                hi_x = wp.max(hi_x, s[0] + p)
                hi_y = wp.max(hi_y, s[1] + p)
                hi_z = wp.max(hi_z, s[2] + p)
            bound_sphere += fixed_block_dim
        union = wp.tile_reduce(
            union_min,
            wp.tile(wp.spatial_vector(lo_x, lo_y, lo_z, -hi_x, -hi_y, -hi_z), preserve_type=True),
        )[0]
        state_lo_x = union[0]
        state_lo_y = union[1]
        state_lo_z = union[2]
        state_hi_x = -union[3]
        state_hi_y = -union[4]
        state_hi_z = -union[5]
        state_obstacles = wp.tile_full(
            shape=(list_storage,), value=BroadObstacle(), storage="shared"
        )
        lane_cost = wp.float32(0.0)
        any_live = wp.int32(0)
        for cohort_base in range(0, fixed_num_spheres, cohort_width):
            contributions = accum_type()
            cohort_live = wp.int32(0)
            for page_base in range(0, max_obstacles, page_capacity):
                state_live_count = wp.int32(0)
                for bound_base in range(0, scale_storage, fixed_block_dim):
                    bound_lane = page_base + bound_base + lane
                    live = wp.int32(0)
                    row = BroadObstacle()
                    flat = bounds_env * obs_set.max_n + bound_lane
                    if bound_lane < wp.min(page_base + page_capacity, scene_count):
                        enabled = wp.float32(0.0)
                        center = wp.vec4(0.0, 0.0, 0.0, 0.0)
                        extent = wp.vec4(0.0, 0.0, 0.0, 0.0)
                        if page_base == 0:
                            prefetch_chunk = bound_base / fixed_block_dim
                            enabled = first_page[prefetch_chunk, 6]
                            center = wp.vec4(
                                first_page[prefetch_chunk, 0],
                                first_page[prefetch_chunk, 1],
                                first_page[prefetch_chunk, 2],
                                0.0,
                            )
                            extent = wp.vec4(
                                first_page[prefetch_chunk, 3],
                                first_page[prefetch_chunk, 4],
                                first_page[prefetch_chunk, 5],
                                0.0,
                            )
                        elif obs_set.enable[flat] == wp.uint8(1):
                            enabled = 1.0
                            center = bounds4[flat * 2]
                            extent = bounds4[flat * 2 + 1]
                        if enabled == 1.0:
                            scale = (
                                wp.abs(center[0])
                                + wp.abs(center[1])
                                + wp.abs(center[2])
                                + wp.abs(extent[0])
                                + wp.abs(extent[1])
                                + wp.abs(extent[2])
                            )
                            row.center_scale = wp.vec4(center[0], center[1], center[2], scale)
                            row.extent = wp.vec3(extent[0], extent[1], extent[2])
                            pad = 2.0e-5 * scale
                            if not (
                                center[0] + extent[0] + pad < state_lo_x
                                or center[0] - extent[0] - pad > state_hi_x
                                or center[1] + extent[1] + pad < state_lo_y
                                or center[1] - extent[1] - pad > state_hi_y
                                or center[2] + extent[2] + pad < state_lo_z
                                or center[2] - extent[2] - pad > state_hi_z
                            ):
                                live = 1
                    if live != 0:
                        # Input-dependent geometry is staged in this same launch,
                        # once per state survivor rather than per sphere survivor.
                        row.layer_bits = obstacle_layers[flat]
                        row.inv_t = load_obstacle_transform(obs_set, bounds_env, bound_lane)
                        row.full_dims = wp.vec3(
                            obs_set.dims[flat, 0], obs_set.dims[flat, 1], obs_set.dims[flat, 2]
                        )
                    lane_bits = mask_type()
                    for flag_word in range(flag_words):
                        bit = wp.uint64(0)
                        if live != 0 and lane / 64 == flag_word:
                            bit = wp.uint64(1) << wp.uint64(lane % 64)
                        lane_bits[flag_word] = bit
                    masks = wp.tile_sum(wp.tile(lane_bits, preserve_type=True))[0]
                    chunk_count = wp.int32(0)
                    local_rank = wp.int32(0)
                    for flag_word in range(flag_words):
                        bits = masks[flag_word]
                        word_count = popcount64(bits)
                        chunk_count += word_count
                        if live != 0:
                            if flag_word < lane / 64:
                                local_rank += word_count
                            elif flag_word == lane / 64:
                                lower_bits = (wp.uint64(1) << wp.uint64(lane % 64)) - wp.uint64(1)
                                local_rank += popcount64(bits & lower_bits) + 1
                    if chunk_count > 0:
                        cohort_live = 1
                        any_live = 1
                        list_destination = state_live_count + local_rank - 1
                        wp.tile_scatter_masked(state_obstacles, list_destination, row, live != 0)
                    state_live_count += chunk_count
                for sphere_group in range(groups_per_cohort):
                    sphere_lane = cohort_base + sphere_group * fixed_block_dim + lane
                    if sphere_lane < fixed_num_spheres and valid_env and state_live_count > 0:
                        sph_flat_idx = state_idx * fixed_num_spheres + sphere_lane
                        query = SphereQueryData()
                        if cohort_base == 0 and sphere_group == 0:
                            query.center = wp.vec3(
                                first_sphere[0], first_sphere[1], first_sphere[2]
                            )
                            query.radius = first_sphere[3]
                            query.radius_adjusted = first_sphere[3] + eta
                        else:
                            query = load_sphere_query(spheres, sph_flat_idx, eta)
                        if query.radius >= 0.0:
                            contribution = wp.vec4(
                                contributions[sphere_group, 0],
                                contributions[sphere_group, 1],
                                contributions[sphere_group, 2],
                                contributions[sphere_group, 3],
                            )
                            obstacle_count = state_live_count
                            sphere_mask = sphere_collision_mask[mask_env_idx, sphere_lane]
                            px = query.center[0]
                            py = query.center[1]
                            pz = query.center[2]
                            point_scale = 1.0 + wp.abs(px) + wp.abs(py) + wp.abs(pz)
                            for word_base in range(0, obstacle_count, 64):
                                candidates = wp.uint64(0)
                                for obstacle_base in range(
                                    word_base, wp.min(word_base + 64, obstacle_count), 4
                                ):
                                    group_mask = wp.uint32(0)
                                    group_bit = wp.uint32(1)
                                    for group_lane in range(4):
                                        list_idx = obstacle_base + group_lane
                                        if list_idx < obstacle_count:
                                            obstacle = state_obstacles[list_idx]
                                            # A miss on any world axis rules out this pair.
                                            bx = obstacle.center_scale[0]
                                            by = obstacle.center_scale[1]
                                            bz = obstacle.center_scale[2]
                                            ex = obstacle.extent[0]
                                            ey = obstacle.extent[1]
                                            ez = obstacle.extent[2]
                                            coordinate_scale = (
                                                point_scale + obstacle.center_scale[3]
                                            )
                                            padding = (
                                                wp.max(query.radius_adjusted, 0.0)
                                                + 1.0e-5 * coordinate_scale
                                            )
                                            separation = wp.max(
                                                wp.abs(px - bx) - ex,
                                                wp.max(wp.abs(py - by) - ey, wp.abs(pz - bz) - ez),
                                            )
                                            if not (separation > padding):
                                                group_mask = group_mask | group_bit
                                        group_bit = group_bit << wp.uint32(1)
                                    candidates = candidates | (
                                        wp.uint64(group_mask)
                                        << wp.uint64(obstacle_base - word_base)
                                    )
                                # Compact independently per sphere, retaining ascending obstacle order.
                                while candidates != wp.uint64(0):
                                    list_idx = word_base + first_obstacle(candidates)
                                    obstacle = state_obstacles[list_idx]
                                    candidates = candidates & (candidates - wp.uint64(1))
                                    if obstacle.layer_bits & sphere_mask != wp.uint32(0):
                                        inv_t = obstacle.inv_t
                                        local_pt = wp.transform_point(inv_t, query.center)
                                        local_result = cuboid_distance_gradient(
                                            obstacle.full_dims,
                                            local_pt,
                                            query.radius_adjusted,
                                            True,
                                        )
                                        sample = evaluate_collision_sample(
                                            local_result, query.radius_adjusted, eta
                                        )
                                        if sample.penetration > 0.0:
                                            grad_world = wp.transform_vector(
                                                wp.transform_inverse(inv_t), sample.gradient
                                            )
                                            contribution += wp.vec4(
                                                sample.cost,
                                                grad_world[0],
                                                grad_world[1],
                                                grad_world[2],
                                            )
                            contributions[sphere_group, 0] = contribution[0]
                            contributions[sphere_group, 1] = contribution[1]
                            contributions[sphere_group, 2] = contribution[2]
                            contributions[sphere_group, 3] = contribution[3]
                if (
                    page_base + page_capacity < max_obstacles
                    or cohort_base + cohort_width < fixed_num_spheres
                ):
                    # A no-write collective is the reader barrier before reuse.
                    wp.tile_scatter_masked(state_obstacles, 0, BroadObstacle(), False)
            # An empty survivor set proves exact +0 outputs for this mode.
            # Other signs, nonfinite weights and accumulation retain the generic path.
            zero_fast = (
                cohort_live == 0
                and valid_env
                and overwrite_output == wp.uint8(1)
                and w > 0.0
                and wp.isfinite(w)
            )
            if zero_fast:
                state_gradient = wp.array(
                    ptr=gradient.ptr + wp.uint64(state_idx) * wp.uint64(fixed_num_spheres * 16),
                    shape=(fixed_num_spheres, 4),
                    dtype=wp.float32,
                )
                zero_gradient = wp.tile_zeros(shape=(cohort_width, 4), dtype=wp.float32)
                wp.tile_store(state_gradient, zero_gradient, offset=(cohort_base, 0))
                if sum_distance != wp.uint8(1):
                    state_distance = wp.array(
                        ptr=distance.ptr + wp.uint64(state_idx) * wp.uint64(fixed_num_spheres * 4),
                        shape=(fixed_num_spheres,),
                        dtype=wp.float32,
                    )
                    zero_distance = wp.tile_zeros(shape=(cohort_width,), dtype=wp.float32)
                    wp.tile_store(state_distance, zero_distance, offset=(cohort_base,))
            else:
                for sphere_group in range(groups_per_cohort):
                    sphere_lane = cohort_base + sphere_group * fixed_block_dim + lane
                    if sphere_lane < fixed_num_spheres:
                        sph_flat_idx = state_idx * fixed_num_spheres + sphere_lane
                        contribution = wp.vec4(
                            contributions[sphere_group, 0],
                            contributions[sphere_group, 1],
                            contributions[sphere_group, 2],
                            contributions[sphere_group, 3],
                        )
                        weighted_contribution = w * contribution
                        if not valid_env:
                            weighted_contribution = wp.vec4(wp.inf, 0.0, 0.0, 0.0)
                        if sum_distance == wp.uint8(1):
                            lane_cost += weighted_contribution[0]
                        elif overwrite_output == wp.uint8(1):
                            distance[sph_flat_idx] = weighted_contribution[0]
                        elif weighted_contribution[0] > 0.0:
                            distance[sph_flat_idx] += weighted_contribution[0]
                        grad_offset = sph_flat_idx * 4
                        if overwrite_output == wp.uint8(1):
                            gradient[grad_offset + 0] = weighted_contribution[1]
                            gradient[grad_offset + 1] = weighted_contribution[2]
                            gradient[grad_offset + 2] = weighted_contribution[3]
                            gradient[grad_offset + 3] = 0.0
                        elif weighted_contribution[0] > 0.0:
                            gradient[grad_offset + 0] += weighted_contribution[1]
                            gradient[grad_offset + 1] += weighted_contribution[2]
                            gradient[grad_offset + 2] += weighted_contribution[3]
        if sum_distance == wp.uint8(1):
            state_cost = wp.float32(0.0)
            empty_overwrite = (
                any_live == 0
                and valid_env
                and overwrite_output == wp.uint8(1)
                and w > 0.0
                and wp.isfinite(w)
            )
            if not empty_overwrite:
                state_cost = wp.tile_sum(wp.tile(lane_cost))[0]
            if lane == 0:
                if overwrite_output == wp.uint8(1):
                    distance[state_idx] = state_cost
                else:
                    distance[state_idx] += state_cost

    return tiled_collision_kernel
