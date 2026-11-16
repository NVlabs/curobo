# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tiled Warp kernel for sphere-obstacle collision detection.

Each block owns one rollout state. Its lanes traverse the state's spheres,
write per-sphere gradients directly, and optionally reduce sphere costs within
the block. Warp resolves the obstacle type through function overloading.
"""

# Standard Library
from functools import lru_cache
from typing import Any

# Third Party
import warp as wp

from curobo._src.geom.collision.wp_collision_common import (
    evaluate_collision_sample,
    load_sphere_query,
)
from curobo._src.geom.data.data_obstacle_query import (
    compute_local_sdf_with_grad,
    is_obs_enabled,
    load_obstacle_transform,
)
from curobo._src.util.warp import warp_constant_suffix, warp_kernel

_SUPPORTED_STATE_BLOCK_DIMS = (32, 64, 128, 256, 512, 1024)


@lru_cache(maxsize=None)
def get_tiled_collision_kernel(
    max_n_obs: int,
    horizon: int,
    num_spheres: int,
    block_dim: int,
) -> Any:
    """Return the tiled collision kernel for any sphere count.

    Each block owns one rollout state. A lane processes all sphere indices
    congruent to its lane index, so sphere counts may exceed the block size.
    Sphere gradients are written directly and lane-local costs are reduced once
    per state.
    """
    if max_n_obs < 1:
        raise ValueError("max_n_obs must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if num_spheres < 1:
        raise ValueError("num_spheres must be positive")
    if block_dim not in _SUPPORTED_STATE_BLOCK_DIMS:
        raise ValueError("block_dim must be one of 32, 64, 128, 256, 512, or 1024")

    max_obstacles = wp.constant(wp.int32(max_n_obs))
    fixed_horizon = wp.constant(wp.int32(horizon))
    fixed_num_spheres = wp.constant(wp.int32(num_spheres))
    fixed_block_dim = wp.constant(wp.int32(block_dim))
    min_blocks_per_sm = max(1, 512 // block_dim)
    suffix = warp_constant_suffix(max_n_obs, horizon, num_spheres, block_dim)

    @warp_kernel(
        f"tiled_collision_{suffix}",
        enable_backward=False,
        launch_bounds=(block_dim, min_blocks_per_sm),
        module_options={"fast_math": True},
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
            and mask_env_idx >= 0
            and mask_env_idx < sphere_collision_mask.shape[0]
        )

        eta = activation_distance[0]
        w = weight[0]
        lane_cost = wp.float32(0.0)
        sphere_lane = lane
        while sphere_lane < fixed_num_spheres:
            sph_flat_idx = state_idx * fixed_num_spheres + sphere_lane
            query = load_sphere_query(spheres, sph_flat_idx, eta)
            contribution = wp.vec4(0.0, 0.0, 0.0, 0.0)
            if query.radius >= 0.0 and valid_env:
                obs = obs_set
                for obs_local_idx in range(max_obstacles):
                    if not valid_env:
                        continue
                    if not is_obs_enabled(obs, env_idx, obs_local_idx):
                        continue
                    if (
                        obstacle_layers[env_idx * obs.max_n + obs_local_idx]
                        & sphere_collision_mask[mask_env_idx, sphere_lane]
                    ) == wp.uint32(0):
                        continue
                    inv_t = load_obstacle_transform(obs, env_idx, obs_local_idx)
                    local_pt = wp.transform_point(inv_t, query.center)
                    local_result = compute_local_sdf_with_grad(
                        obs,
                        env_idx,
                        obs_local_idx,
                        local_pt,
                        query.radius_adjusted,
                    )
                    sample = evaluate_collision_sample(local_result, query.radius_adjusted, eta)
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
            sphere_lane += fixed_block_dim

        if sum_distance == wp.uint8(1):
            state_cost = wp.tile_sum(wp.tile(lane_cost))[0]
            if lane == 0:
                if overwrite_output == wp.uint8(1):
                    distance[state_idx] = state_cost
                else:
                    distance[state_idx] += state_cost

    return tiled_collision_kernel
