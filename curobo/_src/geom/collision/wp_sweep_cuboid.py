# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-wide pruning and ordered pair evaluation for swept cuboid collision."""

from typing import Any

import warp as wp

from curobo._src.geom.collision.wp_sweep_collision_kernel import (
    compute_prepared_swept_collision_contribution,
    speed_metric_from_centers,
)
from curobo._src.geom.data.data_obstacle_bounds import is_swept_sphere_outside_box
from curobo._src.util.warp import warp_constant_suffix, warp_kernel


@wp.func(inline=False)
def sweep_staged_cuboid(
    dims: wp.vec3,
    env_idx: wp.int32,
    obs_local_idx: wp.int32,
    inv_t: wp.transform,
    local_current: wp.vec3,
    local_previous: wp.vec3,
    local_following: wp.vec3,
    has_previous: wp.bool,
    has_following: wp.bool,
    radius_adjusted: wp.float32,
    eta: wp.float32,
) -> wp.vec4:
    """Keep the transformed points rounded before the sweep distance calculation."""
    return compute_prepared_swept_collision_contribution(
        dims,
        env_idx,
        obs_local_idx,
        inv_t,
        local_current,
        local_previous,
        local_following,
        has_previous,
        has_following,
        radius_adjusted,
        eta,
    )


@wp.struct
class StateBounds:
    lower: wp.vec3
    upper: wp.vec3


@wp.struct
class ObstacleGeometry:
    center: wp.vec3
    extent: wp.vec3
    inv_position: wp.vec3
    inv_rotation: wp.quat
    dims: wp.vec3
    layers: wp.uint32


@wp.func
def merge_state_bounds(a: StateBounds, b: StateBounds) -> StateBounds:
    out = StateBounds()
    out.lower = wp.min(a.lower, b.lower)
    out.upper = wp.max(a.upper, b.upper)
    return out


@wp.struct
class SphereQuery:
    current: wp.vec3
    previous: wp.vec3
    following: wp.vec3
    radius: wp.float32
    reach: wp.float32
    mask: wp.uint32


@wp.func(inline=True)
def evaluate_query_obstacle(
    query: SphereQuery,
    geom: ObstacleGeometry,
    has_previous: wp.bool,
    has_following: wp.bool,
    eta: wp.float32,
    env: wp.int32,
    obstacle: wp.int32,
) -> wp.vec4:
    if geom.layers & query.mask == wp.uint32(0):
        return wp.vec4(0.0)
    current = query.current
    previous = query.previous
    following = query.following
    radius = query.radius
    query_reach = query.reach
    dx = wp.abs(current[0] - geom.center[0]) - geom.extent[0] - query_reach
    dy = wp.abs(current[1] - geom.center[1]) - geom.extent[1] - query_reach
    dz = wp.abs(current[2] - geom.center[2]) - geom.extent[2] - query_reach
    if wp.max(wp.max(dx, dy), wp.max(dz, 0.0)) > 0.0:
        return wp.vec4(0.0)
    inv_t = wp.transform(geom.inv_position, geom.inv_rotation)
    local_current = wp.transform_point(inv_t, current)
    local_previous = local_current
    local_following = local_current
    if has_previous:
        local_previous = wp.transform_point(inv_t, previous)
    if has_following:
        local_following = wp.transform_point(inv_t, following)
    if is_swept_sphere_outside_box(
        local_current, local_previous, local_following, wp.vec3(0.0), 0.5 * geom.dims, radius
    ):
        return wp.vec4(0.0)
    return sweep_staged_cuboid(
        geom.dims,
        env,
        obstacle,
        inv_t,
        local_current,
        local_previous,
        local_following,
        has_previous,
        has_following,
        radius,
        eta,
    )


def get_swept_cuboid_kernel(
    max_n_obs: int,
    enable_speed_metric: bool,
    horizon: int,
    num_spheres: int,
    block_dim: int,
) -> Any:
    """Share survivors across a block; use otherwise idle lanes to evaluate long lists."""
    team_width = 1
    if num_spheres * 4 <= block_dim * 3:
        team_width = 4
    team_spheres = block_dim // team_width
    team_phases = (min(num_spheres, block_dim) + team_spheres - 1) // team_spheres
    last_spheres = min(num_spheres, block_dim) - (team_phases - 1) * team_spheres
    last_width = block_dim // (1 << (last_spheres - 1).bit_length())
    team_shift = team_width.bit_length() - 1
    last_shift = last_width.bit_length() - 1
    occupancy_threads = 256 if team_width > 1 else 512
    max_obstacles = wp.constant(wp.int32(max_n_obs))
    use_speed_metric = wp.constant(wp.uint8(int(enable_speed_metric)))
    fixed_horizon = wp.constant(wp.int32(horizon))
    fixed_num_spheres = wp.constant(wp.int32(num_spheres))
    fixed_block_dim = wp.constant(wp.int32(block_dim))
    suffix = warp_constant_suffix(max_n_obs, enable_speed_metric, horizon, num_spheres, block_dim)

    @warp_kernel(
        f"swept_cuboid_{suffix}",
        enable_backward=False,
        launch_bounds=(block_dim, max(1, occupancy_threads // block_dim)),
        module_options={"fast_math": True, "block_dim": block_dim},
    )
    def tiled_swept_collision_kernel(
        obs_set: Any,
        spheres: wp.array(dtype=wp.vec4),
        weight: wp.array(dtype=wp.float32),
        activation_distance: wp.array(dtype=wp.float32),
        speed_dt: wp.array(dtype=wp.float32),
        env_query_idx: wp.array(dtype=wp.int32),
        distance: wp.array(dtype=wp.float32),
        gradient: wp.array(dtype=wp.float32),
        use_multi_env: wp.uint8,
        overwrite_output: wp.uint8,
        sum_distance: wp.uint8,
        obstacle_layers: wp.array(dtype=wp.uint32),
        sphere_collision_mask: wp.array2d(dtype=wp.uint32),
    ):
        state, lane = wp.tid()
        b = state / fixed_horizon
        h = state - b * fixed_horizon
        env = int(0)
        if use_multi_env == wp.uint8(1):
            env = env_query_idx[b]
        mask_env = int(0)
        if sphere_collision_mask.shape[0] > 1:
            mask_env = env
        valid = (
            env >= 0
            and env < obs_set.num_envs
            and mask_env >= 0
            and mask_env < sphere_collision_mask.shape[0]
        )
        eta = activation_distance[0]
        weight_value = weight[0]
        safe_env = int(0)
        obstacle_count = int(0)
        if valid:
            safe_env = env
            obstacle_count = wp.max(0, wp.min(max_obstacles, obs_set.n_per_env[env]))
        base = safe_env * obs_set.max_n
        has_previous = h > 0
        has_following = h < fixed_horizon - 1
        lane_cost = float(0.0)
        # All lanes execute every collective, including tails and sphere groups.
        for sphere_round in range(wp.static((num_spheres + block_dim - 1) // block_dim)):
            sphere_start = sphere_round * fixed_block_dim
            sphere_lane = sphere_start + lane
            sphere_idx = state * fixed_num_spheres + sphere_lane
            sphere = wp.vec4(0.0, 0.0, 0.0, -1.0)
            mask = wp.uint32(0)
            current = wp.vec3(0.0)
            previous = current
            following = current
            radius = float(0.0)
            query_reach = float(0.0)
            if sphere_lane < fixed_num_spheres:
                sphere = spheres[sphere_idx]
                if valid:
                    mask = sphere_collision_mask[mask_env, sphere_lane]
                current = wp.vec3(sphere[0], sphere[1], sphere[2])
                previous = current
                following = current
                reach = float(0.0)
                if has_previous:
                    previous_sphere = spheres[sphere_idx - fixed_num_spheres]
                    previous = wp.vec3(previous_sphere[0], previous_sphere[1], previous_sphere[2])
                    delta = previous - current
                    reach = 0.5 * wp.max(
                        wp.abs(delta[0]), wp.max(wp.abs(delta[1]), wp.abs(delta[2]))
                    )
                if has_following:
                    following_sphere = spheres[sphere_idx + fixed_num_spheres]
                    following = wp.vec3(
                        following_sphere[0], following_sphere[1], following_sphere[2]
                    )
                    delta = following - current
                    reach = wp.max(
                        reach,
                        0.5 * wp.max(wp.abs(delta[0]), wp.max(wp.abs(delta[1]), wp.abs(delta[2]))),
                    )
                radius = sphere[3] + eta
                query_reach = radius + reach
                if sphere[3] < 0.0:
                    mask = wp.uint32(0)
            query = SphereQuery()
            query.current = current
            query.previous = previous
            query.following = following
            query.radius = radius
            query.reach = query_reach
            query.mask = mask
            bounds = StateBounds()
            bounds.lower = wp.vec3(wp.inf)
            bounds.upper = wp.vec3(-wp.inf)
            if sphere_lane < fixed_num_spheres and mask != wp.uint32(0):
                if (
                    wp.isfinite(current[0])
                    and wp.isfinite(current[1])
                    and wp.isfinite(current[2])
                    and wp.isfinite(query_reach)
                ):
                    r = wp.vec3(wp.abs(query_reach))
                    bounds.lower = current - r
                    bounds.upper = current + r
                else:
                    bounds.lower = wp.vec3(-wp.inf)
                    bounds.upper = wp.vec3(wp.inf)
            state_box = wp.tile_reduce(merge_state_bounds, wp.tile(bounds, preserve_type=True))[0]
            total = wp.vec4(0.0)
            # Chunking retains original obstacle order without a capacity limit.
            for obstacle_chunk in range((obstacle_count + fixed_block_dim - 1) / fixed_block_dim):
                obstacle_start = obstacle_chunk * fixed_block_dim
                obs_local = obstacle_start + lane
                box_idx = base + obs_local
                candidate = int(0)
                geometry = ObstacleGeometry()
                if obs_local < obstacle_count:
                    if (
                        obs_set.enable[box_idx] == wp.uint8(1)
                        and state_box.lower[0] <= state_box.upper[0]
                    ):
                        geometry.center = wp.vec3(
                            obs_set.world_bounds[box_idx, 0],
                            obs_set.world_bounds[box_idx, 1],
                            obs_set.world_bounds[box_idx, 2],
                        )
                        geometry.extent = wp.vec3(
                            obs_set.world_bounds[box_idx, 4],
                            obs_set.world_bounds[box_idx, 5],
                            obs_set.world_bounds[box_idx, 6],
                        )
                        candidate = 1
                        center = geometry.center
                        ext = geometry.extent
                        if (
                            wp.isfinite(center[0])
                            and wp.isfinite(center[1])
                            and wp.isfinite(center[2])
                            and wp.isfinite(ext[0])
                            and wp.isfinite(ext[1])
                            and wp.isfinite(ext[2])
                        ):
                            ext = wp.vec3(wp.abs(ext[0]), wp.abs(ext[1]), wp.abs(ext[2]))
                            separated = False
                            for axis in range(3):
                                separation = wp.max(
                                    state_box.lower[axis] - (center[axis] + ext[axis]),
                                    (center[axis] - ext[axis]) - state_box.upper[axis],
                                )
                                scale = (
                                    1.0
                                    + wp.abs(center[axis])
                                    + ext[axis]
                                    + wp.max(
                                        wp.abs(state_box.lower[axis]),
                                        wp.abs(state_box.upper[axis]),
                                    )
                                )
                                # Allow for rounding at large world coordinates.
                                slack = 7.62939453125e-6 * scale
                                if separation > slack:
                                    separated = True
                            if separated:
                                candidate = 0
                prefix = wp.tile_scan_inclusive(wp.tile(candidate))
                active_count = prefix[fixed_block_dim - 1]
                if active_count > 0:
                    slot = wp.max(prefix[lane] - 1, 0)
                    compact = wp.tile_zeros(
                        shape=(wp.static(block_dim),), dtype=int, storage="shared"
                    )
                    wp.tile_scatter_add(compact, slot, lane, candidate != 0, atomic=False)
                    if candidate != 0:
                        geometry.layers = obstacle_layers[box_idx]
                        geometry.inv_position = wp.vec3(
                            obs_set.inv_pose[box_idx, 0],
                            obs_set.inv_pose[box_idx, 1],
                            obs_set.inv_pose[box_idx, 2],
                        )
                        geometry.inv_rotation = wp.quat(
                            obs_set.inv_pose[box_idx, 4],
                            obs_set.inv_pose[box_idx, 5],
                            obs_set.inv_pose[box_idx, 6],
                            obs_set.inv_pose[box_idx, 3],
                        )
                        geometry.dims = wp.vec3(
                            obs_set.dims[box_idx, 0],
                            obs_set.dims[box_idx, 1],
                            obs_set.dims[box_idx, 2],
                        )
                    geometry_tile = wp.tile(geometry, preserve_type=True)
                    # Across each team, lanes evaluate adjacent boxes for the same sphere.
                    # Its owner then sums the staged values in scene order.
                    if wp.static(team_width > 1) and active_count >= 8:
                        queries = wp.tile(query, preserve_type=True)
                        sphere_phase = int(0)
                        while sphere_phase < wp.static(team_phases):
                            phase_shift = int(wp.static(team_shift))
                            if sphere_phase == wp.static(team_phases - 1):
                                phase_shift = wp.static(last_shift)
                            phase_width = 1 << phase_shift
                            obstacle_lane_id = lane & (phase_width - 1)
                            query_index = sphere_phase * wp.static(team_spheres) + (
                                lane >> phase_shift
                            )
                            team_query = queries[query_index]
                            obstacle_phase = int(0)
                            while (obstacle_phase << phase_shift) < active_count:
                                pair_index = (obstacle_phase << phase_shift) + obstacle_lane_id
                                pair_value = wp.vec4(0.0)
                                if pair_index < active_count and team_query.mask != wp.uint32(0):
                                    box_lane = compact[pair_index]
                                    geom = geometry_tile[box_lane]
                                    pair_value = evaluate_query_obstacle(
                                        team_query,
                                        geom,
                                        has_previous,
                                        has_following,
                                        eta,
                                        env,
                                        obstacle_start + box_lane,
                                    )
                                contributions = wp.tile(pair_value)
                                if (
                                    lane >= sphere_phase * wp.static(team_spheres)
                                    and lane < (sphere_phase + 1) * wp.static(team_spheres)
                                    and sphere_lane < fixed_num_spheres
                                ):
                                    local_query = lane - sphere_phase * wp.static(team_spheres)
                                    obstacles_in_wave = wp.min(
                                        phase_width, active_count - (obstacle_phase << phase_shift)
                                    )
                                    for obstacle_lane in range(obstacles_in_wave):
                                        element = (local_query << phase_shift) + obstacle_lane
                                        total += wp.vec4(
                                            contributions[0, element],
                                            contributions[1, element],
                                            contributions[2, element],
                                            contributions[3, element],
                                        )
                                obstacle_phase += 1
                            sphere_phase += 1
                    else:
                        if sphere_lane < fixed_num_spheres and valid and mask != wp.uint32(0):
                            for k in range(active_count):
                                box_lane = compact[k]
                                geom = geometry_tile[box_lane]
                                total += evaluate_query_obstacle(
                                    query,
                                    geom,
                                    has_previous,
                                    has_following,
                                    eta,
                                    env,
                                    obstacle_start + box_lane,
                                )
            if sphere_lane < fixed_num_spheres:
                value = weight_value * total
                if (
                    value[0] > 0.0
                    and use_speed_metric == wp.uint8(1)
                    and has_previous
                    and has_following
                ):
                    value = speed_metric_from_centers(
                        value, previous, current, following, speed_dt
                    )
                if sphere[3] < 0.0:
                    value = wp.vec4(0.0)
                if not valid:
                    value = wp.vec4(wp.inf, 0.0, 0.0, 0.0)
                if sum_distance == wp.uint8(1):
                    lane_cost += value[0]
                elif overwrite_output == wp.uint8(1):
                    distance[sphere_idx] = value[0]
                elif value[0] > 0.0:
                    distance[sphere_idx] += value[0]
                gradient_offset = sphere_idx * 4
                if overwrite_output == wp.uint8(1):
                    gradient[gradient_offset] = value[1]
                    gradient[gradient_offset + 1] = value[2]
                    gradient[gradient_offset + 2] = value[3]
                    gradient[gradient_offset + 3] = 0.0
                elif value[0] > 0.0:
                    gradient[gradient_offset] += value[1]
                    gradient[gradient_offset + 1] += value[2]
                    gradient[gradient_offset + 2] += value[3]
        if sum_distance == wp.uint8(1):
            cost = wp.tile_sum(wp.tile(lane_cost))[0]
            if lane == 0:
                if overwrite_output == wp.uint8(1):
                    distance[state] = cost
                else:
                    distance[state] += cost

    return tiled_swept_collision_kernel
