# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Exact campaign winner f6e7687d49ddeccd95f19727f0673e916ae7397cfb1bf152b9b95d9b7d9dcb82.
"""Balanced mesh sweeps for the qualified native trajectory domain."""

from typing import Any, Callable

import warp as wp

from curobo._src.geom.data.data_mesh import MeshDataWarp

wp.init()
_SDF_EPS = 1e-06
SWEEP_STEPS = wp.constant(3)
_SUPPORTED_STATE_BLOCK_DIMS = (32, 64, 128, 256, 512, 1024)


def _set_warp_name(func: Callable[..., Any], name: str) -> Callable[..., Any]:
    func.__name__ = name
    func.__qualname__ = name
    return func


def warp_kernel(name: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    """Register a named Warp kernel with its original compiler options."""

    def _decorate(func: Callable[..., Any]) -> Any:
        kernel = wp.kernel(module="unique", **kwargs)(_set_warp_name(func, func.__name__))
        return _set_warp_name(kernel, name)

    return _decorate


def warp_constant_suffix(*values: object) -> str:
    """Encode compile-time constants into a stable Warp symbol suffix."""
    return repr(values).encode("ascii").hex()


@wp.func
def get_obs_idx(env_idx: wp.int32, local_idx: wp.int32, max_n: wp.int32) -> wp.int32:
    """Return the flattened environment/obstacle array index."""
    return env_idx * max_n + local_idx


@wp.func
def load_inv_position(inv_pose: wp.array2d(dtype=wp.float32), idx: wp.int32) -> wp.vec3:
    """Load xyz from the obstacle inverse-pose array."""
    return wp.vec3(inv_pose[idx, 0], inv_pose[idx, 1], inv_pose[idx, 2])


@wp.func
def load_inv_quat(inv_pose: wp.array2d(dtype=wp.float32), idx: wp.int32) -> wp.quat:
    """Load the inverse quaternion from the obstacle pose array."""
    return wp.quat(inv_pose[idx, 4], inv_pose[idx, 5], inv_pose[idx, 6], inv_pose[idx, 3])


@wp.func
def load_transform_from_inv_pose(
    inv_pose: wp.array2d(dtype=wp.float32), flat_idx: wp.int32
) -> wp.transform:
    """Load the obstacle world-to-local transform."""
    inv_pos = load_inv_position(inv_pose, flat_idx)
    inv_quat = load_inv_quat(inv_pose, flat_idx)
    return wp.transform(inv_pos, inv_quat)


@wp.func
def is_obs_enabled(obs_set: MeshDataWarp, env_idx: wp.int32, local_idx: wp.int32) -> wp.bool:
    """Check the native obstacle count and enable flags."""
    if local_idx >= obs_set.n_per_env[env_idx]:
        return False
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return obs_set.enable[flat_idx] == wp.uint8(1)


@wp.func
def load_obstacle_transform(
    obs_set: MeshDataWarp, env_idx: wp.int32, local_idx: wp.int32
) -> wp.transform:
    """Load one enabled obstacle transform."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return load_transform_from_inv_pose(obs_set.inv_pose, flat_idx)


@wp.func
def compute_swept_sphere_clearance_lower_bound(
    obs_set: MeshDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    current_world: wp.vec3,
    sweep_min_world: wp.vec3,
    sweep_max_world: wp.vec3,
    radius_adjusted: wp.float32,
    sweep_reach: wp.float32,
) -> wp.float32:
    """Reject swept sphere bounds that miss the mesh world bounds."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    center_x = obs_set.world_bounds[flat_idx, 0]
    center_y = obs_set.world_bounds[flat_idx, 1]
    center_z = obs_set.world_bounds[flat_idx, 2]
    extent_x = obs_set.world_bounds[flat_idx, 4]
    extent_y = obs_set.world_bounds[flat_idx, 5]
    extent_z = obs_set.world_bounds[flat_idx, 6]
    if sweep_max_world[0] < center_x - extent_x:
        return center_x - extent_x - sweep_max_world[0]
    if sweep_min_world[0] > center_x + extent_x:
        return sweep_min_world[0] - center_x - extent_x
    if sweep_max_world[1] < center_y - extent_y:
        return center_y - extent_y - sweep_max_world[1]
    if sweep_min_world[1] > center_y + extent_y:
        return sweep_min_world[1] - center_y - extent_y
    if sweep_max_world[2] < center_z - extent_z:
        return center_z - extent_z - sweep_max_world[2]
    if sweep_min_world[2] > center_z + extent_z:
        return sweep_min_world[2] - center_z - extent_z
    return wp.float32(0.0)


@wp.func_native("""
    const wp::Mesh mesh = wp::mesh_get(mesh_id);
    const int root = *mesh.bvh.root;
    const wp::BVHPackedNodeHalf lower = wp::bvh_load_node(mesh.bvh.node_lowers, root);
    const wp::BVHPackedNodeHalf upper = wp::bvh_load_node(mesh.bvh.node_uppers, root);
    return local_pt[0] < lower.x || local_pt[0] > upper.x
        || local_pt[1] < lower.y || local_pt[1] > upper.y
        || local_pt[2] < lower.z || local_pt[2] > upper.z;
""")
def mesh_point_outside_root(mesh_id: wp.uint64, local_pt: wp.vec3) -> wp.bool:
    """Check whether a query point lies outside the mesh BVH root bounds."""
    ...


@wp.func_native("""
    float hit_sign = 0.0f;
    return wp::mesh_query_ray_closest_sign(wp::mesh_get(mesh_id), local_pt, direction, hit_sign)
        && hit_sign < 0.0f;
""")
def mesh_ray_inside_vote(mesh_id: wp.uint64, local_pt: wp.vec3, direction: wp.vec3) -> wp.bool:
    """Return the signed closest-ray vote used by the mesh query."""
    ...


@wp.func
def compute_local_sdf_with_deferred_grad(
    mesh_id: wp.uint64, max_distance: wp.float32, local_pt: wp.vec3
) -> wp.vec4:
    """Return signed distance and an unnormalized local gradient."""
    result = wp.mesh_query_point_no_sign(mesh_id, local_pt, max_distance)
    if not result.result:
        return wp.vec4(max_distance, 0.0, 0.0, 0.0)
    inside = False
    if not mesh_point_outside_root(mesh_id, local_pt):
        inside_x = mesh_ray_inside_vote(mesh_id, local_pt, wp.vec3(1.0, 0.0, 0.0))
        inside_y = mesh_ray_inside_vote(mesh_id, local_pt, wp.vec3(0.0, 1.0, 0.0))
        inside = inside_x
        if inside_x != inside_y:
            inside = mesh_ray_inside_vote(mesh_id, local_pt, wp.vec3(0.0, 0.0, 1.0))
    sign = wp.float32(1.0)
    if inside:
        sign = wp.float32(-1.0)
    cl_pt = wp.mesh_eval_position(mesh_id, result.face, result.u, result.v)
    delta = cl_pt - local_pt
    dis_length = wp.length(delta)
    signed_dist = dis_length * sign
    grad_local = -delta
    return wp.vec4(signed_dist, grad_local[0], grad_local[1], grad_local[2])


@wp.func
def compute_activated_sdf_contribution(
    sdf_result: wp.vec4, radius_adjusted: wp.float32, eta: wp.float32
) -> wp.vec4:
    """Activate one distance sample and normalize only active gradients."""
    penetration = -sdf_result[0] + radius_adjusted
    if penetration <= 0.0:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    grad_local = wp.vec3(0.0, 0.0, 0.0)
    dis_length = wp.abs(sdf_result[0])
    if dis_length > _SDF_EPS:
        grad_local = wp.vec3(sdf_result[1], sdf_result[2], sdf_result[3]) / dis_length
    cost = wp.float32(0.0)
    gradient_scale = wp.float32(0.0)
    if penetration > eta:
        cost = penetration - 0.5 * eta
        gradient_scale = 1.0
    else:
        cost = 0.5 * penetration * penetration / eta
        gradient_scale = penetration / eta
    return wp.vec4(
        cost,
        gradient_scale * grad_local[0],
        gradient_scale * grad_local[1],
        gradient_scale * grad_local[2],
    )


@wp.func
def compute_sweep_jump(penetration: wp.float32, radius_adjusted: wp.float32) -> wp.float32:
    """Preserve the native adaptive step and missing-query behavior."""
    if penetration > 0.0:
        return penetration
    clearance = -penetration
    if clearance >= 1000.0:
        return radius_adjusted
    return wp.max(clearance, radius_adjusted)


@wp.func
def compute_sweep_segment_contribution(
    mesh_id: wp.uint64,
    max_distance: wp.float32,
    local_current: wp.vec3,
    local_neighbor: wp.vec3,
    radius_adjusted: wp.float32,
    eta: wp.float32,
    current_penetration: wp.float32,
    current_contribution: wp.vec4,
) -> wp.vec4:
    """Accumulate one half-sweep while reusing its current-point query."""
    contribution = wp.vec4(0.0, 0.0, 0.0, 0.0)
    segment = local_neighbor - local_current
    half_dist = wp.length(segment) * 0.5
    if half_dist > 0.0:
        contribution += current_contribution
        jump = compute_sweep_jump(current_penetration, radius_adjusted)
        sweep_direction = 0.5 / wp.max(half_dist, 0.001) * segment
        for _ in range(SWEEP_STEPS - 1):
            if jump >= half_dist:
                break
            local_pt = local_current + jump * sweep_direction
            sdf_result = compute_local_sdf_with_deferred_grad(mesh_id, max_distance, local_pt)
            penetration = -sdf_result[0] + radius_adjusted
            contribution += compute_activated_sdf_contribution(sdf_result, radius_adjusted, eta)
            jump += compute_sweep_jump(penetration, radius_adjusted)
    return contribution


@wp.func
def compute_prepared_swept_collision_contribution(
    obs_set: MeshDataWarp,
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
    """Accumulate both half-sweeps and rotate the gradient to world space."""
    flat_idx = get_obs_idx(env_idx, obs_local_idx, obs_set.max_n)
    mesh_id = obs_set.mesh_ids[flat_idx]
    bounding_box_size = wp.vec3(
        obs_set.dims[flat_idx, 0], obs_set.dims[flat_idx, 1], obs_set.dims[flat_idx, 2]
    )
    max_distance = wp.length(bounding_box_size) * 0.5
    max_distance = wp.max(max_distance, radius_adjusted)
    current_sdf_result = compute_local_sdf_with_deferred_grad(mesh_id, max_distance, local_current)
    current_penetration = -current_sdf_result[0] + radius_adjusted
    contribution = compute_activated_sdf_contribution(current_sdf_result, radius_adjusted, eta)
    if has_previous and SWEEP_STEPS > 0:
        contribution += compute_sweep_segment_contribution(
            mesh_id,
            max_distance,
            local_current,
            local_previous,
            radius_adjusted,
            eta,
            current_penetration,
            contribution,
        )
    if has_following and SWEEP_STEPS > 0:
        current_contribution = compute_activated_sdf_contribution(
            current_sdf_result, radius_adjusted, eta
        )
        contribution += compute_sweep_segment_contribution(
            mesh_id,
            max_distance,
            local_current,
            local_following,
            radius_adjusted,
            eta,
            current_penetration,
            current_contribution,
        )
    if contribution[0] > 0.0:
        grad_sum_local = wp.vec3(contribution[1], contribution[2], contribution[3])
        inv_quat = wp.transform_get_rotation(inv_t)
        grad_world = wp.quat_rotate(wp.quat_inverse(inv_quat), grad_sum_local)
        return wp.vec4(contribution[0], grad_world[0], grad_world[1], grad_world[2])
    return wp.vec4(0.0, 0.0, 0.0, 0.0)


@wp.func
def apply_speed_metric_to_contribution(
    contribution: wp.vec4,
    spheres: wp.array(dtype=wp.vec4),
    speed_dt: wp.array(dtype=wp.float32),
    sph_flat_idx: wp.int32,
    h_idx: wp.int32,
    horizon: wp.int32,
    num_spheres: wp.int32,
) -> wp.vec4:
    """Apply the native trajectory speed metric to the weighted contribution."""
    if h_idx == 0 or h_idx >= horizon - 1 or contribution[0] <= 0.0:
        return contribution
    prev_sphere = spheres[sph_flat_idx - num_spheres]
    current_sphere = spheres[sph_flat_idx]
    next_sphere = spheres[sph_flat_idx + num_spheres]
    dt = wp.max(speed_dt[0], 1e-06)
    prev_pt = wp.vec3(prev_sphere[0], prev_sphere[1], prev_sphere[2])
    current_pt = wp.vec3(current_sphere[0], current_sphere[1], current_sphere[2])
    next_pt = wp.vec3(next_sphere[0], next_sphere[1], next_sphere[2])
    velocity = 0.5 / dt * (next_pt - prev_pt)
    sphere_speed = wp.length(velocity)
    if sphere_speed < 0.001:
        return contribution
    acceleration = 1.0 / (dt * dt) * (prev_pt + next_pt - 2.0 * current_pt)
    normalized_velocity = velocity / sphere_speed
    curvature = acceleration / (sphere_speed * sphere_speed)
    grad = wp.vec3(contribution[1], contribution[2], contribution[3])
    orthogonal_grad = grad - wp.dot(normalized_velocity, grad) * normalized_velocity
    orthogonal_curvature = curvature - wp.dot(normalized_velocity, curvature) * normalized_velocity
    grad = sphere_speed * (orthogonal_grad - contribution[0] * orthogonal_curvature)
    return wp.vec4(sphere_speed * contribution[0], grad[0], grad[1], grad[2])


wp.set_module_options({"fast_math": True, "enable_backward": False, "block_dim": 64})

BLOCK_DIM = wp.constant(wp.int32(64))


@wp.kernel(enable_backward=False, launch_bounds=(64, 4))
def obstacle_swept_collision(
    obs_set: MeshDataWarp,
    spheres: wp.array(dtype=wp.vec4),
    activation_distance: wp.array(dtype=wp.float32),
    env_query_idx: wp.array(dtype=wp.int32),
    scratch: wp.array(dtype=wp.vec4),
    use_multi_env: wp.uint8,
    obstacle_layers: wp.array(dtype=wp.uint32),
    sphere_collision_mask: wp.array2d(dtype=wp.uint32),
    horizon: wp.int32,
    num_spheres: wp.int32,
):
    """Write scratch contributions for each state, obstacle, and sphere."""
    state_obstacle_idx, lane = wp.tid()
    state_idx = state_obstacle_idx / obs_set.max_n
    obs_local_idx = state_obstacle_idx - state_idx * obs_set.max_n
    b_idx = state_idx / horizon
    h_idx = state_idx - b_idx * horizon
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
    sphere_lane = lane
    while sphere_lane < num_spheres:
        sph_flat_idx = state_idx * num_spheres + sphere_lane
        sphere = spheres[sph_flat_idx]
        radius_adjusted = sphere[3] + eta
        current_world = wp.vec3(sphere[0], sphere[1], sphere[2])
        previous_world = current_world
        following_world = current_world
        has_previous = h_idx > 0
        has_following = h_idx < horizon - 1
        sweep_reach = wp.float32(0.0)
        sweep_min_world = current_world
        sweep_max_world = current_world
        if has_previous:
            previous_sphere = spheres[sph_flat_idx - num_spheres]
            previous_world = wp.vec3(previous_sphere[0], previous_sphere[1], previous_sphere[2])
            previous_delta = previous_world - current_world
            sweep_reach = 0.5 * wp.max(
                wp.abs(previous_delta[0]),
                wp.max(wp.abs(previous_delta[1]), wp.abs(previous_delta[2])),
            )
            previous_midpoint_world = 0.5 * (current_world + previous_world)
            sweep_min_world = wp.min(sweep_min_world, previous_midpoint_world)
            sweep_max_world = wp.max(sweep_max_world, previous_midpoint_world)
        if has_following:
            following_sphere = spheres[sph_flat_idx + num_spheres]
            following_world = wp.vec3(
                following_sphere[0], following_sphere[1], following_sphere[2]
            )
            following_delta = following_world - current_world
            following_reach = 0.5 * wp.max(
                wp.abs(following_delta[0]),
                wp.max(wp.abs(following_delta[1]), wp.abs(following_delta[2])),
            )
            sweep_reach = wp.max(sweep_reach, following_reach)
            following_midpoint_world = 0.5 * (current_world + following_world)
            sweep_min_world = wp.min(sweep_min_world, following_midpoint_world)
            sweep_max_world = wp.max(sweep_max_world, following_midpoint_world)
        radius_vector = wp.vec3(radius_adjusted, radius_adjusted, radius_adjusted)
        sweep_min_world -= radius_vector
        sweep_max_world += radius_vector
        contribution = wp.vec4(0.0, 0.0, 0.0, 0.0)
        obs = obs_set
        if valid_env:
            if is_obs_enabled(obs, env_idx, obs_local_idx):
                if obstacle_layers[env_idx * obs.max_n + obs_local_idx] & sphere_collision_mask[
                    mask_env_idx, sphere_lane
                ] != wp.uint32(0):
                    clearance_lower_bound = compute_swept_sphere_clearance_lower_bound(
                        obs,
                        env_idx,
                        obs_local_idx,
                        current_world,
                        sweep_min_world,
                        sweep_max_world,
                        radius_adjusted,
                        sweep_reach,
                    )
                    if not (clearance_lower_bound > 0.0):
                        inv_t = load_obstacle_transform(obs, env_idx, obs_local_idx)
                        local_current = wp.transform_point(inv_t, current_world)
                        local_previous = local_current
                        local_following = local_current
                        if has_previous:
                            local_previous = wp.transform_point(inv_t, previous_world)
                        if has_following:
                            local_following = wp.transform_point(inv_t, following_world)
                        contribution = compute_prepared_swept_collision_contribution(
                            obs,
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
        scratch_idx = (state_idx * obs_set.max_n + obs_local_idx) * num_spheres + sphere_lane
        scratch[scratch_idx] = contribution
        sphere_lane += BLOCK_DIM


@wp.kernel(enable_backward=False, launch_bounds=(64, 4))
def reduce_obstacle_contributions(
    obs_set: MeshDataWarp,
    spheres: wp.array(dtype=wp.vec4),
    weight: wp.array(dtype=wp.float32),
    speed_dt: wp.array(dtype=wp.float32),
    env_query_idx: wp.array(dtype=wp.int32),
    scratch: wp.array(dtype=wp.vec4),
    distance: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    use_multi_env: wp.uint8,
    overwrite_output: wp.uint8,
    sum_distance: wp.uint8,
    sphere_collision_mask: wp.array2d(dtype=wp.uint32),
    enable_speed_metric: wp.uint8,
    horizon: wp.int32,
    num_spheres: wp.int32,
):
    """Reduce scratch contributions and write native costs and gradients."""
    state_idx, lane = wp.tid()
    b_idx = state_idx / horizon
    h_idx = state_idx - b_idx * horizon
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
    lane_cost = wp.float32(0.0)
    sphere_lane = lane
    while sphere_lane < num_spheres:
        sph_flat_idx = state_idx * num_spheres + sphere_lane
        sphere = spheres[sph_flat_idx]
        total = wp.vec4(0.0, 0.0, 0.0, 0.0)
        for obs_local_idx in range(obs_set.max_n):
            scratch_idx = (state_idx * obs_set.max_n + obs_local_idx) * num_spheres + sphere_lane
            total += scratch[scratch_idx]
        weighted_contribution = weight[0] * total
        if total[0] > 0.0 and enable_speed_metric == wp.uint8(1):
            weighted_contribution = apply_speed_metric_to_contribution(
                weighted_contribution, spheres, speed_dt, sph_flat_idx, h_idx, horizon, num_spheres
            )
        if sphere[3] < 0.0:
            weighted_contribution = wp.vec4(0.0, 0.0, 0.0, 0.0)
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
        sphere_lane += BLOCK_DIM
    if sum_distance == wp.uint8(1):
        state_cost = wp.tile_sum(wp.tile(lane_cost))[0]
        if lane == 0:
            if overwrite_output == wp.uint8(1):
                distance[state_idx] = state_cost
            else:
                distance[state_idx] += state_cost


KERNELS = {}
max_obstacles_0 = wp.constant(wp.int32(8))
use_speed_metric_0 = wp.constant(wp.uint8(0))
fixed_horizon_0 = wp.constant(wp.int32(81))
fixed_num_spheres_0 = wp.constant(wp.int32(61))
fixed_block_dim_0 = wp.constant(wp.int32(64))


@wp.kernel(enable_backward=False, launch_bounds=(64, 4))
def tiled_swept_collision_0(
    obs_set: MeshDataWarp,
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
    """Compute the fixed-domain native sweep without speed weighting."""
    state_idx, lane = wp.tid()
    b_idx = state_idx / fixed_horizon_0
    h_idx = state_idx - b_idx * fixed_horizon_0
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
    lane_cost = wp.float32(0.0)
    sphere_lane = lane
    while sphere_lane < fixed_num_spheres_0:
        sph_flat_idx = state_idx * fixed_num_spheres_0 + sphere_lane
        sphere = spheres[sph_flat_idx]
        radius_adjusted = sphere[3] + eta
        current_world = wp.vec3(sphere[0], sphere[1], sphere[2])
        previous_world = current_world
        following_world = current_world
        has_previous = h_idx > 0
        has_following = h_idx < fixed_horizon_0 - 1
        sweep_reach = wp.float32(0.0)
        sweep_min_world = current_world
        sweep_max_world = current_world
        if has_previous:
            previous_sphere = spheres[sph_flat_idx - fixed_num_spheres_0]
            previous_world = wp.vec3(previous_sphere[0], previous_sphere[1], previous_sphere[2])
            previous_delta = previous_world - current_world
            sweep_reach = 0.5 * wp.max(
                wp.abs(previous_delta[0]),
                wp.max(wp.abs(previous_delta[1]), wp.abs(previous_delta[2])),
            )
            previous_midpoint_world = 0.5 * (current_world + previous_world)
            sweep_min_world = wp.min(sweep_min_world, previous_midpoint_world)
            sweep_max_world = wp.max(sweep_max_world, previous_midpoint_world)
        if has_following:
            following_sphere = spheres[sph_flat_idx + fixed_num_spheres_0]
            following_world = wp.vec3(
                following_sphere[0], following_sphere[1], following_sphere[2]
            )
            following_delta = following_world - current_world
            following_reach = 0.5 * wp.max(
                wp.abs(following_delta[0]),
                wp.max(wp.abs(following_delta[1]), wp.abs(following_delta[2])),
            )
            sweep_reach = wp.max(sweep_reach, following_reach)
            following_midpoint_world = 0.5 * (current_world + following_world)
            sweep_min_world = wp.min(sweep_min_world, following_midpoint_world)
            sweep_max_world = wp.max(sweep_max_world, following_midpoint_world)
        radius_vector = wp.vec3(radius_adjusted, radius_adjusted, radius_adjusted)
        sweep_min_world -= radius_vector
        sweep_max_world += radius_vector
        total = wp.vec4(0.0, 0.0, 0.0, 0.0)
        obs = obs_set
        for obs_local_idx in range(max_obstacles_0):
            if not valid_env:
                continue
            if not is_obs_enabled(obs, env_idx, obs_local_idx):
                continue
            if obstacle_layers[env_idx * obs.max_n + obs_local_idx] & sphere_collision_mask[
                mask_env_idx, sphere_lane
            ] == wp.uint32(0):
                continue
            clearance_lower_bound = compute_swept_sphere_clearance_lower_bound(
                obs,
                env_idx,
                obs_local_idx,
                current_world,
                sweep_min_world,
                sweep_max_world,
                radius_adjusted,
                sweep_reach,
            )
            if clearance_lower_bound > 0.0:
                continue
            inv_t = load_obstacle_transform(obs, env_idx, obs_local_idx)
            local_current = wp.transform_point(inv_t, current_world)
            local_previous = local_current
            local_following = local_current
            if has_previous:
                local_previous = wp.transform_point(inv_t, previous_world)
            if has_following:
                local_following = wp.transform_point(inv_t, following_world)
            total += compute_prepared_swept_collision_contribution(
                obs,
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
        weighted_contribution = weight[0] * total
        if total[0] > 0.0 and use_speed_metric_0 == wp.uint8(1):
            weighted_contribution = apply_speed_metric_to_contribution(
                weighted_contribution,
                spheres,
                speed_dt,
                sph_flat_idx,
                h_idx,
                fixed_horizon_0,
                fixed_num_spheres_0,
            )
        if sphere[3] < 0.0:
            weighted_contribution = wp.vec4(0.0, 0.0, 0.0, 0.0)
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
        sphere_lane += fixed_block_dim_0
    if sum_distance == wp.uint8(1):
        state_cost = wp.tile_sum(wp.tile(lane_cost))[0]
        if lane == 0:
            if overwrite_output == wp.uint8(1):
                distance[state_idx] = state_cost
            else:
                distance[state_idx] += state_cost


KERNELS[8, False, 81, 61, 64] = tiled_swept_collision_0
max_obstacles_1 = wp.constant(wp.int32(8))
use_speed_metric_1 = wp.constant(wp.uint8(1))
fixed_horizon_1 = wp.constant(wp.int32(81))
fixed_num_spheres_1 = wp.constant(wp.int32(61))
fixed_block_dim_1 = wp.constant(wp.int32(64))


@wp.kernel(enable_backward=False, launch_bounds=(64, 4))
def tiled_swept_collision_1(
    obs_set: MeshDataWarp,
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
    """Compute the fixed-domain native sweep with speed weighting."""
    state_idx, lane = wp.tid()
    b_idx = state_idx / fixed_horizon_1
    h_idx = state_idx - b_idx * fixed_horizon_1
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
    lane_cost = wp.float32(0.0)
    sphere_lane = lane
    while sphere_lane < fixed_num_spheres_1:
        sph_flat_idx = state_idx * fixed_num_spheres_1 + sphere_lane
        sphere = spheres[sph_flat_idx]
        radius_adjusted = sphere[3] + eta
        current_world = wp.vec3(sphere[0], sphere[1], sphere[2])
        previous_world = current_world
        following_world = current_world
        has_previous = h_idx > 0
        has_following = h_idx < fixed_horizon_1 - 1
        sweep_reach = wp.float32(0.0)
        sweep_min_world = current_world
        sweep_max_world = current_world
        if has_previous:
            previous_sphere = spheres[sph_flat_idx - fixed_num_spheres_1]
            previous_world = wp.vec3(previous_sphere[0], previous_sphere[1], previous_sphere[2])
            previous_delta = previous_world - current_world
            sweep_reach = 0.5 * wp.max(
                wp.abs(previous_delta[0]),
                wp.max(wp.abs(previous_delta[1]), wp.abs(previous_delta[2])),
            )
            previous_midpoint_world = 0.5 * (current_world + previous_world)
            sweep_min_world = wp.min(sweep_min_world, previous_midpoint_world)
            sweep_max_world = wp.max(sweep_max_world, previous_midpoint_world)
        if has_following:
            following_sphere = spheres[sph_flat_idx + fixed_num_spheres_1]
            following_world = wp.vec3(
                following_sphere[0], following_sphere[1], following_sphere[2]
            )
            following_delta = following_world - current_world
            following_reach = 0.5 * wp.max(
                wp.abs(following_delta[0]),
                wp.max(wp.abs(following_delta[1]), wp.abs(following_delta[2])),
            )
            sweep_reach = wp.max(sweep_reach, following_reach)
            following_midpoint_world = 0.5 * (current_world + following_world)
            sweep_min_world = wp.min(sweep_min_world, following_midpoint_world)
            sweep_max_world = wp.max(sweep_max_world, following_midpoint_world)
        radius_vector = wp.vec3(radius_adjusted, radius_adjusted, radius_adjusted)
        sweep_min_world -= radius_vector
        sweep_max_world += radius_vector
        total = wp.vec4(0.0, 0.0, 0.0, 0.0)
        obs = obs_set
        for obs_local_idx in range(max_obstacles_1):
            if not valid_env:
                continue
            if not is_obs_enabled(obs, env_idx, obs_local_idx):
                continue
            if obstacle_layers[env_idx * obs.max_n + obs_local_idx] & sphere_collision_mask[
                mask_env_idx, sphere_lane
            ] == wp.uint32(0):
                continue
            clearance_lower_bound = compute_swept_sphere_clearance_lower_bound(
                obs,
                env_idx,
                obs_local_idx,
                current_world,
                sweep_min_world,
                sweep_max_world,
                radius_adjusted,
                sweep_reach,
            )
            if clearance_lower_bound > 0.0:
                continue
            inv_t = load_obstacle_transform(obs, env_idx, obs_local_idx)
            local_current = wp.transform_point(inv_t, current_world)
            local_previous = local_current
            local_following = local_current
            if has_previous:
                local_previous = wp.transform_point(inv_t, previous_world)
            if has_following:
                local_following = wp.transform_point(inv_t, following_world)
            total += compute_prepared_swept_collision_contribution(
                obs,
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
        weighted_contribution = weight[0] * total
        if total[0] > 0.0 and use_speed_metric_1 == wp.uint8(1):
            weighted_contribution = apply_speed_metric_to_contribution(
                weighted_contribution,
                spheres,
                speed_dt,
                sph_flat_idx,
                h_idx,
                fixed_horizon_1,
                fixed_num_spheres_1,
            )
        if sphere[3] < 0.0:
            weighted_contribution = wp.vec4(0.0, 0.0, 0.0, 0.0)
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
        sphere_lane += fixed_block_dim_1
    if sum_distance == wp.uint8(1):
        state_cost = wp.tile_sum(wp.tile(lane_cost))[0]
        if lane == 0:
            if overwrite_output == wp.uint8(1):
                distance[state_idx] = state_cost
            else:
                distance[state_idx] += state_cost


KERNELS[8, True, 81, 61, 64] = tiled_swept_collision_1
