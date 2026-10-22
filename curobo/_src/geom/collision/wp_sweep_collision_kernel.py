# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Generic Warp kernels for swept-sphere obstacle collision detection.

This module provides type-generic collision kernels that work with any obstacle
type via Warp function overloading. The kernels use `typing.Any` for the obstacle
set parameter, and Warp dispatches to the correct overloaded functions based on
the struct type at runtime.

Key features:
- Single kernel for all obstacle types (cuboid, mesh, voxel)
- Obstacle-coherent tiles across sphere trajectories
- Sparse atomic accumulation: only non-zero costs write to output
- Optimized frame transforms: load once, accumulate in local frame, transform once

Performance optimization:
- Frame transform is loaded once per (sphere, obstacle) pair
- Gradients are accumulated in obstacle local frame
- Single gradient-to-world transform at the end
- Reduces quaternion operations by ~N for N sweep queries

Kernels:
- swept_sphere_obstacle_collision_kernel: Swept collision (trajectory interpolation)
"""

# Standard Library
# =============================================================================
# Automatic SDF Function Overload Registration
# =============================================================================
#
# Registers SDF function overloads from all obstacle data modules.
# Module list is centralized in curobo._src.geom.data.registry.OBSTACLE_SDF_MODULES.
#
# How it works:
# - wp.func uses scope_locals.get(func.func.__name__) to find existing Functions
# - By keeping variables named "is_obs_enabled" and "compute_sdf_with_grad" in scope,
#   subsequent wp.func calls detect the existing Function and add overloads
from importlib import import_module
from inspect import signature
from typing import Any

# Third Party
import warp as wp

from curobo._src.geom.collision.wp_collision_common import (
    apply_collision_activation,
)
from curobo._src.geom.data.data_mesh import MeshDataWarp
from curobo._src.geom.data.registry import OBSTACLE_SDF_MODULES

# Initialize to None - will be set by first iteration
is_obs_enabled = None
load_obstacle_transform = None
compute_local_sdf = None
compute_local_sdf_with_grad = None


for _module_path in OBSTACLE_SDF_MODULES:
    _data_module = import_module(_module_path)
    _obs_fn = getattr(_data_module, "is_obs_enabled")
    _transform_fn = getattr(_data_module, "load_obstacle_transform")
    _sdf_only_fn = getattr(_data_module, "compute_local_sdf")
    _sdf_fn = getattr(_data_module, "compute_local_sdf_with_grad")

    # Register with wp.func - adds overloads on subsequent iterations
    is_obs_enabled = wp.func(_obs_fn, module=__name__)
    load_obstacle_transform = wp.func(_transform_fn, module=__name__)
    compute_local_sdf = wp.func(_sdf_only_fn, module=__name__)
    compute_local_sdf_with_grad = wp.func(_sdf_fn, module=__name__)

del _module_path, _data_module, _obs_fn, _transform_fn, _sdf_fn


SWEEP_STEPS = wp.constant(3)
SPHERE_TILE_SIZE = wp.constant(128)

# Keep older Warp versions on their supported compiler options.
_kernel_options = {"enable_backward": False}
_kernel_parameters = signature(wp.kernel).parameters
if "launch_bounds" in _kernel_parameters:
    _kernel_options["launch_bounds"] = (128, 6)
if "enable_cuda_smem_spilling" in _kernel_parameters:
    _kernel_options["enable_cuda_smem_spilling"] = True
_mesh_query_options = {}
if "inline" in signature(wp.func).parameters:
    _mesh_query_options["inline"] = False

wp.set_module_options({"enable_backward": False, "block_dim": 128})


@wp.func(**_mesh_query_options)
def mesh_sdf(
    mesh_id: wp.uint64, point: wp.vec3, query_limit: wp.float32, radius: wp.float32
) -> wp.vec4:
    """Query mesh distance and compute a normal only inside the activation band."""
    query = wp.mesh_query_point(mesh_id, point, query_limit)
    if not query.result:
        return wp.vec4(query_limit, 0.0, 0.0, 0.0)
    nearest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    delta = nearest - point
    length = wp.length(delta)
    grad = wp.vec3(0.0, 0.0, 0.0)
    if length > 1.0e-6 and -length * query.sign + radius > 0.0:
        grad = query.sign * delta / length
    return wp.vec4(length * query.sign, grad[0], grad[1], grad[2])


@wp.func
def _sweep_sdf(
    obs_set: Any,
    env_idx: wp.int32,
    obs_idx: wp.int32,
    point: wp.vec3,
    radius: wp.float32,
) -> wp.vec4:
    """Use the registered SDF overload for cuboid and voxel obstacles."""
    return compute_local_sdf_with_grad(obs_set, env_idx, obs_idx, point, radius)


@wp.func
def _sweep_sdf(
    obs_set: MeshDataWarp,
    env_idx: wp.int32,
    obs_idx: wp.int32,
    point: wp.vec3,
    radius: wp.float32,
) -> wp.vec4:
    """Use the outlined mesh query without changing other collision kernels."""
    index = env_idx * obs_set.max_n + obs_idx
    dimensions = wp.vec3(obs_set.dims[index, 0], obs_set.dims[index, 1], obs_set.dims[index, 2])
    query_limit = wp.max(wp.length(dimensions) * 0.5, radius)
    return mesh_sdf(obs_set.mesh_ids[index], point, query_limit, radius)


@wp.func
def collision_term(
    sdf: wp.vec4, radius: wp.float32, eta: wp.float32, accumulated: wp.vec4
) -> wp.vec4:
    """Accumulate one activated sample in the obstacle frame."""
    penetration = -sdf[0] + radius
    if penetration <= 0.0:
        return accumulated
    activation = apply_collision_activation(penetration, eta)
    cost_sum = accumulated[0] + activation[0]
    grad_sum = wp.vec3(accumulated[1], accumulated[2], accumulated[3])
    grad_sum += activation[1] * wp.vec3(sdf[1], sdf[2], sdf[3])
    return wp.vec4(cost_sum, grad_sum[0], grad_sum[1], grad_sum[2])


@wp.func
def sweep_jump(penetration: wp.float32, radius: wp.float32) -> wp.float32:
    """Preserve the adaptive stepping rule, including missing-SDF queries."""
    if penetration > 0.0:
        return penetration
    if -penetration >= 1000.0:
        return radius
    return wp.max(-penetration, radius)


@wp.func
def sweep_segment(
    obs_set: Any,
    env_idx: wp.int32,
    obs_idx: wp.int32,
    local_current: wp.vec3,
    local_adjacent: wp.vec3,
    radius: wp.float32,
    eta: wp.float32,
    current_sdf: wp.vec4,
    first_jump: wp.float32,
    accumulated: wp.vec4,
) -> wp.vec4:
    """Reuse the center query as the first sample, then advance up to two times."""
    half_dist = wp.length(local_adjacent - local_current) * 0.5
    result = accumulated
    if half_dist > 0.0:
        result = collision_term(current_sdf, radius, eta, result)
        jump = first_jump
        inv_half_dist = 1.0 / wp.max(half_dist, 0.001)
        for step in range(SWEEP_STEPS - 1):
            if jump >= half_dist:
                break
            fraction = 1.0 - 0.5 * jump * inv_half_dist
            point = fraction * local_current + (1.0 - fraction) * local_adjacent
            sdf = _sweep_sdf(obs_set, env_idx, obs_idx, point, radius)
            result = collision_term(sdf, radius, eta, result)
            jump += sweep_jump(-sdf[0] + radius, radius)
    return result


@wp.kernel(**_kernel_options)
def swept_sphere_obstacle_collision_kernel(
    obs_set: Any,
    spheres: wp.array(dtype=wp.vec4),
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    env_query_idx: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    batch_size: wp.int32,
    horizon: wp.int32,
    num_spheres: wp.int32,
    max_n_obs: wp.int32,
    use_multi_env: wp.uint8,
):
    """Accumulate swept collision costs for [batch, horizon, sphere] inputs.

    Each full tile evaluates 128 adjacent points on sphere trajectories against
    one obstacle. Reordering threads changes floating-point atomic summation
    order but preserves the native arguments, sample weights, and output layout.
    """
    thread_idx = wp.tid()
    total_spheres = batch_size * horizon * num_spheres
    if total_spheres == 0:
        return
    complete_tiles = total_spheres / SPHERE_TILE_SIZE
    complete_threads = complete_tiles * SPHERE_TILE_SIZE * max_n_obs
    obstacle_idx = wp.int32(0)
    trajectory_idx = wp.int32(0)
    # Adjacent threads traverse time for the same sphere and obstacle.
    if thread_idx < complete_threads:
        tile_idx = thread_idx / SPHERE_TILE_SIZE
        obstacle_idx = tile_idx % max_n_obs
        trajectory_idx = (tile_idx / max_n_obs) * SPHERE_TILE_SIZE + thread_idx % SPHERE_TILE_SIZE
    else:
        # A partial tile still covers every remaining sphere/obstacle pair once.
        tail_size = total_spheres - complete_tiles * SPHERE_TILE_SIZE
        if tail_size == 0:
            return
        tail_idx = thread_idx - complete_threads
        obstacle_idx = tail_idx / tail_size
        trajectory_idx = complete_tiles * SPHERE_TILE_SIZE + tail_idx % tail_size
    if obstacle_idx >= max_n_obs:
        return
    batch_idx = trajectory_idx / (horizon * num_spheres)
    within_batch = trajectory_idx - batch_idx * horizon * num_spheres
    sphere_in_robot = within_batch / horizon
    timestep = within_batch - sphere_in_robot * horizon
    sphere_idx = (batch_idx * horizon + timestep) * num_spheres + sphere_in_robot
    environment = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        environment = env_query_idx[batch_idx]
    if not is_obs_enabled(obs_set, environment, obstacle_idx):
        return
    obs = obs_set
    sphere = spheres[sphere_idx]
    if sphere[3] < 0.0:
        return
    eta = activation_distance[0]
    radius = sphere[3] + eta
    inverse_transform = load_obstacle_transform(obs, environment, obstacle_idx)
    local_current = wp.transform_point(inverse_transform, wp.vec3(sphere[0], sphere[1], sphere[2]))
    current_sdf = _sweep_sdf(obs, environment, obstacle_idx, local_current, radius)
    first_jump = sweep_jump(-current_sdf[0] + radius, radius)
    accumulated = collision_term(current_sdf, radius, eta, wp.vec4(0.0, 0.0, 0.0, 0.0))
    if timestep > 0 and SWEEP_STEPS > 0:
        previous = spheres[sphere_idx - num_spheres]
        local_previous = wp.transform_point(
            inverse_transform, wp.vec3(previous[0], previous[1], previous[2])
        )
        accumulated = sweep_segment(
            obs,
            environment,
            obstacle_idx,
            local_current,
            local_previous,
            radius,
            eta,
            current_sdf,
            first_jump,
            accumulated,
        )
    if timestep < horizon - 1 and SWEEP_STEPS > 0:
        following = spheres[sphere_idx + num_spheres]
        local_following = wp.transform_point(
            inverse_transform, wp.vec3(following[0], following[1], following[2])
        )
        accumulated = sweep_segment(
            obs,
            environment,
            obstacle_idx,
            local_current,
            local_following,
            radius,
            eta,
            current_sdf,
            first_jump,
            accumulated,
        )
    if accumulated[0] > 0.0:
        forward_transform = wp.transform_inverse(inverse_transform)
        grad_world = wp.transform_vector(
            forward_transform, wp.vec3(accumulated[1], accumulated[2], accumulated[3])
        )
        collision_weight = weight[0]
        wp.atomic_add(distance, sphere_idx, collision_weight * accumulated[0])
        wp.atomic_add(gradient, sphere_idx * 4, collision_weight * grad_world[0])
        wp.atomic_add(gradient, sphere_idx * 4 + 1, collision_weight * grad_world[1])
        wp.atomic_add(gradient, sphere_idx * 4 + 2, collision_weight * grad_world[2])
