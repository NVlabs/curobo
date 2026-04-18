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
- 2D parallelization: threads = num_spheres × max_n_obstacles
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
# Module list is centralized in curobo._src.geom.data.OBSTACLE_SDF_MODULES.
#
# How it works:
# - wp.func uses scope_locals.get(func.func.__name__) to find existing Functions
# - By keeping variables named "is_obs_enabled" and "compute_sdf_with_grad" in scope,
#   subsequent wp.func calls detect the existing Function and add overloads
from importlib import import_module
from typing import Any

# Third Party
import warp as wp

from curobo._src.geom.collision.wp_collision_common import (
    accumulate_collision,
    apply_collision_activation,
    load_sphere_query,
)
from curobo._src.geom.data import OBSTACLE_SDF_MODULES

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


# =============================================================================
# Generic Swept-Sphere Collision Kernel
# =============================================================================


@wp.kernel(enable_backward=False)
def swept_sphere_obstacle_collision_kernel(
    # Obstacle set (generic - uses Warp function overloading)
    obs_set: Any,
    # Sphere data
    spheres: wp.array(dtype=wp.vec4),  # [batch * horizon * num_spheres]
    # Parameters
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    env_query_idx: wp.array(dtype=wp.int32),
    # Outputs (pre-zeroed)
    distance: wp.array(dtype=wp.float32),  # [batch * horizon * num_spheres]
    gradient: wp.array(dtype=wp.float32),  # [batch * horizon * num_spheres * 4]
    # Dimensions
    batch_size: wp.int32,
    horizon: wp.int32,
    num_spheres: wp.int32,
    max_n_obs: wp.int32,
    use_multi_env: wp.uint8,
):
    """Compute swept-sphere collision with 2D parallelization.

    Sweeps toward previous and next timesteps in the trajectory.
    Uses adaptive stepping based on penetration distance.
    The number of sweep steps is fixed at compile time via SWEEP_STEPS constant.

    Thread layout: tid = sph_flat_idx * max_n_obs + obs_local_idx

    Args:
        obs_set: Obstacle set (CuboidDataWarp, MeshDataWarp, or VoxelDataWarp).
        spheres: Sphere positions and radii for all timesteps.
        weight: Collision cost weight.
        activation_distance: Distance threshold for collision activation.
        env_query_idx: Environment index per batch element.
        distance: Output distance buffer (pre-zeroed).
        gradient: Output gradient buffer (pre-zeroed).
        batch_size: Number of batch elements.
        horizon: Trajectory horizon.
        num_spheres: Number of spheres per timestep.
        max_n_obs: Maximum obstacles per environment.
        use_multi_env: Whether to use batch-specific environments.
    """
    tid = wp.tid()

    # 2D decode: (sphere, obstacle)
    sph_flat_idx = tid / max_n_obs
    obs_local_idx = tid - sph_flat_idx * max_n_obs

    total_spheres = batch_size * horizon * num_spheres
    if sph_flat_idx >= total_spheres:
        return



    # Decode batch/horizon indices
    b_idx = sph_flat_idx / (horizon * num_spheres)
    h_idx = (sph_flat_idx - b_idx * horizon * num_spheres) / num_spheres

    # Get environment index
    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[b_idx]


    # Check if obstacle exists and is enabled
    if not is_obs_enabled(obs_set, env_idx, obs_local_idx):
        return

    # Copy struct from param space to local stack to prevent nvcc
    # from generating cvta.to.local on the param pointer
    obs = obs_set

    # Load parameters
    eta = activation_distance[0]
    w = weight[0]

    # Load sphere - extract radius_adjusted as scalar so query struct can die early
    query = load_sphere_query(spheres, sph_flat_idx, eta)
    if query.radius < 0.0:
        return
    radius_adjusted = query.radius_adjusted

    # Load obstacle transform ONCE for all sweep queries
    inv_t = load_obstacle_transform(obs, env_idx, obs_local_idx)

    # Accumulators in local frame (apply weight and transform at end)
    cost_sum = wp.float32(0.0)
    grad_sum_local = wp.vec3(0.0, 0.0, 0.0)

    # Transform current sphere center to local frame.
    # After this, query.center is no longer needed - sweep distances
    # are computed in local frame (rigid transforms preserve distances).
    local_current = wp.transform_point(inv_t, query.center)

    # Process current timestep
    sdf_result = compute_local_sdf_with_grad(obs, env_idx, obs_local_idx, local_current)
    penetration = -sdf_result[0] + radius_adjusted

    if penetration > 0.0:
        activation_result = apply_collision_activation(penetration, eta)
        cost_sum += activation_result[0]
        grad_sum_local += activation_result[1] * wp.vec3(sdf_result[1], sdf_result[2], sdf_result[3])

    # Sweep toward h-1 (previous timestep)
    # Interpolation and distance are computed in local frame to:
    # 1. Avoid per-step quaternion rotations (lerp in local space)
    # 2. Free query.center registers (rigid transforms preserve distances)
    if h_idx > 0 and SWEEP_STEPS > 0:
        prev_sphere = spheres[sph_flat_idx - num_spheres]
        local_prev = wp.transform_point(
            inv_t, wp.vec3(prev_sphere[0], prev_sphere[1], prev_sphere[2])
        )
        half_dist = wp.length(local_prev - local_current) * 0.5
        inv_half_dist = 1.0 / wp.max(half_dist, 0.001)
        jump = wp.float32(0.0)
        for _ in range(SWEEP_STEPS):
            if jump >= half_dist:
                break
            t = 1.0 - 0.5 * jump * inv_half_dist
            local_pt = t * local_current + (1.0 - t) * local_prev

            sdf = compute_local_sdf(obs, env_idx, obs_local_idx, local_pt)
            penetration = -sdf + radius_adjusted

            if penetration > 0.0:
                sdf_result = compute_local_sdf_with_grad(obs, env_idx, obs_local_idx, local_pt)
                activation_result = apply_collision_activation(penetration, eta)
                cost_sum += activation_result[0]
                grad_sum_local += activation_result[1] * wp.vec3(sdf_result[1], sdf_result[2], sdf_result[3])
                jump += penetration
            else:
                if -penetration >= 1000.0:
                    jump += radius_adjusted
                else:
                    jump += wp.max(-penetration, radius_adjusted)

    # Sweep toward h+1 (next timestep)
    if h_idx < horizon - 1 and SWEEP_STEPS > 0:
        next_sphere = spheres[sph_flat_idx + num_spheres]
        local_next = wp.transform_point(
            inv_t, wp.vec3(next_sphere[0], next_sphere[1], next_sphere[2])
        )
        half_dist = wp.length(local_next - local_current) * 0.5
        inv_half_dist = 1.0 / wp.max(half_dist, 0.001)
        jump = wp.float32(0.0)
        for _ in range(SWEEP_STEPS):
            if jump >= half_dist:
                break
            t = 1.0 - 0.5 * jump * inv_half_dist
            local_pt = t * local_current + (1.0 - t) * local_next

            sdf = compute_local_sdf(obs, env_idx, obs_local_idx, local_pt)
            penetration = -sdf + radius_adjusted

            if penetration > 0.0:
                sdf_result = compute_local_sdf_with_grad(obs, env_idx, obs_local_idx, local_pt)
                activation_result = apply_collision_activation(penetration, eta)
                cost_sum += activation_result[0]
                grad_sum_local += activation_result[1] * wp.vec3(sdf_result[1], sdf_result[2], sdf_result[3])
                jump += penetration
            else:
                if -penetration >= 1000.0:
                    jump += radius_adjusted
                else:
                    jump += wp.max(-penetration, radius_adjusted)

    # Transform accumulated gradient to world frame ONCE and apply weight
    if cost_sum > 0.0:
        fwd_t = wp.transform_inverse(inv_t)
        grad_world = wp.transform_vector(fwd_t, grad_sum_local)
        accumulate_collision(sph_flat_idx, w * cost_sum, w * grad_world, distance, gradient)
