# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Generic Warp kernels for sphere-obstacle collision detection.

This module provides type-generic collision kernels that work with any obstacle
type via Warp function overloading. The kernels use `typing.Any` for the obstacle
set parameter, and Warp dispatches to the correct overloaded functions based on
the struct type at runtime.

Key features:
- Single kernel for all obstacle types (cuboid, mesh, voxel)
- 2D parallelization: threads = num_spheres × max_n_obstacles
- Sparse atomic accumulation: only non-zero costs write to output
- Optimized frame transforms: load once per (sphere, obstacle) pair

Kernels:
- sphere_obstacle_collision_kernel: Regular collision (single timestep)
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
compute_local_sdf_with_grad = None

for _module_path in OBSTACLE_SDF_MODULES:
    _data_module = import_module(_module_path)
    _obs_fn = getattr(_data_module, "is_obs_enabled")
    _transform_fn = getattr(_data_module, "load_obstacle_transform")
    _sdf_fn = getattr(_data_module, "compute_local_sdf_with_grad")

    # Register with wp.func - adds overloads on subsequent iterations
    is_obs_enabled = wp.func(_obs_fn, module=__name__)
    load_obstacle_transform = wp.func(_transform_fn, module=__name__)
    compute_local_sdf_with_grad = wp.func(_sdf_fn, module=__name__)

del _module_path, _data_module, _obs_fn, _transform_fn, _sdf_fn


# =============================================================================
# Generic Sphere-Obstacle Collision Kernel
# =============================================================================


@wp.kernel
def sphere_obstacle_collision_kernel(
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
    """Compute sphere-obstacle collision with 2D parallelization.

    This is a type-generic kernel that works with any obstacle type via
    Warp function overloading. The correct `compute_sdf_with_grad` and
    `is_obs_enabled` functions are selected based on the struct type of `obs_set`.

    Thread layout: tid = sph_flat_idx * max_n_obs + obs_local_idx

    Args:
        obs_set: Obstacle set (CuboidDataWarp, MeshDataWarp, or VoxelDataWarp).
        spheres: Sphere positions and radii.
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

    # Decode batch index and get environment
    b_idx = sph_flat_idx / (horizon * num_spheres)
    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[b_idx]

    # Check if obstacle exists and is enabled
    if not is_obs_enabled(obs_set, env_idx, obs_local_idx):
        return

    # Load parameters
    eta = activation_distance[0]
    w = weight[0]

    # Load sphere
    query = load_sphere_query(spheres, sph_flat_idx, eta)
    if query.radius < 0.0:
        return

    # Load obstacle transform
    inv_t = load_obstacle_transform(obs_set, env_idx, obs_local_idx)

    # Transform sphere center to local frame
    local_pt = wp.transform_point(inv_t, query.center)

    # Compute local SDF and gradient
    local_result = compute_local_sdf_with_grad(obs_set, env_idx, obs_local_idx, local_pt)

    # Only compute forward transform and accumulate when in collision
    penetration = -local_result[0] + query.radius_adjusted
    if penetration > 0.0:
        fwd_t = wp.transform_inverse(inv_t)
        grad_local = wp.vec3(local_result[1], local_result[2], local_result[3])
        grad_world = wp.transform_vector(fwd_t, grad_local)
        activation_result = apply_collision_activation(penetration, eta)
        accumulate_collision(
            sph_flat_idx,
            w * activation_result[0],
            w * activation_result[1] * grad_world,
            distance,
            gradient,
        )
