# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp data structures and functions for obstacle collision kernels."""

# Third Party
import warp as wp

# =============================================================================
# Collision Activation Function
# =============================================================================


@wp.func
def apply_collision_activation(dist: wp.float32, eta: wp.float32) -> wp.vec2:
    """Apply smooth activation function for collision cost.

    Uses quadratic near boundary, linear beyond for C1 continuity.

    Args:
        dist: Penetration distance (positive = collision).
        eta: Activation distance threshold.

    Returns:
        vec2(cost, grad_scale) where:
        - cost: Smoothed collision cost
        - grad_scale: Gradient scaling factor for chain rule
    """
    if dist <= 0.0:
        return wp.vec2(0.0, 0.0)

    if dist > eta:
        # Linear region: cost = dist - 0.5 * eta
        cost = dist - 0.5 * eta
        grad_scale = 1.0
    else:
        # Quadratic region: cost = 0.5 * dist^2 / eta
        cost = 0.5 * dist * dist / eta
        grad_scale = dist / eta

    return wp.vec2(cost, grad_scale)


# =============================================================================
# Helper Structs
# =============================================================================


@wp.struct
class SphereQueryData:
    """Sphere query data."""

    center: wp.vec3  # Sphere center in world frame
    radius: wp.float32  # Sphere radius
    radius_adjusted: wp.float32  # Radius + activation distance


@wp.struct
class CollisionSample:
    """Activated collision contribution for one local-frame SDF sample."""

    penetration: wp.float32  # Positive when the sphere overlaps the activation band
    cost: wp.float32  # Unweighted activated collision cost
    gradient: wp.vec3  # Unweighted activated gradient in the obstacle local frame


# =============================================================================
# Helper Functions
# =============================================================================


@wp.func
def load_sphere_query(
    spheres: wp.array(dtype=wp.vec4),
    idx: wp.int32,
    eta: wp.float32,
) -> SphereQueryData:
    """Load sphere query from array.

    Args:
        spheres: Sphere array with (x, y, z, radius).
        idx: Flat index into spheres array.
        eta: Activation distance.

    Returns:
        SphereQueryData with center, radius, and adjusted radius.
    """
    s = spheres[idx]
    query = SphereQueryData()
    query.center = wp.vec3(s[0], s[1], s[2])
    query.radius = s[3]
    query.radius_adjusted = s[3] + eta
    return query


@wp.func
def evaluate_collision_sample(
    sdf_result: wp.vec4,
    radius_adjusted: wp.float32,
    eta: wp.float32,
) -> CollisionSample:
    """Evaluate one local SDF sample and return its activated contribution."""
    sample = CollisionSample()
    sample.penetration = -sdf_result[0] + radius_adjusted
    sample.cost = wp.float32(0.0)
    sample.gradient = wp.vec3(0.0, 0.0, 0.0)

    if sample.penetration > 0.0:
        activation_result = apply_collision_activation(sample.penetration, eta)
        sample.cost = activation_result[0]
        sample.gradient = activation_result[1] * wp.vec3(
            sdf_result[1], sdf_result[2], sdf_result[3]
        )

    return sample


@wp.func
def accumulate_collision(
    sph_flat_idx: wp.int32,
    cost: wp.float32,
    grad: wp.vec3,
    distance: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
):
    """Atomically accumulate collision cost and gradient."""
    wp.atomic_add(distance, sph_flat_idx, cost)
    wp.atomic_add(gradient, sph_flat_idx * 4 + 0, grad[0])
    wp.atomic_add(gradient, sph_flat_idx * 4 + 1, grad[1])
    wp.atomic_add(gradient, sph_flat_idx * 4 + 2, grad[2])
