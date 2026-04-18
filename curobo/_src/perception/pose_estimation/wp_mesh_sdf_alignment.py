# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Warp kernels for mesh surface distance-based pose alignment.

Two-pass kernel design for reduced register pressure:
- Pass 1 (mesh_surface_distance_query_kernel): Query mesh surface distance, store distances/gradients
- Pass 2 (jacobian_reduce_kernel): Compute Jacobian, block-reduce J^T@J, J^T@r

This split improves GPU occupancy by separating the high-register BVH traversal
from the tile-based reduction operations.

Uses unsigned surface distance (not signed SDF) since observed points from depth
sensors are always external to the mesh surface.

Used by SDFPoseDetector for object pose estimation.
"""

from __future__ import annotations

import warp as wp


@wp.func
def quat_from_wxyz_array(q: wp.array(dtype=wp.float32)) -> wp.quat:
    """Convert quaternion array [w, x, y, z] to Warp quaternion."""
    return wp.quat(q[1], q[2], q[3], q[0])


@wp.func
def vec3_from_array(v: wp.array(dtype=wp.float32)) -> wp.vec3:
    """Convert float array to vec3."""
    return wp.vec3(v[0], v[1], v[2])


@wp.func
def transform_point_inverse(
    position: wp.vec3,
    quat: wp.quat,
    point: wp.vec3,
) -> wp.vec3:
    """Transform point from world frame to object frame.

    Computes: p_object = R^T @ (p_world - t)

    Args:
        position: Object position in world frame.
        quat: Object orientation as quaternion.
        point: Point in world frame.

    Returns:
        Point in object frame.
    """
    translated = point - position
    quat_inv = wp.quat_inverse(quat)
    return wp.quat_rotate(quat_inv, translated)


@wp.func
def transform_vector(
    quat: wp.quat,
    vec: wp.vec3,
) -> wp.vec3:
    """Transform vector by rotation (no translation).

    Computes: v_world = R @ v_object

    Args:
        quat: Object orientation as quaternion.
        vec: Vector in object frame.

    Returns:
        Vector in world frame.
    """
    return wp.quat_rotate(quat, vec)


# =============================================================================
# Pass 1: Mesh Surface Distance Query Kernel
# =============================================================================


@wp.kernel(enable_backward=False)
def mesh_surface_distance_query_kernel(
    # Input: observed points in world frame
    observed_points: wp.array(dtype=wp.vec3),
    n_points: wp.int32,
    # Object pose (transforms from object frame to world frame)
    obj_position: wp.array(dtype=wp.float32),  # [3]
    obj_quaternion: wp.array(dtype=wp.float32),  # [4] wxyz
    # Mesh
    mesh_id: wp.uint64,
    max_distance: wp.float32,
    distance_threshold: wp.float32,
    # Outputs: intermediate buffers for Pass 2
    distance_values: wp.array(dtype=wp.float32),  # [N] unsigned distance to surface
    gradients_world: wp.array(dtype=wp.vec3),  # [N] gradient in world frame
    valid_mask: wp.array(dtype=wp.int32),  # [N] 1 if valid, 0 otherwise
):
    """Pass 1: Query mesh surface distance and store intermediate results.

    Separates BVH traversal from Jacobian computation to reduce register pressure.
    Each thread queries the unsigned distance to mesh surface for one observed point.

    Uses unsigned distance since observed points from depth sensors are always
    external to the mesh surface.
    """
    tid = wp.tid()
    if tid >= n_points:
        return

    # Get observed point in world frame
    p_world = observed_points[tid]

    # Get object pose
    obj_pos = vec3_from_array(obj_position)
    obj_quat = quat_from_wxyz_array(obj_quaternion)

    # Transform point to mesh (object) frame
    p_mesh = transform_point_inverse(obj_pos, obj_quat, p_world)

    # Query mesh for closest point (unsigned distance)
    result = wp.mesh_query_point_no_sign(mesh_id, p_mesh, max_distance)

    if result.result:
        # Get closest point on mesh
        closest_pt = wp.mesh_eval_position(mesh_id, result.face, result.u, result.v)

        # Compute unsigned distance
        delta = closest_pt - p_mesh
        dist = wp.length(delta)

        # Skip if too far from surface or too close (numerical stability)
        if dist <= distance_threshold and dist > 1e-8:
            # Gradient in mesh frame: normalized direction from surface to query point
            grad_mesh = delta / dist

            # Transform gradient to world frame
            grad_world = transform_vector(obj_quat, grad_mesh)

            # Store results
            distance_values[tid] = dist
            gradients_world[tid] = grad_world
            valid_mask[tid] = 1
            return


# =============================================================================
# Pass 2: Jacobian Computation and Block Reduction
# =============================================================================


@wp.kernel(enable_backward=False)
def jacobian_reduce_kernel(
    # Input: observed points and intermediate results from Pass 1
    observed_points: wp.array(dtype=wp.vec3),
    distance_values: wp.array(dtype=wp.float32),
    gradients_world: wp.array(dtype=wp.vec3),
    valid_mask: wp.array(dtype=wp.int32),
    n_points: wp.int32,
    # Huber parameters
    use_huber: wp.int32,
    huber_delta: wp.float32,
    # Outputs (accumulated via atomic add)
    JtJ_out: wp.array(dtype=wp.float32),  # [36] flattened 6x6
    Jtr_out: wp.array(dtype=wp.float32),  # [6]
    sum_sq_residuals: wp.array(dtype=wp.float32),  # [1]
    valid_count: wp.array(dtype=wp.int32),  # [1]
):
    """Pass 2: Compute Jacobian and block-reduce to accumulate J^T@J, J^T@r.

    Uses tile operations for efficient block-level reduction.
    Lower register pressure than combined kernel.
    """
    tid = wp.tid()

    # Per-thread accumulators (initialized to zero)
    j0 = float(0.0)
    j1 = float(0.0)
    j2 = float(0.0)
    j3 = float(0.0)
    j4 = float(0.0)
    j5 = float(0.0)
    r = float(0.0)
    valid = wp.int32(0)

    if tid < n_points:
        if valid_mask[tid] != 0:
            # Load intermediate results
            p_world = observed_points[tid]
            dist = distance_values[tid]
            grad_world = gradients_world[tid]

            gx = grad_world[0]
            gy = grad_world[1]
            gz = grad_world[2]

            px = p_world[0]
            py = p_world[1]
            pz = p_world[2]

            # Residual is the unsigned distance to surface
            r = dist

            # Huber weighting for robustness
            huber_scale = float(1.0)
            if use_huber != 0:
                if r > huber_delta:
                    huber_scale = wp.sqrt(huber_delta / r)
                r = r * huber_scale

            # 6-DoF Jacobian: [-gx, -gy, -gz, -(gz*py - gy*pz), ...]
            # gradient points from points towards mesh surface
            j0 = gx * huber_scale
            j1 = gy * huber_scale
            j2 = gz * huber_scale
            j3 = (gz * py - gy * pz) * huber_scale
            j4 = (gx * pz - gz * px) * huber_scale
            j5 = (gy * px - gx * py) * huber_scale

            valid = wp.int32(1)

    # =========================================================================
    # Block reduction using tile operations
    # =========================================================================

    # Create tiles from per-thread scalar values
    t_j0 = wp.tile(j0)
    t_j1 = wp.tile(j1)
    t_j2 = wp.tile(j2)
    t_j3 = wp.tile(j3)
    t_j4 = wp.tile(j4)
    t_j5 = wp.tile(j5)
    t_r = wp.tile(r)
    t_valid = wp.tile(valid)

    # Reduce valid count first
    sum_valid = wp.tile_sum(t_valid)

    # Skip all atomic operations if block has no valid samples
    if sum_valid[0] > 0.0:
        # Reduce sum(r^2)
        sum_r_sq = wp.tile_sum(wp.tile_map(wp.mul, t_r, t_r))
        wp.tile_atomic_add(sum_sq_residuals, sum_r_sq, 0, False)
        wp.tile_atomic_add(valid_count, sum_valid, 0, False)

        # Reduce J^T @ r (6 elements)
        Jtr_0 = wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_r))
        Jtr_1 = wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_r))
        Jtr_2 = wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_r))
        Jtr_3 = wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_r))
        Jtr_4 = wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_r))
        Jtr_5 = wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_r))

        wp.tile_atomic_add(Jtr_out, Jtr_0, 0, False)
        wp.tile_atomic_add(Jtr_out, Jtr_1, 1, False)
        wp.tile_atomic_add(Jtr_out, Jtr_2, 2, False)
        wp.tile_atomic_add(Jtr_out, Jtr_3, 3, False)
        wp.tile_atomic_add(Jtr_out, Jtr_4, 4, False)
        wp.tile_atomic_add(Jtr_out, Jtr_5, 5, False)

        # Reduce J^T @ J (6x6 = 36 elements, flattened row-major)
        # Row 0: indices 0-5
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j0)), 0, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j1)), 1, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j2)), 2, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j3)), 3, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j4)), 4, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j5)), 5, False
        )
        # Row 1: indices 6-11
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j0)), 6, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j1)), 7, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j2)), 8, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j3)), 9, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j4)), 10, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j5)), 11, False
        )
        # Row 2: indices 12-17
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j0)), 12, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j1)), 13, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j2)), 14, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j3)), 15, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j4)), 16, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j5)), 17, False
        )
        # Row 3: indices 18-23
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j0)), 18, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j1)), 19, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j2)), 20, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j3)), 21, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j4)), 22, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j5)), 23, False
        )
        # Row 4: indices 24-29
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j0)), 24, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j1)), 25, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j2)), 26, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j3)), 27, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j4)), 28, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j5)), 29, False
        )
        # Row 5: indices 30-35
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j0)), 30, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j1)), 31, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j2)), 32, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j3)), 33, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j4)), 34, False
        )
        wp.tile_atomic_add(
            JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j5)), 35, False
        )
