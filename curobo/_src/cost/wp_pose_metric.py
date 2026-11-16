# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp functions for projected position and rotation metrics."""

# Third Party
import warp as wp


@wp.func
def angular_cotangent_to_wxyz_quaternion_cotangent(
    angular_cotangent: wp.vec3, current_quat: wp.quat
):
    """Map a world-left angular cotangent to cuRobo's wxyz quaternion layout.

    For ``q(delta) = quaternion(Exp(delta) R)``, the quaternion tangent is
    ``0.5 * [delta, 0] * q`` in Warp's xyzw convention. The minimum-norm
    Euclidean cotangent with the same virtual work is therefore
    ``2 * [angular_cotangent, 0] * q / dot(q, q)``. The denominator is one for
    kinematics output, while also matching the forward metric's normalization
    if a caller supplies a scaled quaternion. The return value is reordered to
    the wxyz tensor convention used by cuRobo kinematics.
    """
    angular_quat = wp.quat(angular_cotangent[0], angular_cotangent[1], angular_cotangent[2], 0.0)
    quaternion_cotangent = (
        2.0 * wp.mul(angular_quat, current_quat) / wp.dot(current_quat, current_quat)
    )
    return wp.vec4(
        quaternion_cotangent[3],
        quaternion_cotangent[0],
        quaternion_cotangent[1],
        quaternion_cotangent[2],
    )


@wp.func
def _canonicalize_error_quaternion(error_quat: wp.quat):
    """Normalize a rotation-error quaternion and choose its shortest-arc hemisphere."""
    canonical_quat = wp.normalize(error_quat)
    flip_sign = canonical_quat[3] < 0.0

    # At exactly pi, w does not distinguish q from -q. Resolve the logarithm's
    # unavoidable branch ambiguity lexicographically so both representations
    # still produce the same finite result.
    if canonical_quat[3] == 0.0:
        if canonical_quat[0] != 0.0:
            flip_sign = canonical_quat[0] < 0.0
        elif canonical_quat[1] != 0.0:
            flip_sign = canonical_quat[1] < 0.0
        elif canonical_quat[2] != 0.0:
            flip_sign = canonical_quat[2] < 0.0

    if flip_sign:
        canonical_quat = wp.quat(
            -canonical_quat[0],
            -canonical_quat[1],
            -canonical_quat[2],
            -canonical_quat[3],
        )
    return canonical_quat


@wp.func
def compute_rotation_error(current_quat: wp.quat, target_quat: wp.quat):
    """Compute a shortest-arc quaternion-vector error and its exact Jacobian.

    The returned residual is twice the vector part of the canonicalized error
    quaternion. It agrees with the SO(3) logarithm to first order while avoiding
    transcendental functions. The returned matrix exactly linearizes this
    residual under ``R_current(delta) = Exp(delta) R_current``.
    """
    error_quat = wp.mul(current_quat, wp.quat_inverse(target_quat))
    error_quat = _canonicalize_error_quaternion(error_quat)
    vector = wp.vec3(error_quat[0], error_quat[1], error_quat[2])
    scalar = error_quat[3]
    rotation_error = 2.0 * vector
    linearization = wp.mat33(
        scalar,
        vector[2],
        -vector[1],
        -vector[2],
        scalar,
        vector[0],
        vector[1],
        -vector[0],
        scalar,
    )
    return rotation_error, linearization


@wp.func
def compute_projected_position_metric(
    current_position: wp.vec3,
    target_position: wp.vec3,
    projector: wp.mat33,
    axis_scale: wp.vec3,
    tolerance: wp.float32,
    weight: wp.float32,
):
    """Compute projected position cost, reported distance, and cotangent."""
    position_delta = current_position - target_position
    projected_delta = projector * position_delta
    scaled_delta = wp.cw_mul(axis_scale, projected_delta)
    distance = wp.length(scaled_delta)

    cost = 0.5 * weight * wp.dot(scaled_delta, scaled_delta)
    scaled_gradient = wp.cw_mul(axis_scale, wp.cw_mul(axis_scale, projected_delta))
    gradient = weight * (wp.transpose(projector) * scaled_gradient)

    if distance <= tolerance:
        cost = 0.0
        distance = 0.0
        gradient = wp.vec3(0.0, 0.0, 0.0)
    return cost, distance, gradient


@wp.func
def compute_projected_rotation_metric(
    current_quat: wp.quat,
    target_quat: wp.quat,
    projector: wp.mat33,
    axis_scale: wp.vec3,
    tolerance: wp.float32,
    weight: wp.float32,
):
    """Compute projected rotation cost, reported distance, and angular cotangent."""
    rotation_error, error_linearization = compute_rotation_error(current_quat, target_quat)
    projected_error = projector * rotation_error
    scaled_error = wp.cw_mul(axis_scale, projected_error)
    distance = wp.length(scaled_error)

    cost = weight * wp.dot(scaled_error, scaled_error)
    scaled_gradient = wp.cw_mul(axis_scale, wp.cw_mul(axis_scale, projected_error))
    error_gradient = 2.0 * weight * (wp.transpose(projector) * scaled_gradient)
    angular_gradient = wp.transpose(error_linearization) * error_gradient

    if distance <= tolerance:
        cost = 0.0
        distance = 0.0
        angular_gradient = wp.vec3(0.0, 0.0, 0.0)
    return cost, distance, angular_gradient


@wp.func
def compute_projected_pose_metric(
    current_position: wp.vec3,
    target_position: wp.vec3,
    current_quat: wp.quat,
    target_quat: wp.quat,
    position_projector: wp.mat33,
    rotation_projector: wp.mat33,
    position_axis_scale: wp.vec3,
    rotation_axis_scale: wp.vec3,
    position_tolerance: wp.float32,
    rotation_tolerance: wp.float32,
    position_weight: wp.float32,
    rotation_weight: wp.float32,
):
    """Compute the independent position and rotation parts of a projected pose metric."""
    position_cost, position_distance, position_gradient = compute_projected_position_metric(
        current_position,
        target_position,
        position_projector,
        position_axis_scale,
        position_tolerance,
        position_weight,
    )
    rotation_cost, rotation_distance, rotation_gradient = compute_projected_rotation_metric(
        current_quat,
        target_quat,
        rotation_projector,
        rotation_axis_scale,
        rotation_tolerance,
        rotation_weight,
    )
    return (
        position_cost,
        rotation_cost,
        position_distance,
        rotation_distance,
        position_gradient,
        rotation_gradient,
    )


@wp.func
def compute_quadratic_pose_metric(
    current_position: wp.vec3,
    target_position: wp.vec3,
    current_quat: wp.quat,
    target_quat: wp.quat,
    position_metric: wp.mat33,
    rotation_metric: wp.mat33,
    position_tolerance: wp.float32,
    rotation_tolerance: wp.float32,
    position_weight: wp.float32,
    rotation_weight: wp.float32,
):
    """Compute pose cost from precomputed positive-semidefinite quadratic metrics."""
    position_delta = current_position - target_position
    metric_position_delta = position_metric * position_delta
    position_distance_squared = wp.max(wp.dot(position_delta, metric_position_delta), 0.0)
    position_distance = wp.sqrt(position_distance_squared)
    position_cost = 0.5 * position_weight * position_distance_squared
    position_gradient = position_weight * metric_position_delta
    if position_distance <= position_tolerance:
        position_cost = 0.0
        position_distance = 0.0
        position_gradient = wp.vec3(0.0, 0.0, 0.0)

    rotation_cost = wp.float32(0.0)
    rotation_distance = wp.float32(0.0)
    rotation_gradient = wp.vec3(0.0, 0.0, 0.0)
    rotation_metric_trace = (
        rotation_metric[0, 0] + rotation_metric[1, 1] + rotation_metric[2, 2]
    )
    if rotation_metric_trace > 0.0 and rotation_weight != 0.0:
        rotation_error, error_linearization = compute_rotation_error(current_quat, target_quat)
        metric_rotation_error = rotation_metric * rotation_error
        rotation_distance_squared = wp.max(
            wp.dot(rotation_error, metric_rotation_error), 0.0
        )
        rotation_distance = wp.sqrt(rotation_distance_squared)
        rotation_cost = rotation_weight * rotation_distance_squared
        error_gradient = 2.0 * rotation_weight * metric_rotation_error
        rotation_gradient = wp.transpose(error_linearization) * error_gradient
        if rotation_distance <= rotation_tolerance:
            rotation_cost = 0.0
            rotation_distance = 0.0
            rotation_gradient = wp.vec3(0.0, 0.0, 0.0)

    return (
        position_cost,
        rotation_cost,
        position_distance,
        rotation_distance,
        position_gradient,
        rotation_gradient,
    )
