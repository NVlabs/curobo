# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library
from typing import Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream


@wp.func
def scale_quaternion_difference_by_axis(q: wp.quat, weights: wp.vec3):
    """Scale quaternion rotation per axis while maintaining valid rotation.

    Args:
        q: Quaternion (x,y,z,w format)
        weights: Scale factors for each axis

    Returns:
        Scaled quaternion
    """
    # Extract vector and scalar parts
    vec_part = wp.vec3(q[0], q[1], q[2])
    w = q[3]

    # Get the original half angle
    half_angle = wp.atan2(wp.length(vec_part), w)

    # If we have a non-zero rotation
    vec_length = wp.length(vec_part)
    if vec_length > 0.0:
        # Normalize the axis vector
        axis = vec_part * (1.0 / vec_length)

        # Scale the axis components
        scaled_axis = wp.vec3(axis[0] * weights[0], axis[1] * weights[1], axis[2] * weights[2])

        # Normalize the scaled axis
        scaled_length = wp.length(scaled_axis)
        if scaled_length > 1e-6:  # Use a small epsilon to avoid division by zero
            scaled_axis = scaled_axis * (1.0 / scaled_length)

            # Compute new quaternion components
            new_sin = wp.sin(half_angle)
            new_cos = wp.cos(half_angle)

            # Construct new quaternion with scaled axis but same angle
            return wp.quat(
                scaled_axis[0] * new_sin,
                scaled_axis[1] * new_sin,
                scaled_axis[2] * new_sin,
                new_cos,
            )
        else:
            # If all components are zero or close to zero after scaling,
            # return identity quaternion (no rotation)
            return wp.quat(0.0, 0.0, 0.0, 1.0)

    return q


@wp.func
def compute_position_error(
    current_position: wp.vec3,
    goal_position: wp.vec3,
    dof_weight: wp.vec3,
    position_weight: wp.float32,
    convergence_tolerance: wp.float32,
):
    """Compute the error in position and its gradient.

    Args:
        current_position: The current position.
        goal_position: The goal position.
        dof_weight: The weight for each axis.
        position_weight: The weight for the position.
        convergence_tolerance: The tolerance for the position.

    Returns:
        position_distance: The distance between the current and goal position.
        position_gradient: The gradient of the position.
    """
    position_delta = current_position - goal_position
    weighted_position_delta = wp.vec3(
        position_delta[0] * dof_weight[0],
        position_delta[1] * dof_weight[1],
        position_delta[2] * dof_weight[2],
    )
    position_distance = (
        0.5 * position_weight * wp.dot(weighted_position_delta, weighted_position_delta)
    )
    # Correct gradient: position_weight * W^T * W * delta = position_weight * W^2 * delta
    position_gradient = wp.vec3(
        position_weight * dof_weight[0] * dof_weight[0] * position_delta[0],
        position_weight * dof_weight[1] * dof_weight[1] * position_delta[1],
        position_weight * dof_weight[2] * dof_weight[2] * position_delta[2],
    )
    if position_distance < convergence_tolerance:
        position_distance = 0.0
        position_gradient = wp.vec3(0.0, 0.0, 0.0)
    return position_distance, position_gradient


@wp.func
def convert_angular_velocity_to_quaternion_rate(angular_velocity: wp.vec3, current_quat: wp.quat):
    """Convert angular velocity to quaternion rate.

    This function converts angular velocity to quaternion rate using the quaternion multiplication.
    The quaternion rate is then scaled by 0.5.

    Args:
        angular_velocity: The angular velocity. In x, y, z format.
        current_quat: The current quaternion. In x, y, z, w format.

    Returns:
        quat_rate: The quaternion rate. In x, y, z, w format.
    """
    omega_quat = wp.quat(angular_velocity[0], angular_velocity[1], angular_velocity[2], 0.0)

    quat_rate = wp.mul(current_quat, omega_quat)
    # quat_rate = 0.5 * quat_rate
    return quat_rate


@wp.func
def compute_rotation_error_axis_angle(
    current_quat: wp.quat,
    goal_quat: wp.quat,
    rotation_dof_weight: wp.vec3,
    rotation_weight: wp.float32,
    convergence_tolerance: wp.float32,
):
    """Compute the distance between two quaternions and its gradient using axis-angle representation.

    This function computes the relative rotation first, then converts to axis angle representation.
    The axis angle representation is then weighted and converted back to a quaternion rate as the
    gradient. The distance is the dot product of the weighted axis angle with itself.

    Args:
        current_quat: The current quaternion. In x, y, z, w format.
        goal_quat: The goal quaternion. In x, y, z, w format.
        rotation_dof_weight: The weight for each axis. This is a 3D vector. Applied to the angular velocity.
        rotation_weight: The weight for the rotation.
        convergence_tolerance: The tolerance for the rotation.

    Returns:
        angular_distance: The distance between the two quaternions.
        gradient_as_angular_velocity: The gradient of the rotation as a angular velocity. In x, y, z format.
        angle: The angle of rotation between the quaternions.
    """
    # Compute the relative rotation between current and goal

    inverse_goal_quat = wp.quat_inverse(goal_quat)

    quaternion_delta = wp.mul(current_quat, inverse_goal_quat)

    # Extract the vector part of the quaternion delta
    q_xyz = wp.vec3(quaternion_delta[0], quaternion_delta[1], quaternion_delta[2])

    q_xyz = wp.cw_mul(rotation_dof_weight, q_xyz)  #

    # Calculate the angle from the quaternion
    vec_length = wp.length(q_xyz)

    angle = 2.0 * wp.atan2(vec_length, wp.abs(quaternion_delta[3]))
    if rotation_weight == 0.0:
        angle = 0.0

    # Compute the rotation axis
    if vec_length < 1e-15:
        axis = wp.vec3(0.0, 0.0, 0.0)
    else:
        axis = q_xyz / vec_length
    # Convert to angular velocity (omega)
    omega = angle * axis

    # Compute the angular distance
    # Apply rotation weight

    angular_distance = rotation_weight * wp.dot(omega, omega)

    # Initialize gradient as zero quaternion
    gradient_as_angular_velocity = wp.vec3(0.0, 0.0, 0.0)

    if angular_distance < convergence_tolerance:
        angular_distance = 0.0
    else:
        # Scale factor for gradient
        scale_factor = 2.0

        if quaternion_delta[3] < 0.0:
            scale_factor = -1.0 * scale_factor

        scaled_gradient = scale_factor * rotation_weight * omega

        gradient_as_angular_velocity[0] = scaled_gradient[0]  # x
        gradient_as_angular_velocity[1] = scaled_gradient[1]  # y
        gradient_as_angular_velocity[2] = scaled_gradient[2]  # z

    return angular_distance, gradient_as_angular_velocity, angle


@wp.func
def compute_rotation_error_lie_group(
    current_quat: wp.quat,
    goal_quat: wp.quat,
    rotation_dof_weight: wp.vec3,
    rotation_weight: wp.float32,
    convergence_tolerance: wp.float32,
):
    """Compute the distance between two quaternions using Lie group theory.

    This function uses the logarithmic map to work in the tangent space (Lie algebra so(3))
    of the rotation group SO(3). The error is computed as the logarithm of the relative
    rotation, which gives us the axis-angle representation in the tangent space.

    Args:
        current_quat: The current quaternion. In x, y, z, w format.
        goal_quat: The goal quaternion. In x, y, z, w format.
        rotation_dof_weight: The weight for each axis. This is a 3D vector. Applied to the tangent space vector.
        rotation_weight: The weight for the rotation.
        convergence_tolerance: The tolerance for the rotation.

    Returns:
        angular_distance: The distance between the two quaternions.
        gradient_as_angular_velocity: The gradient of the rotation as angular velocity. In x, y, z format.
        angle: The weighted angle of rotation between the quaternions.
    """
    # Compute the relative rotation between current and goal
    inverse_goal_quat = wp.quat_inverse(goal_quat)
    quaternion_delta = wp.mul(current_quat, inverse_goal_quat)

    # Ensure quaternion is in the positive hemisphere (w >= 0) for consistent logarithm
    if quaternion_delta[3] < 0.0:
        quaternion_delta = wp.quat(
            -quaternion_delta[0], -quaternion_delta[1], -quaternion_delta[2], -quaternion_delta[3]
        )

    # Compute the logarithmic map: log(q) = (theta/2) * (v/|v|) where q = [v*sin(theta/2), cos(theta/2)]
    # This maps from SO(3) to its Lie algebra so(3)
    w = quaternion_delta[3]  # scalar part
    v = wp.vec3(quaternion_delta[0], quaternion_delta[1], quaternion_delta[2])  # vector part

    # Compute UNWEIGHTED angle first (geometrically correct)
    v_norm = wp.length(v)
    half_angle = wp.atan2(v_norm, wp.abs(w))
    if rotation_weight == 0.0:
        half_angle = 0.0

    geometric_angle = 2.0 * half_angle  # This is the true geometric angle

    # Compute logarithmic map (tangent space vector)
    if v_norm < 1e-10:
        # Near identity, use first-order approximation: log(q) ≈ 2*v
        tangent_vector = 2.0 * v
    else:
        # General case: log(q) = (theta/sin(theta/2)) * v where theta = 2*atan2(|v|, w)
        if wp.abs(half_angle) < 1e-15:
            # Near identity, use series expansion to avoid numerical issues
            # log(q) ≈ 2*v * (1 + (|v|^2)/(6*w^2))
            correction = 1.0 + (v_norm * v_norm) / (6.0 * w * w)
            tangent_vector = 2.0 * v * correction
        else:
            # Standard case: log(q) = (angle/sin(angle/2)) * v
            sinc_factor = geometric_angle / (2.0 * wp.sin(half_angle))  # angle / (2 * sin(angle/2))
            tangent_vector = sinc_factor * v

    # Apply DOF weights to tangent space vector
    weighted_tangent_vector = wp.cw_mul(rotation_dof_weight, tangent_vector)

    # Calculate weighted angle from the weighted tangent vector
    # This gives an "effective rotation angle" that reflects DOF weighting
    angle = wp.length(weighted_tangent_vector)

    # Compute the angular distance as the squared norm in the weighted tangent space
    angular_distance = rotation_weight * wp.dot(weighted_tangent_vector, weighted_tangent_vector)

    # Initialize gradient
    gradient_as_angular_velocity = wp.vec3(0.0, 0.0, 0.0)

    if angular_distance < convergence_tolerance:
        angular_distance = 0.0
    else:
        # The gradient in the tangent space is simply the weighted tangent vector
        # scaled by the derivative of the cost function

        # Apply the chain rule: gradient w.r.t. current quaternion
        # The gradient of the logarithmic map provides the proper manifold-aware derivative
        gradient_as_angular_velocity = 2.0 * rotation_weight * weighted_tangent_vector

    return angular_distance, gradient_as_angular_velocity, angle


@wp.func
def compute_rotation_error_lie_group_advanced(
    current_quat: wp.quat,
    goal_quat: wp.quat,
    rotation_dof_weight: wp.vec3,
    rotation_weight: wp.float32,
    convergence_tolerance: wp.float32,
):
    """Advanced Lie group version with proper Jacobian computation.

    This version computes the proper Jacobian of the logarithmic map for more accurate
    gradients, especially important for optimization on the manifold.

    Args:
        current_quat: The current quaternion. In x, y, z, w format.
        goal_quat: The goal quaternion. In x, y, z, w format.
        rotation_dof_weight: The weight for each axis. This is a 3D vector. Applied to the tangent space vector.
        rotation_weight: The weight for the rotation.
        convergence_tolerance: The tolerance for the rotation.

    Returns:
        angular_distance: The distance between the two quaternions.
        gradient_as_angular_velocity: The gradient of the rotation as angular velocity. In x, y, z format.
        angle: The weighted angle of rotation between the quaternions.
    """
    # Compute the relative rotation
    inverse_goal_quat = wp.quat_inverse(goal_quat)
    quaternion_delta = wp.mul(current_quat, inverse_goal_quat)

    # Ensure positive hemisphere
    if quaternion_delta[3] < 0.0:
        quaternion_delta = wp.quat(
            -quaternion_delta[0], -quaternion_delta[1], -quaternion_delta[2], -quaternion_delta[3]
        )

    w = quaternion_delta[3]
    v = wp.vec3(quaternion_delta[0], quaternion_delta[1], quaternion_delta[2])

    # Compute UNWEIGHTED angle first (geometrically correct)
    v_norm = wp.length(v)
    half_angle = wp.atan2(v_norm, wp.abs(w))
    if rotation_weight == 0.0:
        half_angle = 0.0

    geometric_angle = 2.0 * half_angle  # This is the true geometric angle

    # Compute logarithmic map (tangent space vector) from UNWEIGHTED quaternion
    if v_norm < 1e-10:
        # Near identity
        tangent_vector = 2.0 * v
        # Jacobian is approximately 2*I (identity matrix effect)
        jacobian_scale = 2.0
    else:
        if wp.abs(half_angle) < 1e-15:
            # Series expansion
            correction = 1.0 + (v_norm * v_norm) / (6.0 * w * w)
            tangent_vector = 2.0 * v * correction
            jacobian_scale = 2.0 * correction
        else:
            # Standard case: log(q) = (angle/sin(angle/2)) * v
            sinc_factor = geometric_angle / (2.0 * wp.sin(half_angle))
            tangent_vector = sinc_factor * v

            # Jacobian of log map: J = (sin(θ/2)/θ/2) * I + (1-sin(θ/2)/θ/2)/(θ/2)^2 * v*v^T
            # For our purposes, we use the dominant term
            jacobian_scale = sinc_factor

    # Apply DOF weights to tangent space vector
    weighted_tangent_vector = wp.cw_mul(rotation_dof_weight, tangent_vector)

    # Calculate weighted angle from the weighted tangent vector
    # This gives an "effective rotation angle" that reflects DOF weighting
    angle = wp.length(weighted_tangent_vector)

    # Compute distance - use consistent weighted product for proper scaling
    angular_distance = rotation_weight * wp.dot(weighted_tangent_vector, weighted_tangent_vector)

    # Initialize gradient
    gradient_as_angular_velocity = wp.vec3(0.0, 0.0, 0.0)

    if angular_distance < convergence_tolerance:
        angular_distance = 0.0
    else:
        # Proper manifold-aware gradient computation
        # The gradient includes the Jacobian of the logarithmic map
        gradient_as_angular_velocity = 2.0 * rotation_weight * weighted_tangent_vector

    return angular_distance, gradient_as_angular_velocity, angle


@wp.func
def compute_rotation_error(
    current_quat: wp.quat,
    goal_quat: wp.quat,
    rotation_dof_weight: wp.vec3,
    rotation_weight: wp.float32,
    convergence_tolerance: wp.float32,
    rotation_error_method: wp.int32,
):
    """Meta function to compute rotation error using different methods.

    Args:
        current_quat: The current quaternion. In x, y, z, w format.
        goal_quat: The goal quaternion. In x, y, z, w format.
        rotation_dof_weight: The weight for each axis. This is a 3D vector.
        rotation_weight: The weight for the rotation.
        convergence_tolerance: The tolerance for the rotation.
        method: The method to use for computation:
                0 = axis-angle (original method)
                1 = lie group (basic)
                2 = lie group advanced (with proper Jacobian)

    Returns:
        angular_distance: The distance between the two quaternions.
        gradient_as_angular_velocity: The gradient of the rotation as angular velocity. In x, y, z format.
        angle: The angle of rotation between the quaternions.
    """
    if rotation_error_method == 0:
        angular_distance, gradient_as_angular_velocity, angle = compute_rotation_error_axis_angle(
            current_quat,
            goal_quat,
            rotation_dof_weight,
            rotation_weight,
            convergence_tolerance,
        )
    elif rotation_error_method == 1:
        angular_distance, gradient_as_angular_velocity, angle = compute_rotation_error_lie_group(
            current_quat, goal_quat, rotation_dof_weight, rotation_weight, convergence_tolerance
        )
    elif rotation_error_method == 2:
        angular_distance, gradient_as_angular_velocity, angle = (
            compute_rotation_error_lie_group_advanced(
                current_quat, goal_quat, rotation_dof_weight, rotation_weight, convergence_tolerance
            )
        )
    else:
        assert False
    return angular_distance, gradient_as_angular_velocity, angle


def create_goalset_pose_distance_kernel_with_constants(
    num_goalset: int,
    rotation_method: int = 0,
):
    """Create kernel with constants for pose distance computation.

    Args:
        num_goalset: Number of goal sets (used for compile-time loop bounds)
        rotation_method: Method for rotation error computation:
                        0 = axis-angle (original method)
                        1 = lie group (basic)
                        2 = lie group advanced (with proper Jacobian)

    Note:
        The kernel uses the runtime num_goalset value (extracted from actual tensor shape)
        for both loop bounds and tensor indexing to handle cases where num_goalset != num_links.
    """

    # Kernel computes pose distance and gradient:
    @wp.kernel
    def goalset_pose_distance(
        current_position: wp.array(dtype=wp.vec3),  # shape is [batch_size * horizon * nlinks, 3]
        current_quat: wp.array(dtype=wp.vec4),  # shape is [batch_size * horizon * nlinks, 4]
        goal_position: wp.array(dtype=wp.vec3),  # shape is [batch_goals, num_links, num_goalset, 3]
        goal_quat: wp.array(dtype=wp.vec4),  # shape is [batch_goals, num_links, num_goalset, 4]
        idxs_goal: wp.array(dtype=wp.int32),  # shape is [batch_size]
        position_orientation_weight: wp.array(dtype=wp.float32),  # shape is [2]
        terminal_pose_axes_weight_factor: wp.array(dtype=wp.float32),  # shape is [num_links, 6]
        non_terminal_pose_axes_weight_factor: wp.array(
            dtype=wp.float32
        ),  # shape is [num_links, 6] position, rotation
        terminal_pose_convergence_tolerance: wp.array(
            dtype=wp.float32
        ),  # shape is [num_links, 2] position, rotation
        non_terminal_pose_convergence_tolerance: wp.array(
            dtype=wp.float32
        ),  # shape is [num_links, 2] position, rotation
        project_distance_to_goal: wp.array(dtype=wp.uint8),  # shape is [num_links, 1]
        out_distance: wp.array(dtype=wp.float32),  # shape is [batch_size * horizon * num_links, 2]
        out_position_distance: wp.array(
            dtype=wp.float32
        ),  # shape is [batch_size * horizon * num_links, 1]
        out_rotation_distance: wp.array(
            dtype=wp.float32
        ),  # shape is [batch_size * horizon * num_links, 1]
        out_position_gradient: wp.array(
            dtype=wp.vec3
        ),  # shape is [batch_size * horizon * num_links, 3]
        out_rotation_gradient: wp.array(
            dtype=wp.vec4
        ),  # shape is [batch_size * horizon * num_links, 4]
        out_goalset_idx: wp.array(dtype=wp.int32),  # shape is [batch_size * horizon * num_links, 1]
        batch_size: wp.int32,
        horizon: wp.int32,
        num_links: wp.int32,
    ):
        tid = wp.tid()  # threads launched = batch_size * horizon
        if tid >= batch_size * horizon * num_links:
            return
        # there are batch_size * horizon * num_links threads launched
        # get batch index
        b_idx = tid / (horizon * num_links)

        # get horizon index
        h_idx = (tid - b_idx * horizon * num_links) / num_links

        # get link index
        link_idx = tid - b_idx * horizon * num_links - h_idx * num_links

        position_axes_weight = wp.vec3(0.0, 0.0, 0.0)
        rotation_axes_weight = wp.vec3(0.0, 0.0, 0.0)
        convergence_tolerance = wp.vec2(0.0, 0.0)

        if h_idx < (horizon - 1) and horizon > 1:
            position_axes_weight = wp.vec3(
                non_terminal_pose_axes_weight_factor[link_idx * 6 + 0],
                non_terminal_pose_axes_weight_factor[link_idx * 6 + 1],
                non_terminal_pose_axes_weight_factor[link_idx * 6 + 2],
            )
            rotation_axes_weight = wp.vec3(
                non_terminal_pose_axes_weight_factor[link_idx * 6 + 3],
                non_terminal_pose_axes_weight_factor[link_idx * 6 + 4],
                non_terminal_pose_axes_weight_factor[link_idx * 6 + 5],
            )
            convergence_tolerance = wp.vec2(
                non_terminal_pose_convergence_tolerance[link_idx * 2 + 0],
                non_terminal_pose_convergence_tolerance[link_idx * 2 + 1],
            )
        else:
            position_axes_weight = wp.vec3(
                terminal_pose_axes_weight_factor[link_idx * 6 + 0],
                terminal_pose_axes_weight_factor[link_idx * 6 + 1],
                terminal_pose_axes_weight_factor[link_idx * 6 + 2],
            )
            rotation_axes_weight = wp.vec3(
                terminal_pose_axes_weight_factor[link_idx * 6 + 3],
                terminal_pose_axes_weight_factor[link_idx * 6 + 4],
                terminal_pose_axes_weight_factor[link_idx * 6 + 5],
            )
            convergence_tolerance = wp.vec2(
                terminal_pose_convergence_tolerance[link_idx * 2 + 0],
                terminal_pose_convergence_tolerance[link_idx * 2 + 1],
            )

        # read dof weights:
        position_weight = position_orientation_weight[0]
        rotation_weight = position_orientation_weight[1]
        convergence_tolerance[0] = convergence_tolerance[0] ** 2.0
        convergence_tolerance[1] = convergence_tolerance[1] ** 2.0

        # read batch index for current thread:
        goal_idx = idxs_goal[b_idx]

        local_project_distance_to_goal = project_distance_to_goal[link_idx]

        # read current position and quat
        c_position = current_position[b_idx * horizon * num_links + h_idx * num_links + link_idx]
        c_quat = current_quat[b_idx * horizon * num_links + h_idx * num_links + link_idx]  # w, x, y, z
        c_quaternion = wp.quaternion(c_quat[1], c_quat[2], c_quat[3], c_quat[0])  # x, y, z, w

        best_distance = wp.float32(-1)
        best_goal_idx = wp.int32(0)
        best_position_distance = wp.float32(-1)
        best_rotation_distance = wp.float32(-1)
        best_position_gradient = wp.vec3(0.0, 0.0, 0.0)
        best_rotation_gradient = wp.vec3(0.0, 0.0, 0.0)
        best_current_quaternion = wp.quat(0.0, 0.0, 0.0, 1.0)
        best_angle = wp.float32(-1)

        current_position_in_frame = wp.vec3(0.0, 0.0, 0.0)
        current_quaternion_in_frame = wp.quat(0.0, 0.0, 0.0, 1.0)
        goal_position_in_frame = wp.vec3(0.0, 0.0, 0.0)
        goal_quaternion_in_frame = wp.quat(0.0, 0.0, 0.0, 1.0)
        best_g_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))

        # loop over num_goalset
        for g_idx in range(num_goalset):
            # read goal position and quat
            # goal_position is shape (batch_goals, num_links, num_goalset, 3)
            # Use runtime num_goalset for indexing to handle cases where num_goalset != num_links
            g_goal_position = goal_position[
                goal_idx * num_links * num_goalset + link_idx * num_goalset + g_idx
            ]
            g_goal_quat = goal_quat[
                goal_idx * num_links * num_goalset + link_idx * num_goalset + g_idx
            ]
            g_goal_quaternion = wp.quaternion(
                g_goal_quat[1], g_goal_quat[2], g_goal_quat[3], g_goal_quat[0]
            )

            # project current pose to goal frame:
            if local_project_distance_to_goal == 1:
                # project current pose to goal frame:
                g_transform = wp.transform(g_goal_position, g_goal_quaternion)
                c_transform = wp.transform(c_position, c_quaternion)
                c_in_g_frame = wp.transform_multiply(wp.transform_inverse(g_transform), c_transform)
                c_in_g_frame_position = wp.transform_get_translation(c_in_g_frame)
                c_in_g_frame_quaternion = wp.transform_get_rotation(c_in_g_frame)

                g_in_g_frame = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
                g_in_g_frame_position = wp.transform_get_translation(g_in_g_frame)
                g_in_g_frame_quaternion = wp.transform_get_rotation(g_in_g_frame)

                current_position_in_frame = c_in_g_frame_position
                current_quaternion_in_frame = c_in_g_frame_quaternion
                goal_position_in_frame = g_in_g_frame_position
                goal_quaternion_in_frame = g_in_g_frame_quaternion
            else:
                current_position_in_frame = c_position
                current_quaternion_in_frame = c_quaternion
                goal_position_in_frame = g_goal_position
                goal_quaternion_in_frame = g_goal_quaternion

            # calculate error in goal frame:

            position_distance, position_gradient = compute_position_error(
                current_position_in_frame,
                goal_position_in_frame,
                position_axes_weight,
                position_weight,
                convergence_tolerance[0],
            )

            angular_distance, gradient_as_angular_velocity, angle = compute_rotation_error(
                current_quaternion_in_frame,
                goal_quaternion_in_frame,
                rotation_axes_weight,
                rotation_weight,
                convergence_tolerance[1],
                rotation_method,
            )

            total_distance = position_distance + angular_distance

            if best_distance < 0 or total_distance < best_distance:
                best_distance = total_distance
                best_goal_idx = g_idx
                best_position_distance = position_distance
                best_rotation_distance = angular_distance
                best_position_gradient = position_gradient
                best_rotation_gradient = gradient_as_angular_velocity
                best_current_quaternion = c_quaternion
                if local_project_distance_to_goal == 1:
                    best_g_transform = g_transform
                best_angle = angle

        scaled_linear_distance = wp.float32(0.0)
        scaled_angular_distance = wp.float32(0.0)
        scaled_linear_distance = best_position_distance
        scaled_angular_distance = best_rotation_distance


        # Project gradients from goal frame to world frame:
        if local_project_distance_to_goal == 1:
            best_position_gradient = wp.transform_vector(best_g_transform, best_position_gradient)
            best_rotation_gradient = wp.transform_vector(best_g_transform, best_rotation_gradient)

        # Compute weight-independent error reporting for convergence checking
        # Extract the weighted geometric distance by removing only the position/rotation weight scaling
        # This gives ||weighted_position_delta|| and ||weighted_rotation_error|| which are weight-independent
        geometric_position_distance = (
            wp.sqrt(2.0 * best_position_distance / position_weight)
            if position_weight > 0.0
            else 0.0
        )
        geometric_rotation_distance = best_angle  # best_angle is already weight-independent

        quaternion_rate_gradient = convert_angular_velocity_to_quaternion_rate(
            best_rotation_gradient, best_current_quaternion
        )

        out_distance[2 * (b_idx * horizon * num_links + h_idx * num_links + link_idx)] = (
            scaled_linear_distance
        )
        out_distance[2 * (b_idx * horizon * num_links + h_idx * num_links + link_idx) + 1] = (
            scaled_angular_distance
        )
        out_goalset_idx[b_idx * horizon * num_links + h_idx * num_links + link_idx] = best_goal_idx

        out_position_distance[b_idx * horizon * num_links + h_idx * num_links + link_idx] = (
            geometric_position_distance
        )
        out_rotation_distance[b_idx * horizon * num_links + h_idx * num_links + link_idx] = (
            geometric_rotation_distance
        )

        out_position_gradient[b_idx * horizon * num_links + h_idx * num_links + link_idx] = (
            best_position_gradient
        )

        out_rotation_gradient[b_idx * horizon * num_links + h_idx * num_links + link_idx] = wp.vec4(
            quaternion_rate_gradient[3],  # w
            quaternion_rate_gradient[0],  # x
            quaternion_rate_gradient[1],  # y
            quaternion_rate_gradient[2],  # z
        )

    return goalset_pose_distance


class ToolPoseDistance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        current_position: torch.Tensor,
        current_quat: torch.Tensor,
        goal_position: torch.Tensor,
        goal_quat: torch.Tensor,
        idxs_goal: torch.Tensor,
        position_orientation_weight: torch.Tensor,
        terminal_pose_axes_weight_factor: torch.Tensor,
        non_terminal_pose_axes_weight_factor: torch.Tensor,
        terminal_pose_convergence_tolerance: torch.Tensor,
        non_terminal_pose_convergence_tolerance: torch.Tensor,
        project_distance_to_goal: torch.Tensor,
        out_distance: torch.Tensor,
        out_position_distance: torch.Tensor,
        out_rotation_distance: torch.Tensor,
        out_position_gradient: torch.Tensor,
        out_rotation_gradient: torch.Tensor,
        out_goalset_idx: torch.Tensor,
        use_grad_input: bool,
        warp_kernel,
    ):
        """Compute the pose distance between the current pose and the goal pose.

        Only current_position and current_quat are supported for gradient computation.

        Args:
            ctx: context
            current_position: Shape is (b, h, num_links, 3).
            current_quat: Shape is (b, h, num_links, 4)
            goal_position: Shape is (-1, num_links, num_goalset, 3). The tensor is accessed using idxs_goal.
            goal_quat: Shape is (-1, num_links, num_goalset, 4). The tensor is accessed using idxs_goal.
            idxs_goal: Shape is (b,1). The tensor is accessed using b_idx.
            position_orientation_weight: Shape is (2,). The first element is the weight for
                the position, the second element is the weight for the rotation.
            terminal_pose_axes_weight_factor: Shape is (num_links, 6). The first three elements are the weight factors for the
                position, the last three elements are the weight factors for the rotation.
            non_terminal_pose_axes_weight_factor: Shape is (num_links, 6). The first three elements are the weight factors for
                the position, the last three elements are the weight factors for the rotation.
            terminal_pose_convergence_tolerance: Shape is (num_links, 2). The first element is the convergence
                tolerance for the position, the second element is the convergence tolerance for the rotation.
            non_terminal_pose_convergence_tolerance: Shape is (num_links, 2). The first element is the convergence
                tolerance for the position, the second element is the convergence tolerance for the rotation.
            project_distance_to_goal: Shape is (num_links,1). The tensor is accessed using
                project_distance_to_goal[0]. Only 0 is supported for now.
            out_distance: Shape is (b,h,num_links*2). The distance between the current pose and the goal pose.
            out_position_distance: Shape is (b,h,num_links). The position distance between the current pose
                and the goal pose.
            out_rotation_distance: Shape is (b,h,num_links). The rotation distance between the current pose
                and the goal pose.
            out_position_gradient: Shape is (b,h,num_links,3). The gradient of the position distance with
                respect to the current position.
            out_rotation_gradient: Shape is (b,h,num_links,4). The gradient of the rotation distance with
                respect to the current rotation.
            out_goalset_idx: Shape is (b,h,num_links). The index of the goal pose in the goal set.
            use_grad_input: bool. If True, the gradient of the input is used.

        Returns:
            out_distance: Shape is (b,h,num_links*2). Gradient is supported.
            out_position_distance: Shape is (b,h,num_links). Gradient is not supported.
            out_rotation_distance: Shape is (b,h,num_links). Gradient is not supported.
            out_goalset_idx: Shape is (b,h,num_links). Gradient is not supported.
        """
        ctx.set_materialize_grads(False)
        if current_position.ndim != 4:
            log_and_raise("current_position must be a 4D tensor")
        if current_quat.ndim != 4:
            log_and_raise("current_quat must be a 4D tensor")

        if goal_position.ndim != 4:
            log_and_raise("goal_position must be a 4D tensor with shape (-1, num_links, num_goalset, 3)")
        if goal_quat.ndim != 4:
            log_and_raise("goal_quat must be a 4D tensor with shape (-1, num_links, num_goalset, 4)")

        num_goalset = goal_position.shape[-2]
        b, h, num_links, _ = current_position.shape
        if current_position.shape != (b, h, num_links, 3):
            log_and_raise("current_position must have shape (b, h, num_links, 3)")
        if current_quat.shape != (b, h, num_links, 4):
            log_and_raise("current_quat must have shape (b, h, num_links, 4)")

        if idxs_goal.shape != (b, 1):
            log_and_raise(
                "idxs_goal must have shape ({},)".format(b) + " but got {}".format(idxs_goal.shape)
            )
        if position_orientation_weight.shape != (2,):
            log_and_raise("position_orientation_weight must have shape (2,)")

        # check gradient shape:
        if out_position_gradient.shape != (b, h, num_links, 3):
            log_and_raise("out_position_gradient must have shape (b, h, num_links, 3)")
        if out_rotation_gradient.shape != (b, h, num_links, 4):
            log_and_raise("out_rotation_gradient must have shape (b, h, num_links, 4)")
        if out_distance.shape != (b, h, num_links * 2):
            log_and_raise("out_distance must have shape (b, h, num_links*2)")
        if out_position_distance.shape != (b, h, num_links):
            log_and_raise("out_position_distance must have shape (b, h, num_links)")
        if out_rotation_distance.shape != (b, h, num_links):
            log_and_raise("out_rotation_distance must have shape (b, h, num_links)")
        if out_goalset_idx.shape != (b, h, num_links):
            log_and_raise("out_goalset_idx must have shape (b, h, num_links)")

        ctx.use_grad_input = use_grad_input
        wp_device, wp_stream = get_warp_device_stream(current_position)

        dim = b * h * num_links

        wp.launch(
            kernel=warp_kernel,
            dim=dim,
            inputs=[
                wp.from_torch(current_position.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(current_quat.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(goal_position.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(goal_quat.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(idxs_goal.detach().view(-1), dtype=wp.int32),
                wp.from_torch(position_orientation_weight.view(-1), dtype=wp.float32),
                wp.from_torch(terminal_pose_axes_weight_factor.view(-1), dtype=wp.float32),
                wp.from_torch(non_terminal_pose_axes_weight_factor.view(-1), dtype=wp.float32),
                wp.from_torch(terminal_pose_convergence_tolerance.view(-1), dtype=wp.float32),
                wp.from_torch(non_terminal_pose_convergence_tolerance.view(-1), dtype=wp.float32),
                wp.from_torch(project_distance_to_goal.view(-1), dtype=wp.uint8),
                wp.from_torch(out_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_rotation_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_gradient.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(out_rotation_gradient.view(-1, 4), dtype=wp.vec4),
                wp.from_torch(out_goalset_idx.view(-1), dtype=wp.int32),
                b,
                h,
                num_links,
            ],
            device=wp_device,
            stream=wp_stream,
            adjoint=False,
        )

        # Save tensors needed for backward

        ctx.mark_non_differentiable(
            out_position_distance,
            out_rotation_distance,
            out_goalset_idx,
            goal_position,
            goal_quat,
            idxs_goal,
            position_orientation_weight,
            terminal_pose_axes_weight,
            non_terminal_pose_axes_weight,
            terminal_pose_convergence_tolerance,
            non_terminal_pose_convergence_tolerance,
            project_distance_to_goal,
        )
        ctx.save_for_backward(out_position_gradient, out_rotation_gradient)

        return out_distance, out_position_distance, out_rotation_distance, out_goalset_idx

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_distance: Optional[torch.Tensor],  # shape: batch_size, horizon, num_links*2
        grad_position_distance: Optional[torch.Tensor],
        grad_rotation_distance: Optional[torch.Tensor],
        grad_goalset_idx: Optional[torch.Tensor],
    ):
        # Extract saved tensors
        use_grad_input = ctx.use_grad_input
        pos_grad = None
        quat_grad = None
        if grad_distance is not None:
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                out_position_gradient, out_rotation_gradient = ctx.saved_tensors

            # Only compute gradients if use_grad_input is True
            # grad_distance has shape (batch, horizon, num_links*2) with interleaved pos/ori costs
            # out_position_gradient has shape (batch, horizon, num_links, 3)
            # out_rotation_gradient has shape (batch, horizon, num_links, 4)
            if ctx.needs_input_grad[0]:
                if use_grad_input:
                    # Extract position gradient multipliers (indices 0, 2, 4, ...)
                    grad_pos = grad_distance[:, :, 0::2].unsqueeze(-1)
                    pos_grad = out_position_gradient * grad_pos
                else:
                    pos_grad = out_position_gradient
            if ctx.needs_input_grad[1]:
                if use_grad_input:
                    # Extract orientation gradient multipliers (indices 1, 3, 5, ...)
                    grad_ori = grad_distance[:, :, 1::2].unsqueeze(-1)
                    quat_grad = out_rotation_gradient * grad_ori
                else:
                    quat_grad = out_rotation_gradient

        # Return gradients for each input (None for inputs that don't need gradients)
        return (
            pos_grad,
            quat_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
