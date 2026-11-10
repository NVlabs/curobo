# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical tests for the shared projected-pose metric Warp functions."""

# Standard Library
import math

# Third Party
import numpy as np
import pytest
import warp as wp

# CuRobo
from curobo._src.cost.wp_pose_metric import (
    angular_cotangent_to_wxyz_quaternion_cotangent,
    compute_projected_pose_metric,
    compute_rotation_error,
)
from curobo._src.util.warp import init_warp


@wp.kernel
def _rotation_error_kernel(
    current_quat: wp.array(dtype=wp.quat),
    target_quat: wp.array(dtype=wp.quat),
    error_out: wp.array(dtype=wp.vec3),
    linearization_out: wp.array(dtype=wp.mat33),
):
    index = wp.tid()
    error, linearization = compute_rotation_error(current_quat[index], target_quat[index])
    error_out[index] = error
    linearization_out[index] = linearization


@wp.kernel
def _quaternion_cotangent_kernel(
    angular_cotangent: wp.array(dtype=wp.vec3),
    current_quat: wp.array(dtype=wp.quat),
    quaternion_cotangent_out: wp.array(dtype=wp.vec4),
):
    index = wp.tid()
    quaternion_cotangent_out[index] = angular_cotangent_to_wxyz_quaternion_cotangent(
        angular_cotangent[index], current_quat[index]
    )


@wp.kernel
def _projected_pose_metric_kernel(
    current_position: wp.array(dtype=wp.vec3),
    target_position: wp.array(dtype=wp.vec3),
    current_quat: wp.array(dtype=wp.quat),
    target_quat: wp.array(dtype=wp.quat),
    position_projector: wp.array(dtype=wp.mat33),
    rotation_projector: wp.array(dtype=wp.mat33),
    position_scale: wp.array(dtype=wp.vec3),
    rotation_scale: wp.array(dtype=wp.vec3),
    tolerance: wp.array(dtype=wp.vec2),
    weight: wp.array(dtype=wp.vec2),
    cost_out: wp.array(dtype=wp.vec2),
    distance_out: wp.array(dtype=wp.vec2),
    position_gradient_out: wp.array(dtype=wp.vec3),
    rotation_gradient_out: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    (
        position_cost,
        rotation_cost,
        position_distance,
        rotation_distance,
        position_gradient,
        rotation_gradient,
    ) = compute_projected_pose_metric(
        current_position[index],
        target_position[index],
        current_quat[index],
        target_quat[index],
        position_projector[index],
        rotation_projector[index],
        position_scale[index],
        rotation_scale[index],
        tolerance[index][0],
        tolerance[index][1],
        weight[index][0],
        weight[index][1],
    )
    cost_out[index] = wp.vec2(position_cost, rotation_cost)
    distance_out[index] = wp.vec2(position_distance, rotation_distance)
    position_gradient_out[index] = position_gradient
    rotation_gradient_out[index] = rotation_gradient


def _normalize_quaternion(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _multiply_quaternions(lhs, rhs):
    lhs_xyz = lhs[:3]
    rhs_xyz = rhs[:3]
    return np.concatenate(
        (
            lhs[3] * rhs_xyz + rhs[3] * lhs_xyz + np.cross(lhs_xyz, rhs_xyz),
            [lhs[3] * rhs[3] - np.dot(lhs_xyz, rhs_xyz)],
        )
    )


def _axis_angle_quaternion(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half_angle = 0.5 * angle
    return np.concatenate((axis * math.sin(half_angle), [math.cos(half_angle)]))


def _canonicalize_quaternion(quaternion):
    quaternion = _normalize_quaternion(quaternion)
    should_flip = quaternion[3] < 0.0
    if quaternion[3] == 0.0:
        for component in quaternion[:3]:
            if component != 0.0:
                should_flip = component < 0.0
                break
    return -quaternion if should_flip else quaternion


def _rotation_error_reference(current_quat, target_quat):
    target_inverse = np.concatenate((-target_quat[:3], [target_quat[3]]))
    error_quat = _canonicalize_quaternion(_multiply_quaternions(current_quat, target_inverse))
    return 2.0 * error_quat[:3]


def _left_perturb_quaternion(current_quat, delta):
    angle = np.linalg.norm(delta)
    if angle < 1.0e-14:
        perturbation = np.concatenate((0.5 * delta, [1.0]))
    else:
        perturbation = _axis_angle_quaternion(delta / angle, angle)
    return _normalize_quaternion(_multiply_quaternions(perturbation, current_quat))


def _launch_rotation_error(current_quat, target_quat, device="cpu"):
    count = len(current_quat)
    current_wp = wp.array(current_quat, dtype=wp.quat, device=device)
    target_wp = wp.array(target_quat, dtype=wp.quat, device=device)
    error_wp = wp.zeros(count, dtype=wp.vec3, device=device)
    linearization_wp = wp.zeros(count, dtype=wp.mat33, device=device)
    wp.launch(
        _rotation_error_kernel,
        dim=count,
        inputs=[current_wp, target_wp, error_wp, linearization_wp],
        device=device,
    )
    return error_wp.numpy(), linearization_wp.numpy()


@pytest.fixture(scope="module", autouse=True)
def setup_warp():
    """Initialize Warp before compiling the test kernels."""
    init_warp(quiet=True)


def test_rotation_error_matches_shortest_arc_quaternion_vector_reference():
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    current = np.stack(
        (
            identity,
            _axis_angle_quaternion([1.0, -2.0, 0.5], 1.0e-5),
            _axis_angle_quaternion([0.3, -0.4, 0.8], 1.37),
            _axis_angle_quaternion([-0.2, 0.9, 0.3], math.pi - 1.0e-4),
        )
    )
    target = np.stack(
        (
            identity,
            identity,
            _axis_angle_quaternion([0.6, 0.1, -0.7], -0.42),
            identity,
        )
    )

    error, linearization = _launch_rotation_error(current, target)
    expected = np.stack(
        [_rotation_error_reference(current[index], target[index]) for index in range(len(current))]
    )

    np.testing.assert_allclose(error, expected, atol=2.0e-6, rtol=2.0e-6)
    np.testing.assert_allclose(
        np.linalg.norm(error, axis=-1),
        np.linalg.norm(expected, axis=-1),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    np.testing.assert_array_equal(linearization[0], np.eye(3, dtype=np.float32))


def test_rotation_error_is_quaternion_sign_invariant():
    current = _axis_angle_quaternion([0.3, -0.7, 0.2], 1.8)
    target = _axis_angle_quaternion([-0.5, 0.1, 0.4], -0.8)
    error, linearization = _launch_rotation_error(
        np.stack((current, -current)), np.stack((target, target))
    )

    np.testing.assert_allclose(error[0], error[1], atol=2.0e-6)
    np.testing.assert_allclose(linearization[0], linearization[1], atol=2.0e-6)


def test_rotation_error_is_quaternion_sign_invariant_at_pi():
    axis = np.array([-0.3, 0.8, -0.4])
    axis /= np.linalg.norm(axis)
    current = np.concatenate((axis, [0.0]))
    target = np.array([0.0, 0.0, 0.0, 1.0])
    error, linearization = _launch_rotation_error(
        np.stack((current, -current)), np.stack((target, target))
    )

    np.testing.assert_allclose(error[0], error[1], atol=2.0e-6)
    np.testing.assert_allclose(linearization[0], linearization[1], atol=2.0e-6)
    assert np.isfinite(error).all()
    assert np.isfinite(linearization).all()


def test_quaternion_cotangent_preserves_world_left_virtual_work():
    current = _axis_angle_quaternion([0.3, -0.8, 0.4], 1.3)
    angular_cotangent = np.array([0.7, -1.1, 0.2])
    output_wp = wp.zeros(1, dtype=wp.vec4, device="cpu")
    wp.launch(
        _quaternion_cotangent_kernel,
        dim=1,
        inputs=[
            wp.array(angular_cotangent[None], dtype=wp.vec3, device="cpu"),
            wp.array(current[None], dtype=wp.quat, device="cpu"),
            output_wp,
        ],
        device="cpu",
    )
    quaternion_cotangent = output_wp.numpy()[0]

    epsilon = 2.0e-4
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = epsilon
        plus_xyzw = _left_perturb_quaternion(current, delta)
        minus_xyzw = _left_perturb_quaternion(current, -delta)
        tangent_xyzw = (plus_xyzw - minus_xyzw) / (2.0 * epsilon)
        tangent_wxyz = tangent_xyzw[[3, 0, 1, 2]]
        np.testing.assert_allclose(
            quaternion_cotangent @ tangent_wxyz,
            angular_cotangent[axis],
            atol=3.0e-4,
            rtol=3.0e-4,
        )


@pytest.mark.parametrize("angle", [1.0e-4, 0.73, 2.4])
def test_rotation_error_linearization_matches_left_perturbation(angle):
    current = _axis_angle_quaternion([0.2, -0.7, 0.5], angle)
    target = _axis_angle_quaternion([-0.4, 0.1, 0.9], -0.61)
    error, linearization = _launch_rotation_error(current[None], target[None])
    epsilon = 2.0e-4
    finite_difference = np.empty((3, 3), dtype=np.float64)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = epsilon
        plus = _rotation_error_reference(_left_perturb_quaternion(current, delta), target)
        minus = _rotation_error_reference(_left_perturb_quaternion(current, -delta), target)
        finite_difference[:, axis] = (plus - minus) / (2.0 * epsilon)

    np.testing.assert_allclose(error[0], _rotation_error_reference(current, target), atol=2.0e-6)
    np.testing.assert_allclose(linearization[0], finite_difference, atol=7.0e-4, rtol=7.0e-4)


def _launch_pose_metric(
    current_position,
    target_position,
    current_quat,
    target_quat,
    position_projector,
    rotation_projector,
    position_scale,
    rotation_scale,
    tolerance,
    weight,
    device="cpu",
):
    count = len(current_position)
    inputs = [
        wp.array(current_position, dtype=wp.vec3, device=device),
        wp.array(target_position, dtype=wp.vec3, device=device),
        wp.array(current_quat, dtype=wp.quat, device=device),
        wp.array(target_quat, dtype=wp.quat, device=device),
        wp.array(position_projector, dtype=wp.mat33, device=device),
        wp.array(rotation_projector, dtype=wp.mat33, device=device),
        wp.array(position_scale, dtype=wp.vec3, device=device),
        wp.array(rotation_scale, dtype=wp.vec3, device=device),
        wp.array(tolerance, dtype=wp.vec2, device=device),
        wp.array(weight, dtype=wp.vec2, device=device),
    ]
    outputs = [
        wp.zeros(count, dtype=wp.vec2, device=device),
        wp.zeros(count, dtype=wp.vec2, device=device),
        wp.zeros(count, dtype=wp.vec3, device=device),
        wp.zeros(count, dtype=wp.vec3, device=device),
    ]
    wp.launch(
        _projected_pose_metric_kernel,
        dim=count,
        inputs=inputs + outputs,
        device=device,
    )
    return tuple(output.numpy() for output in outputs)


def test_projected_pose_metric_applies_projection_before_scale_and_has_correct_gradients():
    direction = np.array([1.0, -2.0, 0.7])
    direction /= np.linalg.norm(direction)
    position_projector = np.eye(3) - np.outer(direction, direction)
    rotation_projector = np.eye(3) - np.outer(direction[::-1], direction[::-1])
    current_position = np.array([[0.9, -0.2, 0.5]])
    target_position = np.array([[-0.1, 0.4, 0.2]])
    current_quat = _axis_angle_quaternion([0.3, -0.6, 0.4], 1.2)[None]
    target_quat = _axis_angle_quaternion([-0.2, 0.9, 0.1], -0.47)[None]
    position_scale = np.array([[2.0, 0.5, 1.4]])
    rotation_scale = np.array([[0.7, 1.8, 1.1]])
    tolerance = np.zeros((1, 2))
    weight = np.array([[1.3, 0.8]])

    cost, distance, position_gradient, rotation_gradient = _launch_pose_metric(
        current_position,
        target_position,
        current_quat,
        target_quat,
        position_projector[None],
        rotation_projector[None],
        position_scale,
        rotation_scale,
        tolerance,
        weight,
    )

    position_delta = current_position[0] - target_position[0]
    projected_position = position_projector @ position_delta
    scaled_position = position_scale[0] * projected_position
    expected_position_gradient = (
        weight[0, 0]
        * position_projector.T
        @ (position_scale[0] * position_scale[0] * projected_position)
    )
    rotation_error = _rotation_error_reference(current_quat[0], target_quat[0])
    scaled_rotation = rotation_scale[0] * (rotation_projector @ rotation_error)

    np.testing.assert_allclose(distance[0, 0], np.linalg.norm(scaled_position), rtol=2.0e-6)
    np.testing.assert_allclose(distance[0, 1], np.linalg.norm(scaled_rotation), rtol=2.0e-6)
    np.testing.assert_allclose(
        cost[0, 0], 0.5 * weight[0, 0] * scaled_position @ scaled_position, rtol=2.0e-6
    )
    np.testing.assert_allclose(
        cost[0, 1], weight[0, 1] * scaled_rotation @ scaled_rotation, rtol=2.0e-6
    )
    np.testing.assert_allclose(position_gradient[0], expected_position_gradient, atol=2.0e-6)

    epsilon = 2.0e-4
    finite_difference = np.empty(3)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = epsilon
        perturbed_quaternions = np.stack(
            (
                _left_perturb_quaternion(current_quat[0], delta),
                _left_perturb_quaternion(current_quat[0], -delta),
            )
        )
        perturbed_error = np.stack(
            [
                _rotation_error_reference(perturbed_quaternion, target_quat[0])
                for perturbed_quaternion in perturbed_quaternions
            ]
        )
        perturbed_scaled = rotation_scale[0] * (rotation_projector @ perturbed_error.T).T
        perturbed_cost = weight[0, 1] * np.sum(perturbed_scaled * perturbed_scaled, axis=-1)
        finite_difference[axis] = (perturbed_cost[0] - perturbed_cost[1]) / (2.0 * epsilon)

    np.testing.assert_allclose(rotation_gradient[0], finite_difference, atol=1.5e-3, rtol=1.5e-3)


def test_projected_pose_metric_zeroes_values_and_gradients_inside_tolerance():
    identity_quat = np.array([[0.0, 0.0, 0.0, 1.0]])
    cost, distance, position_gradient, rotation_gradient = _launch_pose_metric(
        np.array([[1.0e-3, 0.0, 0.0]]),
        np.zeros((1, 3)),
        _axis_angle_quaternion([1.0, 0.0, 0.0], 1.0e-3)[None],
        identity_quat,
        np.eye(3)[None],
        np.eye(3)[None],
        np.ones((1, 3)),
        np.ones((1, 3)),
        np.array([[2.0e-3, 2.0e-3]]),
        np.array([[4.0, 7.0]]),
    )

    np.testing.assert_array_equal(cost, 0.0)
    np.testing.assert_array_equal(distance, 0.0)
    np.testing.assert_array_equal(position_gradient, 0.0)
    np.testing.assert_array_equal(rotation_gradient, 0.0)


def test_projected_pose_metric_zero_projectors_disable_both_components():
    identity_quat = np.array([[0.0, 0.0, 0.0, 1.0]])
    cost, distance, position_gradient, rotation_gradient = _launch_pose_metric(
        np.array([[8.0, -4.0, 2.0]]),
        np.zeros((1, 3)),
        _axis_angle_quaternion([1.0, -2.0, 3.0], 2.1)[None],
        identity_quat,
        np.zeros((1, 3, 3)),
        np.zeros((1, 3, 3)),
        np.array([[2.0, 3.0, 4.0]]),
        np.array([[5.0, 6.0, 7.0]]),
        np.zeros((1, 2)),
        np.array([[11.0, 13.0]]),
    )

    np.testing.assert_array_equal(cost, 0.0)
    np.testing.assert_array_equal(distance, 0.0)
    np.testing.assert_array_equal(position_gradient, 0.0)
    np.testing.assert_array_equal(rotation_gradient, 0.0)


def test_projected_pose_metric_compiles_and_runs_on_cuda():
    if not wp.is_cuda_available():
        pytest.skip("CUDA not available")

    identity_quat = np.array([[0.0, 0.0, 0.0, 1.0]])
    cost, distance, position_gradient, rotation_gradient = _launch_pose_metric(
        np.array([[0.1, -0.2, 0.3]]),
        np.zeros((1, 3)),
        _axis_angle_quaternion([0.2, -0.4, 0.7], 0.5)[None],
        identity_quat,
        np.eye(3)[None],
        np.eye(3)[None],
        np.ones((1, 3)),
        np.ones((1, 3)),
        np.zeros((1, 2)),
        np.ones((1, 2)),
        device="cuda:0",
    )

    assert np.isfinite(cost).all()
    assert np.isfinite(distance).all()
    assert np.isfinite(position_gradient).all()
    assert np.isfinite(rotation_gradient).all()
