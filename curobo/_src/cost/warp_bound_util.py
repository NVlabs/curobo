# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Third Party
import warp as wp


@wp.func
def shrink_bounds_with_activation_distance(
    lower: wp.float32,
    upper: wp.float32,
    activation_distance: wp.float32,
):
    full_range = upper - lower

    new_lower = lower + (activation_distance * full_range)
    new_upper = upper - (activation_distance * full_range)

    return new_lower, new_upper


@wp.func
def aggregate_bound_cost(
    position: wp.float32,
    position_limit_lower: wp.float32,
    position_limit_upper: wp.float32,
    weight: wp.float32,
    cost: wp.float32,
    gradient: wp.float32,
):
    delta = wp.float32(0.0)
    if position < position_limit_lower:
        delta = position - position_limit_lower
    elif position > position_limit_upper:
        delta = position - position_limit_upper
    else:
        return cost, gradient

    cost, gradient = aggregate_squared_l2_regularization(delta, weight, cost, gradient)

    return cost, gradient


@wp.func
def aggregate_bound_cost_l1(
    position: wp.float32,
    position_limit_lower: wp.float32,
    position_limit_upper: wp.float32,
    weight: wp.float32,
    cost: wp.float32,
    gradient: wp.float32,
):
    delta = wp.float32(0.0)
    if position < position_limit_lower:
        delta = position - position_limit_lower
    elif position > position_limit_upper:
        delta = position - position_limit_upper
    else:
        return cost, gradient

    cost += weight * wp.abs(delta)
    gradient += weight * wp.sign(delta)

    # cost, gradient = aggregate_squared_l2_regularization(delta, weight, cost, gradient)

    return cost, gradient


@wp.func
def aggregate_squared_l2_regularization(
    value: wp.float32,
    weight: wp.float32,
    cost: wp.float32,
    gradient: wp.float32,
):
    w_v = weight * value
    cost += 0.5 * w_v * value
    gradient += w_v
    return cost, gradient


@wp.func
def aggregate_energy_regularization(
    torque: wp.float32,
    velocity: wp.float32,
    dt: wp.float32,
    weight: wp.float32,
    cost: wp.float32,
    gradient_torque: wp.float32,
    gradient_velocity: wp.float32,
):
    cost_energy = torque * velocity * dt
    cost += weight * cost_energy * cost_energy
    gradient_torque += 2.0 * weight * cost_energy * velocity * dt
    gradient_velocity += 2.0 * weight * cost_energy * torque * dt
    return cost, gradient_torque, gradient_velocity



