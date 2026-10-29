# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp glue kernels exported for the C++ trajectory-optimization runtime."""

from __future__ import annotations

from typing import Any

import warp as wp

from cpp_runtime.robot.kernels.lbfgs import (
    create_lbfgs_runtime_kernels,
)
from curobo._src.util.warp import warp_kernel


def create_trajopt_runtime_kernels(
    dof: int,
    n_knots: int,
    horizon: int,
    validation_horizon: int,
    num_seeds: int,
    num_line_search: int,
) -> dict[str, Any]:
    """Create fixed-shape optimizer orchestration kernels.

    Cost, collision, kinematics, B-spline, line-search, and L-BFGS math stays in
    the existing cuRobo kernels. These kernels only initialize, join, and select
    their buffers for a standalone native runtime.
    """
    for name, value in {
        "dof": dof,
        "n_knots": n_knots,
        "horizon": horizon,
        "validation_horizon": validation_horizon,
        "num_seeds": num_seeds,
        "num_line_search": num_line_search,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")

    fixed_dof = wp.constant(dof)
    fixed_knots = wp.constant(n_knots)
    fixed_horizon = wp.constant(horizon)
    fixed_validation_horizon = wp.constant(validation_horizon)
    opt_dim = wp.constant(dof * n_knots)
    combine_block = wp.constant(128)
    combine_chunks = wp.constant((dof * horizon + 127) // 128)

    def _prepare_pose_criteria_template(
        position_metric: wp.array(dtype=wp.mat33),
        rotation_metric: wp.array(dtype=wp.mat33),
        pose_tolerance: wp.array(dtype=wp.vec2),
        error_frame: wp.array(dtype=wp.int32),
        direction_x: wp.float32,
        direction_y: wp.float32,
        direction_z: wp.float32,
        path_position_tolerance: wp.float32,
        path_orientation_tolerance: wp.float32,
        direction_in_goal_frame: wp.uint8,
        num_tool_frames: wp.int32,
    ):
        criteria_idx = wp.tid()
        tool_idx = criteria_idx % num_tool_frames
        terminal = criteria_idx < num_tool_frames
        identity = wp.mat33(
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        position_projector = identity
        tolerance = wp.vec2(0.0, 0.0)
        if not terminal:
            direction = wp.normalize(wp.vec3(direction_x, direction_y, direction_z))
            position_projector = identity - wp.outer(direction, direction)
            tolerance = wp.vec2(path_position_tolerance, path_orientation_tolerance)
        position_metric[criteria_idx] = position_projector
        rotation_metric[criteria_idx] = identity
        pose_tolerance[criteria_idx] = tolerance
        if terminal:
            error_frame[tool_idx] = wp.int32(direction_in_goal_frame)

    prepare_pose_criteria = warp_kernel(
        f"trajopt_prepare_pose_criteria_{num_seeds}",
        enable_backward=False,
    )(_prepare_pose_criteria_template)

    def _initialize_actions_template(
        start_position: wp.array(dtype=wp.float32),
        start_velocity: wp.array(dtype=wp.float32),
        start_acceleration: wp.array(dtype=wp.float32),
        goal_position: wp.array(dtype=wp.float32),
        seed_start_position: wp.array2d(dtype=wp.float32),
        seed_goal_position: wp.array2d(dtype=wp.float32),
        request_seed_valid: wp.array(dtype=wp.uint8),
        request_seeds: wp.array2d(dtype=wp.float32),
        actions: wp.array2d(dtype=wp.float32),
        resolved_start_position: wp.array2d(dtype=wp.float32),
        resolved_start_velocity: wp.array2d(dtype=wp.float32),
        resolved_start_acceleration: wp.array2d(dtype=wp.float32),
        resolved_goal_position: wp.array2d(dtype=wp.float32),
        active_seed_valid: wp.array(dtype=wp.uint8),
        use_request_seeds: wp.uint8,
        use_seed_starts: wp.uint8,
        use_seed_goals: wp.uint8,
        use_seed_valid: wp.uint8,
        initial_dt: wp.float32,
        trajectory_dt: wp.array(dtype=wp.float32),
    ):
        seed_idx, variable_idx = wp.tid()
        joint_idx = variable_idx % fixed_dof
        knot_idx = variable_idx / fixed_dof
        seed_is_valid = wp.bool(True)
        if use_seed_valid == wp.uint8(1):
            seed_is_valid = request_seed_valid[seed_idx] != wp.uint8(0)
        start_value = start_position[joint_idx]
        if seed_is_valid and use_seed_starts == wp.uint8(1):
            start_value = seed_start_position[seed_idx, joint_idx]
        goal_value = goal_position[joint_idx]
        if seed_is_valid and use_seed_goals == wp.uint8(1):
            goal_value = seed_goal_position[seed_idx, joint_idx]
        value = request_seeds[seed_idx, variable_idx]
        if not seed_is_valid or use_request_seeds == wp.uint8(0):
            ratio = wp.float32(knot_idx) / wp.float32(fixed_knots - 1)
            value = start_value + ratio * (goal_value - start_value)
        actions[seed_idx, variable_idx] = value
        if knot_idx == 0:
            resolved_start_position[seed_idx, joint_idx] = start_value
            resolved_start_velocity[seed_idx, joint_idx] = start_velocity[joint_idx]
            resolved_start_acceleration[seed_idx, joint_idx] = start_acceleration[joint_idx]
            resolved_goal_position[seed_idx, joint_idx] = goal_value
            if joint_idx == 0:
                active_seed_valid[seed_idx] = wp.uint8(1) if seed_is_valid else wp.uint8(0)
                trajectory_dt[seed_idx] = initial_dt

    initialize_actions = warp_kernel(
        f"trajopt_initialize_actions_{dof}_{n_knots}_{num_seeds}",
        enable_backward=False,
    )(_initialize_actions_template)

    def _expand_trajectory_dt_template(
        trajectory_dt: wp.array(dtype=wp.float32),
        expanded_dt: wp.array(dtype=wp.float32),
        repeat_count: wp.int32,
    ):
        trajectory_idx = wp.tid()
        expanded_dt[trajectory_idx] = trajectory_dt[trajectory_idx / repeat_count]

    expand_trajectory_dt = warp_kernel(
        f"trajopt_expand_dt_{num_seeds}_{num_line_search}",
        enable_backward=False,
    )(_expand_trajectory_dt_template)

    def _reduce_required_dt_template(
        required_dt_per_timestep: wp.array(dtype=wp.float32),
        trajectory_dt: wp.array(dtype=wp.float32),
        dt_scale: wp.float32,
        minimum_dt: wp.float32,
        maximum_dt: wp.float32,
    ):
        seed_idx = wp.tid()
        required_dt = wp.float32(0.0)
        offset = seed_idx * fixed_horizon
        for timestep in range(fixed_horizon):
            required_dt = wp.max(required_dt, required_dt_per_timestep[offset + timestep])
        trajectory_dt[seed_idx] = wp.clamp(required_dt * dt_scale, minimum_dt, maximum_dt)

    reduce_required_dt = warp_kernel(
        f"trajopt_reduce_required_dt_{horizon}_{num_seeds}",
        enable_backward=False,
    )(_reduce_required_dt_template)

    def _calculate_interpolation_horizon_template(
        trajectory_dt: wp.array(dtype=wp.float32),
        interpolation_horizon: wp.array(dtype=wp.int32),
        interpolation_dt: wp.float32,
        sparse_interpolation_steps: wp.int32,
        total_knots: wp.int32,
    ):
        seed_idx = wp.tid()
        knot_dt = trajectory_dt[seed_idx] * wp.float32(sparse_interpolation_steps)
        steps_per_waypoint = wp.int32((knot_dt + interpolation_dt) / interpolation_dt)
        interpolation_horizon[seed_idx] = total_knots * steps_per_waypoint + 1

    calculate_interpolation_horizon = warp_kernel(
        f"trajopt_interpolation_horizon_{num_seeds}",
        enable_backward=False,
    )(_calculate_interpolation_horizon_template)

    def _validate_interpolated_template(
        position: wp.array(dtype=wp.float32),
        velocity: wp.array(dtype=wp.float32),
        acceleration: wp.array(dtype=wp.float32),
        jerk: wp.array(dtype=wp.float32),
        effort: wp.array(dtype=wp.float32),
        world_collision_cost: wp.array(dtype=wp.float32),
        self_collision_cost: wp.array(dtype=wp.float32),
        position_lower: wp.array(dtype=wp.float32),
        position_upper: wp.array(dtype=wp.float32),
        velocity_lower: wp.array(dtype=wp.float32),
        velocity_upper: wp.array(dtype=wp.float32),
        acceleration_lower: wp.array(dtype=wp.float32),
        acceleration_upper: wp.array(dtype=wp.float32),
        jerk_lower: wp.array(dtype=wp.float32),
        jerk_upper: wp.array(dtype=wp.float32),
        effort_lower: wp.array(dtype=wp.float32),
        effort_upper: wp.array(dtype=wp.float32),
        violation: wp.array(dtype=wp.int32),
        constraint_tolerance: wp.float32,
        effort_active: wp.uint8,
    ):
        value_idx = wp.tid()
        joint_idx = value_idx % fixed_dof
        state_flat_idx = value_idx / fixed_dof
        seed_idx = state_flat_idx / fixed_validation_horizon
        p = position[value_idx]
        v = velocity[value_idx]
        a = acceleration[value_idx]
        j = jerk[value_idx]
        effort_violation = wp.bool(False)
        if effort_active == wp.uint8(1):
            tau = effort[value_idx]
            effort_violation = (
                tau < effort_lower[joint_idx] - constraint_tolerance
                or tau > effort_upper[joint_idx] + constraint_tolerance
                or not wp.isfinite(tau)
            )
        invalid = (
            p < position_lower[joint_idx] - constraint_tolerance
            or p > position_upper[joint_idx] + constraint_tolerance
            or v < velocity_lower[joint_idx] - constraint_tolerance
            or v > velocity_upper[joint_idx] + constraint_tolerance
            or a < acceleration_lower[joint_idx] - constraint_tolerance
            or a > acceleration_upper[joint_idx] + constraint_tolerance
            or j < jerk_lower[joint_idx] - constraint_tolerance
            or j > jerk_upper[joint_idx] + constraint_tolerance
            or not wp.isfinite(p)
            or not wp.isfinite(v)
            or not wp.isfinite(a)
            or not wp.isfinite(j)
            or effort_violation
        )
        if joint_idx == 0:
            collision = world_collision_cost[state_flat_idx] + self_collision_cost[state_flat_idx]
            invalid = invalid or collision > constraint_tolerance or not wp.isfinite(collision)
        if invalid:
            wp.atomic_max(violation, seed_idx, wp.int32(1))

    validate_interpolated = warp_kernel(
        f"trajopt_validate_interpolated_{dof}_{validation_horizon}_{num_seeds}",
        enable_backward=False,
    )(_validate_interpolated_template)

    lbfgs_kernels = create_lbfgs_runtime_kernels(
        dof * n_knots,
        num_line_search,
        "trajopt",
    )

    def _combine_rollout_template(
        cspace_cost: wp.array(dtype=wp.float32),
        cspace_gradient_position: wp.array(dtype=wp.float32),
        cspace_gradient_velocity: wp.array(dtype=wp.float32),
        cspace_gradient_acceleration: wp.array(dtype=wp.float32),
        cspace_gradient_jerk: wp.array(dtype=wp.float32),
        effort_gradient_position: wp.array(dtype=wp.float32),
        effort_gradient_velocity: wp.array(dtype=wp.float32),
        effort_gradient_acceleration: wp.array(dtype=wp.float32),
        collision_joint_gradient: wp.array(dtype=wp.float32),
        world_collision_cost: wp.array(dtype=wp.float32),
        self_collision_cost: wp.array(dtype=wp.float32),
        pose_cost: wp.array(dtype=wp.float32),
        num_tool_frames: wp.int32,
        pose_constraint_active: wp.uint8,
        effort_active: wp.uint8,
        dense_gradient_position: wp.array(dtype=wp.float32),
        dense_gradient_velocity: wp.array(dtype=wp.float32),
        dense_gradient_acceleration: wp.array(dtype=wp.float32),
        dense_gradient_jerk: wp.array(dtype=wp.float32),
        trajectory_cost: wp.array(dtype=wp.float32),
    ):
        trajectory_idx, lane = wp.tid()
        state_width = fixed_horizon * fixed_dof
        trajectory_offset = trajectory_idx * state_width
        local_cost = wp.float32(0.0)
        for chunk_idx in range(combine_chunks):
            relative_index = lane + chunk_idx * combine_block
            if relative_index < state_width:
                index = trajectory_offset + relative_index
                joint_idx = relative_index % fixed_dof
                state_idx = relative_index / fixed_dof
                collision_idx = trajectory_idx * fixed_horizon + state_idx
                value_cost = cspace_cost[index]
                if joint_idx == 0:
                    value_cost += (
                        world_collision_cost[collision_idx] + self_collision_cost[collision_idx]
                    )
                    if pose_constraint_active == wp.uint8(1):
                        pose_offset = collision_idx * num_tool_frames * 2
                        for pose_idx in range(2 * num_tool_frames):
                            value_cost += pose_cost[pose_offset + pose_idx]
                local_cost += value_cost
                dense_gradient_position[index] = (
                    cspace_gradient_position[index] + collision_joint_gradient[index]
                )
                dense_gradient_velocity[index] = cspace_gradient_velocity[index]
                dense_gradient_acceleration[index] = cspace_gradient_acceleration[index]
                dense_gradient_jerk[index] = cspace_gradient_jerk[index]
                if effort_active == wp.uint8(1):
                    dense_gradient_position[index] += effort_gradient_position[index]
                    dense_gradient_velocity[index] += effort_gradient_velocity[index]
                    dense_gradient_acceleration[index] += effort_gradient_acceleration[index]

        block_cost = wp.tile_sum(wp.tile(local_cost))
        if lane == 0:
            trajectory_cost[trajectory_idx] = block_cost[0]

    combine_rollout = warp_kernel(
        f"trajopt_combine_rollout_{dof}_{horizon}", enable_backward=False
    )(_combine_rollout_template)

    def _accumulate_template(
        candidate_knots: wp.array2d(dtype=wp.float32),
        candidate_cost: wp.array(dtype=wp.float32),
        position: wp.array(dtype=wp.float32),
        velocity: wp.array(dtype=wp.float32),
        acceleration: wp.array(dtype=wp.float32),
        jerk: wp.array(dtype=wp.float32),
        effort: wp.array(dtype=wp.float32),
        world_collision_cost: wp.array(dtype=wp.float32),
        self_collision_cost: wp.array(dtype=wp.float32),
        pose_position_distance: wp.array(dtype=wp.float32),
        pose_orientation_distance: wp.array(dtype=wp.float32),
        position_lower: wp.array(dtype=wp.float32),
        position_upper: wp.array(dtype=wp.float32),
        velocity_lower: wp.array(dtype=wp.float32),
        velocity_upper: wp.array(dtype=wp.float32),
        acceleration_lower: wp.array(dtype=wp.float32),
        acceleration_upper: wp.array(dtype=wp.float32),
        jerk_lower: wp.array(dtype=wp.float32),
        jerk_upper: wp.array(dtype=wp.float32),
        effort_lower: wp.array(dtype=wp.float32),
        effort_upper: wp.array(dtype=wp.float32),
        trajectory_dt: wp.array(dtype=wp.float32),
        retained_knots: wp.array2d(dtype=wp.float32),
        retained_position: wp.array(dtype=wp.float32),
        retained_velocity: wp.array(dtype=wp.float32),
        retained_acceleration: wp.array(dtype=wp.float32),
        retained_jerk: wp.array(dtype=wp.float32),
        retained_success: wp.array(dtype=wp.uint8),
        retained_pose_valid: wp.array(dtype=wp.uint8),
        retained_cost: wp.array(dtype=wp.float32),
        retained_rank_score: wp.array(dtype=wp.float32),
        retained_collision_cost: wp.array(dtype=wp.float32),
        retained_dt: wp.array(dtype=wp.float32),
        active_seed_valid: wp.array(dtype=wp.uint8),
        interpolated_violation: wp.array(dtype=wp.int32),
        finetune_enabled: wp.array(dtype=wp.int32),
        first_pass: wp.uint8,
        constraint_tolerance: wp.float32,
        pose_position_tolerance: wp.float32,
        pose_orientation_tolerance: wp.float32,
        path_position_tolerance: wp.float32,
        path_orientation_tolerance: wp.float32,
        num_tool_frames: wp.int32,
        pose_constraint_active: wp.uint8,
        effort_active: wp.uint8,
    ):
        seed_idx, lane = wp.tid()
        feasible = active_seed_valid[seed_idx] == wp.uint8(1) and interpolated_violation[
            seed_idx
        ] == wp.int32(0)
        collision_total = wp.float32(0.0)
        pose_valid = wp.bool(True)
        state_offset = seed_idx * fixed_horizon * fixed_dof
        acceleration_total = wp.float32(0.0)
        jerk_total = wp.float32(0.0)
        # One block per seed; validate and retain state/joint values in parallel.
        for relative_index in range(lane, fixed_horizon * fixed_dof, combine_block):
            state_idx = relative_index / fixed_dof
            joint_idx = relative_index % fixed_dof
            collision_idx = seed_idx * fixed_horizon + state_idx
            if joint_idx == 0:
                collision_total += (
                    world_collision_cost[collision_idx] + self_collision_cost[collision_idx]
                )
            index = state_offset + state_idx * fixed_dof + joint_idx
            p = position[index]
            v = velocity[index]
            a = acceleration[index]
            j = jerk[index]
            effort_violation = wp.bool(False)
            if effort_active == wp.uint8(1):
                tau = effort[index]
                effort_violation = (
                    tau < effort_lower[joint_idx] - constraint_tolerance
                    or tau > effort_upper[joint_idx] + constraint_tolerance
                    or not wp.isfinite(tau)
                )
            if (
                p < position_lower[joint_idx] - constraint_tolerance
                or p > position_upper[joint_idx] + constraint_tolerance
                or v < velocity_lower[joint_idx] - constraint_tolerance
                or v > velocity_upper[joint_idx] + constraint_tolerance
                or a < acceleration_lower[joint_idx] - constraint_tolerance
                or a > acceleration_upper[joint_idx] + constraint_tolerance
                or j < jerk_lower[joint_idx] - constraint_tolerance
                or j > jerk_upper[joint_idx] + constraint_tolerance
                or not wp.isfinite(p)
                or not wp.isfinite(v)
                or not wp.isfinite(a)
                or not wp.isfinite(j)
                or effort_violation
            ):
                feasible = False

            if state_idx >= 8 and state_idx < fixed_horizon - 8:
                acceleration_total += wp.abs(a)
                jerk_total += wp.abs(j)
            if joint_idx == 0 and pose_constraint_active == wp.uint8(1):
                position_limit = path_position_tolerance
                orientation_limit = path_orientation_tolerance
                if state_idx == fixed_horizon - 1:
                    position_limit = pose_position_tolerance
                    orientation_limit = pose_orientation_tolerance
                pose_offset = collision_idx * num_tool_frames
                for tool_idx in range(num_tool_frames):
                    pose_idx = pose_offset + tool_idx
                    if (
                        pose_position_distance[pose_idx] > position_limit
                        or pose_orientation_distance[pose_idx] > orientation_limit
                        or not wp.isfinite(pose_position_distance[pose_idx])
                        or not wp.isfinite(pose_orientation_distance[pose_idx])
                    ):
                        feasible = False
                        pose_valid = False

        collision_total = wp.tile_sum(wp.tile(collision_total))[0]
        feasible = wp.tile_min(wp.tile(wp.int32(feasible)))[0] != 0
        pose_valid = wp.tile_min(wp.tile(wp.int32(pose_valid)))[0] != 0
        acceleration_total = wp.tile_sum(wp.tile(acceleration_total))[0]
        jerk_total = wp.tile_sum(wp.tile(jerk_total))[0]
        if collision_total > constraint_tolerance:
            feasible = False

        should_update = first_pass == wp.uint8(1)
        if lane == 0 and first_pass == wp.uint8(1) and feasible:
            wp.atomic_max(finetune_enabled, 0, wp.int32(1))
        if first_pass == wp.uint8(0):
            should_update = (
                finetune_enabled[0] == wp.int32(1)
                and feasible
                and trajectory_dt[seed_idx] <= retained_dt[seed_idx]
            )
        if should_update:
            for variable_idx in range(lane, opt_dim, combine_block):
                retained_knots[seed_idx, variable_idx] = candidate_knots[seed_idx, variable_idx]
            for index in range(lane, fixed_horizon * fixed_dof, combine_block):
                source_index = state_offset + index
                retained_position[source_index] = position[source_index]
                retained_velocity[source_index] = velocity[source_index]
                retained_acceleration[source_index] = acceleration[source_index]
                retained_jerk[source_index] = jerk[source_index]
            if lane == 0:
                rank_score = candidate_cost[seed_idx]
                if feasible:
                    sample_count = wp.float32((fixed_horizon - 16) * fixed_dof)
                    rank_score = (
                        trajectory_dt[seed_idx] * 1000.0
                        + acceleration_total / sample_count * 0.01
                        + jerk_total / sample_count * 0.001
                    )
                retained_rank_score[seed_idx] = rank_score
                retained_success[seed_idx] = wp.uint8(1) if feasible else wp.uint8(0)
                retained_pose_valid[seed_idx] = wp.uint8(1) if pose_valid else wp.uint8(0)
                retained_cost[seed_idx] = candidate_cost[seed_idx]
                retained_collision_cost[seed_idx] = collision_total
                retained_dt[seed_idx] = trajectory_dt[seed_idx]

    accumulate = warp_kernel(
        f"trajopt_accumulate_{dof}_{n_knots}_{horizon}_{num_seeds}",
        enable_backward=False,
    )(_accumulate_template)

    def _finalize_template(
        retained_knots: wp.array2d(dtype=wp.float32),
        retained_position: wp.array(dtype=wp.float32),
        retained_velocity: wp.array(dtype=wp.float32),
        retained_acceleration: wp.array(dtype=wp.float32),
        retained_jerk: wp.array(dtype=wp.float32),
        retained_success: wp.array(dtype=wp.uint8),
        retained_cost: wp.array(dtype=wp.float32),
        retained_rank_score: wp.array(dtype=wp.float32),
        retained_collision_cost: wp.array(dtype=wp.float32),
        retained_dt: wp.array(dtype=wp.float32),
        output_knots: wp.array(dtype=wp.float32),
        output_position: wp.array(dtype=wp.float32),
        output_velocity: wp.array(dtype=wp.float32),
        output_acceleration: wp.array(dtype=wp.float32),
        output_jerk: wp.array(dtype=wp.float32),
        output_success: wp.array(dtype=wp.uint8),
        output_cost: wp.array(dtype=wp.float32),
        output_collision_cost: wp.array(dtype=wp.float32),
        output_dt: wp.array(dtype=wp.float32),
        output_selected_seed: wp.array(dtype=wp.int32),
        active_seeds: wp.int32,
    ):
        output_idx, lane = wp.tid()
        best_seed = wp.int32(0)
        best_score = wp.float32(1.0e30)
        found_feasible = wp.bool(False)
        for seed_idx in range(lane, active_seeds, combine_block):
            feasible = retained_success[seed_idx] == wp.uint8(1)
            score = retained_rank_score[seed_idx]
            if (feasible and not found_feasible) or (
                feasible == found_feasible and score < best_score
            ):
                found_feasible = feasible
                best_seed = seed_idx
                best_score = score

        any_feasible = wp.tile_max(wp.tile(wp.int32(found_feasible)))[0]
        candidate_score = wp.float32(wp.inf)
        if wp.int32(found_feasible) == any_feasible:
            candidate_score = best_score
        minimum_score = wp.tile_min(wp.tile(candidate_score))[0]
        candidate_seed = wp.int32(active_seeds)
        if wp.int32(found_feasible) == any_feasible and best_score == minimum_score:
            candidate_seed = best_seed
        best_seed = wp.tile_min(wp.tile(candidate_seed))[0]
        found_feasible = any_feasible != 0

        for variable_idx in range(lane, opt_dim, combine_block):
            output_knots[variable_idx] = retained_knots[best_seed, variable_idx]
        source_offset = best_seed * fixed_horizon * fixed_dof
        for index in range(lane, fixed_horizon * fixed_dof, combine_block):
            output_position[index] = retained_position[source_offset + index]
            output_velocity[index] = retained_velocity[source_offset + index]
            output_acceleration[index] = retained_acceleration[source_offset + index]
            output_jerk[index] = retained_jerk[source_offset + index]
        if lane == 0:
            output_success[0] = wp.uint8(1) if found_feasible else wp.uint8(0)
            output_cost[0] = retained_cost[best_seed]
            output_collision_cost[0] = retained_collision_cost[best_seed]
            output_dt[0] = retained_dt[best_seed]
            output_selected_seed[0] = best_seed if found_feasible else wp.int32(-1)

    finalize = warp_kernel(
        f"trajopt_finalize_{dof}_{n_knots}_{horizon}_{num_seeds}",
        enable_backward=False,
    )(_finalize_template)

    return {
        "initialize_actions": initialize_actions,
        "prepare_pose_criteria": prepare_pose_criteria,
        "expand_trajectory_dt": expand_trajectory_dt,
        "reduce_required_dt": reduce_required_dt,
        "calculate_interpolation_horizon": calculate_interpolation_horizon,
        "validate_interpolated": validate_interpolated,
        **lbfgs_kernels,
        "combine_rollout": combine_rollout,
        "accumulate": accumulate,
        "finalize": finalize,
    }
