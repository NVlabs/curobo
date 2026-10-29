# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp glue kernels exported for the C++ inverse-kinematics runtime."""

from __future__ import annotations

from typing import Any

import warp as wp

from cpp_runtime.robot.kernels.lbfgs import (
    create_lbfgs_runtime_kernels,
)
from curobo._src.cost.wp_pose_metric import compute_quadratic_pose_metric
from curobo._src.util.warp import warp_kernel


def create_seed_ik_kernels(
    dof: int,
    num_links: int,
    num_line_search: int,
) -> dict[str, Any]:
    """Create dimension-specialized kernels used by native IK optimizers."""
    if dof < 1:
        raise ValueError("dof must be positive")
    if num_links < 1:
        raise ValueError("num_links must be positive")
    if num_line_search < 1:
        raise ValueError("num_line_search must be positive")

    tile_threads = wp.constant(128)
    coord = wp.constant(dof)
    links = wp.constant(num_links)
    pose_residuals = wp.constant(6 * num_links)
    residuals = wp.constant(6 * num_links + dof)

    def _initialize_template(
        template_seeds: wp.array2d(dtype=wp.float32),
        request_seeds: wp.array2d(dtype=wp.float32),
        current_position: wp.array2d(dtype=wp.float32),
        default_position: wp.array(dtype=wp.float32),
        action_min: wp.array(dtype=wp.float32),
        action_max: wp.array(dtype=wp.float32),
        joint_position: wp.array2d(dtype=wp.float32),
        damping: wp.array(dtype=wp.float32),
        num_seeds: wp.int32,
        max_template_seeds: wp.int32,
        seed_offset: wp.array(dtype=wp.int32),
        use_request_seeds: wp.uint8,
        use_current_position: wp.uint8,
        initial_damping: wp.float32,
    ):
        problem_idx, joint_idx = wp.tid()
        seed_idx = problem_idx % num_seeds
        batch_idx = problem_idx / num_seeds
        template_idx = seed_idx
        if use_current_position == wp.uint8(1) and seed_idx > 0:
            template_idx = seed_idx - 1
        template_idx = (template_idx + seed_offset[0]) % max_template_seeds
        value = template_seeds[template_idx, joint_idx]
        if use_current_position == wp.uint8(1) and seed_idx == 0:
            value = current_position[batch_idx, joint_idx]
        elif use_current_position == wp.uint8(0) and seed_idx == num_seeds - 1:
            value = default_position[joint_idx]
        if use_request_seeds == wp.uint8(1):
            value = request_seeds[problem_idx, joint_idx]
        joint_position[problem_idx, joint_idx] = value
        if joint_idx == 0:
            damping[problem_idx] = initial_damping

    initialize = warp_kernel(f"seed_ik_initialize_{dof}_{num_links}", enable_backward=False)(
        _initialize_template
    )

    def _error_template(
        current_position: wp.array(dtype=wp.vec3),
        current_quaternion: wp.array(dtype=wp.vec4),
        goal_position: wp.array(dtype=wp.vec3),
        goal_quaternion: wp.array(dtype=wp.vec4),
        selected_goalset_index: wp.array(dtype=wp.int32),
        raw_jacobian: wp.array3d(dtype=wp.float32),
        joint_position: wp.array2d(dtype=wp.float32),
        action_min: wp.array(dtype=wp.float32),
        action_max: wp.array(dtype=wp.float32),
        scaled_jacobian: wp.array3d(dtype=wp.float32),
        jt_error: wp.array2d(dtype=wp.float32),
        error_norm: wp.array(dtype=wp.float32),
        position_error: wp.array(dtype=wp.float32),
        orientation_error: wp.array(dtype=wp.float32),
        num_seeds: wp.int32,
        num_goalset: wp.int32,
        position_weight: wp.float32,
        orientation_weight: wp.float32,
        joint_limit_weight: wp.float32,
        include_joint_limits: wp.uint8,
    ):
        problem_idx, lane = wp.tid()
        goal_idx = problem_idx / num_seeds
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
        total_cost = wp.float32(0.0)
        max_position_error = wp.float32(0.0)
        max_orientation_error = wp.float32(0.0)

        if lane == 0:
            for joint_idx in range(coord):
                jt_error[problem_idx, joint_idx] = 0.0

        for link_idx in range(links):
            pose_idx = problem_idx * links + link_idx
            current_quat_wxyz = current_quaternion[pose_idx]
            current_quat = wp.quat(
                current_quat_wxyz[1],
                current_quat_wxyz[2],
                current_quat_wxyz[3],
                current_quat_wxyz[0],
            )
            # One block per seed; goals are distributed across its lanes.
            best_pose_cost = wp.float32(1.0e30)
            best_goalset_idx = wp.int32(num_goalset)
            goalset_offset = (goal_idx * links + link_idx) * num_goalset
            for goalset_idx in range(lane, num_goalset, tile_threads):
                goal_pose_idx = goalset_offset + goalset_idx
                goal_quat_wxyz = goal_quaternion[goal_pose_idx]
                goal_quat = wp.quat(
                    goal_quat_wxyz[1],
                    goal_quat_wxyz[2],
                    goal_quat_wxyz[3],
                    goal_quat_wxyz[0],
                )
                (
                    position_cost,
                    rotation_cost,
                    linear_distance,
                    angular_distance,
                    position_gradient,
                    rotation_gradient,
                ) = compute_quadratic_pose_metric(
                    current_position[pose_idx],
                    goal_position[goal_pose_idx],
                    current_quat,
                    goal_quat,
                    identity,
                    identity,
                    0.0,
                    0.0,
                    position_weight,
                    orientation_weight,
                )
                pose_cost = position_cost + rotation_cost
                if pose_cost < best_pose_cost:
                    best_pose_cost = pose_cost
                    best_goalset_idx = goalset_idx
            minimum_cost = wp.tile_min(wp.tile(best_pose_cost))[0]
            local_index = wp.int32(num_goalset)
            if best_pose_cost == minimum_cost:
                local_index = best_goalset_idx
            winner = wp.tile_min(wp.tile(local_index))[0]

            if lane == 0:
                # Reuse the exact pose metric for the winning goal and its gradient.
                best_goalset_idx = wp.int32(0)
                best_position_cost = wp.float32(0.0)
                best_rotation_cost = wp.float32(0.0)
                best_linear_distance = wp.float32(0.0)
                best_angular_distance = wp.float32(0.0)
                best_position_gradient = wp.vec3(0.0, 0.0, 0.0)
                best_rotation_gradient = wp.vec3(0.0, 0.0, 0.0)
                if winner < num_goalset:
                    best_goalset_idx = winner
                    goal_pose_idx = goalset_offset + winner
                    goal_quat_wxyz = goal_quaternion[goal_pose_idx]
                    goal_quat = wp.quat(
                        goal_quat_wxyz[1],
                        goal_quat_wxyz[2],
                        goal_quat_wxyz[3],
                        goal_quat_wxyz[0],
                    )
                    (
                        best_position_cost,
                        best_rotation_cost,
                        best_linear_distance,
                        best_angular_distance,
                        best_position_gradient,
                        best_rotation_gradient,
                    ) = compute_quadratic_pose_metric(
                        current_position[pose_idx],
                        goal_position[goal_pose_idx],
                        current_quat,
                        goal_quat,
                        identity,
                        identity,
                        0.0,
                        0.0,
                        position_weight,
                        orientation_weight,
                    )

                selected_goalset_index[problem_idx * links + link_idx] = best_goalset_idx
                total_cost += best_position_cost + best_rotation_cost
                max_position_error = wp.max(max_position_error, best_linear_distance)
                max_orientation_error = wp.max(max_orientation_error, best_angular_distance)

                row_offset = link_idx * 6
                for joint_idx in range(coord):
                    linear_contribution = (
                        raw_jacobian[problem_idx, row_offset + 0, joint_idx]
                        * best_position_gradient[0]
                        + raw_jacobian[problem_idx, row_offset + 1, joint_idx]
                        * best_position_gradient[1]
                        + raw_jacobian[problem_idx, row_offset + 2, joint_idx]
                        * best_position_gradient[2]
                    )
                    angular_contribution = (
                        raw_jacobian[problem_idx, row_offset + 3, joint_idx]
                        * best_rotation_gradient[0]
                        + raw_jacobian[problem_idx, row_offset + 4, joint_idx]
                        * best_rotation_gradient[1]
                        + raw_jacobian[problem_idx, row_offset + 5, joint_idx]
                        * best_rotation_gradient[2]
                    )
                    jt_error[problem_idx, joint_idx] += linear_contribution + angular_contribution
                    scaled_jacobian[problem_idx, row_offset + 0, joint_idx] = (
                        wp.sqrt(position_weight) * raw_jacobian[problem_idx, row_offset + 0, joint_idx]
                    )
                    scaled_jacobian[problem_idx, row_offset + 1, joint_idx] = (
                        wp.sqrt(position_weight) * raw_jacobian[problem_idx, row_offset + 1, joint_idx]
                    )
                    scaled_jacobian[problem_idx, row_offset + 2, joint_idx] = (
                        wp.sqrt(position_weight) * raw_jacobian[problem_idx, row_offset + 2, joint_idx]
                    )
                    scaled_jacobian[problem_idx, row_offset + 3, joint_idx] = (
                        wp.sqrt(2.0 * orientation_weight)
                        * raw_jacobian[problem_idx, row_offset + 3, joint_idx]
                    )
                    scaled_jacobian[problem_idx, row_offset + 4, joint_idx] = (
                        wp.sqrt(2.0 * orientation_weight)
                        * raw_jacobian[problem_idx, row_offset + 4, joint_idx]
                    )
                    scaled_jacobian[problem_idx, row_offset + 5, joint_idx] = (
                        wp.sqrt(2.0 * orientation_weight)
                        * raw_jacobian[problem_idx, row_offset + 5, joint_idx]
                    )

        if lane == 0:
            for joint_idx in range(coord):
                value = joint_position[problem_idx, joint_idx]
                limit_jacobian = wp.float32(0.0)
                if include_joint_limits == wp.uint8(1):
                    lower_error = wp.max(action_min[joint_idx] - value, 0.0)
                    upper_error = wp.max(value - action_max[joint_idx], 0.0)
                    limit_error = joint_limit_weight * (lower_error + upper_error)
                    if lower_error > 0.0:
                        limit_jacobian -= joint_limit_weight
                    if upper_error > 0.0:
                        limit_jacobian += joint_limit_weight
                    jt_error[problem_idx, joint_idx] += limit_jacobian * limit_error
                    total_cost += limit_error
                limit_row = pose_residuals + joint_idx
                for column_idx in range(coord):
                    scaled_jacobian[problem_idx, limit_row, column_idx] = 0.0
                scaled_jacobian[problem_idx, limit_row, joint_idx] = limit_jacobian

            error_norm[problem_idx] = total_cost
            position_error[problem_idx] = max_position_error
            orientation_error[problem_idx] = max_orientation_error

    error = warp_kernel(f"seed_ik_error_{dof}_{num_links}", enable_backward=False)(_error_template)

    def _update_template(
        joint_position: wp.array2d(dtype=wp.float32),
        candidate: wp.array2d(dtype=wp.float32),
        current_error: wp.array(dtype=wp.float32),
        candidate_error: wp.array(dtype=wp.float32),
        predicted_reduction: wp.array(dtype=wp.float32),
        damping: wp.array(dtype=wp.float32),
        rho_min: wp.float32,
        damping_factor: wp.float32,
        damping_min: wp.float32,
        damping_max: wp.float32,
    ):
        problem_idx = wp.tid()
        denominator = predicted_reduction[problem_idx] + 1.0e-8
        trust_ratio = (current_error[problem_idx] - candidate_error[problem_idx]) / denominator
        accepted = trust_ratio >= rho_min and wp.isfinite(candidate_error[problem_idx])
        current_damping = damping[problem_idx]
        if accepted:
            for joint_idx in range(coord):
                joint_position[problem_idx, joint_idx] = candidate[problem_idx, joint_idx]
            current_damping = current_damping / damping_factor
        else:
            current_damping = current_damping * damping_factor
        damping[problem_idx] = wp.clamp(current_damping, damping_min, damping_max)

    update = warp_kernel(f"seed_ik_update_{dof}_{num_links}", enable_backward=False)(
        _update_template
    )

    lbfgs_kernels = create_lbfgs_runtime_kernels(dof, num_line_search, "ik")

    def _combine_lbfgs_rollout_template(
        pose_cost: wp.array(dtype=wp.float32),
        pose_gradient: wp.array2d(dtype=wp.float32),
        cspace_cost: wp.array(dtype=wp.float32),
        cspace_gradient: wp.array(dtype=wp.float32),
        effort_position_gradient: wp.array(dtype=wp.float32),
        collision_joint_gradient: wp.array2d(dtype=wp.float32),
        world_collision_cost: wp.array(dtype=wp.float32),
        self_collision_cost: wp.array(dtype=wp.float32),
    ):
        problem_idx = wp.tid()
        total_cost = (
            pose_cost[problem_idx]
            + world_collision_cost[problem_idx]
            + self_collision_cost[problem_idx]
        )
        offset = problem_idx * coord
        for joint_idx in range(coord):
            total_cost += cspace_cost[offset + joint_idx]
            pose_gradient[problem_idx, joint_idx] += (
                cspace_gradient[offset + joint_idx]
                + effort_position_gradient[offset + joint_idx]
                + collision_joint_gradient[problem_idx, joint_idx]
            )
        pose_cost[problem_idx] = total_cost

    combine_lbfgs_rollout = warp_kernel(
        f"ik_combine_lbfgs_rollout_{dof}_{num_links}",
        enable_backward=False,
    )(_combine_lbfgs_rollout_template)

    def _rank_template(
        joint_position: wp.array2d(dtype=wp.float32),
        current_position: wp.array2d(dtype=wp.float32),
        position_bounds: wp.array(dtype=wp.float32),
        position_error: wp.array(dtype=wp.float32),
        orientation_error: wp.array(dtype=wp.float32),
        collision_cost: wp.array(dtype=wp.float32),
        self_collision_cost: wp.array(dtype=wp.float32),
        effort: wp.array(dtype=wp.float32),
        effort_bounds: wp.array(dtype=wp.float32),
        start_cspace_weight: wp.array(dtype=wp.float32),
        start_cspace_dof_weight: wp.array(dtype=wp.float32),
        rank_success: wp.array(dtype=wp.uint8),
        rank_score: wp.array(dtype=wp.float32),
        num_seeds: wp.int32,
        use_current_position: wp.uint8,
        position_tolerance: wp.float32,
        orientation_tolerance: wp.float32,
        position_rank_weight: wp.float32,
        orientation_rank_weight: wp.float32,
        cspace_rank_weight: wp.float32,
        effort_active: wp.uint8,
    ):
        problem_idx = wp.tid()
        batch_idx = problem_idx / num_seeds
        candidate_pose_success = (
            position_error[problem_idx] < position_tolerance
            and orientation_error[problem_idx] < orientation_tolerance
        )
        candidate_bounds_success = wp.bool(True)
        for joint_idx in range(coord):
            value = joint_position[problem_idx, joint_idx]
            tau = effort[problem_idx * coord + joint_idx]
            if (
                not wp.isfinite(value)
                or value < position_bounds[joint_idx]
                or value > position_bounds[coord + joint_idx]
                or (
                    effort_active == wp.uint8(1)
                    and (
                        not wp.isfinite(tau)
                        or tau < effort_bounds[joint_idx]
                        or tau > effort_bounds[coord + joint_idx]
                    )
                )
            ):
                candidate_bounds_success = False
        candidate_collision_cost = (
            collision_cost[problem_idx] + self_collision_cost[problem_idx]
        )
        candidate_success = (
            candidate_pose_success
            and candidate_bounds_success
            and candidate_collision_cost <= 0.0
        )
        candidate_cspace = wp.float32(0.0)
        if use_current_position == wp.uint8(1):
            for joint_idx in range(coord):
                delta = (
                    joint_position[problem_idx, joint_idx]
                    - current_position[batch_idx, joint_idx]
                )
                candidate_cspace += (
                    start_cspace_weight[0] * start_cspace_dof_weight[joint_idx] * delta * delta
                )
        candidate_score = (
            position_rank_weight * position_error[problem_idx]
            + orientation_rank_weight * orientation_error[problem_idx]
            + cspace_rank_weight * candidate_cspace
        )

        rank_success[problem_idx] = wp.uint8(candidate_success)
        rank_score[problem_idx] = candidate_score

    rank = warp_kernel(f"seed_ik_rank_{dof}_{num_links}", enable_backward=False)(_rank_template)

    def _finalize_template(
        joint_position: wp.array2d(dtype=wp.float32),
        position_error: wp.array(dtype=wp.float32),
        orientation_error: wp.array(dtype=wp.float32),
        collision_cost: wp.array(dtype=wp.float32),
        self_collision_cost: wp.array(dtype=wp.float32),
        goalset_index: wp.array(dtype=wp.int32),
        rank_success: wp.array(dtype=wp.uint8),
        rank_score: wp.array(dtype=wp.float32),
        solution: wp.array2d(dtype=wp.float32),
        success: wp.array(dtype=wp.uint8),
        solution_position_error: wp.array(dtype=wp.float32),
        solution_orientation_error: wp.array(dtype=wp.float32),
        solution_collision_cost: wp.array(dtype=wp.float32),
        solution_goalset_index: wp.array(dtype=wp.int32),
        num_seeds: wp.int32,
        num_solutions: wp.int32,
    ):
        output_idx, lane = wp.tid()
        batch_idx = output_idx / num_solutions
        solution_idx = output_idx % num_solutions
        matched_problem = wp.int32(-1)
        # Rank candidates in parallel using scores/validity computed once per seed.
        for seed_idx in range(lane, num_seeds, tile_threads):
            problem_idx = batch_idx * num_seeds + seed_idx
            candidate_success = rank_success[problem_idx]
            candidate_score = rank_score[problem_idx]
            better_count = wp.int32(0)
            for other_seed_idx in range(num_seeds):
                other_problem_idx = batch_idx * num_seeds + other_seed_idx
                other_success = rank_success[other_problem_idx]
                other_score = rank_score[other_problem_idx]
                if (other_success > candidate_success) or (
                    other_success == candidate_success
                    and (
                        other_score < candidate_score
                        or (other_score == candidate_score and other_seed_idx < seed_idx)
                    )
                ):
                    better_count += 1
            if better_count == solution_idx:
                matched_problem = problem_idx

        matched_problem = wp.tile_max(wp.tile(matched_problem))[0]
        if lane == 0:
            best_problem = wp.max(matched_problem, batch_idx * num_seeds)
            found_success = matched_problem >= 0 and rank_success[best_problem] == wp.uint8(1)
            for joint_idx in range(coord):
                solution[output_idx, joint_idx] = joint_position[best_problem, joint_idx]
            for link_idx in range(links):
                solution_goalset_index[output_idx * links + link_idx] = goalset_index[
                    best_problem * links + link_idx
                ]
            success[output_idx] = wp.uint8(1) if found_success else wp.uint8(0)
            solution_position_error[output_idx] = position_error[best_problem]
            solution_orientation_error[output_idx] = orientation_error[best_problem]
            solution_collision_cost[output_idx] = (
                collision_cost[best_problem] + self_collision_cost[best_problem]
            )

    finalize = warp_kernel(f"seed_ik_finalize_{dof}_{num_links}", enable_backward=False)(
        _finalize_template
    )

    return {
        "initialize": initialize,
        "error": error,
        "update": update,
        "rank": rank,
        "finalize": finalize,
        **lbfgs_kernels,
        "combine_lbfgs_rollout": combine_lbfgs_rollout,
        "num_residuals": int(residuals),
    }
