# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fused Warp kernels for Python and C++ bidirectional RRT."""

from __future__ import annotations

# Standard Library
from functools import lru_cache
from typing import Any

import warp as wp

from curobo._src.util.warp import warp_kernel


@lru_cache(maxsize=None)
def create_rrt_runtime_kernels(
    dof: int,
    max_candidates: int,
    max_steps: int,
    max_direct_steps: int,
) -> dict[str, Any]:
    """Create fixed-shape steering and control kernels for both RRT frontends.

    Tree operations, kinematics, and collision math remain in their existing
    cuRobo kernels. Python launches this bundle directly, while the native
    exporter compiles the same bundle into deployment CUBINs.
    """
    for name, value in {
        "dof": dof,
        "max_candidates": max_candidates,
        "max_steps": max_steps,
        "max_direct_steps": max_direct_steps,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")

    fixed_dof = wp.constant(dof)
    fixed_candidates = wp.constant(max_candidates)
    fixed_steps = wp.constant(max_steps)
    fixed_direct_steps = wp.constant(max_direct_steps)

    def _reset_workspace_template(
        request_starts: wp.array2d(dtype=wp.float32),
        request_goals: wp.array2d(dtype=wp.float32),
        request_env: wp.array(dtype=wp.int32),
        request_goal: wp.array(dtype=wp.int32),
        request_count: wp.int32,
        goal_route_mode: wp.int32,
        start_nodes: wp.array3d(dtype=wp.float32),
        start_parents: wp.array2d(dtype=wp.int64),
        start_counts: wp.array(dtype=wp.int32),
        goal_nodes: wp.array3d(dtype=wp.float32),
        goal_parents: wp.array2d(dtype=wp.int64),
        goal_counts: wp.array(dtype=wp.int32),
        plan_active: wp.array(dtype=wp.bool),
        env_query_idx: wp.array(dtype=wp.int32),
        goal_query_idx: wp.array(dtype=wp.int32),
        connection_indices: wp.array(dtype=wp.int64),
        path_counts: wp.array(dtype=wp.int32),
        path_lengths: wp.array(dtype=wp.float32),
    ):
        plan_idx = wp.tid()
        source_idx = wp.int32(0)
        if plan_idx < request_count:
            source_idx = plan_idx
        for joint_idx in range(fixed_dof):
            start_nodes[plan_idx, 0, joint_idx] = request_starts[source_idx, joint_idx]
            goal_nodes[plan_idx, 0, joint_idx] = request_goals[source_idx, joint_idx]
        start_parents[plan_idx, 0] = wp.int64(-1)
        goal_parents[plan_idx, 0] = wp.int64(-1)
        start_counts[plan_idx] = wp.int32(1)
        goal_counts[plan_idx] = wp.int32(1)
        plan_active[plan_idx] = False
        connection_indices[plan_idx * 2] = wp.int64(-1)
        connection_indices[plan_idx * 2 + 1] = wp.int64(-1)
        path_counts[plan_idx] = wp.int32(0)
        path_lengths[plan_idx] = wp.float32(wp.inf)

        environment = wp.int32(0)
        goal = wp.int32(0)
        if plan_idx < request_count:
            environment = request_env[plan_idx]
            if goal_route_mode == wp.int32(0):
                goal = environment
            elif goal_route_mode == wp.int32(2):
                goal = request_goal[plan_idx]
        env_query_idx[plan_idx] = environment
        goal_query_idx[plan_idx] = goal

    reset_workspace = warp_kernel(
        f"rrt_runtime_reset_workspace_{dof}",
        enable_backward=False,
    )(_reset_workspace_template)

    def _prepare_samples_template(
        samples: wp.array2d(dtype=wp.float32),
        desired: wp.array2d(dtype=wp.float32),
        candidates_per_plan: wp.int32,
    ):
        edge_idx, joint_idx = wp.tid()
        desired[edge_idx, joint_idx] = samples[edge_idx, joint_idx]

    prepare_samples = warp_kernel(
        f"rrt_runtime_prepare_samples_{dof}_{max_candidates}",
        enable_backward=False,
    )(_prepare_samples_template)

    def _steer_edges_template(
        starts: wp.array2d(dtype=wp.float32),
        desired: wp.array2d(dtype=wp.float32),
        distance_weight: wp.array(dtype=wp.float32),
        targets: wp.array2d(dtype=wp.float32),
        line_points: wp.array2d(dtype=wp.float32),
        line_valid: wp.array(dtype=wp.bool),
        max_extension_distance: wp.float32,
        cspace_resolution: wp.float32,
    ):
        edge_idx, lane = wp.tid()
        local_weighted_delta = wp.float32(0.0)
        if lane < fixed_dof:
            local_weighted_delta = wp.abs(
                (desired[edge_idx, lane] - starts[edge_idx, lane]) * distance_weight[lane]
            )
        weighted_max = wp.tile_max(wp.tile(local_weighted_delta))[0]
        scale = wp.min(
            max_extension_distance / wp.max(weighted_max, wp.float32(1.0e-12)),
            wp.float32(1.0),
        )
        limited_distance = weighted_max * scale
        required_steps = wp.clamp(
            wp.int32(wp.ceil(limited_distance / cspace_resolution)) + 1,
            wp.int32(2),
            wp.int32(fixed_steps),
        )
        denominator = wp.float32(required_steps - 1)
        if lane < fixed_dof:
            start = starts[edge_idx, lane]
            targets[edge_idx, lane] = start + scale * (desired[edge_idx, lane] - start)
        if lane < fixed_steps:
            line_valid[edge_idx * fixed_steps + lane] = lane < required_steps
            alpha = wp.min(wp.float32(lane) / denominator, wp.float32(1.0))
            for joint_idx in range(fixed_dof):
                start = starts[edge_idx, joint_idx]
                target = start + scale * (desired[edge_idx, joint_idx] - start)
                line_points[edge_idx * fixed_steps + lane, joint_idx] = start + alpha * (
                    target - start
                )

    steer_edges = warp_kernel(
        f"rrt_runtime_steer_{dof}_{max_candidates}_{max_steps}",
        enable_backward=False,
    )(_steer_edges_template)

    def _evaluate_edges_template(
        starts: wp.array2d(dtype=wp.float32),
        desired: wp.array2d(dtype=wp.float32),
        targets: wp.array2d(dtype=wp.float32),
        line_points: wp.array2d(dtype=wp.float32),
        line_valid: wp.array(dtype=wp.bool),
        precomputed_feasible: wp.array(dtype=wp.bool),
        world_collision_cost: wp.array(dtype=wp.float32),
        self_collision_cost: wp.array(dtype=wp.float32),
        effort: wp.array(dtype=wp.float32),
        position_lower: wp.array(dtype=wp.float32),
        position_upper: wp.array(dtype=wp.float32),
        effort_lower: wp.array(dtype=wp.float32),
        effort_upper: wp.array(dtype=wp.float32),
        plan_active: wp.array(dtype=wp.bool),
        candidates_per_plan: wp.int32,
        use_plan_mask: wp.uint8,
        similarity_threshold: wp.float32,
        effort_active: wp.uint8,
        use_precomputed_feasible: wp.uint8,
        endpoints: wp.array2d(dtype=wp.float32),
        reached: wp.array(dtype=wp.bool),
        progressed: wp.array(dtype=wp.bool),
    ):
        edge_idx, lane = wp.tid()
        state_idx = edge_idx * fixed_steps + lane
        local_infeasible = wp.int32(fixed_steps)
        local_last_valid = wp.int32(0)
        if lane < fixed_steps and line_valid[state_idx]:
            local_last_valid = lane
            feasible = wp.bool(True)
            if use_precomputed_feasible == wp.uint8(1):
                feasible = precomputed_feasible[state_idx]
            else:
                feasible = (
                    world_collision_cost[state_idx] <= 0.0
                    and self_collision_cost[state_idx] <= 0.0
                )
                for joint_idx in range(fixed_dof):
                    value = line_points[state_idx, joint_idx]
                    tau = effort[state_idx * fixed_dof + joint_idx]
                    effort_violation = effort_active == wp.uint8(1) and (
                        not wp.isfinite(tau)
                        or tau < effort_lower[joint_idx]
                        or tau > effort_upper[joint_idx]
                    )
                    if (
                        not wp.isfinite(value)
                        or value < position_lower[joint_idx]
                        or value > position_upper[joint_idx]
                        or effort_violation
                    ):
                        feasible = False
            if not feasible:
                local_infeasible = lane
        first_infeasible = wp.tile_min(wp.tile(local_infeasible))[0]
        last_valid = wp.tile_max(wp.tile(local_last_valid))[0]
        reached_target = first_infeasible == fixed_steps
        endpoint_idx = last_valid
        if first_infeasible < fixed_steps:
            endpoint_idx = wp.max(first_infeasible - 1, wp.int32(0))

        local_desired_distance_squared = wp.float32(0.0)
        local_progress_distance_squared = wp.float32(0.0)
        if lane < fixed_dof:
            endpoint = line_points[edge_idx * fixed_steps + endpoint_idx, lane]
            endpoints[edge_idx, lane] = endpoint
            desired_delta = targets[edge_idx, lane] - desired[edge_idx, lane]
            progress_delta = endpoint - starts[edge_idx, lane]
            local_desired_distance_squared = desired_delta * desired_delta
            local_progress_distance_squared = progress_delta * progress_delta
        desired_distance_squared = wp.tile_sum(wp.tile(local_desired_distance_squared))[0]
        progress_distance_squared = wp.tile_sum(wp.tile(local_progress_distance_squared))[0]
        if lane == wp.int32(0):
            active = wp.bool(True)
            if use_plan_mask == wp.uint8(1):
                active = plan_active[edge_idx / candidates_per_plan]
            threshold_squared = similarity_threshold * similarity_threshold
            reached[edge_idx] = (
                active and reached_target and desired_distance_squared <= threshold_squared
            )
            progressed[edge_idx] = active and progress_distance_squared > threshold_squared

    evaluate_edges = warp_kernel(
        f"rrt_runtime_evaluate_{dof}_{max_candidates}_{max_steps}",
        enable_backward=False,
    )(_evaluate_edges_template)

    def _prepare_direct_edges_template(
        starts: wp.array2d(dtype=wp.float32),
        desired: wp.array2d(dtype=wp.float32),
        distance_weight: wp.array(dtype=wp.float32),
        edge_count: wp.int32,
        cspace_resolution: wp.float32,
        point_count: wp.array(dtype=wp.int32),
    ):
        weighted_max = wp.float32(0.0)
        for edge_idx in range(edge_count):
            for joint_idx in range(fixed_dof):
                delta = wp.abs(
                    (desired[edge_idx, joint_idx] - starts[edge_idx, joint_idx])
                    * distance_weight[joint_idx]
                )
                weighted_max = wp.max(weighted_max, delta)
        point_count[0] = wp.clamp(
            wp.int32(wp.ceil(weighted_max / cspace_resolution)) + 1,
            wp.int32(2),
            wp.int32(fixed_direct_steps),
        )

    prepare_direct_edges = warp_kernel(
        f"rrt_runtime_prepare_direct_{dof}_{max_direct_steps}",
        enable_backward=False,
    )(_prepare_direct_edges_template)

    def _pack_direct_pairs_template(
        request_starts: wp.array2d(dtype=wp.float32),
        request_goals: wp.array2d(dtype=wp.float32),
        request_env: wp.array(dtype=wp.int32),
        request_goal: wp.array(dtype=wp.int32),
        plan_count: wp.int32,
        starts: wp.array2d(dtype=wp.float32),
        desired: wp.array2d(dtype=wp.float32),
        env_query_idx: wp.array(dtype=wp.int32),
        goal_query_idx: wp.array(dtype=wp.int32),
    ):
        edge_idx = wp.tid()
        plan_idx = edge_idx
        reverse = False
        if edge_idx >= plan_count:
            plan_idx = edge_idx - plan_count
            reverse = True
        for joint_idx in range(fixed_dof):
            start = request_starts[plan_idx, joint_idx]
            goal = request_goals[plan_idx, joint_idx]
            if reverse:
                starts[edge_idx, joint_idx] = goal
                desired[edge_idx, joint_idx] = start
            else:
                starts[edge_idx, joint_idx] = start
                desired[edge_idx, joint_idx] = goal
        env_query_idx[edge_idx] = request_env[plan_idx]
        goal_query_idx[edge_idx] = request_goal[plan_idx]

    pack_direct_pairs = warp_kernel(
        f"rrt_runtime_pack_direct_pairs_{dof}",
        enable_backward=False,
    )(_pack_direct_pairs_template)

    def _interpolate_direct_edges_template(
        starts: wp.array2d(dtype=wp.float32),
        desired: wp.array2d(dtype=wp.float32),
        point_count: wp.array(dtype=wp.int32),
        line_points: wp.array2d(dtype=wp.float32),
        line_valid: wp.array(dtype=wp.bool),
    ):
        edge_idx, step_idx = wp.tid()
        count = point_count[0]
        denominator = wp.float32(count - 1)
        valid = step_idx < count
        line_valid[edge_idx * fixed_direct_steps + step_idx] = valid
        alpha = wp.min(wp.float32(step_idx) / denominator, wp.float32(1.0))
        for joint_idx in range(fixed_dof):
            start = starts[edge_idx, joint_idx]
            line_points[edge_idx * fixed_direct_steps + step_idx, joint_idx] = start + alpha * (
                desired[edge_idx, joint_idx] - start
            )

    interpolate_direct_edges = warp_kernel(
        f"rrt_runtime_interpolate_direct_{dof}_{max_direct_steps}",
        enable_backward=False,
    )(_interpolate_direct_edges_template)

    def _reduce_direct_edges_template(
        starts: wp.array2d(dtype=wp.float32),
        line_points: wp.array2d(dtype=wp.float32),
        line_valid: wp.array(dtype=wp.bool),
        feasible: wp.array(dtype=wp.bool),
        distance_weight: wp.array(dtype=wp.float32),
        point_count: wp.array(dtype=wp.int32),
        plan_active: wp.array(dtype=wp.bool),
        plan_count: wp.int32,
        similarity_threshold: wp.float32,
        endpoints: wp.array2d(dtype=wp.float32),
        reached: wp.array(dtype=wp.bool),
        progressed: wp.array(dtype=wp.bool),
        terminal_feasible: wp.array(dtype=wp.bool),
    ):
        edge_idx = wp.tid()
        count = point_count[0]
        first_infeasible = count
        for step_idx in range(fixed_direct_steps):
            state_idx = edge_idx * fixed_direct_steps + step_idx
            if line_valid[state_idx] and not feasible[state_idx]:
                first_infeasible = wp.min(first_infeasible, step_idx)
        reached_target = first_infeasible == count
        endpoint_idx = count - 1
        if not reached_target:
            endpoint_idx = wp.max(first_infeasible - 1, wp.int32(0))
        weighted_progress = wp.float32(0.0)
        for joint_idx in range(fixed_dof):
            endpoint = line_points[edge_idx * fixed_direct_steps + endpoint_idx, joint_idx]
            endpoints[edge_idx, joint_idx] = endpoint
            progress = wp.abs(
                (endpoint - starts[edge_idx, joint_idx]) * distance_weight[joint_idx]
            )
            weighted_progress = wp.max(weighted_progress, progress)
        plan_idx = edge_idx
        if plan_idx >= plan_count:
            plan_idx = plan_idx - plan_count
        active = plan_active[plan_idx]
        reached[edge_idx] = active and reached_target
        progressed[edge_idx] = active and weighted_progress > similarity_threshold
        terminal_feasible[edge_idx] = feasible[edge_idx * fixed_direct_steps]

    reduce_direct_edges = warp_kernel(
        f"rrt_runtime_reduce_direct_{dof}_{max_direct_steps}",
        enable_backward=False,
    )(_reduce_direct_edges_template)

    def _seed_direct_template(
        endpoints: wp.array2d(dtype=wp.float32),
        reached: wp.array(dtype=wp.bool),
        progressed: wp.array(dtype=wp.bool),
        terminal_feasible: wp.array(dtype=wp.bool),
        query_mask: wp.array(dtype=wp.bool),
        request_count: wp.int32,
        start_nodes: wp.array3d(dtype=wp.float32),
        start_parents: wp.array2d(dtype=wp.int64),
        start_counts: wp.array(dtype=wp.int32),
        goal_nodes: wp.array3d(dtype=wp.float32),
        goal_parents: wp.array2d(dtype=wp.int64),
        goal_counts: wp.array(dtype=wp.int32),
        plan_active: wp.array(dtype=wp.bool),
        connection_indices: wp.array(dtype=wp.int64),
        status: wp.array(dtype=wp.int32),
    ):
        plan_idx = wp.tid()
        valid = wp.bool(False)
        direct = wp.bool(False)
        start_progressed = wp.bool(False)
        goal_progressed = wp.bool(False)
        if plan_idx < request_count:
            valid = (
                query_mask[plan_idx]
                and terminal_feasible[plan_idx]
                and terminal_feasible[request_count + plan_idx]
            )
            direct = valid and reached[plan_idx]
            start_progressed = valid and not direct and progressed[plan_idx]
            goal_progressed = valid and not direct and progressed[request_count + plan_idx]

        plan_active[plan_idx] = valid and not direct
        status[plan_idx] = wp.int32(0)
        if valid:
            status[plan_idx] = wp.int32(1)
        if direct:
            status[plan_idx] = wp.int32(2)
            connection_indices[plan_idx * 2] = wp.int64(-2)
            connection_indices[plan_idx * 2 + 1] = wp.int64(-2)

        start_parents[plan_idx, 1] = wp.int64(0)
        goal_parents[plan_idx, 1] = wp.int64(0)
        start_counts[plan_idx] = wp.int32(1)
        goal_counts[plan_idx] = wp.int32(1)
        if start_progressed:
            start_counts[plan_idx] = wp.int32(2)
            for joint_idx in range(fixed_dof):
                start_nodes[plan_idx, 1, joint_idx] = endpoints[plan_idx, joint_idx]
        if goal_progressed:
            goal_counts[plan_idx] = wp.int32(2)
            for joint_idx in range(fixed_dof):
                goal_nodes[plan_idx, 1, joint_idx] = endpoints[request_count + plan_idx, joint_idx]

    seed_direct = warp_kernel(
        f"rrt_runtime_seed_direct_{dof}",
        enable_backward=False,
    )(_seed_direct_template)

    def _classify_states_template(
        positions: wp.array2d(dtype=wp.float32),
        world_collision_cost: wp.array(dtype=wp.float32),
        self_collision_cost: wp.array(dtype=wp.float32),
        effort: wp.array(dtype=wp.float32),
        position_lower: wp.array(dtype=wp.float32),
        position_upper: wp.array(dtype=wp.float32),
        effort_lower: wp.array(dtype=wp.float32),
        effort_upper: wp.array(dtype=wp.float32),
        feasible: wp.array(dtype=wp.bool),
        effort_active: wp.uint8,
    ):
        state_idx = wp.tid()
        state_feasible = (
            world_collision_cost[state_idx] <= 0.0 and self_collision_cost[state_idx] <= 0.0
        )
        for joint_idx in range(fixed_dof):
            value = positions[state_idx, joint_idx]
            tau = effort[state_idx * fixed_dof + joint_idx]
            effort_violation = effort_active == wp.uint8(1) and (
                not wp.isfinite(tau)
                or tau < effort_lower[joint_idx]
                or tau > effort_upper[joint_idx]
            )
            if (
                not wp.isfinite(value)
                or value < position_lower[joint_idx]
                or value > position_upper[joint_idx]
                or effort_violation
            ):
                state_feasible = False
        feasible[state_idx] = state_feasible

    classify_states = warp_kernel(f"rrt_runtime_classify_{dof}", enable_backward=False)(
        _classify_states_template
    )

    def _select_connection_template(
        connected: wp.array(dtype=wp.bool),
        active_slots: wp.array(dtype=wp.int64),
        active_written: wp.array(dtype=wp.bool),
        passive_slots: wp.array(dtype=wp.int64),
        passive_written: wp.array(dtype=wp.bool),
        passive_parents: wp.array(dtype=wp.int64),
        start_counts: wp.array(dtype=wp.int32),
        goal_counts: wp.array(dtype=wp.int32),
        plan_active: wp.array(dtype=wp.bool),
        connection_indices: wp.array(dtype=wp.int64),
        start_is_active: wp.uint8,
        status: wp.array2d(dtype=wp.int64),
    ):
        plan_idx = wp.tid()
        connection_offset = plan_idx * 2
        candidate_offset = plan_idx * fixed_candidates
        selected = wp.int32(fixed_candidates)
        for candidate_idx in range(fixed_candidates):
            flat_candidate = candidate_offset + candidate_idx
            if (
                plan_active[plan_idx]
                and selected == fixed_candidates
                and connected[flat_candidate]
                and active_written[flat_candidate]
            ):
                selected = candidate_idx

        active_index = wp.int64(-1)
        passive_index = wp.int64(-1)
        if plan_active[plan_idx]:
            connection_indices[connection_offset] = wp.int64(-1)
            connection_indices[connection_offset + 1] = wp.int64(-1)
        if selected < fixed_candidates:
            flat_candidate = candidate_offset + selected
            active_index = active_slots[flat_candidate]
            passive_index = passive_parents[flat_candidate]
            if passive_written[flat_candidate]:
                passive_index = passive_slots[flat_candidate]
            if start_is_active == wp.uint8(1):
                connection_indices[connection_offset] = active_index
                connection_indices[connection_offset + 1] = passive_index
            else:
                connection_indices[connection_offset] = passive_index
                connection_indices[connection_offset + 1] = active_index
            plan_active[plan_idx] = False

        status[plan_idx, 0] = wp.int64(selected)
        status[plan_idx, 1] = active_index
        status[plan_idx, 2] = passive_index
        status[plan_idx, 3] = wp.int64(start_counts[plan_idx])
        status[plan_idx, 4] = wp.int64(goal_counts[plan_idx])

    select_connection = warp_kernel(f"rrt_runtime_select_{max_candidates}", enable_backward=False)(
        _select_connection_template
    )

    return {
        "reset_workspace": reset_workspace,
        "prepare_samples": prepare_samples,
        "steer_edges": steer_edges,
        "evaluate_edges": evaluate_edges,
        "prepare_direct_edges": prepare_direct_edges,
        "pack_direct_pairs": pack_direct_pairs,
        "interpolate_direct_edges": interpolate_direct_edges,
        "reduce_direct_edges": reduce_direct_edges,
        "seed_direct": seed_direct,
        "classify_states": classify_states,
        "select_connection": select_connection,
    }
