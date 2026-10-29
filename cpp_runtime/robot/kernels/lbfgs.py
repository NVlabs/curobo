# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp glue kernels shared by the C++ L-BFGS runtimes."""

from __future__ import annotations

from typing import Any

import warp as wp

from curobo._src.util.warp import warp_kernel


def create_lbfgs_runtime_kernels(
    optimization_dimension: int,
    num_line_search: int,
    name_prefix: str,
) -> dict[str, Any]:
    """Create fixed-dimension L-BFGS state and line-search preparation kernels."""
    if optimization_dimension < 1:
        raise ValueError("optimization_dimension must be positive")
    if num_line_search < 1:
        raise ValueError("num_line_search must be positive")

    fixed_dimension = wp.constant(optimization_dimension)
    fixed_lines = wp.constant(num_line_search)

    def _prepare_initial_state_template(
        seed_action: wp.array2d(dtype=wp.float32),
        seed_cost: wp.array(dtype=wp.float32),
        seed_gradient: wp.array2d(dtype=wp.float32),
        action: wp.array2d(dtype=wp.float32),
        gradient: wp.array2d(dtype=wp.float32),
        cost: wp.array(dtype=wp.float32),
        exploration_action: wp.array2d(dtype=wp.float32),
        exploration_gradient: wp.array2d(dtype=wp.float32),
        exploration_cost: wp.array(dtype=wp.float32),
        best_action: wp.array2d(dtype=wp.float32),
        best_cost: wp.array(dtype=wp.float32),
        best_iteration: wp.array(dtype=wp.int32),
        current_iteration: wp.array(dtype=wp.int32),
        converged: wp.array(dtype=wp.uint8),
        step_direction: wp.array2d(dtype=wp.float32),
        reference_action: wp.array2d(dtype=wp.float32),
        reference_gradient: wp.array2d(dtype=wp.float32),
        history_head: wp.array(dtype=wp.int32),
        history_count: wp.array(dtype=wp.int32),
        curvature_update_valid: wp.array(dtype=wp.bool),
        optimization_step_max: wp.array(dtype=wp.float32),
        initial_step_scale: wp.float32,
        clear_optimizer_state: wp.uint8,
        beta: wp.array(dtype=wp.float32),
    ):
        problem_idx = wp.tid()
        beta[problem_idx] = 1.0
        direction_scale = wp.float32(1.0)
        for variable_idx in range(fixed_dimension):
            raw_direction = -initial_step_scale * seed_gradient[problem_idx, variable_idx]
            step_max = optimization_step_max[variable_idx]
            if step_max > 0.0:
                direction_scale = wp.max(direction_scale, wp.abs(raw_direction) / step_max)
        for variable_idx in range(fixed_dimension):
            value = seed_action[problem_idx, variable_idx]
            grad = seed_gradient[problem_idx, variable_idx]
            action[problem_idx, variable_idx] = value
            gradient[problem_idx, variable_idx] = grad
            exploration_action[problem_idx, variable_idx] = value
            exploration_gradient[problem_idx, variable_idx] = grad
            best_action[problem_idx, variable_idx] = value
            reference_action[problem_idx, variable_idx] = value
            reference_gradient[problem_idx, variable_idx] = grad
            step_direction[problem_idx, variable_idx] = (
                -initial_step_scale * grad / direction_scale
            )
        value_cost = seed_cost[problem_idx]
        cost[problem_idx] = value_cost
        exploration_cost[problem_idx] = value_cost
        best_cost[problem_idx] = value_cost
        best_iteration[problem_idx] = 1
        current_iteration[problem_idx] = 1
        converged[problem_idx] = wp.uint8(0)
        if clear_optimizer_state == wp.uint8(1):
            history_head[problem_idx] = 0
            history_count[problem_idx] = 0
        curvature_update_valid[problem_idx] = False

    prepare_initial_state = warp_kernel(
        f"{name_prefix}_lbfgs_prepare_initial_{optimization_dimension}_{num_line_search}",
        enable_backward=False,
    )(_prepare_initial_state_template)

    def make_prepare_search_actions(backtracking: bool) -> Any:
        """Specialize away beta reads when the fixed probe fan is selected."""

        def _prepare_search_actions_template(
            exploration_action: wp.array2d(dtype=wp.float32),
            step_direction: wp.array2d(dtype=wp.float32),
            line_search_scale: wp.array(dtype=wp.float32),
            beta: wp.array(dtype=wp.float32),
            search_action: wp.array2d(dtype=wp.float32),
        ):
            candidate_idx, variable_idx = wp.tid()
            problem_idx = candidate_idx / fixed_lines
            line_idx = candidate_idx - problem_idx * fixed_lines
            scale = line_search_scale[line_idx]
            if wp.static(backtracking):
                scale = beta[problem_idx] * scale
            search_action[candidate_idx, variable_idx] = (
                exploration_action[problem_idx, variable_idx]
                + scale * step_direction[problem_idx, variable_idx]
            )

        return warp_kernel(
            f"{name_prefix}_lbfgs_prepare_search_{optimization_dimension}_{num_line_search}"
            f"_{int(backtracking)}",
            enable_backward=False,
        )(_prepare_search_actions_template)

    return {
        "prepare_initial_state": prepare_initial_state,
        "prepare_search_actions": make_prepare_search_actions(False),
        "prepare_search_actions_backtracking": make_prepare_search_actions(True),
    }
