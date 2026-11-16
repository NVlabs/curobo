# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp glue kernels exported for the C++ motion-planning runtime."""

from __future__ import annotations

from typing import Any

import warp as wp

from curobo._src.util.warp import warp_kernel


def create_motion_runtime_kernels(
    dof: int,
    num_knots: int,
    trajectory_steps: int,
    max_path_nodes: int,
    num_seeds: int,
) -> dict[str, Any]:
    """Create fixed-shape graph-path preparation kernels."""
    for name, value in {
        "dof": dof,
        "num_knots": num_knots,
        "trajectory_steps": trajectory_steps,
        "max_path_nodes": max_path_nodes,
        "num_seeds": num_seeds,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if num_knots < 2:
        raise ValueError("num_knots must be at least two")
    if trajectory_steps < 2:
        raise ValueError("trajectory_steps must be at least two")
    if max_path_nodes < 2:
        raise ValueError("max_path_nodes must be at least two")

    fixed_dof = wp.constant(dof)
    fixed_steps = wp.constant(trajectory_steps)
    fixed_path_nodes = wp.constant(max_path_nodes)

    def _resample_path_template(
        path: wp.array2d(dtype=wp.float32),
        path_count: wp.array(dtype=wp.int32),
        arc_progress: wp.array(dtype=wp.float32),
        dense_path: wp.array2d(dtype=wp.float32),
    ):
        output_idx = wp.tid()
        plan_idx = output_idx / fixed_steps
        output_step = output_idx - plan_idx * fixed_steps
        path_offset = plan_idx * fixed_path_nodes
        count = wp.clamp(path_count[plan_idx], wp.int32(2), wp.int32(fixed_path_nodes))
        total_length = float(0.0)
        for segment_idx in range(fixed_path_nodes - 1):
            if segment_idx + 1 < count:
                length_squared = float(0.0)
                for joint_idx in range(fixed_dof):
                    delta = (
                        path[path_offset + segment_idx + 1, joint_idx]
                        - path[path_offset + segment_idx, joint_idx]
                    )
                    length_squared += delta * delta
                total_length += wp.sqrt(length_squared)

        query_distance = arc_progress[output_step] * total_length
        selected_segment = int(0)
        selected_start = float(0.0)
        selected_length = float(0.0)
        traversed = float(0.0)
        found = bool(False)
        for segment_idx in range(fixed_path_nodes - 1):
            if segment_idx + 1 < count:
                length_squared = float(0.0)
                for joint_idx in range(fixed_dof):
                    delta = (
                        path[path_offset + segment_idx + 1, joint_idx]
                        - path[path_offset + segment_idx, joint_idx]
                    )
                    length_squared += delta * delta
                segment_length = wp.sqrt(length_squared)
                segment_end = traversed + segment_length
                if not found and (query_distance <= segment_end or segment_idx + 2 == count):
                    selected_segment = segment_idx
                    selected_start = traversed
                    selected_length = segment_length
                    found = True
                traversed = segment_end

        alpha = float(0.0)
        if total_length > wp.float32(1.0e-8) and selected_length > wp.float32(1.0e-8):
            alpha = wp.clamp(
                (query_distance - selected_start) / selected_length,
                wp.float32(0.0),
                wp.float32(1.0),
            )
        for joint_idx in range(fixed_dof):
            start = path[path_offset + selected_segment, joint_idx]
            end = path[path_offset + selected_segment + 1, joint_idx]
            dense_path[output_idx, joint_idx] = start + alpha * (end - start)

    resample_path = warp_kernel(
        f"motion_resample_path_{dof}_{trajectory_steps}_{max_path_nodes}",
        enable_backward=False,
    )(_resample_path_template)

    def _prepare_seed_paths_template(
        dense_path: wp.array3d(dtype=wp.float32),
        start_position: wp.array(dtype=wp.float32),
        start_velocity: wp.array(dtype=wp.float32),
        start_acceleration: wp.array(dtype=wp.float32),
        goal_position: wp.array(dtype=wp.float32),
        seed_start_position: wp.array2d(dtype=wp.float32),
        seed_goal_position: wp.array2d(dtype=wp.float32),
        seed_valid: wp.array(dtype=wp.uint8),
        rrt_success: wp.array(dtype=wp.uint8),
        resolved_start_position: wp.array2d(dtype=wp.float32),
        resolved_start_velocity: wp.array2d(dtype=wp.float32),
        resolved_start_acceleration: wp.array2d(dtype=wp.float32),
        resolved_goal_position: wp.array2d(dtype=wp.float32),
        use_linear_path: wp.uint8,
        use_seed_starts: wp.uint8,
        use_seed_goals: wp.uint8,
        use_seed_valid: wp.uint8,
    ):
        seed_idx, variable_idx = wp.tid()
        step_idx = variable_idx / fixed_dof
        joint_idx = variable_idx - step_idx * fixed_dof
        seed_is_valid = wp.bool(True)
        if use_seed_valid == wp.uint8(1):
            seed_is_valid = seed_valid[seed_idx] != wp.uint8(0)
        start_value = start_position[joint_idx]
        if seed_is_valid and use_seed_starts == wp.uint8(1):
            start_value = seed_start_position[seed_idx, joint_idx]
        goal_value = goal_position[joint_idx]
        if seed_is_valid and use_seed_goals == wp.uint8(1):
            goal_value = seed_goal_position[seed_idx, joint_idx]
        linear_seed = use_linear_path == wp.uint8(1) or rrt_success[seed_idx] == wp.uint8(0)
        if not linear_seed:
            start_value = dense_path[seed_idx, 0, joint_idx]
            goal_value = dense_path[seed_idx, fixed_steps - 1, joint_idx]
        if linear_seed:
            alpha = wp.float32(step_idx) / wp.float32(fixed_steps - 1)
            dense_path[seed_idx, step_idx, joint_idx] = start_value + alpha * (
                goal_value - start_value
            )
        if step_idx == 0:
            resolved_start_position[seed_idx, joint_idx] = start_value
            resolved_start_velocity[seed_idx, joint_idx] = start_velocity[joint_idx]
            resolved_start_acceleration[seed_idx, joint_idx] = start_acceleration[joint_idx]
            resolved_goal_position[seed_idx, joint_idx] = goal_value

    prepare_seed_paths = warp_kernel(
        f"motion_prepare_seed_paths_{dof}_{trajectory_steps}_{num_seeds}",
        enable_backward=False,
    )(_prepare_seed_paths_template)

    return {
        "resample_path": resample_path,
        "prepare_seed_paths": prepare_seed_paths,
    }
