# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp kernel and PyTorch adapter for the absolute tool-pose cost."""

# Standard Library
from typing import Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.cost.tool_pose_criteria import PoseErrorFrame
from curobo._src.cost.wp_pose_metric import (
    angular_cotangent_to_wxyz_quaternion_cotangent,
    compute_quadratic_pose_metric,
)
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream, warp_kernel

_GOAL_ERROR_FRAME = wp.constant(wp.int32(PoseErrorFrame.GOAL))


def create_goalset_pose_distance_kernel_with_constants(num_goalset: int):
    """Create a goal-set kernel specialized by its compile-time loop bound."""

    def _goalset_pose_distance_template(
        current_position: wp.array(dtype=wp.vec3),
        current_quat: wp.array(dtype=wp.vec4),
        goal_position: wp.array(dtype=wp.vec3),
        goal_quat: wp.array(dtype=wp.vec4),
        idxs_goal: wp.array(dtype=wp.int32),
        position_orientation_weight: wp.array(dtype=wp.float32),
        position_metric: wp.array(dtype=wp.mat33),
        rotation_metric: wp.array(dtype=wp.mat33),
        pose_tolerance: wp.array(dtype=wp.vec2),
        error_frame: wp.array(dtype=wp.int32),
        out_distance: wp.array(dtype=wp.float32),
        out_position_distance: wp.array(dtype=wp.float32),
        out_rotation_distance: wp.array(dtype=wp.float32),
        out_position_gradient: wp.array(dtype=wp.vec3),
        out_rotation_gradient: wp.array(dtype=wp.vec4),
        out_goalset_idx: wp.array(dtype=wp.int32),
        batch_size: wp.int32,
        horizon: wp.int32,
        num_links: wp.int32,
    ):
        thread_index = wp.tid()
        if thread_index >= batch_size * horizon * num_links:
            return

        batch_index = thread_index / (horizon * num_links)
        horizon_index = (thread_index - batch_index * horizon * num_links) / num_links
        link_index = thread_index - batch_index * horizon * num_links - horizon_index * num_links

        criteria_index = link_index
        if horizon_index < horizon - 1:
            criteria_index = num_links + link_index

        tolerance = pose_tolerance[criteria_index]
        local_position_metric = position_metric[criteria_index]
        local_rotation_metric = rotation_metric[criteria_index]
        metric_trace = (
            local_position_metric[0, 0]
            + local_position_metric[1, 1]
            + local_position_metric[2, 2]
            + local_rotation_metric[0, 0]
            + local_rotation_metric[1, 1]
            + local_rotation_metric[2, 2]
        )
        if metric_trace <= 0.0:
            return
        position_weight = position_orientation_weight[0]
        rotation_weight = position_orientation_weight[1]
        selected_goal_batch = idxs_goal[batch_index]
        use_goal_frame = error_frame[link_index] == _GOAL_ERROR_FRAME

        current_position_world = current_position[thread_index]
        current_quat_wxyz = current_quat[thread_index]
        current_quat_world = wp.quat(
            current_quat_wxyz[1],
            current_quat_wxyz[2],
            current_quat_wxyz[3],
            current_quat_wxyz[0],
        )
        best_cost = wp.float32(-1.0)
        best_goal_index = wp.int32(0)
        best_position_cost = wp.float32(0.0)
        best_rotation_cost = wp.float32(0.0)
        best_position_distance = wp.float32(0.0)
        best_rotation_distance = wp.float32(0.0)
        best_position_gradient = wp.vec3(0.0, 0.0, 0.0)
        best_rotation_gradient = wp.vec3(0.0, 0.0, 0.0)
        best_goal_transform = wp.transform(
            wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0)
        )

        for goalset_index in range(num_goalset):
            goal_flat_index = (
                selected_goal_batch * num_links * num_goalset
                + link_index * num_goalset
                + goalset_index
            )
            goal_position_world = goal_position[goal_flat_index]
            goal_quat_wxyz = goal_quat[goal_flat_index]
            goal_quat_world = wp.quat(
                goal_quat_wxyz[1],
                goal_quat_wxyz[2],
                goal_quat_wxyz[3],
                goal_quat_wxyz[0],
            )
            goal_transform_world = wp.transform(
                wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0)
            )

            metric_current_position = current_position_world
            metric_current_quat = current_quat_world
            metric_goal_position = goal_position_world
            metric_goal_quat = goal_quat_world
            if use_goal_frame:
                goal_transform_world = wp.transform(goal_position_world, goal_quat_world)
                current_transform_world = wp.transform(
                    current_position_world, current_quat_world
                )
                current_transform_goal = wp.transform_multiply(
                    wp.transform_inverse(goal_transform_world),
                    current_transform_world,
                )
                metric_current_position = wp.transform_get_translation(
                    current_transform_goal
                )
                metric_current_quat = wp.transform_get_rotation(current_transform_goal)
                metric_goal_position = wp.vec3(0.0, 0.0, 0.0)
                metric_goal_quat = wp.quat(0.0, 0.0, 0.0, 1.0)

            (
                position_cost,
                rotation_cost,
                position_distance,
                rotation_distance,
                position_gradient,
                rotation_gradient,
            ) = compute_quadratic_pose_metric(
                metric_current_position,
                metric_goal_position,
                metric_current_quat,
                metric_goal_quat,
                local_position_metric,
                local_rotation_metric,
                tolerance[0],
                tolerance[1],
                position_weight,
                rotation_weight,
            )

            total_cost = position_cost + rotation_cost
            if best_cost < 0.0 or total_cost < best_cost:
                best_cost = total_cost
                best_goal_index = goalset_index
                best_position_cost = position_cost
                best_rotation_cost = rotation_cost
                best_position_distance = position_distance
                best_rotation_distance = rotation_distance
                best_position_gradient = position_gradient
                best_rotation_gradient = rotation_gradient
                best_goal_transform = goal_transform_world

        if use_goal_frame:
            best_position_gradient = wp.transform_vector(
                best_goal_transform, best_position_gradient
            )
            best_rotation_gradient = wp.transform_vector(
                best_goal_transform, best_rotation_gradient
            )

        quaternion_gradient_wxyz = angular_cotangent_to_wxyz_quaternion_cotangent(
            best_rotation_gradient, current_quat_world
        )

        out_distance[2 * thread_index] = best_position_cost
        out_distance[2 * thread_index + 1] = best_rotation_cost
        out_position_distance[thread_index] = best_position_distance
        out_rotation_distance[thread_index] = best_rotation_distance
        out_position_gradient[thread_index] = best_position_gradient
        out_rotation_gradient[thread_index] = quaternion_gradient_wxyz
        out_goalset_idx[thread_index] = best_goal_index

    kernel_name = f"goalset_pose_distance_{num_goalset}_inactive_exit"
    return warp_kernel(kernel_name)(_goalset_pose_distance_template)


class ToolPoseDistance(torch.autograd.Function):
    """Evaluate the Warp pose metric and expose its analytic first derivative."""

    @staticmethod
    def forward(
        ctx,
        current_position: torch.Tensor,
        current_quat: torch.Tensor,
        goal_position: torch.Tensor,
        goal_quat: torch.Tensor,
        idxs_goal: torch.Tensor,
        position_orientation_weight: torch.Tensor,
        position_metric: torch.Tensor,
        rotation_metric: torch.Tensor,
        pose_tolerance: torch.Tensor,
        error_frame: torch.Tensor,
        out_distance: torch.Tensor,
        out_position_distance: torch.Tensor,
        out_rotation_distance: torch.Tensor,
        out_position_gradient: torch.Tensor,
        out_rotation_gradient: torch.Tensor,
        out_goalset_idx: torch.Tensor,
        use_grad_input: bool,
        warp_kernel,
    ):
        """Compute absolute tool-pose costs for all batch, horizon, and link entries."""
        ctx.set_materialize_grads(False)
        if current_position.ndim != 4:
            log_and_raise("current_position must be a 4D tensor")
        if current_quat.ndim != 4:
            log_and_raise("current_quat must be a 4D tensor")
        if goal_position.ndim != 4:
            log_and_raise(
                "goal_position must have shape (-1, num_links, num_goalset, 3), "
                + f"got {goal_position.shape}"
            )
        if goal_quat.ndim != 4:
            log_and_raise(
                "goal_quat must have shape (-1, num_links, num_goalset, 4), "
                + f"got {goal_quat.shape}"
            )

        batch_size, horizon, num_links, _ = current_position.shape
        if current_position.shape != (batch_size, horizon, num_links, 3):
            log_and_raise("current_position must have shape (b, h, num_links, 3)")
        if current_quat.shape != (batch_size, horizon, num_links, 4):
            log_and_raise("current_quat must have shape (b, h, num_links, 4)")
        if goal_position.shape[-1] != 3:
            log_and_raise("goal_position must end in dimension 3")
        if goal_quat.shape[-1] != 4:
            log_and_raise("goal_quat must end in dimension 4")
        if goal_position.shape[:-1] != goal_quat.shape[:-1]:
            log_and_raise("goal_position and goal_quat must have matching leading shapes")
        if goal_position.shape[1] != num_links:
            log_and_raise("goal poses must have the same number of links as current poses")
        if idxs_goal.shape != (batch_size, 1):
            log_and_raise(f"idxs_goal must have shape ({batch_size}, 1), got {idxs_goal.shape}")
        if position_orientation_weight.shape != (2,):
            log_and_raise("position_orientation_weight must have shape (2,)")

        matrix_shape = (2 * num_links, 3, 3)
        for name, tensor in (
            ("position_metric", position_metric),
            ("rotation_metric", rotation_metric),
        ):
            if tensor.shape != matrix_shape:
                log_and_raise(f"{name} must have shape {matrix_shape}, got {tensor.shape}")
        tolerance_shape = (2 * num_links, 2)
        if pose_tolerance.shape != tolerance_shape:
            log_and_raise(
                f"pose_tolerance must have shape {tolerance_shape}, got {pose_tolerance.shape}"
            )
        if error_frame.shape != (num_links,):
            log_and_raise(f"error_frame must have shape ({num_links},), got {error_frame.shape}")

        if out_position_gradient.shape != (batch_size, horizon, num_links, 3):
            log_and_raise("out_position_gradient has an invalid shape")
        if out_rotation_gradient.shape != (batch_size, horizon, num_links, 4):
            log_and_raise("out_rotation_gradient has an invalid shape")
        if out_distance.shape != (batch_size, horizon, num_links * 2):
            log_and_raise("out_distance has an invalid shape")
        if out_position_distance.shape != (batch_size, horizon, num_links):
            log_and_raise("out_position_distance has an invalid shape")
        if out_rotation_distance.shape != (batch_size, horizon, num_links):
            log_and_raise("out_rotation_distance has an invalid shape")
        if out_goalset_idx.shape != (batch_size, horizon, num_links):
            log_and_raise("out_goalset_idx has an invalid shape")

        ctx.use_grad_input = use_grad_input
        wp_device, wp_stream = get_warp_device_stream(current_position)
        launch_dim = batch_size * horizon * num_links
        wp.launch(
            kernel=warp_kernel,
            dim=launch_dim,
            inputs=[
                wp.from_torch(current_position.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(current_quat.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(goal_position.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(goal_quat.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(idxs_goal.detach().view(-1), dtype=wp.int32),
                wp.from_torch(position_orientation_weight.view(-1), dtype=wp.float32),
                wp.from_torch(position_metric.view(-1, 3, 3), dtype=wp.mat33),
                wp.from_torch(rotation_metric.view(-1, 3, 3), dtype=wp.mat33),
                wp.from_torch(pose_tolerance.view(-1, 2), dtype=wp.vec2),
                wp.from_torch(error_frame.view(-1), dtype=wp.int32),
                wp.from_torch(out_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_rotation_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_gradient.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(out_rotation_gradient.view(-1, 4), dtype=wp.vec4),
                wp.from_torch(out_goalset_idx.view(-1), dtype=wp.int32),
                batch_size,
                horizon,
                num_links,
            ],
            device=wp_device,
            stream=wp_stream,
            adjoint=False,
        )

        ctx.mark_non_differentiable(
            out_position_distance,
            out_rotation_distance,
            out_goalset_idx,
        )
        ctx.save_for_backward(out_position_gradient, out_rotation_gradient)
        return out_distance, out_position_distance, out_rotation_distance, out_goalset_idx

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_distance: Optional[torch.Tensor],
        grad_position_distance: Optional[torch.Tensor],
        grad_rotation_distance: Optional[torch.Tensor],
        grad_goalset_idx: Optional[torch.Tensor],
    ):
        """Apply the precomputed analytic cotangents to the two differentiable inputs."""
        del grad_position_distance, grad_rotation_distance, grad_goalset_idx
        position_input_gradient = None
        quaternion_input_gradient = None
        if grad_distance is not None and (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            position_gradient, rotation_gradient = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                if ctx.use_grad_input:
                    position_multiplier = grad_distance[:, :, 0::2].unsqueeze(-1)
                    position_input_gradient = position_gradient * position_multiplier
                else:
                    position_input_gradient = position_gradient
            if ctx.needs_input_grad[1]:
                if ctx.use_grad_input:
                    rotation_multiplier = grad_distance[:, :, 1::2].unsqueeze(-1)
                    quaternion_input_gradient = rotation_gradient * rotation_multiplier
                else:
                    quaternion_input_gradient = rotation_gradient

        return (position_input_gradient, quaternion_input_gradient) + (None,) * 16
