# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernel and PyTorch adapter for relative tool-transform cost."""

# Standard Library
from typing import Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.cost.wp_pose_metric import (
    angular_cotangent_to_wxyz_quaternion_cotangent,
    compute_projected_pose_metric,
)
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream, warp_func, warp_kernel


@warp_func("relative_transform_wxyz_to_quaternion")
def _wxyz_to_quaternion(quaternion: wp.vec4):
    """Convert cuRobo's wxyz tensor layout to Warp's quaternion type."""
    return wp.quat(quaternion[1], quaternion[2], quaternion[3], quaternion[0])


@warp_func("relative_transform_current_pose")
def _compute_current_relative_transform(
    reference_transform_world: wp.transform,
    related_transform_world: wp.transform,
):
    """Return ``inverse(T_world_reference) * T_world_related``."""
    return wp.transform_multiply(
        wp.transform_inverse(reference_transform_world), related_transform_world
    )


@warp_kernel("tool_relative_transform_distance")
def _tool_relative_transform_distance_kernel(
    current_position: wp.array(dtype=wp.vec3),
    current_quat: wp.array(dtype=wp.vec4),
    target_position: wp.array(dtype=wp.vec3),
    target_quat: wp.array(dtype=wp.vec4),
    idxs_target: wp.array(dtype=wp.int32),
    position_orientation_weight: wp.array(dtype=wp.float32),
    position_projector: wp.array(dtype=wp.mat33),
    rotation_projector: wp.array(dtype=wp.mat33),
    position_axis_scale: wp.array(dtype=wp.vec3),
    rotation_axis_scale: wp.array(dtype=wp.vec3),
    tolerance: wp.array(dtype=wp.vec2),
    relation_enabled: wp.array(dtype=wp.int32),
    out_cost: wp.array(dtype=wp.float32),
    out_position_distance: wp.array(dtype=wp.float32),
    out_rotation_distance: wp.array(dtype=wp.float32),
    out_position_gradient: wp.array(dtype=wp.vec3),
    out_rotation_gradient: wp.array(dtype=wp.vec4),
    reference_link_index: wp.int32,
    batch_size: wp.int32,
    horizon: wp.int32,
    num_links: wp.int32,
):
    """Evaluate each relation and deterministically accumulate its reference gradient."""
    thread_index = wp.tid()
    if thread_index >= batch_size * horizon * num_links:
        return

    batch_index = thread_index / (horizon * num_links)
    horizon_index = (thread_index - batch_index * horizon * num_links) / num_links
    link_index = thread_index - batch_index * horizon * num_links - horizon_index * num_links
    state_base_index = batch_index * horizon * num_links + horizon_index * num_links
    reference_state_index = state_base_index + reference_link_index

    reference_position_world = current_position[reference_state_index]
    reference_quat_wxyz = current_quat[reference_state_index]
    reference_quat_world = _wxyz_to_quaternion(reference_quat_wxyz)
    reference_transform_world = wp.transform(
        reference_position_world,
        reference_quat_world,
    )
    selected_target_group = idxs_target[batch_index]
    position_weight = position_orientation_weight[0]
    rotation_weight = position_orientation_weight[1]

    out_cost[thread_index] = 0.0
    out_position_distance[thread_index] = 0.0
    out_rotation_distance[thread_index] = 0.0
    out_position_gradient[thread_index] = wp.vec3(0.0, 0.0, 0.0)
    out_rotation_gradient[thread_index] = wp.vec4(0.0, 0.0, 0.0, 0.0)

    if link_index == reference_link_index:
        reference_position_gradient = wp.vec3(0.0, 0.0, 0.0)
        reference_angular_gradient = wp.vec3(0.0, 0.0, 0.0)
        for relation_index in range(num_links):
            if relation_index != reference_link_index and relation_enabled[relation_index] != 0:
                related_state_index = state_base_index + relation_index
                related_position_world = current_position[related_state_index]
                related_quat_world = _wxyz_to_quaternion(current_quat[related_state_index])
                related_transform_world = wp.transform(
                    related_position_world,
                    related_quat_world,
                )
                current_relative_transform = _compute_current_relative_transform(
                    reference_transform_world,
                    related_transform_world,
                )
                current_relative_position = wp.transform_get_translation(
                    current_relative_transform
                )
                current_relative_quat = wp.transform_get_rotation(current_relative_transform)
                target_index = selected_target_group * num_links + relation_index
                target_relative_quat = _wxyz_to_quaternion(target_quat[target_index])
                relation_tolerance = tolerance[relation_index]
                (
                    _,
                    _,
                    _,
                    _,
                    relative_position_gradient,
                    relative_angular_gradient,
                ) = compute_projected_pose_metric(
                    current_relative_position,
                    target_position[target_index],
                    current_relative_quat,
                    target_relative_quat,
                    position_projector[relation_index],
                    rotation_projector[relation_index],
                    position_axis_scale[relation_index],
                    rotation_axis_scale[relation_index],
                    relation_tolerance[0],
                    relation_tolerance[1],
                    position_weight,
                    rotation_weight,
                )
                reference_position_gradient -= wp.transform_vector(
                    reference_transform_world,
                    relative_position_gradient,
                )
                reference_relative_angular_gradient = (
                    wp.cross(
                        relative_position_gradient,
                        current_relative_position,
                    )
                    - relative_angular_gradient
                )
                reference_angular_gradient += wp.transform_vector(
                    reference_transform_world,
                    reference_relative_angular_gradient,
                )

        out_position_gradient[thread_index] = reference_position_gradient
        out_rotation_gradient[thread_index] = angular_cotangent_to_wxyz_quaternion_cotangent(
            reference_angular_gradient,
            reference_quat_world,
        )
        return

    if relation_enabled[link_index] == 0:
        return

    related_position_world = current_position[thread_index]
    related_quat_world = _wxyz_to_quaternion(current_quat[thread_index])
    related_transform_world = wp.transform(related_position_world, related_quat_world)
    current_relative_transform = _compute_current_relative_transform(
        reference_transform_world,
        related_transform_world,
    )
    current_relative_position = wp.transform_get_translation(current_relative_transform)
    current_relative_quat = wp.transform_get_rotation(current_relative_transform)
    target_index = selected_target_group * num_links + link_index
    target_relative_quat = _wxyz_to_quaternion(target_quat[target_index])
    relation_tolerance = tolerance[link_index]
    (
        position_cost,
        rotation_cost,
        position_distance,
        rotation_distance,
        relative_position_gradient,
        relative_angular_gradient,
    ) = compute_projected_pose_metric(
        current_relative_position,
        target_position[target_index],
        current_relative_quat,
        target_relative_quat,
        position_projector[link_index],
        rotation_projector[link_index],
        position_axis_scale[link_index],
        rotation_axis_scale[link_index],
        relation_tolerance[0],
        relation_tolerance[1],
        position_weight,
        rotation_weight,
    )
    related_position_gradient = wp.transform_vector(
        reference_transform_world,
        relative_position_gradient,
    )
    related_angular_gradient = wp.transform_vector(
        reference_transform_world,
        relative_angular_gradient,
    )

    out_cost[thread_index] = position_cost + rotation_cost
    out_position_distance[thread_index] = position_distance
    out_rotation_distance[thread_index] = rotation_distance
    out_position_gradient[thread_index] = related_position_gradient
    out_rotation_gradient[thread_index] = angular_cotangent_to_wxyz_quaternion_cotangent(
        related_angular_gradient,
        related_quat_world,
    )


class ToolRelativeTransformDistance(torch.autograd.Function):
    """Evaluate relative-transform metrics and expose fused analytic gradients."""

    @staticmethod
    def forward(
        ctx,
        current_position: torch.Tensor,
        current_quat: torch.Tensor,
        target_position: torch.Tensor,
        target_quat: torch.Tensor,
        idxs_target: torch.Tensor,
        position_orientation_weight: torch.Tensor,
        position_projector: torch.Tensor,
        rotation_projector: torch.Tensor,
        position_axis_scale: torch.Tensor,
        rotation_axis_scale: torch.Tensor,
        tolerance: torch.Tensor,
        relation_enabled: torch.Tensor,
        out_cost: torch.Tensor,
        out_position_distance: torch.Tensor,
        out_rotation_distance: torch.Tensor,
        out_position_gradient: torch.Tensor,
        out_rotation_gradient: torch.Tensor,
        reference_link_index: int,
    ):
        """Compute one-reference relative costs for all rollout states."""
        ctx.set_materialize_grads(False)
        if current_position.ndim != 4 or current_position.shape[-1] != 3:
            log_and_raise("current_position must have shape (batch, horizon, num_links, 3)")
        if current_quat.ndim != 4 or current_quat.shape[-1] != 4:
            log_and_raise("current_quat must have shape (batch, horizon, num_links, 4)")
        batch_size, horizon, num_links, _ = current_position.shape
        if current_quat.shape != (batch_size, horizon, num_links, 4):
            log_and_raise("current_position and current_quat leading shapes must match")
        if target_position.ndim != 3 or target_position.shape[1:] != (num_links, 3):
            log_and_raise("target_position must have shape (num_target_groups, num_links, 3)")
        if target_quat.shape != (target_position.shape[0], num_links, 4):
            log_and_raise("target_quat must have shape (num_target_groups, num_links, 4)")
        if idxs_target.shape != (batch_size, 1) or idxs_target.dtype != torch.int32:
            log_and_raise(f"idxs_target must be int32 with shape ({batch_size}, 1)")
        if position_orientation_weight.shape != (2,):
            log_and_raise("position_orientation_weight must have shape (2,)")
        if position_projector.shape != (num_links, 3, 3):
            log_and_raise("position_projector has an invalid shape")
        if rotation_projector.shape != (num_links, 3, 3):
            log_and_raise("rotation_projector has an invalid shape")
        if position_axis_scale.shape != (num_links, 3):
            log_and_raise("position_axis_scale has an invalid shape")
        if rotation_axis_scale.shape != (num_links, 3):
            log_and_raise("rotation_axis_scale has an invalid shape")
        if tolerance.shape != (num_links, 2):
            log_and_raise("tolerance has an invalid shape")
        if relation_enabled.shape != (num_links,) or relation_enabled.dtype != torch.int32:
            log_and_raise("relation_enabled must be int32 with shape (num_links,)")
        if reference_link_index < 0 or reference_link_index >= num_links:
            log_and_raise("reference_link_index is out of range")

        output_shape = (batch_size, horizon, num_links)
        if out_cost.shape != output_shape:
            log_and_raise("out_cost has an invalid shape")
        if out_position_distance.shape != output_shape:
            log_and_raise("out_position_distance has an invalid shape")
        if out_rotation_distance.shape != output_shape:
            log_and_raise("out_rotation_distance has an invalid shape")
        if out_position_gradient.shape != (*output_shape, 3):
            log_and_raise("out_position_gradient has an invalid shape")
        if out_rotation_gradient.shape != (*output_shape, 4):
            log_and_raise("out_rotation_gradient has an invalid shape")

        tensors = (
            current_quat,
            target_position,
            target_quat,
            idxs_target,
            position_orientation_weight,
            position_projector,
            rotation_projector,
            position_axis_scale,
            rotation_axis_scale,
            tolerance,
            relation_enabled,
            out_cost,
            out_position_distance,
            out_rotation_distance,
            out_position_gradient,
            out_rotation_gradient,
        )
        if any(tensor.device != current_position.device for tensor in tensors):
            log_and_raise("all relative-transform cost tensors must be on the same device")
        if current_position.dtype != torch.float32 or current_quat.dtype != torch.float32:
            log_and_raise("relative-transform pose tensors must use float32")

        wp_device, wp_stream = get_warp_device_stream(current_position)
        wp.launch(
            kernel=_tool_relative_transform_distance_kernel,
            dim=batch_size * horizon * num_links,
            inputs=[
                wp.from_torch(current_position.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(current_quat.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(target_position.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(target_quat.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(idxs_target.detach().view(-1), dtype=wp.int32),
                wp.from_torch(position_orientation_weight.view(-1), dtype=wp.float32),
                wp.from_torch(position_projector.view(-1, 3, 3), dtype=wp.mat33),
                wp.from_torch(rotation_projector.view(-1, 3, 3), dtype=wp.mat33),
                wp.from_torch(position_axis_scale.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(rotation_axis_scale.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(tolerance.view(-1, 2), dtype=wp.vec2),
                wp.from_torch(relation_enabled.view(-1), dtype=wp.int32),
                wp.from_torch(out_cost.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_rotation_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_gradient.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(out_rotation_gradient.view(-1, 4), dtype=wp.vec4),
                reference_link_index,
                batch_size,
                horizon,
                num_links,
            ],
            device=wp_device,
            stream=wp_stream,
            adjoint=False,
        )

        ctx.mark_non_differentiable(out_position_distance, out_rotation_distance)
        ctx.save_for_backward(out_position_gradient, out_rotation_gradient)
        return out_cost, out_position_distance, out_rotation_distance

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_cost: Optional[torch.Tensor],
        grad_position_distance: Optional[torch.Tensor],
        grad_rotation_distance: Optional[torch.Tensor],
    ):
        """Return fused gradients for the two current-pose inputs."""
        del grad_position_distance, grad_rotation_distance
        position_input_gradient = None
        quaternion_input_gradient = None
        if grad_cost is not None and (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            position_gradient, rotation_gradient = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                position_input_gradient = position_gradient
            if ctx.needs_input_grad[1]:
                quaternion_input_gradient = rotation_gradient
        return (position_input_gradient, quaternion_input_gradient) + (None,) * 16
