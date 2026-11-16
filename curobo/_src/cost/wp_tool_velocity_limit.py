# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp kernel and PyTorch adapter for projected tool-velocity limits."""

from __future__ import annotations

# Standard Library
from typing import Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream


@wp.func
def compute_projected_speed_limit_metric(
    velocity: wp.vec3,
    projector: wp.mat33,
    axis_scale: wp.vec3,
    maximum_speed: wp.float32,
    weight: wp.float32,
):
    """Compute a projected speed violation, squared-hinge cost, and cotangent."""
    projected_velocity = projector * velocity
    scaled_velocity = wp.cw_mul(axis_scale, projected_velocity)
    speed = wp.length(scaled_velocity)
    violation = speed - maximum_speed
    cost = wp.float32(0.0)
    velocity_gradient = wp.vec3(0.0, 0.0, 0.0)
    if violation > 0.0:
        cost = 0.5 * weight * violation * violation
        scaled_gradient = wp.cw_mul(
            axis_scale,
            wp.cw_mul(axis_scale, projected_velocity),
        )
        velocity_gradient = (
            weight * violation / speed * (wp.transpose(projector) * scaled_gradient)
        )
    else:
        violation = 0.0
    return cost, violation, velocity_gradient


@wp.kernel(enable_backward=False)
def tool_velocity_limit_kernel(
    joint_velocity: wp.array(dtype=wp.float32),
    tool_jacobians: wp.array(dtype=wp.float32),
    linear_angular_weight: wp.array(dtype=wp.float32),
    linear_projector: wp.array(dtype=wp.mat33),
    angular_projector: wp.array(dtype=wp.mat33),
    linear_axis_scale: wp.array(dtype=wp.vec3),
    angular_axis_scale: wp.array(dtype=wp.vec3),
    max_linear_speed: wp.array(dtype=wp.float32),
    max_angular_speed: wp.array(dtype=wp.float32),
    out_cost: wp.array(dtype=wp.float32),
    out_linear_violation: wp.array(dtype=wp.float32),
    out_angular_violation: wp.array(dtype=wp.float32),
    out_linear_twist_gradient: wp.array(dtype=wp.vec3),
    out_angular_twist_gradient: wp.array(dtype=wp.vec3),
    batch_size: wp.int32,
    horizon: wp.int32,
    num_links: wp.int32,
    dof: wp.int32,
):
    """Evaluate one projected linear/angular speed limit per tool-link thread."""
    thread_index = wp.tid()
    if thread_index >= batch_size * horizon * num_links:
        return

    state_index = thread_index / num_links
    link_index = thread_index - state_index * num_links
    joint_velocity_offset = state_index * dof
    jacobian_offset = thread_index * 6 * dof

    linear_velocity = wp.vec3(0.0, 0.0, 0.0)
    angular_velocity = wp.vec3(0.0, 0.0, 0.0)
    for joint_index in range(dof):
        velocity = joint_velocity[joint_velocity_offset + joint_index]
        linear_velocity[0] += tool_jacobians[jacobian_offset + joint_index] * velocity
        linear_velocity[1] += tool_jacobians[jacobian_offset + dof + joint_index] * velocity
        linear_velocity[2] += tool_jacobians[jacobian_offset + 2 * dof + joint_index] * velocity
        angular_velocity[0] += tool_jacobians[jacobian_offset + 3 * dof + joint_index] * velocity
        angular_velocity[1] += tool_jacobians[jacobian_offset + 4 * dof + joint_index] * velocity
        angular_velocity[2] += tool_jacobians[jacobian_offset + 5 * dof + joint_index] * velocity

    linear_cost, linear_violation, linear_gradient = compute_projected_speed_limit_metric(
        linear_velocity,
        linear_projector[link_index],
        linear_axis_scale[link_index],
        max_linear_speed[link_index],
        linear_angular_weight[0],
    )
    angular_cost, angular_violation, angular_gradient = compute_projected_speed_limit_metric(
        angular_velocity,
        angular_projector[link_index],
        angular_axis_scale[link_index],
        max_angular_speed[link_index],
        linear_angular_weight[1],
    )

    out_cost[2 * thread_index] = linear_cost
    out_cost[2 * thread_index + 1] = angular_cost
    out_linear_violation[thread_index] = linear_violation
    out_angular_violation[thread_index] = angular_violation
    out_linear_twist_gradient[thread_index] = linear_gradient
    out_angular_twist_gradient[thread_index] = angular_gradient


class ToolVelocityLimitDistance(torch.autograd.Function):
    """Evaluate tool speed limits and expose exact first derivatives for ``qdot`` and ``J``."""

    @staticmethod
    def forward(
        ctx,
        joint_velocity: torch.Tensor,
        tool_jacobians: torch.Tensor,
        linear_angular_weight: torch.Tensor,
        linear_projector: torch.Tensor,
        angular_projector: torch.Tensor,
        linear_axis_scale: torch.Tensor,
        angular_axis_scale: torch.Tensor,
        max_linear_speed: torch.Tensor,
        max_angular_speed: torch.Tensor,
        out_cost: torch.Tensor,
        out_linear_violation: torch.Tensor,
        out_angular_violation: torch.Tensor,
        out_linear_twist_gradient: torch.Tensor,
        out_angular_twist_gradient: torch.Tensor,
        use_grad_input: bool,
    ):
        """Compute projected velocity-limit costs over batch, horizon, and tool links."""
        ctx.set_materialize_grads(False)
        if joint_velocity.ndim != 3:
            log_and_raise(
                "joint_velocity must have shape (batch, horizon, dof), "
                f"got {tuple(joint_velocity.shape)}"
            )
        if tool_jacobians.ndim != 5:
            log_and_raise(
                "tool_jacobians must have shape (batch, horizon, num_links, 6, dof), "
                f"got {tuple(tool_jacobians.shape)}"
            )
        batch_size, horizon, dof = joint_velocity.shape
        jacobian_batch, jacobian_horizon, num_links, twist_dim, jacobian_dof = tool_jacobians.shape
        if (jacobian_batch, jacobian_horizon, jacobian_dof) != (
            batch_size,
            horizon,
            dof,
        ):
            log_and_raise(
                "joint_velocity and tool_jacobians must have the same batch, horizon, and dof"
            )
        if twist_dim != 6:
            log_and_raise(f"tool_jacobians twist dimension must be 6, got {twist_dim}")
        if linear_angular_weight.shape != (2,):
            log_and_raise(
                f"linear_angular_weight must have shape (2,), got {linear_angular_weight.shape}"
            )

        matrix_shape = (num_links, 3, 3)
        vector_shape = (num_links, 3)
        limit_shape = (num_links,)
        for name, tensor in (
            ("linear_projector", linear_projector),
            ("angular_projector", angular_projector),
        ):
            if tensor.shape != matrix_shape:
                log_and_raise(f"{name} must have shape {matrix_shape}, got {tensor.shape}")
        for name, tensor in (
            ("linear_axis_scale", linear_axis_scale),
            ("angular_axis_scale", angular_axis_scale),
        ):
            if tensor.shape != vector_shape:
                log_and_raise(f"{name} must have shape {vector_shape}, got {tensor.shape}")
        for name, tensor in (
            ("max_linear_speed", max_linear_speed),
            ("max_angular_speed", max_angular_speed),
        ):
            if tensor.shape != limit_shape:
                log_and_raise(f"{name} must have shape {limit_shape}, got {tensor.shape}")

        expected_cost_shape = (batch_size, horizon, 2 * num_links)
        expected_metric_shape = (batch_size, horizon, num_links)
        expected_gradient_shape = (batch_size, horizon, num_links, 3)
        if out_cost.shape != expected_cost_shape:
            log_and_raise(f"out_cost must have shape {expected_cost_shape}")
        if out_linear_violation.shape != expected_metric_shape:
            log_and_raise(f"out_linear_violation must have shape {expected_metric_shape}")
        if out_angular_violation.shape != expected_metric_shape:
            log_and_raise(f"out_angular_violation must have shape {expected_metric_shape}")
        if out_linear_twist_gradient.shape != expected_gradient_shape:
            log_and_raise(f"out_linear_twist_gradient must have shape {expected_gradient_shape}")
        if out_angular_twist_gradient.shape != expected_gradient_shape:
            log_and_raise(f"out_angular_twist_gradient must have shape {expected_gradient_shape}")

        check_float32_tensors(
            joint_velocity.device,
            joint_velocity=joint_velocity,
            tool_jacobians=tool_jacobians,
            linear_angular_weight=linear_angular_weight,
            linear_projector=linear_projector,
            angular_projector=angular_projector,
            linear_axis_scale=linear_axis_scale,
            angular_axis_scale=angular_axis_scale,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
        )
        wp_device, wp_stream = get_warp_device_stream(joint_velocity)
        wp.launch(
            kernel=tool_velocity_limit_kernel,
            dim=batch_size * horizon * num_links,
            inputs=[
                wp.from_torch(joint_velocity.detach().contiguous().view(-1), dtype=wp.float32),
                wp.from_torch(tool_jacobians.detach().contiguous().view(-1), dtype=wp.float32),
                wp.from_torch(linear_angular_weight.view(-1), dtype=wp.float32),
                wp.from_torch(linear_projector.view(-1, 3, 3), dtype=wp.mat33),
                wp.from_torch(angular_projector.view(-1, 3, 3), dtype=wp.mat33),
                wp.from_torch(linear_axis_scale.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(angular_axis_scale.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(max_linear_speed.view(-1), dtype=wp.float32),
                wp.from_torch(max_angular_speed.view(-1), dtype=wp.float32),
                wp.from_torch(out_cost.view(-1), dtype=wp.float32),
                wp.from_torch(out_linear_violation.view(-1), dtype=wp.float32),
                wp.from_torch(out_angular_violation.view(-1), dtype=wp.float32),
                wp.from_torch(out_linear_twist_gradient.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(out_angular_twist_gradient.view(-1, 3), dtype=wp.vec3),
                batch_size,
                horizon,
                num_links,
                dof,
            ],
            device=wp_device,
            stream=wp_stream,
        )

        ctx.use_grad_input = use_grad_input
        ctx.mark_non_differentiable(out_linear_violation, out_angular_violation)
        ctx.save_for_backward(
            joint_velocity,
            tool_jacobians,
            out_linear_twist_gradient,
            out_angular_twist_gradient,
        )
        return out_cost, out_linear_violation, out_angular_violation

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_cost: Optional[torch.Tensor],
        grad_linear_violation: Optional[torch.Tensor],
        grad_angular_violation: Optional[torch.Tensor],
    ):
        """Map saved twist cotangents to joint velocity and Jacobian cotangents."""
        del grad_linear_violation, grad_angular_violation
        joint_velocity_gradient = None
        jacobian_gradient = None
        if grad_cost is not None and (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            (
                joint_velocity,
                tool_jacobians,
                linear_twist_gradient,
                angular_twist_gradient,
            ) = ctx.saved_tensors
            if ctx.use_grad_input:
                output_multiplier = grad_cost.reshape(
                    *grad_cost.shape[:2],
                    tool_jacobians.shape[2],
                    2,
                )
                linear_twist_gradient = linear_twist_gradient * output_multiplier[
                    ..., 0
                ].unsqueeze(-1)
                angular_twist_gradient = angular_twist_gradient * output_multiplier[
                    ..., 1
                ].unsqueeze(-1)
            twist_gradient = torch.cat(
                (linear_twist_gradient, angular_twist_gradient),
                dim=-1,
            )
            if ctx.needs_input_grad[0]:
                joint_velocity_gradient = (tool_jacobians * twist_gradient.unsqueeze(-1)).sum(
                    dim=(2, 3)
                )
            if ctx.needs_input_grad[1]:
                jacobian_gradient = twist_gradient.unsqueeze(-1) * joint_velocity.unsqueeze(
                    2
                ).unsqueeze(3)

        return (joint_velocity_gradient, jacobian_gradient) + (None,) * 13
