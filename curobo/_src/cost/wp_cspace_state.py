# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Third Party
from typing import Optional

import torch
import warp as wp

# CuRobo
from curobo._src.cost.warp_bound_util import (
    aggregate_bound_cost,
    aggregate_energy_regularization,
    aggregate_squared_l2_regularization,
    shrink_bounds_with_activation_distance,
)
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream


@wp.kernel(enable_backward=False)
def forward_cspace_state_warp(
    pos: wp.array(dtype=wp.float32),
    vel: wp.array(dtype=wp.float32),
    acc: wp.array(dtype=wp.float32),
    jerk: wp.array(dtype=wp.float32),
    effort: wp.array(dtype=wp.float32),
    state_dt: wp.array(dtype=wp.float32),
    target_joint_position: wp.array(dtype=wp.float32),
    idxs_target_joint_position: wp.array(dtype=wp.int32),
    p_b: wp.array(dtype=wp.float32),
    v_b: wp.array(dtype=wp.float32),
    a_b: wp.array(dtype=wp.float32),
    j_b: wp.array(dtype=wp.float32),
    effort_b: wp.array(dtype=wp.float32),
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    squared_l2_regularization_weights: wp.array(dtype=wp.float32),
    cspace_target_weight: wp.array(dtype=wp.float32),
    cspace_non_terminal_weight_factor: wp.array(dtype=wp.float32),
    cspace_target_dof_weight: wp.array(dtype=wp.float32),
    out_cost: wp.array(dtype=wp.float32),
    out_grad_p: wp.array(dtype=wp.float32),
    out_grad_v: wp.array(dtype=wp.float32),
    out_grad_a: wp.array(dtype=wp.float32),
    out_grad_j: wp.array(dtype=wp.float32),
    out_grad_tau: wp.array(dtype=wp.float32),
    write_grad: wp.uint8,  # this should be a bool
    batch_size: wp.int32,
    horizon: wp.int32,
    dof: wp.int32,
    retime_weights: bool,
    retime_regularization_weights: bool,
):
    tid = wp.tid()
    # initialize variables:
    b_id = wp.int32(0)
    h_id = wp.int32(0)
    d_id = wp.int32(0)
    b_addrs = int(0)


    current_position = wp.float32(0.0)
    current_velocity = wp.float32(0.0)
    current_acceleration = wp.float32(0.0)
    current_jerk = wp.float32(0.0)
    current_effort = wp.float32(0.0)

    gradient_position = wp.float32(0.0)
    gradient_velocity = wp.float32(0.0)
    gradient_acceleration = wp.float32(0.0)
    gradient_jerk = wp.float32(0.0)
    gradient_effort = wp.float32(0.0)

    cost_total = wp.float32(0.0)

    # we launch batch * horizon * dof kernels
    b_id = tid / (horizon * dof)
    h_id = (tid - (b_id * horizon * dof)) / dof
    d_id = tid - (b_id * horizon * dof + h_id * dof)
    if b_id >= batch_size or h_id >= horizon or d_id >= dof:
        return

    dt = state_dt[b_id]
    cspace_target_w = cspace_target_weight[0]
    cspace_target_dof_w = cspace_target_dof_weight[d_id]
    cspace_non_terminal_factor = cspace_non_terminal_weight_factor[0]

    if h_id < horizon - 1:
        cspace_target_w *= cspace_non_terminal_factor
    # read weights:
    w_bounds = wp.vector(weight[0], weight[1], weight[2], weight[3], weight[4])
    w_reg = wp.vector(
        squared_l2_regularization_weights[0],
        squared_l2_regularization_weights[1],
        squared_l2_regularization_weights[2],
        squared_l2_regularization_weights[3],
        squared_l2_regularization_weights[4],
    )

    # reweight terms:
    if retime_weights:
        w_bounds[1] = (dt) * w_bounds[1]
        w_bounds[2] = wp.pow((dt), 2.0) * w_bounds[2]
        w_bounds[3] = wp.pow((dt), 3.0) * w_bounds[3]

    if retime_regularization_weights:
        w_reg[0] = (dt) * w_reg[0]
        w_reg[1] = wp.pow((dt), 2.0) * w_reg[1]
        w_reg[2] = wp.pow((dt), 3.0) * w_reg[2]
        w_reg[4] = (dt) * w_reg[4]

    # compute cost:
    b_addrs = b_id * horizon * dof + h_id * dof + d_id

    # read buffers:
    current_velocity = vel[b_addrs]
    current_acceleration = acc[b_addrs]
    current_position = pos[b_addrs]
    current_jerk = jerk[b_addrs]
    current_effort = effort[b_addrs]
    # if w_j > 0.0:

    local_activation_distance = wp.vector(
        activation_distance[0],
        activation_distance[1],
        activation_distance[2],
        activation_distance[3],
        activation_distance[4],
    )

    # Compute bound terms:

    # read limits:
    position_limit_lower = p_b[d_id]
    position_limit_upper = p_b[dof + d_id]
    velocity_limit_lower = v_b[d_id]
    velocity_limit_upper = v_b[dof + d_id]
    acceleration_limit_lower = a_b[d_id]
    acceleration_limit_upper = a_b[dof + d_id]
    jerk_limit_lower = j_b[d_id]
    jerk_limit_upper = j_b[dof + d_id]
    effort_limit_lower = effort_b[d_id]
    effort_limit_upper = effort_b[dof + d_id]

    position_limit_lower, position_limit_upper = shrink_bounds_with_activation_distance(
        position_limit_lower,
        position_limit_upper,
        local_activation_distance[0],
    )

    velocity_limit_lower, velocity_limit_upper = shrink_bounds_with_activation_distance(
        velocity_limit_lower,
        velocity_limit_upper,
        local_activation_distance[1],
    )

    acceleration_limit_lower, acceleration_limit_upper = shrink_bounds_with_activation_distance(
        acceleration_limit_lower,
        acceleration_limit_upper,
        local_activation_distance[2],
    )

    jerk_limit_lower, jerk_limit_upper = shrink_bounds_with_activation_distance(
        jerk_limit_lower,
        jerk_limit_upper,
        local_activation_distance[3],
    )

    effort_limit_lower, effort_limit_upper = shrink_bounds_with_activation_distance(
        effort_limit_lower,
        effort_limit_upper,
        local_activation_distance[4],
    )

    # bound cost:
    cost_total, gradient_position = aggregate_bound_cost(
        current_position,
        position_limit_lower,
        position_limit_upper,
        w_bounds[0],
        cost_total,
        gradient_position,
    )
    cost_total, gradient_velocity = aggregate_bound_cost(
        current_velocity,
        velocity_limit_lower,
        velocity_limit_upper,
        w_bounds[1],
        cost_total,
        gradient_velocity,
    )
    cost_total, gradient_acceleration = aggregate_bound_cost(
        current_acceleration,
        acceleration_limit_lower,
        acceleration_limit_upper,
        w_bounds[2],
        cost_total,
        gradient_acceleration,
    )
    cost_total, gradient_jerk = aggregate_bound_cost(
        current_jerk,
        jerk_limit_lower,
        jerk_limit_upper,
        w_bounds[3],
        cost_total,
        gradient_jerk,
    )

    cost_total, gradient_effort = aggregate_bound_cost(
        current_effort,
        effort_limit_lower,
        effort_limit_upper,
        w_bounds[4],
        cost_total,
        gradient_effort,
    )



    if cspace_target_w > 0.0:
        cspace_target_w *= cspace_target_dof_w
        target_id = idxs_target_joint_position[b_id]
        target_id = target_id * dof + d_id
        target_position = target_joint_position[target_id]
        error = current_position - target_position
        cost_total += cspace_target_w * error * error
        gradient_position += 2.0 * cspace_target_w * error


    cost_total, gradient_velocity = aggregate_squared_l2_regularization(
        current_velocity,
        w_reg[0],
        cost_total,
        gradient_velocity,
    )

    cost_total, gradient_acceleration = aggregate_squared_l2_regularization(
        current_acceleration,
        w_reg[1],
        cost_total,
        gradient_acceleration,
    )

    cost_total, gradient_jerk = aggregate_squared_l2_regularization(
        current_jerk,
        w_reg[2],
        cost_total,
        gradient_jerk,
    )

    cost_total, gradient_effort = aggregate_squared_l2_regularization(
        current_effort,
        w_reg[3],
        cost_total,
        gradient_effort,
    )

    if w_reg[4] > 0.0:
        (
            cost_total,
            gradient_effort,
            gradient_velocity,
        ) = aggregate_energy_regularization(
            current_effort,
            current_velocity,
            dt,
            w_reg[4],
            cost_total,
            gradient_effort,
            gradient_velocity,
        )


    out_cost[b_addrs] = cost_total

    # compute gradient
    if write_grad == 1:
        out_grad_p[b_addrs] = gradient_position

        out_grad_v[b_addrs] = gradient_velocity

        out_grad_a[b_addrs] = gradient_acceleration
        out_grad_j[b_addrs] = gradient_jerk

        out_grad_tau[b_addrs] = gradient_effort


class StateCSpaceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pos: torch.Tensor,
        vel: torch.Tensor,
        acc: torch.Tensor,
        jerk: torch.Tensor,
        joint_torque: torch.Tensor,
        state_dt: torch.Tensor,
        target_joint_position: torch.Tensor,
        idxs_target_joint_position: torch.Tensor,
        p_b: torch.Tensor,
        v_b: torch.Tensor,
        a_b: torch.Tensor,
        j_b: torch.Tensor,
        effort_limit: torch.Tensor,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        squared_l2_regularization_weights: torch.Tensor,
        cspace_target_weight: torch.Tensor,
        cspace_non_terminal_weight_factor: torch.Tensor,
        cspace_target_dof_weight: torch.Tensor,
        out_cost: torch.Tensor,
        out_gp: torch.Tensor,
        out_gv: torch.Tensor,
        out_ga: torch.Tensor,
        out_gj: torch.Tensor,
        out_gtau: torch.Tensor,
        retime_weights: bool,
        retime_regularization_weights: bool,
        use_grad_input: bool,
    ):
        """Compute the bound cost and gradient.

        Args:
            ctx: torch.autograd.Function context
            pos: CSpace position of shape (batch_size, horizon, dof)
            vel: CSpace velocity of shape (batch_size, horizon, dof)
            acc: CSpace acceleration of shape (batch_size, horizon, dof)
            jerk: CSpace jerk of shape (batch_size, horizon, dof)
            joint_torque: Joint torque of shape (batch_size, horizon, dof)
            state_dt: State time step of shape (batch_size, horizon)
            target_joint_position: Target joint position of shape (-1, dof), where -1 is any number of
                configurations as the elements are accessed using idxs_target_joint_position. E.g., for batch 0,
                the first element is accessed using idxs_target_joint_position[0], the second element is accessed
                using idxs_target_joint_position[1], etc.
            idxs_target_joint_position: Target joint position index of shape (batch_size, dof). Data type is int32.
            p_b: Joint position limits of shape (2, dof). The first element is the lower limit and
                the second element is the upper limit.
            v_b: Joint velocity limits of shape (2, dof). The first element is the lower limit and
                the second element is the upper limit.
            a_b: Joint acceleration limits of shape (2, dof). The first element is the lower limit and
                the second element is the upper limit.
            j_b: Joint jerk limits of shape (2, dof). The first element is the lower limit and
                the second element is the upper limit.
            effort_limit: Joint effort limits of shape (2, dof). The first element is the lower limit
                and the second element is the upper limit.
            weight: Weight to scale the cost terms. Shape is (5,).
            activation_distance: Activation distance of shape (5,).
            squared_l2_regularization_weights: Squared l2 regularization weights of shape (5,).
            cspace_target_weight: Weight to scale the cspace target cost term. Shape is (1).
            cspace_target_dof_weight: Per dof weight for the cspace target cost term. Shape is (dof).
            out_cost: Output cost of shape (batch_size, horizon, dof)
            out_gp: Gradient of the cost with respect to position. Shape is (batch_size, horizon,
                dof).
            out_gv: Gradient of the cost with respect to velocity. Shape is (batch_size, horizon,
                dof).
            out_ga: Gradient of the cost with respect to acceleration. Shape is (batch_size,
                horizon, dof).
            out_gj: Gradient of the cost with respect to jerk. Shape is (batch_size, horizon, dof).
            out_gtau: Gradient of the cost with respect to joint torque. Shape is (batch_size,
                horizon, dof).
            retime_weights: Whether to retime the weights based on the time step.
            retime_regularization_weights: Whether to retime the regularization weights based on
                the time step.

        Returns:
            out_cost: Output cost of shape (batch_size, horizon)
        """
        # scale the weights for smoothness by this dt:
        wp_device, wp_stream = get_warp_device_stream(vel)
        b, h, dof = vel.shape

        if pos.ndim != 3:
            log_and_raise("pos.ndim != 3")
        if vel.ndim != 3:
            log_and_raise("vel.ndim != 3")
        if acc.ndim != 3:
            log_and_raise("acc.ndim != 3")
        if jerk.ndim != 3:
            log_and_raise("jerk.ndim != 3")
        if joint_torque.ndim != 3:
            log_and_raise("joint_torque.ndim != 3")
        if state_dt.ndim != 1:
            log_and_raise("state_dt.ndim != 1")
        if target_joint_position.ndim != 2:
            if target_joint_position.ndim == 1:
                target_joint_position = target_joint_position.unsqueeze(0)
            else:
                log_and_raise("target_joint_position.ndim != 2")
        if idxs_target_joint_position.ndim != 1:
            idxs_target_joint_position = idxs_target_joint_position.view(-1)
        if pos.shape != (b, h, dof):
            log_and_raise("pos.shape != (b, h, dof)")
        if vel.shape != (b, h, dof):
            log_and_raise("vel.shape != (b, h, dof)")
        if acc.shape != (b, h, dof):
            log_and_raise("acc.shape != (b, h, dof)")
        if jerk.shape != (b, h, dof):
            log_and_raise("jerk.shape != (b, h, dof)")
        if joint_torque.shape != (b, h, dof):
            log_and_raise("joint_torque.shape != (b, h, dof)")
        if state_dt.shape != (b,):
            log_and_raise(
                "state_dt.shape != (b), got {}".format(state_dt.shape) + " expected ({})".format(b)
            )
        if target_joint_position.shape[-1] != dof:
            log_and_raise(
                "target_joint_position.shape != (-1, dof), got {}".format(target_joint_position.shape)
                + " expected (-1, dof)"
            )
        if idxs_target_joint_position.shape != (b,):
            log_and_raise(
                "idxs_target_joint_position.shape != (b), got {}".format(idxs_target_joint_position.shape)
                + " expected ({})".format(b)
            )

        if p_b.shape != (2, dof):
            log_and_raise(
                "p_b.shape != (2, dof), got {}".format(p_b.shape)
                + " expected ({})".format(
                    2,
                )
            )
        if v_b.shape != (2, dof):
            log_and_raise(
                "v_b.shape != (2, dof), got {}".format(v_b.shape)
                + " expected ({})".format(
                    2,
                )
            )
        if a_b.shape != (2, dof):
            log_and_raise(
                "a_b.shape != (2, dof), got {}".format(a_b.shape)
                + " expected ({})".format(
                    2,
                )
            )
        if j_b.shape != (2, dof):
            log_and_raise(
                "j_b.shape != (2, dof), got {}".format(j_b.shape)
                + " expected ({})".format(
                    2,
                )
            )
        if effort_limit.shape != (2, dof):
            log_and_raise(
                "effort_limit.shape != (2, dof), got {}".format(effort_limit.shape)
                + " expected ({})".format(
                    2,
                )
            )
        if weight.shape != (5,):
            log_and_raise(
                "weight.shape != (5), got {}".format(weight.shape) + " expected ({})".format(5)
            )
        if activation_distance.shape != (5,):
            log_and_raise(
                "activation_distance.shape != (5), got {}".format(activation_distance.shape)
                + " expected ({})".format(5)
            )

        if cspace_target_weight.shape != (1,):
            log_and_raise(
                "cspace_target_weight.shape != (1), got {}".format(
                    cspace_target_weight.shape
                )
                + " expected ({})".format(1)
            )
        if cspace_non_terminal_weight_factor.shape != (1,):
            log_and_raise(
                "cspace_non_terminal_weight_factor.shape != (1), got {}".format(cspace_non_terminal_weight_factor.shape)
                + " expected ({})".format(1)
            )
        if cspace_target_dof_weight.shape != (dof,):
            log_and_raise(
                "cspace_target_dof_weight.shape != (dof), got {}".format(cspace_target_dof_weight.shape)
                + " expected ({})".format(dof)
            )
        if out_cost.shape != (b, h, dof):
            log_and_raise(
                "out_cost.shape != (b, h, dof), got {}".format(out_cost.shape)
                + " expected ({})".format(
                    b,
                )
            )
        if out_gp.shape != (b, h, dof):
            log_and_raise(
                "out_gp.shape != (b, h, dof), got {}".format(out_gp.shape)
                + " expected ({})".format(
                    b,
                )
            )
        if out_gv.shape != (b, h, dof):
            log_and_raise(
                "out_gv.shape != (b, h, dof), got {}".format(out_gv.shape)
                + " expected ({})".format(
                    b,
                )
            )

        if out_ga.shape != (b, h, dof):
            log_and_raise(
                "out_ga.shape != (b, h, dof), got {}".format(out_ga.shape)
                + " expected ({})".format(
                    b,
                )
            )

        if out_gj.shape != (b, h, dof):
            log_and_raise(
                "out_gj.shape != (b, h, dof), got {}".format(out_gj.shape)
                + " expected ({})".format(
                    b,
                )
            )

        if out_gtau.shape != (b, h, dof):
            log_and_raise(
                "out_gtau.shape != (b, h, dof), got {}".format(out_gtau.shape)
                + " expected ({})".format(
                    b,
                )
            )
        if squared_l2_regularization_weights.shape != (5,):
            log_and_raise(
                "squared_l2_regularization_weights.shape != (5), got {}".format(squared_l2_regularization_weights.shape)
                + " expected ({})".format(5)
            )

        write_grad = pos.requires_grad
        wp.launch(
            kernel=forward_cspace_state_warp,
            dim=b * h * dof,
            inputs=[
                wp.from_torch(pos.detach().view(-1), dtype=wp.float32),
                wp.from_torch(vel.detach().view(-1), dtype=wp.float32),
                wp.from_torch(acc.detach().view(-1), dtype=wp.float32),
                wp.from_torch(jerk.detach().view(-1), dtype=wp.float32),
                wp.from_torch(joint_torque.detach().view(-1), dtype=wp.float32),
                wp.from_torch(state_dt.detach().view(-1), dtype=wp.float32),
                wp.from_torch(target_joint_position.detach().view(-1), dtype=wp.float32),
                wp.from_torch(idxs_target_joint_position.detach().view(-1), dtype=wp.int32),
                wp.from_torch(p_b.view(-1), dtype=wp.float32),
                wp.from_torch(v_b.view(-1), dtype=wp.float32),
                wp.from_torch(a_b.view(-1), dtype=wp.float32),
                wp.from_torch(j_b.view(-1), dtype=wp.float32),
                wp.from_torch(effort_limit.view(-1), dtype=wp.float32),
                wp.from_torch(weight, dtype=wp.float32),
                wp.from_torch(activation_distance, dtype=wp.float32),
                wp.from_torch(squared_l2_regularization_weights, dtype=wp.float32),
                wp.from_torch(cspace_target_weight, dtype=wp.float32),
                wp.from_torch(cspace_non_terminal_weight_factor, dtype=wp.float32),
                wp.from_torch(cspace_target_dof_weight, dtype=wp.float32),
                wp.from_torch(out_cost.view(-1), dtype=wp.float32),
                wp.from_torch(out_gp.view(-1), dtype=wp.float32),
                wp.from_torch(out_gv.view(-1), dtype=wp.float32),
                wp.from_torch(out_ga.view(-1), dtype=wp.float32),
                wp.from_torch(out_gj.view(-1), dtype=wp.float32),
                wp.from_torch(out_gtau.view(-1), dtype=wp.float32),
                write_grad,
                b,
                h,
                dof,
                retime_weights,
                retime_regularization_weights,
            ],
            device=wp_device,
            stream=wp_stream,
            adjoint=False,
        )




        ctx.save_for_backward(out_gp, out_gv, out_ga, out_gj, out_gtau)
        ctx.use_grad_input = use_grad_input
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(
            pos,
            vel,
            acc,
            jerk,
            joint_torque,
            state_dt,
            target_joint_position,
            idxs_target_joint_position,
            p_b,
            v_b,
            a_b,
            j_b,
            effort_limit,
            weight,
            activation_distance,
            cspace_target_weight,
            cspace_non_terminal_weight_factor,
            cspace_target_dof_weight,
            out_gp,
            out_gv,
            out_ga,
            out_gj,
            out_gtau,

        )

        return out_cost

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out_cost: Optional[torch.Tensor]):
        (
            p_grad,
            v_grad,
            a_grad,
            j_grad,
            tau_grad,
        ) = ctx.saved_tensors
        v_g = None
        a_g = None
        p_g = None
        j_g = None
        e_g = None
        if grad_out_cost is not None:
            if ctx.needs_input_grad[0]:
                if ctx.use_grad_input:
                    p_g = p_grad * grad_out_cost
                else:
                    p_g = p_grad
            if ctx.needs_input_grad[1]:
                if ctx.use_grad_input:
                    v_g = v_grad * grad_out_cost
                else:
                    v_g = v_grad
            if ctx.needs_input_grad[2]:
                if ctx.use_grad_input:
                    a_g = a_grad * grad_out_cost
                else:
                    a_g = a_grad
            if ctx.needs_input_grad[3]:
                if ctx.use_grad_input:
                    j_g = j_grad * grad_out_cost
                else:
                    j_g = j_grad
            if ctx.needs_input_grad[4]:
                if ctx.use_grad_input:
                    e_g = tau_grad * grad_out_cost
                else:
                    e_g = tau_grad

        return (
            p_g,
            v_g,
            a_g,
            j_g,
            e_g,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
