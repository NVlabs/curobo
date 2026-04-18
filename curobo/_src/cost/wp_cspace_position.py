# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Third Party
from typing import Optional

import torch
import warp as wp

# CuRobo
from curobo._src.cost.warp_bound_util import (
    aggregate_bound_cost,
    shrink_bounds_with_activation_distance,
)
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream


class PositionCSpaceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pos: torch.Tensor,
        joint_torque: torch.Tensor,
        target_joint_position: torch.Tensor,
        idxs_target_joint_position: torch.Tensor,
        p_l: torch.Tensor,
        effort_limit: torch.Tensor,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        cspace_target_weight: torch.Tensor,
        cspace_target_dof_weight: torch.Tensor,
        squared_l2_regularization_weight: torch.Tensor,
        current_position: torch.Tensor,
        current_velocity: torch.Tensor,
        idxs_current_state: torch.Tensor,
        v_b: torch.Tensor,
        state_dt: torch.Tensor,
        out_cost: torch.Tensor,
        out_gp: torch.Tensor,
        out_gtau: torch.Tensor,
        use_grad_input: bool,
    ):
        """Compute the bound position cost and gradient.

        Args:
            ctx: torch.autograd.Function context
            pos: CSpace position of shape (batch_size, horizon, dof)
            joint_torque: Joint torque of shape (batch_size, horizon, dof)
            target_joint_position: Retract configuration of shape (-1, dof).
            idxs_target_joint_position: Retract index of shape (batch_size,). Data type is int32.
            p_l: Joint position limits of shape (2, dof).
            effort_limit: Joint effort limits of shape (2, dof).
            weight: Weight to scale the cost terms. Shape is (2).
            activation_distance: Activation distance of shape (2).
            cspace_target_weight: Weight to scale the target joint position cost term. Shape is (1).
            cspace_target_dof_weight: Per dof weight for the target joint position cost. Shape is (dof).
            squared_l2_regularization_weight: Shape (2,) for [velocity, acceleration] regularization.
            current_position: Previous joint position of shape (-1, dof).
            current_velocity: Previous joint velocity of shape (-1, dof). Used for acceleration
                regularization when ``squared_l2_regularization_weight[1] > 0``.
            idxs_current_state: Index into current_position/current_velocity per batch. Shape is (batch_size,).
            v_b: Joint velocity limits of shape (2, dof).
            state_dt: Time step per batch. Shape is (batch_size,).
            out_cost: Output cost of shape (batch_size, horizon, dof).
            out_gp: Gradient w.r.t. position. Shape is (batch_size, horizon, dof).
            out_gtau: Gradient w.r.t. joint torque. Shape is (batch_size, horizon, dof).
            use_grad_input: Whether to use the gradient of the input for the backward pass.

        Returns:
            out_cost: Output cost of shape (batch_size, horizon)
        """
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(
            target_joint_position,
            idxs_target_joint_position,
            p_l,
            effort_limit,
            weight,
            activation_distance,
            cspace_target_weight,
            cspace_target_dof_weight,
            squared_l2_regularization_weight,
            current_position,
            current_velocity,
            idxs_current_state,
            v_b,
            state_dt,
        )

        wp_device, wp_stream = get_warp_device_stream(pos)
        b, h, dof = pos.shape

        requires_grad = pos.requires_grad

        if joint_torque.shape != (b, h, dof):
            log_and_raise(
                f"joint_torque.shape: {joint_torque.shape}" + f" != (b, h, dof): {(b, h, dof)}"
            )

        if target_joint_position.shape[-1] != dof:
            log_and_raise(f"target_joint_position.shape[-1]: {target_joint_position.shape[-1]}" + f" != dof: {dof}")
        if idxs_target_joint_position.ndim == 2:
            idxs_target_joint_position = idxs_target_joint_position.squeeze(1)
        if idxs_target_joint_position.shape != (b,):
            log_and_raise(f"idxs_target_joint_position.shape: {idxs_target_joint_position.shape}" + f" != (b,): {(b,)}")
        if p_l.shape != (2, dof):
            log_and_raise(f"p_l.shape: {p_l.shape}" + f" != (2, dof): {(2, dof)}")
        if effort_limit.shape != (2, dof):
            log_and_raise(f"effort_limit.shape: {effort_limit.shape}" + f" != (2, dof): {(2, dof)}")
        if weight.shape != (2,):
            log_and_raise(f"weight.shape: {weight.shape}" + f" != (2,): {(2,)}")
        if activation_distance.shape != (2,):
            log_and_raise(
                f"activation_distance.shape: {activation_distance.shape}" + f" != (2,): {(2,)}"
            )
        if cspace_target_weight.shape != (1,):
            log_and_raise(
                f"cspace_target_weight.shape: {cspace_target_weight.shape}"
                + f" != (1,): {(1,)}"
            )
        if cspace_target_dof_weight.shape != (dof,):
            log_and_raise(
                f"cspace_target_dof_weight.shape: {cspace_target_dof_weight.shape}"
                + f" != (dof,): {(dof,)}"
            )
        if squared_l2_regularization_weight.shape != (2,):
            log_and_raise(
                f"squared_l2_regularization_weight.shape: {squared_l2_regularization_weight.shape}"
                + f" != (2,): {(2,)}"
            )

        if idxs_current_state.ndim == 2:
            idxs_current_state = idxs_current_state.squeeze(1)

        n_states = current_position.shape[0]
        if current_velocity.shape[0] != n_states:
            log_and_raise(
                f"current_velocity rows ({current_velocity.shape[0]}) "
                f"!= current_position rows ({n_states})"
            )
        #if state_dt.view(-1).shape[0] >= n_states:
        #    log_and_raise(
        #        f"state_dt has {state_dt.view(-1).shape[0]} elements, "
        #        f"expected {n_states} (one per current_position row)"
        #     )

        if out_cost.shape != (b, h, dof):
            log_and_raise(f"out_cost.shape: {out_cost.shape}" + f" != (b, h, dof): {(b, h, dof)}")
        if out_gp.shape != (b, h, dof):
            log_and_raise(f"out_gp.shape: {out_gp.shape}" + f" != (b, h, dof): {(b, h, dof)}")


        wp.launch(
            kernel=forward_cspace_position_warp,
            dim=b * h * dof,
            inputs=[
                wp.from_torch(pos.detach().view(-1), dtype=wp.float32),
                wp.from_torch(joint_torque.detach().view(-1), dtype=wp.float32),
                wp.from_torch(target_joint_position.detach().view(-1), dtype=wp.float32),
                wp.from_torch(idxs_target_joint_position.detach().view(-1), dtype=wp.int32),
                wp.from_torch(p_l.view(-1), dtype=wp.float32),
                wp.from_torch(effort_limit.view(-1), dtype=wp.float32),
                wp.from_torch(weight, dtype=wp.float32),
                wp.from_torch(activation_distance, dtype=wp.float32),
                wp.from_torch(cspace_target_weight.view(-1), dtype=wp.float32),
                wp.from_torch(cspace_target_dof_weight.view(-1), dtype=wp.float32),
                wp.from_torch(squared_l2_regularization_weight, dtype=wp.float32),
                wp.from_torch(current_position.detach().view(-1), dtype=wp.float32),
                wp.from_torch(current_velocity.detach().view(-1), dtype=wp.float32),
                wp.from_torch(idxs_current_state.detach().view(-1), dtype=wp.int32),
                wp.from_torch(v_b.view(-1), dtype=wp.float32),
                wp.from_torch(state_dt.detach().view(-1), dtype=wp.float32),
                wp.from_torch(out_cost.view(-1), dtype=wp.float32),
                wp.from_torch(out_gp.view(-1), dtype=wp.float32),
                wp.from_torch(out_gtau.view(-1), dtype=wp.float32),
                requires_grad,
                b,
                h,
                dof,
            ],
            device=wp_device,
            stream=wp_stream,
        )
        ctx.use_grad_input = use_grad_input

        ctx.save_for_backward(out_gp, out_gtau)

        return out_cost

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out_cost: Optional[torch.Tensor]):
        (p_grad, tau_grad) = ctx.saved_tensors

        p_g = None
        e_g = None
        if grad_out_cost is not None:
            if ctx.needs_input_grad[0]:
                p_g = p_grad
                if ctx.use_grad_input:
                    p_g = p_grad * grad_out_cost
            if ctx.needs_input_grad[1]:
                e_g = tau_grad
                if ctx.use_grad_input:
                    e_g = tau_grad * grad_out_cost

        return (
            p_g,
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
        )



@wp.kernel(enable_backward=False)
def forward_cspace_position_warp(
    pos: wp.array(dtype=wp.float32),
    effort: wp.array(dtype=wp.float32),
    cspace_target: wp.array(dtype=wp.float32),
    cspace_target_idx: wp.array(dtype=wp.int32),
    p_b: wp.array(dtype=wp.float32),
    effort_b: wp.array(dtype=wp.float32),
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    cspace_target_weight: wp.array(dtype=wp.float32),
    cspace_target_dof_weight: wp.array(dtype=wp.float32),
    squared_l2_reg_weight: wp.array(dtype=wp.float32),
    current_position: wp.array(dtype=wp.float32),
    current_velocity: wp.array(dtype=wp.float32),
    idxs_current_state: wp.array(dtype=wp.int32),
    v_b: wp.array(dtype=wp.float32),
    state_dt: wp.array(dtype=wp.float32),
    out_cost: wp.array(dtype=wp.float32),
    out_grad_p: wp.array(dtype=wp.float32),
    out_grad_tau: wp.array(dtype=wp.float32),
    write_grad: wp.uint8,
    batch_size: wp.int32,
    horizon: wp.int32,
    dof: wp.int32,
):
    tid = wp.tid()
    b_id = wp.int32(0)
    h_id = wp.int32(0)
    d_id = wp.int32(0)
    b_addrs = int(0)

    w = wp.float32(0.0)
    c_p = wp.float32(0.0)
    g_p = wp.float32(0.0)
    g_tau = wp.float32(0.0)
    c_total = wp.float32(0.0)
    cur_id = wp.int32(0)

    b_id = tid / (horizon * dof)
    h_id = (tid - (b_id * horizon * dof)) / dof
    d_id = tid - (b_id * horizon * dof + h_id * dof)
    if b_id >= batch_size or h_id >= horizon or d_id >= dof:
        return
    # read weights:
    eta_p = activation_distance[0]
    eta_tau = activation_distance[1]
    w = weight[0]
    tau_w = weight[1]
    tau_limit_lower = effort_b[d_id]
    tau_limit_upper = effort_b[dof + d_id]

    tau_limit_lower, tau_limit_upper = shrink_bounds_with_activation_distance(
        tau_limit_lower,
        tau_limit_upper,
        eta_tau,
    )

    p_l = p_b[d_id]  # lower bound
    p_u = p_b[dof + d_id]  # upper bound

    p_l, p_u = shrink_bounds_with_activation_distance(
        p_l,
        p_u,
        eta_p,
    )

    cur_id = idxs_current_state[b_id]
    dt_val = state_dt[cur_id]
    if dt_val > 0.0:
        cur_p = current_position[cur_id * dof + d_id]
        p_l = wp.max(p_l, cur_p + v_b[d_id] * dt_val)
        p_u = wp.min(p_u, cur_p + v_b[dof + d_id] * dt_val)

    # compute cost:
    b_addrs = b_id * horizon * dof + h_id * dof + d_id

    c_p = pos[b_addrs]
    c_tau = effort[b_addrs]

    # bound cost:
    c_total, g_p = aggregate_bound_cost(
        c_p,
        p_l,
        p_u,
        w,
        c_total,
        g_p,
    )
    if tau_w > 0.0:
        c_total, g_tau = aggregate_bound_cost(
            c_tau,
            tau_limit_lower,
            tau_limit_upper,
            tau_w,
            c_total,
            g_tau,
        )

    target_w = wp.float32(0.0)
    target_w = cspace_target_weight[0]
    target_w *= cspace_target_dof_weight[d_id]
    if target_w > 0.0:
        target_id = cspace_target_idx[b_id]
        target_id = target_id * dof + d_id
        target_p = cspace_target[target_id]
        error = c_p - target_p
        c_total += target_w * error * error
        g_p += 2.0 * target_w * error

    # implied velocity / acceleration regularization (active when dt > 0)
    # weights are retimed by dt^n (n=1 for velocity, n=2 for acceleration)
    # to approximate time-integral and be invariant to dt choice
    vel_reg_w = squared_l2_reg_weight[0] * dt_val
    acc_reg_w = squared_l2_reg_weight[1] * dt_val * dt_val
    if dt_val > 0.0 and (vel_reg_w > 0.0 or acc_reg_w > 0.0):
        v_implied = (c_p - cur_p) / dt_val
        if vel_reg_w > 0.0:
            c_total += 0.5 * vel_reg_w * v_implied * v_implied
            g_p += vel_reg_w * v_implied / dt_val
        if acc_reg_w > 0.0:
            cur_v = current_velocity[cur_id * dof + d_id]
            a_implied = (v_implied - cur_v) / dt_val
            c_total += 0.5 * acc_reg_w * a_implied * a_implied
            g_p += acc_reg_w * a_implied / (dt_val * dt_val)

    out_cost[b_addrs] = c_total

    if write_grad == 1:
        out_grad_p[b_addrs] = g_p
        out_grad_tau[b_addrs] = g_tau
