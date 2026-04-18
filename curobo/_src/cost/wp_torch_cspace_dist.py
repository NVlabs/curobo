# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors
from curobo._src.util.warp import get_warp_device_stream


@wp.kernel
def forward_l2_warp(
    pos: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
    target_idx: wp.array(dtype=wp.int32),
    weight: wp.array(dtype=wp.float32),
    terminal_dof_weight: wp.array(dtype=wp.float32),
    non_terminal_dof_weight: wp.array(dtype=wp.float32),
    out_cost: wp.array(dtype=wp.float32),
    out_grad_p: wp.array(dtype=wp.float32),
    write_grad: wp.uint8,  # this should be a bool
    batch_size: wp.int32,
    horizon: wp.int32,
    dof: wp.int32,
):
    tid = wp.tid()
    # initialize variables:
    b_id = wp.int32(0)
    h_id = wp.int32(0)
    d_id = wp.int32(0)
    b_addrs = wp.int32(0)
    target_id = wp.int32(0)
    w = wp.float32(0.0)
    c_p = wp.float32(0.0)
    target_p = wp.float32(0.0)
    g_p = wp.float32(0.0)
    r_w = wp.float32(0.0)
    c_total = wp.float32(0.0)

    # we launch batch * horizon * dof kernels
    b_id = tid / (horizon * dof)
    h_id = (tid - (b_id * horizon * dof)) / dof
    d_id = tid - (b_id * horizon * dof + h_id * dof)

    if b_id >= batch_size or h_id >= horizon or d_id >= dof:
        return

    # read weights:
    w = weight[0]
    if h_id < horizon - 1:
        r_w = non_terminal_dof_weight[d_id]
    else:
        r_w = terminal_dof_weight[d_id]

    w = r_w * w
    if w == 0.0:
        return
    # compute cost:
    b_addrs = b_id * horizon * dof + h_id * dof + d_id

    # read buffers:

    c_p = pos[b_addrs]
    target_id = target_idx[b_id]
    target_id = target_id * dof + d_id
    target_p = target[target_id]
    error = c_p - target_p

    c_total = w * error * error
    g_p = 2.0 * w * error

    out_cost[b_addrs] = c_total

    # compute gradient
    if write_grad == 1:
        out_grad_p[b_addrs] = g_p


class L2DistFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pos,
        target,
        target_idx,
        weight,
        terminal_dof_weight,
        non_terminal_dof_weight,
        out_cost_dof,
        out_gp,
        use_grad_input: bool,
    ):
        """Forward pass for the L2 distance cost. This calls a warp kernel.

        Make sure to call init_warp() before using this function.

        Args:
            pos: The current position. Shape is (batch, horizon, dof).
            target: The target position. Shape is (-1, dof). The first dimension can have any size as
            the index is accessed via the `target_idx` tensor. i.e., target for batch element 0 is
            target[target_idx[0]].
            target_idx: The indices of the target position. Shape is (batch). This is int32.
            weight: The weight of the cost. Shape is (1).
            terminal_dof_weight: The weight of the terminal dof. Shape is (dof). This is used for
            the last step of the horizon.
            non_terminal_dof_weight: The weight of the non-terminal dof. Shape is (horizon, dof).
            This is used for all steps except the last step.
            out_cost_dof: The output cost. Shape is (batch, horizon, dof).
            out_gp: The output gradient. Shape is (batch, horizon, dof).
            only_terminal_cost: Whether the cost is only terminal. If this is True, then the cost is
            only computed for the last step of the horizon. It is a torch.uint8 tensor of shape [1].
            use_grad_input: Whether to use the gradient input.

        Returns:
            cost: The cost. Shape is (batch, horizon).
        """
        wp_device, wp_stream = get_warp_device_stream(pos)
        b, h, dof = pos.shape
        requires_grad = pos.requires_grad
        pos_flat = pos.detach().view(-1)
        check_float32_tensors(pos_flat.device, pos_flat=pos_flat)
        wp.launch(
            kernel=forward_l2_warp,
            dim=b * h * dof,
            inputs=[
                wp.from_torch(pos_flat, dtype=wp.float32),
                wp.from_torch(target.view(-1), dtype=wp.float32),
                wp.from_torch(target_idx.view(-1), dtype=wp.int32),
                wp.from_torch(weight, dtype=wp.float32),
                wp.from_torch(terminal_dof_weight, dtype=wp.float32),
                wp.from_torch(non_terminal_dof_weight, dtype=wp.float32),
                wp.from_torch(out_cost_dof.view(-1), dtype=wp.float32),
                wp.from_torch(out_gp.view(-1), dtype=wp.float32),
                requires_grad,
                b,
                h,
                dof,
            ],
            device=wp_device,
            stream=wp_stream,
        )

        cost = torch.sum(out_cost_dof, dim=-1)
        ctx.save_for_backward(out_gp)
        ctx.use_grad_input = use_grad_input
        return cost

    @staticmethod
    def backward(ctx, grad_out_cost):
        (p_grad,) = ctx.saved_tensors

        p_g = None
        if ctx.needs_input_grad[0]:
            if ctx.use_grad_input:
                p_grad = p_grad * grad_out_cost.unsqueeze(-1)
            p_g = p_grad
        return p_g, None, None, None, None, None, None, None, None, None
