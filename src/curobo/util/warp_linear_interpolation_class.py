#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Third Party
import torch
import warp as wp

# CuRobo
from curobo.types.robot import JointState
from curobo.util.warp import init_warp
from curobo.util.trajectory import calculate_traj_steps

wp.set_module_options({"fast_math": False})

@wp.kernel
def linear_interpolate_trajectory_kernel(
    raw_position: wp.array(dtype=wp.float32),
    raw_velocity: wp.array(dtype=wp.float32),
    raw_acceleration: wp.array(dtype=wp.float32),
    raw_jerk: wp.array(dtype=wp.float32),
    opt_dt: wp.array(dtype=wp.float32),
    traj_tsteps: wp.array(dtype=wp.int32),
    batch_size: wp.int32,
    raw_horizon: wp.int32,
    dof: wp.int32,
    out_horizon: wp.int32,
    out_position: wp.array(dtype=wp.float32),
    out_velocity: wp.array(dtype=wp.float32),
    out_acceleration: wp.array(dtype=wp.float32),
    out_jerk: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    b_idx = int(0)
    h_idx = int(0)

    oh_idx = int(0)
    d_idx = int(0)
    b_idx = tid / (out_horizon * dof)

    oh_idx = (tid - (b_idx * (out_horizon * dof))) / dof
    d_idx = tid - (b_idx * out_horizon * dof) - (oh_idx * dof)
    if b_idx >= batch_size or oh_idx >= out_horizon or d_idx >= dof:
        return

    nh_idx = int(0)
    weight = float(0)
    n_weight = float(0)
    max_tstep = int(0)
    int_steps = float(0)
    op_dt = float(0)
    op_dt = opt_dt[b_idx]
    max_tstep = traj_tsteps[b_idx]
    int_steps = float((float(max_tstep) / float(raw_horizon - 1)))

    # print(oh_idx)
    h_idx = int(wp.ceil(float(oh_idx) / int_steps))

    if oh_idx >= (max_tstep) or h_idx >= raw_horizon:  # - int(int_steps):
        # write last tstep data:
        h_idx = raw_horizon - 1
        out_position[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = raw_position[
            b_idx * raw_horizon * dof + h_idx * dof + d_idx
        ]
        out_velocity[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
            raw_velocity[b_idx * raw_horizon * dof + h_idx * dof + d_idx]
        )
        out_acceleration[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
            raw_acceleration[b_idx * raw_horizon * dof + h_idx * dof + d_idx] 
        )
        out_jerk[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
            raw_jerk[b_idx * raw_horizon * dof + h_idx * dof + d_idx]
        )
        return
    # we find the current h_idx and interpolate backwards:
    # find the h_idx -1 and h_idx

    # Find current tstep:
    # print(h_idx)

    if h_idx == 0:
        h_idx = 1
    nh_idx = h_idx - 1
    weight = (float(oh_idx) / int_steps) - float(nh_idx)

    n_weight = 1.0 - weight

    # do linear interpolation of position, velocity, acceleration and jerk:
    out_position[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
        weight * raw_position[b_idx * raw_horizon * dof + h_idx * dof + d_idx]
        + n_weight * raw_position[b_idx * raw_horizon * dof + nh_idx * dof + d_idx]
    )
    out_velocity[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
        weight * raw_velocity[b_idx * raw_horizon * dof + h_idx * dof + d_idx]
        + n_weight * raw_velocity[b_idx * raw_horizon * dof + nh_idx * dof + d_idx]
    )
    out_acceleration[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
        (
            weight * raw_acceleration[b_idx * raw_horizon * dof + h_idx * dof + d_idx]
            + n_weight * raw_acceleration[b_idx * raw_horizon * dof + nh_idx * dof + d_idx]
        )

    )
    out_jerk[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
        (
            weight * raw_jerk[b_idx * raw_horizon * dof + h_idx * dof + d_idx]
            + n_weight * raw_jerk[b_idx * raw_horizon * dof + nh_idx * dof + d_idx]
        )

    )



class LinearInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                raw_pos: torch.Tensor,
                raw_vel: torch.Tensor,
                raw_acc: torch.Tensor,
                raw_jerk: torch.Tensor,
                # traj_tsteps: torch.Tensor,
                # out_traj: JointState,
                opt_dt: torch.Tensor,
                interp_dt: float,
                ):
        
        # Store for backward computation
        # ctx.save_for_backward(raw_traj, opt_dt, traj_tsteps)

    
        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        init_warp()
        batch, horizon, dof = raw_pos.shape
        # horizon = raw_pos.shape[1]
        traj_tsteps, steps_max = calculate_traj_steps(opt_dt, interp_dt, horizon)
        steps_max = steps_max.to(dtype=torch.int32).item()

        # inputs
        ctx.raw_pos =  wp.from_torch(raw_pos.view(-1), requires_grad=True)
        ctx.raw_vel =  wp.from_torch(raw_vel.view(-1), requires_grad=True)
        ctx.raw_acc =  wp.from_torch(raw_acc.view(-1))                    
        ctx.raw_jerk = wp.from_torch(raw_jerk.view(-1))                   
        ctx.opt_dt = wp.from_torch(opt_dt.view(-1))
        ctx.traj_tsteps = wp.from_torch(traj_tsteps.clone().view(-1).to(dtype=torch.int32), dtype=wp.int32)
        ctx.batch = batch
        ctx.horizon = horizon
        ctx.dof = dof
        ctx.steps_max = steps_max

        # allocate output

        ctx.out_pos = wp.zeros(batch * steps_max * dof, requires_grad=True)  # wp.from_torch(out_traj.position.view(-1), requires_grad=True)
        ctx.out_vel = wp.zeros(batch * steps_max * dof, requires_grad=True)  # wp.from_torch(out_traj.velocity.view(-1))
        ctx.out_acc = wp.zeros(batch * steps_max * dof, requires_grad=True)  # wp.from_torch(out_traj.acceleration.view(-1))
        ctx.out_jerk =wp.zeros(batch * steps_max * dof, requires_grad=True)  # wp.from_torch(out_traj.jerk.view(-1))

        assert ctx.traj_tsteps.dtype == wp.int32, "Incorrect type"
        wp.launch(
            kernel=linear_interpolate_trajectory_kernel,
            dim=batch * steps_max * dof, # number of threads
            inputs=[
                ctx.raw_pos,
                ctx.raw_vel,
                ctx.raw_acc,
                ctx.raw_jerk,
                ctx.opt_dt,
                ctx.traj_tsteps,
                ctx.batch,
                ctx.horizon,
                ctx.dof,
                ctx.steps_max,
            ],
            outputs=[
                ctx.out_pos,
                ctx.out_vel,
                ctx.out_acc,
                ctx.out_jerk,
            ],
            stream=wp.stream_from_torch(raw_pos.device),
        )

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()
        return wp.to_torch(ctx.out_pos).view(batch, steps_max, dof), wp.to_torch(ctx.out_vel).view(batch, steps_max, dof), wp.to_torch(ctx.out_acc).view(batch, steps_max, dof), wp.to_torch(ctx.out_jerk).view(batch, steps_max, dof)
    
    @staticmethod
    def backward(ctx, grad_out_pos, grad_out_vel, grad_out_acc, grad_out_jerk):
        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        # init_warp()

        # map incoming Torch grads to our output variables
        ctx.out_pos.grad = wp.from_torch(grad_out_pos.view(-1))
        ctx.out_vel.grad = wp.from_torch(grad_out_vel.view(-1))  
        ctx.out_acc.grad = wp.from_torch(grad_out_acc.view(-1))  
        ctx.out_jerk.grad = wp.from_torch(grad_out_jerk.view(-1))
        assert ctx.traj_tsteps.dtype == wp.int32, "Incorrect type"

        wp.launch(
            kernel=linear_interpolate_trajectory_kernel,
            dim=ctx.batch * ctx.steps_max * ctx.dof,
            inputs=[
                ctx.raw_pos,
                ctx.raw_vel,
                ctx.raw_acc,
                ctx.raw_jerk,
                ctx.opt_dt,
                ctx.traj_tsteps,
                ctx.batch,
                ctx.horizon,
                ctx.dof,
                ctx.steps_max,
            ],
            outputs=[
                ctx.out_pos,
                ctx.out_vel,
                ctx.out_acc,
                ctx.out_jerk,
            ],
            # When an adjoint input dos not have gradient, it must be Noneif it's a warp array and 0 if primitive type
            adj_inputs=[ctx.raw_pos.grad, ctx.raw_vel.grad, None, None, None, None, wp.int32(0), wp.int32(0), wp.int32(0), wp.int32(0)],
            adj_outputs=[ctx.out_pos.grad, ctx.out_vel.grad, ctx.out_acc.grad, ctx.out_jerk.grad],
            adjoint=True,
            stream=wp.stream_from_torch(ctx.raw_pos.device),
        )

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        return (wp.to_torch(ctx.raw_pos.grad).view(ctx.batch, ctx.horizon, ctx.dof),
                wp.to_torch(ctx.raw_vel.grad).view(ctx.batch, ctx.horizon, ctx.dof)
                , None, None, None, None, None)
