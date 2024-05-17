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
def cubic_interpolate_trajectory_kernel(
    raw_position: wp.array(dtype=wp.float32), # input position
    raw_velocity: wp.array(dtype=wp.float32), # input velocity
    raw_acceleration: wp.array(dtype=wp.float32), # input acceleration
    raw_jerk: wp.array(dtype=wp.float32), # input jerk
    opt_dt: wp.array(dtype=wp.float32), # optimized dt after retiming (different for every traj in batch) -  used for scaling vel/acc/jerk and for calculating delta_t in interpolation
    traj_tsteps: wp.array(dtype=wp.int32), # number of knots in interpolated (output) trajectory  = (horizon-1) * opt_dt / interpolation_dt + 1 (less than or equal to steps max). Note this is an array!
    batch_size: wp.int32, # batch size
    raw_horizon: wp.int32, # number of knots in input trajectory
    dof: wp.int32, # degrees of freedom
    out_horizon: wp.int32, # This is the steps_max (ie size of interpolation buffer). If a single trajectory, then this is equal to traj_tsteps. Note this is an integer value
    raw_dt: wp.float32, # original dt (solver dt)
    out_position: wp.array(dtype=wp.float32), # output position
    out_velocity: wp.array(dtype=wp.float32), # output velocity
    out_acceleration: wp.array(dtype=wp.float32), # output acceleration
    out_jerk: wp.array(dtype=wp.float32), # output jerk
):
    tid = wp.tid() # current thread idx

    b_idx = int(0)
    h_idx = int(0)

    oh_idx = int(0)
    d_idx = int(0)
    b_idx = tid / (out_horizon * dof) # batch idx

    oh_idx = (tid - (b_idx * (out_horizon * dof))) / dof
    d_idx = tid - (b_idx * out_horizon * dof) - (oh_idx * dof)
    if b_idx >= batch_size or oh_idx >= out_horizon or d_idx >= dof:
        return

    nh_idx = int(0)
    out_idx = int(0)
    curr_idx = int(0)
    prev_idx = int(0)
    max_tstep = int(0)
    int_steps = float(0)
    op_dt = float(0)
    op_dt = opt_dt[b_idx] 
    max_tstep = traj_tsteps[b_idx]
    int_steps = float((float(max_tstep) / float(raw_horizon - 1)))
    scale = float(1.0)
    scale = raw_dt / op_dt # scale 
    # scale = 1.0 #scale * int_steps * (0.01) # Bug is here
    # print(oh_idx)
    h_idx = int(wp.ceil(float(oh_idx) / int_steps))


    out_idx = b_idx * out_horizon * dof + oh_idx * dof + d_idx
    curr_idx = b_idx * raw_horizon * dof + h_idx * dof + d_idx
    if oh_idx >= (max_tstep) or h_idx >= raw_horizon:  # - int(int_steps):
        # write last tstep data:
        h_idx = raw_horizon - 1
        out_position[out_idx] = raw_position[curr_idx]
        out_velocity[out_idx] = (
            raw_velocity[curr_idx] * scale
        )
        out_acceleration[out_idx] = (
            raw_acceleration[curr_idx] * scale * scale
        )
        out_jerk[out_idx] = (
            raw_jerk[curr_idx] * scale * scale * scale
        )
        return
    # we find the current h_idx and interpolate backwards:
    # find the h_idx -1 and h_idx

    # Find current tstep:
    # print(h_idx)

    if h_idx == 0:
        h_idx = 1
    nh_idx = h_idx - 1

    prev_idx = b_idx * raw_horizon * dof + nh_idx * dof + d_idx

    c0 = float(0)
    c1 = float(0)
    c2 = float(0)
    c3 = float(0)
    # c0 = (dydx0 + dydx1 - 2 * (y1 - y0) / dx) / dx**2
    c0 = (raw_velocity[prev_idx] * scale + raw_velocity[curr_idx] * scale - 2.0 * (raw_position[curr_idx] - raw_position[prev_idx]) / op_dt) / op_dt**2.0
    # c1 = (3 * (y1 - y0) / dx - 2 * dydx0 - dydx1) / dx
    c1 = (3.0 * (raw_position[curr_idx] - raw_position[prev_idx]) / op_dt - 2.0 * raw_velocity[prev_idx] * scale - raw_velocity[curr_idx] * scale) / op_dt
    # c2 = dydx0
    c2 = raw_velocity[prev_idx] * scale
    # c3 = y0
    c3 = raw_position[prev_idx]

    dt = float(0)
    dt = ((float(oh_idx) / int_steps) - float(nh_idx)) * op_dt

    # do cubic interpolation of position, velocity, acceleration and jerk:
    # out_position = ((c0 * dt + c1) * dt + c2) * dt + c3
    out_position[out_idx] = (
        ((c0 * dt + c1) * dt + c2) * dt + c3
    )
    # out_velocity = (3 * c0 * dt + 2 * c1) * dt + c2
    out_velocity[out_idx] = (
        (3.0 * c0 * dt + 2.0 * c1) * dt + c2
    ) #* scale
    # out_acceleration = 6 * c0 * dt + 2 * c1
    out_acceleration[out_idx] = (
        (
            6.0 * c0 * dt + 2.0 * c1
        )
        # * scale
        # * scale
    )
    # out_jerk = 6 * c0
    out_jerk[out_idx] = (
        (
            6.0 * c0
        )
        # * scale
        # * scale
        # * scale
    )


class CubicInterpolation(torch.autograd.Function):
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
                raw_dt: float = 0.5,
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
        ctx.raw_dt = raw_dt

        # allocate output

        ctx.out_pos = wp.zeros(batch * steps_max * dof, requires_grad=True)  # wp.from_torch(out_traj.position.view(-1), requires_grad=True)
        ctx.out_vel = wp.zeros(batch * steps_max * dof, requires_grad=True)  # wp.from_torch(out_traj.velocity.view(-1))
        ctx.out_acc = wp.zeros(batch * steps_max * dof, requires_grad=True)  # wp.from_torch(out_traj.acceleration.view(-1))
        ctx.out_jerk =wp.zeros(batch * steps_max * dof, requires_grad=True)  # wp.from_torch(out_traj.jerk.view(-1))

        assert ctx.traj_tsteps.dtype == wp.int32, "Incorrect type"
        wp.launch(
            kernel=cubic_interpolate_trajectory_kernel,
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
                ctx.raw_dt,
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
            kernel=cubic_interpolate_trajectory_kernel,
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
                ctx.raw_dt,
            ],
            outputs=[
                ctx.out_pos,
                ctx.out_vel,
                ctx.out_acc,
                ctx.out_jerk,
            ],
            # When an adjoint input dos not have gradient, it must be Noneif it's a warp array and 0 if primitive type
            adj_inputs=[ctx.raw_pos.grad, ctx.raw_vel.grad, None, None, None, None, wp.int32(0), wp.int32(0), wp.int32(0), wp.int32(0), wp.float32(0.0)],
            adj_outputs=[ctx.out_pos.grad, ctx.out_vel.grad, ctx.out_acc.grad, ctx.out_jerk.grad],
            adjoint=True,
            stream=wp.stream_from_torch(ctx.raw_pos.device),
        )

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        return (wp.to_torch(ctx.raw_pos.grad).view(ctx.batch, ctx.horizon, ctx.dof),
                wp.to_torch(ctx.raw_vel.grad).view(ctx.batch, ctx.horizon, ctx.dof)
                , None, None, None, None, None)
