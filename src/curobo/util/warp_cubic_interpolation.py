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

wp.set_module_options({"fast_math": False})


@wp.kernel
def cubic_interpolate_trajectory_kernel(
    raw_position: wp.array(dtype=wp.float32),
    raw_velocity: wp.array(dtype=wp.float32),
    raw_acceleration: wp.array(dtype=wp.float32),
    raw_jerk: wp.array(dtype=wp.float32),
    out_position: wp.array(dtype=wp.float32),
    out_velocity: wp.array(dtype=wp.float32),
    out_acceleration: wp.array(dtype=wp.float32),
    out_jerk: wp.array(dtype=wp.float32),
    opt_dt: wp.array(dtype=wp.float32),
    traj_tsteps: wp.array(dtype=wp.int32),
    batch_size: wp.int32,
    raw_horizon: wp.int32,
    dof: wp.int32,
    out_horizon: wp.int32,
    raw_dt: wp.float32, # solver dt
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
    op_dt = opt_dt[b_idx] # optimized dt after retiming (different for every traj in batch)
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


def get_cuda_cubic_interpolation(
    raw_traj: JointState,
    traj_tsteps: torch.Tensor,
    out_traj: JointState,
    opt_dt: torch.Tensor,
    raw_dt: float = 0.5,
):
    """Use warp to perform linear interpolation on GPU for a batch of trajectories.

    #NOTE: There is a bug in the indexing which makes the last horizon step in the trajectory to be
    not missed. This will not affect solutions solved by our arm_base class as we make last 3
    timesteps the same.

    Args:
        raw_traj (JointState): _description_
        traj_tsteps (torch.Tensor): _description_
        out_traj (JointState): _description_
        opt_dt (torch.Tensor): _description_
        raw_dt (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    init_warp()
    batch, int_horizon, dof = out_traj.position.shape
    horizon = raw_traj.position.shape[1]

    wp.launch(
        kernel=cubic_interpolate_trajectory_kernel,
        dim=batch * int_horizon * dof,
        inputs=[
            wp.from_torch(raw_traj.position.view(-1)),
            wp.from_torch(raw_traj.velocity.view(-1)),
            wp.from_torch(raw_traj.acceleration.view(-1)),
            wp.from_torch(raw_traj.jerk.view(-1)),
            wp.from_torch(out_traj.position.view(-1)),
            wp.from_torch(out_traj.velocity.view(-1)),
            wp.from_torch(out_traj.acceleration.view(-1)),
            wp.from_torch(out_traj.jerk.view(-1)),
            wp.from_torch(opt_dt.view(-1)),
            wp.from_torch(traj_tsteps.view(-1)),
            batch,
            horizon,
            dof,
            int_horizon,
            raw_dt,
        ],
        stream=wp.stream_from_torch(raw_traj.position.device),
    )
    return out_traj
