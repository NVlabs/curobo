# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.util.warp import get_warp_device_stream, init_warp

wp.set_module_options({"fast_math": False})


@wp.kernel
def linear_interpolate_batch_dt_trajectory_kernel(
    raw_position: wp.array(dtype=wp.float32),
    raw_velocity: wp.array(dtype=wp.float32),
    raw_acceleration: wp.array(dtype=wp.float32),
    raw_jerk: wp.array(dtype=wp.float32),
    raw_dt: wp.array(dtype=wp.float32),
    out_position: wp.array(dtype=wp.float32),
    out_velocity: wp.array(dtype=wp.float32),
    out_acceleration: wp.array(dtype=wp.float32),
    out_jerk: wp.array(dtype=wp.float32),
    out_dt: wp.array(dtype=wp.float32),
    traj_tsteps: wp.array(dtype=wp.int32),
    batch_size: wp.int32,
    raw_horizon: wp.int32,
    dof: wp.int32,
    out_horizon: wp.int32,
    # raw_dt_batch: wp.array(dtype=wp.float32),
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
    op_dt = raw_dt[b_idx]
    max_tstep = traj_tsteps[b_idx]
    int_steps = float((float(max_tstep - 1) / float(raw_horizon - 1)))

    scale = float(1.0)
    # raw_dt = op_dt #float(1.0) #raw_dt_batch[b_idx]
    # scale = raw_dt / op_dt
    # scale = 1.0 #scale * int_steps * (0.01) # Bug is here
    # print(oh_idx)
    h_idx = int(wp.ceil(float(oh_idx) / int_steps))

    if oh_idx >= (max_tstep) or h_idx >= raw_horizon:  # - int(int_steps):
        # write last tstep data:
        h_idx = raw_horizon - 1
        out_position[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = raw_position[
            b_idx * raw_horizon * dof + h_idx * dof + d_idx
        ]
        out_velocity[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
            raw_velocity[b_idx * raw_horizon * dof + h_idx * dof + d_idx] * scale
        )
        out_acceleration[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
            raw_acceleration[b_idx * raw_horizon * dof + h_idx * dof + d_idx] * scale * scale
        )
        out_jerk[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
            raw_jerk[b_idx * raw_horizon * dof + h_idx * dof + d_idx] * scale * scale * scale
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
    ) * scale
    out_acceleration[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
        (
            weight * raw_acceleration[b_idx * raw_horizon * dof + h_idx * dof + d_idx]
            + n_weight * raw_acceleration[b_idx * raw_horizon * dof + nh_idx * dof + d_idx]
        )
        * scale
        * scale
    )
    out_jerk[b_idx * out_horizon * dof + oh_idx * dof + d_idx] = (
        (
            weight * raw_jerk[b_idx * raw_horizon * dof + h_idx * dof + d_idx]
            + n_weight * raw_jerk[b_idx * raw_horizon * dof + nh_idx * dof + d_idx]
        )
        * scale
        * scale
        * scale
    )
    if oh_idx == 0 and d_idx == 0:
        out_dt[b_idx] = op_dt / int_steps


def get_cuda_linear_interpolation(
    raw_traj: JointState,
    traj_tsteps: torch.Tensor,
    out_traj: JointState,
):
    """Use warp to perform linear interpolation on GPU for a batch of trajectories.

    #NOTE: There is a bug in the indexing which makes the last horizon step in the trajectory to be
    not missed. This will not affect solutions solved by our robot rollout as we make last 3
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
        kernel=linear_interpolate_batch_dt_trajectory_kernel,
        dim=batch * int_horizon * dof,
        inputs=[
            wp.from_torch(raw_traj.position.view(-1)),
            wp.from_torch(raw_traj.velocity.view(-1)),
            wp.from_torch(raw_traj.acceleration.view(-1)),
            wp.from_torch(raw_traj.jerk.view(-1)),
            wp.from_torch(raw_traj.dt.view(-1)),
            wp.from_torch(out_traj.position.view(-1)),
            wp.from_torch(out_traj.velocity.view(-1)),
            wp.from_torch(out_traj.acceleration.view(-1)),
            wp.from_torch(out_traj.jerk.view(-1)),
            wp.from_torch(out_traj.dt.view(-1)),
            wp.from_torch(traj_tsteps.view(-1)),
            batch,
            horizon,
            dof,
            int_horizon,
            # wp.from_torch(raw_dt.view(-1)),
        ],
        stream=get_warp_device_stream(raw_traj.position)[1],
    )
    return out_traj
