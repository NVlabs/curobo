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
def linear_interpolate_trajectory_kernel(
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
    raw_dt: wp.float32,
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
    scale = float(1.0)
    scale = raw_dt / op_dt
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


def get_cuda_linear_interpolation(
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
        kernel=linear_interpolate_trajectory_kernel,
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
