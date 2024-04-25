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

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.util.trajectory import InterpolateType, get_batch_interpolated_trajectory


def test_linear_interpolation():
    tensor_args = TensorDeviceType()

    b, h, dof = 1, 24, 1
    raw_dt = tensor_args.to_device(0.4)
    int_dt = 0.01
    # initialize raw trajectory:
    in_traj = JointState.zeros((b, h, dof), tensor_args)
    in_traj.position = torch.zeros((b, h, dof), device=tensor_args.device)
    in_traj.position[:, 1, :] = 0.1

    in_traj.position[:, -2, :] = -0.01
    in_traj.position[:, 10, :] = -0.01

    in_traj.position[:, -1, :] = 0.01
    in_traj.velocity = in_traj.position - torch.roll(in_traj.position, -1, dims=1)
    in_traj.velocity[:, 0, :] = 0.0
    in_traj.velocity[:, -1, :] = 0.0

    max_vel = torch.ones((1, 1, dof), device=tensor_args.device, dtype=tensor_args.dtype)
    max_acc = torch.ones((1, 1, dof), device=tensor_args.device, dtype=tensor_args.dtype) * 25
    max_jerk = torch.ones((1, 1, dof), device=tensor_args.device, dtype=tensor_args.dtype) * 500

    # create max_velocity buffer:
    out_traj_gpu, _, _ = get_batch_interpolated_trajectory(
        in_traj,
        raw_dt,
        int_dt,
        max_vel,
        max_acc=max_acc,
        max_jerk=max_jerk,
    )
    #
    out_traj_gpu = out_traj_gpu.clone()

    out_traj_cpu, _, _ = get_batch_interpolated_trajectory(
        in_traj,
        raw_dt,
        int_dt,
        max_vel,
        kind=InterpolateType.LINEAR,
        max_acc=max_acc,
        max_jerk=max_jerk,
    )
    assert (
        torch.max(
            torch.abs(out_traj_gpu.position[:, -5:, :] - out_traj_cpu.position[:, -5:, :])
        ).item()
        < 0.05
    )
