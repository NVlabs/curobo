# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import torch

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.trajectory import TrajInterpolationType, get_batch_interpolated_trajectory


def test_linear_interpolation():
    device_cfg = DeviceCfg()

    b, h, dof = 1, 12, 1
    raw_dt = device_cfg.to_device([0.4])
    int_dt = device_cfg.to_device([0.1])
    # initialize raw trajectory:
    in_traj = JointState.zeros((b, h, dof), device_cfg)
    in_traj.position = torch.zeros((b, h, dof), device=device_cfg.device)
    # in_traj.position[:, 1, :] = 0.1

    in_traj.position[:, -2:, :] = 1.0
    # in_traj.position[:, 10, :] = -0.01

    in_traj.position[:, -1, :] = 0.01

    in_traj.dt = raw_dt

    # create max_velocity buffer:
    out_traj_gpu, _ = get_batch_interpolated_trajectory(
        raw_traj=in_traj,
        interpolation_dt=int_dt,
        kind=TrajInterpolationType.LINEAR_CUDA,
    )
    #
    out_traj_gpu = out_traj_gpu.clone()

    out_traj_cpu, _ = get_batch_interpolated_trajectory(
        raw_traj=in_traj,
        interpolation_dt=int_dt,
        kind=TrajInterpolationType.LINEAR,
    )

    assert (
        torch.max(
            torch.abs(out_traj_gpu.position[:, -5:, :] - out_traj_cpu.position[:, -5:, :])
        ).item()
        < 0.001
    )
