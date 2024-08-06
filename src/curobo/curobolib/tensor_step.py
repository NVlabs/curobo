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
from curobo.util.logger import log_warn

try:
    # CuRobo
    from curobo.curobolib import tensor_step_cu

except ImportError:
    # Third Party
    from torch.utils.cpp_extension import load

    # CuRobo
    from curobo.util_file import add_cpp_path

    log_warn("tensor_step_cu not found, jit compiling...")
    tensor_step_cu = load(
        name="tensor_step_cu",
        sources=add_cpp_path(["tensor_step_cuda.cpp", "tensor_step_kernel.cu"]),
    )


def tensor_step_pos_clique_idx_fwd(
    out_position,
    out_velocity,
    out_acceleration,
    out_jerk,
    u_position,
    start_position,
    start_velocity,
    start_acceleration,
    start_idx,
    traj_dt,
    batch_size,
    horizon,
    dof,
    mode=-1,
):
    r = tensor_step_cu.step_idx_position2(
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        u_position,
        start_position,
        start_velocity,
        start_acceleration,
        start_idx,
        traj_dt,
        batch_size,
        horizon,
        dof,
        mode,
    )
    return (r[0], r[1], r[2], r[3])


def tensor_step_pos_clique_fwd(
    out_position,
    out_velocity,
    out_acceleration,
    out_jerk,
    u_position,
    start_position,
    start_velocity,
    start_acceleration,
    traj_dt,
    batch_size,
    horizon,
    dof,
    mode=-1,
):
    r = tensor_step_cu.step_position2(
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        u_position,
        start_position,
        start_velocity,
        start_acceleration,
        traj_dt,
        batch_size,
        horizon,
        dof,
        mode,
    )
    return (r[0], r[1], r[2], r[3])


def tensor_step_acc_fwd(
    out_position,
    out_velocity,
    out_acceleration,
    out_jerk,
    u_acc,
    start_position,
    start_velocity,
    start_acceleration,
    traj_dt,
    batch_size,
    horizon,
    dof,
    use_rk2=True,
):
    r = tensor_step_cu.step_acceleration(
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        u_acc,
        start_position,
        start_velocity,
        start_acceleration,
        traj_dt,
        batch_size,
        horizon,
        dof,
        use_rk2,
    )
    return (r[0], r[1], r[2], r[3])  # output: best_cost, best_q, best_iteration


def tensor_step_acc_idx_fwd(
    out_position,
    out_velocity,
    out_acceleration,
    out_jerk,
    u_acc,
    start_position,
    start_velocity,
    start_acceleration,
    start_idx,
    traj_dt,
    batch_size,
    horizon,
    dof,
    use_rk2=True,
):
    r = tensor_step_cu.step_acceleration_idx(
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        u_acc,
        start_position,
        start_velocity,
        start_acceleration,
        start_idx,
        traj_dt,
        batch_size,
        horizon,
        dof,
        use_rk2,
    )
    return (r[0], r[1], r[2], r[3])  # output: best_cost, best_q, best_iteration


def tensor_step_pos_clique_bwd(
    out_grad_position,
    grad_position,
    grad_velocity,
    grad_acceleration,
    grad_jerk,
    traj_dt,
    batch_size,
    horizon,
    dof,
    mode=-1,
):
    r = tensor_step_cu.step_position_backward2(
        out_grad_position,
        grad_position,
        grad_velocity,
        grad_acceleration,
        grad_jerk,
        traj_dt,
        batch_size,
        horizon,
        dof,
        mode,
    )
    return r[0]
