# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standard Library
from enum import Enum
from typing import List, Optional, Tuple

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler
from scipy import interpolate

# CuRobo
from curobo._src.curobolib.cuda_ops.trajectory import get_bspline_interpolation
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.torch_util import get_torch_jit_decorator
from curobo._src.util.warp_interpolation import get_cuda_linear_interpolation


class TrajInterpolationType(Enum):
    #: linear interpolation using scipy
    LINEAR = "linear"
    #: cubic interpolation using scipy
    CUBIC = "cubic"
    #: quintic interpolation using scipy
    QUARTIC = "quartic"
    #: quintic interpolation using scipy
    QUINTIC = "quintic"
    #: cuda accelerated linear interpolation
    LINEAR_CUDA = "linear_cuda"
    #: cuda accelerated bspline interpolation. This requires knots to be given.
    BSPLINE_KNOTS_CUDA = "bspline_knots_cuda"


def get_batch_interpolated_trajectory(
    raw_traj: JointState,
    interpolation_dt: torch.Tensor,
    kind: TrajInterpolationType = TrajInterpolationType.LINEAR_CUDA,
    out_traj_state: Optional[JointState] = None,
    device_cfg: DeviceCfg = DeviceCfg(),
    current_state: Optional[JointState] = None,
    goal_state: Optional[JointState] = None,
    start_idx: Optional[torch.Tensor] = None,
    goal_idx: Optional[torch.Tensor] = None,
    use_implicit_goal_state: Optional[torch.Tensor] = None,
):
    if len(raw_traj.shape) == 2:
        raw_traj = raw_traj.unsqueeze(0)
    nearest_int = False
    if kind == TrajInterpolationType.BSPLINE_KNOTS_CUDA:
        nearest_int = True
        raw_dt = raw_traj.knot_dt
        b, horizon, dof = raw_traj.knot.shape
        if raw_traj.control_space is None:
            log_and_raise("control_space is required")
        horizon = ControlSpace.spline_total_knots(raw_traj.control_space, horizon) + 1

    else:
        raw_dt = raw_traj.dt
        b, horizon, dof = raw_traj.shape

    traj_steps, steps_max = calculate_traj_steps(
        raw_dt, interpolation_dt, int(horizon), nearest_int=nearest_int
    )

    # traj_steps contains the tsteps for each trajectory
    if steps_max <= 1:
        log_and_raise(
            f"Steps max is <= 1: {steps_max}, {raw_dt}, {interpolation_dt}, {horizon},"
            + f"{nearest_int}"
        )

    if out_traj_state is not None:
        if len(out_traj_state.shape) == 2:
            out_traj_state = out_traj_state.unsqueeze(0)

        if out_traj_state.position.shape[1] < steps_max:
            log_warn(
                "Interpolation buffer shape is smaller than steps_max: "
                + str(out_traj_state.position.shape)
                + " creating new buffer of shape "
                + str(steps_max)
            )
            out_traj_state = None

    if out_traj_state is None:
        if steps_max > 10000:
            log_and_raise(
                f"steps_max is too large: {steps_max}"
                + f"raw_dt: {raw_dt}"
                + f"interpolation_dt: {interpolation_dt}"
                + f"horizon: {horizon}"
                + f"nearest_int: {nearest_int}"
            )
        out_traj_state = JointState.zeros(
            [b, steps_max, dof], device_cfg, joint_names=raw_traj.joint_names
        )
    if kind in [
        TrajInterpolationType.LINEAR,
        TrajInterpolationType.CUBIC,
        TrajInterpolationType.QUARTIC,
        TrajInterpolationType.QUINTIC,
    ]:
        # plot and save:
        out_traj_state = get_cpu_linear_interpolation(
            raw_traj,
            traj_steps,
            out_traj_state,
            kind,
            interpolation_dt=interpolation_dt,
        )

    elif kind == TrajInterpolationType.LINEAR_CUDA:
        out_traj_state = get_cuda_linear_interpolation(
            raw_traj,
            traj_steps,
            out_traj_state,
        )
    elif kind == TrajInterpolationType.BSPLINE_KNOTS_CUDA:
        bspline_degree = ControlSpace.spline_degree(raw_traj.control_space)

        out_traj_state = get_bspline_interpolation(
            raw_traj,
            out_traj_state,
            interpolation_dt,
            current_state=current_state,
            goal_state=goal_state,
            start_idx=start_idx,
            goal_idx=goal_idx,
            use_implicit_goal_state=use_implicit_goal_state,
            bspline_degree=bspline_degree,
            interpolated_horizon=traj_steps,
        )

    else:
        log_and_raise("Unknown interpolation type")
    return out_traj_state, traj_steps


def get_cpu_linear_interpolation(
    raw_traj, traj_steps, out_traj_state, kind: TrajInterpolationType, interpolation_dt=None
):
    cpu_traj = raw_traj.position.cpu().numpy()
    out_traj = out_traj_state.position
    retimed_traj = out_traj.cpu()
    int_dt = interpolation_dt.item()
    for k in range(out_traj.shape[0]):
        tstep = traj_steps[k].item()
        opt_d = raw_traj.dt[k].item()
        for i in range(cpu_traj.shape[-1]):
            retimed_traj[k, :tstep, i] = linear_smooth(
                cpu_traj[k, :, i],
                y=None,
                n=tstep,
                kind=kind,
                last_step=tstep,
                opt_dt=opt_d,
                interpolation_dt=int_dt,
            )
        retimed_traj[k, tstep:, :] = retimed_traj[k, tstep - 1 : tstep, :]
        # print(cpu_traj[k, :, i])
        # print(retimed_traj[k, :tstep, i])

    out_traj_state.position[:] = retimed_traj.to(device=raw_traj.position.device)
    return out_traj_state


@profiler.record_function("interpolation/1D")
def linear_smooth(
    x: np.array,
    y=None,
    n=10,
    kind=TrajInterpolationType.CUBIC,
    last_step=None,
    opt_dt=None,
    interpolation_dt=None,
):
    if last_step is None:
        last_step = n  # min(x.shape[0],n)
    extra_dt = 1.0
    if opt_dt is not None:
        y = np.ravel([i * opt_dt for i in range(x.shape[0])])
        extra_dt = opt_dt
    if y is None:
        step = float(last_step - 1) / float(x.shape[0] - 1)
        y = np.ravel([float(i) * step for i in range(x.shape[0])])

    if kind == TrajInterpolationType.QUINTIC:
        if len(y) < 6:
            # linearly interpolate between waypoints to get 6 values
            y = np.concatenate(
                (y, np.ravel([y[-1] + (i + 1) * extra_dt for i in range(6 - len(y))]))
            )
            x = np.concatenate((x, np.ravel([x[-1] for _ in range(6 - len(x))])))

        f = interpolate.make_interp_spline(y, x, k=5)
    elif kind == TrajInterpolationType.QUARTIC:
        if len(y) < 5:
            # linearly interpolate between waypoints to get 5 values
            y = np.concatenate(
                (y, np.ravel([y[-1] + (i + 1) * extra_dt for i in range(5 - len(y))]))
            )
            x = np.concatenate((x, np.ravel([x[-1] for _ in range(5 - len(x))])))
        log_and_raise("Not Implemented yet")
    else:
        if kind == TrajInterpolationType.CUBIC:
            if len(y) < 4:
                y = np.concatenate(
                    (y, np.ravel([y[-1] + (i + 1) * extra_dt for i in range(4 - len(y))]))
                )
                x = np.concatenate((x, np.ravel([x[-1] for _ in range(4 - len(x))])))

        f = interpolate.interp1d(
            x=y,
            y=x,
            kind=kind.value,
            assume_sorted=True,
            fill_value=np.ravel([x[-1]]),
            bounds_error=False,
        )
    if opt_dt is None:
        y_new = np.ravel([i for i in range(last_step)])
    else:
        y_new = np.ravel([i * interpolation_dt for i in range(last_step)])
    ynew = f(y_new)
    y_new = torch.as_tensor(ynew)
    return y_new


@get_torch_jit_decorator(force_jit=False)
def calculate_dt_no_clamp(
    vel: torch.Tensor,
    acc: torch.Tensor,
    jerk: torch.Tensor,
    max_vel: torch.Tensor,
    max_acc: torch.Tensor,
    max_jerk: torch.Tensor,
    epsilon: float = 1e-5,
):
    # compute scaled dt:
    max_v_arr = torch.max(torch.abs(vel), dim=-2)[0]  # output is batch, dof

    max_acc_arr = torch.max(torch.abs(acc), dim=-2)[0]
    max_jerk_arr = torch.max(torch.abs(jerk), dim=-2)[0]
    vel_scale_dt = (max_v_arr) / (max_vel.view(1, max_v_arr.shape[-1]))  # batch,dof
    acc_scale_dt = (max_acc_arr) / (max_acc.view(1, max_acc_arr.shape[-1]))
    jerk_scale_dt = (max_jerk_arr) / (max_jerk.view(1, max_jerk_arr.shape[-1]))
    dt_score_vel = torch.max(vel_scale_dt, dim=-1)[0]  # batch, 1
    dt_score_acc = torch.pow((torch.max(acc_scale_dt, dim=-1)[0]), 1.0 / 2.0)
    dt_score_jerk = torch.pow((torch.max(jerk_scale_dt, dim=-1)[0]), 1.0 / 3.0)
    dt_score_0 = torch.maximum(dt_score_vel, dt_score_acc)
    dt_score_1 = torch.maximum(dt_score_0, dt_score_jerk)
    dt_score_new = dt_score_1 * (1.0 + epsilon)
    return dt_score_new


@get_torch_jit_decorator()
def calculate_traj_steps(
    opt_dt: torch.Tensor,
    interpolation_dt: torch.Tensor,
    horizon: int,
    nearest_int: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if nearest_int:
        interpolation_steps_per_waypoint = (opt_dt + interpolation_dt) / interpolation_dt
    else:
        interpolation_steps_per_waypoint = (opt_dt + opt_dt % interpolation_dt) / interpolation_dt
    interpolation_steps = interpolation_steps_per_waypoint.to(dtype=torch.int64)

    traj_steps = ((horizon - 1) * (interpolation_steps)).to(dtype=torch.int64)
    steps_max = torch.max(traj_steps)
    traj_steps += 1
    steps_max += 1
    traj_steps = traj_steps.to(dtype=torch.int32)
    steps_max = steps_max.to(dtype=torch.int32)
    return traj_steps, steps_max


def get_interpolated_trajectory(
    trajectory: List[torch.Tensor],
    out_traj_state: JointState,
    des_horizon: Optional[int] = None,
    interpolation_dt: float = 0.02,
    kind=TrajInterpolationType.CUBIC,
    device_cfg: DeviceCfg = DeviceCfg(),
    max_joint_velocity: Optional[torch.Tensor] = None,
) -> JointState:
    dof = trajectory[0].shape[-1]
    last_tsteps = []
    opt_dt = []
    if des_horizon is None:
        interpolation_steps = out_traj_state.position.shape[1]
    else:
        interpolation_steps = des_horizon
    y = None
    for b in range(len(trajectory)):
        raw_traj = trajectory[b].cpu().view(-1, dof).numpy()
        current_kind = kind

        if current_kind in [
            TrajInterpolationType.LINEAR,
            TrajInterpolationType.CUBIC,
            TrajInterpolationType.QUARTIC,
            TrajInterpolationType.QUINTIC,
        ]:
            retimed_traj = torch.zeros((interpolation_steps, raw_traj.shape[-1]))
            if raw_traj.shape[0] < 5:
                current_kind = TrajInterpolationType.LINEAR
            if max_joint_velocity is not None:
                v_diff = np.linalg.norm(np.diff(raw_traj, axis=0), axis=-1)
                v_total = np.cumsum(v_diff)
                step = float(des_horizon - 1) / float((raw_traj.shape[0]) - 1)
                v_total = (v_total / v_total[-1]) * step * (raw_traj.shape[0] - 1)
                y = [0] + v_total.tolist()

            for i in range(raw_traj.shape[-1]):  # interpolate per joint
                retimed_traj[:, i] = linear_smooth(
                    raw_traj[:, i],
                    y=y,
                    n=interpolation_steps,
                    kind=kind,
                    last_step=des_horizon,
                )
            retimed_traj = retimed_traj.to(**(device_cfg.as_torch_dict()))
            out_traj_state.position[b, :interpolation_steps, :] = retimed_traj
            out_traj_state.position[b, interpolation_steps:, :] = retimed_traj[
                interpolation_steps - 1 : interpolation_steps, :
            ]
            last_tsteps.append(interpolation_steps)
            opt_dt.append(interpolation_dt)
        else:
            log_and_raise("Unsupported interpolation type: {}".format(kind))
    opt_dt = device_cfg.to_device(opt_dt)
    return out_traj_state, last_tsteps, opt_dt
