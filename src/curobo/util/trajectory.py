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
# Standard Library
import math
from enum import Enum
from typing import List, Optional, Tuple

# Third Party
import numpy as np
import torch

# SRL
import torch.autograd.profiler as profiler

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.sample_lib import bspline
from curobo.util.torch_utils import get_torch_jit_decorator
from curobo.util.warp_interpolation import get_cuda_linear_interpolation


class InterpolateType(Enum):
    #: linear interpolation using scipy
    LINEAR = "linear"
    #: cubic interpolation using scipy
    CUBIC = "cubic"
    #: quintic interpolation using scipy
    QUINTIC = "quintic"
    #: cuda accelerated linear interpolation using warp-lang
    #: custom kernel :meth: get_cuda_linear_interpolation
    LINEAR_CUDA = "linear_cuda"
    #: Uses "Time-optimal trajectory generation for path following with bounded acceleration
    #: and velocity." Robotics: Science and Systems VIII (2012): 1-8, Kunz & Stillman.
    KUNZ_STILMAN_OPTIMAL = "kunz_stilman_optimal"


def get_linear_traj(
    positions,
    dt=0.5,
    duration=20,
    tensor_args={"device": "cpu", "dtype": torch.float32},
    max_traj_pts=None,
    compute_dynamics=True,
):
    with profiler.record_function("linear_traj"):
        if max_traj_pts is not None:
            duration = max_traj_pts * dt
            # max_pts = max_traj_pts

        p_arr = torch.as_tensor(positions)
        # create path:
        path = torch.zeros((p_arr.shape[0] - 1, 2, p_arr.shape[1]))
        for i in range(p_arr.shape[0] - 1):
            path[i, 0] = p_arr[i]
            path[i, 1] = p_arr[i + 1]
        max_pts = math.ceil(duration / dt)

        n_pts = int(max_pts / (p_arr.shape[0] - 1))
        pts = torch.zeros((math.ceil(duration / dt), p_arr.shape[-1]))

        linear_pts = torch.zeros((n_pts, p_arr.shape[-1]))

        for i in range(1, p_arr.shape[0]):
            weight = torch.as_tensor([(i + 1) / n_pts for i in range(n_pts)])
            # do linear interplation between p_arr[i-1], p_arr[i]
            for j in range(linear_pts.shape[0]):
                linear_pts[j] = p_arr[i - 1] + weight[j] * (p_arr[i] - p_arr[i - 1])
            pts[(i - 1) * n_pts : (i) * n_pts] = linear_pts

        # compute velocity and acceleration:

        # pts[0] = path[0, 0]
        # pts[-1] = path[-1, 1]
        # pts = pts[: i * n_pts]
        pts[i * n_pts - 1 :] = pts[i * n_pts - 1].clone()
        pts = pts.to(**tensor_args)
        vel = (pts.clone().roll(-1, dims=0) - pts) / dt
        vel = vel.roll(1, dims=0)
        vel[0] = 0.0
        acc = (vel.clone().roll(-1, dims=0) - vel) / dt
        acc = acc.roll(1, dims=0)
        acc[0] = 0.0
        trajectory = {
            "position": pts,
            "velocity": vel,
            "acceleration": acc,
            "traj_buffer": torch.cat((pts, vel, acc), dim=-1),
        }
    return trajectory


def get_smooth_trajectory(raw_traj: torch.Tensor, degree: int = 5):
    cpu_traj = raw_traj.cpu()

    smooth_traj = torch.zeros_like(cpu_traj)
    for i in range(cpu_traj.shape[-1]):
        smooth_traj[:, i] = bspline(cpu_traj[:, i], n=cpu_traj.shape[0], degree=degree)
    return smooth_traj.to(dtype=raw_traj.dtype, device=raw_traj.device)


def get_spline_interpolated_trajectory(raw_traj: torch.Tensor, des_horizon: int, degree: int = 5):
    retimed_traj = torch.zeros((des_horizon, raw_traj.shape[-1]))
    tensor_args = TensorDeviceType(device=raw_traj.device, dtype=raw_traj.dtype)
    cpu_traj = raw_traj.cpu()
    for i in range(cpu_traj.shape[-1]):
        retimed_traj[:, i] = bspline(cpu_traj[:, i], n=des_horizon, degree=degree)
    retimed_traj = retimed_traj.to(**(tensor_args.as_torch_dict()))
    return retimed_traj


def get_batch_interpolated_trajectory(
    raw_traj: JointState,
    raw_dt: torch.Tensor,
    interpolation_dt: float,
    max_vel: Optional[torch.Tensor] = None,
    max_acc: Optional[torch.Tensor] = None,
    max_jerk: Optional[torch.Tensor] = None,
    kind: InterpolateType = InterpolateType.LINEAR_CUDA,
    out_traj_state: Optional[JointState] = None,
    tensor_args: TensorDeviceType = TensorDeviceType(),
    max_deviation: float = 0.1,
    min_dt: float = 0.02,
    max_dt: float = 0.15,
    optimize_dt: bool = True,
):
    # compute dt across trajectory:
    if len(raw_traj.shape) == 2:
        raw_traj = raw_traj.unsqueeze(0)
    if out_traj_state is not None and len(out_traj_state.shape) == 2:
        out_traj_state = out_traj_state.unsqueeze(0)
    b, horizon, dof = raw_traj.position.shape  # horizon
    # given the dt required to run trajectory at maximum velocity,
    # we find the number of timesteps required:
    if optimize_dt:
        if max_vel is None:
            log_error("Max velocity not provided")
        if max_acc is None:
            log_error("Max acceleration not provided")
        if max_jerk is None:
            log_error("Max jerk not provided")
    if max_vel is not None and max_acc is not None and max_jerk is not None:
        traj_vel = raw_traj.velocity
        traj_acc = raw_traj.acceleration
        traj_jerk = raw_traj.jerk
        if "raw_velocity" in raw_traj.aux_data:
            traj_vel = raw_traj.aux_data["raw_velocity"]
        if "raw_acceleration" in raw_traj.aux_data:
            traj_acc = raw_traj.aux_data["raw_acceleration"]
        if "raw_jerk" in raw_traj.aux_data:
            traj_jerk = raw_traj.aux_data["raw_jerk"]
        traj_steps, steps_max, opt_dt = calculate_tsteps(
            traj_vel,
            traj_acc,
            traj_jerk,
            interpolation_dt,
            max_vel,
            max_acc,
            max_jerk,
            raw_dt,
            min_dt,
            max_dt,
            horizon,
            optimize_dt,
        )
    else:
        traj_steps, steps_max = calculate_traj_steps(raw_dt, interpolation_dt, horizon)
        opt_dt = torch.zeros(b, device=tensor_args.device)
        opt_dt[:] = raw_dt
    # traj_steps contains the tsteps for each trajectory
    if steps_max <= 0:
        log_error("Steps max is less than 1, with a value: " + str(steps_max))

    if out_traj_state is not None and out_traj_state.position.shape[1] < steps_max:
        log_warn(
            "Interpolation buffer shape is smaller than steps_max: "
            + str(out_traj_state.position.shape)
            + " creating new buffer of shape "
            + str(steps_max)
        )
        out_traj_state = None

    if out_traj_state is None:
        out_traj_state = JointState.zeros(
            [b, steps_max, dof], tensor_args, joint_names=raw_traj.joint_names
        )

    if kind in [InterpolateType.LINEAR, InterpolateType.CUBIC]:
        # plot and save:
        out_traj_state = get_cpu_linear_interpolation(
            raw_traj,
            traj_steps,
            out_traj_state,
            kind,
            opt_dt=opt_dt,
            interpolation_dt=interpolation_dt,
        )

    elif kind == InterpolateType.LINEAR_CUDA:
        out_traj_state = get_cuda_linear_interpolation(
            raw_traj, traj_steps, out_traj_state, opt_dt, raw_dt
        )
    elif kind == InterpolateType.KUNZ_STILMAN_OPTIMAL:
        out_traj_state = get_cpu_kunz_stilman_interpolation(
            raw_traj,
            traj_steps,
            out_traj_state,
            opt_dt=opt_dt,
            interpolation_dt=interpolation_dt,
            max_velocity=max_vel,
            max_acceleration=max_acc,
            max_deviation=max_deviation,
        )
    else:
        raise ValueError("Unknown interpolation type")

    return out_traj_state, traj_steps, opt_dt


def get_cpu_linear_interpolation(
    raw_traj, traj_steps, out_traj_state, kind: InterpolateType, opt_dt=None, interpolation_dt=None
):
    cpu_traj = raw_traj.position.cpu().numpy()
    out_traj = out_traj_state.position
    retimed_traj = out_traj.cpu()
    for k in range(out_traj.shape[0]):
        tstep = traj_steps[k].item()
        opt_d = opt_dt[k].item()
        for i in range(cpu_traj.shape[-1]):
            retimed_traj[k, :tstep, i] = linear_smooth(
                cpu_traj[k, :, i],
                y=None,
                n=tstep,
                kind=kind,
                last_step=tstep,
                opt_dt=opt_d,
                interpolation_dt=interpolation_dt,
            )
        retimed_traj[k, tstep:, :] = retimed_traj[k, tstep - 1 : tstep, :]

    out_traj_state.position[:] = retimed_traj.to(device=raw_traj.position.device)
    return out_traj_state


def get_cpu_kunz_stilman_interpolation(
    raw_traj: JointState,
    traj_steps: int,
    out_traj_state: JointState,
    max_velocity: torch.Tensor,
    max_acceleration: torch.Tensor,
    opt_dt: float,
    interpolation_dt: float,
    max_deviation: float = 0.1,
):
    try:
        # Third Party
        from trajectory_smoothing import TrajectorySmoother
    except:
        log_info(
            "trajectory_smoothing package not found, try installing curobo with "
            + "pip install .[smooth]"
        )
        return get_cpu_linear_interpolation(
            raw_traj, traj_steps, out_traj_state, InterpolateType.LINEAR, opt_dt, interpolation_dt
        )

    cpu_traj = raw_traj.position.cpu().numpy()
    out_traj = out_traj_state.position
    retimed_traj = out_traj.cpu()
    out_traj_vel = out_traj_state.velocity.cpu()
    out_traj_acc = out_traj_state.acceleration.cpu()
    out_traj_jerk = out_traj_state.jerk.cpu()
    dof = cpu_traj.shape[-1]
    trajectory_sm = TrajectorySmoother(
        dof,
        max_velocity.cpu().view(dof).numpy(),
        max_acceleration.cpu().view(dof).numpy() * 0.5,
        max_deviation,
    )
    for k in range(out_traj.shape[0]):
        tstep = traj_steps[k].item()
        opt_d = opt_dt[k].item()
        in_traj = np.copy(cpu_traj[k])

        if np.sum(in_traj[-1]) != 0.0:
            out = trajectory_sm.smooth_interpolate(
                in_traj, traj_dt=0.001, interpolation_dt=interpolation_dt, max_tsteps=tstep
            )
            if out.success:
                retimed_traj[k, : out.length, :] = torch.as_tensor(out.position)
                out_traj_vel[k, : out.length, :] = torch.as_tensor(out.velocity)
                out_traj_acc[k, : out.length, :] = torch.as_tensor(out.acceleration)
                out_traj_jerk[k, : out.length, :] = torch.as_tensor(out.jerk)
                retimed_traj[k, out.length :, :] = retimed_traj[k, out.length - 1 : out.length, :]

                out_traj_vel[k, out.length :, :] = out_traj_vel[k, out.length - 1 : out.length, :]
                out_traj_acc[k, out.length :, :] = out_traj_acc[k, out.length - 1 : out.length, :]
                out_traj_jerk[k, out.length :, :] = out_traj_jerk[k, out.length - 1 : out.length, :]
            else:
                log_warn("Kunz Stilman interpolation failed, using linear")
                for i in range(cpu_traj.shape[-1]):
                    retimed_traj[k, :tstep, i] = linear_smooth(
                        cpu_traj[k, :, i],
                        y=None,
                        n=tstep,
                        kind=InterpolateType.LINEAR,
                        last_step=tstep,
                        opt_dt=opt_d,
                        interpolation_dt=interpolation_dt,
                    )
                retimed_traj[k, tstep:, :] = retimed_traj[k, tstep - 1 : tstep, :]
        else:
            for i in range(cpu_traj.shape[-1]):
                retimed_traj[k, :tstep, i] = linear_smooth(
                    cpu_traj[k, :, i],
                    y=None,
                    n=tstep,
                    kind=InterpolateType.LINEAR,
                    last_step=tstep,
                    opt_dt=opt_d,
                    interpolation_dt=interpolation_dt,
                )
            retimed_traj[k, tstep:, :] = retimed_traj[k, tstep - 1 : tstep, :]
    out_traj_state.position[:] = retimed_traj.to(device=raw_traj.position.device)
    out_traj_state.velocity[:] = out_traj_vel.to(device=raw_traj.position.device)
    out_traj_state.acceleration[:] = out_traj_acc.to(device=raw_traj.position.device)
    out_traj_state.jerk[:] = out_traj_jerk.to(device=raw_traj.position.device)

    return out_traj_state


def get_interpolated_trajectory(
    trajectory: List[torch.Tensor],
    out_traj_state: JointState,
    des_horizon: Optional[int] = None,
    interpolation_dt: float = 0.02,
    max_velocity: Optional[torch.Tensor] = None,
    max_acceleration: Optional[torch.Tensor] = None,
    max_jerk: Optional[torch.Tensor] = None,
    kind=InterpolateType.CUBIC,
    max_deviation: float = 0.05,
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> JointState:
    try:
        # Third Party
        from trajectory_smoothing import TrajectorySmoother

    except:
        log_info(
            "trajectory_smoothing package not found, InterpolateType.KUNZ_STILMAN_OPTIMAL"
            + " is disabled. to enable, try installing curobo with"
            + " pip install .[smooth]"
        )
        kind = InterpolateType.LINEAR
    dof = trajectory[0].shape[-1]
    last_tsteps = []
    opt_dt = []
    if des_horizon is None:
        interpolation_steps = out_traj_state.position.shape[1]
    else:
        interpolation_steps = des_horizon

    # create an empty state message to fill data:
    if kind == InterpolateType.KUNZ_STILMAN_OPTIMAL:
        trajectory_sm = TrajectorySmoother(
            dof,
            max_velocity.cpu().view(dof).numpy(),
            max_acceleration.cpu().view(dof).numpy(),
            max_deviation,
        )

    for b in range(len(trajectory)):
        raw_traj = trajectory[b].cpu().view(-1, dof).numpy()
        current_kind = kind

        if current_kind == InterpolateType.KUNZ_STILMAN_OPTIMAL:
            out = trajectory_sm.smooth_interpolate(
                raw_traj, interpolation_dt=interpolation_dt, traj_dt=0.001, max_tsteps=des_horizon
            )
            if out.success:
                out_traj_state.position[b, : out.length, :] = tensor_args.to_device(out.position)
                out_traj_state.position[b, out.length :, :] = out_traj_state.position[
                    b, out.length - 1 : out.length, :
                ]
                out_traj_state.velocity[b, : out.length, :] = tensor_args.to_device(out.velocity)
                out_traj_state.velocity[b, out.length :, :] = out_traj_state.velocity[
                    b, out.length - 1 : out.length, :
                ]
                out_traj_state.acceleration[b, : out.length, :] = tensor_args.to_device(
                    out.acceleration
                )
                out_traj_state.acceleration[b, out.length :, :] = out_traj_state.acceleration[
                    b, out.length - 1 : out.length, :
                ]
                out_traj_state.jerk[b, : out.length, :] = tensor_args.to_device(out.jerk)
                out_traj_state.jerk[b, out.length :, :] = out_traj_state.jerk[
                    b, out.length - 1 : out.length, :
                ]
                last_tsteps.append(out.length)
                opt_dt.append(out.interpolation_dt)
            else:
                current_kind = InterpolateType.LINEAR
        if current_kind in [InterpolateType.LINEAR, InterpolateType.CUBIC, InterpolateType.QUINTIC]:
            retimed_traj = torch.zeros((interpolation_steps, raw_traj.shape[-1]))
            if raw_traj.shape[0] < 5:
                current_kind = InterpolateType.LINEAR
            for i in range(raw_traj.shape[-1]):
                retimed_traj[:, i] = linear_smooth(
                    raw_traj[:, i],
                    y=None,
                    n=interpolation_steps,
                    kind=kind,
                    last_step=des_horizon,
                )
            retimed_traj = retimed_traj.to(**(tensor_args.as_torch_dict()))
            out_traj_state.position[b, :interpolation_steps, :] = retimed_traj
            out_traj_state.position[b, interpolation_steps:, :] = retimed_traj[
                interpolation_steps - 1 : interpolation_steps, :
            ]
            last_tsteps.append(interpolation_steps)
            opt_dt.append(interpolation_dt)
    opt_dt = tensor_args.to_device(opt_dt)
    return out_traj_state, last_tsteps, opt_dt


@profiler.record_function("interpolation/1D")
def linear_smooth(
    x: np.array,
    y=None,
    n=10,
    kind=InterpolateType.CUBIC,
    last_step=None,
    opt_dt=None,
    interpolation_dt=None,
):
    # Third Party
    import numpy as np
    from scipy import interpolate

    if last_step is None:
        last_step = n  # min(x.shape[0],n)

    if opt_dt is not None:
        y = np.ravel([i * opt_dt for i in range(x.shape[0])])

    if kind == InterpolateType.CUBIC and y is None:
        y = np.linspace(0, last_step + 3, x.shape[0] + 4)
        x = np.concatenate((x, x[-1:], x[-1:], x[-1:], x[-1:]))
    elif y is None:
        step = float(last_step - 1) / float(x.shape[0] - 1)
        y = np.ravel([float(i) * step for i in range(x.shape[0])])
        # y[-1] = np.floor(y[-1])

    if kind == InterpolateType.QUINTIC:
        f = interpolate.make_interp_spline(y, x, k=5)

    else:
        f = interpolate.interp1d(y, x, kind=kind.value, assume_sorted=True)
    if opt_dt is None:
        x_new = np.ravel([i for i in range(last_step)])
    else:
        x_new = np.ravel([i * interpolation_dt for i in range(last_step)])
    ynew = f(x_new)
    y_new = torch.as_tensor(ynew)
    return y_new


@get_torch_jit_decorator()
def calculate_dt_fixed(
    vel: torch.Tensor,
    acc: torch.Tensor,
    jerk: torch.Tensor,
    max_vel: torch.Tensor,
    max_acc: torch.Tensor,
    max_jerk: torch.Tensor,
    raw_dt: torch.Tensor,
    min_dt: float,
    max_dt: float,
    epsilon: float = 1e-4,
):
    # compute scaled dt:
    max_v_arr = torch.max(torch.abs(vel), dim=-2)[0]  # output is batch, dof

    max_acc_arr = torch.max(torch.abs(acc), dim=-2)[0]
    max_jerk_arr = torch.max(torch.abs(jerk), dim=-2)[0]

    vel_scale_dt = (max_v_arr) / (max_vel.view(1, max_v_arr.shape[-1]))  # batch,dof
    acc_scale_dt = max_acc_arr / (max_acc.view(1, max_acc_arr.shape[-1]))
    jerk_scale_dt = max_jerk_arr / (max_jerk.view(1, max_jerk_arr.shape[-1]))

    dt_score_vel = raw_dt * torch.max(vel_scale_dt, dim=-1)[0]  # batch, 1
    dt_score_acc = raw_dt * torch.sqrt((torch.max(acc_scale_dt, dim=-1)[0]))
    dt_score_jerk = raw_dt * torch.pow((torch.max(jerk_scale_dt, dim=-1)[0]), 1 / 3)
    dt_score = torch.maximum(dt_score_vel, dt_score_acc)
    dt_score = torch.maximum(dt_score, dt_score_jerk)

    dt_score = torch.clamp(dt_score * (1.0 + epsilon), min_dt, max_dt)

    return dt_score


@get_torch_jit_decorator(force_jit=True)
def calculate_dt(
    vel: torch.Tensor,
    acc: torch.Tensor,
    jerk: torch.Tensor,
    max_vel: torch.Tensor,
    max_acc: torch.Tensor,
    max_jerk: torch.Tensor,
    raw_dt: float,
    min_dt: float,
    epsilon: float = 1e-4,
):
    # compute scaled dt:
    max_v_arr = torch.max(torch.abs(vel), dim=-2)[0]  # output is batch, dof

    max_acc_arr = torch.max(torch.abs(acc), dim=-2)[0]
    max_jerk_arr = torch.max(torch.abs(jerk), dim=-2)[0]

    vel_scale_dt = (max_v_arr) / (max_vel.view(1, max_v_arr.shape[-1]))  # batch,dof
    acc_scale_dt = max_acc_arr / (max_acc.view(1, max_acc_arr.shape[-1]))
    jerk_scale_dt = max_jerk_arr / (max_jerk.view(1, max_jerk_arr.shape[-1]))

    dt_score_vel = raw_dt * torch.max(vel_scale_dt, dim=-1)[0]  # batch, 1
    dt_score_acc = raw_dt * torch.sqrt((torch.max(acc_scale_dt, dim=-1)[0]))
    dt_score_jerk = raw_dt * torch.pow((torch.max(jerk_scale_dt, dim=-1)[0]), 1 / 3)
    dt_score = torch.maximum(dt_score_vel, dt_score_acc)
    dt_score = torch.maximum(dt_score, dt_score_jerk)
    dt_score = torch.clamp(dt_score * (1.0 + epsilon), min_dt, raw_dt)

    # NOTE: this dt score is not dt, rather a scaling to convert velocity, acc, jerk that was
    # computed with raw_dt to a new dt
    return dt_score


@get_torch_jit_decorator(force_jit=True)
def calculate_dt_no_clamp(
    vel: torch.Tensor,
    acc: torch.Tensor,
    jerk: torch.Tensor,
    max_vel: torch.Tensor,
    max_acc: torch.Tensor,
    max_jerk: torch.Tensor,
    epsilon: float = 1e-4,
):
    # compute scaled dt:
    max_v_arr = torch.max(torch.abs(vel), dim=-2)[0]  # output is batch, dof

    max_acc_arr = torch.max(torch.abs(acc), dim=-2)[0]
    max_jerk_arr = torch.max(torch.abs(jerk), dim=-2)[0]

    # max_v_arr = torch.clamp(max_v_arr, None, max_vel.view(1, max_v_arr.shape[-1]))
    vel_scale_dt = (max_v_arr) / (max_vel.view(1, max_v_arr.shape[-1]))  # batch,dof
    acc_scale_dt = max_acc_arr / (max_acc.view(1, max_acc_arr.shape[-1]))
    jerk_scale_dt = max_jerk_arr / (max_jerk.view(1, max_jerk_arr.shape[-1]))
    dt_score_vel = torch.max(vel_scale_dt, dim=-1)[0]  # batch, 1
    dt_score_acc = torch.sqrt((torch.max(acc_scale_dt, dim=-1)[0]))
    dt_score_jerk = torch.pow((torch.max(jerk_scale_dt, dim=-1)[0]), 1 / 3)
    dt_score = torch.maximum(dt_score_vel, dt_score_acc)
    dt_score = torch.maximum(dt_score, dt_score_jerk)
    dt_score = dt_score * (1.0 + epsilon)
    return dt_score


@get_torch_jit_decorator()
def calculate_traj_steps(
    opt_dt: torch.Tensor, interpolation_dt: float, horizon: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    traj_steps = (torch.ceil((horizon - 1) * ((opt_dt) / interpolation_dt))).to(dtype=torch.int32)
    steps_max = torch.max(traj_steps)
    return traj_steps, steps_max


@get_torch_jit_decorator()
def calculate_tsteps(
    vel: torch.Tensor,
    acc: torch.Tensor,
    jerk: torch.Tensor,
    interpolation_dt: float,
    max_vel: torch.Tensor,
    max_acc: torch.Tensor,
    max_jerk: torch.Tensor,
    raw_dt: torch.Tensor,
    min_dt: float,
    max_dt: float,
    horizon: int,
    optimize_dt: bool = True,
):
    # compute scaled dt:
    opt_dt = calculate_dt_fixed(
        vel,
        acc,
        jerk,
        max_vel,
        max_acc,
        max_jerk,
        raw_dt,
        min_dt,
        max_dt,
    )
    if not optimize_dt:
        opt_dt[:] = raw_dt
    # check for nan:
    opt_dt = torch.nan_to_num(opt_dt, nan=min_dt)
    traj_steps, steps_max = calculate_traj_steps(opt_dt, interpolation_dt, horizon)
    return traj_steps, steps_max, opt_dt
