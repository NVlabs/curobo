# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Third Party

# Third Party
import torch

from curobo._src.curobolib.backends import trajectory as trajectory_cu
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float32_tensors,
    check_int32_tensors,
    check_uint8_tensors,
)

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise


def get_bspline_interpolation(
    input_trajectory: JointState,
    output_trajectory: JointState,
    interpolation_dt: torch.Tensor,
    current_state: JointState,
    goal_state: JointState,
    start_idx: torch.Tensor,
    goal_idx: torch.Tensor,
    use_implicit_goal_state: torch.Tensor,
    interpolated_horizon: torch.Tensor,
    bspline_degree=4,
):
    batch_size, max_out_tsteps, dof = output_trajectory.shape
    n_knots = input_trajectory.knot.shape[-2]

    device = output_trajectory.position.device
    check_float32_tensors(
        device,
        out_position=output_trajectory.position,
        out_velocity=output_trajectory.velocity,
        out_acceleration=output_trajectory.acceleration,
        out_jerk=output_trajectory.jerk,
        out_dt=output_trajectory.dt,
        knots=input_trajectory.knot,
        knot_dt=input_trajectory.knot_dt,
        cur_position=current_state.position,
        cur_velocity=current_state.velocity,
        cur_acceleration=current_state.acceleration,
        cur_jerk=current_state.jerk,
        goal_position=goal_state.position,
        goal_velocity=goal_state.velocity,
        goal_acceleration=goal_state.acceleration,
        goal_jerk=goal_state.jerk,
        interpolation_dt=interpolation_dt,
    )
    check_int32_tensors(
        device,
        start_idx=start_idx,
        goal_idx=goal_idx,
        interpolated_horizon=interpolated_horizon,
    )
    check_uint8_tensors(device, use_implicit_goal_state=use_implicit_goal_state)

    trajectory_cu.launch_bspline_interpolation_single_dt_kernel(
        output_trajectory.position,
        output_trajectory.velocity,
        output_trajectory.acceleration,
        output_trajectory.jerk,
        output_trajectory.dt,
        input_trajectory.knot,
        input_trajectory.knot_dt,
        current_state.position,
        current_state.velocity,
        current_state.acceleration,
        current_state.jerk,
        goal_state.position,
        goal_state.velocity,
        goal_state.acceleration,
        goal_state.jerk,
        start_idx,
        goal_idx,
        interpolation_dt,
        use_implicit_goal_state,
        interpolated_horizon,
        batch_size,
        max_out_tsteps,
        dof,
        n_knots,
        bspline_degree,
    )

    return output_trajectory


class CliqueTensorStepIdxKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        start_velocity,
        start_acceleration,
        goal_position,
        goal_velocity,
        goal_acceleration,
        start_idx,
        goal_idx,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        out_dt,
        traj_dt,
        use_implicit_goal_state,
        out_grad_position,
    ):
        horizon = out_position.shape[1]
        # if len(u_act.shape) != 3:
        #    log_and_raise("Action should be 3-dimensional tensor, current shape: " + str(u_act.shape))
        if u_act.shape[-2] != horizon - 4:
            log_and_raise("Action shape is not compatible with horizon: " + str(u_act.shape))
        device = u_act.device
        check_float32_tensors(
            device,
            u_act=u_act,
            start_position=start_position,
            start_velocity=start_velocity,
            start_acceleration=start_acceleration,
            goal_position=goal_position,
            goal_velocity=goal_velocity,
            goal_acceleration=goal_acceleration,
            out_position=out_position,
            out_velocity=out_velocity,
            out_acceleration=out_acceleration,
            out_jerk=out_jerk,
            out_dt=out_dt,
            traj_dt=traj_dt,
            out_grad_position=out_grad_position,
        )
        check_int32_tensors(device, start_idx=start_idx, goal_idx=goal_idx)
        check_uint8_tensors(device, use_implicit_goal_state=use_implicit_goal_state)

        trajectory_cu.launch_differentiation_position_forward_kernel(
            out_position,
            out_velocity,
            out_acceleration,
            out_jerk,
            out_dt,
            u_act,
            start_position,
            start_velocity,
            start_acceleration,
            goal_position,
            goal_velocity,
            goal_acceleration,
            start_idx,
            goal_idx,
            traj_dt,
            use_implicit_goal_state,  # shape (num_problems), 1 per seed.
            out_position.shape[0],
            out_position.shape[1],
            out_position.shape[-1],
        )

        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(traj_dt, out_grad_position, goal_idx, use_implicit_goal_state)
        return out_position, out_velocity, out_acceleration, out_jerk

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        u_grad = None

        if ctx.needs_input_grad[0]:
            (traj_dt, out_grad_position, goal_idx, use_implicit_goal_state) = ctx.saved_tensors
            batch_size = grad_out_p.shape[0]
            horizon = grad_out_p.shape[1]
            dof = grad_out_p.shape[2]
            if grad_out_p is None:
                log_and_raise("grad_out_p is None")
            if grad_out_v is None:
                log_and_raise("grad_out_v is None")
            if grad_out_a is None:
                log_and_raise("grad_out_a is None")
            if grad_out_j is None:
                log_and_raise("grad_out_j is None")
            if out_grad_position is None:
                log_and_raise("out_grad_position is None")
            device = grad_out_p.device
            check_float32_tensors(
                device,
                out_grad_position=out_grad_position,
                grad_out_p=grad_out_p,
                grad_out_v=grad_out_v,
                grad_out_a=grad_out_a,
                grad_out_j=grad_out_j,
                traj_dt=traj_dt,
            )
            check_int32_tensors(device, goal_idx=goal_idx)
            check_uint8_tensors(device, use_implicit_goal_state=use_implicit_goal_state)

            trajectory_cu.launch_differentiation_position_backward_kernel(
                out_grad_position,
                grad_out_p,
                grad_out_v,
                grad_out_a,
                grad_out_j,
                traj_dt,
                goal_idx,  # shape: (batch_size, )
                use_implicit_goal_state,  # should have at least max(goal_idx).
                batch_size,
                horizon,
                dof,
            )
            u_grad = out_grad_position

        return (
            u_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class AccelerationTensorStepIdxKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        start_velocity,
        start_acceleration,
        start_idx,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        traj_dt,
        out_grad_position,
    ):
        device = u_act.device
        check_float32_tensors(
            device,
            u_act=u_act,
            start_position=start_position,
            start_velocity=start_velocity,
            start_acceleration=start_acceleration,
            out_position=out_position,
            out_velocity=out_velocity,
            out_acceleration=out_acceleration,
            out_jerk=out_jerk,
            traj_dt=traj_dt,
            out_grad_position=out_grad_position,
        )
        check_int32_tensors(device, start_idx=start_idx)

        trajectory_cu.launch_integration_acceleration_kernel(
            out_position,
            out_velocity,
            out_acceleration,
            out_jerk,
            u_act,
            start_position,
            start_velocity,
            start_acceleration,
            start_idx,
            traj_dt,
            out_position.shape[0],
            out_position.shape[1],
            out_position.shape[-1],
            True,
        )

        ctx.save_for_backward(traj_dt, out_grad_position)
        return out_position, out_velocity, out_acceleration, out_jerk

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        u_grad = None
        (traj_dt, out_grad_position) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            raise NotImplementedError()

        return u_grad, None, None, None, None, None, None, None, None, None, None


class BSplineIdxKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        start_velocity,
        start_acceleration,
        start_jerk,
        goal_position,
        goal_velocity,
        goal_acceleration,
        goal_jerk,
        start_idx,
        goal_idx,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        out_dt,
        traj_dt,
        use_implicit_goal_state,
        out_grad_position,
        bspline_degree,
        use_flat_gradient=False,
    ):
        n_knots = u_act.shape[-2]
        device = u_act.device
        check_float32_tensors(
            device,
            u_act=u_act,
            start_position=start_position,
            start_velocity=start_velocity,
            start_acceleration=start_acceleration,
            start_jerk=start_jerk,
            goal_position=goal_position,
            goal_velocity=goal_velocity,
            goal_acceleration=goal_acceleration,
            goal_jerk=goal_jerk,
            out_position=out_position,
            out_velocity=out_velocity,
            out_acceleration=out_acceleration,
            out_jerk=out_jerk,
            out_dt=out_dt,
            traj_dt=traj_dt,
            out_grad_position=out_grad_position,
        )
        check_int32_tensors(device, start_idx=start_idx, goal_idx=goal_idx)
        check_uint8_tensors(device, use_implicit_goal_state=use_implicit_goal_state)

        trajectory_cu.launch_bspline_interpolation_forward_kernel(
            out_position,
            out_velocity,
            out_acceleration,
            out_jerk,
            out_dt,
            u_act,
            start_position,
            start_velocity,
            start_acceleration,
            start_jerk,
            goal_position,
            goal_velocity,
            goal_acceleration,
            goal_jerk,
            start_idx,
            goal_idx,
            traj_dt,
            use_implicit_goal_state,
            out_position.shape[0],
            out_position.shape[1],
            out_position.shape[-1],
            n_knots,
            bspline_degree,
        )

        ctx.use_flat_gradient = use_flat_gradient

        ctx.save_for_backward(traj_dt, out_grad_position, goal_idx, use_implicit_goal_state)
        ctx.n_knots = n_knots
        ctx.bspline_degree = bspline_degree
        return out_position, out_velocity, out_acceleration, out_jerk

    # @profiler.record_function("BSplineKernel/bwd")
    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        u_grad = None
        use_flat = ctx.use_flat_gradient
        if ctx.needs_input_grad[0]:
            (traj_dt, out_grad_position, dt_idx, use_implicit_goal_state) = ctx.saved_tensors
            n_knots = ctx.n_knots
            padded_horizon = grad_out_p.shape[1]
            if (
                grad_out_v.shape[1] != padded_horizon
                or grad_out_a.shape[1] != padded_horizon
                or grad_out_j.shape[1] != padded_horizon
            ):
                log_and_raise(
                    f"BSpline backward: grad tensor dim-1 mismatch: "
                    f"p={padded_horizon}, v={grad_out_v.shape[1]}, "
                    f"a={grad_out_a.shape[1]}, j={grad_out_j.shape[1]}"
                )

            device = grad_out_p.device
            check_float32_tensors(
                device,
                out_grad_position=out_grad_position,
                grad_out_p=grad_out_p,
                grad_out_v=grad_out_v,
                grad_out_a=grad_out_a,
                grad_out_j=grad_out_j,
                traj_dt=traj_dt,
            )
            check_int32_tensors(device, dt_idx=dt_idx)
            check_uint8_tensors(device, use_implicit_goal_state=use_implicit_goal_state)

            trajectory_cu.launch_bspline_interpolation_backward_kernel(
                out_grad_position,
                grad_out_p,
                grad_out_v,
                grad_out_a,
                grad_out_j,
                traj_dt,
                dt_idx,
                use_implicit_goal_state,
                grad_out_p.shape[0],
                grad_out_p.shape[1],
                grad_out_p.shape[2],
                n_knots,
                ctx.bspline_degree,
                use_flat,
            )
            u_grad = out_grad_position

        return (
            u_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )  # , None, None, None, None,None
