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
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

# Third Party
import torch
import warp as wp

# CuRobo
from curobo.cuda_robot_model.types import JointLimits
from curobo.types.robot import JointState
from curobo.types.tensor import T_DOF
from curobo.util.logger import log_error, log_warn
from curobo.util.torch_utils import get_cache_fn_decorator, get_torch_jit_decorator
from curobo.util.warp import init_warp, is_runtime_warp_kernel_enabled, warp_support_kernel_key

# Local Folder
from .cost_base import CostBase, CostConfig

wp.set_module_options({"fast_math": False})


class BoundCostType(Enum):
    POSITION = 0
    BOUNDS = 1
    BOUNDS_SMOOTH = 2


@dataclass
class BoundCostConfig(CostConfig):
    joint_limits: Optional[JointLimits] = None
    smooth_weight: Optional[List[float]] = None
    run_weight_velocity: float = 0.0
    run_weight_acceleration: float = 0.0
    run_weight_jerk: float = 0.0
    cspace_distance_weight: Optional[T_DOF] = None
    cost_type: Optional[BoundCostType] = None
    activation_distance: Union[torch.Tensor, float] = 0.0
    state_finite_difference_mode: str = "BACKWARD"
    null_space_weight: Optional[List[float]] = None
    use_l2_kernel: bool = False

    def set_bounds(self, bounds: JointLimits, teleport_mode: bool = False):
        self.joint_limits = bounds.clone()
        if teleport_mode:
            self.cost_type = BoundCostType.POSITION
        if self.cost_type != BoundCostType.POSITION:
            if torch.max(self.joint_limits.velocity[1] - self.joint_limits.velocity[0]) == 0.0:
                log_error("Joint velocity limits is zero")
            if (
                torch.max(self.joint_limits.acceleration[1] - self.joint_limits.acceleration[0])
                == 0.0
            ):
                log_error("Joint acceleration limits is zero")
            if torch.max(self.joint_limits.jerk[1] - self.joint_limits.jerk[0]) == 0.0:
                log_error("Joint jerk limits is zero")

    def __post_init__(self):
        if isinstance(self.activation_distance, List):
            self.activation_distance = self.tensor_args.to_device(self.activation_distance)
        elif isinstance(self.activation_distance, float):
            raise ValueError("Activation distance is a list for bound cost.")
        if self.smooth_weight is not None:
            self.smooth_weight = self.tensor_args.to_device(self.smooth_weight)

        if self.cost_type is None:
            if self.smooth_weight is not None:
                self.cost_type = BoundCostType.BOUNDS_SMOOTH
            else:
                self.cost_type = BoundCostType.BOUNDS

        if self.cspace_distance_weight is not None:
            self.cspace_distance_weight = 1.0 + self.cspace_distance_weight / torch.max(
                self.cspace_distance_weight
            )
        if self.null_space_weight is None:
            self.null_space_weight = self.tensor_args.to_device([0.0])
        else:
            self.null_space_weight = self.tensor_args.to_device(self.null_space_weight)
        if self.vec_weight is None:
            self.vec_weight = self.tensor_args.to_device([0.0])
        return super().__post_init__()


class BoundCost(CostBase, BoundCostConfig):
    def __init__(self, config: BoundCostConfig):
        BoundCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        init_warp()
        self._batch_size = -1
        self._horizon = -1
        self._dof = -1
        empty_buffer = torch.tensor(
            (0, 0, 0), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self._out_gv_buffer = self._out_ga_buffer = self._out_gj_buffer = empty_buffer
        if self.use_l2_kernel:
            if not is_runtime_warp_kernel_enabled():
                log_warn("Runtime warp kernel generation is disabled.")
                self.use_l2_kernel = False
            if not warp_support_kernel_key():
                # define a compile-time constant so that warp hash is different for different dof
                # this is required in older warp versions < 1.2.1 as warp hash didn't consider the
                # name of kernels. Newer warp versions have fixed this issue.
                WARP_CUROBO_BOUNDCOST_DOF_GLOBAL_CONSTANT = wp.constant(self.dof)

            if self.cost_type == BoundCostType.POSITION:
                self._l2_cost = make_bound_pos_kernel(self.dof)

            if self.cost_type == BoundCostType.BOUNDS_SMOOTH:
                self._l2_cost = make_bound_pos_smooth_kernel(self.dof)

    def update_batch_size(self, batch, horizon, dof):
        if self._batch_size != batch or self._horizon != horizon or self._dof != dof:
            self._out_c_buffer = torch.zeros(
                (batch, horizon, dof), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            self._out_c_sum_buffer = torch.zeros(
                (batch, horizon), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )

            self._out_gp_buffer = torch.zeros(
                (batch, horizon, dof), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            if not self.cost_type == BoundCostType.POSITION:
                self._out_gv_buffer = torch.zeros(
                    (batch, horizon, dof),
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )
                self._out_ga_buffer = torch.zeros(
                    (batch, horizon, dof),
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )
                self._out_gj_buffer = torch.zeros(
                    (batch, horizon, dof),
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )

            # create terminal buffers:
            if self.cost_type == BoundCostType.BOUNDS_SMOOTH:
                self._run_weight_vel = torch.ones(
                    (1, horizon),
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )
                # 2: -3
                if self.state_finite_difference_mode == "BACKWARD":
                    self._run_weight_vel[:, :-4] *= self.run_weight_velocity
                elif self.state_finite_difference_mode == "CENTRAL":
                    self._run_weight_vel[:, :] *= self.run_weight_velocity

                # print(self._run_weight_vel)
                # exit()
                self._run_weight_acc = torch.ones(
                    (1, horizon),
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )

                # self._run_weight_acc[:, 3:-3] *= self.run_weight_acceleration
                self._run_weight_jerk = torch.ones(
                    (1, horizon),
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )

                # self._run_weight_jerk[:, :] *= self.run_weight_jerk

            self._batch_size = batch
            self._horizon = horizon
            self._dof = dof
            self._retract_cfg = torch.zeros(
                (dof), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            self._retract_cfg_idx = torch.zeros(
                (batch), device=self.tensor_args.device, dtype=torch.int32
            )

    def forward(
        self,
        state_batch: JointState,
        retract_config: Optional[torch.Tensor] = None,
        retract_idx: Optional[torch.Tensor] = None,
    ):
        b, h, dof = state_batch.position.shape
        self.update_batch_size(b, h, dof)
        if retract_config is None:
            retract_config = self._retract_cfg
        if retract_idx is None:
            retract_idx = self._retract_cfg_idx

        if self.cost_type == BoundCostType.BOUNDS_SMOOTH:
            if self.use_l2_kernel:
                cost = WarpBoundSmoothL2Function.apply(
                    state_batch.position,
                    state_batch.velocity,
                    state_batch.acceleration,
                    state_batch.jerk,
                    retract_config,
                    retract_idx,
                    self.joint_limits.position,
                    self.joint_limits.velocity,
                    self.joint_limits.acceleration,
                    self.joint_limits.jerk,
                    self.weight,
                    self.activation_distance,
                    self.smooth_weight,
                    self.cspace_distance_weight,
                    self.null_space_weight,
                    self.vec_weight,
                    self._run_weight_vel,
                    self._run_weight_acc,
                    self._run_weight_jerk,
                    self._out_c_sum_buffer,
                    self._out_gp_buffer,
                    self._out_gv_buffer,
                    self._out_ga_buffer,
                    self._out_gj_buffer,
                    self._l2_cost,
                )
            else:
                cost = WarpBoundSmoothFunction.apply(
                    state_batch.position,
                    state_batch.velocity,
                    state_batch.acceleration,
                    state_batch.jerk,
                    retract_config,
                    retract_idx,
                    self.joint_limits.position,
                    self.joint_limits.velocity,
                    self.joint_limits.acceleration,
                    self.joint_limits.jerk,
                    self.weight,
                    self.activation_distance,
                    self.smooth_weight,
                    self.cspace_distance_weight,
                    self.null_space_weight,
                    self.vec_weight,
                    self._run_weight_vel,
                    self._run_weight_acc,
                    self._run_weight_jerk,
                    self._out_c_buffer,
                    self._out_gp_buffer,
                    self._out_gv_buffer,
                    self._out_ga_buffer,
                    self._out_gj_buffer,
                )
        elif self.cost_type == BoundCostType.BOUNDS:
            cost = WarpBoundFunction.apply(
                state_batch.position,
                state_batch.velocity,
                state_batch.acceleration,
                state_batch.jerk,
                retract_config,
                retract_idx,
                self.joint_limits.position,
                self.joint_limits.velocity,
                self.joint_limits.acceleration,
                self.joint_limits.jerk,
                self.weight,
                self.activation_distance,
                self.null_space_weight,
                self.vec_weight,
                self._out_c_buffer,
                self._out_gp_buffer,
                self._out_gv_buffer,
                self._out_ga_buffer,
                self._out_gj_buffer,
            )
        elif self.cost_type == BoundCostType.POSITION:
            if self.use_l2_kernel:
                cost = WarpBoundPosL2Function.apply(
                    state_batch.position,
                    retract_config,
                    retract_idx,
                    self.joint_limits.position,
                    self.weight,
                    self.activation_distance,
                    self.null_space_weight,
                    self.vec_weight,
                    self._out_c_sum_buffer,
                    self._out_gp_buffer,
                    self._l2_cost,
                    self.return_loss,
                )
            else:
                cost = WarpBoundPosFunction.apply(
                    state_batch.position,
                    retract_config,
                    retract_idx,
                    self.joint_limits.position,
                    self.weight,
                    self.activation_distance,
                    self.null_space_weight,
                    self.vec_weight,
                    self._out_c_buffer,
                    self._out_gp_buffer,
                    self.return_loss,
                )

        else:
            raise ValueError("No bounds set in BoundCost")
        return cost

    def update_dt(self, dt: Union[float, torch.Tensor]):
        if self.cost_type == BoundCostType.BOUNDS_SMOOTH:
            v_scale = dt / self._dt
            a_scale = v_scale**2
            j_scale = v_scale**3
            self.smooth_weight[1] *= a_scale
            self.smooth_weight[2] *= j_scale

        return super().update_dt(dt)


@get_torch_jit_decorator()
def forward_bound_cost(p, lower_bounds, upper_bounds, weight):
    # c = weight * torch.sum(torch.nn.functional.relu(torch.max(lower_bounds - p, p - upper_bounds)), dim=-1)

    # This version does more work but fuses to 1 kernel
    # c = torch.sum(weight * torch.nn.functional.relu(torch.max(lower_bounds - p, p - upper_bounds)), dim=-1)
    c = torch.sum(
        weight
        * (torch.nn.functional.relu(lower_bounds - p) + torch.nn.functional.relu(p - upper_bounds)),
        dim=(-1),
    )
    return c


@get_torch_jit_decorator()
def forward_all_bound_cost(
    p,
    v,
    a,
    p_lower_bounds,
    p_upper_bounds,
    v_lower_bounds,
    v_upper_bounds,
    a_lower_bounds,
    a_upper_bounds,
    weight,
):
    # c = torch.sum(
    #    weight *
    #
    #    (
    #    torch.nn.functional.relu(torch.max(p_lower_bounds - p, p - p_upper_bounds))
    #    + torch.nn.functional.relu(torch.max(v_lower_bounds - v, v - v_upper_bounds))
    #    + torch.nn.functional.relu(torch.max(a_lower_bounds - a, a - a_upper_bounds))),
    #    dim=-1,
    # )

    c = torch.sum(
        weight
        * (
            torch.nn.functional.relu(p_lower_bounds - p)
            + torch.nn.functional.relu(p - p_upper_bounds)
            + torch.nn.functional.relu(v_lower_bounds - v)
            + torch.nn.functional.relu(v - v_upper_bounds)
            + torch.nn.functional.relu(a_lower_bounds - a)
            + torch.nn.functional.relu(a - a_upper_bounds)
        ),
        dim=-1,
    )

    return c


# create a bound cost tensor:
class WarpBoundSmoothFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pos,
        vel,
        acc,
        jerk,
        retract_config,
        retract_idx,
        p_b,
        v_b,
        a_b,
        j_b,
        weight,
        activation_distance,
        smooth_weight,
        cspace_weight,
        null_space_weight,
        vec_weight,
        run_weight_vel,
        run_weight_acc,
        run_weight_jerk,
        out_cost,
        out_gp,
        out_gv,
        out_ga,
        out_gj,
    ):
        # scale the weights for smoothness by this dt:
        wp_device = wp.device_from_torch(vel.device)
        # assert smooth_weight.shape[0] == 7
        b, h, dof = vel.shape
        requires_grad = pos.requires_grad

        wp.launch(
            kernel=forward_bound_smooth_warp,
            dim=b * h * dof,
            inputs=[
                wp.from_torch(pos.detach().view(-1), dtype=wp.float32),
                wp.from_torch(vel.detach().view(-1), dtype=wp.float32),
                wp.from_torch(acc.detach().view(-1), dtype=wp.float32),
                wp.from_torch(jerk.detach().view(-1), dtype=wp.float32),
                wp.from_torch(retract_config.detach().view(-1), dtype=wp.float32),
                wp.from_torch(retract_idx.detach().view(-1), dtype=wp.int32),
                wp.from_torch(p_b.view(-1), dtype=wp.float32),
                wp.from_torch(v_b.view(-1), dtype=wp.float32),
                wp.from_torch(a_b.view(-1), dtype=wp.float32),
                wp.from_torch(j_b.view(-1), dtype=wp.float32),
                wp.from_torch(weight, dtype=wp.float32),
                wp.from_torch(activation_distance, dtype=wp.float32),
                wp.from_torch(smooth_weight, dtype=wp.float32),
                wp.from_torch(cspace_weight, dtype=wp.float32),
                wp.from_torch(null_space_weight.view(-1), dtype=wp.float32),
                wp.from_torch(vec_weight.view(-1), dtype=wp.float32),
                wp.from_torch(run_weight_vel.view(-1), dtype=wp.float32),
                wp.from_torch(run_weight_acc.view(-1), dtype=wp.float32),
                wp.from_torch(run_weight_jerk.view(-1), dtype=wp.float32),
                wp.from_torch(out_cost.view(-1), dtype=wp.float32),
                wp.from_torch(out_gp.view(-1), dtype=wp.float32),
                wp.from_torch(out_gv.view(-1), dtype=wp.float32),
                wp.from_torch(out_ga.view(-1), dtype=wp.float32),
                wp.from_torch(out_gj.view(-1), dtype=wp.float32),
                requires_grad,
                b,
                h,
                dof,
            ],
            device=wp_device,
            stream=wp.stream_from_torch(vel.device),
        )
        ctx.save_for_backward(out_gp, out_gv, out_ga, out_gj)
        # out_c = out_cost
        # out_c = torch.linalg.norm(out_cost, dim=-1)
        out_c = torch.sum(out_cost, dim=-1)
        return out_c

    @staticmethod
    def backward(ctx, grad_out_cost):
        (
            p_grad,
            v_grad,
            a_grad,
            j_grad,
        ) = ctx.saved_tensors
        v_g = None
        a_g = None
        p_g = None
        j_g = None
        if ctx.needs_input_grad[0]:
            p_g = p_grad  # * grad_out_cost#.unsqueeze(-1)
        if ctx.needs_input_grad[1]:
            v_g = v_grad  # * grad_out_cost#.unsqueeze(-1)
        if ctx.needs_input_grad[2]:
            a_g = a_grad  # * grad_out_cost#.unsqueeze(-1)
        if ctx.needs_input_grad[3]:
            j_g = j_grad  # * grad_out_cost#.unsqueeze(-1)
        return (
            p_g,
            v_g,
            a_g,
            j_g,
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
        )


# create a bound cost tensor:
class WarpBoundSmoothL2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pos,
        vel,
        acc,
        jerk,
        retract_config,
        retract_idx,
        p_b,
        v_b,
        a_b,
        j_b,
        weight,
        activation_distance,
        smooth_weight,
        cspace_weight,
        null_space_weight,
        vec_weight,
        run_weight_vel,
        run_weight_acc,
        run_weight_jerk,
        out_cost,
        out_gp,
        out_gv,
        out_ga,
        out_gj,
        warp_function,
    ):
        # scale the weights for smoothness by this dt:
        wp_device = wp.device_from_torch(vel.device)
        # assert smooth_weight.shape[0] == 7
        b, h, dof = vel.shape
        requires_grad = pos.requires_grad

        wp.launch(
            kernel=warp_function,
            dim=b * h,
            inputs=[
                wp.from_torch(pos.detach().view(-1), dtype=wp.float32),
                wp.from_torch(vel.detach().view(-1), dtype=wp.float32),
                wp.from_torch(acc.detach().view(-1), dtype=wp.float32),
                wp.from_torch(jerk.detach().view(-1), dtype=wp.float32),
                wp.from_torch(retract_config.detach().view(-1), dtype=wp.float32),
                wp.from_torch(retract_idx.detach().view(-1), dtype=wp.int32),
                wp.from_torch(p_b.view(-1), dtype=wp.float32),
                wp.from_torch(v_b.view(-1), dtype=wp.float32),
                wp.from_torch(a_b.view(-1), dtype=wp.float32),
                wp.from_torch(j_b.view(-1), dtype=wp.float32),
                wp.from_torch(weight, dtype=wp.float32),
                wp.from_torch(activation_distance, dtype=wp.float32),
                wp.from_torch(smooth_weight, dtype=wp.float32),
                wp.from_torch(cspace_weight, dtype=wp.float32),
                wp.from_torch(null_space_weight.view(-1), dtype=wp.float32),
                wp.from_torch(vec_weight.view(-1), dtype=wp.float32),
                wp.from_torch(run_weight_vel.view(-1), dtype=wp.float32),
                wp.from_torch(run_weight_acc.view(-1), dtype=wp.float32),
                wp.from_torch(run_weight_jerk.view(-1), dtype=wp.float32),
                wp.from_torch(out_cost.view(-1), dtype=wp.float32),
                wp.from_torch(out_gp.view(-1), dtype=wp.float32),
                wp.from_torch(out_gv.view(-1), dtype=wp.float32),
                wp.from_torch(out_ga.view(-1), dtype=wp.float32),
                wp.from_torch(out_gj.view(-1), dtype=wp.float32),
                requires_grad,
                b,
                h,
                dof,
            ],
            device=wp_device,
            stream=wp.stream_from_torch(vel.device),
        )
        ctx.save_for_backward(out_gp, out_gv, out_ga, out_gj)

        return out_cost

    @staticmethod
    def backward(ctx, grad_out_cost):
        (
            p_grad,
            v_grad,
            a_grad,
            j_grad,
        ) = ctx.saved_tensors
        v_g = None
        a_g = None
        p_g = None
        j_g = None
        if ctx.needs_input_grad[0]:
            p_g = p_grad  # * grad_out_cost#.unsqueeze(-1)
        if ctx.needs_input_grad[1]:
            v_g = v_grad  # * grad_out_cost#.unsqueeze(-1)
        if ctx.needs_input_grad[2]:
            a_g = a_grad  # * grad_out_cost#.unsqueeze(-1)
        if ctx.needs_input_grad[3]:
            j_g = j_grad  # * grad_out_cost#.unsqueeze(-1)
        return (
            p_g,
            v_g,
            a_g,
            j_g,
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
        )


class WarpBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pos,
        vel,
        acc,
        jerk,
        retract_config,
        retract_idx,
        p_b,
        v_b,
        a_b,
        j_b,
        weight,
        activation_distance,
        null_space_weight,
        vec_weight,
        out_cost,
        out_gp,
        out_gv,
        out_ga,
        out_gj,
    ):
        wp_device = wp.device_from_torch(vel.device)
        b, h, dof = vel.shape
        requires_grad = pos.requires_grad
        wp.launch(
            kernel=forward_bound_warp,
            dim=b * h * dof,
            inputs=[
                wp.from_torch(pos.detach().view(-1), dtype=wp.float32),
                wp.from_torch(vel.detach().view(-1), dtype=wp.float32),
                wp.from_torch(acc.detach().view(-1), dtype=wp.float32),
                wp.from_torch(jerk.detach().view(-1), dtype=wp.float32),
                wp.from_torch(retract_config.detach().view(-1), dtype=wp.float32),
                wp.from_torch(retract_idx.detach().view(-1), dtype=wp.int32),
                wp.from_torch(p_b.view(-1), dtype=wp.float32),
                wp.from_torch(v_b.view(-1), dtype=wp.float32),
                wp.from_torch(a_b.view(-1), dtype=wp.float32),
                wp.from_torch(j_b.view(-1), dtype=wp.float32),
                wp.from_torch(weight, dtype=wp.float32),
                wp.from_torch(activation_distance, dtype=wp.float32),
                wp.from_torch(null_space_weight.view(-1), dtype=wp.float32),
                wp.from_torch(vec_weight.view(-1), dtype=wp.float32),
                wp.from_torch(out_cost.view(-1), dtype=wp.float32),
                wp.from_torch(out_gp.view(-1), dtype=wp.float32),
                wp.from_torch(out_gv.view(-1), dtype=wp.float32),
                wp.from_torch(out_ga.view(-1), dtype=wp.float32),
                wp.from_torch(out_gj.view(-1), dtype=wp.float32),
                requires_grad,
                b,
                h,
                dof,
            ],
            device=wp_device,
            stream=wp.stream_from_torch(vel.device),
        )
        ctx.save_for_backward(out_gp, out_gv, out_ga, out_gj)
        # out_c = out_cost
        # out_c = torch.linalg.norm(out_cost, dim=-1)

        out_c = torch.sum(out_cost, dim=-1)
        return out_c

    @staticmethod
    def backward(ctx, grad_out_cost):
        (
            p_grad,
            v_grad,
            a_grad,
            j_grad,
        ) = ctx.saved_tensors
        v_g = None
        a_g = None
        p_g = None
        j_g = None
        if ctx.needs_input_grad[0]:
            p_g = p_grad  # * grad_out_cost#.unsqueeze(-1)
        if ctx.needs_input_grad[1]:
            v_g = v_grad  # * grad_out_cost#.unsqueeze(-1)
        if ctx.needs_input_grad[2]:
            a_g = a_grad  # * grad_out_cost#.unsqueeze(-1)
        if ctx.needs_input_grad[3]:
            j_g = j_grad
        return (
            p_g,
            v_g,
            a_g,
            j_g,
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


# create a bound cost tensor:
class WarpBoundPosFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pos,
        retract_config,
        retract_idx,
        p_l,
        weight,
        activation_distance,
        null_space_weight,
        vec_weight,
        out_cost,
        out_gp,
        return_loss=False,
    ):
        wp_device = wp.device_from_torch(pos.device)
        b, h, dof = pos.shape
        requires_grad = pos.requires_grad
        wp.launch(
            kernel=forward_bound_pos_warp,
            dim=b * h * dof,
            inputs=[
                wp.from_torch(pos.detach().view(-1), dtype=wp.float32),
                wp.from_torch(retract_config.detach().view(-1), dtype=wp.float32),
                wp.from_torch(retract_idx.detach().view(-1), dtype=wp.int32),
                wp.from_torch(p_l.view(-1), dtype=wp.float32),
                wp.from_torch(weight, dtype=wp.float32),
                wp.from_torch(activation_distance, dtype=wp.float32),
                wp.from_torch(null_space_weight.view(-1), dtype=wp.float32),
                wp.from_torch(vec_weight.view(-1), dtype=wp.float32),
                wp.from_torch(out_cost.view(-1), dtype=wp.float32),
                wp.from_torch(out_gp.view(-1), dtype=wp.float32),
                requires_grad,
                b,
                h,
                dof,
            ],
            device=wp_device,
            stream=wp.stream_from_torch(pos.device),
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(out_gp)
        cost = torch.sum(out_cost, dim=-1)
        return cost

    @staticmethod
    def backward(ctx, grad_out_cost):
        (p_grad,) = ctx.saved_tensors

        p_g = None
        if ctx.needs_input_grad[0]:
            p_g = p_grad
            if ctx.return_loss:
                p_g = p_grad * grad_out_cost.unsqueeze(-1)

        return p_g, None, None, None, None, None, None, None, None, None, None


# create a bound cost tensor:
class WarpBoundPosL2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pos,
        retract_config,
        retract_idx,
        p_l,
        weight,
        activation_distance,
        null_space_weight,
        vec_weight,
        out_cost,
        out_gp,
        warp_function,
        return_loss=False,
    ):
        wp_device = wp.device_from_torch(pos.device)
        b, h, dof = pos.shape
        requires_grad = pos.requires_grad
        wp.launch(
            kernel=warp_function,
            dim=b * h,
            inputs=[
                wp.from_torch(pos.detach().view(-1), dtype=wp.float32),
                wp.from_torch(retract_config.detach().view(-1), dtype=wp.float32),
                wp.from_torch(retract_idx.detach().view(-1), dtype=wp.int32),
                wp.from_torch(p_l.view(-1), dtype=wp.float32),
                wp.from_torch(weight, dtype=wp.float32),
                wp.from_torch(activation_distance, dtype=wp.float32),
                wp.from_torch(null_space_weight.view(-1), dtype=wp.float32),
                wp.from_torch(vec_weight.view(-1), dtype=wp.float32),
                wp.from_torch(out_cost.view(-1), dtype=wp.float32),
                wp.from_torch(out_gp.view(-1), dtype=wp.float32),
                requires_grad,
                b,
                h,
                dof,
            ],
            device=wp_device,
            stream=wp.stream_from_torch(pos.device),
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(out_gp)
        # cost = torch.sum(out_cost, dim=-1)
        return out_cost

    @staticmethod
    def backward(ctx, grad_out_cost):
        (p_grad,) = ctx.saved_tensors

        p_g = None
        if ctx.needs_input_grad[0]:
            p_g = p_grad  # * grad_out_cost.unsqueeze(-1)
            if ctx.return_loss:
                p_g = p_grad * grad_out_cost.unsqueeze(-1)
        return p_g, None, None, None, None, None, None, None, None, None, None, None


@wp.kernel(enable_backward=False)
def forward_bound_pos_warp(
    pos: wp.array(dtype=wp.float32),
    retract_config: wp.array(dtype=wp.float32),
    retract_idx: wp.array(dtype=wp.int32),
    p_b: wp.array(dtype=wp.float32),
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    null_weight: wp.array(dtype=wp.float32),
    vec_weight: wp.array(dtype=wp.float32),
    out_cost: wp.array(dtype=wp.float32),
    out_grad_p: wp.array(dtype=wp.float32),
    write_grad: wp.uint8,  # this should be a bool
    batch_size: wp.int32,
    horizon: wp.int32,
    dof: wp.int32,
):
    tid = wp.tid()
    # initialize variables:
    b_id = wp.int32(0)
    h_id = wp.int32(0)
    d_id = wp.int32(0)
    b_addrs = int(0)

    w = wp.float32(0.0)
    c_p = wp.float32(0.0)
    g_p = wp.float32(0.0)
    c_total = wp.float32(0.0)

    # we launch batch * horizon * dof kernels
    b_id = tid / (horizon * dof)
    h_id = (tid - (b_id * horizon * dof)) / dof
    d_id = tid - (b_id * horizon * dof + h_id * dof)
    if b_id >= batch_size or h_id >= horizon or d_id >= dof:
        return

    # read weights:
    eta_p = activation_distance[0]
    w = weight[0]

    n_w = wp.float32(0.0)
    n_w = null_weight[0]
    target_p = wp.float32(0.0)
    target_id = wp.int32(0.0)
    if n_w > 0.0:
        n_w *= vec_weight[d_id]
        target_id = retract_idx[b_id]
        target_id = target_id * dof + d_id
        target_p = retract_config[target_id]
    p_l = p_b[d_id]
    p_u = p_b[dof + d_id]

    p_range = p_u - p_l
    eta_p = eta_p * (p_range)
    p_l += eta_p
    p_u -= eta_p
    if p_range < 1.0:
        w = (1.0 / p_range) * w

    # compute cost:
    b_addrs = b_id * horizon * dof + h_id * dof + d_id

    # read buffers:

    c_p = pos[b_addrs]

    if n_w > 0.0:
        error = c_p - target_p
        c_total = n_w * error * error
        g_p = 2.0 * n_w * error

    # bound cost:

    delta = 0.0
    if c_p < p_l:
        delta = c_p - p_l
    elif c_p > p_u:
        delta = c_p - p_u

    c_total += w * delta * delta
    g_p += 2.0 * w * delta
    out_cost[b_addrs] = c_total

    # compute gradient
    if write_grad == 1:
        out_grad_p[b_addrs] = g_p


@wp.kernel(enable_backward=False)
def forward_bound_warp(
    pos: wp.array(dtype=wp.float32),
    vel: wp.array(dtype=wp.float32),
    acc: wp.array(dtype=wp.float32),
    jerk: wp.array(dtype=wp.float32),
    retract_config: wp.array(dtype=wp.float32),
    retract_idx: wp.array(dtype=wp.int32),
    p_b: wp.array(dtype=wp.float32),
    v_b: wp.array(dtype=wp.float32),
    a_b: wp.array(dtype=wp.float32),
    j_b: wp.array(dtype=wp.float32),
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    null_weight: wp.array(dtype=wp.float32),
    vec_weight: wp.array(dtype=wp.float32),
    out_cost: wp.array(dtype=wp.float32),
    out_grad_p: wp.array(dtype=wp.float32),
    out_grad_v: wp.array(dtype=wp.float32),
    out_grad_a: wp.array(dtype=wp.float32),
    out_grad_j: wp.array(dtype=wp.float32),
    write_grad: wp.uint8,  # this should be a bool
    batch_size: wp.int32,
    horizon: wp.int32,
    dof: wp.int32,
):
    tid = wp.tid()
    # initialize variables:
    b_id = wp.int32(0)
    h_id = wp.int32(0)
    d_id = wp.int32(0)
    b_addrs = int(0)

    w = wp.float32(0.0)
    c_v = wp.float32(0.0)
    c_a = wp.float32(0.0)
    c_p = wp.float32(0.0)
    g_p = wp.float32(0.0)
    g_v = wp.float32(0.0)
    g_a = wp.float32(0.0)
    b_wv = float(0.0)
    b_wa = float(0.0)
    b_wj = float(0.0)
    c_total = wp.float32(0.0)

    # we launch batch * horizon * dof kernels
    b_id = tid / (horizon * dof)
    h_id = (tid - (b_id * horizon * dof)) / dof
    d_id = tid - (b_id * horizon * dof + h_id * dof)
    if b_id >= batch_size or h_id >= horizon or d_id >= dof:
        return
    n_w = wp.float32(0.0)
    n_w = null_weight[0]
    target_p = wp.float32(0.0)
    target_id = wp.int32(0.0)
    if n_w > 0.0:
        n_w *= vec_weight[d_id]
        target_id = retract_idx[b_id]
        target_id = target_id * dof + d_id
        target_p = retract_config[target_id]

    # read weights:
    w = weight[0]
    b_wv = weight[1]
    b_wa = weight[2]
    b_wj = weight[3]

    # compute cost:
    b_addrs = b_id * horizon * dof + h_id * dof + d_id

    # read buffers:
    c_v = vel[b_addrs]
    c_a = acc[b_addrs]
    c_p = pos[b_addrs]

    # if w_j > 0.0:
    eta_p = activation_distance[0]
    eta_v = activation_distance[1]
    eta_a = activation_distance[2]
    eta_j = activation_distance[3]

    c_j = jerk[b_addrs]

    p_l = p_b[d_id]
    p_u = p_b[dof + d_id]
    p_range = p_u - p_l
    eta_p = eta_p * (p_range)
    p_l += eta_p
    p_u -= eta_p

    v_l = v_b[d_id]
    v_u = v_b[dof + d_id]
    v_range = v_u - v_l
    eta_v = eta_v * (v_range)
    v_l += eta_v
    v_u -= eta_v

    a_l = a_b[d_id]
    a_u = a_b[dof + d_id]
    a_range = a_u - a_l
    eta_a = eta_a * (a_range)
    a_l += eta_a
    a_u -= eta_a

    j_l = j_b[d_id]
    j_u = j_b[dof + d_id]
    j_range = j_u - j_l
    eta_j = eta_j * (j_range)
    j_l += eta_j
    j_u -= eta_j

    delta = float(0.0)
    if n_w > 0.0:
        error = c_p - target_p
        c_total = n_w * error * error
        g_p = 2.0 * n_w * error

    # bound cost:
    delta = 0.0

    if c_p < p_l:
        delta = c_p - p_l
    elif c_p > p_u:
        delta = c_p - p_u

    c_total += w * delta * delta
    g_p += 2.0 * w * delta

    # bound cost:
    delta = 0.0
    if c_v < v_l:
        delta = c_v - v_l
    elif c_v > v_u:
        delta = c_v - v_u

    c_total += b_wv * delta * delta
    g_v = 2.0 * b_wv * delta

    delta = 0.0

    if c_a < a_l:
        delta = c_a - a_l
    elif c_a > a_u:
        delta = c_a - a_u

    c_total += b_wa * delta * delta
    g_a = b_wa * 2.0 * delta

    delta = 0.0
    if c_j < j_l:
        delta = c_j - j_l
    elif c_j > j_u:
        delta = c_j - j_u

    c_total += b_wj * delta * delta
    g_j = b_wj * delta * 2.0

    out_cost[b_addrs] = c_total

    # compute gradient
    if write_grad == 1:
        out_grad_p[b_addrs] = g_p
        out_grad_v[b_addrs] = g_v
        out_grad_a[b_addrs] = g_a
        out_grad_j[b_addrs] = g_j


@wp.kernel(enable_backward=False)
def forward_bound_smooth_warp(
    pos: wp.array(dtype=wp.float32),
    vel: wp.array(dtype=wp.float32),
    acc: wp.array(dtype=wp.float32),
    jerk: wp.array(dtype=wp.float32),
    retract_config: wp.array(dtype=wp.float32),
    retract_idx: wp.array(dtype=wp.int32),
    p_b: wp.array(dtype=wp.float32),
    v_b: wp.array(dtype=wp.float32),
    a_b: wp.array(dtype=wp.float32),
    j_b: wp.array(dtype=wp.float32),
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    smooth_weight: wp.array(dtype=wp.float32),
    cspace_weight: wp.array(dtype=wp.float32),
    null_weight: wp.array(dtype=wp.float32),
    vec_weight: wp.array(dtype=wp.float32),
    run_weight_vel: wp.array(dtype=wp.float32),
    run_weight_acc: wp.array(dtype=wp.float32),
    run_weight_jerk: wp.array(dtype=wp.float32),
    out_cost: wp.array(dtype=wp.float32),
    out_grad_p: wp.array(dtype=wp.float32),
    out_grad_v: wp.array(dtype=wp.float32),
    out_grad_a: wp.array(dtype=wp.float32),
    out_grad_j: wp.array(dtype=wp.float32),
    write_grad: wp.uint8,  # this should be a bool
    batch_size: wp.int32,
    horizon: wp.int32,
    dof: wp.int32,
):
    tid = wp.tid()
    # initialize variables:
    b_id = wp.int32(0)
    h_id = wp.int32(0)
    d_id = wp.int32(0)
    b_addrs = int(0)

    w = wp.float32(0.0)
    b_wv = float(0.0)
    b_wa = float(0.0)
    b_wj = float(0.0)
    cspace_w = wp.float32(0.0)
    c_p = wp.float32(0.0)
    c_v = wp.float32(0.0)
    c_a = wp.float32(0.0)
    c_j = wp.float32(0.0)
    g_p = wp.float32(0.0)
    g_v = wp.float32(0.0)
    g_a = wp.float32(0.0)
    g_j = wp.float32(0.0)
    r_wv = wp.float32(0.0)
    r_wa = wp.float32(0.0)
    r_wj = wp.float32(0.0)
    alpha_v = wp.float32(2.0)

    w_v = wp.float32(0.0)
    w_a = wp.float32(0.0)
    w_j = wp.float32(0.0)

    s_v = wp.float32(0.0)
    s_a = wp.float32(0.0)
    s_j = wp.float32(0.0)

    c_total = wp.float32(0.0)

    # we launch batch * horizon * dof kernels
    b_id = tid / (horizon * dof)
    h_id = (tid - (b_id * horizon * dof)) / dof
    d_id = tid - (b_id * horizon * dof + h_id * dof)
    if b_id >= batch_size or h_id >= horizon or d_id >= dof:
        return

    n_w = wp.float32(0.0)
    n_w = null_weight[0]
    target_p = wp.float32(0.0)
    target_id = wp.int32(0.0)
    if n_w > 0.0:
        n_w *= vec_weight[d_id]
        target_id = retract_idx[b_id]
        target_id = target_id * dof + d_id
        target_p = retract_config[target_id]
    # read weights:
    w = weight[0]
    b_wv = weight[1]
    b_wa = weight[2]
    b_wj = weight[3]
    cspace_w = cspace_weight[d_id]
    r_wv = run_weight_vel[h_id]
    r_wa = run_weight_acc[h_id]

    w_v = smooth_weight[0]
    w_a = smooth_weight[1]
    w_j = smooth_weight[2]
    alpha_v = smooth_weight[3]

    # scale all smooth weights by cspace weight:
    if r_wv < 1.0:
        r_wv *= cspace_w
        w_v *= cspace_w

    r_wa *= cspace_w
    r_wj *= cspace_w
    w_a *= cspace_w
    w_j *= cspace_w

    # compute cost:
    b_addrs = b_id * horizon * dof + h_id * dof + d_id

    # read buffers:
    c_v = vel[b_addrs]
    c_a = acc[b_addrs]
    c_p = pos[b_addrs]
    # if w_j > 0.0:
    eta_p = activation_distance[0]
    eta_v = activation_distance[1]
    eta_a = activation_distance[2]
    eta_j = activation_distance[3]

    r_wj = run_weight_jerk[h_id]
    c_j = jerk[b_addrs]

    p_l = p_b[d_id]
    p_u = p_b[dof + d_id]
    p_range = p_u - p_l
    eta_p = eta_p * (p_range)
    p_l += eta_p
    p_u -= eta_p

    v_l = v_b[d_id]
    v_u = v_b[dof + d_id]
    v_range = v_u - v_l
    eta_v = eta_v * (v_range)
    v_l += eta_v
    v_u -= eta_v

    a_l = a_b[d_id]
    a_u = a_b[dof + d_id]
    a_range = a_u - a_l
    eta_a = eta_a * (a_range)
    a_l += eta_a
    a_u -= eta_a

    j_l = j_b[d_id]
    j_u = j_b[dof + d_id]
    j_range = j_u - j_l
    eta_j = eta_j * (j_range)
    j_l += eta_j
    j_u -= eta_j

    delta = float(0.0)

    if p_range < 1.0:
        w = w * (1.0 / (p_range * p_range))

    if v_range < 1.0:
        b_wv = b_wv * (1.0 / (v_range * v_range))
    if a_range < 1.0:
        b_wa = b_wa * (1.0 / (a_range * a_range))
        w_a = w_a * (1.0 / (a_range * a_range))
    if j_range < 1.0:
        b_wj = b_wj * (1.0 / (j_range * j_range))
        w_j = w_j * (1.0 / (j_range * j_range))
    # position:
    if n_w > 0.0:
        error = c_p - target_p
        c_total = n_w * error * error
        g_p = 2.0 * n_w * error

    # bound cost:
    # bound cost:
    delta = 0.0
    if c_p < p_l:
        delta = c_p - p_l
    elif c_p > p_u:
        delta = c_p - p_u

    c_total += w * delta * delta
    g_p += 2.0 * w * delta

    # bound cost:
    delta = 0.0
    if c_v < v_l:
        delta = c_v - v_l
    elif c_v > v_u:
        delta = c_v - v_u

    c_total += b_wv * delta * delta
    g_v = 2.0 * b_wv * delta

    delta = 0.0

    if c_a < a_l:
        delta = c_a - a_l
    elif c_a > a_u:
        delta = c_a - a_u

    c_total += b_wa * delta * delta
    g_a = b_wa * 2.0 * delta

    delta = 0.0
    if c_j < j_l:
        delta = c_j - j_l
    elif c_j > j_u:
        delta = c_j - j_u

    c_total += b_wj * delta * delta
    g_j = b_wj * delta * 2.0

    # do l2 regularization for velocity:
    if r_wv < 1.0:
        s_v = w_v * r_wv * c_v * c_v
        g_v += 2.0 * r_wv * w_v * c_v
    else:
        s_v = w_v * r_wv * wp.log2(wp.cosh(alpha_v * c_v))
        g_v += w_v * r_wv * alpha_v * wp.sinh(alpha_v * c_v) / wp.cosh(alpha_v * c_v)

    s_a = w_a * (r_wa) * c_a * c_a
    g_a += 2.0 * w_a * (r_wa) * c_a
    s_j = w_j * r_wj * c_j * c_j
    g_j += 2.0 * w_j * r_wj * c_j

    c_total += s_v + s_a + s_j

    out_cost[b_addrs] = c_total

    # compute gradient
    if write_grad == 1:
        out_grad_p[b_addrs] = g_p
        out_grad_v[b_addrs] = g_v
        out_grad_a[b_addrs] = g_a
        out_grad_j[b_addrs] = g_j


@get_cache_fn_decorator()
def make_bound_pos_smooth_kernel(dof_template: int):
    def forward_bound_smooth_loop_warp(
        pos: wp.array(dtype=wp.float32),
        vel: wp.array(dtype=wp.float32),
        acc: wp.array(dtype=wp.float32),
        jerk: wp.array(dtype=wp.float32),
        retract_config: wp.array(dtype=wp.float32),
        retract_idx: wp.array(dtype=wp.int32),
        p_b: wp.array(dtype=wp.float32),
        v_b: wp.array(dtype=wp.float32),
        a_b: wp.array(dtype=wp.float32),
        j_b: wp.array(dtype=wp.float32),
        weight: wp.array(dtype=wp.float32),
        activation_distance: wp.array(dtype=wp.float32),
        smooth_weight: wp.array(dtype=wp.float32),
        cspace_weight: wp.array(dtype=wp.float32),
        null_weight: wp.array(dtype=wp.float32),
        vec_weight: wp.array(dtype=wp.float32),
        run_weight_vel: wp.array(dtype=wp.float32),
        run_weight_acc: wp.array(dtype=wp.float32),
        run_weight_jerk: wp.array(dtype=wp.float32),
        out_cost: wp.array(dtype=wp.float32),
        out_grad_p: wp.array(dtype=wp.float32),
        out_grad_v: wp.array(dtype=wp.float32),
        out_grad_a: wp.array(dtype=wp.float32),
        out_grad_j: wp.array(dtype=wp.float32),
        write_grad: wp.uint8,  # this should be a bool
        batch_size: wp.int32,
        horizon: wp.int32,
        dof: wp.int32,
    ):
        tid = wp.tid()
        # initialize variables:
        b_id = wp.int32(0)
        h_id = wp.int32(0)
        b_addrs = int(0)

        w = wp.float32(0.0)
        b_wv = float(0.0)
        b_wa = float(0.0)
        b_wj = float(0.0)

        r_wa = wp.float32(0.0)
        r_wj = wp.float32(0.0)

        w_a = wp.float32(0.0)
        w_j = wp.float32(0.0)

        s_a = wp.float32(0.0)
        s_j = wp.float32(0.0)

        c_total = wp.float32(0.0)

        # we launch batch * horizon * dof kernels
        b_id = tid / (horizon)
        h_id = tid - (b_id * horizon)
        if b_id >= batch_size or h_id >= horizon:
            return

        n_w = wp.float32(0.0)
        n_w = null_weight[0]

        vec_weight_local = wp.vector(dtype=wp.float32, length=dof_template)
        cspace_weight_local = wp.vector(dtype=wp.float32, length=dof_template)
        p_l = wp.vector(dtype=wp.float32, length=dof_template)
        p_u = wp.vector(dtype=wp.float32, length=dof_template)
        v_l = wp.vector(dtype=wp.float32, length=dof_template)
        v_u = wp.vector(dtype=wp.float32, length=dof_template)
        a_l = wp.vector(dtype=wp.float32, length=dof_template)
        a_u = wp.vector(dtype=wp.float32, length=dof_template)
        j_l = wp.vector(dtype=wp.float32, length=dof_template)
        j_u = wp.vector(dtype=wp.float32, length=dof_template)
        g_p = wp.vector(dtype=wp.float32, length=dof_template)
        g_v = wp.vector(dtype=wp.float32, length=dof_template)
        g_a = wp.vector(dtype=wp.float32, length=dof_template)
        g_j = wp.vector(dtype=wp.float32, length=dof_template)
        target_id = wp.int32(0.0)
        target_id = retract_idx[b_id]
        current_p = wp.vector(dtype=wp.float32, length=dof_template)
        current_v = wp.vector(dtype=wp.float32, length=dof_template)
        current_a = wp.vector(dtype=wp.float32, length=dof_template)
        current_j = wp.vector(dtype=wp.float32, length=dof_template)

        target_p = wp.vector(dtype=wp.float32, length=dof_template)

        b_addrs = b_id * horizon * dof + h_id * dof

        for i in range(dof_template):
            vec_weight_local[i] = vec_weight[i]
            target_p[i] = retract_config[target_id * dof + i]
            current_p[i] = pos[b_addrs + i]
            current_v[i] = vel[b_addrs + i]
            current_a[i] = acc[b_addrs + i]
            current_j[i] = jerk[b_addrs + i]
            p_l[i] = p_b[i]
            p_u[i] = p_b[dof + i]
            v_l[i] = v_b[i]
            v_u[i] = v_b[dof + i]
            a_l[i] = a_b[i]
            a_u[i] = a_b[dof + i]
            j_l[i] = j_b[i]
            j_u[i] = j_b[dof + i]
            cspace_weight_local[i] = cspace_weight[i]

        # read weights:
        w = weight[0]
        b_wv = weight[1]
        b_wa = weight[2]
        b_wj = weight[3]
        r_wa = run_weight_acc[h_id]
        r_wj = run_weight_jerk[h_id]

        w_a = smooth_weight[1]
        w_j = smooth_weight[2]

        # compute cost:

        # read buffers:

        # if w_j > 0.0:
        eta_p = activation_distance[0]
        eta_v = activation_distance[1]
        eta_a = activation_distance[2]
        eta_j = activation_distance[3]

        p_range = p_u - p_l
        p_l = p_l + (p_range) * eta_p
        p_u = p_u - (p_range) * eta_p
        v_l = v_l + (v_u - v_l) * eta_v
        v_u = v_u - (v_u - v_l) * eta_v
        a_l = a_l + (a_u - a_l) * eta_a
        a_u = a_u - (a_u - a_l) * eta_a
        j_l = j_l + (j_u - j_l) * eta_j
        j_u = j_u - (j_u - j_l) * eta_j

        # position:
        if n_w > 0.0:
            error = wp.cw_mul(vec_weight_local, current_p - target_p)
            error_length = wp.length(error)
            c_total = n_w * error_length
            if error_length > 0.0:
                g_p = n_w * error / error_length

        # bound cost:
        # bound cost:
        bound_delta_p = wp.vector(dtype=wp.float32, length=dof_template)
        bound_delta_v = wp.vector(dtype=wp.float32, length=dof_template)
        bound_delta_a = wp.vector(dtype=wp.float32, length=dof_template)
        bound_delta_j = wp.vector(dtype=wp.float32, length=dof_template)
        for i in range(dof_template):
            if current_p[i] < p_l[i]:
                bound_delta_p[i] = current_p[i] - p_l[i]
            elif current_p[i] > p_u[i]:
                bound_delta_p[i] = current_p[i] - p_u[i]
            if current_v[i] < v_l[i]:
                bound_delta_v[i] = current_v[i] - v_l[i]
            elif current_v[i] > v_u[i]:
                bound_delta_v[i] = current_v[i] - v_u[i]
            if current_a[i] < a_l[i]:
                bound_delta_a[i] = current_a[i] - a_l[i]
            elif current_a[i] > a_u[i]:
                bound_delta_a[i] = current_a[i] - a_u[i]
            if current_j[i] < j_l[i]:
                bound_delta_j[i] = current_j[i] - j_l[i]
            elif current_j[i] > j_u[i]:
                bound_delta_j[i] = current_j[i] - j_u[i]
        delta_p = wp.length(bound_delta_p)
        if delta_p > 0.0:
            g_p += w * bound_delta_p / delta_p
            c_total += w * delta_p
        delta_v = wp.length(bound_delta_v)
        if delta_v > 0.0:
            g_v = b_wv * bound_delta_v / delta_v
            c_total += b_wv * delta_v
        delta_a = wp.length(bound_delta_a)
        if delta_a > 0.0:
            g_a = b_wa * bound_delta_a / delta_a
            c_total += b_wa * delta_a
        delta_j = wp.length(bound_delta_j)
        if delta_j > 0.0:
            g_j = b_wj * bound_delta_j / delta_j
            c_total += b_wj * delta_j

        delta_acc = wp.cw_mul(cspace_weight_local, current_a)
        # delta_acc = wp.cw_div(delta_acc, a_u - a_l)
        acc_length = wp.length_sq(delta_acc)
        s_a = w_a * r_wa * acc_length
        if acc_length > 0.0:
            g_a += 2.0 * w_a * r_wa * delta_acc  # / acc_length

        delta_jerk = wp.cw_mul(cspace_weight_local, current_j)
        # delta_jerk = wp.cw_div(delta_jerk, j_u - j_l)
        jerk_length = wp.length_sq(delta_jerk)
        s_j = w_j * r_wj * jerk_length
        if jerk_length > 0.0:
            g_j += 2.0 * w_j * r_wj * delta_jerk  # / jerk_length

        c_total += s_a + s_j

        out_cost[b_id * horizon + h_id] = c_total

        # compute gradient
        if write_grad == 1:
            b_addrs = b_id * horizon * dof + h_id * dof
            for i in range(dof_template):
                out_grad_p[b_addrs + i] = g_p[i]
                out_grad_v[b_addrs + i] = g_v[i]
                out_grad_a[b_addrs + i] = g_a[i]
                out_grad_j[b_addrs + i] = g_j[i]

    module = wp.get_module(forward_bound_smooth_loop_warp.__module__)
    key = "forward_bound_smooth_loop_warp_" + str(dof_template)
    if key in module.kernels:
        new_kernel = module.kernels[key]
    else:
        new_kernel = wp.Kernel(forward_bound_smooth_loop_warp, key=key, module=module)

    return new_kernel


@get_cache_fn_decorator()
def make_bound_pos_kernel(dof_template: int):

    def forward_bound_pos_loop_warp(
        pos: wp.array(dtype=wp.float32),
        retract_config: wp.array(dtype=wp.float32),
        retract_idx: wp.array(dtype=wp.int32),
        p_b: wp.array(dtype=wp.float32),
        weight: wp.array(dtype=wp.float32),
        activation_distance: wp.array(dtype=wp.float32),
        null_weight: wp.array(dtype=wp.float32),
        vec_weight: wp.array(dtype=wp.float32),
        out_cost: wp.array(dtype=wp.float32),
        out_grad_p: wp.array(dtype=wp.float32),
        write_grad: wp.uint8,  # this should be a bool
        batch_size: wp.int32,
        horizon: wp.int32,
        dof: wp.int32,
    ):
        tid = wp.tid()
        # initialize variables:
        b_id = wp.int32(0)
        h_id = wp.int32(0)
        b_addrs = int(0)

        w = wp.float32(0.0)
        c_total = wp.float32(0.0)

        # we launch batch * horizon * dof kernels
        b_id = tid / (horizon)
        h_id = tid - (b_id * horizon)
        if b_id >= batch_size or h_id >= horizon:
            return

        # read weights:
        eta_p = activation_distance[0]
        w = weight[0]

        n_w = wp.float32(0.0)
        n_w = null_weight[0]
        target_id = wp.int32(0.0)
        vec_weight_local = wp.vector(dtype=wp.float32, length=dof_template)
        p_l = wp.vector(dtype=wp.float32, length=dof_template)
        p_u = wp.vector(dtype=wp.float32, length=dof_template)
        bound_delta = wp.vector(dtype=wp.float32, length=dof_template)
        target_p = wp.vector(dtype=wp.float32, length=dof_template)
        current_p = wp.vector(dtype=wp.float32, length=dof_template)
        g_p = wp.vector(dtype=wp.float32, length=dof_template)

        target_id = retract_idx[b_id]
        b_addrs = b_id * horizon * dof + h_id * dof

        for i in range(dof_template):
            vec_weight_local[i] = vec_weight[i]
            target_p[i] = retract_config[target_id * dof + i]
            current_p[i] = pos[b_addrs + i]
            p_l[i] = p_b[i]
            p_u[i] = p_b[dof + i]

        p_range = p_u - p_l
        eta_percent = eta_p * (p_range)
        p_l += eta_percent
        p_u -= eta_percent

        # compute retract cost:

        if n_w > 0.0:
            error = wp.cw_mul(vec_weight_local, current_p - target_p)
            error_length = wp.length(error)
            c_total = n_w * error_length
            if error_length > 0.0:
                g_p = n_w * error / error_length

        # bound cost:

        for i in range(dof_template):
            if current_p[i] < p_l[i]:
                bound_delta[i] = current_p[i] - p_l[i]
            elif current_p[i] > p_u[i]:
                bound_delta[i] = current_p[i] - p_u[i]

        delta = wp.length(bound_delta)
        if delta > 0.0:
            g_p += w * bound_delta / delta
            c_total += w * delta

        out_cost[b_id * horizon + h_id] = c_total

        # compute gradient
        if write_grad == 1:
            b_addrs = b_id * horizon * dof + h_id * dof
            for i in range(dof_template):
                out_grad_p[b_addrs + i] = g_p[i]

    module = wp.get_module(forward_bound_pos_loop_warp.__module__)
    key = "bound_pos_loop_warp_" + str(dof_template)
    if key in module.kernels:
        new_kernel = module.kernels[key]
    else:
        new_kernel = wp.Kernel(forward_bound_pos_loop_warp, key=key, module=module)
    return new_kernel
