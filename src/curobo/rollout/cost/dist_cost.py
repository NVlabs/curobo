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
from typing import Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo.util.torch_utils import get_torch_jit_decorator
from curobo.util.warp import init_warp

# Local Folder
from .cost_base import CostBase, CostConfig

wp.set_module_options({"fast_math": False})


class DistType(Enum):
    L1 = 0
    L2 = 1
    SQUARED_L2 = 2


@dataclass
class DistCostConfig(CostConfig):
    dist_type: DistType = DistType.L2
    use_null_space: bool = False

    def __post_init__(self):
        return super().__post_init__()


@get_torch_jit_decorator()
def L2_DistCost_jit(vec_weight, disp_vec):
    return torch.norm(vec_weight * disp_vec, p=2, dim=-1, keepdim=False)


@get_torch_jit_decorator()
def fwd_SQL2_DistCost_jit(vec_weight, disp_vec):
    return torch.sum(torch.square(vec_weight * disp_vec), dim=-1, keepdim=False)


@get_torch_jit_decorator()
def fwd_L1_DistCost_jit(vec_weight, disp_vec):
    return torch.sum(torch.abs(vec_weight * disp_vec), dim=-1, keepdim=False)


@get_torch_jit_decorator()
def L2_DistCost_target_jit(vec_weight, g_vec, c_vec, weight):
    return torch.norm(weight * vec_weight * (g_vec - c_vec), p=2, dim=-1, keepdim=False)


@get_torch_jit_decorator()
def fwd_SQL2_DistCost_target_jit(vec_weight, g_vec, c_vec, weight):
    return torch.sum(torch.square(weight * vec_weight * (g_vec - c_vec)), dim=-1, keepdim=False)


@get_torch_jit_decorator()
def fwd_L1_DistCost_target_jit(vec_weight, g_vec, c_vec, weight):
    return torch.sum(torch.abs(weight * vec_weight * (g_vec - c_vec)), dim=-1, keepdim=False)


@wp.kernel
def forward_l2_warp(
    pos: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
    target_idx: wp.array(dtype=wp.int32),
    weight: wp.array(dtype=wp.float32),
    run_weight: wp.array(dtype=wp.float32),
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
    b_addrs = wp.int32(0)
    target_id = wp.int32(0)
    w = wp.float32(0.0)
    c_p = wp.float32(0.0)
    target_p = wp.float32(0.0)
    g_p = wp.float32(0.0)
    r_w = wp.float32(0.0)
    c_total = wp.float32(0.0)

    # we launch batch * horizon * dof kernels
    b_id = tid / (horizon * dof)
    h_id = (tid - (b_id * horizon * dof)) / dof
    d_id = tid - (b_id * horizon * dof + h_id * dof)

    if b_id >= batch_size or h_id >= horizon or d_id >= dof:
        return

    # read weights:
    w = weight[0]
    r_w = run_weight[h_id]
    w = r_w * w
    r_w = vec_weight[d_id]
    w = r_w * w
    if w == 0.0:
        return
    # compute cost:
    b_addrs = b_id * horizon * dof + h_id * dof + d_id

    # read buffers:

    c_p = pos[b_addrs]
    target_id = target_idx[b_id]
    target_id = target_id * dof + d_id
    target_p = target[target_id]
    error = c_p - target_p

    # if r_w >= 1.0 and w > 100.0:
    #    c_total = w * wp.log2(wp.cosh(10.0 * error))
    #    g_p = w * 10.0 * wp.sinh(10.0 * error) / (wp.cosh(10.0 * error))
    # else:
    c_total = w * error * error
    g_p = 2.0 * w * error

    out_cost[b_addrs] = c_total

    # compute gradient
    if write_grad == 1:
        out_grad_p[b_addrs] = g_p


# create a bound cost tensor:
class L2DistFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pos,
        target,
        target_idx,
        weight,
        run_weight,
        vec_weight,
        out_cost,
        out_cost_v,
        out_gp,
    ):
        wp_device = wp.device_from_torch(pos.device)
        b, h, dof = pos.shape
        requires_grad = pos.requires_grad
        wp.launch(
            kernel=forward_l2_warp,
            dim=b * h * dof,
            inputs=[
                wp.from_torch(pos.detach().reshape(-1), dtype=wp.float32),
                wp.from_torch(target.view(-1), dtype=wp.float32),
                wp.from_torch(target_idx.view(-1), dtype=wp.int32),
                wp.from_torch(weight, dtype=wp.float32),
                wp.from_torch(run_weight.view(-1), dtype=wp.float32),
                wp.from_torch(vec_weight.view(-1), dtype=wp.float32),
                wp.from_torch(out_cost_v.view(-1), dtype=wp.float32),
                wp.from_torch(out_gp.view(-1), dtype=wp.float32),
                requires_grad,
                b,
                h,
                dof,
            ],
            device=wp_device,
            stream=wp.stream_from_torch(pos.device),
        )

        cost = torch.sum(out_cost_v, dim=-1)
        ctx.save_for_backward(out_gp)
        return cost

    @staticmethod
    def backward(ctx, grad_out_cost):
        (p_grad,) = ctx.saved_tensors

        p_g = None
        if ctx.needs_input_grad[0]:
            p_g = p_grad
        return p_g, None, None, None, None, None, None, None, None


class DistCost(CostBase, DistCostConfig):
    def __init__(self, config: Optional[DistCostConfig] = None):
        if config is not None:
            DistCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self._init_post_config()
        init_warp()

    def _init_post_config(self):
        if self.vec_weight is not None:
            self.vec_weight = self.tensor_args.to_device(self.vec_weight)
            if not self.use_null_space:
                self.vec_weight = self.vec_weight * 0.0 + 1.0

    def update_batch_size(self, batch, horizon, dof):
        if self._batch_size != batch or self._horizon != horizon or self._dof != dof:
            self._out_cv_buffer = torch.zeros(
                (batch, horizon, dof), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            self._out_c_buffer = torch.zeros(
                (batch, horizon), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )

            self._out_g_buffer = torch.zeros(
                (batch, horizon, dof), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )

            self._batch_size = batch
            self._horizon = horizon
            self._dof = dof
        if self.vec_weight is None:
            self.vec_weight = torch.ones(
                (1, 1, self._dof), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )

    def forward(self, disp_vec, RETURN_GOAL_DIST=False):
        if self.dist_type == DistType.L2:
            # dist = torch.norm(disp_vec, p=2, dim=-1, keepdim=False)
            dist = L2_DistCost_jit(self.vec_weight, disp_vec)
        elif self.dist_type == DistType.SQUARED_L2:
            # cost = weight * (0.5 * torch.square(torch.norm(disp_vec, p=2, dim=-1)))
            # dist = torch.sum(torch.square(disp_vec), dim=-1, keepdim=False)
            dist = SQL2_DistCost_jit(self.vec_weight, disp_vec)
        elif self.dist_type == DistType.L1:
            # dist = torch.sum(torch.abs(disp_vec), dim=-1, keepdim=False)
            dist = L1_DistCost_jit(self.vec_weight, disp_vec)

        cost = self.weight * dist
        if self.terminal and self.run_weight is not None:
            if self._run_weight_vec is None or self._run_weight_vec.shape[1] != cost.shape[1]:
                self._run_weight_vec = torch.ones(
                    (1, cost.shape[1]), device=self.tensor_args.device, dtype=self.tensor_args.dtype
                )
                self._run_weight_vec[:, :-1] *= self.run_weight
        if RETURN_GOAL_DIST:
            return cost, dist
        return cost

    def forward_target(self, goal_vec, current_vec, RETURN_GOAL_DIST=False):
        if self.dist_type == DistType.L2:
            # dist = torch.norm(disp_vec, p=2, dim=-1, keepdim=False)
            cost = L2_DistCost_target_jit(self.vec_weight, goal_vec, current_vec, self.weight)
        elif self.dist_type == DistType.SQUARED_L2:
            # cost = weight * (0.5 * torch.square(torch.norm(disp_vec, p=2, dim=-1)))
            # dist = torch.sum(torch.square(disp_vec), dim=-1, keepdim=False)
            cost = fwd_SQL2_DistCost_target_jit(self.vec_weight, goal_vec, current_vec, self.weight)
        elif self.dist_type == DistType.L1:
            # dist = torch.sum(torch.abs(disp_vec), dim=-1, keepdim=False)
            cost = fwd_L1_DistCost_target_jit(self.vec_weight, goal_vec, current_vec, self.weight)
        dist = cost
        if self.terminal and self.run_weight is not None:
            if self._run_weight_vec is None or self._run_weight_vec.shape[1] != cost.shape[1]:
                self._run_weight_vec = torch.ones(
                    (1, cost.shape[1]), device=self.tensor_args.device, dtype=self.tensor_args.dtype
                )
                self._run_weight_vec[:, :-1] *= self.run_weight
            cost = self._run_weight_vec * dist
        if RETURN_GOAL_DIST:
            dist_scale = torch.nan_to_num(
                1.0 / torch.sqrt((self.weight * self._run_weight_vec)), 0.0
            )

            return cost, dist * dist_scale
        return cost

    def forward_target_idx(self, goal_vec, current_vec, goal_idx, RETURN_GOAL_DIST=False):
        b, h, dof = current_vec.shape
        self.update_batch_size(b, h, dof)

        if self.terminal and self.run_weight is not None:
            if self._run_weight_vec is None or self._run_weight_vec.shape[1] != h:
                self._run_weight_vec = torch.ones(
                    (1, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype
                )
                self._run_weight_vec[:, :-1] *= self.run_weight
        else:
            raise NotImplementedError("terminal flag needs to be set to true")
        if self.dist_type == DistType.L2:
            # print(goal_idx.shape, goal_vec.shape)
            cost = L2DistFunction.apply(
                current_vec,
                goal_vec,
                goal_idx,
                self.weight,
                self._run_weight_vec,
                self.vec_weight,
                self._out_c_buffer,
                self._out_cv_buffer,
                self._out_g_buffer,
            )

        else:
            raise NotImplementedError()
        if RETURN_GOAL_DIST:
            dist_scale = torch.nan_to_num(
                1.0 / torch.sqrt((self.weight * self._run_weight_vec)), 0.0
            )
            return cost, cost * dist_scale
        return cost
