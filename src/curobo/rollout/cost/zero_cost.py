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
from curobo.util.torch_utils import get_torch_jit_decorator

# Local Folder
from .cost_base import CostBase


@get_torch_jit_decorator()
def squared_sum(cost: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # return  weight * torch.square(torch.linalg.norm(cost, dim=-1, ord=1))
    # return  weight * torch.sum(torch.square(cost), dim=-1)

    # return  torch.sum(torch.abs(cost) * weight, dim=-1)
    return torch.sum(torch.square(cost) * weight, dim=-1)


@get_torch_jit_decorator()
def run_squared_sum(
    cost: torch.Tensor, weight: torch.Tensor, run_weight: torch.Tensor
) -> torch.Tensor:
    # return torch.sum(torch.abs(cost)* weight * run_weight.unsqueeze(-1), dim=-1)
    ## below is smaller compute but more kernels
    return torch.sum(torch.square(cost) * weight * run_weight.unsqueeze(-1), dim=-1)

    # return torch.sum(torch.square(cost), dim=-1) * weight * run_weight


@get_torch_jit_decorator()
def backward_squared_sum(cost_vec, w):
    return 2.0 * w * cost_vec  # * g_out.unsqueeze(-1)
    # return   w * g_out.unsqueeze(-1)


@get_torch_jit_decorator()
def backward_run_squared_sum(cost_vec, w, r_w):
    return 2.0 * w * r_w.unsqueeze(-1) * cost_vec  # * g_out.unsqueeze(-1)
    # return   w * r_w.unsqueeze(-1) * cost_vec * g_out.unsqueeze(-1)


class SquaredSum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cost_vec,
        weight,
    ):
        cost = squared_sum(cost_vec, weight)
        ctx.save_for_backward(cost_vec, weight)
        return cost

    @staticmethod
    def backward(ctx, grad_out_cost):
        (cost_vec, w) = ctx.saved_tensors
        c_grad = None
        if ctx.needs_input_grad[0]:
            c_grad = backward_squared_sum(cost_vec, w)
        return c_grad, None


class RunSquaredSum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cost_vec,
        weight,
        run_weight,
    ):
        cost = run_squared_sum(cost_vec, weight, run_weight)
        ctx.save_for_backward(cost_vec, weight, run_weight)
        return cost

    @staticmethod
    def backward(ctx, grad_out_cost):
        (cost_vec, w, r_w) = ctx.saved_tensors
        c_grad = None
        if ctx.needs_input_grad[0]:
            c_grad = backward_run_squared_sum(cost_vec, w, r_w)
        return c_grad, None, None


class ZeroCost(CostBase):
    """Zero Cost"""

    def forward(self, x, goal_dist):
        err = x

        if self.max_value is not None:
            err = torch.nn.functional.relu(torch.abs(err) - self.max_value)

        if self.hinge_value is not None:
            err = torch.where(goal_dist <= self.hinge_value, err, self._z_scalar)  # soft hinge
        if self.threshold_value is not None:
            err = torch.where(err <= self.distance_threshold, self._z_scalar, err)
        if not self.terminal:  # or self.run_weight is not None:
            cost = SquaredSum.apply(err, self.weight)
        else:
            if self._run_weight_vec is None or self._run_weight_vec.shape[1] != err.shape[1]:
                self._run_weight_vec = torch.ones(
                    (1, err.shape[1]), device=self.tensor_args.device, dtype=self.tensor_args.dtype
                )
                self._run_weight_vec[:, 1:-1] *= self.run_weight
            cost = RunSquaredSum.apply(
                err, self.weight, self._run_weight_vec
            )  # cost * self._run_weight_vec

        return cost
