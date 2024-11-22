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
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.curobolib.ls import update_best, wolfe_line_search
from curobo.opt.opt_base import Optimizer, OptimizerConfig
from curobo.rollout.dynamics_model.integration_utils import build_fd_matrix
from curobo.types.base import TensorDeviceType
from curobo.types.tensor import T_BDOF, T_BHDOF_float, T_BHValue_float, T_BValue_float, T_HDOF_float
from curobo.util.torch_utils import get_torch_jit_decorator


class LineSearchType(Enum):
    GREEDY = "greedy"
    ARMIJO = "armijo"
    WOLFE = "wolfe"
    STRONG_WOLFE = "strong_wolfe"
    APPROX_WOLFE = "approx_wolfe"


@dataclass
class NewtonOptConfig(OptimizerConfig):
    line_search_scale: List[int]
    cost_convergence: float
    cost_delta_threshold: float
    fixed_iters: bool
    inner_iters: int  # used for cuda graph
    line_search_type: LineSearchType
    use_cuda_line_search_kernel: bool
    use_cuda_update_best_kernel: bool
    min_iters: int
    step_scale: float
    last_best: float = 0
    use_temporal_smooth: bool = False
    cost_relative_threshold: float = 0.999
    fix_terminal_action: bool = False

    # use_update_best_kernel: bool
    # c_1: float
    # c_2: float
    def __post_init__(self):
        self.num_particles = len(self.line_search_scale)
        self.line_search_type = LineSearchType(self.line_search_type)
        if self.fixed_iters:
            self.cost_delta_threshold = 0.0001
            self.cost_relative_threshold = 1.0
        return super().__post_init__()


class NewtonOptBase(Optimizer, NewtonOptConfig):
    @profiler.record_function("newton_opt/init")
    def __init__(
        self,
        config: Optional[NewtonOptConfig] = None,
    ):
        if config is not None:
            NewtonOptConfig.__init__(self, **vars(config))
        self.d_opt = self.action_horizon * self.d_action
        self.line_scale = self._create_box_line_search(self.line_search_scale)
        Optimizer.__init__(self)
        self.i = -1
        self.outer_iters = math.ceil(self.n_iters / self.inner_iters)

        # create line search
        self.update_nproblems(self.n_problems)

        self.reset()

        # reshape action lows and highs:
        self.action_lows = self.action_lows.repeat(self.action_horizon)
        self.action_highs = self.action_highs.repeat(self.action_horizon)
        self.action_range = self.action_highs - self.action_lows
        self.action_step_max = self.step_scale * torch.abs(self.action_range)
        self.c_1 = 1e-5
        self.c_2 = 0.9
        self._out_m_idx = None
        self._out_best_x = None
        self._out_best_c = None
        self._out_best_grad = None
        self.cu_opt_graph = None
        if self.d_opt >= 1024:
            self.use_cuda_line_search_kernel = False
        if self.use_temporal_smooth:
            self._temporal_mat = build_fd_matrix(
                self.action_horizon,
                order=2,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            ).unsqueeze(0)
            eye_mat = torch.eye(
                self.action_horizon, device=self.tensor_args.device, dtype=self.tensor_args.dtype
            ).unsqueeze(0)
            self._temporal_mat += eye_mat
        self.rollout_fn.sum_horizon = True

    def reset_cuda_graph(self):
        if self.cu_opt_graph is not None:
            self.cu_opt_graph.reset()
        super().reset_cuda_graph()

    @torch.no_grad()
    def _get_step_direction(self, cost, q, grad_q):
        """
        Reimplement this function in derived class. Gradient Descent is implemented here.
        """
        return -1.0 * grad_q.view(-1, self.d_opt)

    def _shift(self, shift_steps=1):
        # TODO: shift best q?:
        self.best_cost[:] = 5000000.0
        self.best_iteration[:] = 0
        self.current_iteration[:] = 0
        return True

    def _optimize(self, q: T_BHDOF_float, shift_steps=0, n_iters=None):
        with profiler.record_function("newton_base/shift"):
            self._shift(shift_steps)
        # reshape q:
        if self.store_debug:
            self.debug.append(q.view(-1, self.action_horizon, self.d_action).clone())
        with profiler.record_function("newton_base/init_opt"):
            q = q.view(self.n_problems, self.action_horizon * self.d_action)
            grad_q = q.detach() * 0.0
        # run opt graph
        if not self.cu_opt_init:
            self._initialize_opt_iters_graph(q, grad_q, shift_steps=shift_steps)
        for i in range(self.outer_iters):
            best_q, best_cost, q, grad_q = self._call_opt_iters_graph(q, grad_q)
            if (
                not self.fixed_iters
                and self.use_cuda_update_best_kernel
                and (i + 1) * self.inner_iters >= self.min_iters
            ):
                if check_convergence(self.best_iteration, self.current_iteration, self.last_best):
                    break

        best_q = best_q.view(self.n_problems, self.action_horizon, self.d_action)
        return best_q

    def reset(self):
        with profiler.record_function("newton/reset"):
            self.i = -1
            self._opt_finished = False
            self.best_cost[:] = 5000000.0
            self.best_iteration[:] = 0
            self.current_iteration[:] = 0

        super().reset()

    def _opt_iters(self, q, grad_q, shift_steps=0):
        q = q.detach()  # .clone()
        grad_q = grad_q.detach()  # .clone()
        for _ in range(self.inner_iters):
            self.i += 1
            cost_n, q, grad_q = self._opt_step(q.detach(), grad_q.detach())
        if self.store_debug:
            self.debug.append(self.best_q.view(-1, self.action_horizon, self.d_action).clone())
            self.debug_cost.append(self.best_cost.detach().view(-1, 1).clone())

        return self.best_q.detach(), self.best_cost.detach(), q.detach(), grad_q.detach()

    def _opt_step(self, q, grad_q):
        with profiler.record_function("newton/line_search"):
            q_n, cost_n, grad_q_n = self._approx_line_search(q, grad_q)
        with profiler.record_function("newton/step_direction"):
            grad_q = self._get_step_direction(cost_n, q_n, grad_q_n)
        with profiler.record_function("newton/update_best"):
            self._update_best(q_n, grad_q_n, cost_n)
        return cost_n, q_n, grad_q

    def clip_bounds(self, x):
        x = torch.clamp(x, self.action_lows, self.action_highs)
        return x

    def scale_step_direction(self, dx):
        if self.use_temporal_smooth:
            dx_v = dx.view(-1, self.action_horizon, self.d_action)
            dx_new = self._temporal_mat @ dx_v  # 1,h,h x b, h, dof -> b, h, dof
            dx = dx_new.view(-1, self.action_horizon * self.d_action)
        dx_scaled = scale_action(dx, self.action_step_max)

        return dx_scaled

    def project_bounds(self, x):
        # Find maximum value along all joint angles:
        max_tensor = torch.maximum((self.action_lows - x), (x - self.action_highs)) / (
            (self.action_highs - self.action_lows)
        )

        # all values greater than 0 are out of bounds:
        scale = torch.max(max_tensor, dim=-1, keepdim=True)[0]
        scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
        scale[scale <= 0.0] = 1.0

        x = (1.0 / scale) * x

        # If we hit nans in scaling:
        x = torch.nan_to_num(x, nan=0.0)
        #
        # max_val = torch.max()
        # x = torch.clamp(x, self.action_lows, self.action_highs)
        return x

    def _compute_cost_gradient(self, x):
        x_n = x.detach().requires_grad_(True)
        x_in = x_n.view(
            self.n_problems * self.num_particles, self.action_horizon, self.rollout_fn.d_action
        )
        trajectories = self.rollout_fn(x_in)  # x_n = (batch*line_search_scale) x horizon x d_action
        if len(trajectories.costs.shape) == 2:
            cost = torch.sum(
                trajectories.costs.view(self.n_problems, self.num_particles, self.horizon),
                dim=-1,
                keepdim=True,
            )
        else:
            cost = trajectories.costs.view(self.n_problems, self.num_particles, 1)
        g_x = cost.backward(gradient=self.l_vec, retain_graph=False)
        g_x = x_n.grad.detach()
        return (
            cost,
            g_x,
        )  # cost: [n_problems, n_particles, 1], g_x: [n_problems, n_particles, horizon*d_action]

    def _wolfe_line_search(self, x, step_direction):
        # x_set = get_x_set_jit(step_direction, x, self.alpha_list, self.action_lows, self.action_highs)

        step_direction = step_direction.detach()
        step_vec = step_direction.unsqueeze(-2)
        x_set = get_x_set_jit(step_vec, x, self.alpha_list, self.action_lows, self.action_highs)
        # x_set = x.unsqueeze(-2) + self.alpha_list * step_vec
        # x_set = self.clip_bounds(x_set)
        # x_set = self.project_bounds(x_set)
        x_set = x_set.detach().requires_grad_(True)

        b, h, _ = x_set.shape
        c, g_x = self._compute_cost_gradient(x_set)
        with torch.no_grad():
            if not self.use_cuda_line_search_kernel:
                c_0 = c[:, 0:1]
                step_vec_T = step_vec.transpose(-1, -2)
                g_full_step = g_x @ step_vec_T

                # g_step = g_x[:,0:1] @ step_vec_T

                g_step = g_full_step[:, 0:1]
                # condition 1:
                wolfe_1 = c <= c_0 + self.c_1 * self.zero_alpha_list * g_step  # dot product

                # condition 2:
                if self.line_search_type == LineSearchType.STRONG_WOLFE:
                    wolfe_2 = torch.abs(g_full_step) <= self.c_2 * torch.abs(g_step)
                else:
                    wolfe_2 = g_full_step >= self.c_2 * g_step  # dot product

                wolfe = torch.logical_and(wolfe_1, wolfe_2)

                # get the last occurence of true (this will be the largest admissable alpha value):
                # wolfe will have 1 for indices that satisfy.
                step_success = wolfe * (self.zero_alpha_list + 0.1)

                _, m_idx = torch.max(step_success, dim=-2)

                # The below can also be moved into approx wolfe?
                if self.line_search_type != LineSearchType.APPROX_WOLFE:
                    step_success_w1 = wolfe_1 * (self.zero_alpha_list + 0.1)

                    _, m1_idx = torch.max(step_success_w1, dim=-2)

                    m_idx = torch.where(m_idx == 0, m1_idx, m_idx)

                # From ICRA23, we know that noisy update helps, so we don't check for zero here

                if self.line_search_type != LineSearchType.APPROX_WOLFE:
                    m_idx[m_idx == 0] = 1

                m = m_idx.squeeze() + self.c_idx
                g_x_1 = g_x.view(b * h, -1)
                xs_1 = x_set.view(b * h, -1)
                cs_1 = c.view(b * h, -1)
                best_c = cs_1[m]

                best_x = xs_1[m]
                best_grad = g_x_1[m].view(b, 1, self.d_opt)
                return best_x.detach(), best_c.detach(), best_grad.detach()
            else:
                if (
                    self._out_best_x is None
                    or self._out_best_x.shape[0] * self._out_best_x.shape[1]
                    != x_set.shape[0] * x_set.shape[2]
                ):
                    self._out_best_x = torch.zeros(
                        (x_set.shape[0], x_set.shape[2]),
                        dtype=self.tensor_args.dtype,
                        device=self.tensor_args.device,
                    )
                if (
                    self._out_best_c is None
                    or self._out_best_c.shape[0] * self._out_best_c.shape[1]
                    != c.shape[0] * c.shape[2]
                ):
                    self._out_best_c = torch.zeros(
                        (c.shape[0], c.shape[2]),
                        dtype=self.tensor_args.dtype,
                        device=self.tensor_args.device,
                    )
                if (
                    self._out_best_grad is None
                    or self._out_best_grad.shape[0] * self._out_best_grad.shape[1]
                    != g_x.shape[0] * g_x.shape[2]
                ):
                    self._out_best_grad = torch.zeros(
                        (g_x.shape[0], g_x.shape[2]),
                        dtype=self.tensor_args.dtype,
                        device=self.tensor_args.device,
                    )

                (best_x_n, best_c_n, best_grad_n) = wolfe_line_search(
                    self._out_best_x,  # * 0.0,
                    self._out_best_c,  # * 0.0,
                    self._out_best_grad,  # * 0.0,
                    g_x,
                    x_set,
                    step_vec,
                    c,
                    self.c_idx,
                    self.c_1,
                    self.c_2,
                    self.zero_alpha_list,
                    self.line_search_type == LineSearchType.STRONG_WOLFE,
                    self.line_search_type == LineSearchType.APPROX_WOLFE,
                )
                # c_0 = c[:, 0:1]
                # g_0 = g_x[:, 0:1]
                best_grad_n = best_grad_n.view(b, 1, self.d_opt)
                return best_x_n, best_c_n, best_grad_n

    def _greedy_line_search(self, x, step_direction):
        step_direction = step_direction.detach()
        x_set = x.unsqueeze(-2) + self.alpha_list * step_direction.unsqueeze(-2)
        x_set = self.clip_bounds(x_set)
        x_set = x_set.detach()

        x_set = x_set.detach().requires_grad_(True)
        b, h, _ = x_set.shape

        c, g_x = self._compute_cost_gradient(x_set)

        best_c, m_idx = torch.min(c, dim=-2)

        m = m_idx.squeeze() + self.c_idx
        g_x = g_x.view(b * h, -1)
        xs = x_set.view(b * h, -1)

        best_x = xs[m]
        best_grad = g_x[m].view(b, 1, self.d_opt)

        return best_x, best_c, best_grad

    def _armijo_line_search(self, x, step_direction):
        step_direction = step_direction.detach()
        step_vec = step_direction.unsqueeze(-2)
        x_set = x.unsqueeze(-2) + self.alpha_list * step_vec
        x_set = self.clip_bounds(x_set)
        x_set = x_set.detach().requires_grad_(True)
        b, h, _ = x_set.shape

        c, g_x = self._compute_cost_gradient(x_set)

        c_0 = c[:, 0:1]
        g_0 = g_x[:, 0:1]

        step_vec_T = step_vec.transpose(-1, -2)
        g_step = g_0 @ step_vec_T
        # condition 1:
        armjio_1 = c <= c_0 + self.c_1 * self.zero_alpha_list * g_step  # dot product

        # get the last occurence of true (this will be the largest admissable alpha value):
        # wolfe will have 1 for indices that satisfy.
        # find the
        step_success = armjio_1 * (self.zero_alpha_list + 0.1)

        _, m_idx = torch.max(step_success, dim=-2)

        m_idx[m_idx == 0] = 1

        m = m_idx.squeeze() + self.c_idx

        g_x = g_x.view(b * h, -1)
        xs = x_set.view(b * h, -1)
        cs = c.view(b * h, -1)
        best_c = cs[m]

        best_x = xs[m]
        best_grad = g_x[m].view(b, 1, self.d_opt)
        return best_x, best_c, best_grad

    def _approx_line_search(self, x, step_direction):
        if self.step_scale != 0.0 and self.step_scale != 1.0:
            step_direction = self.scale_step_direction(step_direction)
        if self.fix_terminal_action and self.action_horizon > 1:
            step_direction[..., (self.action_horizon - 1) * self.d_action :] = 0.0
        if self.line_search_type == LineSearchType.GREEDY:
            best_x, best_c, best_grad = self._greedy_line_search(x, step_direction)
        elif self.line_search_type == LineSearchType.ARMIJO:
            best_x, best_c, best_grad = self._armijo_line_search(x, step_direction)
        elif self.line_search_type in [
            LineSearchType.WOLFE,
            LineSearchType.STRONG_WOLFE,
            LineSearchType.APPROX_WOLFE,
        ]:
            best_x, best_c, best_grad = self._wolfe_line_search(x, step_direction)
        if self.fix_terminal_action and self.action_horizon > 1:
            best_grad[..., (self.action_horizon - 1) * self.d_action :] = 0.0
        return best_x, best_c, best_grad

    def check_convergence(self, cost):
        above_threshold = cost > self.cost_convergence
        above_threshold = torch.count_nonzero(above_threshold)
        if above_threshold == 0:
            self._opt_finished = True
            return True
        return False

    def _update_best(self, q, grad_q, cost):
        if self.use_cuda_update_best_kernel:
            (self.best_cost, self.best_q, self.best_iteration) = update_best(
                self.best_cost,
                self.best_q,
                self.best_iteration,
                self.current_iteration,
                cost,
                q,
                self.d_opt,
                self.last_best,
                self.cost_delta_threshold,
                self.cost_relative_threshold,
            )
            # print(self.best_cost[0], self.best_q[0])
        else:
            cost = cost.detach()
            q = q.detach()
            mask = cost < self.best_cost
            self.best_cost.copy_(torch.where(mask, cost, self.best_cost))
            mask = mask.view(mask.shape[0])
            mask_q = mask.unsqueeze(-1).expand(-1, self.d_opt)
            self.best_q.copy_(torch.where(mask_q, q, self.best_q))

    def update_nproblems(self, n_problems):
        self.l_vec = torch.ones(
            (n_problems, self.num_particles, 1),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        self.best_cost = (
            torch.ones(
                (n_problems, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            * 5000000.0
        )
        self.best_q = torch.zeros(
            (n_problems, self.d_opt), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self.best_grad_q = torch.zeros(
            (n_problems, 1, self.d_opt),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )

        # create list:
        self.alpha_list = self.line_scale.repeat(n_problems, 1, 1)
        self.zero_alpha_list = self.alpha_list[:, :, 0:1].contiguous()
        h = self.alpha_list.shape[1]
        self.c_idx = torch.arange(
            0, n_problems * h, step=(h), device=self.tensor_args.device, dtype=torch.long
        )
        self.best_iteration = torch.zeros(
            (n_problems), device=self.tensor_args.device, dtype=torch.int16
        )
        self.current_iteration = torch.zeros((1), device=self.tensor_args.device, dtype=torch.int16)
        self.cu_opt_init = False
        super().update_nproblems(n_problems)

    def _initialize_opt_iters_graph(self, q, grad_q, shift_steps):
        if self.use_cuda_graph:
            self._create_opt_iters_graph(q, grad_q, shift_steps)
        self.cu_opt_init = True

    def _create_box_line_search(self, line_search_scale):
        """

        Args:
            line_search_scale (_type_): should have n values
        """
        d = []
        dof_vec = torch.zeros(
            (self.d_opt), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        for i in line_search_scale:
            d.append(dof_vec + i)
        d = torch.stack(d, dim=0).unsqueeze(0)
        return d

    def _call_opt_iters_graph(self, q, grad_q):
        if self.use_cuda_graph:
            self._cu_opt_q_in.copy_(q.detach())
            self._cu_opt_gq_in.copy_(grad_q.detach())
            self.cu_opt_graph.replay()
            return (
                self._cu_opt_q.clone(),
                self._cu_opt_cost.clone(),
                self._cu_q.clone(),
                self._cu_gq.clone(),
            )
        else:
            return self._opt_iters(q, grad_q)

    def _create_opt_iters_graph(self, q, grad_q, shift_steps):
        # create a new stream:
        self.reset()
        self._cu_opt_q_in = q.detach().clone()
        self._cu_opt_gq_in = grad_q.detach().clone()
        s = torch.cuda.Stream(device=self.tensor_args.device)
        s.wait_stream(torch.cuda.current_stream(device=self.tensor_args.device))

        with torch.cuda.stream(s):
            for _ in range(3):
                self._cu_opt_q, self._cu_opt_cost, self._cu_q, self._cu_gq = self._opt_iters(
                    self._cu_opt_q_in, self._cu_opt_gq_in, shift_steps
                )
        torch.cuda.current_stream(device=self.tensor_args.device).wait_stream(s)
        self.reset()
        self.cu_opt_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cu_opt_graph, stream=s):
            self._cu_opt_q, self._cu_opt_cost, self._cu_q, self._cu_gq = self._opt_iters(
                self._cu_opt_q_in, self._cu_opt_gq_in, shift_steps
            )
        torch.cuda.current_stream(device=self.tensor_args.device).wait_stream(s)


@get_torch_jit_decorator()
def get_x_set_jit(step_vec, x, alpha_list, action_lows, action_highs):
    # step_direction = step_direction.detach()
    x_set = torch.clamp(x.unsqueeze(-2) + alpha_list * step_vec, action_lows, action_highs)
    # x_set = x.unsqueeze(-2) + alpha_list * step_vec
    return x_set


@get_torch_jit_decorator()
def _armijo_line_search_tail_jit(c, g_x, step_direction, c_1, alpha_list, c_idx, x_set, d_opt):
    c_0 = c[:, 0:1]
    g_0 = g_x[:, 0:1]
    step_vec = step_direction.unsqueeze(-2)
    step_vec_T = step_vec.transpose(-1, -2)
    g_step = g_0 @ step_vec_T
    # condition 1:
    armjio_1 = c <= c_0 + c_1 * alpha_list * g_step  # dot product

    # get the last occurence of true (this will be the largest admissable alpha value):
    # wolfe will have 1 for indices that satisfy.
    # find the
    step_success = armjio_1 * (alpha_list[:, :, 0:1] + 0.1)

    _, m_idx = torch.max(step_success, dim=-2)

    m_idx[m_idx == 0] = 1

    m = m_idx.squeeze() + c_idx
    b, h, _ = x_set.shape
    g_x = g_x.view(b * h, -1)
    xs = x_set.view(b * h, -1)
    cs = c.view(b * h, -1)
    best_c = cs[m]

    best_x = xs[m]
    best_grad = g_x[m].view(b, 1, d_opt)
    return (best_x, best_c, best_grad)


@get_torch_jit_decorator()
def _wolfe_search_tail_jit(c, g_x, x_set, m, d_opt: int):
    b, h, _ = x_set.shape
    g_x = g_x.view(b * h, -1)
    xs = x_set.view(b * h, -1)
    cs = c.view(b * h, -1)
    best_c = cs[m]
    best_x = xs[m]
    best_grad = g_x[m].view(b, 1, d_opt)
    return (best_x, best_c, best_grad)


@get_torch_jit_decorator()
def scale_action_old(dx, action_step_max):
    scale_value = torch.max(torch.abs(dx) / action_step_max, dim=-1, keepdim=True)[0]
    scale_value = torch.clamp(scale_value, 1.0)
    dx_scaled = dx / scale_value
    return dx_scaled


@get_torch_jit_decorator()
def scale_action(dx, action_step_max):

    # get largest dx scaled by bounds across optimization variables
    scale_value = torch.max(torch.abs(dx) / action_step_max, dim=-1, keepdim=True)[0]

    # scale dx to bring all dx within bounds:
    # only perfom for dx that are greater than 1:

    new_scale = torch.where(scale_value <= 1.0, 1.0, scale_value)
    dx_scaled = dx / new_scale

    # scale_value = torch.clamp(scale_value, 1.0)
    # dx_scaled = dx / scale_value
    return dx_scaled


@get_torch_jit_decorator()
def check_convergence(
    best_iteration: torch.Tensor, current_iteration: torch.Tensor, last_best: int
) -> bool:
    success = False
    if torch.max(best_iteration).item() <= (-1.0 * (last_best)):
        success = True
    return success
