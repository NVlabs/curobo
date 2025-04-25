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
from typing import Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.curobolib.opt import LBFGScu
from curobo.opt.newton.newton_base import NewtonOptBase, NewtonOptConfig
from curobo.util.logger import log_info
from curobo.util.torch_utils import get_torch_jit_decorator


# kernel for l-bfgs:
@get_torch_jit_decorator(only_valid_for_compile=True)
def jit_lbfgs_compute_step_direction(
    alpha_buffer: torch.Tensor,
    rho_buffer: torch.Tensor,
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    grad_q: torch.Tensor,
    m: int,
    epsilon: float,
    stable_mode: bool = True,
):
    grad_q = grad_q.transpose(-1, -2)

    # m = 15 (int)
    # y_buffer, s_buffer: m x b x 175
    # q, grad_q: b x 175
    # rho_buffer: m x b x 1
    # alpha_buffer: m x b x 1 # this can be dynamically created
    gq = grad_q.detach().clone()
    rho_s = rho_buffer * s_buffer.transpose(-1, -2)  # batched m_scalar-m_vector product
    for i in range(m - 1, -1, -1):
        alpha_buffer[i] = rho_s[i] @ gq  # batched vector-vector dot product
        gq = gq - (alpha_buffer[i] * y_buffer[i])  # batched scalar-vector product

    var1 = (s_buffer[-1].transpose(-1, -2) @ y_buffer[-1]) / (
        y_buffer[-1].transpose(-1, -2) @ y_buffer[-1]
    )
    if stable_mode:
        var1 = torch.nan_to_num(var1, epsilon, epsilon, epsilon)
    gamma = torch.nn.functional.relu(var1)  # + epsilon
    r = gamma * gq.clone()
    rho_y = rho_buffer * y_buffer.transpose(-1, -2)  # batched m_scalar-m_vector product
    for i in range(m):
        # batched dot product, scalar-vector product
        r = r + (alpha_buffer[i] - (rho_y[i] @ r)) * s_buffer[i]
    return -1.0 * r


@get_torch_jit_decorator()
def jit_lbfgs_update_buffers(
    q, grad_q, s_buffer, y_buffer, rho_buffer, x_0, grad_0, stable_mode: bool
):
    grad_q = grad_q.transpose(-1, -2)
    q = q.unsqueeze(-1)

    y = grad_q - grad_0
    s = q - x_0
    rho = 1 / (y.transpose(-1, -2) @ s)
    if stable_mode:
        rho = torch.nan_to_num(rho, 0.0, 0.0, 0.0)
    s_buffer[0] = s
    s_buffer[:] = torch.roll(s_buffer, -1, dims=0)
    y_buffer[0] = y
    y_buffer[:] = torch.roll(y_buffer, -1, dims=0)  # .copy_(y_buff)

    rho_buffer[0] = rho

    rho_buffer[:] = torch.roll(rho_buffer, -1, dims=0)

    x_0.copy_(q)
    grad_0.copy_(grad_q)
    return s_buffer, y_buffer, rho_buffer, x_0, grad_0


@dataclass
class LBFGSOptConfig(NewtonOptConfig):
    history: int = 10
    epsilon: float = 0.01
    use_cuda_kernel: bool = True
    stable_mode: bool = True
    use_shared_buffers_kernel: bool = True

    def __post_init__(self):
        return super().__post_init__()


class LBFGSOpt(NewtonOptBase, LBFGSOptConfig):
    @profiler.record_function("lbfgs_opt/init")
    def __init__(self, config: Optional[LBFGSOptConfig] = None):
        if config is not None:
            LBFGSOptConfig.__init__(self, **vars(config))
        NewtonOptBase.__init__(self)
        if (
            self.d_opt >= 1024
            or self.history > 31
            or ((self.d_opt * self.history + 33) * 4 >= 48000)
        ):
            log_info("LBFGS: Not using LBFGS Cuda Kernel as d_opt>1024 or history>15")
            self.use_cuda_kernel = False
        if self.history > self.d_opt:
            log_info("LBFGS: history >= d_opt, reducing history to d_opt-1")
            self.history = self.d_opt - 1
            self.init_hessian(self.n_problems)

    @profiler.record_function("lbfgs/reset")
    def reset(self):
        self.x_0[:] = 0.0
        self.grad_0[:] = 0.0
        self.s_buffer[:] = 0.0
        self.y_buffer[:] = 0.0
        self.rho_buffer[:] = 0.0
        self.alpha_buffer[:] = 0.0
        self.step_q_buffer[:] = 0.0
        return super().reset()

    def update_nproblems(self, n_problems):
        self.init_hessian(b=n_problems)
        return super().update_nproblems(n_problems)

    def init_hessian(self, b=1):
        self.x_0 = torch.zeros(
            (b, self.d_opt, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self.grad_0 = torch.zeros(
            (b, self.d_opt, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self.y_buffer = torch.zeros(
            (self.history, b, self.d_opt, 1),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )  # + 1.0
        self.s_buffer = torch.zeros(
            (self.history, b, self.d_opt, 1),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )  # + 1.0
        self.rho_buffer = torch.zeros(
            (self.history, b, 1, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )  # + 1.0
        self.step_q_buffer = torch.zeros(
            (b, self.d_opt), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self.alpha_buffer = torch.zeros(
            (self.history, b, 1, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )  # + 1.0

    @torch.no_grad()
    def _get_step_direction(self, cost, q, grad_q):
        if self.use_cuda_kernel:
            with profiler.record_function("lbfgs/fused"):
                dq = LBFGScu.apply(
                    self.step_q_buffer,
                    self.rho_buffer,
                    self.y_buffer,
                    self.s_buffer,
                    q,
                    grad_q,
                    self.x_0,
                    self.grad_0,
                    self.epsilon,
                    self.stable_mode,
                    self.use_shared_buffers_kernel,
                )

        else:

            self._update_buffers(q, grad_q)

            dq = jit_lbfgs_compute_step_direction(
                self.alpha_buffer,
                self.rho_buffer,
                self.y_buffer,
                self.s_buffer,
                grad_q,
                self.history,
                self.epsilon,
                self.stable_mode,
            )

            dq = dq.view(-1, self.d_opt)
        return dq

    def _update_q(self, grad_q):
        q = grad_q.detach().clone()
        rho_s = self.rho_buffer * self.s_buffer.transpose(-1, -2)
        for i in range(self.history - 1, -1, -1):
            self.alpha_buffer[i] = rho_s[i] @ q
            q = q - (self.alpha_buffer[i] * self.y_buffer[i])
        return q

    def _update_r(self, r_init):
        r = r_init.clone()
        rho_y = self.rho_buffer * self.y_buffer.transpose(-1, -2)
        for i in range(self.history):
            r = r + self.s_buffer[i] * (self.alpha_buffer[i] - rho_y[i] @ r)
        return -1.0 * r

    def _update_buffers(self, q, grad_q):
        if True:
            self.s_buffer, self.y_buffer, self.rho_buffer, self.x_0, self.grad_0 = (
                jit_lbfgs_update_buffers(
                    q,
                    grad_q,
                    self.s_buffer,
                    self.y_buffer,
                    self.rho_buffer,
                    self.x_0,
                    self.grad_0,
                    self.stable_mode,
                )
            )
            return
        grad_q = grad_q.transpose(-1, -2)
        q = q.unsqueeze(-1)

        y = grad_q - self.grad_0
        s = q - self.x_0
        rho = 1 / (y.transpose(-1, -2) @ s)
        if self.stable_mode:
            rho = torch.nan_to_num(rho, 0, 0, 0)
        self.s_buffer[0] = s
        self.s_buffer[:] = torch.roll(self.s_buffer, -1, dims=0)
        self.y_buffer[0] = y
        self.y_buffer[:] = torch.roll(self.y_buffer, -1, dims=0)  # .copy_(y_buff)

        self.rho_buffer[0] = rho

        self.rho_buffer[:] = torch.roll(self.rho_buffer, -1, dims=0)

        self.x_0.copy_(q)
        self.grad_0.copy_(grad_q)

    def _shift(self, shift_steps=0):
        """Shift the optimizer by shift_steps * d_opt

        Args:
            shift_steps (int, optional): _description_. Defaults to 0.
        """
        if shift_steps == 0:
            return
        self.reset()
        shift_d = shift_steps * self.d_action
        self.x_0 = self._shift_buffer(self.x_0, shift_d, shift_steps)
        self.grad_0 = self._shift_buffer(self.grad_0, shift_d, shift_steps)
        self.y_buffer = self._shift_buffer(self.y_buffer, shift_d, shift_steps)
        self.s_buffer = self._shift_buffer(self.s_buffer, shift_d, shift_steps)
        super()._shift(shift_steps=shift_steps)

    def _shift_buffer(self, buffer, shift_d, shift_steps: int = 1):
        buffer = buffer.roll(-shift_d, -2)
        end_value = -(shift_steps - 1) * self.d_action
        if end_value == 0:
            end_value = buffer.shape[-2]
        buffer[..., -shift_d:end_value, :] = buffer[
            ..., -shift_d - self.d_action : -shift_d, :
        ].clone()

        return buffer
