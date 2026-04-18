# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import warp as wp

from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream


@dataclass
class LevenbergMarquardtState:
    jacobian: torch.Tensor  # shape: (batch_size, n_residuals, action_dim)
    jTerror: torch.Tensor  # shape: (batch_size, action_dim)
    lambda_damping: torch.Tensor  # shape: (batch_size)
    joint_position_in: torch.Tensor  # shape: (batch_size, action_dim)
    joint_position_out: torch.Tensor  # shape: (batch_size, action_dim)
    pred_reduction: torch.Tensor  # shape: (batch_size)
    _batch_size: Optional[int] = None
    _action_dim: Optional[int] = None
    _n_residuals: Optional[int] = None

    def __post_init__(self):
        if self.jacobian.ndim != 3:
            log_and_raise(f"jacobian must be a 3D tensor, but got {self.jacobian.shape}")
        if self.jTerror.ndim != 2:
            log_and_raise(f"jTerror must be a 2D tensor, but got {self.jTerror.shape}")
        if self.lambda_damping.ndim != 1:
            log_and_raise(f"lambda_damping must be a 1D tensor, but got {self.lambda_damping.shape}")
        if self.joint_position_in.ndim != 2:
            log_and_raise(f"joint_position_in must be a 2D tensor, but got {self.joint_position_in.shape}")
        if self.joint_position_out.ndim != 2:
            log_and_raise(
                f"joint_position_out must be a 2D tensor, but got {self.joint_position_out.shape}"
            )
        if self.pred_reduction.ndim != 1:
            log_and_raise(f"pred_reduction must be a 1D tensor, but got {self.pred_reduction.shape}")
        self._batch_size, self._n_residuals, self._action_dim = self.jacobian.shape

        if self.jTerror.shape != (self._batch_size, self._action_dim):
            log_and_raise(
                f"jTerror must be a 2D tensor with shape ({self._batch_size}, {self._action_dim}), but got {self.jTerror.shape}"
            )
        if self.lambda_damping.shape != (self._batch_size,):
            log_and_raise(
                f"lambda_damping must be a 1D tensor with shape ({self._batch_size}), but got {self.lambda_damping.shape}"
            )
        if self.joint_position_in.shape != (self._batch_size, self._action_dim):
            log_and_raise(
                f"joint_position_in must be a 2D tensor with shape ({self._batch_size}, {self._action_dim}), but got {self.joint_position_in.shape}"
            )
        if self.joint_position_out.shape != (self._batch_size, self._action_dim):
            log_and_raise(
                f"joint_position_out must be a 2D tensor with shape ({self._batch_size}, {self._action_dim}), but got {self.joint_position_out.shape}"
            )
        if self.pred_reduction.shape != (self._batch_size,):
            log_and_raise(
                f"pred_reduction must be a 1D tensor with shape ({self._batch_size}), but got {self.pred_reduction.shape}"
            )

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def n_residuals(self):
        return self._n_residuals


class LevenbergMarquardtStep:
    def __init__(self, action_dim: int, n_residuals: int, tile_threads: int = 64):
        self._lm_warp_kernel = LevenbergMarquardtStep.create_lm_warp_kernel(action_dim, n_residuals)
        self._action_dim = action_dim
        self._n_residuals = n_residuals
        self._tile_threads = tile_threads

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def n_residuals(self):
        return self._n_residuals

    @property
    def tile_threads(self):
        return self._tile_threads

    def __call__(self, state: LevenbergMarquardtState) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.n_residuals != self.n_residuals:
            log_and_raise(
                f"n_residuals must be the same as the last dimension of joint_position_in, but got {state.n_residuals} and {self.n_residuals}"
            )
        if state.action_dim != self.action_dim:
            log_and_raise(
                f"action_dim must be the same as the last dimension of joint_position_in, but got {state.action_dim} and {self.action_dim}"
            )

        jac_t = state.jacobian.detach().view(state.batch_size, self.n_residuals, self.action_dim)
        jtres_t = state.jTerror.detach().view(state.batch_size, self.action_dim)
        lams_t = state.lambda_damping.detach().view(state.batch_size)
        joint_position_in_t = state.joint_position_in.detach().view(state.batch_size, self.action_dim)
        pred_t = state.pred_reduction.detach().view(state.batch_size)
        joint_position_out_t = state.joint_position_out.detach().view(state.batch_size, self.action_dim)
        check_float32_tensors(
            jac_t.device,
            jac=jac_t,
            jtres=jtres_t,
            lams=lams_t,
            joint_position_in=joint_position_in_t,
            pred=pred_t,
            joint_position_out=joint_position_out_t,
        )
        jac = wp.from_torch(jac_t)  # wp.zeros((batch_size, self.dof, 13), dtype=wp.float32)
        jtres = wp.from_torch(jtres_t)  # wp.zeros((batch_size, self.dof, 1), dtype=wp.float32)
        lams = wp.from_torch(lams_t)  # wp.zeros((batch_size), dtype=wp.float32)
        joint_position_in = wp.from_torch(joint_position_in_t)  # wp.zeros((batch_size, self.dof), dtype=wp.float32)
        pred = wp.from_torch(pred_t)  # wp.zeros((batch_size), dtype=wp.float32)
        joint_position_out = wp.from_torch(joint_position_out_t)
        wp.launch_tiled(
            self._lm_warp_kernel,
            dim=[state.batch_size],
            inputs=[
                jac,
                jtres,
                lams,
                joint_position_in,
                joint_position_out,
                pred,
            ],
            block_dim=self.tile_threads,
            stream=get_warp_device_stream(state.joint_position_in)[1],
            device=get_warp_device_stream(state.joint_position_in)[0],
        )

        return state.joint_position_out, state.pred_reduction

    @staticmethod
    def create_lm_warp_kernel(dof: int, n_res: int):
        RES = wp.constant(n_res)
        COORD = wp.constant(dof)

        # create warp kernel here:
        def _lm_step_template(
            jacobians: wp.array3d(dtype=wp.float32),  # (num_problems, n_residuals, n_coords)
            Jtresiduals: wp.array2d(dtype=wp.float32),  # (num_problems, n_coords)
            lambda_values: wp.array1d(dtype=wp.float32),  # (num_problems)
            joint_position_in: wp.array2d(dtype=wp.float32),  # (num_problems, n_coords)
            # outputs
            new_joint_position: wp.array2d(dtype=wp.float32),  # (num_problems, n_coords)
            pred_reduction_out: wp.array1d(dtype=wp.float32),  # (num_problems)
        ):
            problem_idx = wp.tid()
            J = wp.tile_load(jacobians[problem_idx], shape=(RES, COORD))
            lam = lambda_values[problem_idx]
            Jt = wp.tile_transpose(J)
            JtJ = wp.tile_zeros(shape=(COORD, COORD), dtype=wp.float32)
            wp.tile_matmul(Jt, J, JtJ)

            diag = wp.tile_zeros(shape=(COORD,), dtype=wp.float32)
            for i in range(COORD):
                diag[i] = lam

            A = wp.tile_diag_add(JtJ, diag)

            g = wp.tile_load(Jtresiduals[problem_idx], shape=(COORD,))

            rhs = wp.tile_map(wp.neg, g)
            L = wp.tile_cholesky(A)
            delta = wp.tile_cholesky_solve(L, rhs)

            local_joint_position = wp.tile_load(joint_position_in[problem_idx], shape=(COORD))

            updated_joint_position = wp.tile_map(wp.add, local_joint_position, delta)

            wp.tile_store(new_joint_position[problem_idx], updated_joint_position)

            lambda_delta = wp.tile_zeros(shape=(COORD), dtype=wp.float32)
            for i in range(COORD):
                lambda_delta[i] = lam * delta[i]

            diff = wp.tile_map(wp.sub, lambda_delta, g)
            prod = wp.tile_map(wp.mul, delta, diff)
            red = wp.tile_sum(prod)[0]
            pred_reduction_out[problem_idx] = 0.5 * red

        _lm_step_template.__name__ = f"_lm_solve_tiled_{COORD}_{RES}"
        _lm_step_template.__qualname__ = f"_lm_solve_tiled_{COORD}_{RES}"
        lm_solve_tiled = wp.kernel(enable_backward=False, module="unique")(_lm_step_template)

        return lm_solve_tiled
