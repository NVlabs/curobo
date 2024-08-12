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
from abc import abstractmethod
from enum import Enum
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.util.torch_utils import get_torch_jit_decorator

# Local Folder
from .integration_utils import (
    AccelerationTensorStepIdxKernel,
    AccelerationTensorStepKernel,
    CliqueTensorStepCentralDifferenceKernel,
    CliqueTensorStepIdxCentralDifferenceKernel,
    CliqueTensorStepIdxKernel,
    CliqueTensorStepKernel,
    build_fd_matrix,
    build_int_matrix,
    build_start_state_mask,
    tensor_step_acc_semi_euler,
    tensor_step_pos,
    tensor_step_pos_clique,
)


class TensorStepType(Enum):
    POSITION_TELEPORT = 0
    POSITION_CLIQUE_KERNEL = 1
    VELOCITY = 2  # Not implemented
    ACCELERATION_KERNEL = 3
    JERK = 4  # Not implemented
    POSITION = 5  # deprecated
    POSITION_CLIQUE = 6  # deprecated
    ACCELERATION = 7  # deprecated


class TensorStepBase:
    def __init__(
        self, tensor_args: TensorDeviceType, batch_size: int = 1, horizon: int = 1
    ) -> None:
        self.batch_size = -1
        self.horizon = -1
        self.tensor_args = tensor_args
        self._diag_dt = None
        self._inv_dt_h = None
        self.action_horizon = horizon
        self.update_batch_size(batch_size, horizon)

    def update_dt(self, dt: float):
        self._dt_h[:] = dt
        if self._inv_dt_h is not None:
            self._inv_dt_h[:] = 1.0 / dt

    @abstractmethod
    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        self.horizon = horizon
        self.batch_size = batch_size

    @abstractmethod
    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        pass


class TensorStepAcceleration(TensorStepBase):
    def __init__(
        self,
        tensor_args: TensorDeviceType,
        dt_h: torch.Tensor,
        batch_size: int = 1,
        horizon: int = 1,
    ) -> None:
        super().__init__(tensor_args, batch_size=batch_size, horizon=horizon)
        self._dt_h = dt_h
        self._diag_dt_h = torch.diag(self._dt_h)
        self._integrate_matrix_pos = None
        self._integrate_matrix_vel = None

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        if self.horizon != horizon:
            self._integrate_matrix_pos = (
                build_int_matrix(
                    horizon,
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                    diagonal=0,
                )
                @ self._diag_dt_h
            )
            self._integrate_matrix_vel = self._integrate_matrix_pos @ self._diag_dt_h
        return super().update_batch_size(batch_size, horizon)

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        if start_state_idx is None:
            state_seq = tensor_step_acc_semi_euler(
                start_state,
                u_act,
                out_state_seq,
                self._diag_dt_h,
                self._integrate_matrix_vel,
                self._integrate_matrix_pos,
            )
        else:
            state_seq = tensor_step_acc_semi_euler(
                start_state[start_state_idx],
                u_act,
                out_state_seq,
                self._diag_dt_h,
                self._integrate_matrix_vel,
                self._integrate_matrix_pos,
            )
        return state_seq


class TensorStepPositionTeleport(TensorStepBase):
    def __init__(
        self,
        tensor_args: TensorDeviceType,
        batch_size: int = 1,
        horizon: int = 1,
    ) -> None:
        super().__init__(tensor_args, batch_size=batch_size, horizon=horizon)

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        out_state_seq.position = u_act
        return out_state_seq


class TensorStepPosition(TensorStepBase):
    def __init__(
        self,
        tensor_args: TensorDeviceType,
        dt_h: torch.Tensor,
        batch_size: int = 1,
        horizon: int = 1,
    ) -> None:
        super().__init__(tensor_args, batch_size=batch_size, horizon=horizon)

        self._dt_h = dt_h
        # self._diag_dt_h = torch.diag(1 / self._dt_h)
        self._fd_matrix = None

    def update_dt(self, dt: float):
        super().update_dt(dt)
        self._fd_matrix = build_fd_matrix(
            self.horizon,
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
            order=1,
        )
        self._fd_matrix = torch.diag(1.0 / self._dt_h) @ self._fd_matrix

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        if horizon != self.horizon:
            self._fd_matrix = build_fd_matrix(
                horizon,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
                order=1,
            )
            self._fd_matrix = torch.diag(1.0 / self._dt_h) @ self._fd_matrix

        return super().update_batch_size(batch_size, horizon)

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        if start_state_idx is None:
            state_seq = tensor_step_pos(start_state, u_act, out_state_seq, self._fd_matrix)
        else:
            state_seq = tensor_step_pos(
                start_state[start_state_idx], u_act, out_state_seq, self._fd_matrix
            )
        return state_seq


class TensorStepPositionClique(TensorStepBase):
    def __init__(
        self,
        tensor_args: TensorDeviceType,
        dt_h: torch.Tensor,
        batch_size: int = 1,
        horizon: int = 1,
    ) -> None:
        super().__init__(tensor_args, batch_size=batch_size, horizon=horizon)

        self._dt_h = dt_h
        self._inv_dt_h = 1.0 / dt_h
        self._fd_matrix = None
        self._start_mask_matrix = None

    def update_dt(self, dt: float):
        super().update_dt(dt)
        self._fd_matrix = []
        for i in range(3):
            self._fd_matrix.append(
                build_fd_matrix(
                    self.horizon,
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                    order=i + 1,
                    SHIFT=True,
                )
            )
        self._diag_dt_h = torch.diag(self._inv_dt_h)

        self._fd_matrix[0] = self._diag_dt_h @ self._fd_matrix[0]
        self._fd_matrix[1] = self._diag_dt_h**2 @ self._fd_matrix[1]
        self._fd_matrix[2] = self._diag_dt_h**3 @ self._fd_matrix[2]

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        if self.horizon != horizon:
            self._fd_matrix = []
            for i in range(3):
                self._fd_matrix.append(
                    build_fd_matrix(
                        horizon,
                        device=self.tensor_args.device,
                        dtype=self.tensor_args.dtype,
                        order=i + 1,
                        SHIFT=True,
                    )
                )
            self._diag_dt_h = torch.diag(self._inv_dt_h)

            self._fd_matrix[0] = self._diag_dt_h @ self._fd_matrix[0]
            self._fd_matrix[1] = self._diag_dt_h**2 @ self._fd_matrix[1]
            self._fd_matrix[2] = self._diag_dt_h**3 @ self._fd_matrix[2]
            self._start_mask_matrix = list(build_start_state_mask(horizon, self.tensor_args))
        return super().update_batch_size(batch_size, horizon)

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        if start_state_idx is None:
            state_seq = tensor_step_pos_clique(
                start_state, u_act, out_state_seq, self._start_mask_matrix, self._fd_matrix
            )
        else:
            state_seq = tensor_step_pos_clique(
                start_state[start_state_idx],
                u_act,
                out_state_seq,
                self._start_mask_matrix,
                self._fd_matrix,
            )
        return state_seq


class TensorStepAccelerationKernel(TensorStepBase):
    def __init__(
        self,
        tensor_args: TensorDeviceType,
        dt_h: torch.Tensor,
        dof: int,
        batch_size: int = 1,
        horizon: int = 1,
    ) -> None:
        self.dof = dof

        super().__init__(tensor_args, batch_size=batch_size, horizon=horizon)

        self._dt_h = dt_h
        self._u_grad = None

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        if batch_size != self.batch_size or horizon != self.horizon:
            self._u_grad = torch.zeros(
                (batch_size, horizon, self.dof),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
        if force_update:
            self._u_grad = self._u_grad.detach()
        return super().update_batch_size(batch_size, horizon)

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        if start_state_idx is None:
            (
                out_state_seq.position,
                out_state_seq.velocity,
                out_state_seq.acceleration,
                out_state_seq.jerk,
            ) = AccelerationTensorStepKernel.apply(
                u_act,
                start_state.position,  # .contiguous(),
                start_state.velocity,  # .contiguous(),
                start_state.acceleration,  # .contiguous(),
                out_state_seq.position,  # .contiguous(),
                out_state_seq.velocity,  # .contiguous(),
                out_state_seq.acceleration,  # .contiguous(),
                out_state_seq.jerk,  # .contiguous(),
                self._dt_h,
                self._u_grad,
            )

        else:
            (
                out_state_seq.position,
                out_state_seq.velocity,
                out_state_seq.acceleration,
                out_state_seq.jerk,
            ) = AccelerationTensorStepIdxKernel.apply(
                u_act,
                start_state.position,  # .contiguous(),
                start_state.velocity,  # .contiguous(),
                start_state.acceleration,  # .contiguous(),
                start_state_idx,
                out_state_seq.position,  # .contiguous(),
                out_state_seq.velocity,  # .contiguous(),
                out_state_seq.acceleration,  # .contiguous(),
                out_state_seq.jerk,  # .contiguous(),
                self._dt_h,
                self._u_grad,
            )

        return out_state_seq


class TensorStepPositionCliqueKernel(TensorStepBase):
    def __init__(
        self,
        tensor_args: TensorDeviceType,
        dt_h: torch.Tensor,
        dof: int,
        finite_difference_mode: int = -1,
        filter_velocity: bool = False,
        filter_acceleration: bool = False,
        filter_jerk: bool = False,
        batch_size: int = 1,
        horizon: int = 1,
    ) -> None:
        self.dof = dof
        self._fd_mode = finite_difference_mode
        super().__init__(tensor_args, batch_size=batch_size, horizon=horizon)
        self._dt_h = dt_h
        self._inv_dt_h = 1.0 / dt_h
        self._u_grad = None
        self._filter_velocity = filter_velocity
        self._filter_acceleration = filter_acceleration
        self._filter_jerk = filter_jerk

        if self._filter_velocity or self._filter_acceleration or self._filter_jerk:
            kernel = self.tensor_args.to_device([[[0.06136, 0.24477, 0.38774, 0.24477, 0.06136]]])
            self._sma = torch.nn.functional.conv1d

            weights = kernel
            self._sma_kernel = weights

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        if batch_size != self.batch_size or horizon != self.horizon:
            self.action_horizon = horizon
            if self._fd_mode == 0:
                self.action_horizon = horizon - 4
            self._u_grad = torch.zeros(
                (batch_size, self.action_horizon, self.dof),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
        if force_update:
            self._u_grad = self._u_grad.detach()
        return super().update_batch_size(batch_size, horizon)

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        if start_state_idx is None:
            if self._fd_mode == -1:
                (
                    out_state_seq.position,
                    out_state_seq.velocity,
                    out_state_seq.acceleration,
                    out_state_seq.jerk,
                ) = CliqueTensorStepKernel.apply(
                    u_act,
                    start_state.position,  # .contiguous(),
                    start_state.velocity,  # .contiguous(),
                    start_state.acceleration,  # .contiguous(),
                    out_state_seq.position,  # .contiguous(),
                    out_state_seq.velocity,  # .contiguous(),
                    out_state_seq.acceleration,  # .contiguous(),
                    out_state_seq.jerk,  # .contiguous(),
                    self._inv_dt_h,
                    self._u_grad,
                )
            else:
                (
                    out_state_seq.position,
                    out_state_seq.velocity,
                    out_state_seq.acceleration,
                    out_state_seq.jerk,
                ) = CliqueTensorStepCentralDifferenceKernel.apply(
                    u_act,
                    start_state.position,  # .contiguous(),
                    start_state.velocity,  # .contiguous(),
                    start_state.acceleration,  # .contiguous(),
                    out_state_seq.position,  # .contiguous(),
                    out_state_seq.velocity,  # .contiguous(),
                    out_state_seq.acceleration,  # .contiguous(),
                    out_state_seq.jerk,  # .contiguous(),
                    self._inv_dt_h,
                    self._u_grad,
                )

        else:
            if self._fd_mode == -1:
                (
                    out_state_seq.position,
                    out_state_seq.velocity,
                    out_state_seq.acceleration,
                    out_state_seq.jerk,
                ) = CliqueTensorStepIdxKernel.apply(
                    u_act,
                    start_state.position,  # .contiguous(),
                    start_state.velocity,  # .contiguous(),
                    start_state.acceleration,  # .contiguous(),
                    start_state_idx,
                    out_state_seq.position,  # .contiguous(),
                    out_state_seq.velocity,  # .contiguous(),
                    out_state_seq.acceleration,  # .contiguous(),
                    out_state_seq.jerk,  # .contiguous(),
                    self._inv_dt_h,
                    self._u_grad,
                )
            else:
                (
                    out_state_seq.position,
                    out_state_seq.velocity,
                    out_state_seq.acceleration,
                    out_state_seq.jerk,
                ) = CliqueTensorStepIdxCentralDifferenceKernel.apply(
                    u_act,
                    start_state.position,  # .contiguous(),
                    start_state.velocity,  # .contiguous(),
                    start_state.acceleration,  # .contiguous(),
                    start_state_idx,
                    out_state_seq.position,  # .contiguous(),
                    out_state_seq.velocity,  # .contiguous(),
                    out_state_seq.acceleration,  # .contiguous(),
                    out_state_seq.jerk,  # .contiguous(),
                    self._inv_dt_h,
                    self._u_grad,
                )
            if self._filter_velocity:
                out_state_seq.aux_data["raw_velocity"] = out_state_seq.velocity
                out_state_seq.velocity = self.filter_signal(out_state_seq.velocity)

            if self._filter_acceleration:
                out_state_seq.aux_data["raw_acceleration"] = out_state_seq.acceleration
                out_state_seq.acceleration = self.filter_signal(out_state_seq.acceleration)

            if self._filter_jerk:
                out_state_seq.aux_data["raw_jerk"] = out_state_seq.jerk
                out_state_seq.jerk = self.filter_signal(out_state_seq.jerk)
        return out_state_seq

    def filter_signal(self, signal: torch.Tensor):
        return filter_signal_jit(signal, self._sma_kernel)


@get_torch_jit_decorator(force_jit=True)
def filter_signal_jit(signal, kernel):
    b, h, dof = signal.shape

    new_signal = (
        torch.nn.functional.conv1d(
            signal.transpose(-1, -2).reshape(b * dof, 1, h), kernel, padding="same"
        )
        .view(b, dof, h)
        .transpose(-1, -2)
        .contiguous()
    )
    return new_signal
