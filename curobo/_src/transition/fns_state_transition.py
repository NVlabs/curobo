# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standard Library
from abc import abstractmethod
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float16_tensors,
    check_float32_tensors,
)
from curobo._src.curobolib.cuda_ops.trajectory import (
    AccelerationTensorStepIdxKernel,
    BSplineIdxKernel,
    CliqueTensorStepIdxKernel,
)
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


class StateFromBase:
    def __init__(
        self, device_cfg: DeviceCfg, batch_size: int = 1, horizon: int = 1
    ) -> None:
        self.batch_size = -1
        self.horizon = -1
        self.device_cfg = device_cfg
        self._diag_dt = None
        self._inv_dt_h = None
        self.action_horizon = horizon
        self.update_batch_size(batch_size, horizon)

    def update_dt(self, dt: float):
        if self._dt_h is not None:
            self._dt_h[:] = dt
        if self._inv_dt_h is not None:
            self._inv_dt_h[:] = 1.0 / dt

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
        **kwargs,
    ) -> JointState:
        pass


class StateFromPositionTeleport(StateFromBase):
    def __init__(
        self,
        device_cfg: DeviceCfg,
        batch_size: int = 1,
        horizon: int = 1,
    ) -> None:
        super().__init__(device_cfg, batch_size=batch_size, horizon=horizon)

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        if u_act.shape != out_state_seq.position.shape:
            log_and_raise(f"Shape mismatch: u_act.shape != out_state_seq.position.shape: {u_act.shape} != {out_state_seq.position.shape}")
        out_state_seq.position.copy_(u_act)
        return out_state_seq


class StateFromAcceleration(StateFromBase):
    def __init__(
        self,
        device_cfg: DeviceCfg,
        dt_h: torch.Tensor,
        dof: int,
        batch_size: int = 1,
        horizon: int = 1,
    ) -> None:
        self.dof = dof

        super().__init__(device_cfg, batch_size=batch_size, horizon=horizon)

        self._dt_h = dt_h
        self._u_grad = torch.zeros(
            (batch_size, horizon, self.dof),
            device=device_cfg.device,
            dtype=device_cfg.dtype,
        )

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        if batch_size != self.batch_size or horizon != self.horizon:
            self._u_grad = torch.zeros(
                (batch_size, horizon, self.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
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
        **kwargs,
    ) -> JointState:
        if start_state_idx is None:
            log_and_raise("Start state index is required for Acceleration kernel")

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


class StateFromPositionClique(StateFromBase):
    def __init__(
        self,
        device_cfg: DeviceCfg,
        dt_h: torch.Tensor,
        dof: int,
        filter_velocity: bool = False,
        filter_acceleration: bool = False,
        filter_jerk: bool = False,
        batch_size: int = 1,
        horizon: int = 1,
    ) -> None:
        """Compute state sequence from action using Clique kernel.

        Only CUDA devices are supported.

        Args:
            device_cfg: Device and dtype for the tensors.
            dt_h: Time step for the state sequence. Shape is (batch_size,).
            dof: Number of degrees of freedom.
            filter_velocity: Whether to filter the velocity.
            filter_acceleration: Whether to filter the acceleration.
            filter_jerk: Whether to filter the jerk.
            batch_size: Number of batch elements.
            horizon: Number of steps in the state sequence.
        """
        self.dof = dof

        self._dt_h = dt_h
        self._inv_dt_h = 1.0 / dt_h
        self._u_grad = None
        self._filter_velocity = filter_velocity
        self._filter_acceleration = filter_acceleration
        self._filter_jerk = filter_jerk
        super().__init__(device_cfg, batch_size=batch_size, horizon=horizon)

        if self._filter_velocity or self._filter_acceleration or self._filter_jerk:
            kernel = self.device_cfg.to_device([[[0.06136, 0.24477, 0.38774, 0.24477, 0.06136]]])
            self._sma = torch.nn.functional.conv1d

            weights = kernel
            self._sma_kernel = weights

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        if batch_size != self.batch_size or horizon != self.horizon or self._u_grad is None:
            self.action_horizon = horizon - 4
            self._u_grad = torch.zeros(
                (batch_size, self.action_horizon, self.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
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
        goal_state: Optional[JointState] = None,
        goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        """Compute state sequence from action using Clique kernel.

        Args:
            start_state: Start state of the sequence. Shape is (-1, dof). The index is obtained
                using start_state_idx. E.g., for batch_index = 0,
                start_state_value = start_state_seq[start_state_idx[0]].
            u_act: Action to apply to the start state. Shape is (batch_size, horizon - 4, dof).
            out_state_seq: Output state sequence. Shape is (batch_size, horizon, dof).
            start_state_idx: Index of the start state in the sequence. Shape is (batch_size,). This
                is of dtype torch.int32.
            goal_state: Goal state of the sequence. Shape is (-1, dof). The index is obtained
                using goal_state_idx. E.g., for batch_index = 0,
                goal_state_value = goal_state_seq[goal_state_idx[0]].
            goal_state_idx: Index of the goal state in the sequence. Shape is (batch_size,). This
                is of dtype torch.int32.
            use_implicit_goal_state: Whether to use the goal state as the goal state of the sequence.
                Shape is (-1,), indexed using goal_state_idx. E.g., for batch_index = 0,
                use_implicit_goal_state_value = use_implicit_goal_state[goal_state_idx[0]].
                This is of dtype torch.uint8.

        Returns:
            out_state_seq: Output state sequence. Shape is (batch_size, horizon, dof).
        """
        if start_state_idx is None:
            log_and_raise("Start state index is required for Clique kernel")

        if goal_state is None:
            goal_state = start_state
        if goal_state_idx is None:
            goal_state_idx = start_state_idx
        if goal_state.dt.shape != goal_state.position.shape[0:2]:
            log_and_raise(
                f"Shape mismatch: goal_state.dt.shape[0] != goal_state.position.shape[0:2]: {goal_state.dt.shape} != {goal_state.position.shape[0:2]}"
            )
        if use_implicit_goal_state.shape != goal_state.position.shape[0:2]:
            log_and_raise(
                f"Shape mismatch: use_implicit_goal_state.shape[0] != goal_state.position.shape[0]: {use_implicit_goal_state.shape[0]} != {goal_state.position.view(-1, goal_state.position.shape[-1]).shape[0]}"
            )

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
            goal_state.position,
            goal_state.velocity,
            goal_state.acceleration,
            start_state_idx,
            goal_state_idx,
            out_state_seq.position,  # .contiguous(),
            out_state_seq.velocity,  # .contiguous(),
            out_state_seq.acceleration,  # .contiguous(),
            out_state_seq.jerk,  # .contiguous(),
            out_state_seq.dt,
            goal_state.dt,
            use_implicit_goal_state,
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


class StateFromBSplineKnot(StateFromBase):
    def __init__(
        self,
        device_cfg: DeviceCfg,
        dof: int,
        batch_size: int = 1,
        horizon: int = 1,
        n_knots: int = 4,
        interpolation_steps: int = 1,
        use_implicit_goal_state: bool = False,
        control_space: ControlSpace = ControlSpace.BSPLINE_4,
    ) -> None:
        """Compute state sequence from action using BSpline kernel.

        Only CUDA devices are supported. Double is not supported (only float32 is supported).

        n_knots needs to be greater than 5. Given n_knots, the horizon is computed as:
        - BSPLINE_3: horizon = (n_knots + 2) * interpolation_steps + 1
        - BSPLINE_4: horizon = (n_knots + 3) * interpolation_steps + 1
        - BSPLINE_5: horizon = (n_knots + 4) * interpolation_steps + 1
        A good value for interpolation_steps is 4.

        Args:
            device_cfg: Device and dtype for the tensors.
            dof: Number of degrees of freedom.
            batch_size: Number of batch elements.
            horizon: Number of steps in the state sequence.
            n_knots: Number of knots in the BSpline.
            use_implicit_goal_state: Whether to use the goal state as the goal state of the sequence.
            control_space: Control space to use. This should be one of BSPLINE_3, BSPLINE_4, or BSPLINE_5.
        """
        self.dof = dof
        self._u_grad = None
        self.n_knots = n_knots
        self.use_implicit_goal_state = use_implicit_goal_state
        self.control_space = control_space
        self.bspline_degree = ControlSpace.spline_degree(control_space)
        self.interpolation_steps = interpolation_steps
        self.padded_horizon = ControlSpace.spline_total_interpolation_steps(
            control_space, n_knots, interpolation_steps
        )
        super().__init__(device_cfg, batch_size=batch_size, horizon=horizon)

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        if batch_size != self.batch_size or horizon != self.horizon:
            self.action_horizon = self.n_knots
            self._u_grad = torch.zeros(
                (batch_size, self.n_knots, self.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
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
        goal_state: Optional[JointState] = None,
        goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        """Compute state sequence from action using BSpline kernel.

        Args:
            start_state: Start state of the sequence. Shape is (-1, dof). The index is obtained
                using start_state_idx. E.g., for batch_index = 0,
                start_state_value = start_state_seq[start_state_idx[0]].
            u_act: Action to apply to the start state. Shape is (batch_size, n_knots, dof).
            out_state_seq: Output state sequence. Shape is (batch_size, horizon, dof). Note that
                the horizon is computed using interpolation steps and also the type of BSpline. See
                the docstring of StateFromBSplineKnot for more details.
            start_state_idx: Index of the start state in the sequence. Shape is (batch_size,). This
                is of dtype torch.int32.
            goal_state: Goal state of the sequence. Shape is (-1, dof). The index is obtained
                using goal_state_idx. E.g., for batch_index = 0,
                goal_state_value = goal_state_seq[goal_state_idx[0]].
            goal_state_idx: Index of the goal state in the sequence. Shape is (batch_size,). This
                is of dtype torch.int32.
            use_implicit_goal_state: Whether to use the goal state as the goal state of the sequence.
                Shape is (-1,), indexed using goal_state_idx. E.g., for batch_index = 0,
                use_implicit_goal_state_value = use_implicit_goal_state[goal_state_idx[0]].

        Returns:
            out_state_seq: Output state sequence. Shape is (batch_size, horizon, dof).
        """
        if self.use_implicit_goal_state:
            if goal_state is None:
                log_and_raise("Goal state is not provided for implicit goal state")
            if start_state_idx is not None:
                if goal_state_idx is None:
                    log_and_raise("Goal state index is not provided for implicit goal state")
        else:
            if goal_state is None:
                goal_state = start_state
            if goal_state_idx is None:
                goal_state_idx = start_state_idx
        if start_state_idx is None:
            log_and_raise("Start state index is required for BSpline kernel")
        if goal_state_idx is None:
            log_and_raise("idx is None")
        if goal_state.dt is None:
            log_and_raise("dt is None")
        if use_implicit_goal_state is None:
            log_and_raise("use_implicit_goal_state is None")
        if goal_state_idx.shape[0] != u_act.shape[0]:
            log_and_raise(
                f"Shape mismatch: goal_state_idx.shape[0] != u_act.shape[0]: {goal_state_idx.shape[0]} != {u_act.shape[0]}"
            )
        if use_implicit_goal_state.shape[0] != goal_state.shape[0]:
            log_and_raise(
                f"Shape mismatch: use_implicit_goal_state.shape[0] != goal_state.shape[0]: {use_implicit_goal_state.shape[0]} != {goal_state.shape[0]}"
            )

        if start_state.jerk is None:
            log_and_raise("start jerk is None")
        if out_state_seq.dt is None:
            log_and_raise("out dt is None")
        if u_act.shape[1] != self.n_knots:
            log_and_raise(f"u_act.shape[1] != self.n_knots: {u_act.shape[1]} != {self.n_knots}")
        if self.padded_horizon != out_state_seq.shape[1]:
            log_and_raise(
                f"padded_horizon != out_state_seq.shape[1]: {self.padded_horizon} != {out_state_seq.shape[1]}"
            )

        (
            out_state_seq.position,
            out_state_seq.velocity,
            out_state_seq.acceleration,
            out_state_seq.jerk,
        ) = BSplineIdxKernel.apply(
            u_act,
            start_state.position,  # .contiguous(),
            start_state.velocity,  # .contiguous(),
            start_state.acceleration,  # .contiguous(),
            start_state.jerk,
            goal_state.position,  # .contiguous(),
            goal_state.velocity,  # .contiguous(),
            goal_state.acceleration,  # .contiguous(),
            goal_state.jerk,
            start_state_idx,
            goal_state_idx,
            out_state_seq.position,  # .contiguous(),
            out_state_seq.velocity,  # .contiguous(),
            out_state_seq.acceleration,  # .contiguous(),
            out_state_seq.jerk,  # .contiguous(),
            out_state_seq.dt,
            goal_state.dt,
            use_implicit_goal_state,
            self._u_grad,
            self.bspline_degree,
        )

        return out_state_seq


@get_torch_jit_decorator(force_jit=True, slow_to_compile=True)
def _filter_signal_jit_core(signal, kernel):
    b, h, dof = signal.shape

    return (
        torch.nn.functional.conv1d(
            signal.transpose(-1, -2).reshape(b * dof, 1, h), kernel, padding="same"
        )
        .view(b, dof, h)
        .transpose(-1, -2)
        .reshape(b * h * dof)
        .reshape(b, h, dof)
    )


def filter_signal_jit(signal, kernel):
    out = _filter_signal_jit_core(signal, kernel)
    (
        check_float16_tensors if signal.dtype == torch.float16 else check_float32_tensors
    )(out.device, filtered=out)
    return out
