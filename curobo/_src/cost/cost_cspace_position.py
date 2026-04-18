# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from curobo._src.cost.cost_cspace_base import BaseCSpaceCost
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.cost.wp_cspace_position import PositionCSpaceFunction
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg


class PositionCSpaceCost(BaseCSpaceCost):
    def __init__(self, config: CSpaceCostCfg):

        if config.cost_type != CSpaceCostType.POSITION:
            log_and_raise("CSpacePositionCost can only be used for position cost")
        super().__init__(config)
        self._target_joint_state = None
        self._idxs_target_joint_state = None
        self._current_joint_state = None
        self._idxs_current_joint_state = None
        self._default_dt = None
        self._current_velocity_buffer = None


    def setup_batch_tensors(self, batch: int, horizon: int):
        if self._batch_size != batch or self._horizon != horizon:
            self._out_c_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            self._out_gp_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._out_gtau_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            self._target_joint_state = JointState.from_position(position=torch.zeros(
                (self.config.dof), device=self.device_cfg.device, dtype=self.device_cfg.dtype
            ))

            self._idxs_target_joint_state = torch.zeros(
                (batch), device=self.device_cfg.device, dtype=torch.int32
            )
            self._joint_torque_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            self._current_joint_state = JointState.from_position(position=torch.zeros(
                (self.config.dof), device=self.device_cfg.device, dtype=self.device_cfg.dtype
            ))
            self._idxs_current_joint_state = torch.zeros(
                (batch), device=self.device_cfg.device, dtype=torch.int32
            )
            self._default_dt = torch.zeros(
                (batch), device=self.device_cfg.device, dtype=self.device_cfg.dtype
            )
            self._current_velocity_buffer = torch.zeros(
                (self.config.dof), device=self.device_cfg.device, dtype=self.device_cfg.dtype
            )

            super().setup_batch_tensors(batch, horizon)

    def forward(
        self,
        state_batch: JointState,
        joint_torque: Optional[torch.Tensor] = None,
        target_joint_state: Optional[JointState] = None,
        idxs_target_joint_state: Optional[torch.Tensor] = None,
        current_joint_state: Optional[JointState] = None,
        idxs_current_joint_state: Optional[torch.Tensor] = None,
        current_state_dt: Optional[torch.Tensor] = None,
    ):
        """Computes the bound cost for the position of the robot.

        There are several terms in this cost:
        1. Joint position limit term (optionally tightened by velocity bounds when dt > 0)
        2. Joint torque limit term
        3. Target joint position term (squared error regularization)

        When ``current_state_dt`` (or ``current_joint_state.dt``) is non-zero, position bounds
        are tightened to
        ``[max(p_lower, current_p + v_lower*dt), min(p_upper, current_p + v_upper*dt)]``.

        Args:
            state_batch: The state batch of the robot. Shape is (batch, horizon, dof).
            joint_torque: The joint torque of the robot. Shape is (batch, horizon, dof).
            target_joint_state: The target/retract joint state. Shape is (-1, dof).
            idxs_target_joint_state: Index for target joint state per batch. Shape is (batch,).
            current_joint_state: Current joint state for velocity clamping. Shape is (-1, dof).
            idxs_current_joint_state: Index for current joint state per batch. Shape is (batch,).
            current_state_dt: Time since the current state was observed. Shape: ``(n_problems,)``.
                Takes priority over ``current_joint_state.dt``.

        Returns:
            The bound cost for the position of the robot. Shape is (batch, horizon, dof).
        """
        self.validate_input(
            state_batch,
            joint_torque,
            target_joint_state,
            idxs_target_joint_state,
        )

        if target_joint_state is None:
            target_joint_state = self._target_joint_state
        if idxs_target_joint_state is None:
            idxs_target_joint_state = self._idxs_target_joint_state
        if target_joint_state is None:
            log_and_raise("target_joint_state is required, call setup_batch_tensors() first")
        if joint_torque is None:
            joint_torque = self._joint_torque_buffer

        if current_joint_state is None:
            current_joint_state = self._current_joint_state
        if idxs_current_joint_state is None:
            idxs_current_joint_state = self._idxs_current_joint_state

        if current_state_dt is not None:
            state_dt = current_state_dt
        elif current_joint_state.dt is not None:
            state_dt = current_joint_state.dt
        else:
            state_dt = self._default_dt

        if (
            current_joint_state is not None
            and current_joint_state.velocity is not None
        ):
            current_velocity = current_joint_state.velocity
        else:
            current_velocity = self._current_velocity_buffer

        cost = PositionCSpaceFunction.apply(
            state_batch.position,
            joint_torque,
            target_joint_state.position,
            idxs_target_joint_state,
            self.config.joint_limits.position,
            self.config.joint_limits.effort,
            self._weight,
            self.config.activation_distance,
            self._cspace_target_weight,
            self.config.cspace_target_dof_weight,
            self.config.squared_l2_regularization_weight,
            current_joint_state.position,
            current_velocity,
            idxs_current_joint_state,
            self.config.joint_limits.velocity,
            state_dt,
            self._out_c_buffer,
            self._out_gp_buffer,
            self._out_gtau_buffer,
            self.config.use_grad_input,
        )
        return cost
