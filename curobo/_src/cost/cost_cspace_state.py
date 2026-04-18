# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from curobo._src.cost.cost_cspace_base import BaseCSpaceCost
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.cost.wp_cspace_state import StateCSpaceFunction
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg




class StateCSpaceCost(BaseCSpaceCost):
    def __init__(self, config: CSpaceCostCfg):
        if config.cost_type != CSpaceCostType.STATE:
            log_and_raise("StateCSpaceCost can only be used for state cost")
        super().__init__(config)
        self._target_joint_state = None
        self._idxs_target_joint_state = None

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

            self._out_gv_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._out_ga_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._out_gj_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            # create terminal buffers:
            self._run_weight_vel = torch.ones(
                (1, horizon),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._run_weight_acc = torch.ones(
                (1, horizon),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            self._run_weight_jerk = torch.ones(
                (1, horizon),
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
        self.validate_input(state_batch, joint_torque, target_joint_state, idxs_target_joint_state)

        if target_joint_state is None:
            target_joint_state = self._target_joint_state
        if idxs_target_joint_state is None:
            idxs_target_joint_state = self._idxs_target_joint_state
        if target_joint_state is None:
            log_and_raise("target_joint_state is required, call setup_batch_tensors()  first")
        if joint_torque is None:
            joint_torque = self._joint_torque_buffer

        cost = StateCSpaceFunction.apply(
            state_batch.position,
            state_batch.velocity,
            state_batch.acceleration,
            state_batch.jerk,
            joint_torque,
            state_batch.dt,
            target_joint_state.position,
            idxs_target_joint_state,
            self.config.joint_limits.position,
            self.config.joint_limits.velocity,
            self.config.joint_limits.acceleration,
            self.config.joint_limits.jerk,
            self.config.joint_limits.effort,
            self._weight,
            self.config.activation_distance,
            self.config.squared_l2_regularization_weight,
            self._cspace_target_weight,
            self.config.cspace_non_terminal_weight_factor,
            self.config.cspace_target_dof_weight,
            self._out_c_buffer,
            self._out_gp_buffer,
            self._out_gv_buffer,
            self._out_ga_buffer,
            self._out_gj_buffer,
            self._out_gtau_buffer,
            self.config.retime_weights,
            self.config.retime_regularization_weights,
            self.config.use_grad_input,
        )



        return cost
