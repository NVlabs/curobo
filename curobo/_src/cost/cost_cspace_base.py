# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import init_warp

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg


class BaseCSpaceCost(BaseCost):
    def __init__(self, config: CSpaceCostCfg):
        if config.dof < 0:
            log_and_raise("dof needs to be set in BoundCostConfig")
        if config.joint_limits is None:
            log_and_raise("joint limits needs to be set in BoundCostConfig")

        config.joint_limits.validate_shape(config.dof, check_effort=True)
        super().__init__(config)
        init_warp()
    def _init_post_config(self):
        self._cspace_target_weight = self.config.cspace_target_weight.clone()
        self.disable_cspace_target()
        super()._init_post_config()

    def validate_input(
        self,
        state_batch: JointState,
        joint_torque: Optional[torch.Tensor] = None,
        target_joint_state: Optional[JointState] = None,
        idxs_target_joint_state: Optional[torch.Tensor] = None,
    ):
        b, h, dof = state_batch.position.shape
        if b != self._batch_size:
            log_and_raise("state_batch batch size mismatch: {} != {}".format(b, self._batch_size))
        if h != self._horizon:
            log_and_raise("state_batch horizon mismatch: {} != {}".format(h, self._horizon))
        if dof != self.config.dof:
            log_and_raise("state_batch dof mismatch: {} != {}".format(dof, self.config.dof))
        if joint_torque is not None:
            if joint_torque.shape[0] != b:
                log_and_raise(
                    "joint_torque batch size mismatch: {} != {}".format(joint_torque.shape[0], b)
                )
        if target_joint_state is not None:
            if target_joint_state.position.shape[-1] != dof:
                log_and_raise(
                    "target_joint_state position dof mismatch: {} != {}".format(target_joint_state.position.shape[-1], dof)
                )
        if idxs_target_joint_state is not None:
            if idxs_target_joint_state.shape[0] != b:
                log_and_raise(
                    "idxs_target_joint_state batch size mismatch: {} != {}".format(idxs_target_joint_state.shape[0], b)
                )


    @abstractmethod
    def forward(
        self,
        state_batch: JointState,
        joint_torque: Optional[torch.Tensor] = None,
        target_joint_state: Optional[JointState] = None,
        idxs_target_joint_state: Optional[torch.Tensor] = None,
        current_joint_state: Optional[JointState] = None,
        idxs_current_joint_state: Optional[torch.Tensor] = None,
    ):
        pass

    def enable_cspace_target(self):
        self._cspace_target_weight.copy_(self.config.cspace_target_weight)

    def disable_cspace_target(self):
        self._cspace_target_weight[:] = 0.0
