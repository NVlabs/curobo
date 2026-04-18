# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Type, Union

import torch

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_cspace_dist import CSpaceDistCost
from curobo._src.transition.robot_state_transition import RobotStateTransition


@dataclass
class CSpaceDistCostCfg(BaseCostCfg):
    class_type: Type[CSpaceDistCost] = CSpaceDistCost
    use_null_space: bool = False
    only_terminal_cost: bool = True
    terminal_dof_weight: Union[torch.Tensor, List[float]] = None
    non_terminal_dof_weight: Union[torch.Tensor, List[float]] = None

    def __post_init__(self):
        super().__post_init__()

    def initialize_from_transition_model(self, transition_model: RobotStateTransition):
        self.update_dof(transition_model.action_dim)

        if self.use_null_space:
            self.update_terminal_dof_weight(transition_model.null_space_weight)
            self.update_non_terminal_dof_weight(transition_model.null_space_weight)
        else:
            self.update_terminal_dof_weight(transition_model.cspace_distance_weight)
            self.update_non_terminal_dof_weight(transition_model.cspace_distance_weight)

    def update_terminal_dof_weight(self, dof_weight: Union[torch.Tensor, List[float]]):
        self.terminal_dof_weight[:] = self.device_cfg.to_device(dof_weight)

    def update_non_terminal_dof_weight(
        self, non_terminal_dof_weight: Union[torch.Tensor, List[float]]
    ):
        self.non_terminal_dof_weight[:] = self.device_cfg.to_device(non_terminal_dof_weight)
        if self.only_terminal_cost:
            self.non_terminal_dof_weight[:] = 0.0

    def update_dof(self, dof: int):
        self.dof = dof
        self.terminal_dof_weight = torch.ones(
            dof, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        self.non_terminal_dof_weight = torch.ones(
            dof, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
