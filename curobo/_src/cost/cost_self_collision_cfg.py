# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional, Type

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_self_collision import SelfCollisionCost

# CuRobo
from curobo._src.robot.types.self_collision_params import (
    SelfCollisionKinematicsCfg,
)


@dataclass
class SelfCollisionCostCfg(BaseCostCfg):
    class_type: Type[SelfCollisionCost] = SelfCollisionCost
    self_collision_kin_config: Optional[SelfCollisionKinematicsCfg] = None
    store_pair_distance: bool = False

    def __post_init__(self):
        return super().__post_init__()
