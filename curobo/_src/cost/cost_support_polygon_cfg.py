# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type

import torch

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_support_polygon import CostSupportPolygon


@dataclass
class CostSupportPolygonCfg(BaseCostCfg):
    class_type: Type[CostSupportPolygon] = CostSupportPolygon
    foot_sphere_indices: Optional[torch.Tensor] = None
    foot_link_names: Optional[List[str]] = None
    inside_cost_weight: float = 0.001  # Small weight to encourage being more centered when inside
