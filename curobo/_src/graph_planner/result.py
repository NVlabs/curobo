# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library
from dataclasses import dataclass
from typing import Any, List, Optional, Union

# Third Party
import torch

# CuRobo


@dataclass
class GraphPlannerResult:
    """Data class stores information about the graph planner result."""

    #: Success flag for each query, shape (B)
    success: torch.Tensor

    #: Plan waypoints, shape (B, N, action_dim)
    plan_waypoints: Optional[List[Union[torch.Tensor, None]]] = None

    #: Interpolated waypoints, shape (B, interpolation_steps, action_dim)
    interpolated_waypoints: Optional[torch.Tensor] = None

    #: Joint names for each index in action_dim.
    joint_names: Optional[List[str]] = None

    #: Path length, shape (B)
    path_length: Optional[torch.Tensor] = None

    #: Solve time, shape (B)
    solve_time: float = 0.0

    #: Valid query, shape (B)
    valid_query: bool = True

    #: Debug info
    debug_info: Optional[Any] = None
