# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.state.state_joint import JointState


@dataclass
class MotionPlannerResult:
    """Result of a motion planning operation."""
    success: Optional[torch.Tensor] = None


@dataclass
class GraspPlanResult:
    """Result of a grasp planning operation."""

    success: Optional[torch.Tensor] = None
    approach_success: Optional[torch.Tensor] = None
    grasp_success: Optional[torch.Tensor] = None
    lift_success: Optional[torch.Tensor] = None
    approach_trajectory: Optional[JointState] = None
    approach_trajectory_dt: Optional[torch.Tensor] = None
    approach_interpolated_trajectory: Optional[JointState] = None
    grasp_trajectory: Optional[JointState] = None
    grasp_trajectory_dt: Optional[torch.Tensor] = None
    grasp_interpolated_trajectory: Optional[JointState] = None
    lift_trajectory: Optional[JointState] = None
    lift_trajectory_dt: Optional[torch.Tensor] = None
    lift_interpolated_trajectory: Optional[JointState] = None
    approach_interpolated_last_tstep: Optional[torch.Tensor] = None
    grasp_interpolated_last_tstep: Optional[torch.Tensor] = None
    lift_interpolated_last_tstep: Optional[torch.Tensor] = None
    status: Optional[str] = None
    planning_time: float = 0.0
    goalset_index: Optional[torch.Tensor] = None

