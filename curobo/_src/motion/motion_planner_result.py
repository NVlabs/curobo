# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from curobo._src.util.logging import log_and_raise


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

    @profiler.record_function("grasp_plan_result/get_approach_interpolated_plan")
    def get_approach_interpolated_plan(self) -> Optional[JointState]:
        if self.approach_interpolated_last_tstep is None:
            return self.approach_interpolated_trajectory
        if len(self.approach_interpolated_last_tstep) > 1:
            log_and_raise("only single result is supported")
        return trim_joint_state_trajectory(
            self.approach_interpolated_trajectory, 0, self.approach_interpolated_last_tstep[0],
        )

    @profiler.record_function("grasp_plan_result/get_grasp_interpolated_plan")
    def get_grasp_interpolated_plan(self) -> Optional[JointState]:
        if self.grasp_interpolated_last_tstep is None:
            return self.grasp_interpolated_trajectory
        if len(self.grasp_interpolated_last_tstep) > 1:
            log_and_raise("only single result is supported")
        return trim_joint_state_trajectory(
            self.grasp_interpolated_trajectory, 0, self.grasp_interpolated_last_tstep[0],
        )
        
    @profiler.record_function("grasp_plan_result/get_lift_interpolated_plan")
    def get_lift_interpolated_plan(self) -> Optional[JointState]:
        if self.lift_interpolated_last_tstep is None:
            return self.lift_interpolated_trajectory
        if len(self.lift_interpolated_last_tstep) > 1:
            log_and_raise("only single result is supported")
        return trim_joint_state_trajectory(
            self.lift_interpolated_trajectory, 0, self.lift_interpolated_last_tstep[0],
        )
