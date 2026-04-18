# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional

# CuRobo
from curobo._src.solver.solver_base_result import BaseSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState


@dataclass
class MPCSolverResult(BaseSolverResult):
    """Result specific to the MPC solver."""

    next_action: Optional[JointState] = None
    action_sequence: Optional[JointState] = None
    full_action_sequence: Optional[JointState] = None
    robot_state_sequence: Optional[RobotState] = None
    action_buffer: Optional[torch.Tensor] = None
    action_dt: Optional[float] = None
    def clone(self) -> MPCSolverResult:
        base_clone = super().clone()
        return MPCSolverResult(
            success=base_clone.success,
            solution=base_clone.solution,
            js_solution=base_clone.js_solution,
            position_error=base_clone.position_error,
            rotation_error=base_clone.rotation_error,
            cspace_error=base_clone.cspace_error,
            goalset_index=base_clone.goalset_index,
            solve_time=base_clone.solve_time,
            total_time=base_clone.total_time,
            debug_info=base_clone.debug_info,
            optimized_seeds=base_clone.optimized_seeds,
            metrics=base_clone.metrics,
            position_tolerance=base_clone.position_tolerance,
            orientation_tolerance=base_clone.orientation_tolerance,
            seed_rank=base_clone.seed_rank,
            seed_cost=base_clone.seed_cost,
            batch_size=base_clone.batch_size,
            num_seeds=base_clone.num_seeds,
            total_cost_reshaped=base_clone.total_cost_reshaped,
            solution_state=base_clone.solution_state,
            feasible=base_clone.feasible,
            next_action=self.next_action.clone() if self.next_action is not None else None,
            action_sequence=(
                self.action_sequence.clone() if self.action_sequence is not None else None
            ),
            full_action_sequence=(
                self.full_action_sequence.clone()
                if self.full_action_sequence is not None
                else None
            ),
            robot_state_sequence=(
                self.robot_state_sequence.clone()
                if self.robot_state_sequence is not None
                else None
            ),
            action_buffer=(
                self.action_buffer.clone() if self.action_buffer is not None else None
            ),
            action_dt=self.action_dt,
        )

