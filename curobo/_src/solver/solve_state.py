# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Solve state types for optimization-based solvers."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional

# CuRobo
from curobo._src.solver.solve_mode import SolveMode


@dataclass
class SolveState:
    """Dataclass for storing the current problem type of a solver.

    This stores metadata about the optimization problem being solved,
    including batch sizes, number of seeds, and solve type.

    Attributes:
        solve_type: Batch mode (SINGLE, BATCH, or MULTI_ENV).
        batch_size: Number of problems in the batch.
        num_envs: Number of environments in the batch.
        num_goalset: Number of goals per problem (for goalset problems).
        multi_env: Whether problems use different environments.
        batch_mode: Whether there is more than 1 problem.
        num_seeds: Number of seeds for each problem.
        num_ik_seeds: Number of seeds for IK problems.
        num_graph_seeds: Number of seeds for graph search.
        num_trajopt_seeds: Number of seeds for trajectory optimization.
        tool_frames: Names of target links for goal pose tracking.
    """

    solve_type: SolveMode
    batch_size: int
    num_envs: int
    num_goalset: int = 1
    multi_env: bool = False
    batch_mode: bool = False
    num_seeds: Optional[int] = None
    num_ik_seeds: Optional[int] = None
    num_graph_seeds: Optional[int] = None
    num_trajopt_seeds: Optional[int] = None
    tool_frames: Optional[List[str]] = None

    def __post_init__(self):
        """Post init to set default flags based on input values."""
        if self.num_envs == 1:
            self.multi_env = False
        else:
            self.multi_env = True
        if self.batch_size > 1:
            self.batch_mode = True
        if self.num_seeds is None:
            self.num_seeds = self.num_ik_seeds
        if self.num_seeds is None:
            self.num_seeds = self.num_trajopt_seeds
        if self.num_seeds is None:
            self.num_seeds = self.num_graph_seeds

    def clone(self) -> "SolveState":
        """Create a deep copy of the solve state."""
        return SolveState(
            solve_type=self.solve_type,
            num_envs=self.num_envs,
            batch_size=self.batch_size,
            num_goalset=self.num_goalset,
            multi_env=self.multi_env,
            batch_mode=self.batch_mode,
            num_seeds=self.num_seeds,
            num_ik_seeds=self.num_ik_seeds,
            num_graph_seeds=self.num_graph_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            tool_frames=self.tool_frames,
        )

    def get_batch_size(self) -> int:
        """Get total number of optimization problems including seeds."""
        return self.num_seeds * self.batch_size

    def get_ik_batch_size(self) -> int:
        """Get total number of IK problems including seeds."""
        if self.num_ik_seeds is None:
            return 0
        return self.num_ik_seeds * self.batch_size

    def get_trajopt_batch_size(self) -> int:
        """Get total number of TrajOpt problems including seeds."""
        if self.num_trajopt_seeds is None:
            return 0
        return self.num_trajopt_seeds * self.batch_size


@dataclass
class MotionPlanSolveState:
    """Dataclass for storing the current state of a motion planner."""

    solve_type: SolveMode
    ik_solve_state: SolveState
    trajopt_solve_state: SolveState

