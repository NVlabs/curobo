# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Base result dataclass for solvers."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import Dict, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.runtime import debug as debug_mode
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import copy_joint_state_at_batch_seed_indices
from curobo._src.state.state_robot import RobotState
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


@dataclass
class BaseSolverResult:
    """Base result dataclass for solvers.

    This result structure is shared by IKSolver, TrajOptSolver, and MPCSolver.

    Attributes:
        success: Boolean tensor indicating success per batch item.
        solution: Optimized action sequence or configuration.
        js_solution: Full joint state solution (possibly trajectory).
        position_error: Position error tensor.
        rotation_error: Rotation error tensor.
        cspace_error: Configuration space error.
        goalset_index: Index into goalset for successful solutions.
        solve_time: Time spent in optimization.
        total_time: Total time including setup.
        debug_info: Debug information dictionary.
        optimized_seeds: Optimized seeds for potential reuse.
        metrics: Rollout metrics from evaluation.
        position_tolerance: Position tolerance used for success check.
        orientation_tolerance: Orientation tolerance used for success check.
        seed_rank: Ranking of seeds by cost.
        seed_cost: Cost of each seed.
        batch_size: Number of problems in batch.
        num_seeds: Number of seeds per problem.
        total_cost_reshaped: Total cost reshaped for analysis.
        solution_state: Full robot state at solution.
    """

    success: torch.Tensor
    solution: Optional[torch.Tensor] = None
    js_solution: Optional[JointState] = None
    position_error: Optional[torch.Tensor] = None
    rotation_error: Optional[torch.Tensor] = None
    cspace_error: Optional[torch.Tensor] = None
    goalset_index: Optional[torch.Tensor] = None
    solve_time: float = 0.0
    total_time: float = 0.0
    debug_info: Dict = field(default_factory=dict)
    optimized_seeds: Optional[torch.Tensor] = None
    metrics: Optional[RolloutMetrics] = None
    position_tolerance: float = 0.0
    orientation_tolerance: float = 0.0
    seed_rank: Optional[torch.Tensor] = None
    seed_cost: Optional[torch.Tensor] = None
    batch_size: int = 0
    num_seeds: int = 0
    total_cost_reshaped: Optional[torch.Tensor] = None
    solution_state: Optional[RobotState] = None
    feasible: Optional[torch.Tensor] = None
    """Boolean tensor indicating constraint feasibility per (batch, seed). True when all
    constraints (collision, joint limits) are satisfied, independent of pose convergence."""

    @profiler.record_function("solver_base_result/clone")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def clone(self) -> "BaseSolverResult":
        """Create a deep copy of the result."""
        new_debug = {}
        if self.debug_info is not None:
            for k, v in self.debug_info.items():
                if isinstance(v, torch.Tensor):
                    new_debug[k] = v.clone()
                else:
                    new_debug[k] = v
        if debug_mode:
            if self.seed_rank is not None:
                if torch.max(self.seed_rank) > self.num_seeds:
                    log_and_raise(
                        f"clone() existing self.seed_rank: {self.seed_rank} > "
                        f"self.num_seeds: {self.num_seeds}"
                    )

        return BaseSolverResult(
            success=self.success.clone() if self.success is not None else None,
            solution=self.solution.clone() if self.solution is not None else None,
            js_solution=self.js_solution.clone() if self.js_solution is not None else None,
            position_error=(
                self.position_error.clone() if self.position_error is not None else None
            ),
            rotation_error=(
                self.rotation_error.clone() if self.rotation_error is not None else None
            ),
            cspace_error=self.cspace_error.clone() if self.cspace_error is not None else None,
            goalset_index=self.goalset_index.clone() if self.goalset_index is not None else None,
            solve_time=self.solve_time,
            total_time=self.total_time,
            debug_info=new_debug,
            optimized_seeds=(
                self.optimized_seeds.clone() if self.optimized_seeds is not None else None
            ),
            metrics=self.metrics.clone() if self.metrics is not None else None,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            seed_rank=self.seed_rank.clone() if self.seed_rank is not None else None,
            seed_cost=self.seed_cost.clone() if self.seed_cost is not None else None,
            batch_size=self.batch_size,
            num_seeds=self.num_seeds,
            total_cost_reshaped=(
                self.total_cost_reshaped.clone() if self.total_cost_reshaped is not None else None
            ),
            solution_state=(
                self.solution_state.clone() if self.solution_state is not None else None
            ),
            feasible=self.feasible.clone() if self.feasible is not None else None,
        )

    @profiler.record_function("solver_base_result/copy_successful_solutions")
    def copy_successful_solutions(self, other: "BaseSolverResult") -> None:
        """Copy successful solutions from other result to self.

        Args:
            other: Another result to copy successful solutions from.
        """
        if self.success is None:
            log_and_raise("success is not set")
        if other.success is None:
            log_and_raise("other.success is not set")
        batch_idx, seed_idx = other.success.nonzero(as_tuple=True)
        self.success[batch_idx, seed_idx] = True
        if self.position_error is not None:
            self.position_error[batch_idx, seed_idx] = other.position_error[batch_idx, seed_idx]
        if self.rotation_error is not None:
            self.rotation_error[batch_idx, seed_idx] = other.rotation_error[batch_idx, seed_idx]
        if self.goalset_index is not None:
            self.goalset_index[batch_idx, seed_idx] = other.goalset_index[batch_idx, seed_idx]

        if self.seed_cost is not None:
            if self.seed_cost.shape != other.seed_cost.shape:
                log_and_raise(
                    f"self.seed_cost.shape: {self.seed_cost.shape} != "
                    f"other.seed_cost.shape: {other.seed_cost.shape}"
                )
            self.seed_cost[batch_idx, seed_idx] = other.seed_cost[batch_idx, seed_idx]

        if self.seed_rank is not None:
            if debug_mode:
                if torch.max(other.seed_rank) > self.num_seeds:
                    log_and_raise(
                        f"other.seed_rank: {other.seed_rank} > self.num_seeds: {self.num_seeds}"
                    )
                if torch.min(other.seed_rank) < 0:
                    log_and_raise(f"other.seed_rank: {other.seed_rank} < 0")
                if torch.max(self.seed_rank) > self.num_seeds:
                    log_and_raise(
                        f"existing self.seed_rank: {self.seed_rank} > "
                        f"self.num_seeds: {self.num_seeds}"
                    )
                if torch.min(self.seed_rank) < 0:
                    log_and_raise(f"existing self.seed_rank: {self.seed_rank} < 0")
            if self.seed_rank.shape != other.seed_rank.shape:
                log_and_raise(
                    f"self.seed_rank.shape: {self.seed_rank.shape} != "
                    f"other.seed_rank.shape: {other.seed_rank.shape}"
                )
            self.seed_rank[batch_idx, seed_idx] = other.seed_rank[batch_idx, seed_idx]
            if debug_mode:
                if torch.max(self.seed_rank) > self.num_seeds:
                    log_and_raise(
                        f"copied self.seed_rank: {self.seed_rank} > "
                        f"self.num_seeds: {self.num_seeds}"
                    )
                if torch.min(self.seed_rank) < 0:
                    log_and_raise(f"copied self.seed_rank: {self.seed_rank} < 0")
        if self.total_cost_reshaped is not None:
            self.total_cost_reshaped[batch_idx, seed_idx] = other.total_cost_reshaped[
                batch_idx, seed_idx
            ]

        if self.optimized_seeds is not None:
            self.optimized_seeds[batch_idx, seed_idx, :, :] = other.optimized_seeds[
                batch_idx, seed_idx, :, :
            ]
        if self.solution is not None:
            self.solution[batch_idx, seed_idx, :, :] = other.solution[batch_idx, seed_idx, :, :]
        if self.js_solution is not None:
            copy_joint_state_at_batch_seed_indices(
                self.js_solution, other.js_solution, batch_idx, seed_idx
            )

        if self.feasible is not None and other.feasible is not None:
            self.feasible[batch_idx, seed_idx] = other.feasible[batch_idx, seed_idx]

        if self.metrics is not None:
            flat_indices = batch_idx * self.num_seeds + seed_idx
            self.metrics.copy_only_index(other.metrics, flat_indices)

    def copy_at_batch_indices(self, other: "BaseSolverResult", mask: torch.Tensor) -> None:
        """Overwrite entire batch slices from *other* where *mask* is True.

        Unlike :meth:`copy_successful_solutions` (which copies individual
        ``(batch, seed)`` entries), this copies **all seeds** for the
        selected batch indices.  Useful for first-success-wins merging in
        batch planners.

        Args:
            mask: Boolean tensor of shape ``(batch_size,)`` selecting which
                batch indices to overwrite.
        """
        for attr in ("success", "solution", "position_error", "rotation_error",
                      "cspace_error", "goalset_index", "feasible", "optimized_seeds",
                      "total_cost_reshaped", "seed_rank", "seed_cost"):
            dst = getattr(self, attr, None)
            src = getattr(other, attr, None)
            if dst is not None and src is not None and dst.shape == src.shape:
                dst[mask] = src[mask]

        if self.js_solution is not None and other.js_solution is not None:
            for attr in ("position", "velocity", "acceleration", "jerk", "dt"):
                dst = getattr(self.js_solution, attr, None)
                src = getattr(other.js_solution, attr, None)
                if dst is not None and src is not None and dst.shape == src.shape:
                    dst[mask] = src[mask]

