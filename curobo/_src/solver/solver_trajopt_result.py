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
import curobo._src.runtime as curobo_runtime
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.runtime import debug as debug_mode
from curobo._src.solver.solver_base_result import BaseSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import (
    copy_joint_state_at_batch_seed_indices,
    gather_joint_state_by_seed,
    trim_joint_state_trajectory,
)
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


@dataclass
class TrajOptSolverResult(BaseSolverResult):
    """Result specific to the Trajectory Optimization solver."""

    # Inherits: success, solution (raw opt output), js_solution (JointState traj),
    # position_error, rotation_error, goalset_index, solve_time, debug_info, optimized_seeds
    # Add any TrajOpt-specific result fields here if needed.

    #: Optimized actions returned as a tensor of shape (batch, return_seeds, horizon, dof)
    solution: Optional[torch.Tensor] = None

    #: Optimized actions are rolled out to obtain JointState
    js_solution: Optional[JointState] = None

    #: Optimized actions are interpolated to a user provided dt
    interpolated_trajectory: Optional[JointState] = None

    #: Last tstep of interpolated trajectory
    interpolated_last_tstep: Optional[torch.Tensor] = None

    interpolated_metrics: Optional[RolloutMetrics] = None

    maximum_trajectory_dt: Optional[torch.Tensor] = None
    minimum_trajectory_dt: Optional[torch.Tensor] = None

    def copy_at_batch_indices(self, other: "TrajOptSolverResult", mask: torch.Tensor) -> None:
        """Extend base method to also copy interpolated trajectory fields."""
        super().copy_at_batch_indices(other, mask)
        if self.interpolated_trajectory is not None and other.interpolated_trajectory is not None:
            for attr in ("position", "velocity", "acceleration", "jerk", "dt"):
                dst = getattr(self.interpolated_trajectory, attr, None)
                src = getattr(other.interpolated_trajectory, attr, None)
                if dst is not None and src is not None and dst.shape == src.shape:
                    dst[mask] = src[mask]
        if self.interpolated_last_tstep is not None and other.interpolated_last_tstep is not None:
            self.interpolated_last_tstep[mask] = other.interpolated_last_tstep[mask]

    def motion_time(self) -> torch.Tensor:
        if self.js_solution is None:
            log_and_raise("js_solution is not set")
        horizon = self.js_solution.shape[-2]
        dt = self.js_solution.dt
        motion_time = (horizon - 1) * dt
        return motion_time

    @profiler.record_function("trajopt_solver_result/clone")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def clone(self) -> TrajOptSolverResult:
        if curobo_runtime.debug:
            if self.seed_rank is not None:
                og_data_ptr = self.seed_rank.data_ptr()
        base_clone = super().clone()
        if self.seed_rank is not None:
            if curobo_runtime.debug:
                new_data_ptr = base_clone.seed_rank.data_ptr()
                if og_data_ptr == new_data_ptr:
                    log_and_raise(f"seed_rank is not cloned: {og_data_ptr} == {new_data_ptr}")
                if torch.max(self.seed_rank) >= self.num_seeds:
                    log_and_raise(
                        f"self.seed_rank: {self.seed_rank} >= self.num_seeds: {self.num_seeds}"
                    )
                if torch.max(base_clone.seed_rank) >= self.num_seeds:
                    log_and_raise(
                        f"base_clone.seed_rank: {base_clone.seed_rank} >= self.num_seeds: {self.num_seeds}"
                    )
        return TrajOptSolverResult(
            success=base_clone.success,
            solution=base_clone.solution,
            js_solution=base_clone.js_solution,
            position_error=base_clone.position_error,
            rotation_error=base_clone.rotation_error,
            goalset_index=base_clone.goalset_index,
            solve_time=base_clone.solve_time,
            total_time=base_clone.total_time,
            interpolated_trajectory=(
                self.interpolated_trajectory.clone()
                if self.interpolated_trajectory is not None
                else None
            ),
            interpolated_last_tstep=(
                self.interpolated_last_tstep.clone()
                if self.interpolated_last_tstep is not None
                else None
            ),
            debug_info=base_clone.debug_info,
            optimized_seeds=base_clone.optimized_seeds,
            metrics=base_clone.metrics,
            interpolated_metrics=(
                self.interpolated_metrics.clone() if self.interpolated_metrics is not None else None
            ),
            position_tolerance=base_clone.position_tolerance,
            orientation_tolerance=base_clone.orientation_tolerance,
            seed_rank=base_clone.seed_rank,
            seed_cost=base_clone.seed_cost,
            batch_size=base_clone.batch_size,
            num_seeds=base_clone.num_seeds,
            minimum_trajectory_dt=self.minimum_trajectory_dt,
            maximum_trajectory_dt=self.maximum_trajectory_dt,
            total_cost_reshaped=base_clone.total_cost_reshaped,
            feasible=base_clone.feasible,
            cspace_error=base_clone.cspace_error,
            solution_state=base_clone.solution_state,
        )

    @profiler.record_function("trajopt_solver_result/get_interpolated_plan")
    def get_interpolated_plan(self) -> JointState:
        if self.interpolated_last_tstep is None:
            return self.interpolated_trajectory
        if len(self.interpolated_last_tstep) > 1:
            log_and_raise("only single result is supported")
        return trim_joint_state_trajectory(
            self.interpolated_trajectory, 0, self.interpolated_last_tstep[0]
        )

    @profiler.record_function("trajopt_solver_result/process_metrics_and_rank_seeds")
    def process_metrics_and_rank_seeds(self):
        self._process_metrics()
        self._compute_rank()

    @profiler.record_function("trajopt_solver_result/_process_metrics")
    def _process_metrics(self):
        # get batch size and num seeds:
        batch_size = self.batch_size
        num_seeds = self.num_seeds

        # 1. Check Feasibility (over entire horizon)
        # feasible shape: (batch * num_seeds,)
        feasible = self.metrics.costs_and_constraints.get_feasible(
            include_all_hybrid=False,
            sum_horizon=True,  # Checks constraints over horizon
        )
        # 2. Check Convergence (Position and Orientation at the *last* timestep)
        converge_list = []
        cost_list = []
        goalset_index_list = []
        for k in range(len(self.metrics.convergence.names)):
            metric_name = self.metrics.convergence.names[k]
            # Values shape: (batch*num_seeds, horizon, num_links) or (batch*num_seeds, horizon)
            metric_values = self.metrics.convergence.values[k]

            # Check convergence only at the last timestep (index -1)
            last_step_values = metric_values[
                :, -1:
            ]  # Keep dim: (batch*num_seeds, 1, num_links) or (batch*num_seeds, 1)

            if "position_tolerance" in metric_name:
                cost_list.append(last_step_values)
                converged = last_step_values < self.position_tolerance
                converge_list.append(converged)
            elif "orientation_tolerance" in metric_name:
                cost_list.append(last_step_values)
                converged = last_step_values < self.orientation_tolerance
                converge_list.append(converged)
            elif "goalset_index" in metric_name:
                goalset_index_list.append(last_step_values)  # Keep last step index

        # Combine convergence criteria
        if converge_list:
            converged_all_links = torch.cat(
                converge_list, dim=-1
            )  # Shape: (batch*num_seeds, 1, num_links * 2)
            converged = torch.all(converged_all_links, dim=-1).squeeze(
                -1
            )  # Shape: (batch*num_seeds,)
        else:
            # Handle case with no convergence metrics (e.g. pure smoothing)
            # Assume converged if feasible? Or always False? Let's say always True if feasible.
            converged = torch.ones_like(feasible)

        # Success = Feasible AND Converged

        success = torch.logical_and(converged, feasible)

        # check dt success:
        # dt_success =  self.js_solution.dt < self.maximum_trajectory_dt
        # print(dt_success)
        # print(self.js_solution.dt)
        # success = torch.logical_and(dt_success, success)
        if self.interpolated_metrics is not None:
            interpolated_feasible = self.interpolated_metrics.costs_and_constraints.get_feasible(
                include_all_hybrid=False, sum_horizon=True
            )
            success = torch.logical_and(interpolated_feasible, success)

        goalset_index = torch.cat(goalset_index_list, dim=-1) if goalset_index_list else None

        # compute position and rotation errors
        if cost_list:
            cost = torch.cat(cost_list, dim=-1)  # Shape: (batch*num_seeds, 1, num_links * 2)

        else:
            cost = torch.zeros(
                (self.batch_size, self.num_seeds, 0), device=success.device, dtype=torch.float32
            )
        cost_sum = cost.sum(dim=-1).squeeze(-1)

        pose_error_flat = cost.view(
            batch_size * num_seeds, cost.shape[-1]
        )  # Shape: (batch*num_seeds, num_links * 2)
        num_links = int(pose_error_flat.shape[-1] / 2) if cost.numel() > 0 else 0
        if num_links > 0:
            pose_error_link_view = pose_error_flat.view(batch_size, num_seeds, num_links, 2)
            # position_error shape: (batch, return_seeds)
            position_error = torch.max(pose_error_link_view[..., 0], dim=-1)[0]
            # rotation_error shape: (batch, return_seeds)
            rotation_error = torch.max(pose_error_link_view[..., 1], dim=-1)[0]
        else:
            # Handle case with no pose cost metrics
            dummy_error = torch.zeros_like(success, dtype=torch.float32)
            position_error = dummy_error
            rotation_error = dummy_error

        # Goalset indices if applicable
        if goalset_index is not None:
            goalset_index = goalset_index.view(
                batch_size, num_seeds, goalset_index.shape[-1]
            )  # Shape: (batch*num_seeds, num_links)

        self.success = success.view(self.batch_size, self.num_seeds)
        self.position_error = position_error.view(self.batch_size, self.num_seeds)
        self.rotation_error = rotation_error.view(self.batch_size, self.num_seeds)
        self.goalset_index = goalset_index
        self.seed_cost = cost_sum.view(self.batch_size, self.num_seeds)

    @profiler.record_function("trajopt_solver_result/_compute_rank")
    def _compute_rank(self):
        if self.metrics is None:
            log_and_raise("metrics is not set")

        dt = self.metrics.state.joint_state.dt
        jerk = self.metrics.state.joint_state.jerk
        acc = self.metrics.state.joint_state.acceleration
        seed_cost = self.seed_cost

        success = self.success
        batch_size = self.batch_size
        num_seeds = self.num_seeds

        seed_cost, seed_rank = self._jit_compute_rank(
            dt, acc, jerk, seed_cost, success, batch_size, num_seeds
        )

        # 4. Select Top-K Solutions
        # order seeds based on cost:
        if debug_mode:
            if seed_cost.shape[1] < self.num_seeds:
                log_and_raise(f"seed_cost.shape: {seed_cost.shape}, self.num_seeds: {self.num_seeds}")

            if torch.any(torch.isnan(seed_cost)) or torch.any(torch.isinf(seed_cost)):
                log_and_raise(f"seed_cost has nan or inf: {seed_cost}")

        self.seed_rank = seed_rank
        self.seed_cost = seed_cost
        self.total_cost_reshaped = seed_cost

    @staticmethod
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def _jit_compute_rank(
        trajectory_dt,
        acceleration,
        jerk,
        seed_rollout_cost,
        success,
        batch_size: int,
        num_seeds: int,
    ):
        dt_cost = trajectory_dt.view(batch_size, num_seeds)
        abs_jerk = torch.abs(jerk)
        abs_acc = torch.abs(acceleration)
        horizon = jerk.shape[-2]
        st_horizon = 8
        end_horizon = horizon - 8
        mean_jerk = torch.mean(torch.mean(abs_jerk, dim=-1)[..., st_horizon:end_horizon], dim=-1)
        mean_acc = torch.mean(torch.mean(abs_acc, dim=-1)[..., st_horizon:end_horizon], dim=-1)
        mean_jerk = mean_jerk.view(batch_size, num_seeds)
        mean_acc = mean_acc.view(batch_size, num_seeds)
        smooth_cost = mean_jerk * 0.001 + mean_acc * 0.01 + dt_cost * 1000.0

        # Note: Should probably use total time (dt * horizon) ? Or just dt? Original used dt.
        total_cost = seed_rollout_cost + smooth_cost  # Combine pose error and dt cost
        total_cost = total_cost.view(batch_size, num_seeds)

        total_cost[~success] += 1e16  # Penalize non-successful trajectories

        # Reshape for topk selection per batch item
        total_cost_reshaped = total_cost.view(batch_size, num_seeds)  # Shape: (batch, num_seeds)

        _, seed_rank = torch.topk(total_cost_reshaped, k=num_seeds, largest=False)

        return total_cost_reshaped, seed_rank

    @profiler.record_function("trajopt_solver_result/get_topk_seeds")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def get_topk_seeds(self, topk: int) -> TrajOptSolverResult:
        """Get the topk best seeds per problem from the result.

        Args:
            topk: number of best seeds to get per problem

        Returns:
            TrajOptSolverResult with the topk seeds
        """
        if self.total_cost_reshaped is None:
            log_and_raise("total_cost_reshaped is not set")
        if self.success is None:
            log_and_raise("success is not set")
        if self.seed_rank is None:
            log_and_raise("seed_rank is not set")

        if topk > self.num_seeds:
            log_and_raise(f"topk is greater than num_seeds: {topk} > {self.num_seeds}")
        if topk == self.num_seeds:
            return self

        # take first topk seeds
        if debug_mode:
            if torch.max(self.seed_rank) >= self.num_seeds:
                print("self.seed_cost: ", self.seed_cost)
                print("self.seed_rank: ", self.seed_rank)
                print("self.num_seeds: ", self.num_seeds)
                log_and_raise("seed_rank is out of bounds")
            if torch.min(self.seed_rank) < 0:
                log_and_raise("seed_rank is less than 0: " + str(self.seed_rank))

        topk_seeds = self.seed_rank[:, :topk]
        batch_size = self.batch_size

        topk_seeds = topk_seeds.view(batch_size, topk)

        new_result = self.clone()

        new_result.success = torch.gather(self.success, dim=1, index=topk_seeds)
        new_result.position_error = torch.gather(self.position_error, dim=1, index=topk_seeds)
        new_result.rotation_error = torch.gather(self.rotation_error, dim=1, index=topk_seeds)
        new_result.seed_cost = torch.gather(self.seed_cost, dim=1, index=topk_seeds)
        new_result.seed_rank = torch.gather(self.seed_rank, dim=1, index=topk_seeds)

        num_links = self.goalset_index.shape[-1]
        # make sure topk_seeds has shape (batch_size, topk, num_links) by broadcasting:
        if num_links == 1:
            topk_seeds_goalset_index = topk_seeds.view(batch_size, topk, 1)
        else:
            topk_seeds_goalset_index = topk_seeds.unsqueeze(-1).expand(batch_size, topk, num_links)

        new_result.goalset_index = torch.gather(
            self.goalset_index, dim=1, index=topk_seeds_goalset_index
        )

        # Broadcast topk_seeds has shape (batch_size, topk, horizon, dof) by broadcasting:
        horizon = self.solution.shape[2]
        dof = self.solution.shape[3]

        # Reshape solution to [batch_size*num_seeds, horizon, dof]
        solution_flat = self.solution.view(-1, horizon, dof)

        # Create flat indices [batch_size, topk] -> [batch_size*topk]
        offset = torch.arange(batch_size, device=topk_seeds.device) * self.num_seeds
        offset = offset.view(-1, 1)
        flat_indices = (topk_seeds + offset).view(-1)

        # Select using flat indices
        selected_flat = solution_flat[flat_indices]

        # Reshape back to [batch_size, topk, horizon, dof]
        new_result.solution = selected_flat.view(batch_size, topk, horizon, dof)
        optimized_seeds_flat = self.optimized_seeds.view(batch_size * self.num_seeds, horizon, dof)
        new_result.optimized_seeds = optimized_seeds_flat[flat_indices].view(
            batch_size, topk, horizon, dof
        )

        new_result.batch_size = batch_size
        new_result.num_seeds = topk

        if self.interpolated_trajectory is not None:
            new_result.interpolated_trajectory = gather_joint_state_by_seed(
                self.interpolated_trajectory, topk_seeds
            )
        if self.js_solution is not None:
            new_result.js_solution = gather_joint_state_by_seed(self.js_solution, topk_seeds)

        new_result.interpolated_metrics = None
        new_result.metrics = None
        if self.interpolated_last_tstep is not None:
            new_result.interpolated_last_tstep = torch.gather(
                self.interpolated_last_tstep.view(batch_size, self.num_seeds),
                dim=1,
                index=topk_seeds,
            )

        return new_result

    @profiler.record_function("trajopt_solver_result/copy_successful_solutions")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def copy_successful_solutions(self, other: TrajOptSolverResult):
        """Copy successful solutions from other result to self.

        The result is assumed to have shape: (batch_size, num_seeds)
        """
        if self.success is None:
            log_and_raise("success is not set")
        if other.success is None:
            log_and_raise("other.success is not set")

        # get batch indices and seed indices of successful solutions:
        batch_idx, seed_idx = other.success.nonzero(as_tuple=True)  # shape: (num_success, 2)
        if curobo_runtime.debug:
            if torch.max(seed_idx) >= self.num_seeds:
                log_and_raise("seed_idx is out of bounds: " + str(seed_idx))
            if torch.min(seed_idx) < 0:
                log_and_raise("seed_idx is less than 0: " + str(seed_idx))
            if torch.max(batch_idx) >= self.batch_size:
                log_and_raise("batch_idx is out of bounds: " + str(batch_idx))
            if torch.min(batch_idx) < 0:
                log_and_raise("batch_idx is less than 0: " + str(batch_idx))
        # if self.solution is not None:
        #    self.solution[batch_idx, seed_idx, :, :] = other.solution[batch_idx, seed_idx, :, :]
        # if self.js_solution is not None:
        #    self.js_solution.copy_at_batch_seed_indices(other.js_solution, batch_idx, seed_idx)
        if self.interpolated_trajectory is not None and other.interpolated_trajectory is not None:
            copy_joint_state_at_batch_seed_indices(
                self.interpolated_trajectory, other.interpolated_trajectory, batch_idx, seed_idx
            )
        if self.interpolated_last_tstep is not None and other.interpolated_last_tstep is not None:
            self.interpolated_last_tstep[batch_idx, seed_idx] = other.interpolated_last_tstep[
                batch_idx, seed_idx
            ]

        if self.interpolated_metrics is not None and other.interpolated_metrics is not None:
            flat_indices = batch_idx * self.num_seeds + seed_idx
            self.interpolated_metrics.copy_only_index(other.interpolated_metrics, flat_indices)

        super().copy_successful_solutions(other)

