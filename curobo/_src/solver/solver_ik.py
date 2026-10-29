# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inverse Kinematics solver for reaching target tool-frame poses.

Wraps SolverCore with IK-specific logic: seed generation via a Levenberg-Marquardt
seed solver, multi-link goal handling, batch padding, and solution ranking by pose
error. Supports goalset IK, velocity-aware IK via optimization_dt, and optional
collision checking.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Dict, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

from curobo._src.cost.criteria_tool_relative_transform import ToolRelativeTransformCriteria
from curobo._src.cost.criteria_tool_velocity_limit import ToolVelocityLimitCriteria
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

# CuRobo
from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.kinematics.kinematics import KinematicsState
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.solver.processor_ik_result import (
    IKResultProcessorConfig,
    WarpIKResultProcessor,
)
from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.solver.solver_core import SolverCore
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_ik_result import IKSolverResult
from curobo._src.solver.solver_trajopt_result import _parse_pose_convergence
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.tensor_util import stable_topk


@dataclass(frozen=True)
class _IKConvergenceLayout:
    """Indices into a stable IK convergence metric collection."""

    position_indices: tuple[int, ...]
    orientation_indices: tuple[int, ...]
    goalset_indices: tuple[int, ...]
    cspace_indices: tuple[int, ...]


def _pad_batch_inputs(
    goal_tool_poses: GoalToolPose,
    current_state: Optional[JointState],
    seed_config: Optional[torch.Tensor],
    batch_size: int,
    max_batch: int,
):
    """Pad inputs from batch_size to max_batch by repeating the first element."""
    pad = max_batch - batch_size
    goal_tool_poses = goal_tool_poses.clone()
    pos = goal_tool_poses.position
    quat = goal_tool_poses.quaternion
    expand_dims = [-1] * (pos.ndim - 1)
    goal_tool_poses.position = torch.cat(
        [pos, pos[:1].expand(pad, *expand_dims)],
        dim=0,
    )
    goal_tool_poses.quaternion = torch.cat(
        [quat, quat[:1].expand(pad, *[-1] * (quat.ndim - 1))],
        dim=0,
    )
    if current_state is not None:
        current_state = current_state.clone()
        current_state.position = torch.cat(
            [current_state.position, current_state.position[:1].expand(pad, -1)],
            dim=0,
        )
        if current_state.velocity is not None:
            current_state.velocity = torch.cat(
                [current_state.velocity, current_state.velocity[:1].expand(pad, -1)],
                dim=0,
            )
        if current_state.acceleration is not None:
            current_state.acceleration = torch.cat(
                [current_state.acceleration, current_state.acceleration[:1].expand(pad, -1)],
                dim=0,
            )
    if seed_config is not None:
        seed_config = torch.cat(
            [seed_config, seed_config[:1].expand(pad, *[-1] * (seed_config.ndim - 1))],
            dim=0,
        )
    return goal_tool_poses, current_state, seed_config


def _pad_tool_relative_transform_target(
    target: Optional[ToolRelativeTransformTarget],
    batch_size: int,
    max_batch: int,
) -> Optional[ToolRelativeTransformTarget]:
    """Pad a per-problem relative target while preserving a shared target.

    A single target group is shared by every problem. Otherwise there must be
    exactly one group per unpadded problem, and group zero is repeated for the
    fixed-size padding lanes.
    """
    if target is None:
        return None
    if target.num_target_groups == 1:
        return target
    if target.num_target_groups != batch_size:
        log_and_raise(
            "tool_relative_transform_target must have one shared target group "
            "or one group per problem; got "
            f"{target.num_target_groups} groups for batch_size={batch_size}"
        )
    if batch_size == max_batch:
        return target

    pad = max_batch - batch_size
    return ToolRelativeTransformTarget(
        reference_tool_frame=target.reference_tool_frame,
        tool_frames=target.tool_frames.copy(),
        position=torch.cat([target.position, target.position[:1].expand(pad, -1, -1)], dim=0),
        quaternion=torch.cat(
            [target.quaternion, target.quaternion[:1].expand(pad, -1, -1)], dim=0
        ),
    )


def _slice_batch_result(result, batch_size: int):
    """Slice padded result tensors back to the original batch size."""
    result.success = result.success[:batch_size]
    for attr in (
        "solution",
        "position_error",
        "rotation_error",
        "goalset_index",
        "feasible",
        "optimized_seeds",
        "total_cost_reshaped",
        "seed_rank",
        "seed_cost",
    ):
        val = getattr(result, attr, None)
        if val is not None:
            setattr(result, attr, val[:batch_size])
    if result.js_solution is not None:
        result.js_solution = result.js_solution[:batch_size]
    if result.solution_state is not None:
        result.solution_state = result.solution_state[:batch_size]
    return result


class IKSolver:
    """Inverse Kinematics solver."""

    def __init__(
        self, config: IKSolverCfg, scene_collision_checker: Optional[SceneCollision] = None
    ):
        """Initializes the IK solver."""
        self.config = config
        self.core = SolverCore(config.core_cfg, scene_collision_checker)

        self.seed_ik_solver = None
        self._result_convergence_names: tuple[str, ...] = ()
        self._result_convergence_layout: Optional[_IKConvergenceLayout] = None
        self._result_batch_offsets: Optional[torch.Tensor] = None
        self._result_batch_offset_shape = (-1, -1)

        robot_model = self.auxiliary_rollout.transition_model.robot_model
        locked_joint_state = robot_model.lock_jointstate
        self._result_processor = WarpIKResultProcessor(
            IKResultProcessorConfig(
                device_cfg=self.device_cfg,
                action_dim=self.action_dim,
                joint_names=self.joint_names,
                full_joint_names=robot_model.all_articulated_joint_names,
                locked_joint_names=(
                    [] if locked_joint_state is None else locked_joint_state.joint_names
                ),
                locked_joint_position=(
                    None if locked_joint_state is None else locked_joint_state.position
                ),
                position_tolerance=self.config.position_tolerance,
                orientation_tolerance=self.config.orientation_tolerance,
                position_rank_weight=self.config.position_rank_weight,
                orientation_rank_weight=self.config.orientation_rank_weight,
                cspace_rank_weight=self.config.cspace_rank_weight,
                success_requires_convergence=self.config.success_requires_convergence,
                use_cuda_graph=self.config.use_cuda_graph,
            )
        )

        # create an instance of seed ik solver:
        seed_solver_num_seeds = config.seed_solver_num_seeds
        max_iterations = 16
        tile_threads = 64

        if self.config.override_iters_for_multi_link_ik is not None:
            for k in self.optimizer.solver_names:
                if "lbfgs" in k:
                    index = self.optimizer.solver_names.index(k)
                    if (
                        self.config.optimizer_configs[index].num_iters
                        < self.config.override_iters_for_multi_link_ik
                    ):
                        self.optimizer.update_solver_params(
                            {k: {"num_iters": self.config.override_iters_for_multi_link_ik}}
                        )
                        log_warn(
                            f"IKSolver: increasing number of optimization iterations "
                            f"to {self.config.optimizer_configs[index].num_iters} for multi-link IK."
                        )

        # if number of goal links is more than 1, we increase number of optimization iterations
        if len(self.kinematics.tool_frames) > 1:
            seed_solver_num_seeds = 128
            max_iterations = 20
        if len(self.kinematics.tool_frames) > 2:
            seed_solver_num_seeds = 64
            max_iterations = 30
            tile_threads = 256

        if self.config.num_seeds > seed_solver_num_seeds:
            seed_solver_num_seeds = self.config.num_seeds * 2

        if self.config.use_lm_seed:
            self.seed_ik_solver = SeedIKSolver(
                SeedIKSolverCfg.create(
                    robot=self.config.robot_config,
                    num_seeds=seed_solver_num_seeds,
                    max_iterations=max_iterations,
                    inner_iterations=max_iterations,
                    joint_limit_weight=1.0,
                    lambda_initial=1.0,
                    lambda_factor=2.0,
                    lambda_max=1.0e10,
                    lambda_min=1e-5,
                    rho_min=1e-5,
                    use_backward=True,
                    use_cuda_graph=self.config.use_cuda_graph,
                    tile_threads=tile_threads,
                    device_cfg=self.config.device_cfg,
                    position_weight=self.config.seed_position_weight,
                    orientation_weight=self.config.seed_orientation_weight,
                    velocity_weight=self.config.seed_velocity_weight,
                    acceleration_weight=self.config.seed_acceleration_weight,
                )
            )

    # -------------------------------------------------------------------
    # Delegated properties and methods from SolverCore
    # -------------------------------------------------------------------

    @property
    def optimizer(self):
        return self.core.optimizer

    @property
    def metrics_rollout(self):
        return self.core.metrics_rollout

    @property
    def auxiliary_rollout(self):
        return self.core.auxiliary_rollout

    @property
    def kinematics(self):
        return self.core.kinematics

    @property
    def transition_model(self):
        return self.core.transition_model

    @property
    def action_dim(self) -> int:
        return self.core.action_dim

    @property
    def action_horizon(self) -> int:
        return self.core.action_horizon

    @property
    def joint_names(self):
        return self.core.joint_names

    @property
    def tool_frames(self):
        return self.core.tool_frames

    @property
    def default_joint_position(self):
        return self.core.default_joint_position

    @property
    def default_joint_state(self):
        return self.core.default_joint_state

    @property
    def device_cfg(self):
        return self.core.device_cfg

    @property
    def scene_collision_checker(self):
        return self.core.scene_collision_checker

    @property
    def goal_registry_manager(self):
        return self.core.goal_registry_manager

    @property
    def solve_state(self) -> SolveState:
        return self.core.solve_state

    @property
    def seed_manager(self):
        return self.core.seed_manager

    def get_all_rollout_instances(self, **kwargs):
        return self.core.get_all_rollout_instances(**kwargs)

    def compute_kinematics(self, state: JointState) -> KinematicsState:
        """Run forward kinematics and return tool poses and collision spheres."""
        return self.core.compute_kinematics(state)

    def get_active_js(self, full_js: JointState) -> JointState:
        return self.core.get_active_js(full_js)

    def get_full_js(self, active_js: JointState) -> JointState:
        return self.core.get_full_js(active_js)

    def sample_configs(self, num_samples: int, rejection_ratio: int = 10) -> torch.Tensor:
        return self.core.sample_configs(
            num_samples, rejection_ratio, self.config.optimizer_collision_activation_distance
        )

    def update_link_inertial(self, link_name, mass=None, com=None, inertia=None):
        return self.core.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(self, link_properties):
        return self.core.update_links_inertial(link_properties)

    def debug_dump(self, file_path: str):
        return self.core.debug_dump(file_path)

    def prepare_action_seeds(
        self, batch_size, num_seeds, seed_config=None, current_state=None, seed_traj=None
    ):
        return self.core.prepare_action_seeds(
            batch_size, num_seeds, seed_config, current_state, seed_traj
        )

    def prepare_trajectory_seeds(
        self, batch_size, num_seeds, current_state, seed_config=None, seed_traj=None
    ):
        return self.core.prepare_trajectory_seeds(
            batch_size, num_seeds, current_state, seed_config, seed_traj
        )

    def enable_tool_pose_tracking(self, tool_frames=None):
        """Enable tool-pose cost terms for the specified (or all) tool frames."""
        self.core.enable_tool_pose_tracking(
            tool_frames, self.config.non_terminal_tool_pose_weight_factor
        )

    def disable_tool_pose_tracking(self, tool_frames=None):
        self.core.disable_tool_pose_tracking(tool_frames)

    def enable_joint_position_tracking(self):
        self.core.enable_joint_position_tracking()

    def disable_joint_position_tracking(self):
        self.core.disable_joint_position_tracking()

    def reset_shape(self):
        self._result_processor.reset_shape()
        self.core.reset_shape()

    def reset_cuda_graph(self):
        self.core.reset_cuda_graph()
        self._result_processor.reset_cuda_graph()

    # -------------------------------------------------------------------
    # IK-specific: tool pose criteria also updates seed solver
    # -------------------------------------------------------------------

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        if self.seed_ik_solver is not None:
            self.seed_ik_solver.update_tool_pose_criteria(tool_pose_criteria)
        self.core.update_tool_pose_criteria(tool_pose_criteria)

    def update_tool_relative_transform_criteria(
        self, criteria: Dict[str, ToolRelativeTransformCriteria]
    ) -> None:
        self.core.update_tool_relative_transform_criteria(criteria)

    def update_tool_velocity_limit_criteria(
        self, criteria: Dict[str, ToolVelocityLimitCriteria]
    ) -> None:
        self.core.update_tool_velocity_limit_criteria(criteria)

    def reset_seed(self):
        if self.config.use_lm_seed:
            self.seed_ik_solver.reset_seed()
        self.core.reset_seed()

    def destroy(self):
        self._result_processor.reset_cuda_graph()
        if self.seed_ik_solver is not None:
            self.seed_ik_solver.destroy()
        self.core.destroy()

    # -------------------------------------------------------------------
    # Goal buffer
    # -------------------------------------------------------------------

    @profiler.record_function("ik_solver/prepare_goal_buffer")
    def _prepare_goal_buffer(
        self,
        solve_state: SolveState,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        use_implicit_goal: bool = False,
        seed_goal_state: Optional[JointState] = None,
        goal_state: Optional[JointState] = None,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
    ):
        goal_tool_poses = goal_tool_poses.reorder_links(self.kinematics.tool_frames)
        goal_buffer, update_reference = self.core.prepare_goal_buffer(
            solve_state=solve_state,
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
            use_implicit_goal=use_implicit_goal,
            seed_goal_state=seed_goal_state,
            goal_state=goal_state,
            tool_relative_transform_target=tool_relative_transform_target,
        )
        return goal_buffer, update_reference

    def _update_rollout_params(
        self,
        goal_buffer: GoalRegistry,
        include_auxiliary_rollout: bool = True,
        include_optimizer_rollouts: bool = True,
    ) -> None:
        self.core.update_rollout_params(
            goal_buffer,
            include_auxiliary_rollout=include_auxiliary_rollout,
            include_optimizer_rollouts=include_optimizer_rollouts,
        )

    @property
    def problem_batch_size(self) -> int:
        return self.core._solve_state.get_ik_batch_size()

    # -------------------------------------------------------------------
    # IK-specific methods
    # -------------------------------------------------------------------

    def _fill_current_state_dt(self, current_state: Optional[JointState]) -> Optional[JointState]:
        """Fill ``current_state.dt`` from ``config.optimization_dt`` when not already set."""
        if (
            current_state is not None
            and current_state.dt is None
            and self.config.optimization_dt is not None
        ):
            current_state = current_state.clone()
            current_state.dt = torch.full(
                (current_state.position.shape[0],),
                self.config.optimization_dt,
                device=current_state.device,
                dtype=current_state.dtype,
            )
        return current_state

    @profiler.record_function("ik_solver/solve_impl")
    def _solve_impl(
        self,
        solve_state: SolveState,
        goal_tool_poses: GoalToolPose,
        num_seeds: int,
        current_state: Optional[JointState] = None,
        seed_config=None,
        return_seeds: int = 1,
        run_optimizer: bool = True,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
        use_specialized_result_processor: bool = True,
    ) -> IKSolverResult:
        goal_tool_poses = goal_tool_poses.reorder_links(self.kinematics.tool_frames)
        total_timer = CudaEventTimer().start()
        if goal_tool_poses.ndim != 5:
            log_and_raise(
                f"goal_tool_poses has shape {goal_tool_poses.shape}. "
                f"Expected 5D [batch, horizon, num_links, num_goalset, 3/4]."
            )

        current_state = self._fill_current_state_dt(current_state)

        goal_buffer, update_reference = self._prepare_goal_buffer(
            solve_state=solve_state,
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
            tool_relative_transform_target=tool_relative_transform_target,
        )

        coord_position_seed = self.prepare_action_seeds(
            solve_state.batch_size, num_seeds, seed_config=seed_config
        )

        self._update_rollout_params(
            goal_buffer,
            include_auxiliary_rollout=False,
            include_optimizer_rollouts=False,
        )

        if self.config.exit_early or not run_optimizer:
            metrics_result = self.metrics_rollout.compute_metrics_from_action(coord_position_seed)
            ik_result = self._get_result(
                coord_position_seed,
                metrics_result,
                num_seeds,
                return_seeds,
                use_specialized_result_processor=use_specialized_result_processor,
            )
            success_ratio = torch.count_nonzero(ik_result.success) / ik_result.success.shape[0]
            if (
                success_ratio >= self.config.exit_early_batch_success_threshold
                or not run_optimizer
            ):
                self._compute_solution_velocity(ik_result, current_state, return_seeds)
                ik_result.total_time = total_timer.stop()
                return ik_result

        self.core.update_rollout_params(
            goal_buffer,
            include_auxiliary_rollout=False,
            include_optimizer_rollouts=True,
            include_metrics_rollout=False,
        )
        opt_timer = CudaEventTimer().start()
        self.optimizer.reinitialize(coord_position_seed)
        opt_result = self.optimizer.optimize(coord_position_seed)
        opt_time = opt_timer.stop()

        metrics_result = self.metrics_rollout.compute_metrics_from_action(opt_result)

        ik_result = self._get_result(
            opt_result,
            metrics_result,
            num_seeds,
            return_seeds,
            use_specialized_result_processor=use_specialized_result_processor,
        )
        self._compute_solution_velocity(ik_result, current_state, return_seeds)
        ik_result.total_time = total_timer.stop()
        ik_result.solve_time = opt_time
        return ik_result

    def get_unique_solution(self, roundoff_decimals: int = 2) -> torch.Tensor:
        """Get unique solutions from many feasible solutions for the same problem.

        Filters :attr:`solution` by :attr:`success`, rounds to
        *roundoff_decimals* to merge near-duplicate configurations, and
        returns one representative per unique rounded configuration.

        Args:
            roundoff_decimals: Number of decimal places used when rounding
                joint values to determine uniqueness. A smaller value merges
                more solutions together.

        Returns:
            Tensor of shape ``[num_unique, dof]`` containing the unique
            feasible joint configurations.
        """
        in_solution = self.solution[self.success]
        r_sol = torch.round(in_solution, decimals=roundoff_decimals)
        if not (len(in_solution.shape) == 2):
            log_and_raise("Solution shape is not of length 2")
        s, i = torch.unique(r_sol, dim=-2, return_inverse=True)
        sol = in_solution[i[: s.shape[0]]]
        return sol

    @profiler.record_function("ik_solver/_get_result")
    def _get_result(
        self,
        opt_result: torch.Tensor,
        metrics_result: RolloutMetrics,
        num_seeds: int,
        return_seeds: int,
        use_specialized_result_processor: bool = True,
    ) -> IKSolverResult:
        """Select and assemble IK results with the shape-managed processor."""
        result_processor = getattr(self, "_result_processor", None)
        if use_specialized_result_processor and result_processor is not None:
            result = result_processor.process(
                opt_result,
                metrics_result,
                num_seeds,
                return_seeds,
            )
            if result is not None:
                return result
        return self._get_result_torch(opt_result, metrics_result, num_seeds, return_seeds)

    @profiler.record_function("ik_solver/_get_result_torch")
    def _get_result_torch(
        self,
        opt_result: torch.Tensor,
        metrics_result: RolloutMetrics,
        num_seeds: int,
        return_seeds: int,
    ) -> IKSolverResult:
        problem_count = opt_result.shape[0]
        batch_size = problem_count // num_seeds
        pose_summary = _parse_pose_convergence(
            metrics_result.convergence,
            self.config.position_tolerance,
            self.config.orientation_tolerance,
        )
        candidate_count = opt_result.shape[0]
        converged = (
            pose_summary.converged
            if pose_summary.converged is not None
            else torch.ones(candidate_count, dtype=torch.bool, device=opt_result.device)
        )
        feasible_tensor = self._get_result_feasibility(metrics_result, problem_count)
        if self.config.success_requires_convergence:
            success = torch.logical_and(converged, feasible_tensor)
        else:
            success = feasible_tensor

        cost_sum = torch.zeros(
            candidate_count,
            dtype=opt_result.dtype,
            device=opt_result.device,
        )
        for metric_name, metric_values in zip(
            metrics_result.convergence.names,
            metrics_result.convergence.values,
        ):
            if metric_values.ndim >= 3:
                final_values = metric_values[:, -1]
            elif metric_values.ndim == 2:
                final_values = metric_values[:, -1:]
            else:
                final_values = metric_values
            final_cost = final_values.reshape(candidate_count, -1).sum(dim=-1)
            if metric_name.endswith("_position_tolerance"):
                cost_sum += getattr(self.config, "position_rank_weight", 1.0) * final_cost
            elif metric_name.endswith("_orientation_tolerance"):
                cost_sum += getattr(self.config, "orientation_rank_weight", 1.0) * final_cost
            if metric_name == "start_cspace_dist_tolerance":
                cost_sum += getattr(self.config, "cspace_rank_weight", 1.0) * final_cost
        cost_sum = torch.where(success, cost_sum, cost_sum + 1e16)

        _, topk_abs_idx = stable_topk(
            cost_sum.view(batch_size, num_seeds),
            k=return_seeds,
            largest=False,
            dim=-1,
        )
        topk_abs_idx.add_(self._get_result_batch_offsets(batch_size, num_seeds, opt_result.device))
        flat_indices = topk_abs_idx.view(-1)

        q_sol = (
            opt_result.view(-1, self.action_dim)
            .index_select(0, flat_indices)
            .view(batch_size, return_seeds, self.action_dim)
        )
        success_topk = success.index_select(0, flat_indices).view(batch_size, return_seeds)
        feasible_topk = feasible_tensor.index_select(0, flat_indices).view(
            batch_size, return_seeds
        )
        all_state = metrics_result.state
        all_kinematics = all_state.cuda_robot_model_state
        num_links = len(all_kinematics.tool_poses.tool_frames)
        num_spheres = all_kinematics.robot_spheres.shape[-2]
        solution_state = RobotState(
            cuda_robot_model_state=KinematicsState(
                tool_poses=ToolPose(
                    tool_frames=all_kinematics.tool_poses.tool_frames,
                    position=all_kinematics.tool_poses.position.view(-1, num_links, 3)
                    .index_select(0, flat_indices)
                    .view(batch_size, return_seeds, num_links, 3),
                    quaternion=all_kinematics.tool_poses.quaternion.view(-1, num_links, 4)
                    .index_select(0, flat_indices)
                    .view(batch_size, return_seeds, num_links, 4),
                ),
                robot_spheres=all_kinematics.robot_spheres.view(-1, num_spheres, 4)
                .index_select(0, flat_indices)
                .view(batch_size, return_seeds, num_spheres, 4),
                robot_com=all_kinematics.robot_com.view(-1, 4)
                .index_select(0, flat_indices)
                .view(batch_size, return_seeds, -1, 4),
            ),
            joint_state=JointState.from_position(q_sol, joint_names=self.joint_names),
        )

        goalset_index_topk = None
        if pose_summary.absolute_goalset_index is not None:
            goalset_width = pose_summary.absolute_goalset_index.shape[-1]
            goalset_index_topk = pose_summary.absolute_goalset_index[topk_abs_idx.view(-1)].view(
                batch_size,
                return_seeds,
                goalset_width,
            )

        if pose_summary.absolute_position_error is not None:
            position_error = torch.max(
                pose_summary.absolute_position_error[topk_abs_idx.view(-1)].view(
                    batch_size,
                    return_seeds,
                    -1,
                ),
                dim=-1,
            )[0]
        else:
            position_error = torch.zeros(
                (batch_size, return_seeds),
                dtype=opt_result.dtype,
                device=opt_result.device,
            )
        if pose_summary.absolute_rotation_error is not None:
            rotation_error = torch.max(
                pose_summary.absolute_rotation_error[topk_abs_idx.view(-1)].view(
                    batch_size,
                    return_seeds,
                    -1,
                ),
                dim=-1,
            )[0]
        else:
            rotation_error = torch.zeros_like(position_error)

        transition_model = self.auxiliary_rollout.transition_model
        if hasattr(transition_model, "get_full_dof_from_position"):
            js_solution = transition_model.get_full_dof_from_position(q_sol)
        else:
            js_solution = transition_model.get_full_dof_from_solution(solution_state.joint_state)

        ik_result = IKSolverResult(
            success=success_topk,
            feasible=feasible_topk,
            solution=q_sol,
            js_solution=js_solution,
            position_error=position_error,
            rotation_error=rotation_error,
            goalset_index=(
                goalset_index_topk
                if goalset_index_topk is not None
                else torch.full_like(position_error, -1, dtype=torch.long)
            ),
            solve_time=0.0,
            debug_info={
                "seed_idx": topk_abs_idx,
            },
            optimized_seeds=opt_result.view(batch_size, num_seeds, self.action_dim),
            solution_state=solution_state,
        )

        return ik_result

    def _get_result_convergence_layout(
        self,
        metrics_result: RolloutMetrics,
    ) -> _IKConvergenceLayout:
        """Return cached field indices for a stable convergence collection."""
        names = tuple(metrics_result.convergence.names)
        if names != self._result_convergence_names:
            self._result_convergence_names = names
            self._result_convergence_layout = _IKConvergenceLayout(
                position_indices=tuple(
                    index for index, name in enumerate(names) if "position_tolerance" in name
                ),
                orientation_indices=tuple(
                    index for index, name in enumerate(names) if "orientation_tolerance" in name
                ),
                goalset_indices=tuple(
                    index for index, name in enumerate(names) if "goalset_index" in name
                ),
                cspace_indices=tuple(
                    index
                    for index, name in enumerate(names)
                    if "start_cspace_dist_tolerance" in name
                ),
            )
        layout = self._result_convergence_layout
        if layout is None or not layout.position_indices or not layout.orientation_indices:
            log_and_raise("IK metrics require position and orientation convergence values")
        return layout

    @staticmethod
    def _get_result_convergence_values(
        metrics_result: RolloutMetrics,
        indices: tuple[int, ...],
        problem_count: int,
    ) -> torch.Tensor:
        """Return selected convergence fields flattened after the problem dimension."""
        values = metrics_result.convergence.values
        if len(indices) == 1:
            selected = values[indices[0]]
        else:
            selected = torch.cat([values[index] for index in indices], dim=-1)
        return selected.view(problem_count, -1)

    def _get_result_feasibility(
        self,
        metrics_result: RolloutMetrics,
        problem_count: int,
    ) -> torch.Tensor:
        """Reduce the cached per-timestep feasibility for each IK seed."""
        hybrid_values = metrics_result.costs_and_constraints.hybrid_costs_constraints.values
        feasible = metrics_result.feasible
        if feasible is None or hybrid_values:
            feasible = metrics_result.costs_and_constraints.get_feasible(
                include_all_hybrid=False,
                sum_horizon=True,
            )
        if isinstance(feasible, bool):
            return torch.full(
                (problem_count,),
                feasible,
                device=self.device_cfg.device,
                dtype=torch.bool,
            )
        feasible_view = feasible.view(problem_count, -1)
        return (
            feasible_view[:, 0]
            if feasible_view.shape[-1] == 1
            else torch.all(feasible_view, dim=-1)
        )

    def _get_result_batch_offsets(
        self,
        batch_size: int,
        num_seeds: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return cached flat-index offsets with shape ``[batch_size, 1]``."""
        shape = (batch_size, num_seeds)
        if (
            getattr(self, "_result_batch_offsets", None) is None
            or shape != getattr(self, "_result_batch_offset_shape", None)
            or self._result_batch_offsets.device != device
        ):
            self._result_batch_offset_shape = shape
            self._result_batch_offsets = (
                torch.arange(batch_size, device=device, dtype=torch.long) * num_seeds
            ).view(-1, 1)
        return self._result_batch_offsets

    def _compute_solution_velocity(
        self,
        ik_result: IKSolverResult,
        current_state: Optional[JointState],
        return_seeds: int,
    ) -> None:
        """Set ``js_solution.velocity`` via finite difference when ``optimization_dt`` is set."""
        if (
            self.config.optimization_dt is None
            or current_state is None
            or current_state.dt is None
            or ik_result.js_solution is None
        ):
            return

        active_cur = self.get_active_js(current_state)
        active_sol = self.get_active_js(ik_result.js_solution)

        cur_pos = active_cur.position
        sol_pos = active_sol.position
        dt = current_state.dt

        cur_pos_exp = cur_pos.unsqueeze(1)
        dt_val = dt.view(-1, 1, 1) if dt.ndim == 1 else dt[:, :1].unsqueeze(-1)

        active_vel = (sol_pos - cur_pos_exp) / dt_val

        if sol_pos.shape[-1] == ik_result.js_solution.position.shape[-1]:
            ik_result.js_solution.velocity = active_vel
        else:
            full_vel = torch.zeros_like(ik_result.js_solution.position)
            active_indices = [ik_result.js_solution.joint_names.index(j) for j in self.joint_names]
            full_vel[..., active_indices] = active_vel
            ik_result.js_solution.velocity = full_vel

    # -------------------------------------------------------------------
    # Public solve methods
    # -------------------------------------------------------------------

    @profiler.record_function("ik_solver/solve_pose")
    def solve_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        seed_config=None,
        return_seeds: int = 1,
        run_optimizer: bool = True,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
    ) -> IKSolverResult:
        """Solve IK for target tool poses and return ranked solutions."""
        return self._solve_pose(
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
            seed_config=seed_config,
            return_seeds=return_seeds,
            run_optimizer=run_optimizer,
            tool_relative_transform_target=tool_relative_transform_target,
            use_specialized_result_processor=True,
        )

    @profiler.record_function("ik_solver/solve_goalset")
    def solve_goalset(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        seed_config: Optional[torch.Tensor] = None,
        return_seeds: int = 1,
        run_optimizer: bool = True,
    ) -> IKSolverResult:
        """Solve and rank goalset IK with the low-overhead result selector.

        Args:
            goal_tool_poses: Target poses shaped as
                ``[batch, 1, num_links, num_goalset, 3/4]``.
            current_state: Current joints shaped as ``[batch, dof]``. When
                supplied, configured c-space ranking favors nearby solutions.
            seed_config: Optional IK seeds shaped as ``[batch, num_seeds, dof]``.
            return_seeds: Number of ranked solutions returned per batch row.
            run_optimizer: Run IK optimization instead of evaluating seeds only.

        Returns:
            Ranked IK solutions with ``solution`` shaped as
            ``[batch, return_seeds, dof]``.
        """
        if seed_config is None and current_state is not None:
            seed_config = current_state.position.unsqueeze(1)
        return self._solve_pose(
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
            seed_config=seed_config,
            return_seeds=return_seeds,
            run_optimizer=run_optimizer,
            use_specialized_result_processor=False,
        )

    def _solve_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        seed_config: Optional[torch.Tensor] = None,
        return_seeds: int = 1,
        run_optimizer: bool = True,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
        use_specialized_result_processor: bool = True,
    ) -> IKSolverResult:
        """Solve IK with explicit internal result-processor selection.

        Args:
            goal_tool_poses: Target poses for tool links.
            current_state: Optional current joint state.
            seed_config: Optional initial seeds.
            return_seeds: Number of top solutions to return per problem.
            run_optimizer: When False, skip the main optimizer.
            tool_relative_transform_target: Optional target transforms relative
                to one moving tool frame. One target group is shared by every
                problem; otherwise provide one group per problem.
            use_specialized_result_processor: Use the Warp CUDA-graph result
                selector instead of the lightweight Torch selector.

        Returns:
            IKSolverResult with solutions ranked by cost.
        """
        goal_tool_poses = goal_tool_poses.reorder_links(self.kinematics.tool_frames)

        max_batch = self.config.max_batch_size
        batch_size = goal_tool_poses.batch_size
        timer = CudaEventTimer().start()
        num_seeds = self.config.num_seeds
        if return_seeds > num_seeds:
            log_warn(
                f"Requested {return_seeds} solutions, increasing optimization seeds "
                f"from {num_seeds} to {return_seeds}."
            )
            num_seeds = return_seeds

        num_goalset = goal_tool_poses.num_goalset
        if num_goalset > self.config.max_goalset:
            log_and_raise(
                f"solve_pose: num_goalset={num_goalset} exceeds config.max_goalset="
                f"{self.config.max_goalset}."
            )

        if batch_size > max_batch:
            log_and_raise(
                f"solve_pose: goal_tool_poses.batch_size={batch_size} "
                f"exceeds config.max_batch_size={max_batch}."
            )

        actual_batch_size = batch_size
        needs_pad = batch_size < max_batch
        tool_relative_transform_target = _pad_tool_relative_transform_target(
            tool_relative_transform_target,
            batch_size=batch_size,
            max_batch=max_batch,
        )
        if needs_pad:
            goal_tool_poses, current_state, seed_config = _pad_batch_inputs(
                goal_tool_poses,
                current_state,
                seed_config,
                batch_size,
                max_batch,
            )
        batch_size = max_batch

        if batch_size == 1:
            solve_mode = SolveMode.SINGLE
        elif self.config.multi_env:
            solve_mode = SolveMode.MULTI_ENV
        else:
            solve_mode = SolveMode.BATCH

        solve_state = SolveState(
            solve_type=solve_mode,
            num_ik_seeds=num_seeds,
            batch_size=batch_size,
            num_envs=batch_size if self.config.multi_env else 1,
            num_goalset=num_goalset,
            tool_frames=goal_tool_poses.tool_frames,
        )

        seed_solve_time = 0.0
        current_state = self._fill_current_state_dt(current_state)

        if seed_config is None and current_state is not None:
            seed_config = current_state.position.view(batch_size, 1, -1)

        if self.config.use_lm_seed and run_optimizer:
            seed_result = self.seed_ik_solver.solve_seed_configs(
                goal_tool_poses=goal_tool_poses,
                num_seed_configs=self.config.num_seeds,
                seed_config=seed_config,
                current_state=current_state,
            )
            seed_config = seed_result.js_solution.position.view(
                batch_size, self.config.num_seeds, self.action_dim
            )
            seed_solve_time = seed_result.solve_time

        if seed_config is not None:
            if seed_config.shape[0] != batch_size:
                log_and_raise(
                    f"solve_pose: seed_config batch dimension ({seed_config.shape[0]}) "
                    f"should match config.max_batch_size ({batch_size})."
                )

        if current_state is not None and current_state.shape[0] != batch_size:
            log_and_raise(
                f"solve_pose: current_state batch dimension ({current_state.shape[0]}) "
                f"should match config.max_batch_size ({batch_size})."
            )

        result = self._solve_impl(
            solve_state=solve_state,
            goal_tool_poses=goal_tool_poses,
            num_seeds=num_seeds,
            seed_config=seed_config,
            current_state=current_state,
            return_seeds=return_seeds,
            run_optimizer=run_optimizer,
            tool_relative_transform_target=tool_relative_transform_target,
            use_specialized_result_processor=use_specialized_result_processor,
        )
        result.solve_time += seed_solve_time
        result.total_time = timer.stop()

        if needs_pad:
            result = _slice_batch_result(result, actual_batch_size)
        return result

    @profiler.record_function("ik_solver/update_world")
    def update_world(self, scene_cfg: SceneCfg) -> None:
        """Reload the collision model from a new scene configuration."""
        self.scene_collision_checker.load_collision_model(scene_cfg)
