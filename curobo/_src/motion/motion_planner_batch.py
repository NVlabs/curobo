# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch motion planner for solving multiple planning problems in parallel.

Unlike :class:`MotionPlanner` (which solves one problem with retry logic and
graph-planner seeding), :class:`BatchMotionPlanner` solves ``batch_size``
independent problems in a single IK + TrajOpt pass with no retries.

When ``multi_env=False`` (shared collision world), PRM or RRT can provide
trajectory seeds. With ``multi_env=True`` (per-problem collision worlds), the
flattened-lane RRT path remains available and carries an environment ID for
each problem; PRM graph seeding remains disabled.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch

from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.cost.criteria_tool_relative_transform import ToolRelativeTransformCriteria
from curobo._src.cost.criteria_tool_velocity_limit import ToolVelocityLimitCriteria
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.collision.collision_scene import create_scene_collision
from curobo._src.geom.types import SceneCfg
from curobo._src.motion.motion_planner import (
    _can_evaluate_seed,
    _create_graph_seed_fitter,
    _fit_linear_seed_trajectories,
    _make_warmup_goalset_poses,
    _merge_graph_seed_trajectories,
    _should_evaluate_seed_before_trajopt,
)
from curobo._src.motion.motion_planner_cfg import (
    MotionPlannerCfg,
    is_rrt_graph_planner_algorithm,
)
from curobo._src.motion.motion_planner_result import GraspPlanResult
from curobo._src.motion.planner_grasp import (
    GraspPhasePlanner,
    get_result_goal_configuration,
)
from curobo._src.motion.planner_grasp import axis_string_to_vector as _axis_string_to_vector
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index
from curobo._src.types.pose import Pose
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_and_raise


class BatchMotionPlanner:
    """Batch motion planner: solves ``batch_size`` problems in parallel.

    Single IK → (optional graph seed) → TrajOpt pass with no retries.
    RRT graph seeding supports shared and per-problem collision worlds. PRM
    graph seeding is available only for a shared collision world.

    Args:
        config: Planner configuration.  ``max_batch_size`` and ``multi_env``
            are read from ``config.ik_solver_config``.
    """

    def __init__(self, config: MotionPlannerCfg):
        self.config = config
        self.device_cfg = config.device_cfg
        self.scene_collision_checker = None
        self._destroyed = False
        self._warming_up = False
        self._last_graph_seed_success = None
        self._grasp_phase_planner: GraspPhasePlanner
        self._initialize_components()

    def _initialize_components(self):
        if self.config.scene_collision_cfg is not None:
            self.scene_collision_checker = create_scene_collision(self.config.scene_collision_cfg)

        self.ik_solver = IKSolver(self.config.ik_solver_config, self.scene_collision_checker)
        self.trajopt_solver = TrajOptSolver(
            self.config.trajopt_solver_config, self.scene_collision_checker
        )

        self.graph_planner = None
        self.constrained_rrt_planner = None
        self.constrained_rrt_seed_fitter = None
        self.graph_seed_fitter = _create_graph_seed_fitter(
            self.config,
            self.trajopt_solver,
            self.config.ik_solver_config.max_batch_size * self.trajopt_solver.config.num_seeds,
        )
        if (
            self.config.prm_graph_planner_config is not None
            or self.config.rrt_graph_planner_config is not None
        ) and (
            not self.config.ik_solver_config.multi_env
            or is_rrt_graph_planner_algorithm(self.config.graph_planner_algorithm)
        ):
            if is_rrt_graph_planner_algorithm(self.config.graph_planner_algorithm):
                from curobo._src.graph_planner.rrt import (
                    BiRRTPlanner,
                    RRTPlannerResources,
                )

                rrt_config = self.config.rrt_graph_planner_config
                if rrt_config is None:
                    log_and_raise("RRT selection requires rrt_graph_planner_config")
                seed_count = self.config.trajopt_solver_config.num_seeds
                required_lanes = self.config.ik_solver_config.max_batch_size * seed_count
                if required_lanes > rrt_config.planner_config.max_batched_lanes:
                    log_and_raise(
                        "RRT batch*seed lanes exceed max_batched_lanes: "
                        f"{required_lanes} > {rrt_config.planner_config.max_batched_lanes}"
                    )
                self.graph_planner = BiRRTPlanner(
                    RRTPlannerResources(
                        rrt_config.resources_config,
                        self.scene_collision_checker,
                    ),
                    rrt_config.planner_config,
                )
            else:
                from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner

                self.graph_planner = PRMGraphPlanner(
                    self.config.prm_graph_planner_config,
                    self.scene_collision_checker,
                )
        if self.config.constrained_rrt_planner_config is not None:
            from curobo._src.graph_planner.rrt import BiRRTPlanner, RRTPlannerResources

            constrained_config = self.config.constrained_rrt_planner_config
            required_lanes = self.config.ik_solver_config.max_batch_size
            if required_lanes > constrained_config.planner_config.max_batched_lanes:
                log_and_raise(
                    "Constrained RRT batch exceeds max_batched_lanes: "
                    f"{required_lanes} > "
                    f"{constrained_config.planner_config.max_batched_lanes}"
                )
            self.constrained_rrt_planner = BiRRTPlanner(
                RRTPlannerResources(
                    constrained_config.resources_config,
                    self.scene_collision_checker,
                ),
                constrained_config.planner_config,
            )
            self.constrained_rrt_planner.initialize_query_shape(required_lanes)
            self.constrained_rrt_seed_fitter = _create_graph_seed_fitter(
                self.config,
                self.trajopt_solver,
                num_lanes=required_lanes,
            )
        self._grasp_phase_planner = GraspPhasePlanner(
            config=self.config,
            batch_size=self.config.ik_solver_config.max_batch_size,
            kinematics=self.kinematics,
            ik_solver=self.ik_solver,
            trajopt_solver=self.trajopt_solver,
            constrained_rrt_planner=self.constrained_rrt_planner,
            constrained_rrt_seed_fitter=self.constrained_rrt_seed_fitter,
            plan_pose=lambda *args, **kwargs: self.plan_pose(*args, **kwargs),
            update_tool_pose_criteria=lambda criteria: self.update_tool_pose_criteria(criteria),
            enable_link_collision=lambda links: self.enable_link_collision(links),
            disable_link_collision=lambda links: self.disable_link_collision(links),
        )

    # -- Lifecycle --

    def destroy(self):
        """Release all CUDA graph resources.

        Call before dropping the last reference to avoid ``cudaMallocAsync``
        warnings about captured allocations freed outside graph replay.
        """
        if getattr(self, "_destroyed", True):
            return
        self._destroyed = True
        grasp_phase_planner = getattr(self, "_grasp_phase_planner", None)
        if grasp_phase_planner is not None:
            grasp_phase_planner.destroy()
        self.ik_solver.destroy()
        self.trajopt_solver.destroy()
        if self.graph_planner is not None and hasattr(self.graph_planner, "reset_cuda_graph"):
            self.graph_planner.reset_cuda_graph()
        constrained_rrt_planner = getattr(self, "constrained_rrt_planner", None)
        if constrained_rrt_planner is not None:
            constrained_rrt_planner.reset_cuda_graph()
        graph_seed_fitter = getattr(self, "graph_seed_fitter", None)
        if graph_seed_fitter is not None:
            graph_seed_fitter.reset_cuda_graph()
        constrained_rrt_seed_fitter = getattr(self, "constrained_rrt_seed_fitter", None)
        if constrained_rrt_seed_fitter is not None:
            constrained_rrt_seed_fitter.reset_cuda_graph()

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()
        return False

    @property
    def attachment_manager(self) -> AttachmentManager:
        """Attachment manager for attaching/detaching obstacles to robot links."""
        return self.trajopt_solver.core.attachment_manager

    @property
    def batch_size(self) -> int:
        return self.config.ik_solver_config.max_batch_size

    def warmup(
        self,
        enable_graph: bool = True,
        num_warmup_iterations: int = 5,
    ) -> bool:
        """JIT-warmup the solvers and (optionally) the graph planner.

        When ``max_goalset > 1``, the maximum goalset is solved before
        singleton c-space and pose requests so CUDA graphs retain the largest
        fixed goal buffer and reuse it through padding.

        Args:
            enable_graph: Warm up the graph planner when available.
            num_warmup_iterations: Number of synthetic solve rounds.
        """
        self._warming_up = True
        og_exit_early = self.ik_solver.config.exit_early
        self.ik_solver.config.exit_early = False

        batch_size = self.batch_size
        num_goalset = self.config.ik_solver_config.max_goalset
        for _ in range(num_warmup_iterations):
            current_state = self.default_joint_state.clone().position.unsqueeze(0)
            current_state = current_state.repeat(batch_size, 1)
            current_state = JointState.from_position(
                current_state,
                joint_names=self.joint_names,
            )

            goal_state = current_state.clone()
            goal_state.position[..., 0] += 0.2

            if num_goalset > 1:
                goalset_poses = _make_warmup_goalset_poses(
                    self.compute_kinematics,
                    current_state,
                    num_goalset,
                    joint_index=0,
                    joint_delta=0.2,
                )
                goalset_goal = GoalToolPose.from_poses(
                    goalset_poses,
                    ordered_tool_frames=self.tool_frames,
                    num_goalset=num_goalset,
                )
                self.plan_pose(
                    goalset_goal,
                    current_state,
                    enable_graph_attempt=0 if enable_graph else 1,
                )
            self.plan_cspace(
                goal_state,
                current_state,
                enable_graph_attempt=0 if enable_graph else 1,
            )
            goal_tool_poses = self.compute_kinematics(goal_state).tool_poses.as_goal()
            self.plan_pose(
                goal_tool_poses,
                current_state,
                enable_graph_attempt=0 if enable_graph else 1,
            )
            self.reset_seed()

        if enable_graph and self.graph_planner is not None:
            self.graph_planner.warmup(
                num_warmup_iterations=num_warmup_iterations,
                max_batch_size=batch_size * self.trajopt_solver.config.num_seeds,
            )
        constrained_rrt_planner = getattr(self, "constrained_rrt_planner", None)
        if enable_graph and constrained_rrt_planner is not None:
            constrained_state = JointState.from_position(
                self.default_joint_state.position.unsqueeze(0).repeat(batch_size, 1),
                joint_names=self.joint_names,
            )
            constrained_goal = self.compute_kinematics(constrained_state).tool_poses.as_goal()
            self._grasp_phase_planner.install_rrt_constraint(constrained_goal)
            constrained_rrt_planner.warmup(
                num_warmup_iterations=num_warmup_iterations,
                max_batch_size=batch_size,
            )
            self._grasp_phase_planner.activate_rrt_constraint(
                constrained_goal,
                axis="z",
                in_tool_frame=True,
            )
            self.ik_solver.solve_goalset(
                constrained_goal,
                current_state=constrained_state,
                return_seeds=1,
            )
        graph_seed_fitter = getattr(self, "graph_seed_fitter", None)
        if graph_seed_fitter is not None:
            graph_seed_fitter.warmup()
        constrained_rrt_seed_fitter = getattr(self, "constrained_rrt_seed_fitter", None)
        if constrained_rrt_seed_fitter is not None:
            constrained_rrt_seed_fitter.warmup()

        self.ik_solver.config.exit_early = og_exit_early
        self._warming_up = False
        return True

    def plan_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 1,
        success_ratio: float = 1.0,
        enable_graph_attempt: int = 0,
        finetune_attempts: int = 1,
        initial_iters: Optional[int] = None,
        time_optimal_iters: Optional[int] = 25,
        finetune_iters: Optional[int] = 25,
        finetune_dt_scale: float = 0.6,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
    ) -> Optional[TrajOptSolverResult]:
        """Plan trajectories for a batch of pose targets.

        Runs up to ``max_attempts`` IK -> TrajOpt passes.  On each attempt
        the full batch is re-solved; per-problem results are locked in on
        first success (first-success-wins).  The loop exits early when the
        fraction of solved problems reaches ``success_ratio``.

        Args:
            goal_tool_poses: Target poses as ``GoalToolPose``
                ``[B, H, L, G, 3/4]``.
            current_state: Initial joint states ``(batch_size, dof)``.
            use_implicit_goal: Use IK solution as implicit trajectory goal.
            max_attempts: Maximum number of IK + TrajOpt passes.
            success_ratio: Exit early when this fraction of the batch has
                succeeded (0.0-1.0).  Default 1.0 means wait for all.
            enable_graph_attempt: Attempt index at which to start graph
                seeding (when graph planner is available).
            finetune_attempts: Number of additional time-optimal refinement
                passes after the initial solve.
            initial_iters: Optional optimizer iteration override for the
                initial solve.
            time_optimal_iters: Optional optimizer iteration override for the
                first refinement pass.
            finetune_iters: Optional optimizer iteration override for later
                refinement passes.
            finetune_dt_scale: Multiplicative factor applied to the trajectory
                timestep between refinement passes.
            tool_relative_transform_target: Optional target transforms relative
                to one moving tool frame.

        Returns:
            TrajOptSolverResult with per-problem success, or None if IK
            never found any solution across all attempts.
        """
        batch_size = self.batch_size
        num_seeds = self.trajopt_solver.config.num_seeds
        device = current_state.position.device

        best_result: Optional[TrajOptSolverResult] = None
        solved = torch.zeros(batch_size, dtype=torch.bool, device=device)
        total_time = 0.0

        for attempt in range(max_attempts):
            ik_result = self.ik_solver.solve_pose(
                goal_tool_poses,
                return_seeds=num_seeds,
                current_state=current_state,
                tool_relative_transform_target=tool_relative_transform_target,
            )
            total_time += ik_result.total_time

            if torch.count_nonzero(ik_result.success) == 0:
                continue

            seed_fitter = getattr(self, "graph_seed_fitter", None) if use_implicit_goal else None
            seed_traj = _fit_linear_seed_trajectories(
                current_state,
                ik_result.solution,
                self.trajopt_solver.action_horizon,
                seed_fitter,
            )
            trajectory_source = "linear_fit" if seed_fitter is not None else "linear"
            used_graph_seed = False
            if self.graph_planner is not None and attempt >= enable_graph_attempt:
                seed_traj = self._get_graph_seed_trajectories(
                    current_state,
                    ik_result.solution,
                    goal_valid=ik_result.success,
                    use_bspline_fitter=use_implicit_goal,
                )
                trajectory_source = "graph_fit" if seed_fitter is not None else "graph"
                used_graph_seed = True

            if _should_evaluate_seed_before_trajopt(
                self.config, used_graph_seed
            ) and _can_evaluate_seed(self.config, seed_fitter, self._warming_up):
                seed_result = self.trajopt_solver.evaluate_pose_seed(
                    goal_tool_poses,
                    current_state,
                    seed_traj,
                    use_implicit_goal=use_implicit_goal,
                    trajectory_source=trajectory_source,
                    seed_valid_mask=(self._last_graph_seed_success if used_graph_seed else None),
                    tool_relative_transform_target=tool_relative_transform_target,
                )
                total_time += seed_result.total_time
                if best_result is None:
                    best_result = seed_result
                    solved = seed_result.success.any(dim=-1)
                else:
                    newly_solved = seed_result.success.any(dim=-1) & ~solved
                    if newly_solved.any():
                        best_result.copy_at_batch_indices(seed_result, newly_solved)
                        solved |= newly_solved
                if solved.float().mean() >= success_ratio:
                    break
                if not self.config.enable_trajopt:
                    break

            trajopt_result = self.trajopt_solver.solve_pose(
                goal_tool_poses,
                current_state,
                seed_config=ik_result.solution,
                seed_traj=seed_traj,
                use_implicit_goal=use_implicit_goal,
                finetune_attempts=finetune_attempts,
                initial_iters=initial_iters,
                time_optimal_iters=time_optimal_iters,
                finetune_iters=finetune_iters,
                finetune_dt_scale=finetune_dt_scale,
                tool_relative_transform_target=tool_relative_transform_target,
            )
            total_time += trajopt_result.total_time
            trajopt_result.debug_info["trajectory_source"] = "trajopt"
            trajopt_result.trajectory_source = "trajopt"

            if best_result is None:
                best_result = trajopt_result
                solved = trajopt_result.success.any(dim=-1)
            else:
                previously_solved = solved.clone()
                newly_solved = trajopt_result.success.any(dim=-1) & ~solved
                if newly_solved.any():
                    best_result.copy_at_batch_indices(trajopt_result, newly_solved)
                    solved = solved | newly_solved
                    best_result.debug_info["trajectory_source"] = (
                        "mixed" if previously_solved.any() else "trajopt"
                    )
                    best_result.trajectory_source = best_result.debug_info["trajectory_source"]

            if solved.float().mean() >= success_ratio:
                break

            # CUDA-graph solver results borrow the processor's output buffers,
            # which are overwritten by the next solve.  Take ownership only
            # when this first result must survive another attempt.
            if attempt + 1 < max_attempts and best_result is trajopt_result:
                best_result = best_result.clone()

        if best_result is not None:
            best_result.total_time = total_time
        return best_result

    def plan_cspace(
        self,
        goal_states: JointState,
        current_state: JointState,
        max_attempts: int = 1,
        success_ratio: float = 1.0,
        enable_graph_attempt: int = 0,
        finetune_attempts: int = 1,
    ) -> Optional[TrajOptSolverResult]:
        """Plan trajectories for a batch of joint-space targets.

        Same retry / first-success-wins logic as :meth:`plan_pose`.

        Args:
            goal_states: Target joint configurations ``(batch_size, dof)``.
            current_state: Initial joint states ``(batch_size, dof)``.
            max_attempts: Maximum number of TrajOpt passes.
            success_ratio: Exit early when this fraction of the batch
                has succeeded.
            enable_graph_attempt: Attempt index to start graph seeding.
            finetune_attempts: Additional 25-iteration TrajOpt refinement passes.

        Returns:
            TrajOptSolverResult with per-problem success.
        """
        batch_size = self.batch_size
        device = current_state.position.device

        best_result: Optional[TrajOptSolverResult] = None
        solved = torch.zeros(batch_size, dtype=torch.bool, device=device)
        total_time = 0.0

        num_seeds = self.trajopt_solver.config.num_seeds
        dof = self.trajopt_solver.action_dim

        for attempt in range(max_attempts):
            goal_configs = goal_states.position.view(batch_size, 1, dof)
            goal_configs = goal_configs.repeat(1, num_seeds, 1)
            seed_fitter = getattr(self, "graph_seed_fitter", None)
            seed_traj = _fit_linear_seed_trajectories(
                current_state,
                goal_configs,
                self.trajopt_solver.action_horizon,
                seed_fitter,
            )
            trajectory_source = "linear_fit" if seed_fitter is not None else "linear"
            used_graph_seed = False
            if self.graph_planner is not None and attempt >= enable_graph_attempt:
                seed_traj = self._get_graph_seed_trajectories(
                    current_state,
                    goal_configs,
                )
                trajectory_source = "graph_fit" if seed_fitter is not None else "graph"
                used_graph_seed = True

            if _should_evaluate_seed_before_trajopt(
                self.config, used_graph_seed
            ) and _can_evaluate_seed(self.config, seed_fitter, self._warming_up):
                seed_result = self.trajopt_solver.evaluate_cspace_seed(
                    goal_states,
                    current_state,
                    seed_traj,
                    trajectory_source=trajectory_source,
                    seed_valid_mask=(self._last_graph_seed_success if used_graph_seed else None),
                )
                total_time += seed_result.total_time
                if best_result is None:
                    best_result = seed_result
                    solved = seed_result.success.any(dim=-1)
                else:
                    newly_solved = seed_result.success.any(dim=-1) & ~solved
                    if newly_solved.any():
                        best_result.copy_at_batch_indices(seed_result, newly_solved)
                        solved |= newly_solved
                if solved.float().mean() >= success_ratio:
                    break
                if not self.config.enable_trajopt:
                    break

            trajopt_result = self.trajopt_solver.solve_cspace(
                goal_states,
                current_state,
                seed_traj=seed_traj,
                finetune_attempts=finetune_attempts,
            )
            total_time += trajopt_result.total_time
            trajopt_result.debug_info["trajectory_source"] = "trajopt"
            trajopt_result.trajectory_source = "trajopt"

            if best_result is None:
                best_result = trajopt_result
                solved = trajopt_result.success.any(dim=-1)
            else:
                previously_solved = solved.clone()
                newly_solved = trajopt_result.success.any(dim=-1) & ~solved
                if newly_solved.any():
                    best_result.copy_at_batch_indices(trajopt_result, newly_solved)
                    solved = solved | newly_solved
                    best_result.debug_info["trajectory_source"] = (
                        "mixed" if previously_solved.any() else "trajopt"
                    )
                    best_result.trajectory_source = best_result.debug_info["trajectory_source"]

            if solved.float().mean() >= success_ratio:
                break

            # CUDA-graph solver results borrow the processor's output buffers,
            # which are overwritten by the next solve.  Take ownership only
            # when this first result must survive another attempt.
            if attempt + 1 < max_attempts and best_result is trajopt_result:
                best_result = best_result.clone()

        if best_result is not None:
            best_result.total_time = total_time
        return best_result

    def plan_grasp(
        self,
        grasp_poses: GoalToolPose,
        current_state: JointState,
        grasp_approach_axis: str = "z",
        grasp_approach_offset: float = -0.15,
        grasp_approach_in_tool_frame: bool = True,
        grasp_lift_axis: str = "z",
        grasp_lift_offset: float = -0.15,
        grasp_lift_in_tool_frame: bool = True,
        plan_approach_to_grasp: bool = True,
        plan_grasp_to_lift: bool = True,
        disable_collision_links: Optional[List[str]] = None,
        use_trajopt_for_goalset_selection: bool = False,
        use_rrt_for_constrained_phases: bool = False,
        use_double_pass_trajopt_for_constrained_phases: bool = False,
    ) -> GraspPlanResult:
        """Plan grasp motions for a batch: goalset -> approach -> grasp -> lift.

        All B problems are planned at every stage (CUDA graph stability).
        Problems that fail a stage get their current state substituted as
        the goal for subsequent stages so the optimizer doesn't diverge.
        Goalset selection uses ranked IK and constrained phases use one TrajOpt
        pass by default. Full goalset TrajOpt, projected RRT, and a second
        constrained TrajOpt pass are opt-in.
        """
        if use_rrt_for_constrained_phases and use_double_pass_trajopt_for_constrained_phases:
            log_and_raise(
                "use_rrt_for_constrained_phases and "
                "use_double_pass_trajopt_for_constrained_phases are mutually exclusive"
            )
        if disable_collision_links is None:
            disable_collision_links = (
                self.kinematics.config.kinematics_config.grasp_contact_link_names
            )
        if disable_collision_links is None:
            disable_collision_links = []

        batch_size = self.batch_size
        device = current_state.position.device
        tool_frames = grasp_poses.tool_frames

        result = GraspPlanResult()
        result.success = torch.zeros(batch_size, dtype=torch.bool, device=device)
        result.approach_success = torch.zeros(batch_size, dtype=torch.bool, device=device)
        result.grasp_success = torch.zeros(batch_size, dtype=torch.bool, device=device)
        result.lift_success = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # -- Stage 1: Goalset solve --
        self._grasp_phase_planner.activate_collision(disable_collision_links, enable=False)
        if use_trajopt_for_goalset_selection:
            goalset_result = self.plan_pose(grasp_poses, current_state)
        else:
            goalset_result = self.ik_solver.solve_goalset(
                grasp_poses,
                return_seeds=1,
                current_state=current_state,
            )
        self._grasp_phase_planner.activate_collision(disable_collision_links, enable=True)

        if goalset_result is None:
            result.status = "Goalset planning returned None."
            return result

        goalset_ok = goalset_result.success.any(dim=-1)
        result.goalset_result = goalset_result

        if not goalset_ok.any():
            result.status = "No grasp in goal set was reachable."
            return result

        result.goalset_index = goalset_result.goalset_index.clone()
        grasp_goal_configuration = get_result_goal_configuration(
            goalset_result,
            self.kinematics,
        )

        # Extract per-problem grasp pose using each problem's goalset_index
        selected_grasp = self._extract_per_problem_grasp(
            grasp_poses,
            goalset_result.goalset_index,
            goalset_ok,
            device,
        )

        # -- Stage 2: Plan to approach pose --
        approach_offset = Pose.from_list(
            [grasp_approach_offset * v for v in _axis_string_to_vector(grasp_approach_axis)]
            + [1, 0, 0, 0]
        )
        approach_poses_dict = {}
        for frame in tool_frames:
            grasp_pose = selected_grasp[frame]
            if grasp_approach_in_tool_frame:
                approach_poses_dict[frame] = grasp_pose.multiply(approach_offset)
            else:
                approach_poses_dict[frame] = approach_offset.multiply(grasp_pose)

        approach_tool_pose = GoalToolPose.from_poses(
            approach_poses_dict,
            ordered_tool_frames=tool_frames,
            num_goalset=1,
        )
        approach_result = self.plan_pose(approach_tool_pose, current_state)
        result.approach_result = approach_result

        if approach_result is None:
            result.status = "Planning to approach pose failed."
            return result

        approach_ok = goalset_ok & approach_result.success.any(dim=-1)
        result.approach_success = approach_ok.clone()
        result.approach_trajectory = approach_result.js_solution.clone()
        result.approach_trajectory_dt = result.approach_trajectory.dt
        result.approach_interpolated_trajectory = approach_result.interpolated_trajectory.clone()
        result.approach_interpolated_last_tstep = approach_result.interpolated_last_tstep.clone()

        if not plan_approach_to_grasp:
            result.success = approach_ok.clone()
            result.status = "Planning to approach pose completed."
            return result

        # -- Stage 3: Linear motion approach -> grasp --
        approach_end = get_joint_state_at_horizon_index(approach_result.js_solution, -1).squeeze(1)
        approach_end = self.kinematics.get_active_js(approach_end)
        approach_end.dt = None

        grasp_tool_pose = GoalToolPose.from_poses(
            selected_grasp,
            ordered_tool_frames=tool_frames,
            num_goalset=1,
        )
        # For failed problems, substitute current state as goal
        self._substitute_fallback_goal(grasp_tool_pose, approach_end, ~approach_ok)

        self._grasp_phase_planner.activate_collision(disable_collision_links, enable=False)
        grasp_result = self._grasp_phase_planner.plan_linear_phase(
            grasp_tool_pose,
            approach_end,
            grasp_approach_axis,
            grasp_approach_in_tool_frame,
            use_rrt=use_rrt_for_constrained_phases,
            finetune_attempts=int(use_double_pass_trajopt_for_constrained_phases),
            goal_configuration=grasp_goal_configuration,
            goal_valid=goalset_ok,
        )
        self._grasp_phase_planner.activate_collision(disable_collision_links, enable=True)

        if grasp_result is None:
            result.status = "Planning to grasp pose failed."
            return result

        grasp_ok = approach_ok & grasp_result.success.any(dim=-1)
        result.grasp_success = grasp_ok.clone()
        result.grasp_trajectory = grasp_result.js_solution.clone()
        result.grasp_trajectory_dt = result.grasp_trajectory.dt
        result.grasp_interpolated_trajectory = grasp_result.interpolated_trajectory.clone()
        result.grasp_interpolated_last_tstep = grasp_result.interpolated_last_tstep.clone()

        if not plan_grasp_to_lift:
            result.success = grasp_ok.clone()
            result.status = "Planning to grasp pose completed."
            return result

        # -- Stage 4: Lift --
        lift_start = get_joint_state_at_horizon_index(grasp_result.js_solution, -1).squeeze(1)
        lift_start = self.kinematics.get_active_js(lift_start)
        lift_start.dt = None

        lift_offset = Pose.from_list(
            [grasp_lift_offset * v for v in _axis_string_to_vector(grasp_lift_axis)] + [1, 0, 0, 0]
        )
        lift_poses_dict = {}
        for frame in tool_frames:
            grasp_pose = selected_grasp[frame]
            if grasp_lift_in_tool_frame:
                lift_poses_dict[frame] = grasp_pose.multiply(lift_offset)
            else:
                lift_poses_dict[frame] = lift_offset.multiply(grasp_pose)

        lift_tool_pose = GoalToolPose.from_poses(
            lift_poses_dict,
            ordered_tool_frames=tool_frames,
            num_goalset=1,
        )
        self._substitute_fallback_goal(lift_tool_pose, lift_start, ~grasp_ok)

        self._grasp_phase_planner.activate_collision(disable_collision_links, enable=False)
        lift_result = self._grasp_phase_planner.plan_linear_phase(
            lift_tool_pose,
            lift_start,
            grasp_lift_axis,
            grasp_lift_in_tool_frame,
            use_rrt=use_rrt_for_constrained_phases,
            finetune_attempts=int(use_double_pass_trajopt_for_constrained_phases),
        )
        self._grasp_phase_planner.activate_collision(disable_collision_links, enable=True)

        if lift_result is None:
            result.status = "Planning to lift pose failed."
            return result

        lift_ok = grasp_ok & lift_result.success.any(dim=-1)
        result.lift_success = lift_ok.clone()
        result.lift_trajectory = lift_result.js_solution
        result.lift_trajectory_dt = lift_result.js_solution.dt
        result.lift_interpolated_trajectory = lift_result.interpolated_trajectory
        result.lift_interpolated_last_tstep = lift_result.interpolated_last_tstep
        result.success = lift_ok.clone()
        result.status = "Grasp planning completed."
        return result

    def _extract_per_problem_grasp(
        self,
        grasp_poses: GoalToolPose,
        goalset_index: torch.Tensor,
        goalset_ok: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, Pose]:
        """Extract per-problem grasp pose from goalset using each problem's selected index.

        For failed problems, uses the first goalset entry as a placeholder.
        """
        batch_size = self.batch_size
        idx = goalset_index[:, 0]
        if idx.ndim > 1:
            idx = idx[:, 0]
        idx = idx.long()
        idx[~goalset_ok] = 0

        batch_idx = torch.arange(batch_size, device=device)
        selected: Dict[str, Pose] = {}
        for frame in grasp_poses.tool_frames:
            li = grasp_poses.tool_frames.index(frame)
            pos = grasp_poses.position[:, 0, li, :, :]  # [B, G, 3]
            quat = grasp_poses.quaternion[:, 0, li, :, :]  # [B, G, 4]
            selected[frame] = Pose(
                position=pos[batch_idx, idx, :],
                quaternion=quat[batch_idx, idx, :],
            )
        return selected

    def _substitute_fallback_goal(
        self,
        goal_tool_pose: GoalToolPose,
        current_state: JointState,
        failed_mask: torch.Tensor,
    ) -> None:
        """For failed problems, replace the goal pose with FK of current_state.

        This keeps the optimizer from diverging on unreachable targets.
        Modifies goal_tool_pose in place.
        """
        if not failed_mask.any():
            return
        kin = self.compute_kinematics(current_state)
        fallback = kin.tool_poses
        for i, frame in enumerate(goal_tool_pose.tool_frames):
            fb_pose = fallback.get_link_pose(frame)
            fb_pos = fb_pose.position[failed_mask]
            fb_quat = fb_pose.quaternion[failed_mask]
            goal_tool_pose.position[failed_mask, :, i, :, :] = fb_pos[:, None, None, :]
            goal_tool_pose.quaternion[failed_mask, :, i, :, :] = fb_quat[:, None, None, :]

    def _get_graph_seed_trajectories(
        self,
        current_state: JointState,
        seed_config: torch.Tensor,
        goal_valid: Optional[torch.Tensor] = None,
        use_bspline_fitter: bool = True,
    ) -> Optional[torch.Tensor]:
        """Query graph planner for trajectory seeds across all batch*seed pairs.

        Flattens ``(B, S)`` into ``(B*S,)`` for the graph planner, then
        reshapes back to ``(B, S, action_horizon, dof)`` for TrajOpt.
        Successful graph paths are reparameterized by joint-space arc length
        and fitted to B-spline knots when configured; lanes without a graph
        path fit a straight joint-space line to the same representation.
        ``use_bspline_fitter`` must be false for non-implicit-goal TrajOpt
        because the fitter's boundary operator pins an implicit joint goal.
        """
        from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType

        batch_size = self.batch_size
        num_seeds = seed_config.shape[1] if seed_config.ndim == 3 else seed_config.shape[0]
        dof = self.trajopt_solver.action_dim
        seed_config = seed_config.reshape(batch_size, num_seeds, dof)
        graph_seed_fitter = (
            getattr(self, "graph_seed_fitter", None) if use_bspline_fitter else None
        )
        graph_horizon = (
            self.trajopt_solver.horizon
            if graph_seed_fitter is not None
            else self.trajopt_solver.action_horizon
        )

        graph_starts = current_state.position.view(batch_size, 1, dof)
        graph_starts = graph_starts.repeat(1, num_seeds, 1).reshape(-1, dof)
        graph_goals = seed_config.view(-1, dof)
        if goal_valid is None:
            goal_valid = torch.ones(
                (batch_size, num_seeds),
                device=seed_config.device,
                dtype=torch.bool,
            )
        else:
            goal_valid = goal_valid.reshape(batch_size, num_seeds).to(dtype=torch.bool)

        graph_kwargs = {
            "interpolate_waypoints": True,
            "interpolation_steps": graph_horizon,
            "interpolation_type": TrajInterpolationType.LINEAR,
            "validate_interpolated_trajectory": False,
        }

        use_flattened_rrt = is_rrt_graph_planner_algorithm(self.config.graph_planner_algorithm)
        if use_flattened_rrt:
            lane_env_idx = (
                torch.arange(
                    batch_size,
                    device=seed_config.device,
                    dtype=torch.int32,
                ).repeat_interleave(num_seeds)
                if self.config.ik_solver_config.multi_env
                else None
            )
            lane_group_id = torch.arange(
                batch_size,
                device=seed_config.device,
                dtype=torch.int32,
            ).repeat_interleave(num_seeds)
            graph_result = self.graph_planner.find_path(
                graph_starts,
                graph_goals,
                query_mask=goal_valid.reshape(-1),
                query_group_id=lane_group_id,
                group_count=batch_size,
                env_query_idx=lane_env_idx,
                **graph_kwargs,
            )
        else:
            graph_result = self.graph_planner.find_path(
                graph_starts.clone(),
                graph_goals.clone(),
                **graph_kwargs,
            )

        graph_success = graph_result.success.reshape(batch_size, num_seeds) & goal_valid
        self._last_graph_seed_success = graph_success
        return _merge_graph_seed_trajectories(
            current_state,
            seed_config,
            graph_result.interpolated_waypoints,
            graph_success,
            self.trajopt_solver.action_horizon,
            graph_seed_fitter,
        )

    # -- Scene / state management (mirrors MotionPlanner) --

    @property
    def joint_names(self) -> List[str]:
        return self.ik_solver.joint_names

    @property
    def action_dim(self) -> int:
        return self.trajopt_solver.action_dim

    @property
    def tool_frames(self) -> List[str]:
        return self.trajopt_solver.tool_frames

    @property
    def default_joint_state(self) -> JointState:
        return self.trajopt_solver.default_joint_state

    @property
    def kinematics(self) -> Kinematics:
        return self.trajopt_solver.kinematics

    def compute_kinematics(self, state: JointState) -> KinematicsState:
        return self.trajopt_solver.compute_kinematics(state)

    def update_world(self, scene_cfg: SceneCfg, env_idx: int = 0):
        """Update one fixed collision environment and clear planner nodes only."""
        self.scene_collision_checker.load_collision_model(scene_cfg, env_idx=env_idx)
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
        constrained_rrt_planner = getattr(self, "constrained_rrt_planner", None)
        if constrained_rrt_planner is not None:
            constrained_rrt_planner.reset_buffer()

    def clear_scene_cache(self):
        """Clear fixed world storage and graph-planner nodes, retaining captures."""
        self.scene_collision_checker.clear_cache()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
        constrained_rrt_planner = getattr(self, "constrained_rrt_planner", None)
        if constrained_rrt_planner is not None:
            constrained_rrt_planner.reset_buffer()

    def reset_seed(self):
        self.ik_solver.reset_seed()
        self.trajopt_solver.reset_seed()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
            self.graph_planner.reset_seed()
        constrained_rrt_planner = getattr(self, "constrained_rrt_planner", None)
        if constrained_rrt_planner is not None:
            constrained_rrt_planner.reset_buffer()
            constrained_rrt_planner.reset_seed()

    def enable_link_collision(self, enable_collision_links: List[str]):
        for link_name in enable_collision_links:
            self.kinematics.config.kinematics_config.enable_link_spheres(link_name)

    def disable_link_collision(self, disable_collision_links: List[str]):
        for link_name in disable_collision_links:
            self.kinematics.config.kinematics_config.disable_link_spheres(link_name)

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, "ToolPoseCriteria"]):
        grasp_phase_planner = getattr(self, "_grasp_phase_planner", None)
        if grasp_phase_planner is not None:
            grasp_phase_planner.invalidate_tool_pose_criteria()
        self.ik_solver.update_tool_pose_criteria(tool_pose_criteria)
        self.trajopt_solver.update_tool_pose_criteria(tool_pose_criteria)

    def update_tool_relative_transform_criteria(
        self,
        criteria: Dict[str, ToolRelativeTransformCriteria],
    ) -> None:
        """Update fixed-reference transform criteria in IK and TrajOpt rollouts."""
        self.ik_solver.update_tool_relative_transform_criteria(criteria)
        self.trajopt_solver.update_tool_relative_transform_criteria(criteria)

    def update_tool_velocity_limit_criteria(
        self,
        criteria: Dict[str, ToolVelocityLimitCriteria],
    ) -> None:
        """Update tool-speed limits in time-parameterized TrajOpt rollouts."""
        self.trajopt_solver.update_tool_velocity_limit_criteria(criteria)

    def update_link_inertial(
        self,
        link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ) -> None:
        """Update one link's inertial properties in IK, RRT, and TrajOpt."""
        self.ik_solver.update_link_inertial(link_name, mass, com, inertia)
        if self.graph_planner is not None and is_rrt_graph_planner_algorithm(
            self.config.graph_planner_algorithm
        ):
            self.graph_planner.update_link_inertial(link_name, mass, com, inertia)
        if self.constrained_rrt_planner is not None:
            self.constrained_rrt_planner.update_link_inertial(link_name, mass, com, inertia)
        self.trajopt_solver.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(
        self,
        link_properties: dict[str, dict[str, Union[float, torch.Tensor]]],
    ) -> None:
        """Update multiple links' inertial properties in IK, RRT, and TrajOpt."""
        self.ik_solver.update_links_inertial(link_properties)
        if self.graph_planner is not None and is_rrt_graph_planner_algorithm(
            self.config.graph_planner_algorithm
        ):
            self.graph_planner.update_links_inertial(link_properties)
        if self.constrained_rrt_planner is not None:
            self.constrained_rrt_planner.update_links_inertial(link_properties)
        self.trajopt_solver.update_links_inertial(link_properties)
