# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch motion planner for solving multiple planning problems in parallel.

Unlike :class:`MotionPlanner` (which solves one problem with retry logic and
graph-planner seeding), :class:`BatchMotionPlanner` solves ``batch_size``
independent problems in a single IK + TrajOpt pass with no retries.

When ``multi_env=False`` (shared collision world), the PRM graph planner is
available for trajectory seeding.  When ``multi_env=True`` (per-problem
collision worlds), graph seeding is skipped.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.collision import create_collision_checker
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.geom.types import SceneCfg
from curobo._src.motion.motion_planner import _axis_string_to_vector
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.motion.motion_planner_result import GraspPlanResult
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise


class BatchMotionPlanner:
    """Batch motion planner: solves ``batch_size`` problems in parallel.

    Single IK → (optional graph seed) → TrajOpt pass with no retries.
    When ``multi_env=False``, the PRM graph planner provides trajectory
    seeds.  When ``multi_env=True``, graph seeding is unavailable.

    Args:
        config: Planner configuration.  ``max_batch_size`` and ``multi_env``
            are read from ``config.ik_solver_config``.
    """

    def __init__(self, config: MotionPlannerCfg):
        self.config = config
        self.device_cfg = config.device_cfg
        self.scene_collision_checker = None
        self._initialize_components()

    def _initialize_components(self):
        if self.config.scene_collision_cfg is not None:
            self.scene_collision_checker = create_collision_checker(
                self.config.scene_collision_cfg
            )

        self.ik_solver = IKSolver(self.config.ik_solver_config, self.scene_collision_checker)
        self.trajopt_solver = TrajOptSolver(
            self.config.trajopt_solver_config, self.scene_collision_checker
        )

        self.graph_planner = None
        if (
            not self.config.ik_solver_config.multi_env
            and self.config.graph_planner_config is not None
        ):
            from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner

            self.graph_planner = PRMGraphPlanner(
                self.config.graph_planner_config, self.scene_collision_checker
            )

    # -- Lifecycle --

    def destroy(self):
        """Release all CUDA graph resources.

        Call before dropping the last reference to avoid ``cudaMallocAsync``
        warnings about captured allocations freed outside graph replay.
        """
        self.ik_solver.destroy()
        self.trajopt_solver.destroy()
        if self.graph_planner is not None and hasattr(self.graph_planner, "reset_cuda_graph"):
            self.graph_planner.reset_cuda_graph()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()
        return False

    @property
    def attachment_manager(self):
        """Attachment manager for attaching/detaching obstacles to robot links."""
        return self.trajopt_solver.attachment_manager

    @property
    def batch_size(self) -> int:
        return self.config.ik_solver_config.max_batch_size

    def warmup(
        self, enable_graph: bool = True, num_warmup_iterations: int = 5,
    ) -> bool:
        """JIT-warmup the solvers and (optionally) the graph planner.

        Args:
            enable_graph: Warmup graph planner if available (multi_env=False).
            num_warmup_iterations: Number of dummy solves.
        """
        og_exit_early = self.ik_solver.config.exit_early
        self.ik_solver.config.exit_early = False

        batch_size = self.batch_size
        for _ in range(num_warmup_iterations):
            current_state = self.default_joint_state.clone().position.unsqueeze(0)
            current_state = current_state.repeat(batch_size, 1)
            current_state = JointState.from_position(
                current_state, joint_names=self.joint_names,
            )

            goal_state = current_state.clone()
            goal_state.position[..., 0] += 0.2

            self.plan_cspace(goal_state, current_state)
            self.reset_seed()

        if enable_graph and self.graph_planner is not None:
            self.graph_planner.warmup(num_warmup_iterations=num_warmup_iterations)

        self.ik_solver.config.exit_early = og_exit_early
        return True

    def plan_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 1,
        success_ratio: float = 1.0,
        enable_graph_attempt: int = 0,
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
            )
            total_time += ik_result.total_time

            if torch.count_nonzero(ik_result.success) == 0:
                continue

            seed_traj = None
            if (
                self.graph_planner is not None
                and attempt >= enable_graph_attempt
            ):
                seed_traj = self._get_graph_seed_trajectories(
                    current_state, ik_result.solution,
                )

            trajopt_result = self.trajopt_solver.solve_pose(
                goal_tool_poses,
                current_state,
                seed_config=ik_result.solution,
                seed_traj=seed_traj,
                use_implicit_goal=use_implicit_goal,
            )
            total_time += trajopt_result.total_time

            if best_result is None:
                best_result = trajopt_result
                solved = trajopt_result.success.any(dim=-1)
            else:
                newly_solved = trajopt_result.success.any(dim=-1) & ~solved
                if newly_solved.any():
                    best_result.copy_at_batch_indices(trajopt_result, newly_solved)
                    solved = solved | newly_solved

            if solved.float().mean() >= success_ratio:
                break

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
            seed_traj = None
            if (
                self.graph_planner is not None
                and attempt >= enable_graph_attempt
            ):
                goal_configs = goal_states.position.view(batch_size, 1, dof)
                goal_configs = goal_configs.repeat(1, num_seeds, 1)
                seed_traj = self._get_graph_seed_trajectories(
                    current_state, goal_configs,
                )

            trajopt_result = self.trajopt_solver.solve_cspace(
                goal_states,
                current_state,
                seed_traj=seed_traj,
            )
            total_time += trajopt_result.total_time

            if best_result is None:
                best_result = trajopt_result
                solved = trajopt_result.success.any(dim=-1)
            else:
                newly_solved = trajopt_result.success.any(dim=-1) & ~solved
                if newly_solved.any():
                    best_result.copy_at_batch_indices(trajopt_result, newly_solved)
                    solved = solved | newly_solved

            if solved.float().mean() >= success_ratio:
                break

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
        disable_collision_links: List[str] = None,
    ) -> GraspPlanResult:
        """Plan grasp motions for a batch: goalset -> approach -> grasp -> lift.

        All B problems are planned at every stage (CUDA graph stability).
        Problems that fail a stage get their current state substituted as
        the goal for subsequent stages so the optimizer doesn't diverge.
        """
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
        self.disable_link_collision(disable_collision_links)
        goalset_result = self.plan_pose(grasp_poses, current_state)
        self.enable_link_collision(disable_collision_links)

        if goalset_result is None:
            result.status = "Goalset planning returned None."
            return result

        goalset_ok = goalset_result.success.any(dim=-1)
        result.goalset_result = goalset_result

        if not goalset_ok.any():
            result.status = "No grasp in goal set was reachable."
            return result

        result.goalset_index = goalset_result.goalset_index.clone()

        # Extract per-problem grasp pose using each problem's goalset_index
        selected_grasp = self._extract_per_problem_grasp(
            grasp_poses, goalset_result.goalset_index, goalset_ok, device,
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
            approach_poses_dict, ordered_tool_frames=tool_frames, num_goalset=1,
        )
        approach_result = self.plan_pose(approach_tool_pose, current_state)
        result.approach_result = approach_result

        if approach_result is None:
            result.status = "Planning to approach pose failed."
            return result

        approach_ok = goalset_ok & approach_result.success.any(dim=-1)
        result.approach_success = approach_ok.clone()
        result.approach_trajectory = approach_result.js_solution
        result.approach_trajectory_dt = approach_result.js_solution.dt
        result.approach_interpolated_trajectory = approach_result.interpolated_trajectory
        result.approach_interpolated_last_tstep = approach_result.interpolated_last_tstep

        if not plan_approach_to_grasp:
            result.success = approach_ok.clone()
            result.status = "Planning to approach pose completed."
            return result

        # -- Stage 3: Linear motion approach -> grasp --
        approach_end = get_joint_state_at_horizon_index(
            approach_result.js_solution, -1
        ).squeeze(1)
        approach_end = self.kinematics.get_active_js(approach_end)

        grasp_tool_pose = GoalToolPose.from_poses(
            selected_grasp, ordered_tool_frames=tool_frames, num_goalset=1,
        )
        # For failed problems, substitute current state as goal
        self._substitute_fallback_goal(grasp_tool_pose, approach_end, ~approach_ok)

        linear_motion = ToolPoseCriteria.linear_motion(
            axis=grasp_approach_axis,
            non_terminal_scale=1.0,
            project_distance_to_goal=grasp_approach_in_tool_frame,
        )
        self.update_tool_pose_criteria({k: linear_motion for k in tool_frames})
        self.disable_link_collision(disable_collision_links)
        grasp_result = self.plan_pose(grasp_tool_pose, approach_end)
        self.enable_link_collision(disable_collision_links)

        standard_criteria = {k: ToolPoseCriteria() for k in tool_frames}
        self.update_tool_pose_criteria(standard_criteria)

        if grasp_result is None:
            result.status = "Planning to grasp pose failed."
            return result

        grasp_ok = approach_ok & grasp_result.success.any(dim=-1)
        result.grasp_success = grasp_ok.clone()
        result.grasp_trajectory = grasp_result.js_solution
        result.grasp_trajectory_dt = grasp_result.js_solution.dt
        result.grasp_interpolated_trajectory = grasp_result.interpolated_trajectory
        result.grasp_interpolated_last_tstep = grasp_result.interpolated_last_tstep

        if not plan_grasp_to_lift:
            result.success = grasp_ok.clone()
            result.status = "Planning to grasp pose completed."
            return result

        # -- Stage 4: Lift --
        lift_start = get_joint_state_at_horizon_index(
            grasp_result.js_solution, -1
        ).squeeze(1)
        lift_start = self.kinematics.get_active_js(lift_start)

        lift_offset = Pose.from_list(
            [grasp_lift_offset * v for v in _axis_string_to_vector(grasp_lift_axis)]
            + [1, 0, 0, 0]
        )
        lift_poses_dict = {}
        for frame in tool_frames:
            grasp_pose = selected_grasp[frame]
            if grasp_lift_in_tool_frame:
                lift_poses_dict[frame] = grasp_pose.multiply(lift_offset)
            else:
                lift_poses_dict[frame] = lift_offset.multiply(grasp_pose)

        lift_tool_pose = GoalToolPose.from_poses(
            lift_poses_dict, ordered_tool_frames=tool_frames, num_goalset=1,
        )
        self._substitute_fallback_goal(lift_tool_pose, lift_start, ~grasp_ok)

        lift_linear = ToolPoseCriteria.linear_motion(
            axis=grasp_lift_axis,
            non_terminal_scale=1.0,
            project_distance_to_goal=grasp_lift_in_tool_frame,
        )
        self.update_tool_pose_criteria({k: lift_linear for k in tool_frames})
        self.disable_link_collision(disable_collision_links)
        lift_result = self.plan_pose(lift_tool_pose, lift_start)
        self.enable_link_collision(disable_collision_links)
        self.update_tool_pose_criteria(standard_criteria)

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
    ) -> Optional[torch.Tensor]:
        """Query graph planner for trajectory seeds across all batch*seed pairs.

        Flattens ``(B, S)`` into ``(B*S,)`` for the graph planner, then
        reshapes back to ``(B, S, horizon, dof)`` for TrajOpt.  Failed
        graph paths keep zero-filled waypoints; the optimizer improves them.

        Returns None if no graph paths were found.
        """
        from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType

        batch_size = self.batch_size
        num_seeds = seed_config.shape[1] if seed_config.ndim == 3 else seed_config.shape[0]
        dof = self.trajopt_solver.action_dim

        graph_starts = current_state.position.view(batch_size, 1, dof)
        graph_starts = graph_starts.repeat(1, num_seeds, 1).reshape(-1, dof)
        graph_goals = seed_config.view(-1, dof)

        graph_result = self.graph_planner.find_path(
            graph_starts.clone(),
            graph_goals.clone(),
            interpolate_waypoints=True,
            interpolation_steps=self.trajopt_solver.action_horizon,
            interpolation_type=TrajInterpolationType.LINEAR,
            validate_interpolated_trajectory=False,
        )

        if not graph_result.success.any():
            return None

        horizon = graph_result.interpolated_waypoints.shape[1]
        return graph_result.interpolated_waypoints.view(
            batch_size, num_seeds, horizon, dof
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

    def update_world(self, scene_cfg: SceneCfg):
        self.scene_collision_checker.load_collision_model(scene_cfg)
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()

    def clear_scene_cache(self):
        self.scene_collision_checker.clear_cache()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()

    def reset_seed(self):
        self.ik_solver.reset_seed()
        self.trajopt_solver.reset_seed()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
            self.graph_planner.reset_seed()

    def enable_link_collision(self, enable_collision_links: List[str]):
        for link_name in enable_collision_links:
            self.kinematics.config.kinematics_config.enable_link_spheres(link_name)

    def disable_link_collision(self, disable_collision_links: List[str]):
        for link_name in disable_collision_links:
            self.kinematics.config.kinematics_config.disable_link_spheres(link_name)

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, "ToolPoseCriteria"]):
        self.ik_solver.update_tool_pose_criteria(tool_pose_criteria)
        self.trajopt_solver.update_tool_pose_criteria(tool_pose_criteria)

    def update_link_inertial(
        self,
        link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ) -> None:
        self.ik_solver.update_link_inertial(link_name, mass, com, inertia)
        self.trajopt_solver.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(
        self,
        link_properties: dict[str, dict[str, Union[float, torch.Tensor]]],
    ) -> None:
        self.ik_solver.update_links_inertial(link_properties)
        self.trajopt_solver.update_links_inertial(link_properties)
