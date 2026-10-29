# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared constrained-phase planning for single and batched grasps."""

from __future__ import annotations

# Standard Library
from typing import Callable, Dict, List, Optional

# Third Party
import torch

# CuRobo
from curobo._src.cost.tool_pose_criteria import PoseErrorFrame, ToolPoseCriteria
from curobo._src.graph_planner.result import GraphPlannerResult
from curobo._src.graph_planner.rrt.planner_bi_rrt import BiRRTPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.solver.solver_base_result import BaseSolverResult
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.trajectory.trajectory_interpolation import TrajInterpolationType
from curobo._src.trajectory.trajectory_seed_fitter import BSplineTimeOptimalSeedFitter
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.cuda_graph_util import GraphExecutor
from curobo._src.util.logging import log_and_raise


def axis_string_to_vector(axis: str) -> List[float]:
    """Convert an axis name to an xyz unit vector."""
    if axis == "x":
        return [1.0, 0.0, 0.0]
    if axis == "y":
        return [0.0, 1.0, 0.0]
    if axis == "z":
        return [0.0, 0.0, 1.0]
    log_and_raise(f"Invalid axis: {axis}, must be 'x', 'y', or 'z'")


def get_result_goal_configuration(
    result: BaseSolverResult,
    kinematics: Kinematics,
) -> torch.Tensor:
    """Extract selected active joints shaped as ``[batch, dof]`` from a solver result."""
    if isinstance(result, TrajOptSolverResult):
        if result.js_solution is None:
            log_and_raise("TrajOpt goalset result has no joint trajectory")
        endpoint = JointState.from_position(
            result.js_solution.position[:, 0, -1, :],
            joint_names=result.js_solution.joint_names,
        )
        return kinematics.get_active_js(endpoint).position.clone()
    if result.solution is None:
        log_and_raise("IK goalset result has no joint solution")
    return result.solution[:, 0, :].clone()


class GraspPhasePlanner:
    """Plan and cache constrained grasp phases for one fixed problem batch."""

    def __init__(
        self,
        config: MotionPlannerCfg,
        batch_size: int,
        kinematics: Kinematics,
        ik_solver: IKSolver,
        trajopt_solver: TrajOptSolver,
        constrained_rrt_planner: Optional[BiRRTPlanner],
        constrained_rrt_seed_fitter: Optional[BSplineTimeOptimalSeedFitter],
        plan_pose: Callable[..., Optional[TrajOptSolverResult]],
        update_tool_pose_criteria: Callable[[Dict[str, ToolPoseCriteria]], None],
        enable_link_collision: Callable[[List[str]], None],
        disable_link_collision: Callable[[List[str]], None],
        record_graph_result: Optional[Callable[[GraphPlannerResult], None]] = None,
    ) -> None:
        """Initialize shared solvers, callbacks, and fixed-address update caches."""
        self.config = config
        self.batch_size = batch_size
        self.device_cfg: DeviceCfg = config.device_cfg
        self.kinematics = kinematics
        self.ik_solver = ik_solver
        self.trajopt_solver = trajopt_solver
        self.constrained_rrt_planner = constrained_rrt_planner
        self.constrained_rrt_seed_fitter = constrained_rrt_seed_fitter
        self._plan_pose = plan_pose
        self._update_tool_pose_criteria = update_tool_pose_criteria
        self._enable_link_collision = enable_link_collision
        self._disable_link_collision = disable_link_collision
        self._record_graph_result = record_graph_result
        self._update_executors: Dict[tuple, GraphExecutor] = {}
        self._active_tool_pose_key: Optional[tuple] = None
        self._rrt_goal_executor: Optional[GraphExecutor] = None
        self._rrt_criteria_executors: Dict[tuple, GraphExecutor] = {}
        self._active_rrt_criteria_key: Optional[tuple] = None

    def destroy(self) -> None:
        """Release CUDA graphs owned by cached grasp-state updates."""
        for executor in self._update_executors.values():
            executor.reset()
        self._update_executors.clear()
        self._active_tool_pose_key = None
        if self._rrt_goal_executor is not None:
            self._rrt_goal_executor.reset()
            self._rrt_goal_executor = None
        for executor in self._rrt_criteria_executors.values():
            executor.reset()
        self._rrt_criteria_executors.clear()
        self._active_rrt_criteria_key = None

    def invalidate_tool_pose_criteria(self) -> None:
        """Forget the active preset after an external criteria update."""
        self._active_tool_pose_key = None

    def _create_update_executor(
        self,
        key: tuple,
        update: Callable[[], None],
    ) -> GraphExecutor:
        """Capture and cache one fixed-address grasp-state update."""
        executor = GraphExecutor(
            capture_fn=update,
            device=self.device_cfg.device,
            use_cuda_graph=self.config.trajopt_solver_config.use_cuda_graph,
            clone_outputs=False,
            clone_inputs=False,
        )
        executor.warmup()
        self._update_executors[key] = executor
        return executor

    def activate_tool_pose_criteria(
        self,
        tool_frames: List[str],
        axis: Optional[str],
        in_tool_frame: bool,
    ) -> None:
        """Activate one cached standard or linear pose-criteria preset."""
        key = ("tool_pose", tuple(tool_frames), axis, in_tool_frame)
        if key == self._active_tool_pose_key:
            return
        executor = self._update_executors.get(key)
        if executor is None:
            if axis is None:
                criteria = {
                    frame: ToolPoseCriteria(device_cfg=self.device_cfg) for frame in tool_frames
                }
            else:
                linear_motion = self._make_linear_criteria(axis, in_tool_frame)
                criteria = {frame: linear_motion for frame in tool_frames}
            executor = self._create_update_executor(
                key,
                lambda: self._update_tool_pose_criteria(criteria),
            )
        executor.replay(clone_outputs=False)
        self._active_tool_pose_key = key

    def activate_collision(self, links: List[str], *, enable: bool) -> None:
        """Activate one cached grasp-contact collision state."""
        if not links:
            return
        stable_links = tuple(links)
        key = ("collision", stable_links, enable)
        executor = self._update_executors.get(key)
        if executor is None:
            if self.device_cfg.device.type == "cuda":
                kinematics_config = self.kinematics.config.kinematics_config
                sphere_indices = torch.cat(
                    [
                        kinematics_config.get_sphere_index_from_link_name(link)
                        for link in stable_links
                    ]
                )
                target_radii = kinematics_config.link_spheres[..., 3]
                if enable:
                    source_radii = kinematics_config.reference_link_spheres[..., 3].index_select(
                        1,
                        sphere_indices,
                    )
                else:
                    source_radii = torch.full(
                        (target_radii.shape[0], sphere_indices.shape[0]),
                        -100.0,
                        device=target_radii.device,
                        dtype=target_radii.dtype,
                    )

                def update() -> None:
                    target_radii.index_copy_(1, sphere_indices, source_radii)

            elif enable:

                def update() -> None:
                    self._enable_link_collision(list(stable_links))

            else:

                def update() -> None:
                    self._disable_link_collision(list(stable_links))

            executor = self._create_update_executor(key, update)
        executor.replay(clone_outputs=False)

    def install_rrt_constraint(self, goal_tool_poses: GoalToolPose) -> None:
        """Install fixed projected-RRT buffers and capture reusable goal updates."""
        if self.constrained_rrt_planner is None or self._rrt_goal_executor is not None:
            return
        disabled_criteria = {
            frame: ToolPoseCriteria.disabled(device_cfg=self.device_cfg)
            for frame in goal_tool_poses.tool_frames
        }
        self.constrained_rrt_planner.update_tool_pose_constraint(
            goal_tool_poses,
            disabled_criteria,
        )
        self._rrt_goal_executor = GraphExecutor(
            capture_fn=self.constrained_rrt_planner.update_tool_pose_constraint_goal,
            device=self.device_cfg.device,
            use_cuda_graph=self.config.trajopt_solver_config.use_cuda_graph,
            clone_outputs=False,
        )
        self._rrt_goal_executor.warmup(goal_tool_poses)

    def activate_rrt_constraint(
        self,
        goal_tool_poses: GoalToolPose,
        axis: str,
        in_tool_frame: bool,
    ) -> None:
        """Update RRT targets and activate one cached linear constraint preset."""
        self.install_rrt_constraint(goal_tool_poses)
        if self.constrained_rrt_planner is None or self._rrt_goal_executor is None:
            log_and_raise("constrained RRT is not configured")
        key = (tuple(goal_tool_poses.tool_frames), axis, in_tool_frame)
        if key != self._active_rrt_criteria_key:
            executor = self._rrt_criteria_executors.get(key)
            if executor is None:
                linear_motion = self._make_linear_criteria(axis, in_tool_frame)
                criteria = {frame: linear_motion for frame in goal_tool_poses.tool_frames}
                executor = GraphExecutor(
                    capture_fn=lambda: (
                        self.constrained_rrt_planner.update_tool_pose_constraint_criteria(criteria)
                    ),
                    device=self.device_cfg.device,
                    use_cuda_graph=self.config.trajopt_solver_config.use_cuda_graph,
                    clone_outputs=False,
                    clone_inputs=False,
                )
                executor.warmup()
                self._rrt_criteria_executors[key] = executor
            executor.replay(clone_outputs=False)
            self._active_rrt_criteria_key = key
        self._rrt_goal_executor(goal_tool_poses, clone_outputs=False)

    def _make_linear_criteria(
        self,
        axis: str,
        in_tool_frame: bool,
    ) -> ToolPoseCriteria:
        """Create one linear constraint with the configured pose tolerances."""
        linear_motion = ToolPoseCriteria.linear_motion(
            direction=axis_string_to_vector(axis),
            non_terminal_scale=1.0,
            error_frame=PoseErrorFrame.GOAL if in_tool_frame else PoseErrorFrame.WORLD,
            device_cfg=self.device_cfg,
        )
        linear_motion.path_tolerance.copy_(
            self.device_cfg.to_device(
                [
                    self.config.ik_solver_config.position_tolerance,
                    self.config.ik_solver_config.orientation_tolerance,
                ]
            )
        )
        return linear_motion

    def _plan_rrt_phase(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        goal_configuration: torch.Tensor,
        goal_valid: torch.Tensor,
        axis: str,
        in_tool_frame: bool,
    ) -> Optional[TrajOptSolverResult]:
        """Plan, B-spline fit, and validate one projected-RRT phase."""
        if self.constrained_rrt_planner is None or self.constrained_rrt_seed_fitter is None:
            return None
        self.activate_rrt_constraint(goal_tool_poses, axis, in_tool_frame)
        lane_env_idx = (
            torch.arange(
                self.batch_size,
                device=goal_configuration.device,
                dtype=torch.int32,
            )
            if self.config.ik_solver_config.multi_env
            else None
        )
        graph_result = self.constrained_rrt_planner.find_path(
            current_state.position.reshape(self.batch_size, -1),
            goal_configuration.reshape(self.batch_size, -1),
            query_mask=goal_valid.reshape(-1),
            group_count=self.batch_size,
            env_query_idx=lane_env_idx,
            interpolate_waypoints=True,
            interpolation_steps=self.constrained_rrt_seed_fitter.config.rollout_horizon,
            interpolation_type=TrajInterpolationType.LINEAR,
            validate_interpolated_trajectory=True,
        )
        if self._record_graph_result is not None:
            self._record_graph_result(graph_result)
        graph_success = graph_result.success.reshape(-1) & goal_valid.reshape(-1)
        if not graph_success.any() or graph_result.interpolated_waypoints is None:
            return None

        fitted_knots = self.constrained_rrt_seed_fitter.fit(
            graph_result.interpolated_waypoints,
            current_state,
            goal_configuration,
        )
        validation_seeds = fitted_knots.unsqueeze(1).repeat(
            1,
            self.trajopt_solver.config.num_seeds,
            1,
            1,
        )
        seed_valid = graph_success.view(-1, 1).expand(
            -1,
            self.trajopt_solver.config.num_seeds,
        )
        self.activate_tool_pose_criteria(
            goal_tool_poses.tool_frames,
            axis=axis,
            in_tool_frame=in_tool_frame,
        )
        try:
            fitted_result = self.trajopt_solver.evaluate_pose_seed(
                goal_tool_poses,
                current_state,
                validation_seeds,
                use_implicit_goal=True,
                trajectory_source="constrained_rrt_fit",
                seed_valid_mask=seed_valid,
            )
        finally:
            self.activate_tool_pose_criteria(
                goal_tool_poses.tool_frames,
                axis=None,
                in_tool_frame=True,
            )
        if not fitted_result.success.any():
            return None
        fitted_result.debug_info["constrained_rrt"] = graph_result.debug_info
        return fitted_result

    def plan_linear_phase(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        axis: str,
        in_tool_frame: bool,
        use_rrt: bool,
        finetune_attempts: int,
        goal_configuration: Optional[torch.Tensor] = None,
        goal_valid: Optional[torch.Tensor] = None,
    ) -> Optional[TrajOptSolverResult]:
        """Plan one linear phase with optional RRT and per-lane TrajOpt fallback."""
        rrt_result = None
        if use_rrt and self.constrained_rrt_planner is not None:
            if goal_configuration is None or goal_valid is None:
                ik_result = self.ik_solver.solve_goalset(
                    goal_tool_poses,
                    return_seeds=1,
                    current_state=current_state,
                )
                goal_configuration = ik_result.solution[:, 0, :].clone()
                goal_valid = ik_result.success.reshape(self.batch_size, -1).any(dim=-1)
            if goal_valid.any():
                rrt_result = self._plan_rrt_phase(
                    goal_tool_poses,
                    current_state,
                    goal_configuration,
                    goal_valid,
                    axis,
                    in_tool_frame,
                )
                if rrt_result is not None and rrt_result.success.any(dim=-1).all():
                    return rrt_result

        self.activate_tool_pose_criteria(
            goal_tool_poses.tool_frames,
            axis=axis,
            in_tool_frame=in_tool_frame,
        )
        try:
            fallback = self._plan_pose(
                goal_tool_poses,
                current_state,
                finetune_attempts=finetune_attempts,
            )
        finally:
            self.activate_tool_pose_criteria(
                goal_tool_poses.tool_frames,
                axis=None,
                in_tool_frame=True,
            )
        if rrt_result is None or fallback is None:
            return fallback if rrt_result is None else rrt_result

        rrt_result = rrt_result.clone()
        rrt_success = rrt_result.success.any(dim=-1)
        fallback_success = fallback.success.any(dim=-1)
        fallback_mask = ~rrt_success & fallback_success
        if fallback_mask.any():
            rrt_result.copy_at_batch_indices(fallback, fallback_mask)
            rrt_result.trajectory_source = "mixed"
            rrt_result.debug_info["trajectory_source"] = "mixed"
        return rrt_result
