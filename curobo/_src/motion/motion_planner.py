# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-problem motion planner with retry logic and graph-planner seeding."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.collision import create_collision_checker
from curobo._src.geom.types import SceneCfg
from curobo._src.graph_planner.graph_planner_prm import (
    PRMGraphPlanner,
    TrajInterpolationType,
)
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.motion.motion_planner_result import GraspPlanResult
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise


def _axis_string_to_vector(axis: str) -> List[float]:
    if axis == "x":
        return [1.0, 0.0, 0.0]
    elif axis == "y":
        return [0.0, 1.0, 0.0]
    elif axis == "z":
        return [0.0, 0.0, 1.0]
    else:
        log_and_raise(f"Invalid axis: {axis}, must be 'x', 'y', or 'z'")


class MotionPlanner:
    """Single-problem motion planner with retry logic and graph-planner seeding.

    Solves one planning problem at a time (``batch_size=1``).  For batched
    planning see :class:`BatchMotionPlanner`.
    """

    def __init__(self, config: MotionPlannerCfg):
        self.config = config
        self.device_cfg = config.device_cfg
        self.scene_collision_checker = None
        self._destroyed = False
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
        if self.config.graph_planner_config is not None:
            self.graph_planner = PRMGraphPlanner(
                self.config.graph_planner_config, self.scene_collision_checker
            )

    # -- Lifecycle --

    def destroy(self):
        """Release all CUDA graph resources.

        Call before dropping the last reference to avoid ``cudaMallocAsync``
        warnings about captured allocations freed outside graph replay.

        Does not synchronize, the ``gc.collect()`` + ``synchronize()`` fence
        in :meth:`GraphExecutor._initialize_cuda_graph` ensures stale
        ``cuGraphExecDestroy`` calls complete before any new capture begins.
        """
        if self._destroyed:
            return
        self._destroyed = True
        self.ik_solver.destroy()
        self.trajopt_solver.destroy()
        if self.graph_planner is not None and hasattr(self.graph_planner, "reset_cuda_graph"):
            self.graph_planner.reset_cuda_graph()

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()
        return False

    @property
    def attachment_manager(self):
        """Attachment manager for attaching/detaching obstacles to robot links."""
        return self.trajopt_solver.attachment_manager

    # -- Properties --

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

    # -- Warmup --

    def warmup(
        self,
        enable_graph: bool = True,
        warmup_joint_index: int = 0,
        warmup_joint_delta: float = 0.2,
        num_warmup_iterations: int = 10,
    ) -> bool:
        num_goalset = self.config.ik_solver_config.max_goalset
        og_exit_early = self.ik_solver.config.exit_early
        self.ik_solver.config.exit_early = False

        for _ in range(num_warmup_iterations):
            current_state = self.default_joint_state.clone().unsqueeze(0)

            if num_goalset > 1:
                goalset_poses = self._make_warmup_goalset(
                    current_state, num_goalset, warmup_joint_index, warmup_joint_delta,
                )
                goalset_goal = GoalToolPose.from_poses(
                    goalset_poses, num_goalset=num_goalset,
                )
                result = self.plan_pose(goalset_goal, current_state, max_attempts=1)
                if result is None:
                    goal_tool_poses = GoalToolPose.from_poses(
                        goalset_poses, num_goalset=num_goalset,
                    )
                    seed_config = current_state.position.view(1, 1, -1).repeat(
                        1, self.trajopt_solver.config.num_seeds, 1
                    )
                    self.trajopt_solver.solve_pose(
                        goal_tool_poses, current_state,
                        seed_config=seed_config, use_implicit_goal=True,
                    )
            else:
                goal_state = current_state.clone()
                goal_state.position[..., warmup_joint_index] += warmup_joint_delta
                goal_tool_poses = self.compute_kinematics(goal_state).tool_poses.as_goal()
                self.plan_pose(goal_tool_poses, current_state, max_attempts=1)

            self.reset_seed()

        if enable_graph and self.graph_planner is not None:
            self.graph_planner.warmup(num_warmup_iterations=num_warmup_iterations)

        self.ik_solver.config.exit_early = og_exit_early
        return True

    def _make_warmup_goalset(
        self, current_state: JointState, num_goalset: int,
        joint_index: int, joint_delta: float,
    ) -> Dict[str, Pose]:
        positions_per_frame: Dict[str, list] = {}
        quaternions_per_frame: Dict[str, list] = {}
        for g in range(num_goalset):
            goal_state = current_state.clone()
            delta = joint_delta * (g + 1) / num_goalset
            goal_state.position[..., joint_index] += delta
            kin = self.compute_kinematics(goal_state)
            for frame, pose in kin.tool_poses.to_dict().items():
                positions_per_frame.setdefault(frame, []).append(pose.position)
                quaternions_per_frame.setdefault(frame, []).append(pose.quaternion)

        return {
            frame: Pose(
                position=torch.cat(positions_per_frame[frame], dim=0),
                quaternion=torch.cat(quaternions_per_frame[frame], dim=0),
            )
            for frame in positions_per_frame
        }

    # -- Planning --

    def plan_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
    ) -> Optional[TrajOptSolverResult]:
        """Plan a trajectory to reach target tool poses.

        Goalset is auto-detected from ``goal_tool_poses.num_goalset``.  When
        ``num_goalset > 1`` the planner uses a simpler IK+TrajOpt loop without
        graph seeding.
        """
        if current_state.ndim > 2:
            log_and_raise(f"current_state must be a 2D tensor, got shape: {current_state.shape}")

        if goal_tool_poses.num_goalset > 1:
            return self._plan_pose_goalset(
                goal_tool_poses, current_state, use_implicit_goal, max_attempts,
            )

        return self._plan_pose_single(
            goal_tool_poses, current_state, max_attempts, enable_graph_attempt,
        )

    def _plan_pose_single(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        max_attempts: int,
        enable_graph_attempt: int,
    ) -> Optional[TrajOptSolverResult]:
        """Single-goal planning with retry, seed repair, and graph seeding."""
        trajopt_result = None
        total_time = 0.0
        solve_time = 0.0
        og_current_state = current_state.clone()
        num_seeds = self.trajopt_solver.config.num_seeds

        for current_attempt in range(max_attempts):
            current_state = og_current_state.clone()
            ik_result = self.ik_solver.solve_pose(
                goal_tool_poses,
                return_seeds=num_seeds,
                current_state=current_state,
            )
            total_time += ik_result.total_time
            solve_time += ik_result.solve_time

            success_count = torch.count_nonzero(ik_result.success)
            if success_count == 0:
                continue

            seed_config = ik_result.solution
            if success_count < num_seeds:
                good_solution = seed_config[ik_result.success][0:1, :].clone()
                seed_config[~ik_result.success][:, :] = good_solution

            seed_traj = None
            finetune_attempts = 1
            finetune_dt_scale = 0.55
            if current_attempt >= enable_graph_attempt and self.graph_planner is not None:
                graph_seed = self._get_graph_seed_trajectories(
                    current_state, seed_config,
                )
                if graph_seed is None:
                    continue
                seed_traj = graph_seed
                total_time += 0.0  # graph time already in graph_seed call
                finetune_attempts = 3
                finetune_dt_scale = 0.75

            trajopt_result = self.trajopt_solver.solve_pose(
                goal_tool_poses, current_state,
                seed_config=seed_config, seed_traj=seed_traj,
                use_implicit_goal=True,
                finetune_attempts=finetune_attempts,
                finetune_dt_scale=finetune_dt_scale,
            )
            total_time += trajopt_result.total_time
            solve_time += trajopt_result.solve_time
            if torch.count_nonzero(trajopt_result.success) > 0:
                break

        if trajopt_result is not None:
            trajopt_result.total_time = total_time
            trajopt_result.solve_time = solve_time
        return trajopt_result

    def _plan_pose_goalset(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 10,
    ) -> Optional[TrajOptSolverResult]:
        """Goalset planning: IK + TrajOpt, no graph seeding."""
        for _ in range(max_attempts):
            ik_result = self.ik_solver.solve_pose(
                goal_tool_poses,
                return_seeds=self.trajopt_solver.config.num_seeds,
                current_state=current_state,
            )
            if torch.count_nonzero(ik_result.success) == 0:
                return None

            trajopt_result = self.trajopt_solver.solve_pose(
                goal_tool_poses, current_state,
                seed_config=ik_result.solution,
                use_implicit_goal=use_implicit_goal,
            )
            if torch.count_nonzero(trajopt_result.success) > 0:
                break

        return trajopt_result

    def plan_cspace(
        self,
        goal_state: JointState,
        current_state: JointState,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
    ) -> Optional[TrajOptSolverResult]:
        """Plan a collision-free trajectory to a joint configuration.

        Args:
            goal_state: Target joint configuration.
            current_state: Initial joint state.
            max_attempts: Maximum planning attempts.
            enable_graph_attempt: Attempt at which to start graph seeding.

        Returns:
            TrajOptSolverResult, or None if planning failed.
        """
        if current_state.ndim > 2:
            log_and_raise(f"current_state must be a 2D tensor, got shape: {current_state.shape}")
        if goal_state.ndim > 2:
            log_and_raise(f"goal_state must be a 2D tensor, got shape: {goal_state.shape}")

        trajopt_result = None
        total_time = 0.0
        solve_time = 0.0
        og_current_state = current_state.clone()
        num_seeds = self.trajopt_solver.config.num_seeds

        for current_attempt in range(max_attempts):
            current_state = og_current_state.clone()
            seed_traj = None

            if current_attempt >= enable_graph_attempt and self.graph_planner is not None:
                goal_configs = goal_state.position.view(1, 1, -1).repeat(1, num_seeds, 1)
                graph_seed = self._get_graph_seed_trajectories(
                    current_state, goal_configs,
                )
                if graph_seed is None:
                    continue
                seed_traj = graph_seed

            trajopt_result = self.trajopt_solver.solve_cspace(
                goal_state, current_state, seed_traj=seed_traj,
            )
            total_time += trajopt_result.total_time
            solve_time += trajopt_result.solve_time
            if torch.count_nonzero(trajopt_result.success) > 0:
                break

        if trajopt_result is not None:
            trajopt_result.total_time = total_time
            trajopt_result.solve_time = solve_time
        return trajopt_result

    def _get_graph_seed_trajectories(
        self,
        current_state: JointState,
        seed_config: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Query graph planner for trajectory seeds.

        Args:
            current_state: Start state ``(1, dof)``.
            seed_config: Goal configs ``(1, num_seeds, dof)``.

        Returns:
            Seed trajectories ``(1, n_success, horizon, dof)`` or None.
        """
        dof = self.trajopt_solver.action_dim
        num_seeds = seed_config.shape[1] if seed_config.ndim == 3 else seed_config.shape[0]

        graph_starts = current_state.position.view(1, dof).repeat(num_seeds, 1)
        graph_goals = seed_config.view(num_seeds, dof)

        result = self.graph_planner.find_path(
            graph_starts.clone(), graph_goals.clone(),
            interpolate_waypoints=True,
            interpolation_steps=self.trajopt_solver.action_horizon,
            interpolation_type=TrajInterpolationType.LINEAR,
            validate_interpolated_trajectory=False,
        )
        if torch.count_nonzero(result.success) == 0:
            return None
        return result.interpolated_waypoints[result.success, :, :].unsqueeze(0)

    # -- Grasp planning --

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
        """Plan a grasp motion: goalset -> approach -> grasp -> lift.

        Args:
            grasp_poses: Candidate grasp poses with ``num_goalset`` entries per link.
                Construct via ``GoalToolPose`` with the desired goalset size.
        """
        if disable_collision_links is None:
            disable_collision_links = self.kinematics.config.kinematics_config.grasp_contact_link_names
        if disable_collision_links is None:
            disable_collision_links = []

        device = current_state.device
        result = GraspPlanResult()
        result.success = torch.tensor([False], device=device)
        result.approach_success = torch.tensor([False], device=device)
        result.grasp_success = torch.tensor([False], device=device)
        result.lift_success = torch.tensor([False], device=device)

        # Step 1: Plan to one of the grasp poses (goalset)
        self.disable_link_collision(disable_collision_links)
        goalset_result = self.plan_pose(grasp_poses, current_state)
        self.enable_link_collision(disable_collision_links)

        if goalset_result is None:
            result.status = "Goalset planning returned None."
            return result

        result.success = goalset_result.success.clone()
        result.success[:] = False
        result.goalset_result = goalset_result

        if not goalset_result.success.any():
            result.status = "No grasp in goal set was reachable."
            return result

        result.goalset_index = goalset_result.goalset_index.clone()
        goal_index = int(goalset_result.goalset_index.view(-1)[0].item())

        grasp_poses_dict = {}
        for frame in grasp_poses.tool_frames:
            li = grasp_poses.tool_frames.index(frame)
            pos = grasp_poses.position[:, 0, li, goal_index, :]
            quat = grasp_poses.quaternion[:, 0, li, goal_index, :]
            grasp_poses_dict[frame] = Pose(position=pos, quaternion=quat)

        # Step 2: Plan to approach pose (offset from grasp)
        approach_offset = Pose.from_list([
            grasp_approach_offset * v
            for v in _axis_string_to_vector(grasp_approach_axis)
        ] + [1, 0, 0, 0])

        approach_poses_dict = {}
        for frame, goal_pose in grasp_poses_dict.items():
            if grasp_approach_in_tool_frame:
                approach_poses_dict[frame] = goal_pose.multiply(approach_offset)
            else:
                approach_poses_dict[frame] = approach_offset.multiply(goal_pose)

        approach_tool_pose = GoalToolPose.from_poses(
            approach_poses_dict, ordered_tool_frames=grasp_poses.tool_frames, num_goalset=1,
        )
        approach_result = self.plan_pose(approach_tool_pose, current_state)
        result.approach_result = approach_result

        if approach_result is None or not approach_result.success.any():
            result.status = "Planning to approach pose failed."
            result.approach_success[:] = False
            result.success[:] = False
            return result

        result.approach_success = approach_result.success.clone()
        result.approach_trajectory = approach_result.js_solution
        result.approach_trajectory_dt = approach_result.js_solution.dt
        result.approach_interpolated_trajectory = approach_result.interpolated_trajectory
        result.approach_interpolated_last_tstep = approach_result.interpolated_last_tstep
        result.status = "Planning to approach pose succeeded."

        if not plan_approach_to_grasp:
            result.success[:] = True
            return result

        # Step 3: Linear motion from approach to grasp
        approach_end = get_joint_state_at_horizon_index(approach_result.js_solution, -1).squeeze(0)
        approach_end = self.kinematics.get_active_js(approach_end)

        linear_motion = ToolPoseCriteria.linear_motion(
            axis=grasp_approach_axis, non_terminal_scale=1.0,
            project_distance_to_goal=grasp_approach_in_tool_frame,
        )
        self.update_tool_pose_criteria({k: linear_motion for k in grasp_poses.tool_frames})

        grasp_tool_pose = GoalToolPose.from_poses(
            grasp_poses_dict, ordered_tool_frames=grasp_poses.tool_frames, num_goalset=1,
        )
        self.disable_link_collision(disable_collision_links)
        grasp_result = self.plan_pose(grasp_tool_pose, approach_end)
        self.enable_link_collision(disable_collision_links)

        standard_criteria = {k: ToolPoseCriteria() for k in grasp_poses.tool_frames}
        self.update_tool_pose_criteria(standard_criteria)

        if grasp_result is None or not grasp_result.success.any():
            result.status = "Planning to grasp pose failed."
            result.grasp_success[:] = False
            result.success[:] = False
            return result

        result.grasp_trajectory = grasp_result.js_solution
        result.grasp_trajectory_dt = grasp_result.js_solution.dt
        result.grasp_interpolated_trajectory = grasp_result.interpolated_trajectory
        result.grasp_interpolated_last_tstep = grasp_result.interpolated_last_tstep
        result.success = grasp_result.success.clone()
        result.grasp_success = grasp_result.success.clone()

        if not plan_grasp_to_lift:
            return result

        # Step 4: Lift motion
        lift_start = get_joint_state_at_horizon_index(grasp_result.js_solution, -1).squeeze(0)
        lift_start = self.kinematics.get_active_js(lift_start)

        lift_offset = Pose.from_list([
            grasp_lift_offset * v
            for v in _axis_string_to_vector(grasp_lift_axis)
        ] + [1, 0, 0, 0])

        lift_poses_dict = {}
        for frame, goal_pose in grasp_poses_dict.items():
            if grasp_lift_in_tool_frame:
                lift_poses_dict[frame] = goal_pose.multiply(lift_offset)
            else:
                lift_poses_dict[frame] = lift_offset.multiply(goal_pose)

        lift_tool_pose = GoalToolPose.from_poses(
            lift_poses_dict, ordered_tool_frames=grasp_poses.tool_frames, num_goalset=1,
        )
        lift_linear = ToolPoseCriteria.linear_motion(
            axis=grasp_lift_axis, non_terminal_scale=1.0,
            project_distance_to_goal=grasp_lift_in_tool_frame,
        )
        self.update_tool_pose_criteria({k: lift_linear for k in grasp_poses.tool_frames})

        self.disable_link_collision(disable_collision_links)
        lift_result = self.plan_pose(lift_tool_pose, lift_start)
        self.enable_link_collision(disable_collision_links)
        self.update_tool_pose_criteria(standard_criteria)

        if lift_result is None or not lift_result.success.any():
            result.status = "Planning to lift pose failed."
            result.success[:] = False
            result.lift_success[:] = False
            return result

        result.lift_trajectory = lift_result.js_solution
        result.lift_trajectory_dt = lift_result.js_solution.dt
        result.lift_interpolated_trajectory = lift_result.interpolated_trajectory
        result.lift_interpolated_last_tstep = lift_result.interpolated_last_tstep
        result.success = lift_result.success.clone()
        result.lift_success = lift_result.success.clone()
        result.status = "Planning to lift pose succeeded."
        return result

    # -- Scene / collision management --

    def enable_link_collision(self, enable_collision_links: List[str]):
        for link_name in enable_collision_links:
            self.kinematics.config.kinematics_config.enable_link_spheres(link_name)

    def disable_link_collision(self, disable_collision_links: List[str]):
        for link_name in disable_collision_links:
            self.kinematics.config.kinematics_config.disable_link_spheres(link_name)

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

    def update_link_inertial(
        self, link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ) -> None:
        self.ik_solver.update_link_inertial(link_name, mass, com, inertia)
        self.trajopt_solver.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(
        self, link_properties: dict[str, dict[str, Union[float, torch.Tensor]]],
    ) -> None:
        self.ik_solver.update_links_inertial(link_properties)
        self.trajopt_solver.update_links_inertial(link_properties)

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        self.ik_solver.update_tool_pose_criteria(tool_pose_criteria)
        self.trajopt_solver.update_tool_pose_criteria(tool_pose_criteria)
