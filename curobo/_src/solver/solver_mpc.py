# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MPCSolver - Model Predictive Control solver for reaching targets in Cartesian and Joint space."""

from __future__ import annotations

# Standard Library
from typing import Dict, List, Optional, Tuple

# Third Party
import torch

# CuRobo
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.collision import SceneCollision
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.solver.solver_core import SolverCore
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
from curobo._src.solver.solver_mpc_result import MPCSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.trajectory_execution_manager import TrajectoryExecutionManager


class MPCSolver:
    """Implements model predictive control for reaching targets in Cartesian and Joint space."""

    def __init__(
        self,
        config: MPCSolverCfg,
        scene_collision_checker: Optional[SceneCollision] = None,
    ) -> None:
        self.config = config
        self.core = SolverCore(config.core_cfg, scene_collision_checker)

        # Trajectory execution manager
        self.trajectory_execution_manager = TrajectoryExecutionManager(
            self.config.interpolation_steps
        )

        self._mpc_setup_complete = False
        self._mpc_initialized = False
        self._mpc_warm_start_available = False
        self._goal_tool_poses = None
        self._debug = False
        self._debug_counter = 0

        # Create IKSolver for computing IK solutions when updating goal poses
        ik_solver_cfg = IKSolverCfg.create(
            robot=config.robot_config,
            scene_model=None,
            num_seeds=1,
            position_tolerance=config.position_tolerance,
            orientation_tolerance=config.orientation_tolerance,
            use_cuda_graph=True,
            random_seed=config.random_seed,
            optimizer_collision_activation_distance=config.optimizer_collision_activation_distance,
            device_cfg=config.device_cfg,
            self_collision_check=config.self_collision_check,
            store_debug=config.store_debug,
            optimizer_configs=["ik/lbfgs_retarget_ik.yml"],
            metrics_rollout="metrics_base.yml",
            transition_model="ik/transition_ik.yml",
        )
        self.ik_solver = IKSolver(ik_solver_cfg, self.core.scene_collision_checker)
        self.enable_tool_pose_tracking()
        self.disable_joint_position_tracking()

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
    def additional_metrics_rollouts(self):
        return self.core.additional_metrics_rollouts

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

    def compute_kinematics(self, state):
        """Run forward kinematics and return tool poses and collision spheres."""
        return self.core.compute_kinematics(state)

    def get_active_js(self, full_js):
        return self.core.get_active_js(full_js)

    def get_full_js(self, active_js):
        return self.core.get_full_js(active_js)

    def sample_configs(self, num_samples, rejection_ratio=10):
        return self.core.sample_configs(
            num_samples, rejection_ratio, self.config.optimizer_collision_activation_distance
        )

    def update_pose_cost_metric(self, pose_cost_metric):
        return self.core.update_pose_cost_metric(pose_cost_metric)

    def update_link_inertial(self, link_name, mass=None, com=None, inertia=None):
        return self.core.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(self, link_properties):
        return self.core.update_links_inertial(link_properties)

    def debug_dump(self, file_path):
        return self.core.debug_dump(file_path)

    def prepare_action_seeds(self, batch_size, num_seeds, seed_config=None, current_state=None, seed_traj=None):
        return self.core.prepare_action_seeds(batch_size, num_seeds, seed_config, current_state, seed_traj)

    def prepare_trajectory_seeds(self, batch_size, num_seeds, current_state, seed_config=None, seed_traj=None):
        return self.core.prepare_trajectory_seeds(batch_size, num_seeds, current_state, seed_config, seed_traj)

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
        self.core.reset_shape()

    def reset_seed(self):
        self.core.reset_seed()

    def reset_cuda_graph(self):
        self.core.reset_cuda_graph()

    def destroy(self):
        self.core.destroy()

    # -------------------------------------------------------------------
    # MPC-specific: tool pose criteria also updates ik_solver
    # -------------------------------------------------------------------

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        self.ik_solver.update_tool_pose_criteria(tool_pose_criteria)
        self.core.update_tool_pose_criteria(tool_pose_criteria)

    # -------------------------------------------------------------------
    # Goal buffer
    # -------------------------------------------------------------------

    def _prepare_goal_buffer(
        self,
        solve_state: SolveState,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        use_implicit_goal: bool = False,
        seed_goal_state: Optional[JointState] = None,
        goal_state: Optional[JointState] = None,
    ) -> Tuple[GoalRegistry, bool]:
        if current_state is None:
            log_and_raise("current_state is required for MPC")
        goal_buffer, update_reference = self.core.prepare_goal_buffer(
            solve_state=solve_state,
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
            use_implicit_goal=use_implicit_goal,
            seed_goal_state=seed_goal_state,
            goal_state=goal_state,
        )
        return goal_buffer, update_reference

    def _update_rollout_params(self, goal_buffer, include_auxiliary_rollout=True):
        self.core.update_rollout_params(goal_buffer, include_auxiliary_rollout)

    @property
    def problem_batch_size(self) -> int:
        return self.solve_state.get_trajopt_batch_size()

    # -------------------------------------------------------------------
    # MPC-specific methods
    # -------------------------------------------------------------------

    def setup(
        self,
        current_state: JointState,
        tool_frames: Optional[List[str]] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> None:
        """Setup MPC for tracking."""
        batch_size = self.config.max_batch_size
        max_goalset = self.config.max_goalset

        self._validate_setup_input(batch_size, current_state, max_goalset, tool_frames, dt)

        if dt is None:
            dt = torch.full(
                (batch_size,),
                self.config.optimization_dt,
                device=current_state.device,
                dtype=current_state.dtype,
            )
        current_state.dt = dt

        if tool_frames is None:
            tool_frames = self.tool_frames
        kin_state = self.compute_kinematics(current_state.clone())
        goal_tool_poses = kin_state.tool_poses.as_goal(tool_frames)

        if batch_size == 1:
            solve_mode = SolveMode.SINGLE
        elif self.config.multi_env:
            solve_mode = SolveMode.MULTI_ENV
        else:
            solve_mode = SolveMode.BATCH

        goal_state = current_state.clone()
        solve_state = SolveState(
            solve_type=solve_mode,
            num_envs=batch_size if self.config.multi_env else 1,
            num_goalset=max_goalset,
            batch_size=batch_size,
            tool_frames=tool_frames,
            num_trajopt_seeds=self.config.num_seeds,
        )

        goal_buffer, update_reference = self._prepare_goal_buffer(
            solve_state=solve_state,
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
            use_implicit_goal=False,
            goal_state=goal_state,
            seed_goal_state=goal_state.clone().unsqueeze(1),
        )
        self.optimizer.update_rollout_params(goal=goal_buffer)
        self.metrics_rollout.update_params(goal=goal_buffer)
        seed_trajectory = self.prepare_trajectory_seeds(
            num_seeds=self.solve_state.num_seeds,
            batch_size=self.solve_state.batch_size,
            current_state=current_state.clone(),
        )
        self.trajectory_execution_manager.update_action_buffer(seed_trajectory.clone())
        self.enable_tool_pose_tracking()
        if batch_size > 1:
            self.enable_joint_position_tracking()
        else:
            self.disable_joint_position_tracking()
        self.cold_start_solve(current_state)
        self.reset_robot(current_state)

        self._mpc_setup_complete = True

    def _validate_setup_input(
        self,
        n_robots: int,
        current_state: JointState,
        num_goalset: int,
        tool_frames: Optional[List[str]] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> None:
        if n_robots <= 0:
            log_and_raise(f"n_robots must be greater than 0, got {n_robots}")
        if num_goalset <= 0:
            log_and_raise(f"num_goalset must be greater than 0, got {num_goalset}")
        if tool_frames is not None:
            if tool_frames == []:
                log_and_raise("tool_frames must not be empty list, use None to track all links")
            if not isinstance(tool_frames, list):
                log_and_raise("tool_frames must be a list")
            if not all(isinstance(name, str) for name in tool_frames):
                log_and_raise("tool_frames must be a list of strings")
            if tool_frames not in self.tool_frames:
                log_and_raise(f"tool_frames must be a subset of {self.tool_frames}")
        if current_state.ndim != 2:
            log_and_raise("current_state must be a 2D tensor")
        if current_state.shape[0] != n_robots:
            log_and_raise(f"current_state must have {n_robots} rows, got {current_state.shape[0]}")
        if current_state.shape[1] != self.action_dim:
            log_and_raise(
                f"current_state must have {self.action_dim} columns, got {current_state.shape[1]}"
            )
        if dt is not None:
            if dt.ndim != 1:
                log_and_raise("dt must be a 1D tensor")
            if dt.shape[0] != n_robots:
                log_and_raise(f"dt must have {n_robots} rows, got {dt.shape[0]}")

    def update_goal_tool_poses(
        self,
        goal_tool_poses: GoalToolPose,
        robot_ids: Optional[torch.Tensor] = None,
        run_ik: bool = True,
        use_ik_goal: bool = True,
        use_best_effort_ik: bool = False,
    ) -> bool:
        """Update the Cartesian goal poses for one or more robots.

        When *robot_ids* is provided, only the poses for those robots are
        overwritten inside the previously stored goal; otherwise the full goal
        is replaced. If *run_ik* is ``True``, an IK solve via
        :attr:`ik_solver` is attempted to obtain a feasible goal joint
        configuration, and joint-position tracking is enabled when
        *use_ik_goal* is also ``True``. When *run_ik* is ``False``,
        joint-position tracking is disabled and only the Cartesian goal is
        updated in the rollout parameters.

        Args:
            goal_tool_poses: Target tool-frame poses. Position shape
                ``[batch, horizon, num_links, num_goalset, 3]``; quaternion
                shape ``[batch, horizon, num_links, num_goalset, 4]``.
            robot_ids: Optional 1-D integer tensor selecting which robots in
                the batch to update. When ``None`` all robots are updated.
            run_ik: If ``True``, run :class:`IKSolver` to compute a feasible
                goal joint state from the target poses.
            use_ik_goal: If ``True`` and IK succeeds, also set the goal joint
                state and enable joint-position tracking.
            use_best_effort_ik: If ``True``, accept the IK result even when
                not all problems converge; otherwise require full success.

        Returns:
            ``True`` if the goal was successfully updated (including IK when
            requested), ``False`` if IK was requested but failed.
        """
        if robot_ids is not None:
            if self._goal_tool_poses is None:
                log_and_raise(
                    "goal_tool_poses not set, call update_goal_tool_poses without robot_ids first"
                )
            for i, link_name in enumerate(goal_tool_poses.tool_frames):
                self._goal_tool_poses.position[robot_ids, :, i, :, :] = (
                    goal_tool_poses.position[robot_ids, :, i, :, :]
                )
                self._goal_tool_poses.quaternion[robot_ids, :, i, :, :] = (
                    goal_tool_poses.quaternion[robot_ids, :, i, :, :]
                )
            goal_tool_poses = self._goal_tool_poses

        if run_ik:
            ik_result = self._solve_ik_for_goal(goal_tool_poses, robot_ids)
            if ik_result is not None:
                if use_best_effort_ik or torch.all(ik_result.success):
                    goal_joint_state = JointState.from_position(
                        ik_result.solution.view(-1, self.action_dim), joint_names=self.joint_names
                    )
                    log_info("IK succeeded, updated target joint configuration with IK solution")
                    if use_ik_goal:
                        self.update_goal_state(goal_joint_state)
                        current_state = self.core._goal_buffer.current_js.clone()
                        self.enable_joint_position_tracking()
                    new_goal_buffer = self.goal_registry_manager.update_goal_tool_poses(goal_tool_poses)
                    self._goal_tool_poses = goal_tool_poses
                    self._update_rollout_params(new_goal_buffer)
                    return True
            return False
        else:
            self.disable_joint_position_tracking()
            new_goal_buffer = self.goal_registry_manager.update_goal_tool_poses(goal_tool_poses)
            self._goal_tool_poses = goal_tool_poses
            self._update_rollout_params(new_goal_buffer)
            return True

    def _solve_ik_for_goal(
        self,
        goal_tool_poses: GoalToolPose,
        robot_ids: Optional[torch.Tensor] = None,
    ):
        if self.core._goal_buffer is not None and self.core._goal_buffer.current_js is not None:
            current_state = self.core._goal_buffer.current_js.clone()
        else:
            current_state = self.default_joint_state.clone().unsqueeze(0)

        batch_size = goal_tool_poses.batch_size
        ik_result = self.ik_solver.solve_pose(
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
            seed_config=current_state.position.view(batch_size, 1, self.action_dim).clone(),
            return_seeds=1,
        )
        return ik_result

    def update_goal_state(self, goal_state: JointState, robot_ids: Optional[torch.Tensor] = None):
        """Update the joint-space goal state used for tracking.

        Replaces the goal joint configuration in the
        :attr:`goal_registry_manager` and propagates the change to all
        rollout instances.

        Args:
            goal_state: Desired goal joint state with position tensor of shape
                ``[batch, dof]``.
            robot_ids: Reserved for future per-robot updates. Currently
                raises an error if not ``None``.
        """
        if robot_ids is not None:
            log_and_raise("robot_ids not supported for update_goal_state")
        new_goal_buffer = self.goal_registry_manager.update_goal_state(goal_state)
        self._update_rollout_params(new_goal_buffer)

    def update_current_state(self, current_state: JointState):
        """Update the current robot state in the goal buffer and rollout params.

        If ``current_state.dt`` is ``None``, it is filled with
        :attr:`config.optimization_dt`. The updated state is written to the
        :attr:`goal_registry_manager` and propagated to all rollout
        instances.

        Args:
            current_state: Current joint state with position tensor of shape
                ``[batch, dof]``.
        """
        if current_state.dt is None:
            current_state.dt = torch.full(
                (1,),
                self.config.optimization_dt,
                device=current_state.device,
                dtype=current_state.dtype,
            )
        new_goal_buffer = self.goal_registry_manager.update_current_state(current_state)
        self._update_rollout_params(new_goal_buffer)

    def update_seed_trajectory(self, seed_trajectory: torch.Tensor):
        if seed_trajectory.ndim != 3:
            log_and_raise(f"seed_trajectory must have 3 dimensions, got {seed_trajectory.ndim}")
        if seed_trajectory.shape[0] != self.problem_batch_size:
            log_and_raise(
                f"seed_trajectory must have {self.problem_batch_size} rows, got {seed_trajectory.shape[0]}"
            )
        if seed_trajectory.shape[2] != self.action_dim:
            log_and_raise(
                f"seed_trajectory must have {self.action_dim} columns, got {seed_trajectory.shape[2]}"
            )
        if seed_trajectory.shape[1] != self.action_horizon:
            log_and_raise(
                f"seed_trajectory must have {self.action_horizon} columns, got {seed_trajectory.shape[1]}"
            )
        self.trajectory_execution_manager.update_action_buffer(seed_trajectory.clone())
        self.optimizer.reinitialize(seed_trajectory)

    def update_seed_trajectory_from_goal_state(self, goal_joint_state: JointState) -> None:
        if self.core._goal_buffer is None or self.core._goal_buffer.current_js is None:
            log_and_raise("Current state not available. Call setup first.")

        current_state = self.core._goal_buffer.current_js
        num_seeds = self.solve_state.num_seeds
        batch_size = self.solve_state.batch_size

        goal_config = goal_joint_state.position.unsqueeze(1).repeat(1, num_seeds, 1)
        seed_trajectory = self.prepare_trajectory_seeds(
            num_seeds=num_seeds,
            batch_size=batch_size,
            current_state=current_state.clone(),
            seed_config=goal_config,
        )
        self.update_seed_trajectory(seed_trajectory)

    def optimize_next_action(self, current_state: JointState) -> MPCSolverResult:
        """Return the next single-step action from the MPC action buffer.

        On the first call after :func:`setup` (or after a reset), a cold-start
        optimization is performed. On subsequent calls the solver checks whether
        the trajectory execution manager still has a valid command; if not, a
        warm-start re-optimization is triggered before the next command is
        extracted.

        Args:
            current_state: Current joint state of the robot with position tensor
                of shape ``[batch, dof]``.

        Returns:
            :class:`MPCSolverResult` with :attr:`~MPCSolverResult.next_action`
            set to the single-step :class:`JointState` command,
            :attr:`~MPCSolverResult.action_buffer` containing the full planned
            horizon, and timing information in
            :attr:`~MPCSolverResult.solve_time`.

        Raises:
            RuntimeError: If :func:`setup` has not been called.
        """
        timer = CudaEventTimer().start()
        if not self._mpc_setup_complete:
            log_and_raise("MPC problem not setup, call setup first")
        if not self._mpc_warm_start_available:
            self.cold_start_solve(current_state)
            self._mpc_warm_start_available = True

        if not self.trajectory_execution_manager.has_valid_next_command():
            self.warm_start_solve(current_state)

        next_command = self.trajectory_execution_manager.get_next_command()
        metrics = self.trajectory_execution_manager.get_current_metrics()
        result = self._get_result(metrics)

        result.next_action = next_command
        result.solve_time = timer.stop()
        result.robot_state_sequence = (
            self.trajectory_execution_manager.get_robot_state_sequence()
            if self.trajectory_execution_manager.has_valid_robot_state_trajectory()
            else None
        )
        result.action_buffer = self.trajectory_execution_manager.get_action_buffer()
        result.action_dt = self.config.optimization_dt
        return result

    def optimize_action_sequence(self, current_state: JointState) -> MPCSolverResult:
        """Re-optimize and return the full action sequence for the planning horizon.

        Unlike :func:`optimize_next_action`, which returns only the immediate
        next command, this method always re-optimizes (cold-start on the first
        call, warm-start thereafter) and returns the complete command sequence
        over the action horizon.

        Args:
            current_state: Current joint state of the robot with position tensor
                of shape ``[batch, dof]``.

        Returns:
            :class:`MPCSolverResult` with
            :attr:`~MPCSolverResult.action_sequence` set to the full
            :class:`JointState` command sequence over the horizon,
            :attr:`~MPCSolverResult.action_buffer` containing the raw planned
            actions, and :attr:`~MPCSolverResult.solve_time`.

        Raises:
            RuntimeError: If :func:`setup` has not been called.
        """
        timer = CudaEventTimer().start()
        if not self._mpc_setup_complete:
            log_and_raise("MPC problem not setup, call setup first")
        if not self._mpc_warm_start_available:
            self.cold_start_solve(current_state)
            self._mpc_warm_start_available = True
        else:
            self.warm_start_solve(current_state)
        metrics = self.trajectory_execution_manager.get_current_metrics()
        command_sequence = self.trajectory_execution_manager.get_command_sequence()

        result = self._get_result(metrics)
        result.action_sequence = command_sequence
        result.solve_time = timer.stop()
        result.robot_state_sequence = (
            self.trajectory_execution_manager.get_robot_state_sequence()
            if self.trajectory_execution_manager.has_valid_robot_state_trajectory()
            else None
        )
        result.action_buffer = self.trajectory_execution_manager.get_action_buffer()
        result.action_dt = self.config.optimization_dt
        return result

    def cold_start_solve(self, current_state: JointState):
        """Run a full optimization from scratch with no warm-start information.

        Uses :attr:`config.cold_start_optimization_num_iters` iterations,
        which is typically larger than the warm-start count, to build an
        initial high-quality trajectory.

        Args:
            current_state: Current joint state with position tensor of shape
                ``[batch, dof]``.

        Returns:
            :class:`RolloutMetrics` from the optimization pass.
        """
        metrics = self._solve_impl(current_state, self.config.cold_start_optimization_num_iters)
        return metrics

    def warm_start_solve(self, current_state: JointState):
        """Run a warm-start optimization reusing the shifted action buffer.

        Uses :attr:`config.warm_start_optimization_num_iters` iterations,
        which is typically fewer than the cold-start count, since the
        previous solution (shifted by one step) provides a good initial
        seed.

        Args:
            current_state: Current joint state with position tensor of shape
                ``[batch, dof]``.

        Returns:
            :class:`RolloutMetrics` from the optimization pass.
        """
        metrics = self._solve_impl(current_state, self.config.warm_start_optimization_num_iters)
        return metrics

    def _solve_impl(self, current_state: JointState, optimization_niters: int):
        self.optimizer.update_niters(optimization_niters)
        seed_trajectory = self.trajectory_execution_manager.get_shifteaction_dim_buffer()
        self.optimizer.shift(shift_steps=1)
        self.update_current_state(current_state)

        opt_result = self.optimizer.optimize(seed_trajectory.clone())
        best_actions = opt_result.clone()

        metrics_result = self.metrics_rollout.compute_metrics_from_action(opt_result)

        feasible_mask = metrics_result.costs_and_constraints.get_feasible(
            include_all_hybrid=False, sum_horizon=False
        )
        feasible_mask = torch.all(
            feasible_mask[:, : int(self.config.interpolation_steps * 2)], dim=-1
        )

        if torch.count_nonzero(feasible_mask) < self.problem_batch_size and False:
            new_seed_trajectory = self.prepare_safe_deceleration_trajectory(
                current_state=current_state.clone(), failed_mask=~feasible_mask
            )
            best_actions[~feasible_mask] = new_seed_trajectory[~feasible_mask]
            metrics_result = self.metrics_rollout.compute_metrics_from_action(best_actions)

        self.trajectory_execution_manager.update_robot_state_trajectory(metrics_result.state)
        self.trajectory_execution_manager.update_state_action_metrics_buffers(
            metrics_result.state.joint_state, best_actions, metrics_result
        )

        self._debug_counter += 1
        return metrics_result

    def _update_trajectory_execution_manager(self, actions: torch.Tensor):
        metrics_result = self.metrics_rollout.compute_metrics_from_action(actions)
        self.trajectory_execution_manager.update_robot_state_trajectory(metrics_result.state)
        self.trajectory_execution_manager.update_state_action_metrics_buffers(
            metrics_result.state.joint_state, actions, metrics_result
        )

    def prepare_safe_deceleration_trajectory(
        self,
        current_state: JointState,
        failed_mask: torch.Tensor,
        deceleration_time: Optional[float] = None,
        deceleration_profile: Optional[str] = None,
    ) -> torch.Tensor:
        """Build a safe deceleration trajectory for robots that failed feasibility.

        For robots whose optimized trajectory is infeasible (indicated by
        *failed_mask*), this method generates a trajectory that smoothly
        decelerates the robot to a stop. If the robot has negligible velocity
        or deceleration-on-failure is disabled in config, a static hold
        trajectory from :func:`prepare_trajectory_seeds` is returned instead.

        Args:
            current_state: Current joint state with position tensor of shape
                ``[batch, dof]`` and optional velocity tensor of the same
                shape.
            failed_mask: Boolean tensor of shape ``[batch]`` where ``True``
                marks robots that need a safe deceleration fallback.
            deceleration_time: Time in seconds over which to decelerate.
                Defaults to :attr:`config.deceleration_time`, clamped to
                :attr:`config.max_deceleration_time`.
            deceleration_profile: Name of the deceleration profile (e.g.
                ``"linear"``). Defaults to
                :attr:`config.deceleration_profile`.

        Returns:
            Trajectory seed tensor of shape
            ``[batch * num_seeds, action_horizon, dof]`` containing safe
            deceleration actions for the failed robots and unchanged seeds
            for the rest.
        """
        if deceleration_time is None:
            deceleration_time = self.config.deceleration_time
        if deceleration_profile is None:
            deceleration_profile = self.config.deceleration_profile
        if deceleration_time is not None:
            deceleration_time = min(deceleration_time, self.config.max_deceleration_time)

        use_deceleration = (
            self.config.use_deceleration_on_failure
            and current_state.velocity is not None
            and torch.any(torch.abs(current_state.velocity) > 1e-6)
        )

        if use_deceleration:
            safe_trajectory = self.seed_manager.prepare_deceleration_trajectory_seeds(
                num_seeds=self.solve_state.num_seeds,
                batch_size=self.solve_state.batch_size,
                current_state=current_state,
                deceleration_time=deceleration_time,
                deceleration_profile=deceleration_profile,
            )
        else:
            safe_trajectory = self.prepare_trajectory_seeds(
                num_seeds=self.solve_state.num_seeds,
                batch_size=self.solve_state.batch_size,
                current_state=current_state,
            )
        return safe_trajectory

    def _get_result(self, metrics_result: RolloutMetrics) -> MPCSolverResult:
        if self.config.num_seeds > 1:
            log_and_raise("This is only implemented for 1 seed per mpc problem.")

        feasible = metrics_result.costs_and_constraints.get_feasible(
            include_all_hybrid=False, sum_horizon=True
        )

        converge_list = []
        position_list = []
        orientation_list = []
        cspace_list = []
        goalset_index_list = []
        for k in range(len(metrics_result.convergence.names)):
            metric_name = metrics_result.convergence.names[k]
            metric_values = metrics_result.convergence.values[k]
            first_step_values = metric_values[:, :1]

            if "position_tolerance" in metric_name:
                position_list.append(first_step_values)
                converged = first_step_values < self.config.position_tolerance
                converge_list.append(converged)
            elif "orientation_tolerance" in metric_name:
                orientation_list.append(first_step_values)
                converged = first_step_values < self.config.orientation_tolerance
                converge_list.append(converged)
            elif "cspace_tolerance" in metric_name:
                cspace_list.append(first_step_values)
                converge_list.append(first_step_values < 0.01)
            elif "goalset_index" in metric_name:
                goalset_index_list.append(first_step_values)

        position_error = None
        orientation_error = None
        if len(position_list) > 0:
            position_error = torch.cat(position_list, dim=-1).view(self.problem_batch_size, -1)
        if len(orientation_list) > 0:
            orientation_error = torch.cat(orientation_list, dim=-1).view(
                self.problem_batch_size, -1
            )

        if converge_list:
            converged_all_links = torch.cat(converge_list, dim=-1)
            converged = torch.all(converged_all_links, dim=-1).squeeze(-1)
        else:
            converged = torch.ones_like(feasible)
        goalset_index = torch.cat(goalset_index_list, dim=-1) if goalset_index_list else None

        result = MPCSolverResult(
            success=feasible,
            position_error=(
                torch.max(position_error, dim=-1)[0] if position_error is not None else None
            ),
            rotation_error=(
                torch.max(orientation_error, dim=-1)[0] if orientation_error is not None else None
            ),
            goalset_index=goalset_index,
            solve_time=0.0,
            debug_info={"solver": self.optimizer.get_recorded_trace(), "seed_idx": None},
        )
        return result

    def set_default_goal_from_current_state(
        self, current_state: JointState, robot_ids: Optional[torch.Tensor] = None
    ) -> None:
        kin_state = self.compute_kinematics(current_state.clone())
        target_links = self.solve_state.tool_frames
        goal_tool_poses = kin_state.tool_poses.as_goal(target_links)
        self.update_goal_tool_poses(goal_tool_poses, robot_ids)

    def reset_robot(self, current_state: JointState) -> None:
        if current_state.ndim != 2:
            log_and_raise("current_state must be a 2D tensor")
        if current_state.shape[0] != self.problem_batch_size:
            log_and_raise(
                f"current_state must have {self.problem_batch_size} rows, got {current_state.shape[0]}"
            )
        if current_state.shape[1] != self.action_dim:
            log_and_raise(
                f"current_state must have {self.action_dim} columns, got {current_state.shape[1]}"
            )
        seed_trajectory = self.prepare_trajectory_seeds(
            num_seeds=self.solve_state.num_seeds,
            batch_size=self.solve_state.batch_size,
            current_state=current_state.clone(),
        )
        self.trajectory_execution_manager.update_action_buffer(seed_trajectory.clone())
        self.optimizer.reinitialize(seed_trajectory)
        self._mpc_warm_start_available = False

    def reset_robot_id(self, current_state: JointState, robot_ids: torch.Tensor) -> None:
        if current_state.ndim != 2:
            log_and_raise("current_state must be a 2D tensor")
        if current_state.shape[0] != self.problem_batch_size:
            log_and_raise(
                f"current_state must have {self.problem_batch_size} rows, got {current_state.shape[0]}"
            )
        if current_state.shape[1] != self.action_dim:
            log_and_raise(
                f"current_state must have {self.action_dim} columns, got {current_state.shape[1]}"
            )
        seed_trajectory = self.prepare_trajectory_seeds(
            num_seeds=self.solve_state.num_seeds,
            batch_size=self.solve_state.batch_size,
            current_state=current_state.clone(),
        )
        current_buffer = self.trajectory_execution_manager.get_action_buffer()
        current_buffer[robot_ids] = seed_trajectory[robot_ids]
        self._update_trajectory_execution_manager(current_buffer)

        mask = torch.zeros(
            self.problem_batch_size, dtype=torch.bool, device=robot_ids.device
        )
        mask[robot_ids] = True
        self.optimizer.reinitialize(current_buffer, mask=mask)
