# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory optimization solver for computing collision-free, time-optimal paths.

Wraps SolverCore with trajectory-specific logic: multi-seed trajectory generation,
iterative dt-finetuning for time optimality, interpolation to dense waypoints, and
solution ranking. Supports both Cartesian-space (solve_pose) and configuration-space
(solve_cspace) goals.
"""
from __future__ import annotations

# Standard Library
from typing import Dict, Optional, Tuple

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
import curobo._src.runtime as curobo_runtime
from curobo._src.geom.collision import SceneCollision
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.solver.solver_core import SolverCore
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import log_and_raise, log_info, log_warn
from curobo._src.util.torch_util import get_torch_jit_decorator, is_cuda_graph_reset_available
from curobo._src.util.trajectory import calculate_dt_no_clamp, get_batch_interpolated_trajectory
from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator


class TrajOptSolver:
    """Trajectory Optimization solver."""

    def __init__(
        self,
        config: TrajOptSolverCfg,
        scene_collision_checker: Optional[SceneCollision] = None,
    ):
        """Initializes the TrajOpt solver."""
        self.config = config
        self.core = SolverCore(config.core_cfg, scene_collision_checker)

        # TrajOpt needs an additional interpolated rollout
        rollout_instance = RobotRollout(
            self.config.metrics_rollout_config,
            self.core.scene_collision_checker,
            use_cuda_graph=self.config.use_cuda_graph,
        )
        rollout_instance.rollout_instance_name = "interpolated_rollout"
        self.core.additional_metrics_rollouts["interpolated_rollout"] = rollout_instance

        # TrajOpt-specific: trajectory seed generator and interpolation buffers
        self._trajectory_seed_generator = TrajectorySeedGenerator(
            self.action_horizon, self.action_dim, self.device_cfg
        )
        self._interpolated_traj_buffer = None
        self._interpolation_dt_buffer = None
        self._seed_dt_buffer = None

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
    def optimizer_rollouts(self):
        return self.core.optimizer_rollouts

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
    def horizon(self) -> int:
        return self.auxiliary_rollout.horizon

    @property
    def opt_dim(self) -> int:
        return self.action_dim * self.action_horizon

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

    def update_tool_pose_criteria(self, tool_pose_criteria):
        return self.core.update_tool_pose_criteria(tool_pose_criteria)

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
    # Goal buffer
    # -------------------------------------------------------------------

    @profiler.record_function("trajopt_solver/_prepare_goal_buffer")
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
            log_and_raise("current_state is required for TrajOpt")
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
        return self.core._solve_state.get_trajopt_batch_size()

    @property
    def interpolation_steps(self) -> int:
        return self.auxiliary_rollout.transition_model.interpolation_steps

    # -------------------------------------------------------------------
    # TrajOpt-specific methods
    # -------------------------------------------------------------------

    @profiler.record_function("trajopt_solver/_solve_impl")
    def _solve_impl(
        self,
        current_state: JointState,
        solve_state: SolveState,
        goal_tool_poses: GoalToolPose,
        num_seeds: int,
        seed_config=None,
        seed_traj=None,
        return_seeds: int = 1,
        dt=None,
        use_implicit_goal: bool = False,
        finetune_attempts: int = 0,
        goal_state: Optional[JointState] = None,
        initial_iters: Optional[int] = None,
        time_optimal_iters: Optional[int] = None,
        finetune_iters: Optional[int] = None,
        finetune_dt_scale: float = 0.55,
    ) -> TrajOptSolverResult:
        opt_time = 0.0
        metrics_time = 0.0
        seed_prep_time = 0.0
        seed_goal_time = 0.0
        total_timer = CudaEventTimer().start()
        # enable all optimizers
        for optimizer in self.optimizer.optimizers:
            optimizer.enable()
        # 1. Get Seed Trajectories
        action_seed = self.prepare_trajectory_seeds(
            solve_state.batch_size,
            num_seeds,
            current_state=current_state,
            seed_config=seed_config,
            seed_traj=seed_traj,
        )
        with profiler.record_function("trajopt_solver/calculate_seed_goal_state"):
            seed_goal_state = action_seed[..., -1, :].view(-1, self.action_dim)
            seed_goal_state = seed_goal_state.view(solve_state.batch_size, num_seeds, self.action_dim)
            seed_goal_state = JointState.from_position(seed_goal_state)
            if self._seed_dt_buffer is None or self._seed_dt_buffer.shape != (
                solve_state.batch_size,
                num_seeds,
            ):
                self._seed_dt_buffer = torch.ones(
                    (solve_state.batch_size, num_seeds),
                    device=self.device_cfg.device,
                    dtype=self.device_cfg.dtype,
                )
            seed_goal_state.dt = self._seed_dt_buffer
            if dt is not None:
                seed_goal_state.dt[:] = dt

            if seed_traj is not None or seed_config is not None:
                temp_goal_buffer_for_dt, update_reference = self._prepare_goal_buffer(
                    solve_state,
                    goal_tool_poses,
                    current_state,
                    use_implicit_goal,
                    seed_goal_state,
                    goal_state,
                )
                self.metrics_rollout.update_params(goal=temp_goal_buffer_for_dt)
                self.additional_metrics_rollouts["interpolated_rollout"].update_params(
                    goal=temp_goal_buffer_for_dt
                )
                with torch.no_grad():
                    robot_state = self.metrics_rollout.compute_state_from_action(
                        action_seed.detach()
                    )
                state_seq = robot_state.joint_state
                calculated_dt = self.compute_trajectory_dt(state_seq, scale_dt=True)
                calculated_dt = calculated_dt.view(solve_state.batch_size, num_seeds)
                if dt is None:
                    dt = calculated_dt
                    seed_goal_state.dt = dt

        best_trajopt_result: Optional[TrajOptSolverResult] = None
        best_dt = seed_goal_state.dt.clone()
        goal_buffer = None

        for i in range(finetune_attempts + 1):
            if curobo_runtime.debug:
                if best_trajopt_result is not None:
                    if torch.max(best_trajopt_result.seed_rank) >= best_trajopt_result.num_seeds:
                        log_and_raise(
                            f" i before get_result best_trajopt_result.seed_rank: {best_trajopt_result.seed_rank} >= best_trajopt_result.num_seeds: {best_trajopt_result.num_seeds}"
                        )

            current_dt_for_buffer = best_dt
            current_dt_for_buffer = current_dt_for_buffer * finetune_dt_scale
            current_dt_for_buffer = torch.clamp(
                current_dt_for_buffer,
                min=self.config.minimum_trajectory_dt,
                max=self.config.maximum_trajectory_dt,
            )
            if goal_buffer is not None:
                self._update_trajectory_dt(current_dt_for_buffer, goal_buffer)
            else:
                if current_dt_for_buffer.shape != seed_goal_state.dt.shape:
                    log_and_raise(
                        f"current_dt_for_buffer.shape: {current_dt_for_buffer.shape} != seed_goal_state.dt.shape: {seed_goal_state.dt.shape}"
                    )
                seed_goal_state.dt.copy_(current_dt_for_buffer)
                goal_buffer, update_reference = self._prepare_goal_buffer(
                    solve_state,
                    goal_tool_poses,
                    current_state,
                    use_implicit_goal,
                    seed_goal_state,
                    goal_state,
                )
                self._update_rollout_params(goal_buffer)

            num_iters_backup = None
            if hasattr(self.optimizer.optimizers[-1].config, "num_iters"):
                if i == 0:
                    num_iters_backup = self.optimizer.optimizers[-1].config.num_iters
                    if initial_iters is not None:
                        self.optimizer.optimizers[-1].update_niters(initial_iters)
                if i == 1:
                    if time_optimal_iters is not None:
                        self.optimizer.optimizers[-1].update_niters(time_optimal_iters)
                if i > 1:
                    if finetune_iters is not None:
                        self.optimizer.optimizers[-1].update_niters(finetune_iters)

            if i >= 1:
                if len(self.optimizer.optimizers) > 1:
                    self.optimizer.optimizers[0].disable()

            current_seed = (
                action_seed
                if i == 0
                else best_trajopt_result.optimized_seeds.view(
                    -1, self.action_horizon, self.action_dim
                ).clone()
            )
            opt_timer = CudaEventTimer().start()

            if i < 1:
                self.optimizer.reinitialize(current_seed)
            else:
                self.optimizer.reinitialize(current_seed, clear_optimizer_state=False)

            opt_result = self.optimizer.optimize(current_seed)
            opt_time += opt_timer.stop()

            if num_iters_backup is not None:
                self.optimizer.optimizers[-1].update_niters(num_iters_backup)

            with profiler.record_function("trajopt_solver/post_optimization"):
                optimized_state = self.metrics_rollout.compute_state_from_action(opt_result)
                optimized_joint_state = optimized_state.joint_state
                new_dt = self.compute_trajectory_dt(optimized_joint_state, scale_dt=True)
                new_dt = new_dt.view(solve_state.batch_size, num_seeds)

                if new_dt.shape != seed_goal_state.dt.shape:
                    log_and_raise(
                        f"new_dt.shape: {new_dt.shape} != seed_goal_state.dt.shape: {seed_goal_state.dt.shape}"
                    )
                self._update_trajectory_dt(new_dt, goal_buffer)
                metrics_result = self.metrics_rollout.compute_metrics_from_action(opt_result)

            js_optimized = metrics_result.state.joint_state
            js_optimized.joint_names = self.joint_names
            js_optimized.knot = metrics_result.actions
            js_optimized.knot_dt = (
                metrics_result.state.joint_state.dt
                * self.auxiliary_rollout.transition_model.interpolation_steps
            )
            interpolated_metrics, interpolated_trajectory, last_tstep = (
                self._interpolate_and_compute_metrics(js_optimized)
            )

            trajopt_result = self._get_result(
                opt_result,
                metrics_result,
                interpolated_metrics,
                interpolated_trajectory,
                last_tstep,
                num_seeds,
                num_seeds,
            )

            with profiler.record_function("trajopt_solver/post_get_result"):
                planned_dt = metrics_result.state.joint_state.dt.clone()
                new_dt = planned_dt.view(solve_state.batch_size, num_seeds)

                if best_trajopt_result is None:
                    best_trajopt_result = trajopt_result.clone()
                    best_dt = new_dt.clone()
                else:
                    dt_mask = new_dt <= best_dt
                    update_mask = torch.logical_and(trajopt_result.success, dt_mask)
                    if torch.any(update_mask):
                        trajopt_result.success[:] = update_mask
                        best_trajopt_result.copy_successful_solutions(trajopt_result)
                        best_dt = best_trajopt_result.js_solution.dt.clone()
                    else:
                        break

            if i == 0 and torch.count_nonzero(best_trajopt_result.success) == 0:
                break

        best_trajopt_result = self._get_best_result(best_trajopt_result, return_seeds)
        best_trajopt_result.solve_time = opt_time
        best_trajopt_result.total_time = total_timer.stop()
        return best_trajopt_result

    @profiler.record_function("trajopt_solver/_get_best_result")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def _get_best_result(
        self,
        all_seeds_result: TrajOptSolverResult,
        return_seeds: int,
    ) -> TrajOptSolverResult:
        best_trajopt_result = all_seeds_result.get_topk_seeds(return_seeds)
        best_solution = best_trajopt_result.js_solution
        if best_solution.joint_names is None:
            best_solution.joint_names = self.joint_names
        best_trajopt_result.js_solution = (
            self.auxiliary_rollout.transition_model.get_full_dof_from_solution(best_solution)
        )
        best_trajopt_result.js_solution.knot_dt = best_trajopt_result.js_solution.dt * self.interpolation_steps
        best_trajopt_result.js_solution.knot = best_trajopt_result.solution
        return best_trajopt_result

    @profiler.record_function("trajopt_solver/_interpolate_and_compute_metrics")
    def _interpolate_and_compute_metrics(
        self, js_optimized: JointState
    ) -> Tuple[RolloutMetrics, torch.Tensor, torch.Tensor]:
        batch_size = self.solve_state.batch_size
        num_seeds = self.solve_state.num_seeds
        interpolated_rollout = self.additional_metrics_rollouts["interpolated_rollout"]

        interpolated_js, last_tstep, buffer_updated = self.get_interpolated_trajectory(js_optimized)
        interpolated_robot_state = interpolated_rollout.transition_model.compute_augmented_state(
            interpolated_js
        )
        interpolated_metrics = interpolated_rollout.compute_metrics_from_state(
            interpolated_robot_state
        )
        interpolated_trajectory = interpolated_js.view(
            batch_size, num_seeds, interpolated_js.shape[-2], self.action_dim
        )
        interpolated_trajectory = interpolated_rollout.transition_model.get_full_dof_from_solution(
            interpolated_trajectory
        )
        last_tstep = last_tstep.view(batch_size, num_seeds)
        return interpolated_metrics, interpolated_trajectory, last_tstep

    @profiler.record_function("trajopt_solver/_get_result")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def _get_result(
        self,
        opt_result: torch.Tensor,
        metrics_result: RolloutMetrics,
        interpolated_metrics: RolloutMetrics,
        interpolated_trajectory: torch.Tensor,
        last_tstep: torch.Tensor,
        num_seeds: int,
        return_seeds: int,
    ) -> TrajOptSolverResult:
        batch_size = self.solve_state.batch_size
        num_seeds = self.solve_state.num_seeds

        optimized_seeds = metrics_result.actions.view(
            batch_size, num_seeds, self.action_horizon, self.action_dim
        )
        js_solution_seq = metrics_result.state.joint_state
        js_solution_seq = js_solution_seq.view(batch_size, num_seeds, self.horizon, self.action_dim)
        if js_solution_seq.joint_names is None:
            js_solution_seq.joint_names = self.joint_names
        js_solution_seq.knot_dt = js_solution_seq.dt * self.interpolation_steps
        js_solution_seq.knot = optimized_seeds

        trajopt_result = TrajOptSolverResult(
            success=None,
            solution=optimized_seeds,
            js_solution=js_solution_seq,
            position_error=None,
            rotation_error=None,
            position_tolerance=self.config.position_tolerance,
            orientation_tolerance=self.config.orientation_tolerance,
            goalset_index=None,
            solve_time=0.0,
            debug_info={"solver": self.optimizer.get_recorded_trace()},
            optimized_seeds=optimized_seeds,
            interpolated_trajectory=interpolated_trajectory,
            interpolated_metrics=interpolated_metrics,
            metrics=metrics_result,
            interpolated_last_tstep=last_tstep,
            batch_size=batch_size,
            num_seeds=num_seeds,
            minimum_trajectory_dt=self.config.minimum_trajectory_dt,
            maximum_trajectory_dt=self.config.maximum_trajectory_dt,
        )
        trajopt_result.process_metrics_and_rank_seeds()
        trajopt_result = trajopt_result.get_topk_seeds(return_seeds)
        return trajopt_result

    @profiler.record_function("trajopt_solver/_update_trajectory_dt")
    def _update_trajectory_dt(self, dt: torch.Tensor, goal_buffer: GoalRegistry):
        if goal_buffer.seed_goal_js is None:
            log_and_raise("seed_goal_js is None in goal_buffer")
        if goal_buffer.seed_goal_js.dt is None:
            log_and_raise("dt is None in seed_goal_js in goal_buffer")
        if dt.shape != goal_buffer.seed_goal_js.dt.shape:
            log_and_raise(
                f"dt.shape: {dt.shape} != goal_buffer.seed_goal_js.dt.shape: {goal_buffer.seed_goal_js.dt.shape}"
            )
        goal_buffer.seed_goal_js.dt.copy_(dt)
        self._update_rollout_dt(goal_buffer)

    def _update_rollout_dt(self, goal_buffer: GoalRegistry):
        for rollout in self.get_all_rollout_instances(
            include_optimizer_rollouts=True, include_auxiliary_rollout=False
        ):
            rollout.update_goal_dt(goal_buffer)

    @profiler.record_function("trajopt_solver/get_interpolated_trajectory")
    def get_interpolated_trajectory(
        self,
        js_optimized: JointState,
    ) -> Tuple[JointState, torch.Tensor, bool]:
        batch_size, _, _ = js_optimized.shape
        if (
            self._interpolated_traj_buffer is None
            or js_optimized.shape[0] != self._interpolated_traj_buffer.shape[0]
        ):
            self._interpolated_traj_buffer = JointState.zeros(
                (batch_size, self.config.interpolation_buffer_size, self.action_dim), self.device_cfg
            )
            self._interpolated_traj_buffer.joint_names = self.joint_names
        if (
            self._interpolation_dt_buffer is None
            or self._interpolation_dt_buffer.shape[0] != batch_size
        ):
            self._interpolation_dt_buffer = (
                torch.ones((batch_size,), device=self.device_cfg.device, dtype=torch.float32)
                * self.config.interpolation_dt
            )

        interpolation_buffer_reallocated = False
        goal = self.core._goal_buffer

        if self.auxiliary_rollout.transition_model.control_space in ControlSpace.bspline_types():
            js_optimized.control_space = self.auxiliary_rollout.transition_model.control_space
            if js_optimized.knot is None:
                log_and_raise("Knots not set for bspline interpolation")
            if js_optimized.knot_dt is None:
                log_and_raise("Knot dt not set for bspline interpolation")
        else:
            js_optimized.control_space = ControlSpace.POSITION

        state, last_tstep = get_batch_interpolated_trajectory(
            js_optimized,
            self._interpolation_dt_buffer,
            kind=self.config.interpolation_type,
            device_cfg=self.device_cfg,
            out_traj_state=self._interpolated_traj_buffer,
            current_state=goal.current_js,
            goal_state=goal.seed_goal_js,
            start_idx=goal.idxs_current_js,
            goal_idx=goal.idxs_seed_goal_js,
            use_implicit_goal_state=goal.seed_enable_implicit_goal_js,
        )

        if state.shape != self._interpolated_traj_buffer.shape:
            interpolation_buffer_reallocated = True
            if is_cuda_graph_reset_available():
                log_info("Interpolated trajectory buffer was recreated, reinitializing cuda graph")
                self._interpolated_traj_buffer = state.clone()
            else:
                log_and_raise(
                    "Interpolated trajectory buffer was recreated, but cuda graph is not available"
                )
        return state, last_tstep, interpolation_buffer_reallocated

    @profiler.record_function("trajopt_solver/compute_trajectory_dt")
    def compute_trajectory_dt(
        self,
        trajectory: JointState,
        epsilon: float = 1e-3,
        scale_dt: bool = True,
    ) -> torch.Tensor:
        if (
            trajectory.velocity is None
            or trajectory.acceleration is None
            or trajectory.jerk is None
        ):
            log_and_raise("compute_trajectory_dt requires velocity, acceleration, and jerk.")

        vel_limit = self.transition_model.max_velocity.view(1, -1)
        acc_limit = self.transition_model.max_acceleration.view(1, -1)
        jerk_limit = self.transition_model.max_jerk.view(1, -1)
        while vel_limit.ndim < trajectory.position.ndim - 1:
            vel_limit = vel_limit.unsqueeze(0)
            acc_limit = acc_limit.unsqueeze(0)
            jerk_limit = jerk_limit.unsqueeze(0)

        opt_dt = calculate_dt_no_clamp(
            trajectory.velocity,
            trajectory.acceleration,
            trajectory.jerk,
            vel_limit,
            acc_limit,
            jerk_limit,
            epsilon=epsilon,
        )
        if trajectory.dt is not None and scale_dt:
            opt_dt = opt_dt * trajectory.dt

        opt_dt = torch.clamp(
            opt_dt, min=self.config.minimum_trajectory_dt, max=self.config.maximum_trajectory_dt
        )
        return opt_dt

    # -------------------------------------------------------------------
    # Public solve methods
    # -------------------------------------------------------------------

    @profiler.record_function("trajopt_solver/solve_pose")
    def solve_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        seed_config=None,
        seed_traj=None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        dt=None,
        use_implicit_goal: bool = False,
        finetune_attempts: int = 1,
        goal_state: Optional[JointState] = None,
        initial_iters: Optional[int] = None,
        time_optimal_iters: Optional[int] = None,
        finetune_iters: Optional[int] = None,
        finetune_dt_scale: float = 0.55,
    ) -> TrajOptSolverResult:
        """Solve trajectory optimization for target tool poses in Cartesian space.

        Computes a time-optimal, collision-free joint-space trajectory that reaches
        the desired tool-frame poses from the current robot state. The solver pads
        the batch to :attr:`config.max_batch_size` when the input batch is smaller,
        runs multi-seed optimization with optional iterative dt-finetuning, and
        returns only the requested number of top-ranked seeds.

        Args:
            goal_tool_poses: Target poses for each tool link. Shape of position
                tensor is ``[batch, horizon, num_links, num_goalset, 3]``; quaternion
                tensor is ``[batch, horizon, num_links, num_goalset, 4]``.
            current_state: Current joint state of the robot with position tensor of
                shape ``[batch, dof]``.
            seed_config: Optional initial joint configurations used as trajectory
                seeds. Shape ``[batch, num_seeds, dof]``.
            seed_traj: Optional seed trajectory tensor of shape
                ``[batch, action_horizon, dof]`` used instead of generated seeds.
            return_seeds: Number of top-ranked solutions to return per problem.
                If larger than *num_seeds*, the seed count is increased
                automatically.
            num_seeds: Number of optimization seeds per problem. Defaults to
                :attr:`config.num_seeds` when ``None``.
            dt: Optional trajectory time-step tensor of shape ``[batch, num_seeds]``
                overriding the automatically computed dt.
            use_implicit_goal: When ``True``, the solver treats the last trajectory
                waypoint as a free variable rather than pinning it to the goal.
            finetune_attempts: Number of additional time-optimal refinement passes
                after the initial solve. Each pass shrinks dt by
                *finetune_dt_scale*.
            goal_state: Optional explicit goal joint state with position tensor of
                shape ``[batch, dof]``. Used to seed the goal end of the
                trajectory.
            initial_iters: Override for the number of optimizer iterations on the
                first pass. ``None`` keeps the config default.
            time_optimal_iters: Override for the number of optimizer iterations on
                the second (time-optimal) pass.
            finetune_iters: Override for the number of optimizer iterations on
                subsequent finetune passes.
            finetune_dt_scale: Multiplicative factor applied to dt between finetune
                passes. Defaults to ``0.55``.

        Returns:
            :class:`TrajOptSolverResult` containing the best trajectories ranked by
            cost. Key fields include :attr:`~TrajOptSolverResult.success`,
            :attr:`~TrajOptSolverResult.js_solution` (interpolated joint-state
            trajectory), :attr:`~TrajOptSolverResult.position_error`, and
            :attr:`~TrajOptSolverResult.rotation_error`.
        """
        max_batch = self.config.max_batch_size
        batch_size = goal_tool_poses.batch_size
        if num_seeds is None:
            num_seeds = self.config.num_seeds
        if return_seeds > num_seeds:
            log_warn(
                f"Requested {return_seeds} solutions, increasing optimization seeds "
                f"from {num_seeds} to {return_seeds}."
            )
            num_seeds = return_seeds

        if batch_size > max_batch:
            log_and_raise(
                f"solve_pose: batch_size={batch_size} exceeds "
                f"config.max_batch_size={max_batch}."
            )

        num_goalset = goal_tool_poses.num_goalset
        if num_goalset > self.config.max_goalset:
            log_and_raise(
                f"solve_pose: num_goalset={num_goalset} exceeds config.max_goalset="
                f"{self.config.max_goalset}."
            )

        actual_batch_size = batch_size
        needs_pad = batch_size < max_batch
        if needs_pad:
            from curobo._src.solver.solver_ik import _pad_batch_inputs
            goal_tool_poses, current_state, seed_config = _pad_batch_inputs(
                goal_tool_poses, current_state, seed_config, batch_size, max_batch,
            )
            if goal_state is not None:
                pad = max_batch - batch_size
                goal_state = goal_state.clone()
                goal_state.position = torch.cat(
                    [goal_state.position, goal_state.position[:1].expand(pad, -1)], dim=0,
                )
            if seed_traj is not None:
                pad = max_batch - batch_size
                seed_traj = torch.cat(
                    [seed_traj, seed_traj[:1].expand(pad, *[-1] * (seed_traj.ndim - 1))], dim=0,
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
            num_trajopt_seeds=num_seeds,
            batch_size=batch_size,
            num_envs=batch_size if self.config.multi_env else 1,
            num_goalset=num_goalset,
            tool_frames=goal_tool_poses.tool_frames,
        )

        result = self._solve_impl(
            current_state=current_state,
            solve_state=solve_state,
            goal_tool_poses=goal_tool_poses,
            num_seeds=num_seeds,
            seed_config=seed_config,
            seed_traj=seed_traj,
            return_seeds=return_seeds,
            dt=dt,
            use_implicit_goal=use_implicit_goal,
            finetune_attempts=finetune_attempts,
            goal_state=goal_state,
            initial_iters=initial_iters,
            time_optimal_iters=time_optimal_iters,
            finetune_iters=finetune_iters,
            finetune_dt_scale=finetune_dt_scale,
        )
        if needs_pad:
            from curobo._src.solver.solver_ik import _slice_batch_result
            result = _slice_batch_result(result, actual_batch_size)
        return result

    @profiler.record_function("trajopt_solver/solve_cspace")
    def solve_cspace(
        self,
        goal_state: JointState,
        current_state: JointState,
        seed_traj=None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        dt=None,
        finetune_attempts: int = 1,
        initial_iters: Optional[int] = None,
        time_optimal_iters: Optional[int] = None,
        finetune_iters: Optional[int] = None,
    ) -> TrajOptSolverResult:
        """Solve trajectory optimization in configuration space (joint-to-joint).

        Given a goal joint configuration and the current robot state, computes a
        time-optimal, collision-free trajectory entirely in joint space. Forward
        kinematics is run on *goal_state* to derive the Cartesian tool-pose target
        used internally, and :func:`_solve_impl` is called with
        ``use_implicit_goal=True`` so the final waypoint is pinned to the exact
        goal configuration.

        Args:
            goal_state: Desired goal joint state with position tensor of shape
                ``[batch, dof]``.
            current_state: Current joint state of the robot with position tensor of
                shape ``[batch, dof]``.
            seed_traj: Optional seed trajectory tensor of shape
                ``[batch, action_horizon, dof]`` used instead of generated seeds.
            return_seeds: Number of top-ranked solutions to return per problem.
            num_seeds: Number of optimization seeds per problem. Defaults to
                :attr:`config.num_seeds` when ``None``.
            dt: Optional trajectory time-step tensor of shape
                ``[batch, num_seeds]`` overriding the automatically computed dt.
            finetune_attempts: Number of additional time-optimal refinement passes
                after the initial solve.
            initial_iters: Override for the number of optimizer iterations on the
                first pass. ``None`` keeps the config default.
            time_optimal_iters: Override for the number of optimizer iterations on
                the second (time-optimal) pass.
            finetune_iters: Override for the number of optimizer iterations on
                subsequent finetune passes.

        Returns:
            :class:`TrajOptSolverResult` containing the best joint-space
            trajectories ranked by cost. Key fields include
            :attr:`~TrajOptSolverResult.success`,
            :attr:`~TrajOptSolverResult.js_solution`, and
            :attr:`~TrajOptSolverResult.solve_time`.
        """
        max_batch = self.config.max_batch_size
        batch_size = current_state.shape[0]
        if num_seeds is None:
            num_seeds = self.config.num_seeds
        if return_seeds > num_seeds:
            log_warn(
                f"Requested {return_seeds} solutions, increasing optimization seeds "
                f"from {num_seeds} to {return_seeds}."
            )
            num_seeds = return_seeds

        if batch_size > max_batch:
            log_and_raise(
                f"solve_cspace: batch_size={batch_size} exceeds "
                f"config.max_batch_size={max_batch}."
            )

        actual_batch_size = batch_size
        needs_pad = batch_size < max_batch
        if needs_pad:
            pad = max_batch - batch_size
            current_state = current_state.clone()
            current_state.position = torch.cat(
                [current_state.position, current_state.position[:1].expand(pad, -1)], dim=0,
            )
            if current_state.velocity is not None:
                current_state.velocity = torch.cat(
                    [current_state.velocity, current_state.velocity[:1].expand(pad, -1)], dim=0,
                )
            if current_state.acceleration is not None:
                current_state.acceleration = torch.cat(
                    [current_state.acceleration, current_state.acceleration[:1].expand(pad, -1)], dim=0,
                )
            goal_state = goal_state.clone()
            goal_state.position = torch.cat(
                [goal_state.position, goal_state.position[:1].expand(pad, -1)], dim=0,
            )
            if seed_traj is not None:
                seed_traj = torch.cat(
                    [seed_traj, seed_traj[:1].expand(pad, *[-1] * (seed_traj.ndim - 1))], dim=0,
                )
        batch_size = max_batch

        goal_kin_state = self.compute_kinematics(goal_state)
        fk_tool_poses = goal_kin_state.tool_poses
        goal_tool_poses = GoalToolPose(
            tool_frames=fk_tool_poses.tool_frames,
            position=fk_tool_poses.position.unsqueeze(3),
            quaternion=fk_tool_poses.quaternion.unsqueeze(3),
        )

        seed_config = goal_state.position.view(batch_size, 1, -1).repeat(1, num_seeds, 1)

        if batch_size == 1:
            solve_mode = SolveMode.SINGLE
        elif self.config.multi_env:
            solve_mode = SolveMode.MULTI_ENV
        else:
            solve_mode = SolveMode.BATCH

        solve_state = SolveState(
            solve_type=solve_mode,
            num_trajopt_seeds=num_seeds,
            batch_size=batch_size,
            num_envs=batch_size if self.config.multi_env else 1,
            num_goalset=1,
            tool_frames=goal_tool_poses.tool_frames,
        )

        result = self._solve_impl(
            current_state=current_state,
            solve_state=solve_state,
            goal_tool_poses=goal_tool_poses,
            num_seeds=num_seeds,
            seed_config=seed_config,
            seed_traj=seed_traj,
            return_seeds=return_seeds,
            dt=dt,
            use_implicit_goal=True,
            finetune_attempts=finetune_attempts,
            goal_state=goal_state,
            initial_iters=initial_iters,
            time_optimal_iters=time_optimal_iters,
            finetune_iters=finetune_iters,
        )
        if needs_pad:
            from curobo._src.solver.solver_ik import _slice_batch_result
            result = _slice_batch_result(result, actual_batch_size)
        return result
