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
import math
from typing import Optional, Tuple

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.solver.processor_trajopt_result import (
    TorchTrajOptResultProcessor,
    TrajOptResultCandidate,
    TrajOptResultProcessor,
    TrajOptResultProcessorConfig,
)
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.solver.solver_core import SolverCore
from curobo._src.solver.solver_timeopt import TimeOptSolver
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.trajectory.trajectory_interpolation import (
    calculate_dt_no_clamp,
    get_batch_interpolated_trajectory,
)
from curobo._src.trajectory.trajectory_seed_generator import TrajectorySeedGenerator
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import log_and_raise, log_warn


class TrajOptSolver:
    """Trajectory Optimization solver."""

    def __init__(
        self,
        config: TrajOptSolverCfg,
        scene_collision_checker: Optional[SceneCollision] = None,
    ):
        """Initializes the TrajOpt solver."""
        self.config = config
        self._timeopt: Optional[TimeOptSolver] = None
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
        self._interpolation_buffer_is_validated = False
        self._seed_dt_buffer = None
        result_processor_config = TrajOptResultProcessorConfig(
            device_cfg=self.device_cfg,
            action_horizon=self.action_horizon,
            horizon=self.horizon,
            action_dim=self.action_dim,
            interpolation_steps=self.interpolation_steps,
            goalset_index_dim=max(1, len(self.tool_frames)),
            goalset_index_dtype=torch.int32,
            joint_names=tuple(self.joint_names),
            position_tolerance=self.config.position_tolerance,
            orientation_tolerance=self.config.orientation_tolerance,
            minimum_trajectory_dt=self.config.minimum_trajectory_dt,
            maximum_trajectory_dt=self.config.maximum_trajectory_dt,
            use_cuda_graph=self.config.use_cuda_graph,
        )
        self._result_processor: TrajOptResultProcessor = TorchTrajOptResultProcessor(
            result_processor_config,
            self.auxiliary_rollout.transition_model.get_full_dof_from_solution,
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

    @property
    def result_processor(self) -> TrajOptResultProcessor:
        """Return the result processor used by trajectory optimization."""
        return self._result_processor

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

    def update_tool_pose_criteria(self, tool_pose_criteria):
        return self.core.update_tool_pose_criteria(tool_pose_criteria)

    def update_tool_relative_transform_criteria(self, criteria):
        return self.core.update_tool_relative_transform_criteria(criteria)

    def update_tool_velocity_limit_criteria(self, criteria):
        return self.core.update_tool_velocity_limit_criteria(criteria)

    def update_link_inertial(self, link_name, mass=None, com=None, inertia=None):
        return self.core.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(self, link_properties):
        return self.core.update_links_inertial(link_properties)

    def debug_dump(self, file_path):
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

    def _reset_timeopt(self) -> None:
        """Release timeopt workspaces and graphs bound to their output buffers."""
        if self._timeopt is not None:
            self._timeopt.reset_cuda_graph()
            self._timeopt = None
            self._result_processor.reset_cuda_graph()

    def reset_shape(self):
        self._reset_timeopt()
        self.core.reset_shape()
        self._result_processor.reset_shape()

    def reset_seed(self):
        self.core.reset_seed()

    def reset_cuda_graph(self):
        self._reset_timeopt()
        self.core.reset_cuda_graph()
        self._result_processor.reset_cuda_graph()

    def destroy(self):
        self._reset_timeopt()
        self._result_processor.reset_shape()
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
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
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
            tool_relative_transform_target=tool_relative_transform_target,
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
        time_optimal_iters: Optional[int] = 25,
        finetune_iters: Optional[int] = 25,
        finetune_dt_scale: float = 0.6,
        initial_dt_scale: Optional[float] = 1.25,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
    ) -> TrajOptSolverResult:
        opt_time = 0.0
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
        goal_buffer = None
        with profiler.record_function("trajopt_solver/calculate_seed_goal_state"):
            seed_goal_state = action_seed[..., -1, :].view(-1, self.action_dim)
            seed_goal_state = seed_goal_state.view(
                solve_state.batch_size, num_seeds, self.action_dim
            )
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
                goal_buffer, _ = self._prepare_goal_buffer(
                    solve_state,
                    goal_tool_poses,
                    current_state,
                    use_implicit_goal,
                    seed_goal_state,
                    goal_state,
                    tool_relative_transform_target,
                )
                self.metrics_rollout.update_params(goal=goal_buffer)
                self.additional_metrics_rollouts["interpolated_rollout"].update_params(
                    goal=goal_buffer
                )
                with torch.no_grad():
                    calculated_dt = self.metrics_rollout.compute_time_scaled_dt_from_action(
                        action_seed.detach(),
                        self.transition_model.get_state_bounds(),
                        epsilon=1e-3,
                        minimum_dt=self.config.minimum_trajectory_dt,
                        maximum_dt=self.config.maximum_trajectory_dt,
                    )
                calculated_dt = calculated_dt.view(solve_state.batch_size, num_seeds)
                if dt is None:
                    dt = calculated_dt
                    seed_goal_state.dt = dt

        self._result_processor.update_batch_size(
            solve_state.batch_size,
            num_seeds,
            return_seeds,
        )
        self._result_processor.reset(seed_goal_state.dt)
        next_seed = None

        for i in range(finetune_attempts + 1):
            current_dt_for_buffer = self._result_processor.best_dt
            dt_scale = (
                initial_dt_scale
                if i == 0 and initial_dt_scale is not None
                else finetune_dt_scale
            )
            current_dt_for_buffer = current_dt_for_buffer * dt_scale
            current_dt_for_buffer = torch.clamp(
                current_dt_for_buffer,
                min=self.config.minimum_trajectory_dt,
                max=self.config.maximum_trajectory_dt,
            )
            if goal_buffer is not None:
                self._update_trajectory_dt(current_dt_for_buffer)
            else:
                if current_dt_for_buffer.shape != seed_goal_state.dt.shape:
                    log_and_raise(
                        f"current_dt_for_buffer.shape: {current_dt_for_buffer.shape} != seed_goal_state.dt.shape: {seed_goal_state.dt.shape}"
                    )
                seed_goal_state.dt.copy_(current_dt_for_buffer)
                goal_buffer, _ = self._prepare_goal_buffer(
                    solve_state,
                    goal_tool_poses,
                    current_state,
                    use_implicit_goal,
                    seed_goal_state,
                    goal_state,
                    tool_relative_transform_target,
                )
            if i == 0:
                self._update_rollout_params(goal_buffer)

            num_iters_backup = None
            if hasattr(self.optimizer.optimizers[-1].config, "num_iters"):
                num_iters_backup = self.optimizer.optimizers[-1].config.num_iters
                if i == 0:
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

            current_seed = action_seed
            if i > 0:
                if next_seed is None:
                    log_and_raise("TrajOpt result processor did not provide the next seed")
                current_seed = next_seed.clone()
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
                new_dt = self.metrics_rollout.compute_time_scaled_dt_from_action(
                    opt_result.detach(),
                    self.transition_model.get_state_bounds(),
                    epsilon=1e-3,
                    minimum_dt=self.config.minimum_trajectory_dt,
                    maximum_dt=self.config.maximum_trajectory_dt,
                )
                new_dt = new_dt.view(solve_state.batch_size, num_seeds)

                if new_dt.shape != seed_goal_state.dt.shape:
                    log_and_raise(
                        f"new_dt.shape: {new_dt.shape} != seed_goal_state.dt.shape: {seed_goal_state.dt.shape}"
                    )
                self._update_trajectory_dt(new_dt)
                metrics_result = self.metrics_rollout.compute_metrics_from_action(opt_result)

            js_optimized = metrics_result.state.joint_state
            js_optimized.joint_names = self.joint_names
            js_optimized.knot = metrics_result.actions
            js_optimized.knot_dt = (
                metrics_result.state.joint_state.dt
                * self.auxiliary_rollout.transition_model.interpolation_steps
            )
            interpolated_metrics = self._interpolate_and_compute_metrics(js_optimized)

            result_update = self._result_processor.update(
                TrajOptResultCandidate(
                    metrics=metrics_result,
                    interpolated_metrics=interpolated_metrics,
                    debug_info={"solver": self.optimizer.get_recorded_trace()},
                )
            )
            next_seed = result_update.next_seed

            if i > 0 and not result_update.improved:
                break
            if i == 0 and not result_update.any_success:
                break

        best_trajopt_result = self._finalize_result(current_state, num_seeds, return_seeds)
        best_trajopt_result.solve_time = opt_time
        best_trajopt_result.total_time = total_timer.stop()
        return best_trajopt_result

    def _finalize_result(
        self,
        current_state: JointState,
        num_seeds: int,
        return_seeds: int,
        seed_valid_mask: Optional[torch.Tensor] = None,
    ) -> TrajOptSolverResult:
        """Interpolate or optimize timing for retained seeds, then select the final result."""
        native = self._result_processor.get_interpolation_input()
        batch_size = native.position.shape[0] // num_seeds
        timeopt_success = timeopt_duration = None
        timeopt_time = 0.0
        if self.config.enable_timeopt:
            timer = CudaEventTimer().start()
            lanes = native.position.shape[0]
            if self._timeopt is None or self._timeopt.lanes != lanes:
                self._reset_timeopt()
                self._timeopt = TimeOptSolver(self, lanes)
            interpolated, counts, success, duration = self._timeopt.timeopt(
                native, current_state, num_seeds
            )
            timeopt_success = success.view(batch_size, num_seeds)
            if seed_valid_mask is not None:
                timeopt_success = timeopt_success & seed_valid_mask.to(
                    device=timeopt_success.device, dtype=torch.bool
                )
            timeopt_duration = duration.view(batch_size, num_seeds)
            timeopt_time = timer.stop()
        else:
            interpolated, counts, _ = self.get_interpolated_trajectory(native)
        result = self._result_processor.finalize(
            return_seeds,
            interpolated.view(batch_size, num_seeds, interpolated.shape[-2], self.action_dim),
            counts.view(batch_size, num_seeds),
            timeopt_success,
            timeopt_duration,
        )
        result.timeopt_time = timeopt_time
        return result

    @profiler.record_function("trajopt_solver/_interpolate_and_compute_metrics")
    def _interpolate_and_compute_metrics(self, js_optimized: JointState) -> RolloutMetrics:
        """Interpolate and validate one candidate without retaining the dense trajectory."""
        interpolated_rollout = self.additional_metrics_rollouts["interpolated_rollout"]

        interpolated_js, _, _ = self.get_interpolated_trajectory(js_optimized)
        interpolated_robot_state = interpolated_rollout.transition_model.compute_augmented_state(
            interpolated_js
        )
        interpolated_metrics = interpolated_rollout.compute_metrics_from_state(
            interpolated_robot_state
        )
        return interpolated_metrics

    @profiler.record_function("trajopt_solver/_update_trajectory_dt")
    def _update_trajectory_dt(self, dt: torch.Tensor) -> None:
        self.goal_registry_manager.update_seed_goal_dt(dt)

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
                (batch_size, self.config.interpolation_buffer_size, self.action_dim),
                self.device_cfg,
            )
            self._interpolated_traj_buffer.joint_names = self.joint_names
            self._interpolation_buffer_is_validated = False
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
            if not self._interpolation_buffer_is_validated:
                self._validate_bspline_interpolation_buffer()
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
            use_preallocated_buffer=self._interpolation_buffer_is_validated,
        )

        if state.shape != self._interpolated_traj_buffer.shape:
            self._raise_interpolation_buffer_size_error(state.shape[1])
        return state, last_tstep, interpolation_buffer_reallocated

    def _validate_bspline_interpolation_buffer(self) -> None:
        """Validate the fixed buffer against the configured maximum trajectory dt."""
        control_space = self.auxiliary_rollout.transition_model.control_space
        max_knot_dt = self.config.maximum_trajectory_dt * self.interpolation_steps
        max_steps_per_waypoint = math.ceil(max_knot_dt / self.config.interpolation_dt) + 1
        required_waypoints = ControlSpace.spline_total_interpolation_steps(
            control_space,
            self.action_horizon,
            max_steps_per_waypoint,
        )
        if self.config.interpolation_buffer_size < required_waypoints:
            self._raise_interpolation_buffer_size_error(required_waypoints)
        self._interpolation_buffer_is_validated = True

    def _raise_interpolation_buffer_size_error(self, required_waypoints: int) -> None:
        log_and_raise(
            f"Interpolated trajectory requires {required_waypoints} waypoints, exceeding "
            f"interpolation_buffer_size={self.config.interpolation_buffer_size}. Recreate "
            "the configuration with a larger interpolation_buffer_size or interpolation_dt. "
            "A larger interpolation_dt produces fewer waypoints. Increasing "
            "interpolation_buffer_size can significantly increase GPU memory usage because "
            "position, velocity, acceleration, and jerk buffers are allocated for every "
            "batched seed trajectory."
        )

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

        opt_dt = self._compute_unclamped_trajectory_dt(trajectory, epsilon, scale_dt)
        opt_dt = torch.clamp(
            opt_dt, min=self.config.minimum_trajectory_dt, max=self.config.maximum_trajectory_dt
        )
        return opt_dt

    def _compute_unclamped_trajectory_dt(
        self,
        trajectory: JointState,
        epsilon: float = 1e-3,
        scale_dt: bool = True,
    ) -> torch.Tensor:
        """Return the uniform dt required by velocity, acceleration, and jerk limits."""
        if (
            trajectory.velocity is None
            or trajectory.acceleration is None
            or trajectory.jerk is None
        ):
            log_and_raise("trajectory dt calculation requires velocity, acceleration, and jerk")

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
        return opt_dt

    @torch.no_grad()
    def _evaluate_seed_impl(
        self,
        current_state: JointState,
        solve_state: SolveState,
        goal_tool_poses: GoalToolPose,
        seed_traj: torch.Tensor,
        use_implicit_goal: bool,
        return_seeds: int,
        trajectory_source: str,
        goal_state: Optional[JointState] = None,
        seed_valid_mask: Optional[torch.Tensor] = None,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
    ) -> TrajOptSolverResult:
        """Retime and densely validate fixed-shape action seeds without optimization."""
        total_timer = CudaEventTimer().start()
        batch_size = solve_state.batch_size
        num_seeds = solve_state.num_trajopt_seeds
        expected_shape = (batch_size, num_seeds, self.action_horizon, self.action_dim)
        if tuple(seed_traj.shape) != expected_shape:
            log_and_raise(
                f"seed_traj must have shape {expected_shape}; got {tuple(seed_traj.shape)}"
            )

        action_seed = seed_traj.reshape(-1, self.action_horizon, self.action_dim)
        seed_goal_state = JointState.from_position(
            action_seed[..., -1, :].view(batch_size, num_seeds, self.action_dim)
        )
        seed_goal_state.dt = torch.ones(
            (batch_size, num_seeds),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        goal_buffer, _ = self._prepare_goal_buffer(
            solve_state,
            goal_tool_poses,
            current_state,
            use_implicit_goal,
            seed_goal_state,
            goal_state,
            tool_relative_transform_target,
        )
        self._update_rollout_params(goal_buffer)

        normalized_state = self.metrics_rollout.compute_state_from_action(action_seed).joint_state
        required_dt = self._compute_unclamped_trajectory_dt(
            normalized_state,
            scale_dt=True,
        ).view(batch_size, num_seeds)
        dt_valid = torch.isfinite(required_dt) & (required_dt <= self.config.maximum_trajectory_dt)
        evaluation_dt = torch.nan_to_num(
            required_dt,
            nan=self.config.maximum_trajectory_dt,
            posinf=self.config.maximum_trajectory_dt,
            neginf=self.config.minimum_trajectory_dt,
        ).clamp(
            min=self.config.minimum_trajectory_dt,
            max=self.config.maximum_trajectory_dt,
        )
        self._update_trajectory_dt(evaluation_dt)
        self._result_processor.update_batch_size(
            batch_size,
            num_seeds,
            return_seeds,
        )
        self._result_processor.reset(evaluation_dt)

        metrics_result = self.metrics_rollout.compute_metrics_from_action(action_seed)
        acceptance_mask = dt_valid
        if seed_valid_mask is not None:
            expected_mask_shape = (batch_size, num_seeds)
            if tuple(seed_valid_mask.shape) != expected_mask_shape:
                log_and_raise(
                    f"seed_valid_mask must have shape {expected_mask_shape}; "
                    f"got {tuple(seed_valid_mask.shape)}"
                )
            acceptance_mask = torch.logical_and(
                acceptance_mask,
                seed_valid_mask.to(device=acceptance_mask.device, dtype=torch.bool),
            )
        if isinstance(metrics_result.feasible, bool):
            if metrics_result.feasible:
                metrics_result.feasible = acceptance_mask.reshape(-1)
        else:
            mask_shape = (acceptance_mask.numel(),) + (1,) * (metrics_result.feasible.ndim - 1)
            torch.logical_and(
                metrics_result.feasible,
                acceptance_mask.reshape(mask_shape),
                out=metrics_result.feasible,
            )
        js_seed = metrics_result.state.joint_state
        js_seed.joint_names = self.joint_names
        js_seed.knot = metrics_result.actions
        js_seed.knot_dt = js_seed.dt * self.interpolation_steps
        interpolated_metrics = self._interpolate_and_compute_metrics(js_seed)
        self._result_processor.update(
            TrajOptResultCandidate(
                metrics=metrics_result,
                interpolated_metrics=interpolated_metrics,
                debug_info={"trajectory_source": trajectory_source},
            )
        )
        result = self._finalize_result(current_state, num_seeds, return_seeds, seed_valid_mask)
        result.debug_info["trajectory_source"] = trajectory_source
        result.trajectory_source = trajectory_source
        result.debug_info["required_trajectory_dt"] = required_dt
        result.solve_time = 0.0
        result.total_time = total_timer.stop()
        return result

    def evaluate_pose_seed(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        seed_traj: torch.Tensor,
        use_implicit_goal: bool = True,
        return_seeds: int = 1,
        trajectory_source: str = "fitted_seed",
        seed_valid_mask: Optional[torch.Tensor] = None,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
    ) -> TrajOptSolverResult:
        """Retime and validate pose-planning seeds without running TrajOpt."""
        batch_size, num_seeds = seed_traj.shape[:2]
        if batch_size != self.config.max_batch_size:
            log_and_raise(
                "evaluate_pose_seed requires the configured fixed batch size; "
                f"got {batch_size}, expected {self.config.max_batch_size}"
            )
        solve_mode = (
            SolveMode.SINGLE
            if batch_size == 1
            else SolveMode.MULTI_ENV
            if self.config.multi_env
            else SolveMode.BATCH
        )
        solve_state = SolveState(
            solve_type=solve_mode,
            num_trajopt_seeds=num_seeds,
            batch_size=batch_size,
            num_envs=batch_size if self.config.multi_env else 1,
            num_goalset=goal_tool_poses.num_goalset,
            tool_frames=goal_tool_poses.tool_frames,
        )
        from curobo._src.solver.solver_ik import _pad_tool_relative_transform_target

        tool_relative_transform_target = _pad_tool_relative_transform_target(
            tool_relative_transform_target,
            batch_size=batch_size,
            max_batch=self.config.max_batch_size,
        )
        return self._evaluate_seed_impl(
            current_state,
            solve_state,
            goal_tool_poses,
            seed_traj,
            use_implicit_goal,
            return_seeds,
            trajectory_source,
            seed_valid_mask=seed_valid_mask,
            tool_relative_transform_target=tool_relative_transform_target,
        )

    def evaluate_cspace_seed(
        self,
        goal_state: JointState,
        current_state: JointState,
        seed_traj: torch.Tensor,
        return_seeds: int = 1,
        trajectory_source: str = "fitted_seed",
        seed_valid_mask: Optional[torch.Tensor] = None,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
    ) -> TrajOptSolverResult:
        """Retime and validate configuration-space seeds without running TrajOpt."""
        batch_size, num_seeds = seed_traj.shape[:2]
        if batch_size != self.config.max_batch_size:
            log_and_raise(
                "evaluate_cspace_seed requires the configured fixed batch size; "
                f"got {batch_size}, expected {self.config.max_batch_size}"
            )
        goal_kin_state = self.compute_kinematics(goal_state)
        fk_tool_poses = goal_kin_state.tool_poses
        goal_tool_poses = GoalToolPose(
            tool_frames=fk_tool_poses.tool_frames,
            position=fk_tool_poses.position.unsqueeze(3),
            quaternion=fk_tool_poses.quaternion.unsqueeze(3),
        )
        solve_mode = (
            SolveMode.SINGLE
            if batch_size == 1
            else SolveMode.MULTI_ENV
            if self.config.multi_env
            else SolveMode.BATCH
        )
        solve_state = SolveState(
            solve_type=solve_mode,
            num_trajopt_seeds=num_seeds,
            batch_size=batch_size,
            num_envs=batch_size if self.config.multi_env else 1,
            num_goalset=1,
            tool_frames=goal_tool_poses.tool_frames,
        )
        from curobo._src.solver.solver_ik import _pad_tool_relative_transform_target

        tool_relative_transform_target = _pad_tool_relative_transform_target(
            tool_relative_transform_target,
            batch_size=batch_size,
            max_batch=self.config.max_batch_size,
        )
        return self._evaluate_seed_impl(
            current_state,
            solve_state,
            goal_tool_poses,
            seed_traj,
            True,
            return_seeds,
            trajectory_source,
            seed_valid_mask=seed_valid_mask,
            tool_relative_transform_target=tool_relative_transform_target,
        )

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
        time_optimal_iters: Optional[int] = 25,
        finetune_iters: Optional[int] = 25,
        finetune_dt_scale: float = 0.6,
        initial_dt_scale: Optional[float] = 1.25,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
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
                passes. Defaults to ``0.6``.
            initial_dt_scale: Optional multiplicative factor applied to the
                seed-derived dt before the initial optimization pass (default 1.25). ``None``
                preserves the legacy behavior of using *finetune_dt_scale*.
            tool_relative_transform_target: Optional target transforms relative
                to one moving tool frame. One target group is shared by every
                problem; otherwise provide one group per problem.

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
                f"solve_pose: batch_size={batch_size} exceeds config.max_batch_size={max_batch}."
            )

        num_goalset = goal_tool_poses.num_goalset
        if num_goalset > self.config.max_goalset:
            log_and_raise(
                f"solve_pose: num_goalset={num_goalset} exceeds config.max_goalset="
                f"{self.config.max_goalset}."
            )

        actual_batch_size = batch_size
        needs_pad = batch_size < max_batch
        from curobo._src.solver.solver_ik import _pad_tool_relative_transform_target

        tool_relative_transform_target = _pad_tool_relative_transform_target(
            tool_relative_transform_target,
            batch_size=batch_size,
            max_batch=max_batch,
        )
        if needs_pad:
            from curobo._src.solver.solver_ik import _pad_batch_inputs

            goal_tool_poses, current_state, seed_config = _pad_batch_inputs(
                goal_tool_poses,
                current_state,
                seed_config,
                batch_size,
                max_batch,
            )
            if goal_state is not None:
                pad = max_batch - batch_size
                goal_state = goal_state.clone()
                goal_state.position = torch.cat(
                    [goal_state.position, goal_state.position[:1].expand(pad, -1)],
                    dim=0,
                )
            if seed_traj is not None:
                pad = max_batch - batch_size
                seed_traj = torch.cat(
                    [seed_traj, seed_traj[:1].expand(pad, *[-1] * (seed_traj.ndim - 1))],
                    dim=0,
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
            initial_dt_scale=initial_dt_scale,
            tool_relative_transform_target=tool_relative_transform_target,
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
        time_optimal_iters: Optional[int] = 25,
        finetune_iters: Optional[int] = 25,
        finetune_dt_scale: float = 0.6,
        initial_dt_scale: Optional[float] = 1.25,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
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
            finetune_dt_scale: Multiplicative factor applied to dt between finetune
                passes. Defaults to ``0.6``.
            initial_dt_scale: Optional multiplicative factor applied to the
                seed-derived dt before the initial optimization pass (default 1.25). ``None``
                preserves the legacy behavior of using *finetune_dt_scale*.
            initial_iters: Override for the number of optimizer iterations on the
                first pass. ``None`` keeps the config default.
            time_optimal_iters: Override for the number of optimizer iterations on
                the second (time-optimal) pass.
            finetune_iters: Override for the number of optimizer iterations on
                subsequent finetune passes.
            tool_relative_transform_target: Optional target transforms relative
                to one moving tool frame. One target group is shared by every
                problem; otherwise provide one group per problem.

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
                f"solve_cspace: batch_size={batch_size} exceeds config.max_batch_size={max_batch}."
            )

        actual_batch_size = batch_size
        needs_pad = batch_size < max_batch
        from curobo._src.solver.solver_ik import _pad_tool_relative_transform_target

        tool_relative_transform_target = _pad_tool_relative_transform_target(
            tool_relative_transform_target,
            batch_size=batch_size,
            max_batch=max_batch,
        )
        if needs_pad:
            pad = max_batch - batch_size
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
            goal_state = goal_state.clone()
            goal_state.position = torch.cat(
                [goal_state.position, goal_state.position[:1].expand(pad, -1)],
                dim=0,
            )
            if seed_traj is not None:
                seed_traj = torch.cat(
                    [seed_traj, seed_traj[:1].expand(pad, *[-1] * (seed_traj.ndim - 1))],
                    dim=0,
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
            initial_iters=initial_iters,
            time_optimal_iters=time_optimal_iters,
            finetune_iters=finetune_iters,
            finetune_dt_scale=finetune_dt_scale,
            initial_dt_scale=initial_dt_scale,
            tool_relative_transform_target=tool_relative_transform_target,
        )
        if needs_pad:
            from curobo._src.solver.solver_ik import _slice_batch_result

            result = _slice_batch_result(result, actual_batch_size)
        return result
