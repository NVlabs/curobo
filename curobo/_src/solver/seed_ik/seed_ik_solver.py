# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed IK Solver for CuRobo.

This module implements a Levenberg-Marquardt algorithm for generating seed
configurations for the main IK solver. It provides fast approximate solutions
that are then refined by the full optimizer.
"""

from __future__ import annotations

# Standard Library
from typing import Dict, Optional, Tuple

# Third Party
import torch
import torch.autograd.profiler as profiler

from curobo._src.optim.util.levenberg_marquardt_step import (
    LevenbergMarquardtState,
    LevenbergMarquardtStep,
)

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.solver.solver_ik_result import IKSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.sampling import SampleBuffer
from curobo._src.util.tensor_util import stable_topk, tensor_repeat_seeds
from curobo._src.util.torch_util import get_torch_jit_decorator
from curobo._src.util.warp import init_warp

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from curobo._src.solver.seed_ik.seed_ik_error_calculator import SeedIKErrorCalculator
from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
from curobo._src.solver.seed_ik.seed_ik_state import SeedIKState
from curobo._src.solver.seed_ik.seed_iteration_state_manager import SeedIterationStateManager


class SeedIKSolver:
    """Seed IK Solver using Levenberg-Marquardt algorithm.

    Generates seed configurations for the main IK solver using a fast
    Levenberg-Marquardt algorithm. Supports single and batch solving modes
    with multiple solution returns.

    Features three solver methods:
    - Cholesky decomposition: Fastest, good for well-conditioned jacobians
    - QR decomposition: More numerically stable, better for ill-conditioned jacobians
    - SVD decomposition: Most robust, handles rank-deficient and singular jacobians
    """

    def __init__(self, config: SeedIKSolverCfg):
        """Initialize the Seed IK solver.

        Args:
            config: Configuration for the solver
        """
        # setup_curobo_logger("info")
        init_warp()
        self._batch_size = -1
        self._num_seeds = -1
        self._n_residuals = None
        self.config = config
        self.device_cfg = config.device_cfg
        self._batch_indices = None

        # Create executors - they initialize lazily on first call
        self._initial_state_executor: Optional[GraphExecutor] = None
        self._inner_iterations_executor: Optional[GraphExecutor] = None
        self._goal_tool_poses_buffer: Optional[GoalToolPose] = None
        # Create robot model with jacobian computation enabled
        robot_model_config = config.robot_config.kinematics
        self._robot_model = Kinematics(
            robot_model_config, compute_spheres=False, compute_jacobian=True
        )
        self._aux_robot_model = Kinematics(
            robot_model_config, compute_spheres=False, compute_jacobian=False
        )

        # Store robot properties
        self.dof = self._robot_model.get_dof()
        self.joint_names = self._robot_model.joint_names
        self.tool_frames = self._robot_model.tool_frames
        self.num_links = len(self.tool_frames)
        self.default_joint_position = self._robot_model.default_joint_position

        joint_limit_margin = (
            self.joint_limits.position[1] - self.joint_limits.position[0]
        ) * self.config.joint_limit_margin
        # Joint limits
        self.action_min = self.joint_limits.position[0] + joint_limit_margin
        self.action_max = self.joint_limits.position[1] - joint_limit_margin

        self.action_step_max = self.config.max_step_size * torch.abs(
            self.action_max - self.action_min
        )

        # Velocity-aware IK buffers: allocated in _setup_batch_size, updated in-place in _solve_impl
        self._velocity_current_position = None
        self._velocity_current_velocity = None
        self._velocity_dt = None
        self._velocity_clamping_active = False

        # Setup unified error calculator
        self._setup_error_calculator()

        # Setup iteration state manager
        self._iteration_state_manager = SeedIterationStateManager(
            action_min=self.action_min,
            action_max=self.action_max,
            rho_min=self.config.rho_min,
            lambda_factor=self.config.lambda_factor,
            lambda_min=self.config.lambda_min,
            lambda_max=self.config.lambda_max,
            convergence_position_tolerance=self.config.convergence_position_tolerance,
            convergence_orientation_tolerance=self.config.convergence_orientation_tolerance,
            convergence_joint_limit_weight=self.config.convergence_joint_limit_weight,
        )

        self.act_sample_gen = SampleBuffer.create_halton_sample_buffer(
            ndims=self.dof,
            device_cfg=self.device_cfg,
            up_bounds=self.action_max,
            low_bounds=self.action_min,
            seed=self.config.sampler_seed,
        )

        # create parallel cuda streams:
        log_info(f"Seed IK Solver initialized for robot with {self.dof} DOF")
        log_info(f"Controlled links: {self.tool_frames}")

    def _calculate_n_residuals(self, num_links: int, joint_limit_weight: float):
        """Calculate number of residuals."""
        n_res = 2 * 3 * num_links  # position and orientation
        if joint_limit_weight > 0:
            n_res += self.dof
        if self.config.velocity_weight > 0:
            n_res += self.dof
        if self.config.acceleration_weight > 0:
            n_res += self.dof
        return n_res

    @classmethod
    def _setup_lm_step(self, num_links: int, dof: int, n_residuals: int, tile_threads: int = 64):
        lm_step = LevenbergMarquardtStep(dof, n_residuals, tile_threads=tile_threads)
        return lm_step

    @property
    def n_residuals(self):
        if self._n_residuals is None:
            self._n_residuals = self._calculate_n_residuals(
                self.num_links, self.config.joint_limit_weight
            )
        return self._n_residuals

    def _setup_batch_size(self, batch_size: int, num_seeds: int = 1):
        """Setup the batch size for the solver."""
        if batch_size != self._batch_size or num_seeds != self._num_seeds:
            log_info(f"Setting batch size to {batch_size}")
            self._batch_size = batch_size
            self._num_seeds = num_seeds
            self._num_problems = batch_size * num_seeds
            self.config.num_seeds = num_seeds
            self._lm_step = self._setup_lm_step(
                self.num_links, self.dof, self.n_residuals, tile_threads=self.config.tile_threads
            )
            self.error_calculator.setup_batch_tensors(batch_size * num_seeds, 1)

            self._pred_reduction = torch.zeros(
                (batch_size * num_seeds, 1), device=self.device_cfg.device
            )
            self._new_joint_position = torch.zeros(
                (batch_size * num_seeds, self.dof), device=self.device_cfg.device
            )
            if self.error_calculator.config.use_backward:
                self._new_joint_position.requires_grad_(True)

            idxs_goal = torch.arange(
                0, batch_size, 1, dtype=torch.int32, device=self.device_cfg.device
            ).unsqueeze(-1)

            self._idxs_goal = tensor_repeat_seeds(idxs_goal, num_seeds)

            self._lambda_damping = torch.zeros(
                (batch_size * num_seeds, 1, 1), device=self.device_cfg.device
            )
            self._success = torch.zeros((batch_size * num_seeds), device=self.device_cfg.device)

            # Pre-allocate velocity/accel state buffers with stable addresses for CUDA graphs.
            # _solve_impl uses .copy_() to update these in-place.
            self._velocity_current_position = torch.zeros(
                (batch_size * num_seeds, self.dof), device=self.device_cfg.device,
            )
            self._velocity_current_velocity = torch.zeros(
                (batch_size * num_seeds, self.dof), device=self.device_cfg.device,
            )
            self._velocity_dt = torch.zeros(
                (batch_size * num_seeds,), device=self.device_cfg.device,
            )

    def _setup_error_calculator(self):
        """Setup the unified error and jacobian calculator."""
        self.error_calculator = SeedIKErrorCalculator(
            robot_model=self._robot_model,
            config=self.config,
            device_cfg=self.device_cfg,
            action_min=self.action_min,
            action_max=self.action_max,
        )

    @profiler.record_function("seed_ik_solver/compute_pose_error_and_jacobian")
    def _compute_pose_error_and_jacobian(
        self,
        joint_position: torch.Tensor,
        goal_tool_poses: GoalToolPose,
    ) -> SeedIKState:
        """Compute pose error and its jacobian w.r.t. joint configuration.

        This method now delegates to the unified error calculator.
        """
        result = self.error_calculator.compute_error_and_jacobian(
            joint_position=joint_position,
            goal_poses=goal_tool_poses,
            idxs_goal=self._idxs_goal,
            current_position=self._velocity_current_position,
            current_velocity=self._velocity_current_velocity,
            dt=self._velocity_dt,
            velocity_clamping_active=self._velocity_clamping_active,
        )

        return SeedIKState(
            position_errors=result.position_errors,
            orientation_errors=result.orientation_errors,
            jTerror=result.jTerror,
            jacobian=result.jacobian,
            error_norm=result.error_norm,
            joint_position=result.joint_position,
        )

    def _levenberg_marquardt_step_inner_iterations(
        self,
        iteration_state: SeedIKState,
        goal_tool_poses: GoalToolPose,
    ):
        if self.config.use_cuda_graph:
            if self._inner_iterations_executor is None:
                self._inner_iterations_executor = create_graph_executor(
                    capture_fn=self._levenberg_marquardt_step_inner_iterations_impl,
                    device=self.device_cfg.device,
                )
            return self._inner_iterations_executor(iteration_state, goal_tool_poses)
        else:
            return self._levenberg_marquardt_step_inner_iterations_impl(iteration_state, goal_tool_poses)

    def _levenberg_marquardt_step_inner_iterations_impl(
        self,
        iteration_state: SeedIKState,
        goal_tool_poses: GoalToolPose,
    ):
        for _ in range(self.config.inner_iterations):
            iteration_state = self._levenberg_marquardt_step_impl(iteration_state, goal_tool_poses)
        return iteration_state

    @profiler.record_function("seed_ik_solver/levenberg_marquardt_step_fn")
    def _levenberg_marquardt_step_impl(
        self,
        current_iteration_state: SeedIKState,
        goal_tool_poses: GoalToolPose,
    ) -> SeedIKState:
        """Perform one Levenberg-Marquardt iteration step.

        Args:
            joint_position: Current joint configuration
            goal_tool_poses: Target poses
            lambda_damping: Damping parameter for each batch item

        Returns:
            Tuple of (new_joint_position, error_norm, success_mask)
        """
        state = LevenbergMarquardtState(
            jacobian=current_iteration_state.jacobian,
            jTerror=current_iteration_state.jTerror,
            lambda_damping=current_iteration_state.lambda_damping.view(-1),
            joint_position_in=current_iteration_state.joint_position,
            joint_position_out=self._new_joint_position,
            pred_reduction=self._pred_reduction.view(-1),
        )
        new_joint_position, pred_reduction = self._lm_step(state)

        # New joint configuration is available.
        new_iteration_state = self._compute_pose_error_and_jacobian(new_joint_position, goal_tool_poses)
        new_iteration_state.lambda_damping = current_iteration_state.lambda_damping

        new_iteration_state = self._iteration_state_manager.update_iteration_state(
            current_state=current_iteration_state,
            candidate_state=new_iteration_state,
            predicted_reduction=pred_reduction,
            batch_size=self._num_problems,
        )
        return new_iteration_state

    @profiler.record_function("seed_ik_solver/_check_convergence")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _check_convergence(
        self,
        iteration_state: SeedIKState,
        batch_size: int,
    ):
        pos_errors = iteration_state.position_errors
        ori_errors = iteration_state.orientation_errors
        joint_position = iteration_state.joint_position

        pose_success = torch.logical_and(
            pos_errors < self.config.position_tolerance,
            ori_errors < self.config.orientation_tolerance,
        )
        success = pose_success

        if self.config.joint_limit_weight > 0:
            joint_limit_success = torch.logical_and(
                joint_position > self.joint_limits.position[0],
                joint_position < self.joint_limits.position[1],
            )
            # joint limit success is batch_size, dof
            joint_limit_success = torch.all(joint_limit_success, dim=-1)
            # change this to pose success
            success = torch.logical_and(pose_success, joint_limit_success)
        return success

    def _pad_goal_tool_poses(self, goal_tool_poses: GoalToolPose) -> GoalToolPose:
        """Pad goal poses to match the cached CUDA graph buffer shape.

        On the first call, stores a clone as the buffer. On subsequent calls,
        if num_goalset shrank, copies data into the first slots and fills the
        remainder with the first goal. If num_goalset grew, reallocates the
        buffer and resets CUDA graph executors.
        """
        n = goal_tool_poses.num_goalset

        if (
            self._goal_tool_poses_buffer is None
            or n > self._goal_tool_poses_buffer.num_goalset
            or goal_tool_poses.batch_size != self._goal_tool_poses_buffer.batch_size
        ):
            self._goal_tool_poses_buffer = goal_tool_poses.clone()
            self._initial_state_executor = None
            self._inner_iterations_executor = None
            return self._goal_tool_poses_buffer

        buf_n = self._goal_tool_poses_buffer.num_goalset
        buf_pos = self._goal_tool_poses_buffer.position
        buf_quat = self._goal_tool_poses_buffer.quaternion

        new_pos = goal_tool_poses.position
        new_quat = goal_tool_poses.quaternion

        if n == buf_n:
            buf_pos.copy_(new_pos)
            buf_quat.copy_(new_quat)
        else:
            buf_pos[..., :n, :] = new_pos
            buf_quat[..., :n, :] = new_quat
            buf_pos[..., n:, :] = new_pos[..., :1, :]
            buf_quat[..., n:, :] = new_quat[..., :1, :]

        return self._goal_tool_poses_buffer

    @profiler.record_function("seed_ik_solver/_optimize")
    def _optimize(
        self,
        initial_config: torch.Tensor,  # (batch, num_seeds, dof)
        goal_tool_poses: GoalToolPose,
        success_num_seeds: int = 1,
    ) -> tuple[torch.Tensor, bool, float, float, int]:
        """Solve IK for a single seed configuration.

        Returns:
            Tuple of (final_config, success, position_error, orientation_error, iterations)
        """
        goal_tool_poses = self._pad_goal_tool_poses(goal_tool_poses)
        if initial_config.ndim != 3:
            log_and_raise(
                f"initial_config must be of shape (batch, num_seeds, dof), got {initial_config.shape}"
            )
        joint_position = initial_config  # .clone()
        self._setup_batch_size(initial_config.shape[0], initial_config.shape[1])
        # flatten initial_config:
        joint_position = joint_position.view(self._num_problems, self.dof)
        lambda_damping = self._lambda_damping
        lambda_damping[:] = self.config.lambda_initial

        iteration_state = self._compute_initial_iteration_state(joint_position, goal_tool_poses)
        iteration_state.lambda_damping = lambda_damping
        iteration_state.success = self._success
        max_iterations = self.config.max_iterations // self.config.inner_iterations

        for iteration in range(max_iterations):
            # Perform LM step
            iteration_state = self._levenberg_marquardt_step_inner_iterations(
                iteration_state, goal_tool_poses
            )
            if iteration < max_iterations - 1:
                success = iteration_state.success.view(self._batch_size, self._num_seeds)
                # make sure at least success_num_seeds are successful

                exit_condition = self._calculate_exit_condition(
                    success,
                    success_num_seeds,
                    self.config.batch_success_threshold,
                    self._batch_size,
                )
                if exit_condition:
                    break

        # Check convergence
        success = self._check_convergence(iteration_state, self._batch_size)
        pos_errors = iteration_state.position_errors
        ori_errors = iteration_state.orientation_errors
        joint_position = iteration_state.joint_position
        # calculate success again:

        max_pos_error = pos_errors.view(self._batch_size, self._num_seeds)
        max_ori_error = ori_errors.view(self._batch_size, self._num_seeds)

        joint_position = joint_position.view(self._batch_size, self._num_seeds, self.dof)
        success = success.view(self._batch_size, self._num_seeds)

        return (joint_position, success, max_pos_error, max_ori_error, iteration + 1)

    def _compute_initial_iteration_state(
        self, joint_position: torch.Tensor, goal_tool_poses: GoalToolPose
    ):
        if self.config.use_backward:
            joint_position = joint_position.detach().clone().requires_grad_(True)



        if self.config.use_cuda_graph:
            if self._initial_state_executor is None:
                self._initial_state_executor = create_graph_executor(
                    capture_fn=self._compute_pose_error_and_jacobian,
                    device=self.device_cfg.device,
                )
            return self._initial_state_executor(joint_position, goal_tool_poses)
        else:
            return self._compute_pose_error_and_jacobian(joint_position, goal_tool_poses)

    @profiler.record_function("seed_ik_solver/_calculate_exit_condition")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def _calculate_exit_condition(
        self,
        success: torch.Tensor,
        success_num_seeds: int,
        batch_success_threshold: float,
        batch_size: int,
    ):
        exit_condition = False
        success_count = torch.sum(success, dim=-1)
        success_count = torch.count_nonzero(success_count >= success_num_seeds)
        if success_count >= batch_success_threshold * batch_size:
            exit_condition = True
        return exit_condition

    def _generate_seed_configs(
        self, batch_size: int, seed_config: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate seed configurations for optimization.

        Args:
            batch_size: Number of seed configurations to generate
            seed_config: Optional seed configuration (batch_size, 1-num_seeds, dof)

        Returns:
            Tensor of shape (batch_size, num_seeds, dof)
        """
        if seed_config is not None:
            if seed_config.ndim != 3:
                log_and_raise(
                    f"seed_config must be of shape (batch, num_seeds, dof), got {seed_config.shape}"
                )
            if seed_config.shape[0] != batch_size:
                log_and_raise(
                    f"Provided seed_config batch size {seed_config.shape[0]} "
                    f"!= required {batch_size}"
                )

            if seed_config.shape[1] > self.config.num_seeds:
                log_and_raise(
                    f"Provided seed_config has {seed_config.shape[1]} seeds, "
                    f"but only {self.config.num_seeds} are needed"
                )
            if seed_config.shape[1] < self.config.num_seeds:
                # sample extra seeds
                extra_seeds = self.act_sample_gen.get_samples(
                    batch_size * (self.config.num_seeds - seed_config.shape[1]), bounded=True
                ).view(batch_size, self.config.num_seeds - seed_config.shape[1], self.dof)

                seeds = torch.cat([seed_config, extra_seeds], dim=1)
            else:
                seeds = seed_config
        else:
            # Generate random seeds
            seeds = (
                self.act_sample_gen.get_samples(self.config.num_seeds, bounded=True)
                .view(1, self.config.num_seeds, self.dof)
                .repeat(batch_size, 1, 1)
            )
            seeds = seeds.view(batch_size, self.config.num_seeds, self.dof)
            # make first seed be default joint state:
            seeds[:, -1, :] = self._robot_model.default_joint_state.position.view(1, 1, -1)

        return seeds

    @profiler.record_function("seed_ik_solver/_select_top_solutions")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def _select_top_solutions(
        self,
        solutions: torch.Tensor,
        successes: torch.Tensor,
        position_errors: torch.Tensor,
        orientation_errors: torch.Tensor,
        start_joint_position: Optional[torch.Tensor],
        batch_size: int,
        num_seeds: int,
        return_seeds: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top solutions based on error ranking."""
        # Rank solutions by error (successful solutions first)
        # shape: (batch_size, num_seeds)

        if solutions.shape != (batch_size, num_seeds, self.dof):
            log_and_raise(
                f"solutions must be of shape (batch_size, num_seeds, dof), got {solutions.shape}"
            )

        costs = position_errors + orientation_errors

        if self.config.start_cspace_dist_weight > 0 and start_joint_position is not None:
            if start_joint_position.shape != (batch_size, self.dof):
                log_and_raise(
                    f"start_joint_position must be of shape (batch_size, dof), got {start_joint_position.shape}, or set null_space_weight to 0"
                )
            start_joint_position = start_joint_position.view(batch_size, 1, self.dof)

            start_cspace_dist_cost = torch.norm(
                solutions - start_joint_position, dim=-1
            )  # shape: (batch_size, num_seeds)
            costs += self.config.start_cspace_dist_weight * start_cspace_dist_cost
        costs[~successes] += 1e10  # Penalize failed solutions

        # Select top solutions
        _, top_indices = stable_topk(costs, k=return_seeds, largest=False)
        top_indices = top_indices.view(batch_size, return_seeds)

        # Convert to flat indices for advanced indexing
        top_indices = self._batch_indices.unsqueeze(-1) + top_indices.unsqueeze(0)
        top_indices = top_indices.view(-1)
        # Extract top results using the same pattern for all tensors

        top_solutions = solutions.view(-1, self.dof)[top_indices].view(
            batch_size, return_seeds, self.dof
        )
        top_successes = successes.view(-1)[top_indices].view(batch_size, return_seeds)
        top_pos_errors = position_errors.view(-1)[top_indices].view(batch_size, return_seeds)
        top_ori_errors = orientation_errors.view(-1)[top_indices].view(batch_size, return_seeds)

        return top_successes, top_solutions, top_pos_errors, top_ori_errors

    def _solve_impl(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        seed_config: Optional[torch.Tensor] = None,
        return_seeds: int = 1,
        batch_size: int = 1,
    ) -> IKSolverResult:
        total_timer = CudaEventTimer().start()

        goal_tool_poses = goal_tool_poses.reorder_links(self.tool_frames)

        # Validate input shapes
        if goal_tool_poses.batch_size != batch_size:
            log_and_raise(f"expects batch size {batch_size}, got {goal_tool_poses.batch_size}")

        if seed_config is None and current_state is not None:
            seed_config = current_state.position.view(batch_size, 1, self.dof)

        # Ensure buffers are allocated before updating them.
        # _setup_batch_size is a no-op if size hasn't changed.
        self._setup_batch_size(batch_size, self.config.num_seeds)

        # Update velocity/accel state buffers in-place (addresses stay stable for CUDA graphs).
        self._velocity_dt.fill_(1.0)
        self._velocity_current_position.zero_()
        self._velocity_current_velocity.zero_()
        self._velocity_clamping_active = False
        if current_state is not None and current_state.dt is not None:
            self._velocity_clamping_active = True
            num_seeds = self.config.num_seeds
            self._velocity_current_position.copy_(
                tensor_repeat_seeds(
                    current_state.position.view(batch_size, 1, self.dof), num_seeds
                ).view(batch_size * num_seeds, self.dof)
            )
            dt_per_batch = current_state.dt.reshape(batch_size, -1)[:, 0]
            self._velocity_dt.copy_(dt_per_batch.repeat_interleave(num_seeds))
            if current_state.velocity is not None:
                self._velocity_current_velocity.copy_(
                    tensor_repeat_seeds(
                        current_state.velocity.view(batch_size, 1, self.dof), num_seeds
                    ).view(batch_size * num_seeds, self.dof)
                )

        # Generate seed configurations
        seeds = self._generate_seed_configs(
            batch_size=batch_size, seed_config=seed_config
        )  # (batch_size, num_seeds, dof)

        # solve mini batches here:
        batches_per_mini_batch = self.config.max_problems_mini_batch // self.config.num_seeds

        n_mini_batches = batch_size // batches_per_mini_batch
        solve_timer = CudaEventTimer().start()

        # Store full-batch velocity state for mini-batch slicing
        full_velocity_current_position = self._velocity_current_position.clone()
        full_velocity_current_velocity = self._velocity_current_velocity.clone()
        full_velocity_dt = self._velocity_dt.clone()

        if n_mini_batches > 0:
            solution_list = []
            success_list = []
            position_error_list = []
            orientation_error_list = []
            n_iterations_list = []
            start_idx = 0
            end_idx = batches_per_mini_batch
            mini_batch_seeds_buffer = seeds[start_idx:end_idx].clone()
            mini_batch_goal_tool_poses_buffer = GoalToolPose(
                tool_frames=self.tool_frames,
                position=goal_tool_poses.position[start_idx:end_idx].clone(),
                quaternion=goal_tool_poses.quaternion[start_idx:end_idx].clone(),
            )

            for i in range(n_mini_batches + 1):
                start_idx = i * batches_per_mini_batch
                end_idx = min(start_idx + batches_per_mini_batch, batch_size)
                valid_range = end_idx - start_idx

                mini_batch_seeds_buffer[:(valid_range), :, :] = seeds[start_idx:end_idx, :, :]
                mini_batch_goal_tool_poses_buffer.position[:(valid_range)] = (
                    goal_tool_poses.position[start_idx:end_idx]
                )
                mini_batch_goal_tool_poses_buffer.quaternion[:(valid_range)] = (
                    goal_tool_poses.quaternion[start_idx:end_idx]
                )

                # Copy velocity state slice for this mini-batch (in-place for CUDA graph stability)
                num_seeds = self.config.num_seeds
                seed_start = start_idx * num_seeds
                seed_end = end_idx * num_seeds
                num_problems = seed_end - seed_start
                self._velocity_current_position[:num_problems].copy_(
                    full_velocity_current_position[seed_start:seed_end]
                )
                self._velocity_dt[:num_problems].copy_(
                    full_velocity_dt[seed_start:seed_end]
                )
                self._velocity_current_velocity[:num_problems].copy_(
                    full_velocity_current_velocity[seed_start:seed_end]
                )

                solutions, successes, position_errors, orientation_errors, n_iterations = (
                    self._optimize(mini_batch_seeds_buffer, mini_batch_goal_tool_poses_buffer)
                )

                solution_list.append(solutions[:valid_range].clone())
                success_list.append(successes[:valid_range].clone())
                position_error_list.append(position_errors[:valid_range].clone())
                orientation_error_list.append(orientation_errors[:valid_range].clone())
                n_iterations_list.append(n_iterations)

            solutions = torch.cat(solution_list, dim=0)
            successes = torch.cat(success_list, dim=0)
            position_errors = torch.cat(position_error_list, dim=0)
            orientation_errors = torch.cat(orientation_error_list, dim=0)
            n_iterations = max(n_iterations_list)

        else:
            # solve the problems as a batch
            solutions, successes, position_errors, orientation_errors, n_iterations = (
                self._optimize(seeds, goal_tool_poses)
            )
        solve_time = solve_timer.stop()
        if self._batch_indices is None or self._batch_indices.shape[0] != batch_size:
            # Pre-allocate batch indices for advanced indexing
            self._batch_indices = (
                torch.arange(batch_size, device=self.device_cfg.device) * self.config.num_seeds
            )
        # Select top solutions
        top_successes, top_solutions, top_pos_errors, top_ori_errors = self._select_top_solutions(
            solutions,
            successes,
            position_errors,
            orientation_errors,
            current_state.position if current_state is not None else None,
            batch_size,
            self.config.num_seeds,
            return_seeds,
        )

        # Create joint state solution
        js_solution = JointState.from_position(top_solutions, joint_names=self.joint_names)

        # Create result
        result = IKSolverResult(
            success=top_successes,
            solution=top_solutions,
            js_solution=js_solution,
            position_error=top_pos_errors,
            rotation_error=top_ori_errors,
            goalset_index=torch.zeros_like(top_pos_errors, dtype=torch.long),  # Single goalset
            total_time=total_timer.stop(),
            solve_time=solve_time,
        )

        return result

    @profiler.record_function("seed_ik_solver/solve_single")
    def solve_single(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        seed_config: Optional[torch.Tensor] = None,
        return_seeds: int = 1,
    ) -> IKSolverResult:
        """Solve IK for a single problem with multiple possible goal poses.

        Args:
            goal_tool_poses: Target poses for controlled links
            seed_config: Optional seed joint configuration (1, dof)
            return_seeds: Number of best solutions to return

        Returns:
            IKSolverResult with solutions
        """
        result = self._solve_impl(
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
            seed_config=seed_config,
            return_seeds=return_seeds,
            batch_size=1,
        )
        return result

    @profiler.record_function("seed_ik_solver/solve_batch")
    def solve_batch(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        seed_config: Optional[torch.Tensor] = None,
        return_seeds: int = 1,
    ) -> IKSolverResult:
        """Solve IK for a batch of problems.

        Args:
            goal_tool_poses: Batch of target poses for controlled links
            seed_config: Optional seed joint configurations (batch, dof)
            return_seeds: Number of best solutions to return per problem

        Returns:
            IKSolverResult with solutions for each problem in the batch
        """
        goal_tool_poses = goal_tool_poses.reorder_links(self.tool_frames)
        result = self._solve_impl(
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
            seed_config=seed_config,
            return_seeds=return_seeds,
            batch_size=goal_tool_poses.batch_size,
        )
        return result

    @property
    def joint_limits(self):
        """Get robot joint limits."""
        return self._robot_model.get_joint_limits()

    def get_default_joint_position(self) -> torch.Tensor:
        """Get robot default joint position."""
        return self.default_joint_position.clone()

    @property
    def kinematics(self) -> Kinematics:
        """Get robot kinematics model."""
        return self._aux_robot_model

    def compute_kinematics(self, joint_position: JointState) -> CudaKinematicsState:
        """Compute robot kinematics."""
        return self.kinematics.compute_kinematics(joint_position)

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        self.error_calculator.update_tool_pose_criteria(tool_pose_criteria)

    def reset_seed(self):
        self.act_sample_gen.reset()

    def destroy(self):
        """Release CUDA graph resources held by this solver."""
        if self._initial_state_executor is not None:
            self._initial_state_executor.reset()
            self._initial_state_executor = None
        if self._inner_iterations_executor is not None:
            self._inner_iterations_executor.reset()
            self._inner_iterations_executor = None

