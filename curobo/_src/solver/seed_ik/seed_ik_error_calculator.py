# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Error and Jacobian Calculator for Seed IK Solver.

This module implements unified error and jacobian calculations for the
Levenberg-Marquardt based Seed IK solver, handling pose errors
and joint limit violations.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.state.state_joint import JointState
from curobo._src.cost.cost_tool_pose_cfg import (
    ToolPoseCostCfg,
)
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.transform import quaternion_rate_to_axis_angle_rate
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.cuda_stream_util import (
    create_cuda_stream_pair,
    cuda_stream_context,
    synchronize_cuda_streams,
)
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.torch_util import get_torch_jit_decorator


@dataclass
class ErrorJacobianResult:
    """Container for error and jacobian calculation results."""

    position_errors: torch.Tensor
    orientation_errors: torch.Tensor
    jTerror: torch.Tensor
    jacobian: torch.Tensor
    error_norm: torch.Tensor
    joint_position: torch.Tensor


class SeedIKErrorCalculator:
    """Unified calculator for IK error types and jacobians (pose + joint limits)."""

    def __init__(self, robot_model, config, action_min, action_max, device_cfg):
        self.robot_model = robot_model
        self.config = config
        self.device_cfg = device_cfg
        self.action_min = action_min
        self.action_max = action_max
        self.velocity_limits = robot_model.get_joint_limits().velocity
        self.num_links = len(robot_model.tool_frames)
        self.dof = robot_model.get_dof()
        self._batch_size = -1
        self._num_seeds = -1
        self._num_problems = -1
        self._cost_shape = None

        # Setup pose cost function internally
        self.pose_cost = self._setup_cost_function()

        # Cache for cost_shape tensors based on batch_size
        self._cost_shape_cache = {}

        # Initialize CUDA streams and events for parallel computation
        self._streams = {}
        self._events = {}
        self._streams["pose_residual"], self._events["pose_residual"] = create_cuda_stream_pair(
            self.device_cfg.device
        )
        self._streams["joint_limit_residual"], self._events["joint_limit_residual"] = (
            create_cuda_stream_pair(self.device_cfg.device)
        )
        self._streams["acceleration_residual"], self._events["acceleration_residual"] = (
            create_cuda_stream_pair(self.device_cfg.device)
        )
        self._streams["velocity_residual"], self._events["velocity_residual"] = (
            create_cuda_stream_pair(self.device_cfg.device)
        )

    def setup_batch_tensors(self, batch_size: int, num_seeds: int = 1):
        """Setup the batch size for the solver."""
        if batch_size != self._batch_size or num_seeds != self._num_seeds:
            log_info(f"Setting batch size to {batch_size}")
            self._batch_size = batch_size
            self._num_seeds = num_seeds
            self._num_problems = batch_size * num_seeds
            self._cost_shape = torch.ones(
                (self._num_problems, 1, 2 * self.num_links),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self.pose_cost.setup_batch_tensors(self._num_problems, 1)


    def _setup_cost_function(self):
        """Setup the multi-link pose cost function."""
        # Create link pose criteria for all controlled links
        tool_pose_criteria = {}
        for link_name in self.robot_model.tool_frames:
            tool_pose_criteria[link_name] = ToolPoseCriteria(
                terminal_pose_convergence_tolerance=[0.0, 0.0],
                terminal_pose_axes_weight_factor=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                device_cfg=self.device_cfg,
            )

        # Create cost configuration
        cost_config = ToolPoseCostCfg(
            weight=[self.config.position_weight, self.config.orientation_weight],
            tool_frames=self.robot_model.tool_frames,
            tool_pose_criteria=tool_pose_criteria,
            device_cfg=self.device_cfg,
            use_lie_group=False,
        )

        # Create and return cost function
        return ToolPoseCost(cost_config)

    @profiler.record_function("seed_ik_error_calculator/compute_all_errors")
    def compute_error_and_jacobian(
        self,
        joint_position: torch.Tensor,
        goal_poses: GoalToolPose,
        idxs_goal: torch.Tensor,
        current_position: Optional[torch.Tensor] = None,
        current_velocity: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        velocity_clamping_active: bool = False,
    ) -> ErrorJacobianResult:
        """Compute all error types and jacobians in one unified calculation.

        Uses cuda_stream_context to automatically handle stream-based parallelism
        when CUDA streams are enabled. When disabled, the stream contexts become no-ops.

        Args:
            joint_position: Joint configurations. Shape (num_problems, dof).
            goal_poses: Target tool poses.
            idxs_goal: Goal indices.
            current_position: Current joint position for velocity clamping. Shape (num_problems, dof).
            current_velocity: Current joint velocity for acceleration regularization.
                Shape (num_problems, dof). When None, acceleration cost is skipped.
            dt: Time step per problem for velocity clamping. Shape (num_problems,).
            velocity_clamping_active: When True, velocity clamping and regularization
                costs are active. When False, ``current_position``/``dt`` buffers are
                ignored for joint limit tightening even if they are non-None tensors.
        """
        num_problems = joint_position.shape[0]

        if num_problems != self._num_problems:
            log_and_raise(f"num_problems size mismatch: {num_problems} != {self._num_problems}")

        # Compute pose errors in stream (no-op context when streams disabled)
        with self.stream_context("pose_residual"):
            pose_jTerror, pose_jacobian, position_errors, orientation_errors, pose_error_norm = (
                self._compute_pose_errors(joint_position, goal_poses, idxs_goal, num_problems)
            )

        # Compute joint limit errors in stream (no-op context when streams disabled)
        with self.stream_context("joint_limit_residual"):
            joint_limit_jTerror, joint_limit_jacobian, joint_limit_error = (
                self._compute_joint_limit_errors(
                    joint_position, num_problems,
                    current_position=current_position,
                    dt=dt,
                    velocity_clamping_active=velocity_clamping_active,
                )
            )


        # Always compute velocity/accel errors when weights > 0 for CUDA graph consistency.
        # When dt=0, the clamped dt produces near-zero residuals safely.
        vel_jTerror = None
        vel_jacobian = None
        vel_error_norm = None
        accel_jTerror = None
        accel_jacobian = None
        accel_error_norm = None
        if self.config.velocity_weight > 0:
            with self.stream_context("velocity_residual"):
                vel_jTerror, vel_jacobian, vel_error_norm = (
                    self._compute_velocity_errors(
                        joint_position, current_position, dt, num_problems,
                    )
                )


        if self.config.acceleration_weight > 0:
            with self.stream_context("acceleration_residual"):
                accel_jTerror, accel_jacobian, accel_error_norm = (
                    self._compute_acceleration_errors(
                        joint_position, current_position, current_velocity, dt, num_problems,
                    )
                )


        # Synchronize streams (no-op when streams disabled)
        synchronize_cuda_streams(self._events, self.device_cfg.device)

        # Combine errors in current stream
        total_jTerror, total_jacobian, total_error_norm = self._combine_errors(
            pose_jTerror,
            pose_jacobian,
            pose_error_norm,
            joint_limit_jTerror,
            joint_limit_jacobian,
            joint_limit_error,
            vel_jTerror,
            vel_jacobian,
            vel_error_norm,
            accel_jTerror,
            accel_jacobian,
            accel_error_norm,
        )

        return ErrorJacobianResult(
            position_errors=position_errors,
            orientation_errors=orientation_errors,
            jTerror=total_jTerror,
            jacobian=total_jacobian,
            error_norm=total_error_norm,
            joint_position=joint_position.detach(),
        )

    @profiler.record_function("seed_ik_error_calculator/compute_pose_errors")
    def _compute_pose_errors(
        self,
        joint_position: torch.Tensor,
        goal_poses: GoalToolPose,
        idxs_goal: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pose errors and jacobians."""
        num_problems = joint_position.shape[0]
        if num_problems != self._num_problems:
            log_and_raise(f"num_problems size mismatch: {num_problems} != {self._num_problems}")
        cost_shape = self._cost_shape
        if self.config.use_backward:
            joint_position = joint_position.detach().requires_grad_(True)

        # Forward kinematics
        kin_state = self.robot_model.compute_kinematics(
            JointState.from_position(joint_position, joint_names=self.robot_model.joint_names)
        )
        num_links = len(kin_state.tool_frames)

        current_poses = ToolPose(
            tool_frames=kin_state.tool_frames,
            position=kin_state.tool_poses.position.view(batch_size, 1, num_links, 3),
            quaternion=kin_state.tool_poses.quaternion.view(batch_size, 1, num_links, 4),
        )

        # Set goal poses to require gradients for automatic differentiation
        if not self.config.use_backward:
            current_poses.detach().requires_grad_(True)

        # Compute cost and get gradients
        cost, linear_distance, angular_distance, goalset_idx = self.pose_cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        # Trigger backward pass
        cost.backward(cost_shape)

        # Get jacobian from kinematics
        jacobian = kin_state.tool_jacobians.view(batch_size, -1, self.dof).detach()

        # Reduce errors
        position_errors, orientation_errors, error_norm = self._reduce_pose_errors(
            linear_distance, angular_distance, cost, batch_size
        )

        # Compute jacobian transpose times error
        if self.config.use_backward:
            jTerror = joint_position.grad.view(batch_size, self.dof)
        else:
            jTerror = self._compute_analytical_pose_jTerror(current_poses, jacobian, batch_size)

        return jTerror, jacobian, position_errors, orientation_errors, error_norm

    @profiler.record_function("seed_ik_error_calculator/reduce_pose_errors")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _reduce_pose_errors(
        self,
        position_errors: torch.Tensor,
        orientation_errors: torch.Tensor,
        cost: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce pose errors to scalar values per batch."""
        position_errors = torch.max(position_errors.squeeze(1), dim=-1)[0]
        orientation_errors = torch.max(orientation_errors.squeeze(1), dim=-1)[0]
        error_norm = torch.sum(cost.view(batch_size, -1), dim=-1)
        return position_errors, orientation_errors, error_norm

    @profiler.record_function("seed_ik_error_calculator/compute_analytical_pose_jTerror")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def _compute_analytical_pose_jTerror(
        self,
        current_poses: ToolPose,
        jacobian: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute analytical jacobian transpose times pose error."""
        position_residual = current_poses.position.grad.view(batch_size, self.num_links, 3)
        quaternion_residual = current_poses.quaternion.grad.view(batch_size, self.num_links, 4)
        current_quaternion = current_poses.quaternion.view(batch_size, self.num_links, 4).detach()

        # Convert quaternion residual to axis-angle
        axis_angle_residual = quaternion_rate_to_axis_angle_rate(
            quaternion_residual,
            current_quaternion,  # .clone(),
        )

        # Combine residuals
        residual = torch.cat(
            [
                position_residual.view(batch_size, self.num_links, 3),
                axis_angle_residual.view(batch_size, self.num_links, 3),
            ],
            dim=-1,
        ).view(batch_size, -1)

        # Compute J^T * residual
        jTerror = jacobian.transpose(-2, -1) @ residual.view(batch_size, -1, 1)
        return jTerror.squeeze(-1)

    @profiler.record_function("seed_ik_error_calculator/add_joint_limit_errors")
    def _compute_joint_limit_errors(
        self,
        joint_position: torch.Tensor,
        batch_size: int,
        current_position: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        velocity_clamping_active: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add joint limit violation errors.

        When ``velocity_clamping_active`` is True and ``current_position``/``dt``
        are provided, bounds are tightened to
        ``[max(action_min, current + v_lower*dt), min(action_max, current + v_upper*dt)]``.
        """
        action_min = self.action_min
        action_max = self.action_max

        if velocity_clamping_active and current_position is not None and dt is not None:
            v_lower = self.velocity_limits[0]  # (dof,) negative
            v_upper = self.velocity_limits[1]  # (dof,) positive
            dt_col = dt.unsqueeze(-1)  # (batch, 1)
            action_min = torch.max(self.action_min.unsqueeze(0), current_position + v_lower * dt_col)
            action_max = torch.min(self.action_max.unsqueeze(0), current_position + v_upper * dt_col)

        # Calculate violations
        upper_violation = torch.clamp(joint_position - action_max, min=0.0)
        lower_violation = torch.clamp(action_min - joint_position, min=0.0)

        # Combined joint limit errors
        joint_limit_errors = self.config.joint_limit_weight * (lower_violation + upper_violation)

        # Combined jacobian diagonal
        # Derivative: d/dq[violation] = -1 for lower, +1 for upper
        joint_limit_jacobian_diag = self.config.joint_limit_weight * (
            torch.where(lower_violation > 0, -1.0, 0.0) + torch.where(upper_violation > 0, 1.0, 0.0)
        )

        # Add to jTerror (for diagonal jacobian, jTerror = diag * error)
        jTlimit_error = joint_limit_jacobian_diag * joint_limit_errors

        # Add joint limit jacobian
        joint_limit_jacobian = torch.diag_embed(joint_limit_jacobian_diag)

        joint_limit_error = torch.sum(joint_limit_errors.view(batch_size, -1), dim=-1)

        # Add to error norm

        return jTlimit_error, joint_limit_jacobian, joint_limit_error

    @profiler.record_function("seed_ik_error_calculator/compute_velocity_errors")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _compute_velocity_errors(
        self,
        joint_position: torch.Tensor,
        current_position: torch.Tensor,
        dt: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute velocity regularization errors.

        The residual is ``r = sqrt(w * dt) * v_implied`` where
        ``v_implied = (q - q_current) / dt``. This gives:

        - ``||r||^2 = w * dt * v^2``  (matches LBFGS cost scaling)
        - ``J^T r  = w * v``          (gradient invariant to dt choice)

        When dt=0 (no velocity state), all outputs are zero.
        """
        dt_col = torch.clamp(dt.unsqueeze(-1), min=1e-10)
        sqrt_w_dt = (self.config.velocity_weight * dt_col) ** 0.5
        inv_dt = 1.0 / dt_col

        v_implied = (joint_position - current_position) * inv_dt

        vel_error = sqrt_w_dt * v_implied
        vel_jac_diag = (sqrt_w_dt * inv_dt).expand_as(joint_position)

        jT_vel = vel_jac_diag * vel_error
        vel_jacobian = torch.diag_embed(vel_jac_diag)
        vel_error_norm = torch.sum(vel_error * vel_error, dim=-1)

        return jT_vel, vel_jacobian, vel_error_norm

    @profiler.record_function("seed_ik_error_calculator/compute_acceleration_errors")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _compute_acceleration_errors(
        self,
        joint_position: torch.Tensor,
        current_position: torch.Tensor,
        current_velocity: torch.Tensor,
        dt: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute acceleration regularization errors.

        The residual is defined as ``r = sqrt(w) * (v_implied - v_current)``
        where ``v_implied = (q - q_current) / dt``. This gives:

        - ``||r||^2 = w * dt^2 * a^2``  (matches LBFGS cost scaling)
        - ``J^T r  = w * a``            (gradient invariant to dt choice)

        Args:
            joint_position: Candidate joint configuration. Shape (batch_size, dof).
            current_position: Current joint position. Shape (batch_size, dof).
            current_velocity: Current joint velocity. Shape (batch_size, dof).
            dt: Time step per problem. Shape (batch_size,).
            batch_size: Number of problems.
        """
        sqrt_w = self.config.acceleration_weight ** 0.5
        inv_dt = 1.0 / torch.clamp(dt.unsqueeze(-1), min=1e-10)

        v_implied = (joint_position - current_position) * inv_dt

        # r = sqrt(w) * (v_implied - v_current) = sqrt(w) * dt * a
        accel_error = sqrt_w * (v_implied - current_velocity)
        # dr/dq = sqrt(w) / dt
        accel_jac_diag = (sqrt_w * inv_dt).expand_as(joint_position)

        jT_accel = accel_jac_diag * accel_error
        accel_jacobian = torch.diag_embed(accel_jac_diag)
        accel_error_norm = torch.sum(accel_error * accel_error, dim=-1)

        return jT_accel, accel_jacobian, accel_error_norm

    @profiler.record_function("seed_ik_error_calculator/combine_errors")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def _combine_errors(
        self,
        pose_jTerror,
        pose_jacobian,
        pose_error_norm,
        joint_limit_jTerror,
        joint_limit_jacobian,
        joint_limit_error,
        vel_jTerror=None,
        vel_jacobian=None,
        vel_error_norm=None,
        accel_jTerror=None,
        accel_jacobian=None,
        accel_error_norm=None,
    ):
        jTerror = pose_jTerror + joint_limit_jTerror

        jacobian_parts = [pose_jacobian, joint_limit_jacobian]
        error_norm = pose_error_norm + joint_limit_error

        if vel_jTerror is not None:
            jTerror = jTerror + vel_jTerror
            jacobian_parts.append(vel_jacobian)
            error_norm = error_norm + vel_error_norm

        if accel_jTerror is not None:
            jTerror = jTerror + accel_jTerror
            jacobian_parts.append(accel_jacobian)
            error_norm = error_norm + accel_error_norm

        jacobian = torch.cat(jacobian_parts, dim=1)
        return jTerror, jacobian, error_norm

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        self.pose_cost.update_tool_pose_criteria(tool_pose_criteria)

    def stream_context(self, stream_name: str):
        """Context manager for computation with optional CUDA streams.

        This is a convenience wrapper around cuda_util.cuda_stream_context.

        Args:
            stream_name: Name of the stream ("pose_residual" or "joint_limit_residual")

        Yields:
            Context for the computation

        Example:
            >>> with self.stream_context("pose_residual"):
            ...     result = compute_something()
        """
        return cuda_stream_context(
            stream_name, self._streams, self._events, self.device_cfg.device
        )

