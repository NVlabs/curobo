# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iteration State Manager for Seed IK Solver.

This module handles the logic for updating iteration states in the LM algorithm,
including step acceptance, damping parameter updates, and convergence checking.
"""

from __future__ import annotations

from dataclasses import dataclass

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo._src.solver.seed_ik.seed_ik_state import SeedIKState
from curobo._src.util.torch_util import get_torch_jit_decorator


class SeedIterationStateManager:
    """Manages iteration state updates for Levenberg-Marquardt optimization.

    This class encapsulates the logic for:
    - Step acceptance based on trust region ratio
    - Damping parameter updates
    - State value selection
    - Convergence checking

    Following the Single Responsibility Principle, this class focuses solely
    on iteration state management, making it easier to test and maintain.
    """

    # Constants
    EPSILON_DIVISION_SAFETY = 1e-8

    def __init__(
        self,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        rho_min: float,
        lambda_factor: float,
        lambda_min: float,
        lambda_max: float,
        convergence_position_tolerance: float,
        convergence_orientation_tolerance: float,
        convergence_joint_limit_weight: float,
    ):
        """Initialize the iteration state manager.

        Args:
            action_min: Minimum joint limits
            action_max: Maximum joint limits
            rho_min: Minimum trust region ratio for step acceptance
            lambda_factor: Factor for damping parameter updates
            lambda_min: Minimum damping parameter value
            lambda_max: Maximum damping parameter value
            convergence_position_tolerance: Position error tolerance for convergence
            convergence_orientation_tolerance: Orientation error tolerance for convergence
            convergence_joint_limit_weight: Weight for joint limit convergence check
        """
        self.action_min = action_min
        self.action_max = action_max
        self.rho_min = rho_min
        self.lambda_factor = lambda_factor
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.convergence_position_tolerance = convergence_position_tolerance
        self.convergence_orientation_tolerance = convergence_orientation_tolerance
        self.convergence_joint_limit_weight = convergence_joint_limit_weight

    @profiler.record_function("iteration_state_manager/update_state")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def update_iteration_state(
        self,
        current_state: SeedIKState,
        candidate_state: SeedIKState,
        predicted_reduction: torch.Tensor,
        batch_size: int,
    ) -> SeedIKState:
        """Update iteration state based on step acceptance criteria.

        Args:
            current_state: Current iteration state
            candidate_state: Candidate new state from LM step
            predicted_reduction: Predicted cost reduction from linear model
            batch_size: Number of problems in batch

        Returns:
            Updated iteration state
        """
        # Calculate step acceptance
        trust_ratio = self._calculate_trust_region_ratio(
            current_state.error_norm, candidate_state.error_norm, predicted_reduction, batch_size
        )

        step_accepted = self._determine_step_acceptance(trust_ratio, batch_size)

        # Update components
        updated_damping = self._update_damping_parameter(
            current_state.lambda_damping, step_accepted, batch_size
        )

        selected_values = self._select_state_values(current_state, candidate_state, step_accepted)

        convergence_status = self._check_convergence(
            selected_values.joint_position,
            selected_values.position_errors,
            selected_values.orientation_errors,
        )

        # Construct updated state
        return SeedIKState(
            joint_position=selected_values.joint_position,
            lambda_damping=updated_damping,
            error_norm=candidate_state.error_norm,
            success=convergence_status,
            improvement=step_accepted,
            position_errors=selected_values.position_errors,
            orientation_errors=selected_values.orientation_errors,
            jTerror=selected_values.jTerror,
            jacobian=selected_values.jacobian,
        )

    @profiler.record_function("iteration_state_manager/calculate_trust_ratio")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _calculate_trust_region_ratio(
        self,
        old_error_norm: torch.Tensor,
        new_error_norm: torch.Tensor,
        predicted_reduction: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Calculate trust region ratio for step acceptance.

        The trust region ratio compares actual vs predicted reduction:
        rho = (actual_reduction) / (predicted_reduction)
        """
        actual_reduction = old_error_norm - new_error_norm
        safe_predicted_reduction = (
            predicted_reduction.view(batch_size) + self.EPSILON_DIVISION_SAFETY
        )
        return actual_reduction / safe_predicted_reduction

    @profiler.record_function("iteration_state_manager/determine_step_acceptance")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _determine_step_acceptance(
        self,
        trust_ratio: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Determine whether to accept the proposed step."""
        return (trust_ratio >= self.rho_min).view(batch_size)

    @profiler.record_function("iteration_state_manager/update_damping")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _update_damping_parameter(
        self,
        current_damping: torch.Tensor,
        step_accepted: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Update Levenberg-Marquardt damping parameter.

        Decrease damping for accepted steps (more Gauss-Newton-like),
        increase damping for rejected steps (more gradient descent-like).
        """
        acceptance_mask = step_accepted.view(batch_size, 1, 1)

        updated_damping = torch.where(
            acceptance_mask,
            current_damping / self.lambda_factor,
            current_damping * self.lambda_factor,
        )

        return torch.clamp(updated_damping, min=self.lambda_min, max=self.lambda_max)

    @dataclass
    class SelectedStateValues:
        """Container for selected state values."""

        joint_position: torch.Tensor
        jTerror: torch.Tensor
        jacobian: torch.Tensor
        position_errors: torch.Tensor
        orientation_errors: torch.Tensor

    @profiler.record_function("iteration_state_manager/select_values")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def _select_state_values(
        self,
        current_state: SeedIKState,
        candidate_state: SeedIKState,
        step_accepted: torch.Tensor,
    ) -> SelectedStateValues:
        """Select state values based on step acceptance.

        For accepted steps, use candidate values.
        For rejected steps, keep current values.
        """
        mask_1d = step_accepted.view(-1)
        mask_2d = step_accepted.view(-1, 1)
        mask_3d = step_accepted.view(-1, 1, 1)

        return self.SelectedStateValues(
            joint_position=torch.where(
                mask_2d, candidate_state.joint_position, current_state.joint_position
            ),
            jTerror=torch.where(mask_2d, candidate_state.jTerror, current_state.jTerror),
            jacobian=torch.where(mask_3d, candidate_state.jacobian, current_state.jacobian),
            position_errors=torch.where(
                mask_1d, candidate_state.position_errors, current_state.position_errors
            ),
            orientation_errors=torch.where(
                mask_1d, candidate_state.orientation_errors, current_state.orientation_errors
            ),
        )

    @profiler.record_function("iteration_state_manager/check_convergence")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _check_convergence(
        self,
        joint_position: torch.Tensor,
        position_errors: torch.Tensor,
        orientation_errors: torch.Tensor,
    ) -> torch.Tensor:
        """Check convergence based on pose and joint limit criteria."""
        pose_converged = self._check_pose_convergence(position_errors, orientation_errors)

        if self.convergence_joint_limit_weight <= 0:
            return pose_converged

        joint_limits_satisfied = self._check_joint_limit_satisfaction(joint_position)
        return torch.logical_and(pose_converged, joint_limits_satisfied)

    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _check_pose_convergence(
        self,
        position_errors: torch.Tensor,
        orientation_errors: torch.Tensor,
    ) -> torch.Tensor:
        """Check if pose errors are within tolerance."""
        return torch.logical_and(
            position_errors < self.convergence_position_tolerance,
            orientation_errors < self.convergence_orientation_tolerance,
        )

    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _check_joint_limit_satisfaction(
        self,
        joint_position: torch.Tensor,
    ) -> torch.Tensor:
        """Check if joint limits are satisfied."""
        within_limits = torch.logical_and(
            joint_position > self.action_min,
            joint_position < self.action_max,
        )
        return torch.all(within_limits, dim=-1)

