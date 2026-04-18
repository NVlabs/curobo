# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

from curobo._src.state.state_joint import JointState

# CuRobo
from curobo._src.util.logging import log_and_raise


@dataclass
class OptimizationIterationState:
    """Variables needed for an optimization iteration are stored here."""

    #: action tensor to optimize. Shape: (num_problems, action_horizon,  action_dim)
    action: torch.Tensor

    #: cost tensor. Shape: (num_problems)
    cost: Optional[torch.Tensor] = None

    #: gradient tensor. Shape: (num_problems, action_dim)
    gradient: Optional[torch.Tensor] = None

    #: exploration action tensor. Shape: (num_problems, action_horizon, action_dim)
    exploration_action: Optional[torch.Tensor] = None

    #: exploration gradient tensor. Shape: (num_problems, action_horizon, action_dim)
    exploration_gradient: Optional[torch.Tensor] = None

    #: exploration cost tensor. Shape: (num_problems)
    exploration_cost: Optional[torch.Tensor] = None

    #: step direction tensor. Shape: (num_problems, action_dim)
    step_direction: Optional[torch.Tensor] = None

    #: best action tensor. Shape: (num_problems, action_horizon, action_dim)
    best_action: Optional[torch.Tensor] = None

    #: best cost tensor. Shape: (num_problems)
    best_cost: Optional[torch.Tensor] = None

    #: best iteration tensor. Shape: (num_problems). Type: int16
    best_iteration: Optional[torch.Tensor] = None

    #: current iteration tensor. Shape: (num_problems). Type: int16
    current_iteration: Optional[torch.Tensor] = None

    #: state from rolling out action.
    state: Optional[Union[JointState, torch.Tensor]] = None

    #: converged tensor. Shape: (num_problems). Type: bool
    converged: Optional[torch.Tensor] = None

    jacobian: Optional[torch.Tensor] = None

    def data_ptr(self):
        return self.action.data_ptr()

    @profiler.record_function("iteration_state/post_init")
    def __post_init__(self):
        if len(self.action.shape) != 3:
            log_and_raise(
                f"Action tensor must have shape (num_problems, action_horizon, action_dim). Got {self.action.shape}"
            )
        if self.exploration_action is not None and len(self.exploration_action.shape) != 3:
            log_and_raise(
                f"Exploration action tensor must have shape (num_problems, action_horizon, action_dim). Got {self.exploration_action.shape}"
            )
        if self.exploration_gradient is not None and len(self.exploration_gradient.shape) != 3:
            log_and_raise(
                f"Exploration gradient tensor must have shape (num_problems, action_horizon, action_dim). Got {self.exploration_gradient.shape}"
            )
        if self.exploration_cost is not None and len(self.exploration_cost.shape) != 1:
            log_and_raise(
                f"Exploration cost tensor must have shape (num_problems). Got {self.exploration_cost.shape}"
            )

        if self.best_action is not None and len(self.best_action.shape) != 3:
            log_and_raise(
                f"Best action tensor must have shape (num_problems, action_horizon, action_dim). Got {self.best_action.shape}"
            )
        if self.gradient is not None and len(self.gradient.shape) != 3:
            log_and_raise(
                f"Gradient tensor must have shape (num_problems, action_horizon, action_dim). Got {self.gradient.shape}"
            )

        if self.step_direction is not None and len(self.step_direction.shape) != 3:
            log_and_raise(
                f"Step direction tensor must have shape (num_problems, action_horizon, action_dim). Got {self.step_direction.shape}"
            )
        if self.cost is not None and len(self.cost.shape) != 1:
            log_and_raise(f"Cost tensor must have shape (num_problems). Got {self.cost.shape}")
        if self.best_cost is not None and len(self.best_cost.shape) != 1:
            log_and_raise(f"Best cost tensor must have shape (num_problems). Got {self.best_cost.shape}")
        if self.best_iteration is not None and len(self.best_iteration.shape) != 1:
            log_and_raise(
                f"Best iteration tensor must have shape (num_problems). Got {self.best_iteration.shape}"
            )
        if self.converged is not None and len(self.converged.shape) != 1:
            log_and_raise(f"Converged tensor must have shape (num_problems). Got {self.converged.shape}")

    @profiler.record_function("iteration_state/clone")
    def clone(self) -> OptimizationIterationState:
        """Clone the optimization iteration state."""
        return OptimizationIterationState(
            action=self.action.clone(),
            cost=self.cost.clone() if self.cost is not None else None,
            gradient=self.gradient.clone() if self.gradient is not None else None,
            exploration_action=(
                self.exploration_action.clone() if self.exploration_action is not None else None
            ),
            exploration_gradient=(
                self.exploration_gradient.clone() if self.exploration_gradient is not None else None
            ),
            exploration_cost=(
                self.exploration_cost.clone() if self.exploration_cost is not None else None
            ),
            step_direction=self.step_direction.clone() if self.step_direction is not None else None,
            best_action=self.best_action.clone() if self.best_action is not None else None,
            best_cost=self.best_cost.clone() if self.best_cost is not None else None,
            best_iteration=self.best_iteration.clone() if self.best_iteration is not None else None,
            current_iteration=(
                self.current_iteration.clone() if self.current_iteration is not None else None
            ),
            state=self.state.clone() if self.state is not None else None,
            jacobian=self.jacobian.clone() if self.jacobian is not None else None,
            converged=self.converged.clone() if self.converged is not None else None,
        )

    @profiler.record_function("iteration_state/copy_")
    def copy_(self, other: OptimizationIterationState):
        """Copy the optimization iteration state from another instance.

        Only copies the variables that are not None in both instances.

        Args:
            other: The instance to copy from.
        """
        if self.action is not None and other.action is not None:
            if self.action.shape != other.action.shape:
                log_and_raise(f"Action shape mismatch: {self.action.shape} != {other.action.shape}")
            self.action.copy_(other.action)

        if self.cost is not None and other.cost is not None:
            if self.cost.shape != other.cost.shape:
                log_and_raise(f"Cost shape mismatch: {self.cost.shape} != {other.cost.shape}")
            self.cost.copy_(other.cost)
        if self.gradient is not None and other.gradient is not None:
            if self.gradient.shape != other.gradient.shape:
                log_and_raise(
                    f"Gradient shape mismatch: {self.gradient.shape} != {other.gradient.shape}"
                )
            self.gradient.copy_(other.gradient)

        if self.step_direction is not None and other.step_direction is not None:
            if self.step_direction.shape != other.step_direction.shape:
                log_and_raise(
                    f"Step direction shape mismatch: {self.step_direction.shape} != {other.step_direction.shape}"
                )
            self.step_direction.copy_(other.step_direction)

        if self.exploration_action is not None and other.exploration_action is not None:
            if self.exploration_action.shape != other.exploration_action.shape:
                log_and_raise(
                    f"Exploration action shape mismatch: {self.exploration_action.shape} != {other.exploration_action.shape}"
                )
            self.exploration_action.copy_(other.exploration_action)

        if self.exploration_gradient is not None and other.exploration_gradient is not None:
            if self.exploration_gradient.shape != other.exploration_gradient.shape:
                log_and_raise(
                    f"Exploration gradient shape mismatch: {self.exploration_gradient.shape} != {other.exploration_gradient.shape}"
                )
            self.exploration_gradient.copy_(other.exploration_gradient)

        if self.exploration_cost is not None and other.exploration_cost is not None:
            if self.exploration_cost.shape != other.exploration_cost.shape:
                log_and_raise(
                    f"Exploration cost shape mismatch: {self.exploration_cost.shape} != {other.exploration_cost.shape}"
                )
            self.exploration_cost.copy_(other.exploration_cost)

        if self.best_action is not None and other.best_action is not None:
            if self.best_action.shape != other.best_action.shape:
                log_and_raise(
                    f"Best action shape mismatch: {self.best_action.shape} != {other.best_action.shape}"
                )
            self.best_action.copy_(other.best_action)

        if self.best_cost is not None and other.best_cost is not None:
            if self.best_cost.shape != other.best_cost.shape:
                log_and_raise(
                    f"Best cost shape mismatch: {self.best_cost.shape} != {other.best_cost.shape}"
                )
            self.best_cost.copy_(other.best_cost)

        if self.best_iteration is not None and other.best_iteration is not None:
            if self.best_iteration.shape != other.best_iteration.shape:
                log_and_raise(
                    f"Best iteration shape mismatch: {self.best_iteration.shape} != {other.best_iteration.shape}"
                )
            self.best_iteration.copy_(other.best_iteration)

        if self.current_iteration is not None and other.current_iteration is not None:
            if self.current_iteration.shape != other.current_iteration.shape:
                log_and_raise(
                    f"Current iteration shape mismatch: {self.current_iteration.shape} != {other.current_iteration.shape}"
                )
            self.current_iteration.copy_(other.current_iteration)

        if self.converged is not None and other.converged is not None:
            if self.converged.shape != other.converged.shape:
                log_and_raise(
                    f"Converged shape mismatch: {self.converged.shape} != {other.converged.shape}"
                )
            self.converged.copy_(other.converged)

        if self.state is not None and other.state is not None:
            if self.state.shape != other.state.shape:
                log_and_raise(f"State shape mismatch: {self.state.shape} != {other.state.shape}")
            self.state.copy_(other.state)

        if self.jacobian is not None and other.jacobian is not None:
            if self.jacobian.shape != other.jacobian.shape:
                log_and_raise(f"Jacobian shape mismatch: {self.jacobian.shape} != {other.jacobian.shape}")
            self.jacobian.copy_(other.jacobian)
