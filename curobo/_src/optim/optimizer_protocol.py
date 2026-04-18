# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-checkable protocol defining the optimizer interface for cuRobo solvers.

Declares the properties (config, enabled, action_horizon, action_dim) and methods
(optimize, reinitialize, shift, lifecycle hooks) that MultiStageOptimizer and
solver layers depend on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import torch


@runtime_checkable
class Optimizer(Protocol):
    """Runtime-checkable optimizer interface expected by solvers and MultiStageOptimizer.

    Defines configuration access, action shape properties, core optimization methods,
    lifecycle hooks for resizing and rollout updates, and debug/state-management helpers.
    """

    # -- Properties --

    @property
    def config(self) -> Any:
        """Optimizer configuration."""
        ...

    @property
    def enabled(self) -> bool:
        """Whether this optimizer is active."""
        ...

    @property
    def action_horizon(self) -> int:
        """Number of timesteps in the action sequence."""
        ...

    @property
    def action_dim(self) -> int:
        """Dimensionality of each action."""
        ...

    # -- Core --

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        """Run optimization from a seed action and return the best result.

        Performs one full optimization pass (all configured iterations)
        starting from the provided seed.  The optimizer evaluates the
        rollout, updates its distribution or gradient state, and returns
        the best action sequence found.

        Args:
            seed_action: Initial action tensor of shape
                ``(batch, action_horizon, action_dim)``.

        Returns:
            Optimized action tensor with the same shape as
            ``seed_action``.
        """
        ...

    def reinitialize(
        self,
        action: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        clear_optimizer_state: bool = True,
        reset_num_iters: bool = False,
    ) -> None:
        """Reinitialize the optimizer distribution or state at a given action.

        Used between successive solves or MPC steps to warm-start from a
        known trajectory while optionally resetting internal statistics.

        Args:
            action: Action tensor of shape
                ``(batch, action_horizon, action_dim)`` to center the
                distribution on.
            mask: Bool tensor of shape ``(batch,)`` selecting which
                problems to reinitialize.  All problems when ``None``.
            clear_optimizer_state: If ``True``, reset momentum / running
                statistics accumulated from prior iterations.
            reset_num_iters: If ``True``, reset the iteration counter
                back to zero.
        """
        ...

    def shift(self, shift_steps: int = 0) -> None:
        """Shift the action distribution forward for MPC warm-starting.

        Rolls the mean/best action sequence by ``shift_steps`` timesteps
        and pads the tail with the last valid action.

        Args:
            shift_steps: Number of timesteps to shift forward.
        """
        ...

    # -- Lifecycle --

    def update_num_problems(self, num_problems: int) -> None:
        """Resize internal buffers for a new number of parallel problems.

        Called when the batch size changes between solves.  Re-allocates
        particle buffers, rollout batch tensors, and cost accumulators.

        Args:
            num_problems: New number of independent problems to solve
                in parallel.
        """
        ...

    def update_rollout_params(self, goal: Any) -> None:
        """Update goal parameters in all owned rollout instances.

        Args:
            goal: Goal registry (typically :class:`GoalRegistry`) passed
                through to each rollout's :meth:`update_params`.
        """
        ...

    def get_all_rollout_instances(self) -> list:
        """Return all rollout instances owned by this optimizer."""
        ...

    def compute_metrics(self, action: torch.Tensor) -> Any:
        """Compute full metrics (costs, constraints, convergence) for an action.

        Args:
            action: Action tensor of shape
                ``(batch, action_horizon, action_dim)``.

        Returns:
            :class:`RolloutMetrics` from the metrics rollout path.
        """
        ...

    def reset_shape(self) -> None:
        """Reset shape-dependent internal buffers."""
        ...

    def reset_seed(self) -> None:
        """Reset random number generator seeds."""
        ...

    def reset_cuda_graph(self) -> None:
        """Reset CUDA graph executors so they re-record on next call."""
        ...

    # -- Debug / config --

    def get_recorded_trace(self) -> Dict[str, Any]:
        """Return per-iteration debug history from the last optimization.

        Returns:
            Dict mapping trace keys (e.g. ``"cost"``, ``"best_cost"``)
            to lists of per-iteration values.
        """
        ...

    def update_solver_params(self, solver_params: Dict[str, Dict[str, Any]]) -> bool:
        """Update optimizer parameters by name from a nested dict.

        Args:
            solver_params: Mapping of optimizer name to a dict of
                parameter names and values to update.

        Returns:
            ``True`` if any parameters were updated.
        """
        ...

    def enable(self) -> None:
        """Enable this optimizer."""
        ...

    def disable(self) -> None:
        """Disable this optimizer."""
        ...
