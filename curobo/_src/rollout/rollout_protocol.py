# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural typing interface that optimizers and solvers use to drive rollouts.

Defines the properties and methods every rollout must expose (action bounds,
horizon, forward pass, CUDA-graph replay, etc.).  Data types used in the
interface (RolloutResult, RolloutMetrics, CostsAndConstraints, CostCollection)
are re-exported from metrics.py for convenience.
"""

from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

import torch

from curobo._src.rollout.metrics import (
    CostCollection,
    CostsAndConstraints,
    RolloutMetrics,
    RolloutResult,
)
from curobo._src.state.state_joint import JointState

__all__ = [
    "Rollout",
    "RolloutResult",
    "RolloutMetrics",
    "CostsAndConstraints",
    "CostCollection",
]


@runtime_checkable
class Rollout(Protocol):
    """Structural typing interface for rollout classes used by optimizers and solvers.

    Declares action-space properties, forward-pass entry points, CUDA-graph
    replay hooks, and goal/start-state management that every rollout must
    provide.
    """

    # -- Properties --

    @property
    def action_dim(self) -> int:
        """Number of actuated joints (dimensionality of each action)."""
        ...

    @property
    def action_horizon(self) -> int:
        """Number of timesteps in an action sequence."""
        ...

    @property
    def action_bound_lows(self) -> torch.Tensor:
        """Lower joint-position limits, shape ``(action_dim,)``."""
        ...

    @property
    def action_bound_highs(self) -> torch.Tensor:
        """Upper joint-position limits, shape ``(action_dim,)``."""
        ...

    @property
    def dt(self) -> float:
        """Integration timestep in seconds."""
        ...

    @property
    def sum_horizon(self) -> bool:
        """Whether costs are summed across the horizon before returning."""
        ...

    # -- Core --

    def evaluate_action(self, act_seq: torch.Tensor, **kwargs) -> RolloutResult:
        """Forward-simulate an action sequence and return costs/constraints.

        This is the primary entry point used by the optimizer during each
        iteration.

        Args:
            act_seq: Action tensor of shape
                ``(batch, action_horizon, action_dim)``.

        Returns:
            :class:`RolloutResult` with actions, state trajectory, and
            aggregated costs and constraints.
        """
        ...

    def compute_metrics_from_state(
        self, state: JointState, **kwargs
    ) -> RolloutMetrics:
        """Evaluate costs, constraints, and convergence for a state trajectory.

        Args:
            state: Joint state with position/velocity of shape
                ``(batch, horizon, n_dof)``.

        Returns:
            :class:`RolloutMetrics` including feasibility and convergence.
        """
        ...

    def compute_metrics_from_action(
        self, act_seq: torch.Tensor, **kwargs
    ) -> RolloutMetrics:
        """Forward-simulate actions and compute full metrics with convergence.

        Args:
            act_seq: Action tensor of shape
                ``(batch, action_horizon, action_dim)``.

        Returns:
            :class:`RolloutMetrics` with the ``actions`` field populated.
        """
        ...

    # -- Lifecycle --

    def update_params(self, **kwargs) -> bool:
        """Update goal targets and internal parameters for the next solve.

        Returns:
            ``True`` if the update succeeded.
        """
        ...

    def update_batch_size(self, batch_size: int) -> None:
        """Resize internal buffers to match a new batch size.

        Args:
            batch_size: Number of parallel problems (particles).
        """
        ...

    def update_dt(
        self, dt: Union[float, torch.Tensor], **kwargs
    ) -> bool:
        """Change the integration timestep for the transition model.

        Args:
            dt: New timestep in seconds (scalar or per-problem tensor).

        Returns:
            ``True`` if the update succeeded.
        """
        ...

    def reset(
        self, reset_problem_ids: Optional[torch.Tensor] = None, **kwargs
    ) -> bool:
        """Reset cost-manager state for specified or all problems.

        Args:
            reset_problem_ids: Int tensor of problem indices to reset.
                Resets all problems when ``None``.

        Returns:
            ``True`` if the reset succeeded.
        """
        ...

    def reset_shape(self) -> bool:
        """Clear cached goal registries so they are rebuilt on next update.

        Returns:
            ``True`` if the reset succeeded.
        """
        ...

    def reset_seed(self) -> None:
        """Reset the Halton sampler to its initial state."""
        ...
