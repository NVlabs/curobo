# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iteration-level optimizer state recorder for debugging.

When enabled via ``store_debug``, captures actions, costs, and line-search
outcomes so convergence behavior can be inspected after optimization completes.
"""

from typing import Any, Dict, List

import torch

from curobo._src.optim.optimization_iteration_state import OptimizationIterationState


class DebugRecorder:
    """Records iteration state history for debugging."""

    def __init__(self) -> None:
        self.actions: List[torch.Tensor] = []
        self.costs: List[torch.Tensor] = []
        self.line_search_indices: List[torch.Tensor] = []
        self.line_search_statuses: List[torch.Tensor] = []
        self.curvature_update_valid: List[torch.Tensor] = []

    def record(
        self,
        iteration_state: OptimizationIterationState,
        action_horizon: int,
        action_dim: int,
    ) -> None:
        """Record one iteration snapshot."""
        action = iteration_state.action.view(-1, action_horizon, action_dim).clone()
        self.actions.append(action)
        if iteration_state.cost is not None:
            self.costs.append(iteration_state.cost.clone())
        for value, storage in (
            (iteration_state.line_search_index, self.line_search_indices),
            (iteration_state.line_search_status, self.line_search_statuses),
            (iteration_state.curvature_update_valid, self.curvature_update_valid),
        ):
            if value is not None:
                storage.append(value.clone())

    def clear(self) -> None:
        """Clear all recorded history."""
        self.actions = []
        self.costs = []
        self.line_search_indices = []
        self.line_search_statuses = []
        self.curvature_update_valid = []

    def get_trace(self) -> Dict[str, Any]:
        """Return recorded history as a dict."""
        return {
            "debug": self.actions,
            "debug_cost": self.costs,
            "debug_line_search_index": self.line_search_indices,
            "debug_line_search_status": self.line_search_statuses,
            "debug_curvature_update_valid": self.curvature_update_valid,
        }
