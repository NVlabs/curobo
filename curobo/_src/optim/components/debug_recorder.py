# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iteration-level action and cost history recorder for optimizer debugging.

When enabled via ``store_debug``, captures cloned snapshots of the action tensor
and cost at each iteration so they can be inspected after optimization completes.
"""

from typing import Any, Dict, List

import torch

from curobo._src.optim.optimization_iteration_state import OptimizationIterationState


class DebugRecorder:
    """Records iteration state history for debugging."""

    def __init__(self):
        self.actions: List[torch.Tensor] = []
        self.costs: List[torch.Tensor] = []

    def record(self, iteration_state: OptimizationIterationState,
               action_horizon: int, action_dim: int):
        """Record one iteration snapshot."""
        action = iteration_state.action.view(-1, action_horizon, action_dim).clone()
        self.actions.append(action)
        if iteration_state.cost is not None:
            self.costs.append(iteration_state.cost.clone())

    def clear(self):
        """Clear all recorded history."""
        self.actions = []
        self.costs = []

    def get_trace(self) -> Dict[str, Any]:
        """Return recorded history as a dict."""
        return {"debug": self.actions, "debug_cost": self.costs}
