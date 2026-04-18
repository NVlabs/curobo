# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-problem best-solution tracking for gradient optimizers.

Maintains the lowest cost, corresponding action, iteration index, and convergence
state across optimization iterations for each parallel problem.
"""

from typing import Optional

import torch

from curobo._src.optim.gradient.update_best_solution import update_best_solution
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.torch_util import get_torch_jit_decorator


class BestTracker:
    """Tracks best cost, action, iteration, and convergence state per problem."""

    def __init__(self, device_cfg: DeviceCfg):
        self.device_cfg = device_cfg
        self.cost: Optional[torch.Tensor] = None
        self.action: Optional[torch.Tensor] = None
        self.iteration: Optional[torch.Tensor] = None
        self.current_iteration: Optional[torch.Tensor] = None
        self.previous_step_direction: Optional[torch.Tensor] = None
        self.converged: Optional[torch.Tensor] = None

    def resize(self, num_problems: int, action_horizon: int, action_dim: int):
        """Allocate or reallocate tracking buffers for a given number of problems."""
        self.cost = (
            torch.ones(
                (num_problems,), device=self.device_cfg.device, dtype=self.device_cfg.dtype
            )
            * 5000000.0
        )
        self.action = torch.zeros(
            (num_problems, action_horizon, action_dim),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        self.iteration = torch.zeros(
            (num_problems,), device=self.device_cfg.device, dtype=torch.int16
        )
        self.current_iteration = torch.zeros(
            (num_problems,), device=self.device_cfg.device, dtype=torch.int16
        )
        self.previous_step_direction = torch.zeros(
            (num_problems, action_horizon, action_dim),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        self.converged = torch.zeros(
            (num_problems,),
            device=self.device_cfg.device,
            dtype=torch.uint8,
        )

    def clear(self, mask: Optional[torch.Tensor] = None):
        """Clear tracking buffers. CUDA graph compatible.

        Args:
            mask: Boolean mask [num_problems]. If None, clear all.
        """
        if mask is None:
            self.cost[:] = 5000000000000.0
            self.iteration[:] = 0
            self.current_iteration[:] = 0
            self.previous_step_direction[:] = 0.0
            self.converged[:] = 0
        else:
            self.cost[mask] = 5000000000000.0
            self.iteration[mask] = 0
            self.current_iteration[mask] = 0
            self.previous_step_direction[mask] = 0.0
            self.converged[mask] = 0

    def reset(self):
        """Reset cost and iteration tracking (e.g., after MPC shift)."""
        self.cost[:] = 5000000000000.0
        self.iteration[:] = 0
        self.current_iteration[:] = 0
        self.previous_step_direction[:] = 0.0
        self.converged[:] = 0

    def update(
        self,
        iteration_state: OptimizationIterationState,
        action_horizon: int,
        action_dim: int,
        cost_delta_threshold: float,
        cost_relative_threshold: float,
        convergence_iteration: int,
    ) -> OptimizationIterationState:
        """Update best solution tracking from current iteration state."""
        return update_best_solution(
            iteration_state,
            action_horizon,
            action_dim,
            cost_delta_threshold,
            cost_relative_threshold,
            convergence_iteration,
        )

    @staticmethod
    @get_torch_jit_decorator()
    def check_convergence(
        converged: torch.Tensor,
        converged_ratio: float,
    ) -> bool:
        """Check if enough problems have converged."""
        success = False
        success_count = torch.count_nonzero(converged)
        if success_count > len(converged) * converged_ratio:
            success = True
        return success
