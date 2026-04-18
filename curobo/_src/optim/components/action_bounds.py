# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Horizon-expanded action bound tensors and per-step limits for optimizers.

Computes and caches bound tensors tiled across the action horizon, along with
maximum step sizes derived from the action range and a configurable scale factor.
"""

from typing import Optional

import torch


class ActionBounds:
    """Computes and caches horizon-expanded action bounds and step limits."""

    def __init__(self, action_bound_lows: torch.Tensor, action_bound_highs: torch.Tensor,
                 action_horizon: int, step_scale: float):
        self._action_horizon = action_horizon
        self._step_scale = step_scale
        self.lows = action_bound_lows
        self.highs = action_bound_highs
        self._compute(action_bound_lows, action_bound_highs, action_horizon, step_scale)

    def _compute(self, lows: torch.Tensor, highs: torch.Tensor, action_horizon: int,
                 step_scale: float):
        self.horizon_lows = lows.view(1, -1).repeat(action_horizon, 1).view(-1)
        self.horizon_highs = highs.view(1, -1).repeat(action_horizon, 1).view(-1)
        action_range_horizon = self.horizon_highs - self.horizon_lows
        self.horizon_step_max = step_scale * torch.abs(action_range_horizon)
        self.step_max = step_scale * torch.abs(highs - lows)

    def refresh(self, action_bound_lows: torch.Tensor, action_bound_highs: torch.Tensor,
                action_horizon: int):
        """Recompute if action_horizon changed."""
        if action_horizon != self._action_horizon:
            self._action_horizon = action_horizon
            self.lows = action_bound_lows
            self.highs = action_bound_highs
            self._compute(action_bound_lows, action_bound_highs, action_horizon, self._step_scale)
