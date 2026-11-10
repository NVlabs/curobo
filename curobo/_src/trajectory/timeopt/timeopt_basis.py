# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cubic squared-speed basis and fixed-path problem data."""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import BSpline

from curobo._src.util.logging import log_and_raise


class CubicTimingBasis:
    """Fixed clamped cubic basis, with setup on CPU and runtime evaluation on GPU."""

    def __init__(self, count: int, device: torch.device) -> None:
        """Create count positive timing coefficients, independently of geometric controls."""
        self.count = count
        self.breaks_cpu = np.linspace(0.0, 1.0, count - 2)
        knots = np.r_[np.zeros(3), self.breaks_cpu, np.ones(3)]
        self.spline = BSpline(knots, np.eye(count), 3)
        self.device = device
        left = self.breaks_cpu[:-1]
        self.polynomial = self.tensor(
            np.stack([self.spline(left, nu=k) / (1, 1, 2, 6)[k] for k in (3, 2, 1, 0)], 1)
        )

    def tensor(self, value: np.ndarray) -> torch.Tensor:
        """Copy static setup data to contiguous float32 device storage."""
        return torch.as_tensor(
            np.asarray(value, dtype=np.float32), device=self.device
        ).contiguous()

    def at(self, points: np.ndarray) -> torch.Tensor:
        """Precompute basis derivatives [3,points,coefficients] at fixed positions."""
        return self.tensor(np.stack([self.spline(points, nu=k) for k in range(3)]))


class TimeOptProblem:
    """Borrow the fixed path, dynamics coefficients, and limits used by the CUDA solver."""

    def __init__(
        self,
        basis: CubicTimingBasis,
        grid: np.ndarray,
        path: torch.Tensor,
        torque_coefficients: torch.Tensor,
        limits: tuple[torch.Tensor, ...],
        num_intervals: int,
    ) -> None:
        """Borrow path [4,B,G,D], torque coefficients [B,3,G,D], and scaled limits.

        The grid starts with num_intervals + 1 uniform nodes on [0,1], shared by
        constraints and the mean inverse-speed objective. Remaining nodes only constrain.
        """
        if type(num_intervals) is not int or num_intervals < 1:
            log_and_raise("num_intervals must be an integer >= 1")
        self.basis = basis
        self.path = path
        self.torque_coefficients = torque_coefficients
        self.limits = limits
        self.batch_size = path.shape[1]
        self.grid_cpu = grid
        self.num_intervals = num_intervals
        self.stationary = torch.zeros(self.batch_size, dtype=torch.bool, device=path.device)
