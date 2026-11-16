# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize complete cubic controls from retained native solver states."""

from __future__ import annotations

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.trajectory.trajectory_spline import SplineTrajectory, UniformTiming


def _boundary(state: JointState, index: int, dt: torch.Tensor) -> torch.Tensor:
    """Solve the cubic boundary support from analytic endpoint derivatives."""
    p = state.position[:, index, :].unsqueeze(1)
    v = state.velocity[:, index, :].unsqueeze(1) * dt[:, None, None]
    a = state.acceleration[:, index, :].unsqueeze(1) * dt[:, None, None].square()
    j = state.jerk[:, index, :].unsqueeze(1) * dt[:, None, None].pow(3)
    if index == -1:
        v, j = -v, -j
    coefficients = p.new_tensor([-1, 0, 1, 2])[None, :, None]
    acceleration = p.new_tensor([1 / 3, -1 / 6, 1 / 3, 11 / 6])[None, :, None]
    jerk = p.new_tensor([0, 0, 0, 1])[None, :, None]
    points = p + v * coefficients + a * acceleration + j * jerk
    return points.flip(1) if index == -1 else points


def native_spline(state: JointState) -> SplineTrajectory:
    """Export the exact native cubic domain, including both boundary spans."""
    dt = state.knot_dt.reshape(-1)
    controls = torch.cat(
        (_boundary(state, 0, dt), state.knot[:, :-1], _boundary(state, -1, dt)), dim=1
    )
    return SplineTrajectory(
        controls.contiguous(), state.joint_names, UniformTiming(dt.clone().contiguous())
    )
