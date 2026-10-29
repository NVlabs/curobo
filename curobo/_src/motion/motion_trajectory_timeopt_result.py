# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Borrowed trajectory timeopt results."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from curobo._src.state.state_joint import JointState


@dataclass
class TrajectoryTimeOptResult:
    """Borrowed results; success denotes sampled feasibility, not global optimality."""

    #: Original path with a cubic squared-speed time law; shape [B,32].
    timing_controls: torch.Tensor
    #: Common uniform path-coordinate output grid [G].
    path_parameter: torch.Tensor
    #: Integrated times [B,G], including zero and final duration.
    time: torch.Tensor
    #: Position/velocity/acceleration/jerk [B,G,D].
    joint_state: JointState
    #: Rigid-body torque at the output grid [B,G,D].
    torque: torch.Tensor
    #: Sampled velocity/acceleration/jerk/torque feasibility [B].
    success: torch.Tensor
    #: 0 feasible, 1 no uniform feasible initializer, 2 output validation failure, 3 bad input.
    status: torch.Tensor
    #: All attempted Newton systems were numerically valid [B]; not a convergence certificate.
    optimization_ok: torch.Tensor
    #: Constant-speed seed duration [B]; static reference timing may be made feasible.
    uniform_duration: torch.Tensor
    #: Exact stationary geometry flag [B].
    stationary: torch.Tensor

    @property
    def duration(self) -> torch.Tensor:
        """Return integrated final time [B]."""
        return self.time[:, -1]

    def clone(self) -> TrajectoryTimeOptResult:
        """Retain outputs across subsequent calls."""
        return TrajectoryTimeOptResult(
            **{field.name: getattr(self, field.name).clone() for field in fields(self)}
        )
