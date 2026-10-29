# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for fixed-path trajectory timeopt."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg


@dataclass(frozen=True)
class TrajectoryTimeOptCfg:
    """Fixed robot, geometry, timing basis, limits, and iteration budget."""

    #: CUDA float32 model with rigid-body inertial parameters and payload.
    robot: KinematicsCfg
    #: Independent paths sharing one model and limit set.
    batch_size: int = 1
    #: Native cubic action controls, including the replaced final action.
    n_knots: int = 16
    #: Independent positive squared-speed spline coefficients; fixed at 32.
    timing_controls: int = 32
    #: Uniform duration/constraint intervals (512 nodes); path knots are also checked.
    num_intervals: int = 511
    #: Total predictor-corrector iterations, each with one Hessian factorization.
    max_iterations: int = 10
    #: Joint velocity, acceleration, and asymmetric torque planning scale.
    limit_scale: float = 0.95
    #: Jerk planning scale, also used by sampled acceptance with a small FP32 tolerance.
    jerk_limit_scale: float = 0.95
    #: Explicit symmetric velocity bounds [D], or the model's symmetric subset.
    velocity_limits: torch.Tensor | None = None
    #: Explicit symmetric acceleration bounds [D], or the model's symmetric subset.
    acceleration_limits: torch.Tensor | None = None
    #: Explicit symmetric jerk bounds [D], or the model's symmetric subset.
    jerk_limits: torch.Tensor | None = None
    #: Asymmetric lower effort bounds [D], otherwise from the robot.
    torque_lower: torch.Tensor | None = None
    #: Asymmetric upper effort bounds [D], otherwise from the robot.
    torque_upper: torch.Tensor | None = None
    #: World-frame gravity in m/s^2.
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    #: Capture preparation, optimization, time construction, and grid-state evaluation.
    use_cuda_graph: bool = True
