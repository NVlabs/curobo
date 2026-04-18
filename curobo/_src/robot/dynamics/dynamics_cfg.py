# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for robot dynamics (native CUDA RNEA)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class DynamicsCfg:
    """Configuration for native CUDA RNEA dynamics.

    Everything is derived from KinematicsParams; no external library or URDF
    export is needed.

    Attributes:
        kinematics_config: KinematicsParams with inertial properties.
        device_cfg: Device configuration.
        gravity: Gravity vector in world frame [gx, gy, gz]. Default [0, 0, -9.81].
    """

    #: Kinematics configuration with inertial properties (mass, COM, inertia tensors).
    kinematics_config: KinematicsParams

    #: Device configuration (cuda/cpu) and floating point precision.
    device_cfg: DeviceCfg

    #: Gravity vector in world frame [gx, gy, gz].
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])

    def get_gravity_spatial(self) -> torch.Tensor:
        """Build the 6D spatial gravity vector (Featherstone convention).

        The Featherstone trick: instead of adding gravity to each link, we set
        the base link's acceleration to -gravity (upward). So if gravity = [0,0,-9.81],
        the spatial gravity acceleration = [0,0,0, 0,0,+9.81].

        Returns:
            [6] float32 tensor on the configured device.
        """
        gravity_spatial = torch.zeros(6, dtype=torch.float32)
        gravity_spatial[3] = -self.gravity[0]
        gravity_spatial[4] = -self.gravity[1]
        gravity_spatial[5] = -self.gravity[2]
        return gravity_spatial.to(self.device_cfg.device)
