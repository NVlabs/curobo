# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Joint limits definitions for robot kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import copy_or_clone


@dataclass
class JointLimits:
    """Joint limits for a robot."""

    #: Names of the joints. All tensors are indexed by joint names.
    joint_names: List[str]

    #: Position limits for each joint. Shape [2, n_joints] with rows having [min, max] values.
    position: torch.Tensor

    #: Velocity limits for each joint. Shape [2, n_joints] with rows having [min, max] values.
    velocity: torch.Tensor

    #: Acceleration limits for each joint. Shape [2, n_joints] with rows having [min, max]
    #: values.
    acceleration: torch.Tensor

    #: Jerk limits for each joint. Shape [2, n_joints] with rows having [min, max] values.
    jerk: torch.Tensor

    #: Effort limits for each joint. Shape [2, n_joints] with rows having [min, max] values.
    effort: Optional[torch.Tensor] = None

    #: Device and floating point precision for tensors.
    device_cfg: DeviceCfg = DeviceCfg()

    def __post_init__(self):
        """Post initialization checks and data transfer to device tensors."""
        dof = len(self.joint_names)

        # Validate shapes using the validate_shape method
        self.validate_shape(dof, check_effort=True)

        # Validate that lower limits are less than upper limits
        if (self.position[0, :] >= self.position[1, :]).any():
            log_and_raise("lower limits must be less than upper limits")
        if (self.velocity[0, :] >= self.velocity[1, :]).any():
            log_and_raise("lower velocity limits must be less than upper velocity limits")
        if (self.acceleration[0, :] >= self.acceleration[1, :]).any():
            log_and_raise("lower acceleration limits must be less than upper acceleration limits")
        if (self.jerk[0, :] >= self.jerk[1, :]).any():
            log_and_raise("lower jerk limits must be less than upper jerk limits")
        if self.effort is not None and (self.effort[0, :] >= self.effort[1, :]).any():
            log_and_raise("lower effort limits must be less than upper effort limits")

    @staticmethod
    def from_data_dict(
        data: Dict, device_cfg: DeviceCfg = DeviceCfg()
    ) -> JointLimits:
        """Create JointLimits from a dictionary.

        Args:
            data: Dictionary containing joint limits. E.g., {"position": [0, 1], ...}.
            device_cfg: Device and floating point precision for tensors.

        Returns:
            JointLimits: Joint limits instance.
        """
        p = device_cfg.to_device(data["position"])
        v = device_cfg.to_device(data["velocity"])
        a = device_cfg.to_device(data["acceleration"])
        j = device_cfg.to_device(data["jerk"])
        e = None
        if "effort" in data and data["effort"] is not None:
            e = device_cfg.to_device(data["effort"])

        return JointLimits(data["joint_names"], p, v, a, j, e)

    def clone(self) -> JointLimits:
        """Clone joint limits."""
        return JointLimits(
            self.joint_names.copy(),
            self.position.clone(),
            self.velocity.clone(),
            self.acceleration.clone(),
            self.jerk.clone(),
            self.effort.clone() if self.effort is not None else None,
            self.device_cfg,
        )

    def copy_(self, new_jl: JointLimits) -> JointLimits:
        """Copy joint limits from another instance. This maintains reference and copies the data.

        Args:
            new_jl: JointLimits instance to copy from.

        Returns:
            JointLimits: Data copied joint limits.
        """
        self.joint_names = new_jl.joint_names.copy()
        self.position.copy_(new_jl.position)
        self.velocity.copy_(new_jl.velocity)
        self.acceleration.copy_(new_jl.acceleration)
        self.jerk.copy_(new_jl.jerk)
        self.effort = copy_or_clone(new_jl.effort, self.effort)
        return self

    def validate_shape(self, dof: int, check_effort: bool = True):
        """Validate that all limit tensors have correct shape.

        Args:
            dof: Expected degrees of freedom.
            check_effort: Whether to check effort limits shape.

        Raises:
            ValueError: If any tensor shape doesn't match expected (2, dof).
        """
        errors = []
        if self.position.shape != (2, dof):
            errors.append(f"position shape does not match dof: {self.position.shape} != {2, dof}")
        if self.velocity.shape != (2, dof):
            errors.append(f"velocity shape does not match dof: {self.velocity.shape} != {2, dof}")
        if self.acceleration.shape != (2, dof):
            errors.append(
                f"acceleration shape does not match dof: {self.acceleration.shape} != {2, dof}"
            )
        if self.jerk.shape != (2, dof):
            errors.append(f"jerk shape does not match dof: {self.jerk.shape} != {2, dof}")
        if check_effort and self.effort is not None and self.effort.shape != (2, dof):
            errors.append(f"effort shape does not match dof: {self.effort.shape} != {2, dof}")

        if errors:
            log_and_raise("Joint limits validation failed:\n  " + "\n  ".join(errors))

    @property
    def position_lower_limits(self) -> torch.Tensor:
        return self.position[0, :]

    @property
    def position_upper_limits(self) -> torch.Tensor:
        return self.position[1, :]

