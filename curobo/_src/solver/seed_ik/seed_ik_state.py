# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SeedIKState:
    """State container for Seed IK Solver iterations."""

    success: Optional[torch.Tensor] = None
    improvement: Optional[torch.Tensor] = None
    joint_position: Optional[torch.Tensor] = None
    error_norm: Optional[torch.Tensor] = None
    jTerror: Optional[torch.Tensor] = None
    jacobian: Optional[torch.Tensor] = None
    lambda_damping: Optional[torch.Tensor] = None
    position_errors: Optional[torch.Tensor] = None
    orientation_errors: Optional[torch.Tensor] = None

    def clone(self):
        return SeedIKState(
            success=self.success.clone() if self.success is not None else None,
            improvement=self.improvement.clone() if self.improvement is not None else None,
            joint_position=self.joint_position.clone() if self.joint_position is not None else None,
            error_norm=self.error_norm.clone() if self.error_norm is not None else None,
            jTerror=self.jTerror.clone() if self.jTerror is not None else None,
            jacobian=self.jacobian.clone() if self.jacobian is not None else None,
            lambda_damping=self.lambda_damping.clone() if self.lambda_damping is not None else None,
            position_errors=(
                self.position_errors.clone() if self.position_errors is not None else None
            ),
            orientation_errors=(
                self.orientation_errors.clone() if self.orientation_errors is not None else None
            ),
        )

    def copy_(self, other: "SeedIKState"):
        if self.success is not None:
            self.success.copy_(other.success)
        if self.improvement is not None:
            self.improvement.copy_(other.improvement)
        if self.joint_position is not None:
            self.joint_position.copy_(other.joint_position)
        if self.error_norm is not None:
            self.error_norm.copy_(other.error_norm)
        if self.jTerror is not None:
            self.jTerror.copy_(other.jTerror)
        if self.jacobian is not None:
            self.jacobian.copy_(other.jacobian)
        if self.lambda_damping is not None:
            self.lambda_damping.copy_(other.lambda_damping)
        if self.position_errors is not None:
            self.position_errors.copy_(other.position_errors)
        if self.orientation_errors is not None:
            self.orientation_errors.copy_(other.orientation_errors)
