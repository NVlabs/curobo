# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projected end-effector linear and angular velocity-limit cost."""

from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Dict, Tuple

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.criteria_tool_velocity_limit import (
    StackedToolVelocityLimitCriteria,
)
from curobo._src.cost.wp_tool_velocity_limit import ToolVelocityLimitDistance
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_tool_velocity_limit_cfg import ToolVelocityLimitCostCfg
    from curobo._src.cost.criteria_tool_velocity_limit import ToolVelocityLimitCriteria


class ToolVelocityLimitCost(BaseCost):
    """Limit projected world-frame tool speeds computed as ``J(q) qdot``."""

    def __init__(self, config: ToolVelocityLimitCostCfg):
        """Initialize fixed per-tool criteria from ``config``."""
        self.config: ToolVelocityLimitCostCfg = config
        if config.tool_frames is None or not config.tool_frames:
            log_and_raise("ToolVelocityLimitCost requires at least one configured tool frame")
        self._stacked_tool_velocity_limit_criteria = (
            StackedToolVelocityLimitCriteria.from_tool_velocity_limit_criteria(
                config.tool_velocity_limit_criteria
            )
        )
        self.tool_frames = config.tool_frames
        self.num_links = len(self.tool_frames)
        super().__init__(config)

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs) -> None:
        """Allocate reusable output and analytic twist-gradient buffers."""
        if batch_size != self._batch_size or horizon != self._horizon:
            shape = (batch_size, horizon, self.num_links)
            self._out_cost = torch.zeros(
                (batch_size, horizon, 2 * self.num_links),
                **self.device_cfg.as_torch_dict(),
            )
            self._out_linear_violation = torch.zeros(
                shape,
                **self.device_cfg.as_torch_dict(),
            )
            self._out_angular_violation = torch.zeros(
                shape,
                **self.device_cfg.as_torch_dict(),
            )
            self._out_linear_twist_gradient = torch.zeros(
                (*shape, 3),
                **self.device_cfg.as_torch_dict(),
            )
            self._out_angular_twist_gradient = torch.zeros(
                (*shape, 3),
                **self.device_cfg.as_torch_dict(),
            )
        super().setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        joint_velocity: torch.Tensor,
        tool_jacobians: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return interleaved costs and linear/angular speed violations.

        Args:
            joint_velocity: Joint velocity with shape ``[batch, horizon, dof]``.
            tool_jacobians: World-frame tool Jacobians with shape
                ``[batch, horizon, num_links, 6, dof]``. Linear rows precede
                angular rows.
            kwargs: Ignored rollout data accepted for cost-manager compatibility.
        """
        if joint_velocity is None or tool_jacobians is None:
            log_and_raise("joint_velocity and tool_jacobians must be provided")
        if joint_velocity.ndim != 3 or tool_jacobians.ndim != 5:
            log_and_raise(
                "joint_velocity and tool_jacobians must have shapes [B,H,D] and [B,H,L,6,D]"
            )
        if tool_jacobians.shape[2] != self.num_links:
            log_and_raise(
                "tool_jacobians must match the configured tool frames: "
                f"expected {self.num_links}, got {tool_jacobians.shape[2]}"
            )
        if (
            joint_velocity.shape[0] != tool_jacobians.shape[0]
            or joint_velocity.shape[1] != tool_jacobians.shape[1]
            or joint_velocity.shape[2] != tool_jacobians.shape[4]
        ):
            log_and_raise(
                "joint_velocity and tool_jacobians must have the same batch, horizon, and dof"
            )
        if tool_jacobians.shape[3] != 6:
            log_and_raise("tool_jacobians must have six world-frame twist rows")
        if joint_velocity.device != self.device_cfg.device:
            log_and_raise(
                f"joint_velocity must be on {self.device_cfg.device}, got {joint_velocity.device}"
            )
        if tool_jacobians.device != self.device_cfg.device:
            log_and_raise(
                f"tool_jacobians must be on {self.device_cfg.device}, got {tool_jacobians.device}"
            )

        batch_size, horizon, _ = joint_velocity.shape
        self.setup_batch_tensors(batch_size, horizon)
        criteria = self._stacked_tool_velocity_limit_criteria
        return ToolVelocityLimitDistance.apply(
            joint_velocity,
            tool_jacobians,
            self._weight,
            criteria.linear_projector,
            criteria.angular_projector,
            criteria.linear_axis_scale,
            criteria.angular_axis_scale,
            criteria.max_linear_speed,
            criteria.max_angular_speed,
            self._out_cost,
            self._out_linear_violation,
            self._out_angular_violation,
            self._out_linear_twist_gradient,
            self._out_angular_twist_gradient,
            self.config.use_grad_input,
        )

    def update_tool_velocity_limit_criteria(
        self,
        criteria_by_frame: Dict[str, ToolVelocityLimitCriteria],
    ) -> None:
        """Update selected fixed-shape criteria without replacing their tensors."""
        self._stacked_tool_velocity_limit_criteria.update_tool_velocity_limit_criteria(
            criteria_by_frame
        )
