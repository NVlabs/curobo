# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Absolute multi-link tool-pose cost."""

from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Dict, Optional, Tuple

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.tool_pose_criteria import StackedToolPoseCriteria
from curobo._src.cost.wp_tool_pose import (
    ToolPoseDistance,
    create_goalset_pose_distance_kernel_with_constants,
)
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria


class ToolPoseCost(BaseCost):
    """Cost that computes pose goalset cost for multiple links."""

    def __init__(self, config: ToolPoseCostCfg):
        """Initialize the ToolPoseCost.

        Args:
            config: Configuration for the cost.
        """
        self.config: ToolPoseCostCfg = config
        self._stacked_tool_pose_criteria = StackedToolPoseCriteria.from_tool_pose_criteria(
            config.tool_pose_criteria
        )
        super().__init__(config)

        # Store link names for which this cost applies
        self.tool_frames = config.tool_frames
        self.num_links = len(self.tool_frames)

        # Cache the goal-set-specialized Warp kernel.
        self._warp_kernel: Optional[wp.kernel] = None
        self._constants = -1
        self._position_metric = torch.empty(
            (2 * self.num_links, 3, 3),
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )
        self._rotation_metric = torch.empty(
            (2 * self.num_links, 3, 3),
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )
        self._pose_tolerance = torch.empty(
            (2 * self.num_links, 2),
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )
        self._pose_active = torch.empty(
            (2 * self.num_links,),
            dtype=torch.uint8,
            device=self.device_cfg.device,
        )
        self._refresh_pose_metric_buffers()

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs):
        """Allocate reusable output and analytic-gradient buffers."""
        if batch_size != self._batch_size or horizon != self._horizon:
            num_links = len(self.tool_frames)
            self._out_distance = torch.zeros(
                (batch_size, horizon, 2 * num_links),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_position_distance = torch.zeros(
                (batch_size, horizon, num_links),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_rotation_distance = torch.zeros(
                (batch_size, horizon, num_links),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_goalset_idx = torch.zeros(
                (batch_size, horizon, num_links),
                dtype=torch.int32,
                device=self.device_cfg.device,
            )

            self._out_position_gradient = torch.zeros(
                (batch_size, horizon, num_links, 3),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_rotation_gradient = torch.zeros(
                (batch_size, horizon, num_links, 4),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )

        super().setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        current_tool_poses: ToolPose,
        goal_tool_poses: GoalToolPose,
        idxs_goal: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized implementation for ToolPose."""
        # Input validation
        if current_tool_poses is None or goal_tool_poses is None:
            log_and_raise("current_tool_poses and goal_tool_poses must be provided")
        if current_tool_poses.tool_frames != goal_tool_poses.tool_frames:
            log_and_raise("current_tool_poses and goal_tool_poses must have the same link names")
        if current_tool_poses.tool_frames != self.tool_frames:
            log_and_raise(
                "current_tool_poses and goal_tool_poses must match the cost's "
                "configured tool_frames"
            )

        num_goalset = goal_tool_poses.num_goalset
        if self._constants != num_goalset:
            self._constants = num_goalset
            self._warp_kernel = create_goalset_pose_distance_kernel_with_constants(num_goalset)

        warp_kernel = self._warp_kernel
        goal_pos = goal_tool_poses.position.squeeze(1)
        goal_quat = goal_tool_poses.quaternion.squeeze(1)
        cost, linear_distance, angular_distance, goalset_idx = ToolPoseDistance.apply(
            current_tool_poses.position,
            current_tool_poses.quaternion,
            goal_pos,
            goal_quat,
            idxs_goal,
            self._weight,
            self._position_metric,
            self._rotation_metric,
            self._pose_tolerance,
            self._stacked_tool_pose_criteria.error_frame,
            self._out_distance,
            self._out_position_distance,
            self._out_rotation_distance,
            self._out_position_gradient,
            self._out_rotation_gradient,
            self._out_goalset_idx,
            self.config.use_grad_input,
            warp_kernel,
        )
        return cost, linear_distance, angular_distance, goalset_idx

    def update_tool_pose_criteria(
        self,
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ):
        """Update selected per-link criteria in the existing stacked buffers."""
        self._stacked_tool_pose_criteria.update_tool_pose_criteria(
            tool_pose_criteria=tool_pose_criteria,
        )
        self._refresh_pose_metric_buffers()

    @staticmethod
    def _quadratic_metric(projector: torch.Tensor, axis_scale: torch.Tensor) -> torch.Tensor:
        """Return ``P.T @ diag(axis_scale**2) @ P`` for a projector batch."""
        scaled_projector = axis_scale.unsqueeze(-1) * projector
        return scaled_projector.transpose(-1, -2) @ scaled_projector

    def _refresh_pose_metric_buffers(self) -> None:
        """Refresh fixed-shape quadratic metrics without changing their addresses."""
        criteria = self._stacked_tool_pose_criteria
        position_metric = torch.cat(
            (
                self._quadratic_metric(
                    criteria.terminal_position_projector,
                    criteria.terminal_position_axis_scale,
                ),
                self._quadratic_metric(
                    criteria.path_position_projector,
                    criteria.path_position_axis_scale,
                ),
            )
        )
        rotation_metric = torch.cat(
            (
                self._quadratic_metric(
                    criteria.terminal_rotation_projector,
                    criteria.terminal_rotation_axis_scale,
                ),
                self._quadratic_metric(
                    criteria.path_rotation_projector,
                    criteria.path_rotation_axis_scale,
                ),
            )
        )
        self._position_metric.copy_(position_metric)
        self._rotation_metric.copy_(rotation_metric)
        self._pose_tolerance.copy_(
            torch.cat((criteria.terminal_tolerance, criteria.path_tolerance))
        )
        metric_trace = torch.diagonal(position_metric, dim1=-2, dim2=-1).sum(dim=-1)
        metric_trace += torch.diagonal(rotation_metric, dim1=-2, dim2=-1).sum(dim=-1)
        self._pose_active.copy_((metric_trace > 0.0).to(dtype=torch.uint8))
        self._clear_inactive_pose_outputs()

    def _clear_inactive_pose_outputs(self) -> None:
        """Clear entries whose kernel threads will return without writing."""
        if self._batch_size < 0 or not hasattr(self, "_out_distance"):
            return

        terminal_inactive = self._pose_active[: self.num_links] == 0
        path_inactive = self._pose_active[self.num_links :] == 0
        self._clear_inactive_pose_output_span(
            terminal_inactive,
            slice(self._horizon - 1, self._horizon),
        )
        self._clear_inactive_pose_output_span(
            path_inactive,
            slice(0, self._horizon - 1),
        )

    def _clear_inactive_pose_output_span(
        self,
        inactive_links: torch.Tensor,
        horizon_slice: slice,
    ) -> None:
        """Clear one temporal span using a device-resident per-link mask."""
        link_mask = inactive_links.view(1, 1, self.num_links)
        cost_mask = inactive_links.repeat_interleave(2).view(1, 1, 2 * self.num_links)
        self._out_distance[:, horizon_slice].masked_fill_(cost_mask, 0.0)
        self._out_position_distance[:, horizon_slice].masked_fill_(link_mask, 0.0)
        self._out_rotation_distance[:, horizon_slice].masked_fill_(link_mask, 0.0)
        self._out_position_gradient[:, horizon_slice].masked_fill_(
            link_mask.unsqueeze(-1), 0.0
        )
        self._out_rotation_gradient[:, horizon_slice].masked_fill_(
            link_mask.unsqueeze(-1), 0.0
        )
        self._out_goalset_idx[:, horizon_slice].masked_fill_(link_mask, 0)
