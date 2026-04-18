# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

        # Cache for warp kernels - one per link for efficiency
        self._warp_kernel: Optional[wp.kernel] = None
        self._constants = (
            -1,
            self.config.rotation_method,
        )

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs):
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

        num_goalset = goal_tool_poses.num_goalset
        constants = (
            num_goalset,
            self.config.rotation_method,
        )
        if self._constants != constants:
            self._constants = constants
            self._warp_kernel = create_goalset_pose_distance_kernel_with_constants(
                num_goalset,
                self.config.rotation_method,
            )


        warp_kernel = self._warp_kernel
        goal_pos = goal_tool_poses.position.squeeze(1)
        goal_quat = goal_tool_poses.quaternion.squeeze(1)
        cost, linear_distance, angular_distance, goalset_idx = (
            ToolPoseDistance.apply(
                current_tool_poses.position,
                current_tool_poses.quaternion,
                goal_pos,
                goal_quat,
                idxs_goal,
                self._weight,
                self._stacked_tool_pose_criteria.terminal_pose_axes_weight_factor,
                self._stacked_tool_pose_criteria.non_terminal_pose_axes_weight_factor,
                self._stacked_tool_pose_criteria.terminal_pose_convergence_tolerance,
                self._stacked_tool_pose_criteria.non_terminal_pose_convergence_tolerance,
                self._stacked_tool_pose_criteria.project_distance_to_goal,
                self._out_distance,
                self._out_position_distance,
                self._out_rotation_distance,
                self._out_position_gradient,
                self._out_rotation_gradient,
                self._out_goalset_idx,
                self.config.use_grad_input,
                warp_kernel,
            )
        )
        return cost, linear_distance, angular_distance, goalset_idx


    def update_tool_pose_criteria(
        self,
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ):

        self._stacked_tool_pose_criteria.update_tool_pose_criteria(
            tool_pose_criteria=tool_pose_criteria,
        )
