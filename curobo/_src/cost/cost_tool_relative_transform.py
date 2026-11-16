# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost for transforms relative to one moving tool frame."""

from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Dict, Tuple

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.criteria_tool_relative_transform import (
    StackedToolRelativeTransformCriteria,
    ToolRelativeTransformCriteria,
)
from curobo._src.cost.wp_tool_relative_transform import ToolRelativeTransformDistance
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_tool_relative_transform_cfg import ToolRelativeTransformCostCfg


class ToolRelativeTransformCost(BaseCost):
    """Evaluate a fixed one-reference group of relative tool transforms."""

    def __init__(self, config: ToolRelativeTransformCostCfg):
        """Initialize fixed topology and device-resident criteria buffers."""
        if config.tool_frames is None or config.reference_tool_frame is None:
            log_and_raise(
                "ToolRelativeTransformCost requires configured tool_frames and "
                "reference_tool_frame"
            )
        self.config = config
        self.tool_frames = config.tool_frames.copy()
        self.num_links = len(self.tool_frames)
        self.reference_tool_frame = config.reference_tool_frame
        self.reference_link_index = config.reference_link_index
        self._stacked_criteria = StackedToolRelativeTransformCriteria.from_criteria(
            config.tool_relative_transform_criteria
        )
        super().__init__(config)

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs) -> None:
        """Allocate reusable output and fused-gradient buffers."""
        if batch_size != self._batch_size or horizon != self._horizon:
            output_shape = (batch_size, horizon, self.num_links)
            tensor_kwargs = {
                "device": self.device_cfg.device,
                "dtype": torch.float32,
            }
            self._out_cost = torch.zeros(output_shape, **tensor_kwargs)
            self._out_position_distance = torch.zeros(output_shape, **tensor_kwargs)
            self._out_rotation_distance = torch.zeros(output_shape, **tensor_kwargs)
            self._out_position_gradient = torch.zeros((*output_shape, 3), **tensor_kwargs)
            self._out_rotation_gradient = torch.zeros((*output_shape, 4), **tensor_kwargs)
        super().setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        current_tool_poses: ToolPose,
        tool_relative_transform_target: ToolRelativeTransformTarget,
        idxs_tool_relative_transform_target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate aligned relative targets for the selected batch groups."""
        target = tool_relative_transform_target
        if current_tool_poses.tool_frames != self.tool_frames:
            log_and_raise("current_tool_poses tool_frames must match the cost's fixed tool_frames")
        if target.tool_frames != self.tool_frames:
            log_and_raise("target tool_frames must match the cost's fixed tool_frames")
        if target.reference_tool_frame != self.reference_tool_frame:
            log_and_raise(
                "target reference_tool_frame must match the cost's fixed "
                f"reference_tool_frame: {target.reference_tool_frame!r} != "
                f"{self.reference_tool_frame!r}"
            )
        if target.device != self.device_cfg.device or target.dtype != torch.float32:
            log_and_raise(
                "target tensors must use the cost device and float32 dtype: "
                f"got {target.device} and {target.dtype}"
            )
        batch_size, horizon = current_tool_poses.position.shape[:2]
        if batch_size != self._batch_size or horizon != self._horizon:
            log_and_raise(
                "setup_batch_tensors must be called with the current batch size and horizon "
                "before forward"
            )

        return ToolRelativeTransformDistance.apply(
            current_tool_poses.position,
            current_tool_poses.quaternion,
            target.position,
            target.quaternion,
            idxs_tool_relative_transform_target,
            self._weight,
            self._stacked_criteria.position_projector,
            self._stacked_criteria.rotation_projector,
            self._stacked_criteria.position_axis_scale,
            self._stacked_criteria.rotation_axis_scale,
            self._stacked_criteria.tolerance,
            self._stacked_criteria.relation_enabled,
            self._out_cost,
            self._out_position_distance,
            self._out_rotation_distance,
            self._out_position_gradient,
            self._out_rotation_gradient,
            self.reference_link_index,
        )

    def update_tool_relative_transform_criteria(
        self,
        tool_relative_transform_criteria: Dict[str, ToolRelativeTransformCriteria],
    ) -> None:
        """Update selected relation criteria in existing device buffers."""
        reference_criterion = tool_relative_transform_criteria.get(self.reference_tool_frame)
        if reference_criterion is not None and reference_criterion.enabled:
            log_and_raise("The reference slot must remain disabled")
        self._stacked_criteria.update_criteria(tool_relative_transform_criteria)
