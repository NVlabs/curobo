# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for one-reference relative tool-transform cost."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_tool_relative_transform import ToolRelativeTransformCost
from curobo._src.cost.criteria_tool_relative_transform import ToolRelativeTransformCriteria
from curobo._src.util.logging import log_and_raise


@dataclass
class ToolRelativeTransformCostCfg(BaseCostCfg):
    """Static topology and residual policy for one relative tool group."""

    class_type: Type[ToolRelativeTransformCost] = ToolRelativeTransformCost
    reference_tool_frame: Optional[str] = None
    tool_frames: Optional[List[str]] = None
    default_tool_relative_transform_criteria: Optional[
        Union[ToolRelativeTransformCriteria, Dict[str, Any]]
    ] = None
    tool_relative_transform_criteria: Dict[
        str, Union[ToolRelativeTransformCriteria, Dict[str, Any]]
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize criteria and validate the fixed kernel contract."""
        super().__post_init__()
        if self.device_cfg.dtype != torch.float32:
            log_and_raise(
                "ToolRelativeTransformCostCfg supports only float32 because its Warp kernel "
                f"uses float32 inputs, got {self.device_cfg.dtype}"
            )
        if self.weight.shape != (2,):
            log_and_raise(
                "ToolRelativeTransformCostCfg weight must have shape (2,), "
                f"got {tuple(self.weight.shape)}"
            )
        if self.use_grad_input:
            log_and_raise(
                "ToolRelativeTransformCostCfg requires use_grad_input=False because the fused "
                "kernel precomputes gradients for unit upstream cost summation"
            )
        if self.reference_tool_frame is not None and not self.reference_tool_frame:
            log_and_raise("reference_tool_frame must be a nonempty string")

        if isinstance(self.default_tool_relative_transform_criteria, dict):
            default_data = dict(self.default_tool_relative_transform_criteria)
            default_data["device_cfg"] = self.device_cfg
            self.default_tool_relative_transform_criteria = ToolRelativeTransformCriteria(
                **default_data
            )
        elif self.default_tool_relative_transform_criteria is None:
            self.default_tool_relative_transform_criteria = ToolRelativeTransformCriteria.disabled(
                device_cfg=self.device_cfg
            )
        elif not isinstance(
            self.default_tool_relative_transform_criteria,
            ToolRelativeTransformCriteria,
        ):
            log_and_raise(
                "default_tool_relative_transform_criteria must be a "
                "ToolRelativeTransformCriteria, dictionary, or None"
            )
        elif self.default_tool_relative_transform_criteria.device_cfg != self.device_cfg:
            log_and_raise("default_tool_relative_transform_criteria must use the cost device_cfg")

        normalized_criteria: Dict[str, ToolRelativeTransformCriteria] = {}
        for frame, criterion in self.tool_relative_transform_criteria.items():
            if isinstance(criterion, dict):
                criterion_data = dict(criterion)
                criterion_data["device_cfg"] = self.device_cfg
                criterion = ToolRelativeTransformCriteria(**criterion_data)
            elif not isinstance(criterion, ToolRelativeTransformCriteria):
                log_and_raise(
                    f"tool_relative_transform_criteria[{frame!r}] must be a "
                    "ToolRelativeTransformCriteria or dictionary"
                )
            elif criterion.device_cfg != self.device_cfg:
                log_and_raise(
                    f"tool_relative_transform_criteria[{frame!r}] must use the cost device_cfg"
                )
            normalized_criteria[frame] = criterion
        self.tool_relative_transform_criteria = normalized_criteria

        if self.tool_frames is not None:
            self.set_tool_frames(self.tool_frames)

    def clone(self) -> ToolRelativeTransformCostCfg:
        """Return an independent copy of all owned configuration state."""
        return ToolRelativeTransformCostCfg(
            weight=self.weight.clone(),
            class_type=self.class_type,
            device_cfg=self.device_cfg,
            convert_to_binary=self.convert_to_binary,
            use_grad_input=False,
            reference_tool_frame=self.reference_tool_frame,
            tool_frames=self.tool_frames.copy() if self.tool_frames is not None else None,
            default_tool_relative_transform_criteria=(
                self.default_tool_relative_transform_criteria.clone()
            ),
            tool_relative_transform_criteria={
                frame: criterion.clone()
                for frame, criterion in self.tool_relative_transform_criteria.items()
            },
        )

    def set_tool_frames(self, tool_frames: List[str]) -> None:
        """Set the fixed ordered tool-frame topology and fill missing criteria."""
        if not tool_frames:
            log_and_raise("tool_frames must contain at least one frame")
        if len(set(tool_frames)) != len(tool_frames):
            log_and_raise("tool_frames must be unique")
        if self.reference_tool_frame is None:
            log_and_raise("reference_tool_frame must be set before tool_frames")
        if self.reference_tool_frame not in tool_frames:
            log_and_raise(
                f"reference_tool_frame {self.reference_tool_frame!r} is not in {tool_frames}"
            )
        unknown_frames = set(self.tool_relative_transform_criteria) - set(tool_frames)
        if unknown_frames:
            log_and_raise(
                "tool_relative_transform_criteria contains frames outside tool_frames: "
                f"{sorted(unknown_frames)}"
            )

        existing = self.tool_relative_transform_criteria
        self.tool_frames = list(tool_frames)
        self.tool_relative_transform_criteria = {
            frame: (
                existing[frame]
                if frame in existing
                else self.default_tool_relative_transform_criteria.clone()
            )
            for frame in self.tool_frames
        }
        reference_criterion = self.tool_relative_transform_criteria[self.reference_tool_frame]
        if reference_criterion.enabled:
            log_and_raise(
                "The reference slot must be disabled; it does not define a relation to itself"
            )

    @property
    def num_links(self) -> int:
        """Return the fixed tool-frame capacity."""
        return len(self.tool_frames) if self.tool_frames is not None else 0

    @property
    def reference_link_index(self) -> int:
        """Return the fixed index of the moving reference frame."""
        if self.tool_frames is None or self.reference_tool_frame not in self.tool_frames:
            log_and_raise("reference_tool_frame and tool_frames are not fully configured")
        return self.tool_frames.index(self.reference_tool_frame)
