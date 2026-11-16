# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the end-effector velocity-limit cost."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_tool_velocity_limit import ToolVelocityLimitCost
from curobo._src.cost.criteria_tool_velocity_limit import ToolVelocityLimitCriteria
from curobo._src.util.logging import log_and_raise


@dataclass
class ToolVelocityLimitCostCfg(BaseCostCfg):
    """Static configuration for projected per-tool velocity limits."""

    class_type: Type[ToolVelocityLimitCost] = ToolVelocityLimitCost
    tool_frames: Optional[List[str]] = None
    default_tool_velocity_limit_criteria: Optional[
        Union[ToolVelocityLimitCriteria, Dict[str, Any]]
    ] = None
    tool_velocity_limit_criteria: Dict[
        str,
        Union[ToolVelocityLimitCriteria, Dict[str, Any]],
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize weights and criteria into one canonical representation."""
        super().__post_init__()
        if self.device_cfg.dtype != torch.float32:
            log_and_raise(
                "ToolVelocityLimitCostCfg supports only torch.float32 because its Warp kernel "
                f"uses float32 inputs, got {self.device_cfg.dtype}"
            )
        if self.weight.shape != (2,):
            log_and_raise(f"weight must have shape (2,), got {tuple(self.weight.shape)}")
        if not bool(torch.isfinite(self.weight).all()):
            log_and_raise("weight must contain only finite values")
        if not bool(torch.all(self.weight >= 0.0)):
            log_and_raise("weight must contain nonnegative values")

        if isinstance(self.default_tool_velocity_limit_criteria, dict):
            criteria_data = dict(self.default_tool_velocity_limit_criteria)
            criteria_data["device_cfg"] = self.device_cfg
            self.default_tool_velocity_limit_criteria = ToolVelocityLimitCriteria(**criteria_data)
        elif self.default_tool_velocity_limit_criteria is None:
            self.default_tool_velocity_limit_criteria = ToolVelocityLimitCriteria.disabled(
                self.device_cfg
            )
        elif not isinstance(
            self.default_tool_velocity_limit_criteria,
            ToolVelocityLimitCriteria,
        ):
            log_and_raise(
                "default_tool_velocity_limit_criteria must be a "
                "ToolVelocityLimitCriteria, dictionary, or None"
            )
        elif self.default_tool_velocity_limit_criteria.device_cfg != self.device_cfg:
            log_and_raise(
                "default_tool_velocity_limit_criteria device_cfg must match the cost "
                f"device_cfg: {self.default_tool_velocity_limit_criteria.device_cfg} "
                f"!= {self.device_cfg}"
            )

        normalized_criteria = {}
        for tool_frame, criteria in self.tool_velocity_limit_criteria.items():
            if isinstance(criteria, dict):
                criteria_data = dict(criteria)
                criteria_data["device_cfg"] = self.device_cfg
                criteria = ToolVelocityLimitCriteria(**criteria_data)
            if not isinstance(criteria, ToolVelocityLimitCriteria):
                log_and_raise(
                    f"tool_velocity_limit_criteria[{tool_frame!r}] must be a "
                    f"ToolVelocityLimitCriteria, got {type(criteria)}"
                )
            if criteria.device_cfg != self.device_cfg:
                log_and_raise(
                    f"tool_velocity_limit_criteria[{tool_frame!r}] device_cfg must match "
                    f"the cost device_cfg: {criteria.device_cfg} != {self.device_cfg}"
                )
            normalized_criteria[tool_frame] = criteria
        self.tool_velocity_limit_criteria = normalized_criteria

        if self.tool_frames is not None:
            self.set_tool_frames(self.tool_frames)

    @property
    def num_links(self) -> int:
        """Return the configured fixed tool-frame capacity."""
        return len(self.tool_frames) if self.tool_frames is not None else 0

    def set_tool_frames(self, tool_frames: Optional[List[str]]) -> None:
        """Set the ordered tool frames while preserving explicit criteria."""
        if tool_frames is None:
            self.tool_frames = None
            return
        if len(tool_frames) != len(set(tool_frames)):
            log_and_raise("tool_frames must contain unique names")
        if self.default_tool_velocity_limit_criteria is None:
            log_and_raise("default_tool_velocity_limit_criteria is not set")
        existing = self.tool_velocity_limit_criteria
        self.tool_frames = list(tool_frames)
        self.tool_velocity_limit_criteria = {
            tool_frame: (
                existing[tool_frame]
                if tool_frame in existing
                else self.default_tool_velocity_limit_criteria.clone()
            )
            for tool_frame in self.tool_frames
        }

    def clone(self) -> ToolVelocityLimitCostCfg:
        """Return a deep copy of all owned mutable state."""
        return ToolVelocityLimitCostCfg(
            weight=self.weight.clone(),
            class_type=self.class_type,
            device_cfg=self.device_cfg,
            convert_to_binary=self.convert_to_binary,
            use_grad_input=self.use_grad_input,
            tool_frames=self.tool_frames.copy() if self.tool_frames is not None else None,
            default_tool_velocity_limit_criteria=(
                self.default_tool_velocity_limit_criteria.clone()
                if self.default_tool_velocity_limit_criteria is not None
                else None
            ),
            tool_velocity_limit_criteria={
                tool_frame: criteria.clone()
                for tool_frame, criteria in self.tool_velocity_limit_criteria.items()
            },
        )
