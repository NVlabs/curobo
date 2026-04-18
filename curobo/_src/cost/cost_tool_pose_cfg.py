# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for multi-link pose goalset cost."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.util.logging import log_and_raise


@dataclass
class ToolPoseCostCfg(BaseCostCfg):
    """Configuration for multi-link goalset pose cost."""

    #: Class type of the cost. This is used to initialize the cost class from this configuration.
    class_type: Type[ToolPoseCost] = ToolPoseCost

    #: List of link names for which to compute pose costs
    tool_frames: Optional[List[str]] = None

    tool_pose_criteria: Dict[str, ToolPoseCriteria] = field(default_factory=dict)

    #: If true, the rotation distance and gradient is computed using the Lie group.
    use_lie_group: bool = False

    _terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None
    _non_terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None
    _terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None
    _non_terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None
    _project_distance_to_goal: Union[torch.Tensor, bool] = False
    _pose_criteria: Optional[ToolPoseCriteria] = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # create pose critirea for each link:
        self._pose_criteria = ToolPoseCriteria(
            device_cfg=self.device_cfg,
            terminal_pose_convergence_tolerance=self._terminal_pose_convergence_tolerance,
            non_terminal_pose_convergence_tolerance=self._non_terminal_pose_convergence_tolerance,
            terminal_pose_axes_weight_factor=self._terminal_pose_axes_weight_factor,
            non_terminal_pose_axes_weight_factor=self._non_terminal_pose_axes_weight_factor,
            project_distance_to_goal=self._project_distance_to_goal,
        )
        if self.tool_frames is not None:
            self.set_tool_frames(self.tool_frames)

        super().__post_init__()

    def clone(self):
        """Create a deep copy of this configuration."""
        return ToolPoseCostCfg(
            weight=self.weight.clone(),
            device_cfg=self.device_cfg,
            convert_to_binary=self.convert_to_binary,
            tool_frames=self.tool_frames.copy(),
            tool_pose_criteria={
                link_name: self.tool_pose_criteria[link_name].clone()
                for link_name in self.tool_frames
            },
            use_lie_group=self.use_lie_group,
            _pose_criteria=self._pose_criteria.clone() if self._pose_criteria is not None else None,
        )

    def set_tool_frames(self, tool_frames: List[str]):
        """Update the list of tool frames.

        Args:
            tool_frames: List of tool frames to set.
        """
        self.tool_frames = tool_frames
        if self._pose_criteria is None:
            log_and_raise("pose_criteria is not set, cannot set link names")
        self.tool_pose_criteria = {}
        for link_name in self.tool_frames:
            self.tool_pose_criteria[link_name] = self._pose_criteria.clone()

    @property
    def num_links(self) -> int:
        """Get the number of links."""
        return len(self.tool_frames)

    @property
    def rotation_method(self) -> int:
        """Get the rotation method."""
        return 1 if self.use_lie_group else 0
