# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for multi-link pose goalset cost."""

from __future__ import annotations

# Standard Library
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.util.logging import log_and_raise

_UNSET = object()


@dataclass
class ToolPoseCostCfg(BaseCostCfg):
    """Configuration for multi-link goalset pose cost."""

    #: Class type of the cost. This is used to initialize the cost class from this configuration.
    class_type: Type[ToolPoseCost] = ToolPoseCost

    #: List of link names for which to compute pose costs
    tool_frames: Optional[List[str]] = None

    #: Criteria cloned for every configured tool frame without an explicit criterion.
    #: A raw dictionary is accepted when constructing this config from YAML.
    default_tool_pose_criteria: Optional[Union[ToolPoseCriteria, Dict[str, Any]]] = None

    #: Explicit per-frame criteria. These take precedence over the default criteria.
    tool_pose_criteria: Dict[str, ToolPoseCriteria] = field(default_factory=dict)

    # Compatibility-only constructor inputs. They are converted once into
    # default_tool_pose_criteria and are not stored on the configuration.
    _terminal_pose_convergence_tolerance: InitVar[Optional[Union[torch.Tensor, List[float]]]] = (
        None
    )
    _non_terminal_pose_convergence_tolerance: InitVar[
        Optional[Union[torch.Tensor, List[float]]]
    ] = None
    _terminal_pose_axes_weight_factor: InitVar[Optional[Union[torch.Tensor, List[float]]]] = None
    _non_terminal_pose_axes_weight_factor: InitVar[Optional[Union[torch.Tensor, List[float]]]] = (
        None
    )
    _project_distance_to_goal: InitVar[Union[torch.Tensor, bool, None, object]] = _UNSET
    use_lie_group: InitVar[Optional[bool]] = None

    def __post_init__(
        self,
        _terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]],
        _non_terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]],
        _terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]],
        _non_terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]],
        _project_distance_to_goal: Union[torch.Tensor, bool, None, object],
        use_lie_group: Optional[bool],
    ) -> None:
        """Post-initialization validation and setup."""
        super().__post_init__()
        if self.device_cfg.dtype != torch.float32:
            log_and_raise(
                "ToolPoseCostCfg supports only torch.float32 because its Warp kernel uses "
                f"float32 inputs, got {self.device_cfg.dtype}"
            )
        legacy_criteria = {}
        if _terminal_pose_convergence_tolerance is not None:
            legacy_criteria["terminal_pose_convergence_tolerance"] = (
                _terminal_pose_convergence_tolerance
            )
        if _non_terminal_pose_convergence_tolerance is not None:
            legacy_criteria["non_terminal_pose_convergence_tolerance"] = (
                _non_terminal_pose_convergence_tolerance
            )
        if _terminal_pose_axes_weight_factor is not None:
            legacy_criteria["terminal_pose_axes_weight_factor"] = _terminal_pose_axes_weight_factor
        if _non_terminal_pose_axes_weight_factor is not None:
            legacy_criteria["non_terminal_pose_axes_weight_factor"] = (
                _non_terminal_pose_axes_weight_factor
            )
        if _project_distance_to_goal is not _UNSET:
            legacy_criteria["project_distance_to_goal"] = _project_distance_to_goal

        if self.default_tool_pose_criteria is not None and legacy_criteria:
            log_and_raise(
                "default_tool_pose_criteria cannot be combined with legacy private "
                "tool-pose criteria inputs"
            )

        if isinstance(self.default_tool_pose_criteria, dict):
            criteria_data = dict(self.default_tool_pose_criteria)
            criteria_data["device_cfg"] = self.device_cfg
            self.default_tool_pose_criteria = ToolPoseCriteria(**criteria_data)
        elif self.default_tool_pose_criteria is None:
            self.default_tool_pose_criteria = ToolPoseCriteria(
                device_cfg=self.device_cfg,
                **legacy_criteria,
            )
        elif not isinstance(self.default_tool_pose_criteria, ToolPoseCriteria):
            log_and_raise(
                "default_tool_pose_criteria must be a ToolPoseCriteria, a dictionary, or None, "
                f"got {type(self.default_tool_pose_criteria)}"
            )
        elif self.default_tool_pose_criteria.device_cfg != self.device_cfg:
            log_and_raise(
                "default_tool_pose_criteria must use the ToolPoseCostCfg device_cfg: "
                f"{self.default_tool_pose_criteria.device_cfg} != {self.device_cfg}"
            )

        for link_name, criteria in self.tool_pose_criteria.items():
            if not isinstance(criteria, ToolPoseCriteria):
                log_and_raise(
                    f"tool_pose_criteria[{link_name!r}] must be a ToolPoseCriteria, "
                    f"got {type(criteria)}"
                )
            if criteria.device_cfg != self.device_cfg:
                log_and_raise(
                    f"tool_pose_criteria[{link_name!r}] must use the ToolPoseCostCfg "
                    f"device_cfg: {criteria.device_cfg} != {self.device_cfg}"
                )

        # The compatibility switch is intentionally ignored. Rotation now has
        # one common metric and the choice is no longer runtime configuration.
        _ = use_lie_group

        if self.tool_frames is not None:
            self.set_tool_frames(self.tool_frames)

    def clone(self) -> ToolPoseCostCfg:
        """Create a deep copy of this configuration."""
        return ToolPoseCostCfg(
            weight=self.weight.clone(),
            class_type=self.class_type,
            device_cfg=self.device_cfg,
            convert_to_binary=self.convert_to_binary,
            use_grad_input=self.use_grad_input,
            tool_frames=self.tool_frames.copy() if self.tool_frames is not None else None,
            default_tool_pose_criteria=self.default_tool_pose_criteria.clone(),
            tool_pose_criteria={
                link_name: criteria.clone()
                for link_name, criteria in self.tool_pose_criteria.items()
            },
        )

    def set_tool_frames(self, tool_frames: Optional[List[str]]) -> None:
        """Update the list of tool frames.

        Args:
            tool_frames: List of tool frames to set.
        """
        if tool_frames is None:
            self.tool_frames = None
            return
        if self.default_tool_pose_criteria is None:
            log_and_raise("default_tool_pose_criteria is not set")

        existing_criteria = self.tool_pose_criteria
        self.tool_frames = list(tool_frames)
        self.tool_pose_criteria = {
            link_name: (
                existing_criteria[link_name]
                if link_name in existing_criteria
                else self.default_tool_pose_criteria.clone()
            )
            for link_name in self.tool_frames
        }

    @property
    def num_links(self) -> int:
        """Get the number of links."""
        return len(self.tool_frames) if self.tool_frames is not None else 0
