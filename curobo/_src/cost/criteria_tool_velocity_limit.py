# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Criteria for projected end-effector velocity limits."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple, Union

# Third Party
import torch

# CuRobo
from curobo._src.cost.tool_pose_criteria import TensorLike, _as_axis_scale, _as_projector
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise

ScalarLike = Union[torch.Tensor, float]


def _as_nonnegative_scalar(
    value: ScalarLike,
    name: str,
    device_cfg: DeviceCfg,
) -> torch.Tensor:
    """Convert one finite, nonnegative physical limit to a scalar tensor."""
    tensor = torch.as_tensor(value, device=device_cfg.device, dtype=device_cfg.dtype).clone()
    if tensor.numel() != 1:
        log_and_raise(f"{name} must be a scalar, got shape {tuple(tensor.shape)}")
    tensor = tensor.reshape(())
    if not bool(torch.isfinite(tensor)):
        log_and_raise(f"{name} must be finite")
    if tensor.item() < 0.0:
        log_and_raise(f"{name} must be nonnegative")
    return tensor


@dataclass
class ToolVelocityLimitCriteria:
    """Projected linear and angular speed limits for one tool frame.

    Projectors are expressed in the world frame. Axis scales are applied after
    projection. A zero projector disables that half of the criterion.
    """

    linear_projector: Optional[TensorLike] = None
    angular_projector: Optional[TensorLike] = None
    linear_axis_scale: Optional[TensorLike] = None
    angular_axis_scale: Optional[TensorLike] = None
    max_linear_speed: ScalarLike = 0.0
    max_angular_speed: ScalarLike = 0.0
    device_cfg: DeviceCfg = DeviceCfg()

    _TENSOR_FIELD_NAMES: ClassVar[Tuple[str, ...]] = (
        "linear_projector",
        "angular_projector",
        "linear_axis_scale",
        "angular_axis_scale",
        "max_linear_speed",
        "max_angular_speed",
    )

    def __post_init__(self) -> None:
        """Normalize and validate the canonical tensor representation."""
        zero_projector = torch.zeros((3, 3), **self.device_cfg.as_torch_dict())
        unit_scale = torch.ones(3, **self.device_cfg.as_torch_dict())
        self.linear_projector = _as_projector(
            zero_projector if self.linear_projector is None else self.linear_projector,
            "linear_projector",
            self.device_cfg,
        )
        self.angular_projector = _as_projector(
            zero_projector if self.angular_projector is None else self.angular_projector,
            "angular_projector",
            self.device_cfg,
        )
        self.linear_axis_scale = _as_axis_scale(
            unit_scale if self.linear_axis_scale is None else self.linear_axis_scale,
            "linear_axis_scale",
            self.device_cfg,
        )
        self.angular_axis_scale = _as_axis_scale(
            unit_scale if self.angular_axis_scale is None else self.angular_axis_scale,
            "angular_axis_scale",
            self.device_cfg,
        )
        self.max_linear_speed = _as_nonnegative_scalar(
            self.max_linear_speed,
            "max_linear_speed",
            self.device_cfg,
        )
        self.max_angular_speed = _as_nonnegative_scalar(
            self.max_angular_speed,
            "max_angular_speed",
            self.device_cfg,
        )

    @classmethod
    def tensor_field_names(cls) -> Tuple[str, ...]:
        """Return canonical fields in their stable copy order."""
        return cls._TENSOR_FIELD_NAMES

    @staticmethod
    def limit(
        max_linear_speed: ScalarLike,
        max_angular_speed: ScalarLike,
        *,
        linear_projector: Optional[TensorLike] = None,
        angular_projector: Optional[TensorLike] = None,
        linear_axis_scale: Optional[TensorLike] = None,
        angular_axis_scale: Optional[TensorLike] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> ToolVelocityLimitCriteria:
        """Limit projected world-frame linear and angular speeds."""
        identity = torch.eye(3, **device_cfg.as_torch_dict())
        return ToolVelocityLimitCriteria(
            linear_projector=identity if linear_projector is None else linear_projector,
            angular_projector=identity if angular_projector is None else angular_projector,
            linear_axis_scale=linear_axis_scale,
            angular_axis_scale=angular_axis_scale,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            device_cfg=device_cfg,
        )

    @staticmethod
    def disabled(device_cfg: DeviceCfg = DeviceCfg()) -> ToolVelocityLimitCriteria:
        """Create a criterion with both twist projectors disabled."""
        return ToolVelocityLimitCriteria(device_cfg=device_cfg)

    def clone(self) -> ToolVelocityLimitCriteria:
        """Return an independent criteria value."""
        return ToolVelocityLimitCriteria(
            device_cfg=self.device_cfg,
            **{name: getattr(self, name).clone() for name in self.tensor_field_names()},
        )

    def to(self, device_cfg: DeviceCfg) -> ToolVelocityLimitCriteria:
        """Return criteria on ``device_cfg``."""
        if self.device_cfg == device_cfg:
            return self
        return ToolVelocityLimitCriteria(
            device_cfg=device_cfg,
            **{
                name: getattr(self, name).to(
                    device=device_cfg.device,
                    dtype=device_cfg.dtype,
                )
                for name in self.tensor_field_names()
            },
        )

    def copy_(self, other: ToolVelocityLimitCriteria) -> ToolVelocityLimitCriteria:
        """Copy canonical values without changing tensor addresses."""
        if self.device_cfg != other.device_cfg:
            log_and_raise(f"device_cfg mismatch: {self.device_cfg} != {other.device_cfg}")
        for name in self.tensor_field_names():
            getattr(self, name).copy_(getattr(other, name))
        return self


@dataclass
class StackedToolVelocityLimitCriteria:
    """Fixed-shape velocity-limit criteria for an ordered tool-frame list."""

    tool_frames: List[str]
    linear_projector: torch.Tensor
    angular_projector: torch.Tensor
    linear_axis_scale: torch.Tensor
    angular_axis_scale: torch.Tensor
    max_linear_speed: torch.Tensor
    max_angular_speed: torch.Tensor
    device_cfg: DeviceCfg = DeviceCfg()

    _TENSOR_FIELD_NAMES: ClassVar[Tuple[str, ...]] = ToolVelocityLimitCriteria._TENSOR_FIELD_NAMES

    def __post_init__(self) -> None:
        """Validate stacked tensor shapes and placement."""
        num_links = len(self.tool_frames)
        expected_shapes = {
            "linear_projector": (num_links, 3, 3),
            "angular_projector": (num_links, 3, 3),
            "linear_axis_scale": (num_links, 3),
            "angular_axis_scale": (num_links, 3),
            "max_linear_speed": (num_links,),
            "max_angular_speed": (num_links,),
        }
        for name, expected_shape in expected_shapes.items():
            tensor = getattr(self, name)
            if tensor.shape != expected_shape:
                log_and_raise(
                    f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.device != self.device_cfg.device:
                log_and_raise(f"{name} must be on {self.device_cfg.device}, got {tensor.device}")
            if tensor.dtype != self.device_cfg.dtype:
                log_and_raise(f"{name} must use {self.device_cfg.dtype}, got {tensor.dtype}")

    @classmethod
    def tensor_field_names(cls) -> Tuple[str, ...]:
        """Return canonical fields in their stable copy order."""
        return cls._TENSOR_FIELD_NAMES

    @staticmethod
    def from_tool_velocity_limit_criteria(
        criteria_by_frame: Dict[str, ToolVelocityLimitCriteria],
    ) -> StackedToolVelocityLimitCriteria:
        """Stack per-frame criteria in dictionary insertion order."""
        if not criteria_by_frame:
            log_and_raise("tool_velocity_limit_criteria must contain at least one tool frame")
        tool_frames = list(criteria_by_frame)
        first = criteria_by_frame[tool_frames[0]]
        for tool_frame, criteria in criteria_by_frame.items():
            if criteria.device_cfg != first.device_cfg:
                log_and_raise(
                    f"device_cfg mismatch for {tool_frame}: "
                    f"{criteria.device_cfg} != {first.device_cfg}"
                )
        values = {
            name: torch.stack([getattr(criteria_by_frame[frame], name) for frame in tool_frames])
            for name in StackedToolVelocityLimitCriteria.tensor_field_names()
        }
        return StackedToolVelocityLimitCriteria(
            tool_frames=tool_frames,
            device_cfg=first.device_cfg,
            **values,
        )

    def clone(self) -> StackedToolVelocityLimitCriteria:
        """Return an independent stacked value."""
        return StackedToolVelocityLimitCriteria(
            tool_frames=self.tool_frames.copy(),
            device_cfg=self.device_cfg,
            **{name: getattr(self, name).clone() for name in self.tensor_field_names()},
        )

    def copy_(
        self,
        other: StackedToolVelocityLimitCriteria,
    ) -> StackedToolVelocityLimitCriteria:
        """Copy a shape-compatible stack without changing tensor addresses."""
        if self.tool_frames != other.tool_frames:
            log_and_raise(f"tool_frames mismatch: {self.tool_frames} != {other.tool_frames}")
        if self.device_cfg != other.device_cfg:
            log_and_raise(f"device_cfg mismatch: {self.device_cfg} != {other.device_cfg}")
        for name in self.tensor_field_names():
            getattr(self, name).copy_(getattr(other, name))
        return self

    def update_tool_velocity_limit_criteria(
        self,
        criteria_by_frame: Dict[str, ToolVelocityLimitCriteria],
    ) -> None:
        """Update selected rows without reallocating fixed-shape tensors."""
        for tool_frame, criteria in criteria_by_frame.items():
            if tool_frame not in self.tool_frames:
                log_and_raise(f"tool frame {tool_frame!r} not found in configured tool frames")
            if criteria.device_cfg != self.device_cfg:
                log_and_raise(f"device_cfg mismatch: {self.device_cfg} != {criteria.device_cfg}")
            link_index = self.tool_frames.index(tool_frame)
            for name in self.tensor_field_names():
                getattr(self, name)[link_index].copy_(getattr(criteria, name))
