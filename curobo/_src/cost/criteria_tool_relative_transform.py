# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Criteria for transforms relative to one moving tool frame."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import ClassVar, Dict, Tuple

# Third Party
import torch

# CuRobo
from curobo._src.cost.tool_pose_criteria import (
    TensorLike,
    _as_axis_scale,
    _as_projector,
    _as_tolerance,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


@dataclass
class ToolRelativeTransformCriteria:
    """Projected residual policy for one reference-relative tool transform."""

    position_projector: TensorLike | None = None
    rotation_projector: TensorLike | None = None
    position_axis_scale: TensorLike | None = None
    rotation_axis_scale: TensorLike | None = None
    tolerance: TensorLike | None = None
    device_cfg: DeviceCfg = DeviceCfg()

    _TENSOR_FIELD_NAMES: ClassVar[Tuple[str, ...]] = (
        "position_projector",
        "rotation_projector",
        "position_axis_scale",
        "rotation_axis_scale",
        "tolerance",
    )

    def __post_init__(self) -> None:
        """Normalize and validate the canonical criteria tensors."""
        identity = torch.eye(3, **self.device_cfg.as_torch_dict())
        unit_scale = torch.ones(3, **self.device_cfg.as_torch_dict())
        zero_tolerance = torch.zeros(2, **self.device_cfg.as_torch_dict())
        self.position_projector = _as_projector(
            identity if self.position_projector is None else self.position_projector,
            "position_projector",
            self.device_cfg,
        )
        self.rotation_projector = _as_projector(
            identity if self.rotation_projector is None else self.rotation_projector,
            "rotation_projector",
            self.device_cfg,
        )
        self.position_axis_scale = _as_axis_scale(
            unit_scale if self.position_axis_scale is None else self.position_axis_scale,
            "position_axis_scale",
            self.device_cfg,
        )
        self.rotation_axis_scale = _as_axis_scale(
            unit_scale if self.rotation_axis_scale is None else self.rotation_axis_scale,
            "rotation_axis_scale",
            self.device_cfg,
        )
        self.tolerance = _as_tolerance(
            zero_tolerance if self.tolerance is None else self.tolerance,
            "tolerance",
            self.device_cfg,
        )

    @classmethod
    def tensor_field_names(cls) -> Tuple[str, ...]:
        """Return canonical tensor fields in their stable copy order."""
        return cls._TENSOR_FIELD_NAMES

    @property
    def enabled(self) -> bool:
        """Return whether either projected residual component is active."""
        return bool(torch.any(self.position_projector != 0.0)) or bool(
            torch.any(self.rotation_projector != 0.0)
        )

    def clone(self) -> ToolRelativeTransformCriteria:
        """Return an independent copy of this criterion."""
        return ToolRelativeTransformCriteria(
            **{name: getattr(self, name).clone() for name in self.tensor_field_names()},
            device_cfg=self.device_cfg,
        )

    def to(self, device_cfg: DeviceCfg) -> ToolRelativeTransformCriteria:
        """Return this criterion on ``device_cfg``."""
        if self.device_cfg == device_cfg:
            return self
        return ToolRelativeTransformCriteria(
            **{
                name: getattr(self, name).to(
                    device=device_cfg.device,
                    dtype=device_cfg.dtype,
                )
                for name in self.tensor_field_names()
            },
            device_cfg=device_cfg,
        )

    def copy_(self, other: ToolRelativeTransformCriteria) -> ToolRelativeTransformCriteria:
        """Copy values without replacing any owned tensor storage."""
        if self.device_cfg != other.device_cfg:
            log_and_raise(f"device_cfg mismatch: {self.device_cfg} != {other.device_cfg}")
        for name in self.tensor_field_names():
            getattr(self, name).copy_(getattr(other, name))
        return self

    @staticmethod
    def rigid(*, device_cfg: DeviceCfg = DeviceCfg()) -> ToolRelativeTransformCriteria:
        """Constrain the complete relative position and orientation."""
        return ToolRelativeTransformCriteria(device_cfg=device_cfg)

    @staticmethod
    def track_position(*, device_cfg: DeviceCfg = DeviceCfg()) -> ToolRelativeTransformCriteria:
        """Constrain relative position while leaving orientation free."""
        return ToolRelativeTransformCriteria(
            rotation_projector=torch.zeros((3, 3), **device_cfg.as_torch_dict()),
            device_cfg=device_cfg,
        )

    @staticmethod
    def track_orientation(*, device_cfg: DeviceCfg = DeviceCfg()) -> ToolRelativeTransformCriteria:
        """Constrain relative orientation while leaving position free."""
        return ToolRelativeTransformCriteria(
            position_projector=torch.zeros((3, 3), **device_cfg.as_torch_dict()),
            device_cfg=device_cfg,
        )

    @staticmethod
    def disabled(*, device_cfg: DeviceCfg = DeviceCfg()) -> ToolRelativeTransformCriteria:
        """Disable both relative-transform residual components."""
        zero = torch.zeros((3, 3), **device_cfg.as_torch_dict())
        return ToolRelativeTransformCriteria(
            position_projector=zero,
            rotation_projector=zero,
            device_cfg=device_cfg,
        )


@dataclass
class StackedToolRelativeTransformCriteria:
    """Fixed device cache for per-tool relative-transform criteria."""

    tool_frames: list[str]
    position_projector: torch.Tensor
    rotation_projector: torch.Tensor
    position_axis_scale: torch.Tensor
    rotation_axis_scale: torch.Tensor
    tolerance: torch.Tensor
    relation_enabled: torch.Tensor
    device_cfg: DeviceCfg = DeviceCfg()

    _TENSOR_FIELD_NAMES: ClassVar[Tuple[str, ...]] = (
        "position_projector",
        "rotation_projector",
        "position_axis_scale",
        "rotation_axis_scale",
        "tolerance",
    )

    @staticmethod
    def from_criteria(
        criteria: Dict[str, ToolRelativeTransformCriteria],
    ) -> StackedToolRelativeTransformCriteria:
        """Stack an ordered per-frame criteria mapping."""
        if not criteria:
            log_and_raise("criteria must contain at least one tool frame")
        tool_frames = list(criteria)
        first = criteria[tool_frames[0]]
        for frame in tool_frames:
            if criteria[frame].device_cfg != first.device_cfg:
                log_and_raise(
                    f"device_cfg mismatch for {frame}: "
                    f"{criteria[frame].device_cfg} != {first.device_cfg}"
                )
        values = {
            name: torch.stack([getattr(criteria[frame], name) for frame in tool_frames])
            for name in StackedToolRelativeTransformCriteria._TENSOR_FIELD_NAMES
        }
        relation_enabled = torch.tensor(
            [int(criteria[frame].enabled) for frame in tool_frames],
            device=first.device_cfg.device,
            dtype=torch.int32,
        )
        return StackedToolRelativeTransformCriteria(
            tool_frames=tool_frames,
            relation_enabled=relation_enabled,
            device_cfg=first.device_cfg,
            **values,
        )

    def __post_init__(self) -> None:
        """Validate fixed cache shapes, devices, and dtypes."""
        num_links = len(self.tool_frames)
        expected_shapes = {
            "position_projector": (num_links, 3, 3),
            "rotation_projector": (num_links, 3, 3),
            "position_axis_scale": (num_links, 3),
            "rotation_axis_scale": (num_links, 3),
            "tolerance": (num_links, 2),
            "relation_enabled": (num_links,),
        }
        for name, expected_shape in expected_shapes.items():
            tensor = getattr(self, name)
            if tensor.shape != expected_shape:
                log_and_raise(
                    f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.device != self.device_cfg.device:
                log_and_raise(f"{name} must be on {self.device_cfg.device}, got {tensor.device}")
            expected_dtype = torch.int32 if name == "relation_enabled" else self.device_cfg.dtype
            if tensor.dtype != expected_dtype:
                log_and_raise(f"{name} must use {expected_dtype}, got {tensor.dtype}")

    def update_criteria(
        self,
        criteria: Dict[str, ToolRelativeTransformCriteria],
    ) -> None:
        """Update selected frame criteria without replacing cached tensors."""
        for frame, frame_criteria in criteria.items():
            if frame not in self.tool_frames:
                log_and_raise(f"Unknown tool frame {frame!r}; expected one of {self.tool_frames}")
            if frame_criteria.device_cfg != self.device_cfg:
                log_and_raise(
                    f"device_cfg mismatch for {frame}: "
                    f"{frame_criteria.device_cfg} != {self.device_cfg}"
                )
            frame_index = self.tool_frames.index(frame)
            for name in self._TENSOR_FIELD_NAMES:
                getattr(self, name)[frame_index].copy_(getattr(frame_criteria, name))
            self.relation_enabled[frame_index] = int(frame_criteria.enabled)
