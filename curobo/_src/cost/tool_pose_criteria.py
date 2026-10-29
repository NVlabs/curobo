# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Criteria that select and scale tool-pose residual subspaces."""

from __future__ import annotations

# Standard Library
from dataclasses import InitVar, dataclass
from enum import IntEnum
from math import isfinite
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple, Union

# Third Party
import torch

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise

TensorLike = Union[torch.Tensor, Sequence[float]]
ErrorFrameLike = Union[torch.Tensor, "PoseErrorFrame", int]
DirectionLike = Union[str, TensorLike]

_PROJECTOR_ATOL = 1.0e-5

_AXIS_DIRECTIONS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


class PoseErrorFrame(IntEnum):
    """Coordinate frame in which pose errors and their projectors are expressed."""

    WORLD = 0
    GOAL = 1


def _as_float_tensor(
    value: TensorLike,
    shape: Tuple[int, ...],
    name: str,
    device_cfg: DeviceCfg,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device_cfg.device, dtype=device_cfg.dtype).clone()
    if tensor.shape != shape:
        log_and_raise(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        log_and_raise(f"{name} must contain only finite values")
    return tensor


def _direction_to_orthogonal_projector(
    direction: DirectionLike,
    name: str,
    device_cfg: DeviceCfg,
) -> torch.Tensor:
    """Return the projector orthogonal to one nonzero 3D direction."""
    if isinstance(direction, str):
        if direction not in _AXIS_DIRECTIONS:
            log_and_raise(f"Invalid {name}: {direction}, must be one of {tuple(_AXIS_DIRECTIONS)}")
        direction = _AXIS_DIRECTIONS[direction]
    direction_tensor = _as_float_tensor(direction, (3,), name, device_cfg)
    direction_norm = torch.linalg.vector_norm(direction_tensor)
    if not bool(direction_norm > 1.0e-8):
        log_and_raise(f"{name} must be nonzero")
    unit_direction = direction_tensor / direction_norm
    identity = torch.eye(3, **device_cfg.as_torch_dict())
    return identity - torch.outer(unit_direction, unit_direction)


def _as_projector(
    value: TensorLike,
    name: str,
    device_cfg: DeviceCfg,
) -> torch.Tensor:
    projector = _as_float_tensor(value, (3, 3), name, device_cfg)
    if not torch.allclose(projector, projector.transpose(0, 1), atol=_PROJECTOR_ATOL, rtol=0.0):
        log_and_raise(f"{name} must be symmetric")
    if not torch.allclose(projector @ projector, projector, atol=_PROJECTOR_ATOL, rtol=0.0):
        log_and_raise(f"{name} must be idempotent")
    return projector


def _as_axis_scale(
    value: TensorLike,
    name: str,
    device_cfg: DeviceCfg,
) -> torch.Tensor:
    scale = _as_float_tensor(value, (3,), name, device_cfg)
    if not bool(torch.all(scale > 0.0)):
        log_and_raise(f"{name} must contain strictly positive values")
    return scale


def _as_tolerance(
    value: TensorLike,
    name: str,
    device_cfg: DeviceCfg,
) -> torch.Tensor:
    tolerance = _as_float_tensor(value, (2,), name, device_cfg)
    if not bool(torch.all(tolerance >= 0.0)):
        log_and_raise(f"{name} must contain nonnegative values")
    return tolerance


def _as_error_frame(value: ErrorFrameLike, device_cfg: DeviceCfg) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            log_and_raise(f"error_frame must be a scalar, got shape {tuple(value.shape)}")
        frame_value = value.item()
    else:
        frame_value = value
    if isinstance(frame_value, PoseErrorFrame):
        frame = frame_value
    elif isinstance(frame_value, int) and not isinstance(frame_value, bool):
        try:
            frame = PoseErrorFrame(frame_value)
        except ValueError:
            log_and_raise(f"error_frame must be WORLD or GOAL, got {frame_value}")
    else:
        log_and_raise(f"error_frame must be a PoseErrorFrame or int, got {frame_value}")
    return torch.tensor(int(frame), device=device_cfg.device, dtype=torch.int32)


def _legacy_bool_to_error_frame(value: Union[torch.Tensor, bool]) -> PoseErrorFrame:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            log_and_raise(
                "project_distance_to_goal must be a bool or a scalar tensor, "
                + f"got shape {tuple(value.shape)}"
            )
        value = value.item()
    if isinstance(value, bool):
        return PoseErrorFrame.GOAL if value else PoseErrorFrame.WORLD
    if value in (0, 1):
        return PoseErrorFrame.GOAL if int(value) == 1 else PoseErrorFrame.WORLD
    log_and_raise(f"project_distance_to_goal must be boolean, got {value}")


def _axis_factor_to_projector_and_scale(
    factor: TensorLike,
    name: str,
    device_cfg: DeviceCfg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    factor_tensor = _as_float_tensor(factor, (3,), name, device_cfg)
    if not bool(torch.all(factor_tensor >= 0.0)):
        log_and_raise(f"{name} must contain nonnegative values")
    enabled = factor_tensor != 0.0
    projector = torch.diag(enabled.to(dtype=device_cfg.dtype))
    scale = torch.where(enabled, torch.abs(factor_tensor), torch.ones_like(factor_tensor))
    return projector, scale


def _legacy_factor_to_projectors_and_scales(
    factor: TensorLike,
    name: str,
    device_cfg: DeviceCfg,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    factor_tensor = _as_float_tensor(factor, (6,), name, device_cfg)
    if not bool(torch.all(factor_tensor >= 0.0)):
        log_and_raise(f"{name} must contain nonnegative values")
    position_projector, position_scale = _axis_factor_to_projector_and_scale(
        factor_tensor[:3], name, device_cfg
    )
    rotation_projector, rotation_scale = _axis_factor_to_projector_and_scale(
        factor_tensor[3:], name, device_cfg
    )
    return position_projector, rotation_projector, position_scale, rotation_scale


def _scaled_factor(
    factor: TensorLike,
    scale: float,
    name: str,
    device_cfg: DeviceCfg,
) -> torch.Tensor:
    if not isfinite(scale) or scale < 0.0:
        log_and_raise(f"{name} must be finite and nonnegative, got {scale}")
    return _as_float_tensor(factor, (3,), name, device_cfg) * scale


@dataclass
class ToolPoseCriteria:
    """Projectors, scales, and tolerances for one tool-pose residual.

    Legacy six-axis factors are accepted only as constructor inputs. They are
    converted immediately to diagonal projectors and positive axis scales and
    are not retained as a second runtime representation.
    """

    terminal_position_projector: Optional[TensorLike] = None
    terminal_rotation_projector: Optional[TensorLike] = None
    path_position_projector: Optional[TensorLike] = None
    path_rotation_projector: Optional[TensorLike] = None

    terminal_position_axis_scale: Optional[TensorLike] = None
    terminal_rotation_axis_scale: Optional[TensorLike] = None
    path_position_axis_scale: Optional[TensorLike] = None
    path_rotation_axis_scale: Optional[TensorLike] = None

    terminal_tolerance: Optional[TensorLike] = None
    path_tolerance: Optional[TensorLike] = None
    error_frame: Optional[ErrorFrameLike] = None

    device_cfg: DeviceCfg = DeviceCfg()

    terminal_pose_axes_weight_factor: InitVar[Optional[TensorLike]] = None
    non_terminal_pose_axes_weight_factor: InitVar[Optional[TensorLike]] = None
    terminal_pose_convergence_tolerance: InitVar[Optional[TensorLike]] = None
    non_terminal_pose_convergence_tolerance: InitVar[Optional[TensorLike]] = None
    project_distance_to_goal: InitVar[Optional[Union[torch.Tensor, bool]]] = None

    _TENSOR_FIELD_NAMES: ClassVar[Tuple[str, ...]] = (
        "terminal_position_projector",
        "terminal_rotation_projector",
        "path_position_projector",
        "path_rotation_projector",
        "terminal_position_axis_scale",
        "terminal_rotation_axis_scale",
        "path_position_axis_scale",
        "path_rotation_axis_scale",
        "terminal_tolerance",
        "path_tolerance",
        "error_frame",
    )

    def __post_init__(
        self,
        terminal_pose_axes_weight_factor: Optional[TensorLike],
        non_terminal_pose_axes_weight_factor: Optional[TensorLike],
        terminal_pose_convergence_tolerance: Optional[TensorLike],
        non_terminal_pose_convergence_tolerance: Optional[TensorLike],
        project_distance_to_goal: Optional[Union[torch.Tensor, bool]],
    ) -> None:
        """Normalize canonical fields and consume legacy construction inputs."""
        identity = torch.eye(3, **self.device_cfg.as_torch_dict())
        zero_projector = torch.zeros((3, 3), **self.device_cfg.as_torch_dict())
        unit_scale = torch.ones(3, **self.device_cfg.as_torch_dict())
        zero_tolerance = torch.zeros(2, **self.device_cfg.as_torch_dict())

        terminal_fields = (
            self.terminal_position_projector,
            self.terminal_rotation_projector,
            self.terminal_position_axis_scale,
            self.terminal_rotation_axis_scale,
        )
        if terminal_pose_axes_weight_factor is not None:
            if any(value is not None for value in terminal_fields):
                log_and_raise(
                    "terminal_pose_axes_weight_factor cannot be combined with canonical "
                    "terminal projector or axis-scale fields"
                )
            (
                self.terminal_position_projector,
                self.terminal_rotation_projector,
                self.terminal_position_axis_scale,
                self.terminal_rotation_axis_scale,
            ) = _legacy_factor_to_projectors_and_scales(
                terminal_pose_axes_weight_factor,
                "terminal_pose_axes_weight_factor",
                self.device_cfg,
            )

        path_fields = (
            self.path_position_projector,
            self.path_rotation_projector,
            self.path_position_axis_scale,
            self.path_rotation_axis_scale,
        )
        if non_terminal_pose_axes_weight_factor is not None:
            if any(value is not None for value in path_fields):
                log_and_raise(
                    "non_terminal_pose_axes_weight_factor cannot be combined with canonical "
                    "path projector or axis-scale fields"
                )
            (
                self.path_position_projector,
                self.path_rotation_projector,
                self.path_position_axis_scale,
                self.path_rotation_axis_scale,
            ) = _legacy_factor_to_projectors_and_scales(
                non_terminal_pose_axes_weight_factor,
                "non_terminal_pose_axes_weight_factor",
                self.device_cfg,
            )

        if terminal_pose_convergence_tolerance is not None:
            if self.terminal_tolerance is not None:
                log_and_raise(
                    "terminal_pose_convergence_tolerance cannot be combined with "
                    "terminal_tolerance"
                )
            self.terminal_tolerance = terminal_pose_convergence_tolerance
        if non_terminal_pose_convergence_tolerance is not None:
            if self.path_tolerance is not None:
                log_and_raise(
                    "non_terminal_pose_convergence_tolerance cannot be combined with "
                    "path_tolerance"
                )
            self.path_tolerance = non_terminal_pose_convergence_tolerance
        if project_distance_to_goal is not None:
            if self.error_frame is not None:
                log_and_raise("project_distance_to_goal cannot be combined with error_frame")
            self.error_frame = _legacy_bool_to_error_frame(project_distance_to_goal)

        self.terminal_position_projector = _as_projector(
            identity
            if self.terminal_position_projector is None
            else self.terminal_position_projector,
            "terminal_position_projector",
            self.device_cfg,
        )
        self.terminal_rotation_projector = _as_projector(
            identity
            if self.terminal_rotation_projector is None
            else self.terminal_rotation_projector,
            "terminal_rotation_projector",
            self.device_cfg,
        )
        self.path_position_projector = _as_projector(
            zero_projector
            if self.path_position_projector is None
            else self.path_position_projector,
            "path_position_projector",
            self.device_cfg,
        )
        self.path_rotation_projector = _as_projector(
            zero_projector
            if self.path_rotation_projector is None
            else self.path_rotation_projector,
            "path_rotation_projector",
            self.device_cfg,
        )

        self.terminal_position_axis_scale = _as_axis_scale(
            unit_scale
            if self.terminal_position_axis_scale is None
            else self.terminal_position_axis_scale,
            "terminal_position_axis_scale",
            self.device_cfg,
        )
        self.terminal_rotation_axis_scale = _as_axis_scale(
            unit_scale
            if self.terminal_rotation_axis_scale is None
            else self.terminal_rotation_axis_scale,
            "terminal_rotation_axis_scale",
            self.device_cfg,
        )
        self.path_position_axis_scale = _as_axis_scale(
            unit_scale if self.path_position_axis_scale is None else self.path_position_axis_scale,
            "path_position_axis_scale",
            self.device_cfg,
        )
        self.path_rotation_axis_scale = _as_axis_scale(
            unit_scale if self.path_rotation_axis_scale is None else self.path_rotation_axis_scale,
            "path_rotation_axis_scale",
            self.device_cfg,
        )

        self.terminal_tolerance = _as_tolerance(
            zero_tolerance if self.terminal_tolerance is None else self.terminal_tolerance,
            "terminal_tolerance",
            self.device_cfg,
        )
        self.path_tolerance = _as_tolerance(
            zero_tolerance if self.path_tolerance is None else self.path_tolerance,
            "path_tolerance",
            self.device_cfg,
        )
        self.error_frame = _as_error_frame(
            PoseErrorFrame.WORLD if self.error_frame is None else self.error_frame,
            self.device_cfg,
        )

    @classmethod
    def tensor_field_names(cls) -> Tuple[str, ...]:
        """Return canonical tensor fields in their stable copy order."""
        return cls._TENSOR_FIELD_NAMES

    def clone(self) -> ToolPoseCriteria:
        """Return an independent criteria value with the same canonical tensors."""
        return ToolPoseCriteria(
            terminal_position_projector=self.terminal_position_projector.clone(),
            terminal_rotation_projector=self.terminal_rotation_projector.clone(),
            path_position_projector=self.path_position_projector.clone(),
            path_rotation_projector=self.path_rotation_projector.clone(),
            terminal_position_axis_scale=self.terminal_position_axis_scale.clone(),
            terminal_rotation_axis_scale=self.terminal_rotation_axis_scale.clone(),
            path_position_axis_scale=self.path_position_axis_scale.clone(),
            path_rotation_axis_scale=self.path_rotation_axis_scale.clone(),
            terminal_tolerance=self.terminal_tolerance.clone(),
            path_tolerance=self.path_tolerance.clone(),
            error_frame=self.error_frame.clone(),
            device_cfg=self.device_cfg,
        )

    def to(self, device_cfg: DeviceCfg) -> ToolPoseCriteria:
        """Return criteria on ``device_cfg``, preserving this object when already aligned."""
        if self.device_cfg == device_cfg:
            return self
        values = {}
        for name in self.tensor_field_names():
            dtype = torch.int32 if name == "error_frame" else device_cfg.dtype
            values[name] = getattr(self, name).to(device=device_cfg.device, dtype=dtype)
        return ToolPoseCriteria(device_cfg=device_cfg, **values)

    def copy_(self, other: ToolPoseCriteria) -> ToolPoseCriteria:
        """Copy canonical values without changing any tensor address."""
        if self.device_cfg != other.device_cfg:
            log_and_raise(f"device_cfg mismatch: {self.device_cfg} != {other.device_cfg}")
        for name in self.tensor_field_names():
            getattr(self, name).copy_(getattr(other, name))
        return self

    def copy_path_constraint_(self, other: ToolPoseCriteria) -> ToolPoseCriteria:
        """Copy ``other`` path fields into both fixed-address evaluation slots."""
        if self.device_cfg != other.device_cfg:
            log_and_raise(f"device_cfg mismatch: {self.device_cfg} != {other.device_cfg}")
        for terminal_name, path_name in (
            ("terminal_position_projector", "path_position_projector"),
            ("terminal_rotation_projector", "path_rotation_projector"),
            ("terminal_position_axis_scale", "path_position_axis_scale"),
            ("terminal_rotation_axis_scale", "path_rotation_axis_scale"),
            ("terminal_tolerance", "path_tolerance"),
        ):
            source = getattr(other, path_name)
            getattr(self, terminal_name).copy_(source)
            getattr(self, path_name).copy_(source)
        self.error_frame.copy_(other.error_frame)
        return self

    def as_path_constraint(self) -> ToolPoseCriteria:
        """Copy path criteria into terminal fields for horizon-one evaluation."""
        return ToolPoseCriteria(
            terminal_position_projector=self.path_position_projector.clone(),
            terminal_rotation_projector=self.path_rotation_projector.clone(),
            path_position_projector=self.path_position_projector.clone(),
            path_rotation_projector=self.path_rotation_projector.clone(),
            terminal_position_axis_scale=self.path_position_axis_scale.clone(),
            terminal_rotation_axis_scale=self.path_rotation_axis_scale.clone(),
            path_position_axis_scale=self.path_position_axis_scale.clone(),
            path_rotation_axis_scale=self.path_rotation_axis_scale.clone(),
            terminal_tolerance=self.path_tolerance.clone(),
            path_tolerance=self.path_tolerance.clone(),
            error_frame=self.error_frame.clone(),
            device_cfg=self.device_cfg,
        )

    @staticmethod
    def track_position(
        xyz: TensorLike = (1.0, 1.0, 1.0),
        *,
        error_frame: ErrorFrameLike = PoseErrorFrame.WORLD,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> ToolPoseCriteria:
        """Track position at every state while leaving orientation unconstrained."""
        position_projector, position_scale = _axis_factor_to_projector_and_scale(
            xyz, "xyz", device_cfg
        )
        zero = torch.zeros((3, 3), **device_cfg.as_torch_dict())
        return ToolPoseCriteria(
            terminal_position_projector=position_projector,
            terminal_rotation_projector=zero,
            path_position_projector=position_projector,
            path_rotation_projector=zero,
            terminal_position_axis_scale=position_scale,
            path_position_axis_scale=position_scale,
            error_frame=error_frame,
            device_cfg=device_cfg,
        )

    @staticmethod
    def track_orientation(
        rpy: TensorLike = (0.001, 0.001, 0.001),
        non_terminal_scale: float = 1.0,
        *,
        error_frame: ErrorFrameLike = PoseErrorFrame.WORLD,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> ToolPoseCriteria:
        """Track orientation at every state while leaving position unconstrained."""
        terminal_projector, terminal_scale = _axis_factor_to_projector_and_scale(
            rpy, "rpy", device_cfg
        )
        path_projector, path_scale = _axis_factor_to_projector_and_scale(
            _scaled_factor(rpy, non_terminal_scale, "non_terminal_scale", device_cfg),
            "rpy",
            device_cfg,
        )
        zero = torch.zeros((3, 3), **device_cfg.as_torch_dict())
        return ToolPoseCriteria(
            terminal_position_projector=zero,
            terminal_rotation_projector=terminal_projector,
            path_position_projector=zero,
            path_rotation_projector=path_projector,
            terminal_rotation_axis_scale=terminal_scale,
            path_rotation_axis_scale=path_scale,
            error_frame=error_frame,
            device_cfg=device_cfg,
        )

    @staticmethod
    def track_pose(
        xyz: TensorLike = (1.0, 1.0, 1.0),
        rpy: TensorLike = (1.0, 1.0, 1.0),
        non_terminal_scale: float = 0.1,
        *,
        error_frame: ErrorFrameLike = PoseErrorFrame.WORLD,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> ToolPoseCriteria:
        """Track position and orientation with independently scaled residual axes."""
        terminal_position_projector, terminal_position_scale = _axis_factor_to_projector_and_scale(
            xyz, "xyz", device_cfg
        )
        terminal_rotation_projector, terminal_rotation_scale = _axis_factor_to_projector_and_scale(
            rpy, "rpy", device_cfg
        )
        path_position_projector, path_position_scale = _axis_factor_to_projector_and_scale(
            _scaled_factor(xyz, non_terminal_scale, "non_terminal_scale", device_cfg),
            "xyz",
            device_cfg,
        )
        path_rotation_projector, path_rotation_scale = _axis_factor_to_projector_and_scale(
            _scaled_factor(rpy, non_terminal_scale, "non_terminal_scale", device_cfg),
            "rpy",
            device_cfg,
        )
        return ToolPoseCriteria(
            terminal_position_projector=terminal_position_projector,
            terminal_rotation_projector=terminal_rotation_projector,
            path_position_projector=path_position_projector,
            path_rotation_projector=path_rotation_projector,
            terminal_position_axis_scale=terminal_position_scale,
            terminal_rotation_axis_scale=terminal_rotation_scale,
            path_position_axis_scale=path_position_scale,
            path_rotation_axis_scale=path_rotation_scale,
            error_frame=error_frame,
            device_cfg=device_cfg,
        )

    @staticmethod
    def track_position_and_orientation(
        xyz: TensorLike = (1.0, 1.0, 1.0),
        rpy: TensorLike = (1.0, 1.0, 1.0),
        non_terminal_scale: float = 0.1,
        *,
        error_frame: ErrorFrameLike = PoseErrorFrame.WORLD,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> ToolPoseCriteria:
        """Compatibility alias for :meth:`track_pose`."""
        return ToolPoseCriteria.track_pose(
            xyz=xyz,
            rpy=rpy,
            non_terminal_scale=non_terminal_scale,
            error_frame=error_frame,
            device_cfg=device_cfg,
        )

    @staticmethod
    def linear_motion(
        axis: Optional[DirectionLike] = None,
        non_terminal_scale: float = 1.0,
        project_distance_to_goal: Optional[Union[torch.Tensor, bool]] = None,
        *,
        direction: Optional[TensorLike] = None,
        rotation_axis: Optional[DirectionLike] = None,
        error_frame: Optional[ErrorFrameLike] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> ToolPoseCriteria:
        """Constrain path position to a line with optional free axial rotation.

        ``direction`` is the preferred arbitrary-direction API. ``axis`` and
        ``project_distance_to_goal`` retain compatibility with the former
        axis-aligned factory. When provided, ``rotation_axis`` leaves rotation
        about that arbitrary axis free while constraining orthogonal rotation.
        Omitting it retains full path-orientation tracking.
        """
        if axis is not None and direction is not None:
            log_and_raise("axis and direction cannot both be provided")

        uses_axis_compatibility = direction is None and (axis is None or isinstance(axis, str))
        if direction is None:
            if axis is None:
                direction = (0.0, 0.0, 1.0)
            elif isinstance(axis, str):
                if axis not in _AXIS_DIRECTIONS:
                    log_and_raise(f"Invalid axis: {axis}, must be 'x', 'y', or 'z'")
                direction = _AXIS_DIRECTIONS[axis]
            else:
                direction = axis

        if project_distance_to_goal is not None and error_frame is not None:
            log_and_raise("project_distance_to_goal and error_frame cannot both be provided")
        if project_distance_to_goal is not None:
            selected_error_frame: ErrorFrameLike = _legacy_bool_to_error_frame(
                project_distance_to_goal
            )
        elif error_frame is not None:
            selected_error_frame = error_frame
        elif uses_axis_compatibility:
            selected_error_frame = PoseErrorFrame.GOAL
        else:
            selected_error_frame = PoseErrorFrame.WORLD

        identity = torch.eye(3, **device_cfg.as_torch_dict())
        path_position_projector = _direction_to_orthogonal_projector(
            direction,
            "direction",
            device_cfg,
        )
        path_rotation_projector = (
            identity
            if rotation_axis is None
            else _direction_to_orthogonal_projector(
                rotation_axis,
                "rotation_axis",
                device_cfg,
            )
        )

        if not isfinite(non_terminal_scale) or non_terminal_scale < 0.0:
            log_and_raise(
                "non_terminal_scale must be finite and nonnegative, " + f"got {non_terminal_scale}"
            )
        if non_terminal_scale == 0.0:
            path_position_projector = torch.zeros_like(identity)
            path_rotation_projector = torch.zeros_like(identity)
            path_scale = torch.ones(3, **device_cfg.as_torch_dict())
        else:
            path_scale = torch.full(
                (3,), non_terminal_scale, device=device_cfg.device, dtype=device_cfg.dtype
            )

        return ToolPoseCriteria(
            terminal_position_projector=identity,
            terminal_rotation_projector=identity,
            path_position_projector=path_position_projector,
            path_rotation_projector=path_rotation_projector,
            path_position_axis_scale=path_scale,
            path_rotation_axis_scale=path_scale,
            error_frame=selected_error_frame,
            device_cfg=device_cfg,
        )

    @staticmethod
    def disabled(*, device_cfg: DeviceCfg = DeviceCfg()) -> ToolPoseCriteria:
        """Disable terminal and path pose residuals while retaining valid scales."""
        zero = torch.zeros((3, 3), **device_cfg.as_torch_dict())
        return ToolPoseCriteria(
            terminal_position_projector=zero,
            terminal_rotation_projector=zero,
            path_position_projector=zero,
            path_rotation_projector=zero,
            device_cfg=device_cfg,
        )


@dataclass
class StackedToolPoseCriteria:
    """Fixed-shape canonical criteria tensors for multiple tool frames."""

    tool_frames: List[str]
    terminal_position_projector: torch.Tensor
    terminal_rotation_projector: torch.Tensor
    path_position_projector: torch.Tensor
    path_rotation_projector: torch.Tensor
    terminal_position_axis_scale: torch.Tensor
    terminal_rotation_axis_scale: torch.Tensor
    path_position_axis_scale: torch.Tensor
    path_rotation_axis_scale: torch.Tensor
    terminal_tolerance: torch.Tensor
    path_tolerance: torch.Tensor
    error_frame: torch.Tensor
    device_cfg: DeviceCfg = DeviceCfg()

    _TENSOR_FIELD_NAMES: ClassVar[Tuple[str, ...]] = ToolPoseCriteria._TENSOR_FIELD_NAMES

    @classmethod
    def tensor_field_names(cls) -> Tuple[str, ...]:
        """Return canonical tensor fields in their stable copy order."""
        return cls._TENSOR_FIELD_NAMES

    @staticmethod
    def from_tool_pose_criteria(
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ) -> StackedToolPoseCriteria:
        """Stack per-frame criteria without retaining a second representation."""
        if not tool_pose_criteria:
            log_and_raise("tool_pose_criteria must contain at least one tool frame")
        tool_frames = list(tool_pose_criteria.keys())
        first = tool_pose_criteria[tool_frames[0]]
        for link_name in tool_frames:
            if tool_pose_criteria[link_name].device_cfg != first.device_cfg:
                log_and_raise(
                    f"device_cfg mismatch for {link_name}: "
                    + f"{tool_pose_criteria[link_name].device_cfg} != {first.device_cfg}"
                )
        values = {
            name: torch.stack([getattr(tool_pose_criteria[frame], name) for frame in tool_frames])
            for name in StackedToolPoseCriteria.tensor_field_names()
        }
        return StackedToolPoseCriteria(
            tool_frames=tool_frames,
            device_cfg=first.device_cfg,
            **values,
        )

    def __post_init__(self) -> None:
        """Validate fixed stacked tensor shapes and canonical values."""
        num_links = len(self.tool_frames)
        expected_shapes = {
            "terminal_position_projector": (num_links, 3, 3),
            "terminal_rotation_projector": (num_links, 3, 3),
            "path_position_projector": (num_links, 3, 3),
            "path_rotation_projector": (num_links, 3, 3),
            "terminal_position_axis_scale": (num_links, 3),
            "terminal_rotation_axis_scale": (num_links, 3),
            "path_position_axis_scale": (num_links, 3),
            "path_rotation_axis_scale": (num_links, 3),
            "terminal_tolerance": (num_links, 2),
            "path_tolerance": (num_links, 2),
            "error_frame": (num_links,),
        }
        for name, expected_shape in expected_shapes.items():
            tensor = getattr(self, name)
            if tensor.shape != expected_shape:
                log_and_raise(
                    f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.device != self.device_cfg.device:
                log_and_raise(f"{name} must be on {self.device_cfg.device}, got {tensor.device}")
            if name != "error_frame" and tensor.dtype != self.device_cfg.dtype:
                log_and_raise(f"{name} must use {self.device_cfg.dtype}, got {tensor.dtype}")

        for name in (
            "terminal_position_projector",
            "terminal_rotation_projector",
            "path_position_projector",
            "path_rotation_projector",
        ):
            projector = getattr(self, name)
            if not bool(torch.isfinite(projector).all()):
                log_and_raise(f"{name} must contain only finite values")
            if not torch.allclose(
                projector, projector.transpose(-1, -2), atol=_PROJECTOR_ATOL, rtol=0.0
            ):
                log_and_raise(f"{name} must be symmetric")
            if not torch.allclose(
                projector @ projector, projector, atol=_PROJECTOR_ATOL, rtol=0.0
            ):
                log_and_raise(f"{name} must be idempotent")

        for name in (
            "terminal_position_axis_scale",
            "terminal_rotation_axis_scale",
            "path_position_axis_scale",
            "path_rotation_axis_scale",
        ):
            scale = getattr(self, name)
            if not bool(torch.isfinite(scale).all()) or not bool(torch.all(scale > 0.0)):
                log_and_raise(f"{name} must contain finite, strictly positive values")

        for name in ("terminal_tolerance", "path_tolerance"):
            tolerance = getattr(self, name)
            if not bool(torch.isfinite(tolerance).all()) or not bool(torch.all(tolerance >= 0.0)):
                log_and_raise(f"{name} must contain finite, nonnegative values")

        if self.error_frame.dtype != torch.int32:
            log_and_raise(f"error_frame must use torch.int32, got {self.error_frame.dtype}")
        if not bool(
            torch.all(
                (self.error_frame == int(PoseErrorFrame.WORLD))
                | (self.error_frame == int(PoseErrorFrame.GOAL))
            )
        ):
            log_and_raise("error_frame must contain only WORLD or GOAL values")

    def clone(self) -> StackedToolPoseCriteria:
        """Return an independent stacked criteria value."""
        return StackedToolPoseCriteria(
            tool_frames=self.tool_frames.copy(),
            device_cfg=self.device_cfg,
            **{name: getattr(self, name).clone() for name in self.tensor_field_names()},
        )

    def copy_(self, other: StackedToolPoseCriteria) -> StackedToolPoseCriteria:
        """Copy a shape-compatible stack without changing tensor addresses."""
        if self.tool_frames != other.tool_frames:
            log_and_raise(f"tool_frames mismatch: {self.tool_frames} != {other.tool_frames}")
        if self.device_cfg != other.device_cfg:
            log_and_raise(f"device_cfg mismatch: {self.device_cfg} != {other.device_cfg}")
        for name in self.tensor_field_names():
            getattr(self, name).copy_(getattr(other, name))
        return self

    def update_tool_pose_criteria(
        self,
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ) -> None:
        """Update selected tool-frame rows without reallocating stacked tensors."""
        for link_name, criteria in tool_pose_criteria.items():
            if link_name not in self.tool_frames:
                log_and_raise(f"link_name {link_name} not found in tool_frames")
            if criteria.device_cfg != self.device_cfg:
                log_and_raise(f"device_cfg mismatch: {self.device_cfg} != {criteria.device_cfg}")
            link_idx = self.tool_frames.index(link_name)
            for name in self.tensor_field_names():
                getattr(self, name)[link_idx].copy_(getattr(criteria, name))
