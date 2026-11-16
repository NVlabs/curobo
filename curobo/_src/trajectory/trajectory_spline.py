# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete cardinal cubic paths with uniform or optimized physical timing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise


@dataclass
class UniformTiming:
    """Physical seconds per cardinal spatial span."""

    #: Contiguous float32 intervals [B].
    knot_dt: torch.Tensor


@dataclass
class TimeOptTiming:
    """Clamped cubic squared speed x(s)=(ds/dt)^2 on normalized progress."""

    #: Contiguous float32 coefficients [B,32].
    controls: torch.Tensor
    #: Contiguous float32 integrated seconds [B,G], at uniform progress nodes.
    time: torch.Tensor


@dataclass
class SplineTrajectory:
    """Complete cubic trajectory with physical timing.

    Controls include all boundary support. CPU values support controller handoff;
    interpolation runs on CUDA. Sampling does not certify collision feasibility.
    """

    #: Complete uniform cardinal spatial controls [B,C,D], float32.
    control_points: torch.Tensor
    #: Joint names in the last control/output dimension's order.
    joint_names: list[str]
    #: Exactly one physical timing representation, on the controls' device.
    timing: UniformTiming | TimeOptTiming
    #: Spatial degree; only cubic is supported.
    degree: int = 3

    def __post_init__(self) -> None:
        """Check descriptor layouts without reading GPU values or synchronizing."""
        points = self.control_points
        if not isinstance(points, torch.Tensor) or points.ndim != 3:
            log_and_raise("control_points must have shape [B,C,D]")
        if (
            type(self.degree) is not int
            or self.degree != 3
            or min(points.shape) < 1
            or points.shape[1] < 4
        ):
            log_and_raise("SplineTrajectory requires degree=3 and at least four controls")
        if points.device.type not in ("cpu", "cuda"):
            log_and_raise("SplineTrajectory supports CPU and CUDA storage")
        if any(not isinstance(name, str) or not name for name in self.joint_names):
            log_and_raise("joint_names must contain nonempty strings")
        if len(self.joint_names) != points.shape[-1] or len(set(self.joint_names)) != len(
            self.joint_names
        ):
            log_and_raise("joint_names must uniquely name every control coordinate")
        if not isinstance(self.timing, (UniformTiming, TimeOptTiming)):
            log_and_raise("timing must be UniformTiming or TimeOptTiming")
        tensors = [points]
        batch = points.shape[0]
        if isinstance(self.timing, UniformTiming):
            if self.timing.knot_dt.shape != (batch,):
                log_and_raise("knot_dt must have shape [B]")
            tensors.append(self.timing.knot_dt)
        else:
            if self.timing.controls.shape != (batch, 32):
                log_and_raise("TimeOpt controls must have shape [B,32]")
            time = self.timing.time
            if time.ndim != 2 or time.shape[0] != batch or time.shape[1] < 2:
                log_and_raise("TimeOpt time must have shape [B,G] with G >= 2")
            tensors.extend((self.timing.controls, time))
        if any(
            t.dtype != torch.float32 or t.device != points.device or not t.is_contiguous()
            for t in tensors
        ):
            log_and_raise("All spline buffers must be contiguous float32 on the same device")
        self.joint_names = list(self.joint_names)

    @property
    def duration(self) -> torch.Tensor:
        """Return physical duration [B] on the trajectory device."""
        if isinstance(self.timing, UniformTiming):
            return self.timing.knot_dt * (self.control_points.shape[1] - self.degree)
        return self.timing.time[:, -1]

    def _copy(self, device: torch.device, copy: bool) -> SplineTrajectory:
        """Copy all spatial and timing buffers together on the current stream."""

        def move(value: torch.Tensor) -> torch.Tensor:
            return value.to(device=device, copy=copy)

        timing = self.timing
        timing = (
            UniformTiming(move(timing.knot_dt))
            if isinstance(timing, UniformTiming)
            else TimeOptTiming(move(timing.controls), move(timing.time))
        )
        return SplineTrajectory(move(self.control_points), self.joint_names, timing, self.degree)

    def clone(self) -> SplineTrajectory:
        """Retain independent controls, timing, and names on the current device."""
        return self._copy(self.control_points.device, True)

    def to(self, device: torch.device | str) -> SplineTrajectory:
        """Transfer the complete value; a same-device call may return self."""
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device == self.control_points.device:
            return self
        return self._copy(device, True)

    def _select(self, indices: torch.Tensor) -> SplineTrajectory:
        """Gather lanes while keeping complete path and timing selection aligned."""
        timing = self.timing
        timing = (
            UniformTiming(timing.knot_dt[indices].contiguous())
            if isinstance(timing, UniformTiming)
            else TimeOptTiming(
                timing.controls[indices].contiguous(), timing.time[indices].contiguous()
            )
        )
        return SplineTrajectory(
            self.control_points[indices].contiguous(), self.joint_names, timing
        )

    def _copy_lanes(self, source: SplineTrajectory, mask: torch.Tensor) -> None:
        """Replace compatible result lanes, including their time laws."""
        if (
            type(self.timing) is not type(source.timing)
            or self.control_points.shape != source.control_points.shape
            or self.joint_names != source.joint_names
            or self.control_points.device != source.control_points.device
            or (
                isinstance(self.timing, TimeOptTiming)
                and self.timing.time.shape != source.timing.time.shape
            )
        ):
            log_and_raise("Cannot combine incompatible spline result layouts")
        self.control_points[mask] = source.control_points[mask]
        if isinstance(self.timing, UniformTiming):
            self.timing.knot_dt[mask] = source.timing.knot_dt[mask]
        else:
            self.timing.controls[mask] = source.timing.controls[mask]
            self.timing.time[mask] = source.timing.time[mask]

    def interpolate(self, dt: float) -> JointState:
        """Interpolate one trajectory [1,N,D], including endpoints, with actual dt.

        dt is the maximum interval in seconds. This checked allocating operation
        reads duration and must run outside CUDA capture and control loops.
        """
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            log_and_raise("dt must be finite and positive")
        if self.control_points.device.type != "cuda":
            log_and_raise("Interpolation requires CUDA; CPU values support controller handoff")
        if self.control_points.shape[0] != 1:
            log_and_raise("Select a single trajectory before interpolation")
        with torch.cuda.device(self.control_points.device):
            if torch.cuda.is_current_stream_capturing():
                log_and_raise("interpolate allocates checked output; call it outside CUDA capture")
        duration = float(self.duration.item())
        if not math.isfinite(duration) or duration < 0:
            log_and_raise("Trajectory duration must be finite and nonnegative")
        if not bool(torch.isfinite(self.control_points).all()):
            log_and_raise("Spline control points must be finite")
        if isinstance(self.timing, UniformTiming):
            if duration <= 0:
                log_and_raise("Uniform knot_dt must be positive")
        else:
            time = self.timing.time
            if (
                not bool(torch.isfinite(time).all())
                or float(time[0, 0]) != 0
                or not bool((time[:, 1:] >= time[:, :-1]).all())
                or not bool(torch.isfinite(self.timing.controls).all())
                or not bool((self.timing.controls >= 0).all())
                or (duration > 0 and not bool((self.timing.controls > 0).any()))
            ):
                log_and_raise("TimeOpt timing must be finite, monotone, and nonnegative")
            if duration == 0 and not bool(
                (self.control_points == self.control_points[:, :1]).all()
            ):
                log_and_raise("Zero duration requires a stationary spatial path")
        intervals = max(1, math.ceil(duration / dt)) if duration else 0
        if (intervals + 1) * 4 * self.control_points.shape[-1] > 2**31 - 1:
            log_and_raise("Requested interpolation exceeds the supported sample count")
        actual_dt = duration / intervals if intervals else 0.0
        from curobo._src.curobolib.backends.cuda_core_backend.trajectory_timeopt import (
            interpolate_spline,
        )

        values = interpolate_spline(self, intervals + 1, actual_dt)
        return JointState(
            position=values[0],
            velocity=values[1],
            acceleration=values[2],
            jerk=values[3],
            joint_names=self.joint_names.copy(),
            dt=self.control_points.new_tensor([actual_dt]) if intervals else None,
        )
