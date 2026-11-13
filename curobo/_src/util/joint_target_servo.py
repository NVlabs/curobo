# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Fixed-rate joint target servo for active-joint command smoothing."""

from __future__ import annotations

from typing import Optional

import torch

from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise


class JointTargetServo:
    """Smooth a joint-space target into limit-respecting fixed-rate commands.

    This class is for a servo loop that tracks one current
    :class:`curobo._src.state.state_joint.JointState` target. It is not a trajectory queue and
    does not use timestamps. Call :func:`set_target` whenever a new target arrives, then call
    :func:`step` with the measured robot state once per fixed control tick to generate the next
    command.

    All command, target, and measured states must use the same ``joint_names`` and tensor shape
    ``(batch, dof)``. ``batch`` is the number of independent command streams, and ``dof`` is the
    number of active joints in ``joint_limits``.

    :func:`set_target` only stores a target. The command state is initialized either by
    :func:`reset` or by the first :func:`step`, which starts from the measured state.

    Args:
        joint_limits: Active joint limits as a
            :class:`curobo._src.robot.types.joint_limits.JointLimits` dataclass, such as the
            object returned by ``kinematics.get_joint_limits()``.
    """

    def __init__(
        self,
        joint_limits: JointLimits,
        servo_dt_s: float = 0.02,
        kp: float = 8.0,
        kd: float = 0.1,
    ) -> None:
        """Store active-joint limits and fixed-rate servo gains.

        Args:
            joint_limits: Active-joint :class:`JointLimits` with limit tensors of shape
                ``(2, dof)``.
            servo_dt_s: Fixed servo tick duration in seconds.
            kp: Proportional gain applied to position error.
            kd: Damping gain applied to measured velocity.

        Raises:
            ValueError: If ``servo_dt_s`` or any symmetric motion limit is invalid.
        """
        if servo_dt_s <= 0.0:
            log_and_raise("servo_dt_s must be positive")
        self.joint_names = tuple(joint_limits.joint_names)
        self._dof = len(self.joint_names)
        self.servo_dt_s = servo_dt_s
        self.kp = kp
        self.kd = kd

        position_limits = self._require_tensor(
            joint_limits.position,
            "joint_limits.position",
            shape=(2, self.dof),
        )
        self.q_min = position_limits[0]
        self.q_max = position_limits[1]
        self.qd_max = self._symmetric_max(joint_limits.velocity, "joint_limits.velocity")
        self.qdd_max = self._symmetric_max(joint_limits.acceleration, "joint_limits.acceleration")
        self.jerk_max = self._symmetric_max(joint_limits.jerk, "joint_limits.jerk")
        if torch.any(self.q_min > self.q_max):
            log_and_raise("joint position lower limits must be less than or equal to upper limits")
        if torch.any(self.qd_max <= 0.0):
            log_and_raise("joint velocity limits must be positive")
        if torch.any(self.qdd_max <= 0.0):
            log_and_raise("joint acceleration limits must be positive")
        if torch.any(self.jerk_max <= 0.0):
            log_and_raise("joint jerk limits must be positive")

        self.q_cmd: Optional[torch.Tensor] = None
        self.qd_cmd: Optional[torch.Tensor] = None
        self.qdd_cmd: Optional[torch.Tensor] = None
        self.jerk_cmd: Optional[torch.Tensor] = None
        self.q_target: Optional[torch.Tensor] = None
        self.qd_target: Optional[torch.Tensor] = None
        self._target_has_velocity = False

    @property
    def batch_size(self) -> int:
        """Return the number of independent command streams.

        Returns:
            Batch dimension of the initialized command state.

        Raises:
            ValueError: If no command state has been initialized.
        """
        self._require_initialized()
        return self.q_cmd.shape[0]

    @property
    def dof(self) -> int:
        """Return the number of active joints."""
        return self._dof

    def _require_initialized(self) -> None:
        """Check that command state exists before it is read."""
        if self.q_cmd is None:
            log_and_raise("JointTargetServo has no command state; call step() or reset() first")

    def _require_target(self) -> None:
        """Check that a target exists before stepping."""
        if self.q_target is None:
            log_and_raise("JointTargetServo has no target; call set_target() or reset() first")

    def _require_joint_state(self, state: JointState, name: str) -> None:
        """Validate that a joint state uses this servo's active joint order.

        Args:
            state: Joint state to validate.
            name: Argument name used in error messages.

        Raises:
            TypeError: If ``state`` is not a :class:`JointState`.
            ValueError: If joint names are missing or do not match ``joint_limits`` order.
        """
        if not isinstance(state, JointState):
            log_and_raise(f"{name} must be a JointState", exception_type=TypeError)
        if state.joint_names is None:
            log_and_raise(f"{name}.joint_names must be set")
        if tuple(state.joint_names) != self.joint_names:
            log_and_raise(
                f"{name}.joint_names must match joint_limits.joint_names: "
                f"expected {list(self.joint_names)}, got {state.joint_names}"
            )

    def _position_shape(self, position: torch.Tensor, name: str) -> tuple[int, ...]:
        """Validate the shape of a joint-space tensor.

        Args:
            position: Joint-space tensor expected to have shape ``(batch, dof)``.
            name: Argument name used in error messages.

        Returns:
            Validated tensor shape.

        Raises:
            ValueError: If ``position`` is not rank two, has an empty batch dimension, or has the
                wrong number of joints.
        """
        if not isinstance(position, torch.Tensor):
            return (0, self.dof)
        if position.ndim != 2:
            log_and_raise(
                f"{name} must have shape (batch, {self.dof}), got {tuple(position.shape)}"
            )
        if position.shape[0] < 1:
            log_and_raise(f"{name} batch dimension must be at least 1")
        if position.shape[1] != self.dof:
            log_and_raise(
                f"{name} must have shape (batch, {self.dof}), got {tuple(position.shape)}"
            )
        return tuple(position.shape)

    def _require_position(
        self,
        value: torch.Tensor,
        name: str,
        shape: Optional[tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Validate and clone a joint-space tensor for this servo.

        Args:
            value: Joint-space tensor expected to have shape ``(batch, dof)``.
            name: Argument name used in error messages.
            shape: Required shape. When ``None``, any non-empty batch size is accepted as long as
                the last dimension is ``dof``.

        Returns:
            Detached clone of ``value`` with the validated shape, dtype, and device.

        Raises:
            TypeError: If ``value`` is not a floating point tensor.
            ValueError: If shape, device, or finite-value checks fail.
        """
        if shape is None:
            shape = self._position_shape(value, name)
        return self._require_tensor(
            value,
            name,
            shape=shape,
            ref=self.q_min,
        )

    @staticmethod
    def _require_tensor(
        value: torch.Tensor,
        name: str,
        shape: tuple[int, ...],
        ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Validate and clone a finite floating point tensor.

        Args:
            value: Tensor to validate.
            name: Argument name used in error messages.
            shape: Required tensor shape.
            ref: Optional reference tensor that supplies required dtype and device.

        Returns:
            Detached clone of ``value``.

        Raises:
            TypeError: If ``value`` is not a floating point tensor.
            ValueError: If shape, device, or finite-value checks fail.
        """
        if not isinstance(value, torch.Tensor):
            log_and_raise(f"{name} must be a torch.Tensor", exception_type=TypeError)
        if tuple(value.shape) != shape:
            log_and_raise(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if not torch.is_floating_point(value):
            log_and_raise(f"{name} must have a floating point dtype", exception_type=TypeError)
        if ref is not None and value.dtype != ref.dtype:
            log_and_raise(
                f"{name} must have dtype {ref.dtype}, got {value.dtype}",
                exception_type=TypeError,
            )
        if ref is not None and value.device != ref.device:
            log_and_raise(f"{name} must be on device {ref.device}, got {value.device}")
        if not torch.isfinite(value).all():
            log_and_raise(f"{name} must contain only finite values")
        return value.detach().clone()

    def _symmetric_max(self, limits: torch.Tensor, name: str) -> torch.Tensor:
        """Convert lower and upper joint limits to usable symmetric magnitudes.

        Args:
            limits: Limit tensor with shape ``(2, dof)``.
            name: Argument name used in error messages.

        Returns:
            Tensor with shape ``(dof,)`` containing ``min(abs(lower), abs(upper))``.
        """
        limits = self._require_tensor(limits, name, shape=(2, self.dof), ref=self.q_min)
        return torch.minimum(torch.abs(limits[0]), torch.abs(limits[1]))

    def reset(self, command_state: JointState) -> None:
        """Replace command state with a known joint position and optional velocity.

        Use this when starting the servo from the robot's current state, or after a discontinuity
        where the previous command state should no longer influence acceleration and jerk limits.
        The reset state also becomes the current target, so a following :func:`step` holds that
        state until :func:`set_target` is called.

        Args:
            command_state: Command state with position shape ``(batch, dof)`` and joint names
                matching ``joint_limits``. When velocity is omitted, command velocity is
                initialized to zero.

        Raises:
            TypeError: If ``command_state`` is not a :class:`JointState` or contains non-tensor
                fields.
            ValueError: If joint names, shape, dtype, device, or finite-value checks fail.
        """
        self._require_joint_state(command_state, "command_state")
        self.q_cmd = self._require_position(
            command_state.position,
            "command_state.position",
        )
        self.qd_cmd = self._velocity_or_zero(
            command_state.velocity, "command_state.velocity", self.q_cmd
        )
        self.qdd_cmd = torch.zeros_like(self.q_cmd)
        self.jerk_cmd = torch.zeros_like(self.q_cmd)
        self.q_target = self.q_cmd.clone()
        self.qd_target = self.qd_cmd.clone()
        self._target_has_velocity = command_state.velocity is not None

    def set_target(self, target_state: JointState) -> None:
        """Replace the current target used by future :func:`step` calls.

        The servo stores only this target; it does not queue previous targets or initialize command
        state from the target.

        Args:
            target_state: Target state with position shape ``(batch, dof)`` and joint names
                matching ``joint_limits``. When velocity is omitted, the servo treats the target as
                a position-only target and applies a stopping limit.

        Raises:
            TypeError: If ``target_state`` is not a :class:`JointState` or contains non-tensor
                fields.
            ValueError: If joint names, shape, dtype, device, or finite-value checks fail.
        """
        self._require_joint_state(target_state, "target_state")
        if self.q_cmd is not None:
            shape = tuple(self.q_cmd.shape)
        elif self.q_target is not None:
            shape = tuple(self.q_target.shape)
        else:
            shape = None
        q_target = self._require_position(
            target_state.position,
            "target_state.position",
            shape=shape,
        )
        self.q_target = q_target
        self.qd_target = self._velocity_or_zero(
            target_state.velocity, "target_state.velocity", q_target
        )
        self._target_has_velocity = target_state.velocity is not None

    def command_state(self) -> JointState:
        """Return a copy of the current command state.

        Returns:
            :class:`JointState` with position, velocity, acceleration, and jerk tensors of shape
            ``(batch, dof)`` and active joint names from ``joint_limits``.

        Raises:
            ValueError: If no command state has been initialized.
        """
        self._require_initialized()
        return JointState(
            position=self.q_cmd.clone(),
            velocity=self.qd_cmd.clone(),
            acceleration=self.qdd_cmd.clone(),
            jerk=self.jerk_cmd.clone(),
            joint_names=list(self.joint_names),
        )

    def step(
        self,
        measured_state: JointState,
    ) -> JointState:
        """Generate the next command for one fixed servo tick.

        The method assumes it is called at a fixed rate matching ``servo_dt_s``. It applies a
        position feedback term from measured position, optional target velocity, and
        position/velocity/acceleration/jerk limits before updating the internal command state.

        Args:
            measured_state: Measured state with position and velocity tensors of shape
                ``(batch, dof)`` and joint names matching ``joint_limits``.

        Returns:
            New :class:`JointState` command after applying all joint-space limits.

        Raises:
            TypeError: If ``measured_state`` is not a :class:`JointState` or contains non-tensor
                fields.
            ValueError: If no target has been set, measured velocity is missing, or joint names,
                shape, dtype, device, or finite-value checks fail.
        """

        self._require_target()
        self._require_joint_state(measured_state, "measured_state")
        if self.q_cmd is None:
            shape = tuple(self.q_target.shape)
            measured_q = self._require_position(
                measured_state.position,
                "measured_state.position",
                shape=shape,
            )
            measured_qd = self._require_velocity(
                measured_state.velocity, "measured_state.velocity", shape=shape
            )
            self.q_cmd = torch.clamp(measured_q, min=self.q_min, max=self.q_max)
            self.qd_cmd = measured_qd.clone()
            self.qdd_cmd = torch.zeros_like(self.q_cmd)
            self.jerk_cmd = torch.zeros_like(self.q_cmd)
        else:
            measured_q = self._require_position(
                measured_state.position,
                "measured_state.position",
                shape=tuple(self.q_cmd.shape),
            )
            measured_qd = self._require_velocity(
                measured_state.velocity,
                "measured_state.velocity",
                shape=tuple(self.q_cmd.shape),
            )

        q_error = self.q_target - measured_q
        qd_des = self.qd_target + self.kp * q_error - self.kd * measured_qd
        if not self._target_has_velocity:
            qd_des = self._limit_for_stopping(qd_des, q_error)
        qd_limited, _, _ = self._limit_velocity_acceleration_jerk(qd_des)
        q_next, position_clamped = self._limit_position(
            self.q_cmd,
            self.q_cmd + self.servo_dt_s * qd_limited,
        )

        qd_next = (q_next - self.q_cmd) / self.servo_dt_s
        qdd_next = (qd_next - self.qd_cmd) / self.servo_dt_s
        jerk_next = (qdd_next - self.qdd_cmd) / self.servo_dt_s
        qd_next = torch.where(position_clamped, torch.zeros_like(qd_next), qd_next)
        qdd_next = torch.where(position_clamped, torch.zeros_like(qdd_next), qdd_next)
        jerk_next = torch.where(position_clamped, torch.zeros_like(jerk_next), jerk_next)
        self.q_cmd = q_next
        self.qd_cmd = qd_next
        self.qdd_cmd = qdd_next
        self.jerk_cmd = jerk_next
        return self.command_state()

    def _velocity_or_zero(
        self, velocity: Optional[torch.Tensor], name: str, reference: torch.Tensor
    ) -> torch.Tensor:
        """Return a validated velocity tensor or zeros matching command state.

        Args:
            velocity: Optional velocity tensor with shape ``(batch, dof)``.
            name: Argument name used in error messages.
            reference: Tensor whose shape, dtype, and device are used for zero velocity.

        Returns:
            Velocity tensor with shape ``(batch, dof)``.
        """
        if velocity is None:
            return torch.zeros_like(reference)
        return self._require_position(velocity, name, shape=tuple(reference.shape))

    def _require_velocity(
        self, velocity: Optional[torch.Tensor], name: str, shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Validate a required velocity tensor.

        Args:
            velocity: Velocity tensor with shape ``(batch, dof)``.
            name: Argument name used in error messages.
            shape: Required velocity tensor shape.

        Returns:
            Velocity tensor with shape ``(batch, dof)``.

        Raises:
            ValueError: If ``velocity`` is ``None``.
        """
        if velocity is None:
            log_and_raise(f"{name} must be set")
        return self._require_position(velocity, name, shape=shape)

    def _limit_for_stopping(self, qd_des: torch.Tensor, q_error: torch.Tensor) -> torch.Tensor:
        """Limit velocity for a position-only target so the command can stop at the target.

        Args:
            qd_des: Desired velocity with shape ``(batch, dof)``.
            q_error: Position error with shape ``(batch, dof)``.

        Returns:
            Velocity with shape ``(batch, dof)`` scaled by the acceleration stopping limit.
        """
        qd_stop = torch.sqrt(torch.clamp(2.0 * self.qdd_max * torch.abs(q_error), min=0.0))
        return self._scale_to_limit(qd_des, qd_stop)

    def _limit_velocity_acceleration_jerk(
        self,
        qd_des: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply velocity, acceleration, and jerk limits for one fixed servo tick.

        Args:
            qd_des: Desired velocity with shape ``(batch, dof)``.

        Returns:
            Tuple containing limited velocity, acceleration, and jerk tensors, each with shape
            ``(batch, dof)``.
        """
        qd_des = self._scale_to_limit(qd_des, self.qd_max)
        qdd_des = (qd_des - self.qd_cmd) / self.servo_dt_s
        qdd_des = self._scale_to_limit(qdd_des, self.qdd_max)
        jerk_des = (qdd_des - self.qdd_cmd) / self.servo_dt_s
        jerk_limited = self._scale_to_limit(jerk_des, self.jerk_max)
        qdd_limited = self.qdd_cmd + self.servo_dt_s * jerk_limited
        qd_limited = self.qd_cmd + self.servo_dt_s * qdd_limited
        return qd_limited, qdd_limited, jerk_limited

    def _limit_position(
        self,
        q_prev: torch.Tensor,
        q_candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Limit a candidate command position to joint position limits.

        Args:
            q_prev: Previous command position with shape ``(batch, dof)``.
            q_candidate: Candidate command position with shape ``(batch, dof)``.

        Returns:
            Tuple of limited position with shape ``(batch, dof)`` and a boolean mask with shape
            ``(batch, dof)`` indicating joints whose position limit was crossed.
        """
        dq = q_candidate - q_prev
        ratios = torch.ones_like(q_prev)
        high_mask = q_candidate > self.q_max
        low_mask = q_candidate < self.q_min
        high_ratio = (self.q_max - q_prev) / torch.clamp(dq, min=1.0e-8)
        low_ratio = (self.q_min - q_prev) / torch.clamp(dq, max=-1.0e-8)
        ratios = torch.where(high_mask, high_ratio, ratios)
        ratios = torch.where(low_mask, low_ratio, ratios)
        scale = torch.clamp(torch.min(ratios, dim=-1, keepdim=True).values, 0.0, 1.0)
        return q_prev + scale * dq, high_mask | low_mask

    @staticmethod
    def _scale_to_limit(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
        """Scale each batch row so every joint stays inside its absolute limit.

        Args:
            value: Tensor to scale with shape ``(batch, dof)``.
            limit: Per-joint limits with shape ``(dof,)`` or broadcastable to ``value``.

        Returns:
            Scaled tensor with the same shape as ``value``.
        """
        ratio = limit / torch.clamp(torch.abs(value), min=1.0e-8)
        scale = torch.clamp(torch.min(ratio, dim=-1, keepdim=True).values, 0.0, 1.0)
        return scale * value
