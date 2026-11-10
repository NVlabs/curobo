# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the joint target servo helper."""

from typing import Optional, Sequence

import pytest
import torch

from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.state.state_joint import JointState
from curobo.servo import JointTargetServo


def _make_joint_limits(
    qd_max: tuple[float, float] = (1.0, 2.0),
    qdd_max: tuple[float, float] = (2.0, 2.0),
    jerk_max: tuple[float, float] = (1000.0, 1000.0),
) -> JointLimits:
    qd = torch.tensor(qd_max)
    qdd = torch.tensor(qdd_max)
    jerk = torch.tensor(jerk_max)
    return JointLimits(
        joint_names=["j0", "j1"],
        position=torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        velocity=torch.stack((-qd, qd)),
        acceleration=torch.stack((-qdd, qdd)),
        jerk=torch.stack((-jerk, jerk)),
    )


def _make_asymmetric_joint_limits() -> JointLimits:
    return JointLimits(
        joint_names=["j0", "j1"],
        position=torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        velocity=torch.tensor([[-1.0, -3.0], [2.0, 2.0]]),
        acceleration=torch.tensor([[-2.0, -4.0], [3.0, 3.0]]),
        jerk=torch.tensor([[-5.0, -7.0], [6.0, 6.0]]),
    )


def _make_state(
    position: Sequence[Sequence[float]],
    velocity: Optional[Sequence[Sequence[float]]] = None,
    joint_names: Optional[list[str]] = None,
) -> JointState:
    return JointState(
        position=torch.tensor(position),
        velocity=None if velocity is None else torch.tensor(velocity),
        joint_names=["j0", "j1"] if joint_names is None else joint_names,
    )


def _make_servo(
    command_position: Sequence[Sequence[float]],
    qd_max: tuple[float, float] = (1.0, 2.0),
    qdd_max: tuple[float, float] = (2.0, 2.0),
    jerk_max: tuple[float, float] = (1000.0, 1000.0),
    kp: float = 100.0,
    kd: float = 1.0,
) -> JointTargetServo:
    servo = JointTargetServo(
        _make_joint_limits(qd_max, qdd_max, jerk_max),
        servo_dt_s=0.1,
        kp=kp,
        kd=kd,
    )
    servo.reset(_make_state(command_position))
    return servo


def test_position_boundary_uses_uniform_scale() -> None:
    servo = _make_servo(
        [[0.95, 0.0]],
        qd_max=(10.0, 10.0),
        qdd_max=(10.0, 10.0),
        jerk_max=(1000.0, 1000.0),
        kp=1.0,
    )
    servo.servo_dt_s = 1.0

    servo.set_target(_make_state([[1.15, 0.4]]))
    state = servo.step(servo.command_state())

    assert torch.allclose(state.position[0, 0], torch.tensor(1.0))
    assert state.position[0, 1] > 0.0


def test_position_boundary_zeros_clamped_joint_derivatives() -> None:
    servo = _make_servo(
        [[0.95, 0.0]],
        qd_max=(10.0, 10.0),
        qdd_max=(10.0, 10.0),
        jerk_max=(1000.0, 1000.0),
        kp=1.0,
        kd=0.0,
    )
    servo.servo_dt_s = 1.0
    servo.set_target(_make_state([[1.15, 0.4]], velocity=[[1.0, 0.0]]))

    state = servo.step(servo.command_state())

    assert torch.allclose(state.position[0, 0], torch.tensor(1.0))
    assert state.position[0, 1] > 0.0
    assert torch.allclose(state.velocity[0, 0], torch.tensor(0.0))
    assert torch.allclose(state.acceleration[0, 0], torch.tensor(0.0))
    assert torch.allclose(state.jerk[0, 0], torch.tensor(0.0))
    assert state.velocity[0, 1] > 0.0


def test_position_target_tracks_previous_command_velocity() -> None:
    servo = _make_servo([[0.0, 0.0]])
    servo.set_target(_make_state([[1.0, 1.0]]))

    first_state = servo.step(servo.command_state())
    second_state = servo.step(first_state)

    assert torch.allclose(first_state.position, torch.tensor([[0.02, 0.02]]))
    assert torch.allclose(first_state.velocity, torch.tensor([[0.2, 0.2]]))
    assert torch.allclose(second_state.velocity, torch.tensor([[0.4, 0.4]]))


def test_position_target_reaches_goal() -> None:
    servo = _make_servo([[0.0, 0.0]])
    target = torch.tensor([[1.0, 1.0]])
    servo.set_target(_make_state([[1.0, 1.0]]))

    state = servo.command_state()
    for _ in range(80):
        state = servo.step(state)

    assert torch.allclose(state.position, target)
    assert torch.allclose(state.velocity, torch.zeros_like(target))


def test_velocity_target_moves_without_stopping_limit() -> None:
    servo = _make_servo(
        [[0.0, 0.0]],
        qd_max=(10.0, 10.0),
        qdd_max=(10.0, 10.0),
        jerk_max=(1000.0, 1000.0),
        kp=0.0,
        kd=0.0,
    )
    servo.set_target(_make_state([[0.0, 0.0]], velocity=[[1.0, 0.0]]))

    state = servo.step(servo.command_state())

    assert torch.allclose(state.position, torch.tensor([[0.1, 0.0]]))
    assert torch.allclose(state.velocity, torch.tensor([[1.0, 0.0]]))


def test_jerk_limit_caps_acceleration_change() -> None:
    servo = _make_servo(
        [[0.0, 0.0]],
        qd_max=(10.0, 10.0),
        qdd_max=(10.0, 10.0),
        jerk_max=(1.0, 1.0),
        kp=0.0,
    )
    servo.set_target(_make_state([[0.0, 0.0]], velocity=[[10.0, 0.0]]))

    state = servo.step(servo.command_state())

    assert torch.allclose(state.jerk, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(state.acceleration, torch.tensor([[0.1, 0.0]]))
    assert torch.allclose(state.velocity, torch.tensor([[0.01, 0.0]]))


def test_first_step_initializes_empty_servo_from_measured_state() -> None:
    servo = JointTargetServo(_make_asymmetric_joint_limits(), servo_dt_s=0.1)

    with pytest.raises(ValueError, match="has no command state"):
        servo.command_state()
    with pytest.raises(ValueError, match="has no target"):
        servo.step(_make_state([[0.0, 0.0]], velocity=[[0.0, 0.0]]))

    target = torch.tensor([[0.25, -0.25]])
    target_velocity = torch.tensor([[0.5, -0.5]])
    servo.set_target(_make_state([[0.25, -0.25]], velocity=[[0.5, -0.5]]))

    with pytest.raises(ValueError, match="has no command state"):
        servo.command_state()

    state = servo.step(_make_state([[0.0, 0.0]], velocity=[[0.0, 0.0]]))

    assert state.position.shape == target.shape
    assert state.velocity.shape == target.shape
    assert torch.allclose(servo.q_target, target)
    assert torch.allclose(servo.qd_target, target_velocity)


def test_first_step_clamps_initial_measured_command_position() -> None:
    servo = JointTargetServo(_make_asymmetric_joint_limits(), servo_dt_s=0.1)

    servo.set_target(_make_state([[1.2, -1.3]]))
    state = servo.step(_make_state([[1.2, -1.3]], velocity=[[0.0, 0.0]]))

    assert torch.allclose(state.position, torch.tensor([[1.0, -1.0]]))
    assert torch.allclose(servo.q_target, torch.tensor([[1.2, -1.3]]))


def test_reset_sets_command_position_velocity_and_target() -> None:
    servo = _make_servo([[0.0, 0.0]])
    servo.set_target(_make_state([[1.0, 1.0]]))
    servo.step(servo.command_state())

    servo.reset(_make_state([[0.3, -0.4]], velocity=[[0.1, 0.2]]))
    state = servo.command_state()

    assert torch.allclose(state.position, torch.tensor([[0.3, -0.4]]))
    assert torch.allclose(state.velocity, torch.tensor([[0.1, 0.2]]))
    assert torch.allclose(state.acceleration, torch.zeros(1, 2))
    assert state.joint_names == ["j0", "j1"]
    assert torch.allclose(servo.q_target, torch.tensor([[0.3, -0.4]]))
    assert torch.allclose(servo.qd_target, torch.tensor([[0.1, 0.2]]))


def test_rejects_wrong_target_dimension() -> None:
    servo = _make_servo([[0.0, 0.0]])

    with pytest.raises(ValueError, match="position must have shape"):
        servo.set_target(
            JointState(
                position=torch.tensor([[0.0, 0.0, 0.0]]),
                joint_names=["j0", "j1"],
            )
        )


def test_set_target_rejects_missing_joint_names() -> None:
    servo = _make_servo([[0.0, 0.0]])

    with pytest.raises(ValueError, match=r"target_state\.joint_names must be set"):
        servo.set_target(JointState(position=torch.tensor([[0.0, 0.0]])))


def test_set_target_rejects_joint_name_order_mismatch() -> None:
    servo = _make_servo([[0.0, 0.0]])

    with pytest.raises(ValueError, match=r"target_state\.joint_names must match"):
        servo.set_target(_make_state([[0.0, 0.0]], joint_names=["j1", "j0"]))


def test_reset_rejects_non_joint_state() -> None:
    servo = JointTargetServo(_make_asymmetric_joint_limits(), servo_dt_s=0.1)

    with pytest.raises(TypeError, match="command_state must be a JointState"):
        servo.reset(torch.tensor([[0.0, 0.0]]))


def test_rejects_unbatched_target_shape() -> None:
    servo = _make_servo([[0.0, 0.0]])

    with pytest.raises(ValueError, match="position must have shape"):
        servo.set_target(
            JointState(position=torch.tensor([0.0, 0.0]), joint_names=["j0", "j1"])
        )


def test_reset_rejects_non_tensor_position() -> None:
    servo = JointTargetServo(_make_asymmetric_joint_limits(), servo_dt_s=0.1)

    with pytest.raises(TypeError, match="command_state.position must be a torch.Tensor"):
        servo.reset(JointState(position=[0.0, 0.0], joint_names=["j0", "j1"]))


def test_rejects_dtype_mismatch() -> None:
    joint_limits = JointLimits(
        joint_names=["j0", "j1"],
        position=torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float64),
        velocity=torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        acceleration=torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        jerk=torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
    )
    with pytest.raises(TypeError, match=r"joint_limits\.velocity must have dtype"):
        JointTargetServo(joint_limits, servo_dt_s=0.1)


def test_constructor_uses_active_joint_limits() -> None:
    servo = JointTargetServo(_make_asymmetric_joint_limits())

    assert servo.joint_names == ("j0", "j1")
    assert servo.dof == 2
    assert torch.allclose(servo.qd_max, torch.tensor([1.0, 2.0]))
    assert torch.allclose(servo.qdd_max, torch.tensor([2.0, 3.0]))
    assert torch.allclose(servo.jerk_max, torch.tensor([5.0, 6.0]))
    servo.set_target(_make_state([[0.0, 0.0]]))
    servo.step(_make_state([[0.0, 0.0]], velocity=[[0.0, 0.0]]))
    assert servo.batch_size == 1


def test_servo_steps_each_batch_row_independently() -> None:
    servo = _make_servo([[0.0, 0.0], [0.95, 0.0]])
    servo.set_target(_make_state([[1.0, 1.0], [1.15, 0.4]]))

    state = servo.step(servo.command_state())

    assert state.position.shape == (2, 2)
    assert torch.allclose(state.position[0], torch.tensor([0.02, 0.02]))
    assert torch.allclose(state.velocity[0], torch.tensor([0.2, 0.2]))
    assert state.position[1, 0] > 0.95
    assert state.position[1, 0] <= 1.0


def test_step_uses_measured_joint_state() -> None:
    servo = _make_servo(
        [[0.0, 0.0]],
        qd_max=(10.0, 10.0),
        qdd_max=(10.0, 10.0),
        jerk_max=(1000.0, 1000.0),
        kp=1.0,
        kd=1.0,
    )
    servo.set_target(_make_state([[1.0, 1.0]]))

    state = servo.step(_make_state([[0.0, 0.0]], velocity=[[0.5, 0.5]]))

    assert torch.allclose(state.velocity, torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(state.position, torch.tensor([[0.05, 0.05]]))


def test_step_rejects_measured_state_without_velocity() -> None:
    servo = _make_servo([[0.0, 0.0]])
    servo.set_target(_make_state([[1.0, 0.0]]))

    with pytest.raises(ValueError, match=r"measured_state\.velocity must be set"):
        servo.step(_make_state([[0.0, 0.0]]))


def test_step_rejects_measured_state_joint_name_order_mismatch() -> None:
    servo = _make_servo([[0.0, 0.0]])
    servo.set_target(_make_state([[1.0, 0.0]]))

    with pytest.raises(ValueError, match=r"measured_state\.joint_names must match"):
        servo.step(_make_state([[0.0, 0.0]], velocity=[[0.0, 0.0]], joint_names=["j1", "j0"]))


def test_servo_rejects_unbatched_reset_position() -> None:
    with pytest.raises(ValueError, match="position must have shape"):
        _make_servo([0.0, 0.0])


def test_constructor_accepts_kinematics_joint_limits_with_multiple_batches() -> None:
    class _FakeKinematics:
        def get_joint_limits(self) -> JointLimits:
            return _make_asymmetric_joint_limits()

    servo = JointTargetServo(_FakeKinematics().get_joint_limits())
    servo.set_target(JointState(position=torch.zeros(3, 2), joint_names=["j0", "j1"]))
    servo.step(
        JointState(
            position=torch.zeros(3, 2),
            velocity=torch.zeros(3, 2),
            joint_names=["j0", "j1"],
        )
    )

    assert servo.batch_size == 3
    assert servo.dof == 2
    assert torch.allclose(servo.qd_max, torch.tensor([1.0, 2.0]))
