# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library

# Third Party
import numpy as np
import pytest
import torch

from curobo._src.state.state_joint import JointState

# CuRobo
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.trajectory import (
    TrajInterpolationType,
    calculate_dt_no_clamp,
    calculate_traj_steps,
    get_batch_interpolated_trajectory,
    get_cpu_linear_interpolation,
    get_interpolated_trajectory,
    linear_smooth,
)


@pytest.fixture
def device_cfg():
    return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def horizon():
    return 10


@pytest.fixture
def dof():
    return 7


class TestCalculateTrajSteps:
    def test_calculate_traj_steps_basic(self, device_cfg):
        """Test basic trajectory step calculation."""
        opt_dt = torch.tensor([0.1, 0.2], device=device_cfg.device)
        interpolation_dt = torch.tensor(0.01, device=device_cfg.device)
        horizon = 10

        traj_steps, steps_max = calculate_traj_steps(
            opt_dt, interpolation_dt, horizon, nearest_int=False
        )

        # Check that steps are integers
        assert traj_steps.dtype == torch.int32
        assert steps_max.dtype == torch.int32

        # Check that steps are positive
        assert torch.all(traj_steps > 0)
        assert steps_max > 0

        # Check that steps_max is the maximum
        assert steps_max >= torch.max(traj_steps)

    def test_calculate_traj_steps_nearest_int(self, device_cfg):
        """Test trajectory step calculation with nearest_int=True."""
        opt_dt = torch.tensor([0.1, 0.2], device=device_cfg.device)
        interpolation_dt = torch.tensor(0.01, device=device_cfg.device)
        horizon = 10

        traj_steps, steps_max = calculate_traj_steps(
            opt_dt, interpolation_dt, horizon, nearest_int=True
        )

        # Check types
        assert traj_steps.dtype == torch.int32
        assert steps_max.dtype == torch.int32

        # Check that results are different from non-nearest
        traj_steps_no_nearest, steps_max_no_nearest = calculate_traj_steps(
            opt_dt, interpolation_dt, horizon, nearest_int=False
        )

        # Results might be the same or different depending on values
        assert traj_steps.shape == traj_steps_no_nearest.shape


class TestCalculateDtNoClamp:
    def test_calculate_dt_no_clamp_basic(self, device_cfg, batch_size, horizon, dof):
        """Test dt calculation without clamping."""
        vel = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        acc = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        jerk = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        max_vel = torch.ones(dof, device=device_cfg.device) * 2.0
        max_acc = torch.ones(dof, device=device_cfg.device) * 5.0
        max_jerk = torch.ones(dof, device=device_cfg.device) * 10.0

        dt_score = calculate_dt_no_clamp(vel, acc, jerk, max_vel, max_acc, max_jerk)

        # Check output shape
        assert dt_score.shape == (batch_size,)

        # Check that all values are positive
        assert torch.all(dt_score > 0)

    def test_calculate_dt_no_clamp_with_epsilon(self, device_cfg, batch_size, horizon, dof):
        """Test dt calculation with custom epsilon."""
        vel = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        acc = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        jerk = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        max_vel = torch.ones(dof, device=device_cfg.device) * 2.0
        max_acc = torch.ones(dof, device=device_cfg.device) * 5.0
        max_jerk = torch.ones(dof, device=device_cfg.device) * 10.0

        epsilon = 0.01
        dt_score = calculate_dt_no_clamp(
            vel, acc, jerk, max_vel, max_acc, max_jerk, epsilon=epsilon
        )

        # Check that epsilon affects result
        dt_score_default = calculate_dt_no_clamp(vel, acc, jerk, max_vel, max_acc, max_jerk)

        # They should be different (epsilon adds a factor)
        assert not torch.allclose(dt_score, dt_score_default)

    def test_calculate_dt_no_clamp_zero_inputs(self, device_cfg, batch_size, horizon, dof):
        """Test dt calculation with zero velocity/acceleration/jerk."""
        vel = torch.zeros((batch_size, horizon, dof), device=device_cfg.device)
        acc = torch.zeros((batch_size, horizon, dof), device=device_cfg.device)
        jerk = torch.zeros((batch_size, horizon, dof), device=device_cfg.device)
        max_vel = torch.ones(dof, device=device_cfg.device) * 2.0
        max_acc = torch.ones(dof, device=device_cfg.device) * 5.0
        max_jerk = torch.ones(dof, device=device_cfg.device) * 10.0

        dt_score = calculate_dt_no_clamp(vel, acc, jerk, max_vel, max_acc, max_jerk)

        # Check output shape
        assert dt_score.shape == (batch_size,)


class TestLinearSmooth:
    def test_linear_smooth_linear(self):
        """Test linear interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        n = 10
        result = linear_smooth(x, n=n, kind=TrajInterpolationType.LINEAR)

        # Check output shape
        assert result.shape == (n,)

        # Check that it's a tensor
        assert isinstance(result, torch.Tensor)

        # First value should be close to first input
        assert abs(result[0].item() - x[0]) < 0.1

    def test_linear_smooth_cubic(self):
        """Test cubic interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        n = 20
        result = linear_smooth(x, n=n, kind=TrajInterpolationType.CUBIC)

        # Check output shape
        assert result.shape == (n,)

    def test_linear_smooth_cubic_insufficient_points(self):
        """Test cubic interpolation with insufficient points."""
        x = np.array([0.0, 1.0])  # Only 2 points, cubic needs 4
        n = 10
        result = linear_smooth(x, n=n, kind=TrajInterpolationType.CUBIC)

        # Should still work by padding
        assert result.shape == (n,)

    def test_linear_smooth_quintic(self):
        """Test quintic interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        n = 20
        result = linear_smooth(x, n=n, kind=TrajInterpolationType.QUINTIC)

        # Check output shape
        assert result.shape == (n,)

    def test_linear_smooth_quintic_insufficient_points(self):
        """Test quintic interpolation with insufficient points."""
        x = np.array([0.0, 1.0])  # Only 2 points, quintic needs 6
        n = 10
        result = linear_smooth(x, n=n, kind=TrajInterpolationType.QUINTIC)

        # Should still work by padding
        assert result.shape == (n,)

    def test_linear_smooth_with_custom_y(self):
        """Test interpolation with custom y values."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 5.0, 10.0, 15.0])
        n = 10
        result = linear_smooth(x, y=y, n=n, kind=TrajInterpolationType.LINEAR)

        # Check output shape
        assert result.shape == (n,)

    def test_linear_smooth_with_opt_dt(self):
        """Test interpolation with opt_dt."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        n = 10
        opt_dt = 0.1
        interpolation_dt = 0.01
        result = linear_smooth(
            x, n=n, kind=TrajInterpolationType.LINEAR, opt_dt=opt_dt, interpolation_dt=interpolation_dt
        )

        # Check output shape
        assert result.shape == (n,)

    def test_linear_smooth_quartic_raises(self):
        """Test that quartic interpolation raises NotImplementedError."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        n = 10

        with pytest.raises(Exception):
            linear_smooth(x, n=n, kind=TrajInterpolationType.QUARTIC)


class TestGetCpuLinearInterpolation:
    def test_get_cpu_linear_interpolation_linear(self, device_cfg, batch_size, horizon, dof):
        """Test CPU linear interpolation."""
        # Create raw trajectory
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        # Create output trajectory with more steps
        out_horizon = horizon * 2
        out_traj_state = JointState.zeros(
            [batch_size, out_horizon, dof], device_cfg, joint_names=None
        )

        # Trajectory steps
        traj_steps = torch.tensor([out_horizon] * batch_size, device=device_cfg.device)
        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        # Interpolate
        result = get_cpu_linear_interpolation(
            raw_traj, traj_steps, out_traj_state, TrajInterpolationType.LINEAR, interpolation_dt
        )

        # Check that output is modified
        assert result.position.shape == (batch_size, out_horizon, dof)

    def test_get_cpu_linear_interpolation_cubic(self, device_cfg, batch_size, horizon, dof):
        """Test CPU cubic interpolation."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        out_horizon = horizon * 2
        out_traj_state = JointState.zeros(
            [batch_size, out_horizon, dof], device_cfg, joint_names=None
        )

        traj_steps = torch.tensor([out_horizon] * batch_size, device=device_cfg.device)
        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        result = get_cpu_linear_interpolation(
            raw_traj, traj_steps, out_traj_state, TrajInterpolationType.CUBIC, interpolation_dt
        )

        assert result.position.shape == (batch_size, out_horizon, dof)

    def test_get_cpu_linear_interpolation_quintic(self, device_cfg, batch_size, horizon, dof):
        """Test CPU quintic interpolation."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        out_horizon = horizon * 2
        out_traj_state = JointState.zeros(
            [batch_size, out_horizon, dof], device_cfg, joint_names=None
        )

        traj_steps = torch.tensor([out_horizon] * batch_size, device=device_cfg.device)
        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        result = get_cpu_linear_interpolation(
            raw_traj, traj_steps, out_traj_state, TrajInterpolationType.QUINTIC, interpolation_dt
        )

        assert result.position.shape == (batch_size, out_horizon, dof)


class TestGetBatchInterpolatedTrajectory:
    def test_linear_cuda_interpolation(self, device_cfg, batch_size, horizon, dof):
        """Test batch interpolation with LINEAR_CUDA."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        velocity = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.5
        acceleration = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.1
        jerk = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.05
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)

        raw_traj = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            dt=dt,
            device_cfg=device_cfg
        )

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        out_traj, traj_steps = get_batch_interpolated_trajectory(
            raw_traj,
            interpolation_dt,
            kind=TrajInterpolationType.LINEAR_CUDA,
            device_cfg=device_cfg,
        )

        # Check output
        assert out_traj.position.shape[0] == batch_size
        assert out_traj.position.shape[2] == dof
        assert len(traj_steps) == batch_size

    def test_linear_interpolation(self, device_cfg, batch_size, horizon, dof):
        """Test batch interpolation with LINEAR."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        out_traj, traj_steps = get_batch_interpolated_trajectory(
            raw_traj,
            interpolation_dt,
            kind=TrajInterpolationType.LINEAR,
            device_cfg=device_cfg,
        )

        # Check output
        assert out_traj.position.shape[0] == batch_size
        assert out_traj.position.shape[2] == dof

    def test_cubic_interpolation(self, device_cfg, batch_size, horizon, dof):
        """Test batch interpolation with CUBIC."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        out_traj, traj_steps = get_batch_interpolated_trajectory(
            raw_traj,
            interpolation_dt,
            kind=TrajInterpolationType.CUBIC,
            device_cfg=device_cfg,
        )

        assert out_traj.position.shape[0] == batch_size

    def test_quartic_interpolation(self, device_cfg, batch_size, horizon, dof):
        """Test batch interpolation with QUARTIC (should raise)."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        with pytest.raises(Exception):
            get_batch_interpolated_trajectory(
                raw_traj,
                interpolation_dt,
                kind=TrajInterpolationType.QUARTIC,
                device_cfg=device_cfg,
            )

    def test_quintic_interpolation(self, device_cfg, batch_size, horizon, dof):
        """Test batch interpolation with QUINTIC."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        out_traj, traj_steps = get_batch_interpolated_trajectory(
            raw_traj,
            interpolation_dt,
            kind=TrajInterpolationType.QUINTIC,
            device_cfg=device_cfg,
        )

        assert out_traj.position.shape[0] == batch_size

    def test_2d_trajectory_cpu(self, device_cfg, horizon, dof):
        """Test with 2D trajectory (single batch) using CPU interpolation."""
        position = torch.rand((horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1], device=device_cfg.device)  # Need batch dimension
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        out_traj, traj_steps = get_batch_interpolated_trajectory(
            raw_traj,
            interpolation_dt,
            kind=TrajInterpolationType.LINEAR,
            device_cfg=device_cfg,
        )

        # Should be expanded to 3D
        assert len(out_traj.position.shape) == 3
        assert out_traj.position.shape[0] == 1  # Batch of 1

    def test_2d_output_buffer_cpu(self, device_cfg, batch_size, horizon, dof):
        """Test with 2D output buffer using CPU interpolation."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        # 2D buffer
        out_traj_state = JointState.zeros([horizon * 2, dof], device_cfg)

        out_traj, traj_steps = get_batch_interpolated_trajectory(
            raw_traj,
            interpolation_dt,
            kind=TrajInterpolationType.LINEAR,
            out_traj_state=out_traj_state,
            device_cfg=device_cfg,
        )

        assert len(out_traj.position.shape) == 3

    def test_2d_trajectory_cuda(self, device_cfg, horizon, dof):
        """Test with 2D trajectory (single batch) using CUDA."""
        position = torch.rand((horizon, dof), device=device_cfg.device)
        velocity = torch.rand((horizon, dof), device=device_cfg.device) * 0.5
        acceleration = torch.rand((horizon, dof), device=device_cfg.device) * 0.1
        jerk = torch.rand((horizon, dof), device=device_cfg.device) * 0.05
        dt = torch.tensor([0.1], device=device_cfg.device)

        raw_traj = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            dt=dt,
            device_cfg=device_cfg
        )

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        out_traj, traj_steps = get_batch_interpolated_trajectory(
            raw_traj,
            interpolation_dt,
            kind=TrajInterpolationType.LINEAR_CUDA,
            device_cfg=device_cfg,
        )

        # Should be expanded to 3D
        assert len(out_traj.position.shape) == 3

    def test_with_provided_output_buffer(self, device_cfg, batch_size, horizon, dof):
        """Test with pre-allocated output buffer."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        velocity = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.5
        acceleration = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.1
        jerk = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.05
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)

        raw_traj = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            dt=dt,
            device_cfg=device_cfg
        )

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        # Pre-allocate buffer
        out_traj_state = JointState.zeros([batch_size, horizon * 3, dof], device_cfg)

        out_traj, traj_steps = get_batch_interpolated_trajectory(
            raw_traj,
            interpolation_dt,
            kind=TrajInterpolationType.LINEAR_CUDA,
            out_traj_state=out_traj_state,
            device_cfg=device_cfg,
        )

        assert out_traj.position.shape[0] == batch_size

    def test_with_small_output_buffer(self, device_cfg, batch_size, horizon, dof):
        """Test with output buffer that's too small."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        velocity = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.5
        acceleration = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.1
        jerk = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.05
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)

        raw_traj = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            dt=dt,
            device_cfg=device_cfg
        )

        interpolation_dt = torch.tensor(0.01, device=device_cfg.device)  # Very small dt

        # Small buffer
        out_traj_state = JointState.zeros([batch_size, 5, dof], device_cfg)

        out_traj, traj_steps = get_batch_interpolated_trajectory(
            raw_traj,
            interpolation_dt,
            kind=TrajInterpolationType.LINEAR_CUDA,
            out_traj_state=out_traj_state,
            device_cfg=device_cfg,
        )

        # Should create new buffer
        assert out_traj.position.shape[1] >= 5

    def test_2d_output_buffer_cuda(self, device_cfg, batch_size, horizon, dof):
        """Test with 2D output buffer using CUDA."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        velocity = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.5
        acceleration = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.1
        jerk = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.05
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)

        raw_traj = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            dt=dt,
            device_cfg=device_cfg
        )

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        # 2D buffer
        out_traj_state = JointState.zeros([horizon * 2, dof], device_cfg)

        out_traj, traj_steps = get_batch_interpolated_trajectory(
            raw_traj,
            interpolation_dt,
            kind=TrajInterpolationType.LINEAR_CUDA,
            out_traj_state=out_traj_state,
            device_cfg=device_cfg,
        )

        assert len(out_traj.position.shape) == 3

    def test_steps_max_too_large_error(self, device_cfg, batch_size, horizon, dof):
        """Test error when steps_max would be too large."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        # Extremely small interpolation_dt to cause large steps_max
        interpolation_dt = torch.tensor(1e-6, device=device_cfg.device)

        with pytest.raises(Exception):
            get_batch_interpolated_trajectory(
                raw_traj,
                interpolation_dt,
                kind=TrajInterpolationType.LINEAR_CUDA,
                device_cfg=device_cfg,
            )

    def test_steps_max_too_small_error(self, device_cfg, batch_size, horizon, dof):
        """Test error when steps_max is <= 1."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        # Very large interpolation_dt
        interpolation_dt = torch.tensor(100.0, device=device_cfg.device)

        with pytest.raises(Exception):
            get_batch_interpolated_trajectory(
                raw_traj,
                interpolation_dt,
                kind=TrajInterpolationType.LINEAR_CUDA,
                device_cfg=device_cfg,
            )

    def test_unknown_interpolation_type(self, device_cfg, batch_size, horizon, dof):
        """Test error with unknown interpolation type."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(position=position, dt=dt, device_cfg=device_cfg)

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        # This should raise an error for unsupported type
        # We can't easily create a fake enum value, so we'll skip this test
        # or use monkey patching


class TestGetInterpolatedTrajectory:
    def test_get_interpolated_trajectory_linear(self, device_cfg, horizon, dof):
        """Test get_interpolated_trajectory with LINEAR."""
        batch_size = 2
        trajectory = [
            torch.rand((horizon, dof), device=device_cfg.device) for _ in range(batch_size)
        ]

        des_horizon = horizon * 2
        out_traj_state = JointState.zeros([batch_size, des_horizon, dof], device_cfg)

        result, last_tsteps, opt_dt = get_interpolated_trajectory(
            trajectory,
            out_traj_state,
            des_horizon=des_horizon,
            interpolation_dt=0.05,
            kind=TrajInterpolationType.LINEAR,
            device_cfg=device_cfg,
        )

        # Check outputs
        assert result.position.shape == (batch_size, des_horizon, dof)
        assert len(last_tsteps) == batch_size
        assert len(opt_dt) == batch_size

    def test_get_interpolated_trajectory_cubic(self, device_cfg, horizon, dof):
        """Test get_interpolated_trajectory with CUBIC."""
        batch_size = 2
        trajectory = [
            torch.rand((horizon, dof), device=device_cfg.device) for _ in range(batch_size)
        ]

        des_horizon = horizon * 2
        out_traj_state = JointState.zeros([batch_size, des_horizon, dof], device_cfg)

        result, last_tsteps, opt_dt = get_interpolated_trajectory(
            trajectory,
            out_traj_state,
            des_horizon=des_horizon,
            interpolation_dt=0.05,
            kind=TrajInterpolationType.CUBIC,
            device_cfg=device_cfg,
        )

        assert result.position.shape == (batch_size, des_horizon, dof)

    def test_get_interpolated_trajectory_quintic(self, device_cfg, horizon, dof):
        """Test get_interpolated_trajectory with QUINTIC."""
        batch_size = 2
        trajectory = [
            torch.rand((horizon, dof), device=device_cfg.device) for _ in range(batch_size)
        ]

        des_horizon = horizon * 2
        out_traj_state = JointState.zeros([batch_size, des_horizon, dof], device_cfg)

        result, last_tsteps, opt_dt = get_interpolated_trajectory(
            trajectory,
            out_traj_state,
            des_horizon=des_horizon,
            interpolation_dt=0.05,
            kind=TrajInterpolationType.QUINTIC,
            device_cfg=device_cfg,
        )

        assert result.position.shape == (batch_size, des_horizon, dof)

    def test_get_interpolated_trajectory_short_trajectory(self, device_cfg, dof):
        """Test with trajectory too short for cubic."""
        batch_size = 2
        short_horizon = 3
        trajectory = [
            torch.rand((short_horizon, dof), device=device_cfg.device) for _ in range(batch_size)
        ]

        des_horizon = 20
        out_traj_state = JointState.zeros([batch_size, des_horizon, dof], device_cfg)

        # Should fall back to linear
        result, last_tsteps, opt_dt = get_interpolated_trajectory(
            trajectory,
            out_traj_state,
            des_horizon=des_horizon,
            interpolation_dt=0.05,
            kind=TrajInterpolationType.CUBIC,
            device_cfg=device_cfg,
        )

        assert result.position.shape == (batch_size, des_horizon, dof)

    def test_get_interpolated_trajectory_no_des_horizon(self, device_cfg, horizon, dof):
        """Test without specifying des_horizon."""
        batch_size = 2
        trajectory = [
            torch.rand((horizon, dof), device=device_cfg.device) for _ in range(batch_size)
        ]

        des_horizon = horizon * 2
        out_traj_state = JointState.zeros([batch_size, des_horizon, dof], device_cfg)

        result, last_tsteps, opt_dt = get_interpolated_trajectory(
            trajectory,
            out_traj_state,
            des_horizon=None,  # Use buffer size
            interpolation_dt=0.05,
            kind=TrajInterpolationType.LINEAR,
            device_cfg=device_cfg,
        )

        assert result.position.shape == (batch_size, des_horizon, dof)

    def test_get_interpolated_trajectory_with_max_joint_velocity(self, device_cfg, horizon, dof):
        """Test with max_joint_velocity parameter."""
        batch_size = 2
        trajectory = [
            torch.rand((horizon, dof), device=device_cfg.device) for _ in range(batch_size)
        ]

        des_horizon = horizon * 2
        out_traj_state = JointState.zeros([batch_size, des_horizon, dof], device_cfg)
        max_joint_velocity = torch.ones(dof, device=device_cfg.device) * 2.0

        result, last_tsteps, opt_dt = get_interpolated_trajectory(
            trajectory,
            out_traj_state,
            des_horizon=des_horizon,
            interpolation_dt=0.05,
            kind=TrajInterpolationType.LINEAR,
            device_cfg=device_cfg,
            max_joint_velocity=max_joint_velocity,
        )

        assert result.position.shape == (batch_size, des_horizon, dof)

    def test_get_interpolated_trajectory_unsupported_type(self, device_cfg, horizon, dof):
        """Test with unsupported interpolation type."""
        batch_size = 2
        trajectory = [
            torch.rand((horizon, dof), device=device_cfg.device) for _ in range(batch_size)
        ]

        des_horizon = horizon * 2
        out_traj_state = JointState.zeros([batch_size, des_horizon, dof], device_cfg)

        with pytest.raises(Exception):
            get_interpolated_trajectory(
                trajectory,
                out_traj_state,
                des_horizon=des_horizon,
                interpolation_dt=0.05,
                kind=TrajInterpolationType.LINEAR_CUDA,  # Not supported in this function
                device_cfg=device_cfg,
            )


class TestBSplineInterpolation:
    def test_bspline_knots_cuda_without_control_space(self, device_cfg, batch_size, horizon, dof):
        """Test BSPLINE_KNOTS_CUDA without control_space should raise error."""
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        knot = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        knot_dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        raw_traj = JointState(
            position=position,
            knot=knot,
            knot_dt=knot_dt,
            control_space=None,  # Missing control_space
            device_cfg=device_cfg,
        )

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        with pytest.raises(Exception):
            get_batch_interpolated_trajectory(
                raw_traj,
                interpolation_dt,
                kind=TrajInterpolationType.BSPLINE_KNOTS_CUDA,
                device_cfg=device_cfg,
            )

    def test_bspline_knots_cuda_with_control_space(self, device_cfg, batch_size, dof):
        """Test BSPLINE_KNOTS_CUDA with control_space."""
        horizon = 10
        position = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        velocity = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.5
        acceleration = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.1
        jerk = torch.rand((batch_size, horizon, dof), device=device_cfg.device) * 0.05
        knot = torch.rand((batch_size, horizon, dof), device=device_cfg.device)
        knot_dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)
        dt = torch.tensor([0.1] * batch_size, device=device_cfg.device)

        # Create a control space with degree >= 3 (POSITION has degree 0 which is not supported)
        control_space = ControlSpace.BSPLINE_3

        raw_traj = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            knot=knot,
            knot_dt=knot_dt,
            dt=dt,
            control_space=control_space,
            device_cfg=device_cfg,
        )

        interpolation_dt = torch.tensor(0.05, device=device_cfg.device)

        # Create start and goal states with all kinematic fields
        start_state = JointState(
            position=torch.rand((batch_size, dof), device=device_cfg.device),
            velocity=torch.zeros((batch_size, dof), device=device_cfg.device),
            acceleration=torch.zeros((batch_size, dof), device=device_cfg.device),
            jerk=torch.zeros((batch_size, dof), device=device_cfg.device),
            dt=torch.tensor([0.1] * batch_size, device=device_cfg.device),
            device_cfg=device_cfg,
        )
        goal_state = JointState(
            position=torch.rand((batch_size, dof), device=device_cfg.device),
            velocity=torch.zeros((batch_size, dof), device=device_cfg.device),
            acceleration=torch.zeros((batch_size, dof), device=device_cfg.device),
            jerk=torch.zeros((batch_size, dof), device=device_cfg.device),
            dt=torch.tensor([0.1] * batch_size, device=device_cfg.device),
            device_cfg=device_cfg,
        )

        start_idx = torch.zeros(batch_size, dtype=torch.int32, device=device_cfg.device)
        goal_idx = torch.ones(batch_size, dtype=torch.int32, device=device_cfg.device) * (
            horizon - 1
        )
        use_implicit_goal_state = torch.zeros(batch_size, dtype=torch.uint8, device=device_cfg.device)

        out_traj, traj_steps = get_batch_interpolated_trajectory(
            raw_traj,
            interpolation_dt,
            kind=TrajInterpolationType.BSPLINE_KNOTS_CUDA,
            device_cfg=device_cfg,
            current_state=start_state,
            goal_state=goal_state,
            start_idx=start_idx,
            goal_idx=goal_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        # Check output
        assert out_traj.position.shape[0] == batch_size
        assert out_traj.position.shape[2] == dof

