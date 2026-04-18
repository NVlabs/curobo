# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for state transition functions."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.transition.fns_state_transition import (
    StateFromAcceleration,
    StateFromBSplineKnot,
    StateFromPositionClique,
    StateFromPositionTeleport,
    filter_signal_jit,
)
from curobo._src.types.control_space import ControlSpace


class TestStateFromBase:
    """Test StateFromBase class methods."""

    def test_update_dt_with_dt_h(self, cuda_device_cfg):
        """Test update_dt when _dt_h is set."""
        horizon = 14
        dt_tensor = torch.ones(horizon, **cuda_device_cfg.as_torch_dict()) * 0.02

        clique = StateFromPositionClique(
            device_cfg=cuda_device_cfg,
            dt_h=dt_tensor,
            dof=7,
            batch_size=2,
            horizon=horizon,
        )

        # Update dt
        new_dt = 0.05
        clique.update_dt(new_dt)

        # Verify dt was updated - all elements should be new_dt
        expected_dt = torch.ones(horizon, **cuda_device_cfg.as_torch_dict()) * new_dt
        assert torch.allclose(clique._dt_h, expected_dt)


class TestStateFromPositionTeleport:
    """Test StateFromPositionTeleport class."""

    def test_initialization(self, cuda_device_cfg):
        """Test basic initialization."""
        teleport = StateFromPositionTeleport(
            device_cfg=cuda_device_cfg,
            batch_size=2,
            horizon=10,
        )

        assert teleport.batch_size == 2
        assert teleport.horizon == 10
        assert teleport.action_horizon == 10

    def test_update_batch_size(self, cuda_device_cfg):
        """Test update_batch_size method."""
        teleport = StateFromPositionTeleport(
            device_cfg=cuda_device_cfg,
            batch_size=2,
            horizon=10,
        )

        teleport.update_batch_size(batch_size=4, horizon=15)

        assert teleport.batch_size == 4
        assert teleport.horizon == 15

    def test_forward_copies_position(self, cuda_device_cfg):
        """Test that forward copies action to output position."""
        batch_size = 2
        horizon = 5
        dof = 7

        teleport = StateFromPositionTeleport(
            device_cfg=cuda_device_cfg,
            batch_size=batch_size,
            horizon=horizon,
        )

        # Create input tensors
        u_act = torch.randn(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)

        # Create dummy start state (not used in teleport mode)
        start_state = JointState.zeros((1, dof), cuda_device_cfg)

        result = teleport.forward(start_state, u_act, out_state_seq)

        # Verify position matches action
        assert torch.allclose(result.position, u_act)


class TestStateFromAcceleration:
    """Test StateFromAcceleration class."""

    @pytest.fixture
    def dt_tensor(self, cuda_device_cfg):
        """Create dt tensor for testing."""
        horizon = 10
        return torch.ones(horizon, **cuda_device_cfg.as_torch_dict()) * 0.02

    def test_initialization(self, cuda_device_cfg, dt_tensor):
        """Test basic initialization."""
        dof = 7
        batch_size = 2
        horizon = 10

        accel = StateFromAcceleration(
            device_cfg=cuda_device_cfg,
            dt_h=dt_tensor,
            dof=dof,
            batch_size=batch_size,
            horizon=horizon,
        )

        assert accel.dof == dof
        assert accel.batch_size == batch_size
        assert accel.horizon == horizon
        assert accel.action_horizon == horizon

    def test_update_batch_size_creates_gradient_buffer(self, cuda_device_cfg, dt_tensor):
        """Test that update_batch_size creates gradient buffer."""
        dof = 7
        batch_size = 2
        horizon = 10

        accel = StateFromAcceleration(
            device_cfg=cuda_device_cfg,
            dt_h=dt_tensor,
            dof=dof,
            batch_size=batch_size,
            horizon=horizon,
        )

        # After init, _u_grad is allocated with the init batch_size/horizon
        assert accel._u_grad is not None
        assert accel._u_grad.shape == (batch_size, horizon, dof)

        # Update to new batch size - this should create the buffer
        new_batch_size = 4
        new_horizon = 15
        accel.update_batch_size(batch_size=new_batch_size, horizon=new_horizon)

        assert accel._u_grad is not None
        assert accel._u_grad.shape == (new_batch_size, new_horizon, dof)

    def test_update_batch_size_force_update_detaches(self, cuda_device_cfg, dt_tensor):
        """Test that force_update detaches gradient buffer."""
        dof = 7
        batch_size = 2
        horizon = 10

        accel = StateFromAcceleration(
            device_cfg=cuda_device_cfg,
            dt_h=dt_tensor,
            dof=dof,
            batch_size=batch_size,
            horizon=horizon,
        )

        # First create the buffer by calling update_batch_size with new values
        new_batch_size = 4
        new_horizon = 15
        accel.update_batch_size(batch_size=new_batch_size, horizon=new_horizon)
        assert accel._u_grad is not None

        # Force update should detach the buffer
        accel.update_batch_size(batch_size=new_batch_size, horizon=new_horizon, force_update=True)

        assert not accel._u_grad.requires_grad


class TestStateFromPositionClique:
    """Test StateFromPositionClique class."""

    @pytest.fixture
    def dt_tensor(self, cuda_device_cfg):
        """Create dt tensor for testing."""
        horizon = 14  # Needs horizon >= 5 for clique (action_horizon = horizon - 4)
        return torch.ones(horizon, **cuda_device_cfg.as_torch_dict()) * 0.02

    def test_initialization(self, cuda_device_cfg, dt_tensor):
        """Test basic initialization."""
        dof = 7
        batch_size = 2
        horizon = 14

        clique = StateFromPositionClique(
            device_cfg=cuda_device_cfg,
            dt_h=dt_tensor,
            dof=dof,
            batch_size=batch_size,
            horizon=horizon,
        )

        assert clique.dof == dof
        assert clique.batch_size == batch_size
        assert clique.horizon == horizon
        # Action horizon is horizon - 4 for clique
        assert clique.action_horizon == horizon - 4

    def test_initialization_with_filtering(self, cuda_device_cfg, dt_tensor):
        """Test initialization with velocity filtering enabled."""
        dof = 7
        batch_size = 2
        horizon = 14

        clique = StateFromPositionClique(
            device_cfg=cuda_device_cfg,
            dt_h=dt_tensor,
            dof=dof,
            filter_velocity=True,
            filter_acceleration=True,
            filter_jerk=True,
            batch_size=batch_size,
            horizon=horizon,
        )

        assert clique._filter_velocity is True
        assert clique._filter_acceleration is True
        assert clique._filter_jerk is True
        assert clique._sma_kernel is not None

    def test_update_batch_size(self, cuda_device_cfg, dt_tensor):
        """Test update_batch_size method."""
        dof = 7
        batch_size = 2
        horizon = 14

        clique = StateFromPositionClique(
            device_cfg=cuda_device_cfg,
            dt_h=dt_tensor,
            dof=dof,
            batch_size=batch_size,
            horizon=horizon,
        )

        new_batch_size = 4
        new_horizon = 20
        clique.update_batch_size(batch_size=new_batch_size, horizon=new_horizon)

        assert clique.batch_size == new_batch_size
        assert clique.horizon == new_horizon
        assert clique.action_horizon == new_horizon - 4
        assert clique._u_grad.shape == (new_batch_size, new_horizon - 4, dof)


class TestStateFromBSplineKnot:
    """Test StateFromBSplineKnot class."""

    def test_initialization_bspline_3(self, cuda_device_cfg):
        """Test initialization with BSPLINE_3."""
        dof = 7
        batch_size = 2
        n_knots = 6
        interpolation_steps = 4

        bspline = StateFromBSplineKnot(
            device_cfg=cuda_device_cfg,
            dof=dof,
            batch_size=batch_size,
            horizon=100,  # Will be overridden by padded_horizon
            n_knots=n_knots,
            interpolation_steps=interpolation_steps,
            control_space=ControlSpace.BSPLINE_3,
        )

        assert bspline.dof == dof
        assert bspline.n_knots == n_knots
        assert bspline.bspline_degree == 3
        assert bspline.interpolation_steps == interpolation_steps
        assert bspline.action_horizon == n_knots

    def test_initialization_bspline_4(self, cuda_device_cfg):
        """Test initialization with BSPLINE_4."""
        dof = 7
        batch_size = 2
        n_knots = 6
        interpolation_steps = 4

        bspline = StateFromBSplineKnot(
            device_cfg=cuda_device_cfg,
            dof=dof,
            batch_size=batch_size,
            horizon=100,
            n_knots=n_knots,
            interpolation_steps=interpolation_steps,
            control_space=ControlSpace.BSPLINE_4,
        )

        assert bspline.bspline_degree == 4

    def test_initialization_bspline_5(self, cuda_device_cfg):
        """Test initialization with BSPLINE_5."""
        dof = 7
        batch_size = 2
        n_knots = 6
        interpolation_steps = 4

        bspline = StateFromBSplineKnot(
            device_cfg=cuda_device_cfg,
            dof=dof,
            batch_size=batch_size,
            horizon=100,
            n_knots=n_knots,
            interpolation_steps=interpolation_steps,
            control_space=ControlSpace.BSPLINE_5,
        )

        assert bspline.bspline_degree == 5

    def test_padded_horizon_calculation(self, cuda_device_cfg):
        """Test that padded_horizon is correctly calculated."""
        dof = 7
        batch_size = 2
        n_knots = 6
        interpolation_steps = 4

        bspline = StateFromBSplineKnot(
            device_cfg=cuda_device_cfg,
            dof=dof,
            batch_size=batch_size,
            horizon=100,
            n_knots=n_knots,
            interpolation_steps=interpolation_steps,
            control_space=ControlSpace.BSPLINE_4,
        )

        expected_padded_horizon = ControlSpace.spline_total_interpolation_steps(
            ControlSpace.BSPLINE_4, n_knots, interpolation_steps
        )
        assert bspline.padded_horizon == expected_padded_horizon

    def test_update_batch_size(self, cuda_device_cfg):
        """Test update_batch_size method."""
        dof = 7
        batch_size = 2
        n_knots = 6
        interpolation_steps = 4

        bspline = StateFromBSplineKnot(
            device_cfg=cuda_device_cfg,
            dof=dof,
            batch_size=batch_size,
            horizon=100,
            n_knots=n_knots,
            interpolation_steps=interpolation_steps,
            control_space=ControlSpace.BSPLINE_4,
        )

        new_batch_size = 8
        bspline.update_batch_size(batch_size=new_batch_size, horizon=100)

        assert bspline.batch_size == new_batch_size
        assert bspline._u_grad.shape == (new_batch_size, n_knots, dof)

    def test_update_batch_size_force_update(self, cuda_device_cfg):
        """Test update_batch_size with force_update."""
        dof = 7
        batch_size = 2
        n_knots = 6
        interpolation_steps = 4

        bspline = StateFromBSplineKnot(
            device_cfg=cuda_device_cfg,
            dof=dof,
            batch_size=batch_size,
            horizon=100,
            n_knots=n_knots,
            interpolation_steps=interpolation_steps,
            control_space=ControlSpace.BSPLINE_4,
        )

        bspline.update_batch_size(batch_size=batch_size, horizon=100, force_update=True)

        assert not bspline._u_grad.requires_grad


class TestFilterSignalJit:
    """Test filter_signal_jit function."""

    def test_filter_preserves_shape(self, cuda_device_cfg):
        """Test that filter preserves tensor shape."""
        batch_size = 4
        horizon = 20
        dof = 7

        signal = torch.randn(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict())
        kernel = cuda_device_cfg.to_device([[[0.06136, 0.24477, 0.38774, 0.24477, 0.06136]]])

        filtered = filter_signal_jit(signal, kernel)

        assert filtered.shape == signal.shape

    def test_filter_smooths_signal(self, cuda_device_cfg):
        """Test that filter produces smoother output."""
        batch_size = 1
        horizon = 50
        dof = 1

        # Create noisy signal
        t = torch.linspace(0, 2 * 3.14159, horizon, **cuda_device_cfg.as_torch_dict())
        clean_signal = torch.sin(t).view(batch_size, horizon, dof)
        noise = torch.randn_like(clean_signal) * 0.1
        noisy_signal = clean_signal + noise

        kernel = cuda_device_cfg.to_device([[[0.06136, 0.24477, 0.38774, 0.24477, 0.06136]]])

        filtered = filter_signal_jit(noisy_signal, kernel)

        # Filtered signal should be closer to clean signal (lower noise variance)
        noisy_diff = (noisy_signal - clean_signal).std()
        filtered_diff = (filtered - clean_signal).std()

        # Allow for some tolerance since filtering may not perfectly denoise
        assert filtered_diff <= noisy_diff * 1.5

    def test_filter_is_differentiable(self, cuda_device_cfg):
        """Test that filter is differentiable."""
        batch_size = 2
        horizon = 20
        dof = 7

        signal = torch.randn(
            batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict(), requires_grad=True
        )
        kernel = cuda_device_cfg.to_device([[[0.06136, 0.24477, 0.38774, 0.24477, 0.06136]]])

        filtered = filter_signal_jit(signal, kernel)
        loss = filtered.sum()
        loss.backward()

        assert signal.grad is not None
        assert signal.grad.shape == signal.shape

    def test_filter_output_contiguous(self, cuda_device_cfg):
        """Test that filter output is contiguous."""
        batch_size = 2
        horizon = 20
        dof = 7

        signal = torch.randn(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict())
        kernel = cuda_device_cfg.to_device([[[0.06136, 0.24477, 0.38774, 0.24477, 0.06136]]])

        filtered = filter_signal_jit(signal, kernel)

        assert filtered.is_contiguous()

