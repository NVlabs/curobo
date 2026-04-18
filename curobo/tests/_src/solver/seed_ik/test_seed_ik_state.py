# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for SeedIKState dataclass."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.seed_ik.seed_ik_state import SeedIKState


@pytest.fixture
def sample_state():
    """Create a sample SeedIKState for testing."""
    batch_size = 4
    dof = 7
    jacobian_rows = 6  # 6 DOF pose (3 position + 3 orientation)

    return SeedIKState(
        success=torch.tensor([True, False, True, False]),
        improvement=torch.tensor([True, True, False, False]),
        joint_position=torch.randn(batch_size, dof),
        error_norm=torch.rand(batch_size),
        jTerror=torch.randn(batch_size, dof),
        jacobian=torch.randn(batch_size, jacobian_rows, dof),
        lambda_damping=torch.ones(batch_size, 1, 1) * 0.1,
        position_errors=torch.rand(batch_size),
        orientation_errors=torch.rand(batch_size),
    )


class TestSeedIKStateDataclass:
    """Test SeedIKState dataclass attributes."""

    def test_default_success_is_none(self):
        """Test default success is None."""
        state = SeedIKState()
        assert state.success is None

    def test_default_improvement_is_none(self):
        """Test default improvement is None."""
        state = SeedIKState()
        assert state.improvement is None

    def test_default_joint_position_is_none(self):
        """Test default joint_position is None."""
        state = SeedIKState()
        assert state.joint_position is None

    def test_default_error_norm_is_none(self):
        """Test default error_norm is None."""
        state = SeedIKState()
        assert state.error_norm is None

    def test_default_jTerror_is_none(self):
        """Test default jTerror is None."""
        state = SeedIKState()
        assert state.jTerror is None

    def test_default_jacobian_is_none(self):
        """Test default jacobian is None."""
        state = SeedIKState()
        assert state.jacobian is None

    def test_default_lambda_damping_is_none(self):
        """Test default lambda_damping is None."""
        state = SeedIKState()
        assert state.lambda_damping is None

    def test_default_position_errors_is_none(self):
        """Test default position_errors is None."""
        state = SeedIKState()
        assert state.position_errors is None

    def test_default_orientation_errors_is_none(self):
        """Test default orientation_errors is None."""
        state = SeedIKState()
        assert state.orientation_errors is None


class TestSeedIKStateClone:
    """Test SeedIKState.clone method."""

    def test_clone_creates_new_object(self, sample_state):
        """Test clone creates a new object."""
        cloned = sample_state.clone()
        assert cloned is not sample_state

    def test_clone_preserves_success(self, sample_state):
        """Test clone preserves success tensor."""
        cloned = sample_state.clone()
        assert torch.equal(cloned.success, sample_state.success)

    def test_clone_preserves_improvement(self, sample_state):
        """Test clone preserves improvement tensor."""
        cloned = sample_state.clone()
        assert torch.equal(cloned.improvement, sample_state.improvement)

    def test_clone_preserves_joint_position(self, sample_state):
        """Test clone preserves joint_position tensor."""
        cloned = sample_state.clone()
        assert torch.equal(cloned.joint_position, sample_state.joint_position)

    def test_clone_preserves_error_norm(self, sample_state):
        """Test clone preserves error_norm tensor."""
        cloned = sample_state.clone()
        assert torch.equal(cloned.error_norm, sample_state.error_norm)

    def test_clone_preserves_jTerror(self, sample_state):
        """Test clone preserves jTerror tensor."""
        cloned = sample_state.clone()
        assert torch.equal(cloned.jTerror, sample_state.jTerror)

    def test_clone_preserves_jacobian(self, sample_state):
        """Test clone preserves jacobian tensor."""
        cloned = sample_state.clone()
        assert torch.equal(cloned.jacobian, sample_state.jacobian)

    def test_clone_preserves_lambda_damping(self, sample_state):
        """Test clone preserves lambda_damping tensor."""
        cloned = sample_state.clone()
        assert torch.equal(cloned.lambda_damping, sample_state.lambda_damping)

    def test_clone_preserves_position_errors(self, sample_state):
        """Test clone preserves position_errors tensor."""
        cloned = sample_state.clone()
        assert torch.equal(cloned.position_errors, sample_state.position_errors)

    def test_clone_preserves_orientation_errors(self, sample_state):
        """Test clone preserves orientation_errors tensor."""
        cloned = sample_state.clone()
        assert torch.equal(cloned.orientation_errors, sample_state.orientation_errors)

    def test_clone_is_independent(self, sample_state):
        """Test cloned tensors are independent."""
        cloned = sample_state.clone()

        # Modify original
        sample_state.success[0] = False
        sample_state.joint_position[0, 0] = 999.0

        # Clone should be unaffected
        assert cloned.success[0] == True
        assert cloned.joint_position[0, 0] != 999.0

    def test_clone_with_none_fields(self):
        """Test clone handles None fields correctly."""
        state = SeedIKState(
            success=torch.tensor([True, False]),
            joint_position=torch.randn(2, 7),
        )
        cloned = state.clone()

        assert torch.equal(cloned.success, state.success)
        assert torch.equal(cloned.joint_position, state.joint_position)
        assert cloned.improvement is None
        assert cloned.jacobian is None


class TestSeedIKStateCopy:
    """Test SeedIKState.copy_ method."""

    def test_copy_updates_success(self, sample_state):
        """Test copy_ updates success tensor."""
        other = SeedIKState(
            success=torch.tensor([False, True, False, True]),
            improvement=torch.tensor([False, False, True, True]),
            joint_position=torch.randn(4, 7),
            error_norm=torch.rand(4),
            jTerror=torch.randn(4, 7),
            jacobian=torch.randn(4, 6, 7),
            lambda_damping=torch.ones(4, 1, 1) * 0.2,
            position_errors=torch.rand(4),
            orientation_errors=torch.rand(4),
        )

        sample_state.copy_(other)

        assert torch.equal(sample_state.success, other.success)

    def test_copy_updates_joint_position(self, sample_state):
        """Test copy_ updates joint_position tensor."""
        other = SeedIKState(
            success=torch.tensor([False, True, False, True]),
            improvement=torch.tensor([False, False, True, True]),
            joint_position=torch.randn(4, 7),
            error_norm=torch.rand(4),
            jTerror=torch.randn(4, 7),
            jacobian=torch.randn(4, 6, 7),
            lambda_damping=torch.ones(4, 1, 1) * 0.2,
            position_errors=torch.rand(4),
            orientation_errors=torch.rand(4),
        )

        sample_state.copy_(other)

        assert torch.equal(sample_state.joint_position, other.joint_position)

    def test_copy_updates_error_norm(self, sample_state):
        """Test copy_ updates error_norm tensor."""
        other = SeedIKState(
            success=torch.tensor([False, True, False, True]),
            improvement=torch.tensor([False, False, True, True]),
            joint_position=torch.randn(4, 7),
            error_norm=torch.rand(4),
            jTerror=torch.randn(4, 7),
            jacobian=torch.randn(4, 6, 7),
            lambda_damping=torch.ones(4, 1, 1) * 0.2,
            position_errors=torch.rand(4),
            orientation_errors=torch.rand(4),
        )

        sample_state.copy_(other)

        assert torch.equal(sample_state.error_norm, other.error_norm)


class TestSeedIKStateDeviceHandling:
    """Test SeedIKState device handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensors(self):
        """Test state with CUDA tensors."""
        state = SeedIKState(
            success=torch.tensor([True, False], device="cuda:0"),
            joint_position=torch.randn(2, 7, device="cuda:0"),
        )
        assert state.success.is_cuda
        assert state.joint_position.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_clone_preserves_device(self):
        """Test clone preserves device."""
        state = SeedIKState(
            success=torch.tensor([True, False], device="cuda:0"),
            joint_position=torch.randn(2, 7, device="cuda:0"),
        )
        cloned = state.clone()
        assert cloned.success.is_cuda
        assert cloned.joint_position.is_cuda


class TestSeedIKStateTensorShapes:
    """Test SeedIKState tensor shapes."""

    def test_success_is_1d(self, sample_state):
        """Test success tensor is 1D."""
        assert sample_state.success.ndim == 1

    def test_joint_position_is_2d(self, sample_state):
        """Test joint_position tensor is 2D (batch, dof)."""
        assert sample_state.joint_position.ndim == 2

    def test_jacobian_is_3d(self, sample_state):
        """Test jacobian tensor is 3D (batch, rows, dof)."""
        assert sample_state.jacobian.ndim == 3

    def test_lambda_damping_is_3d(self, sample_state):
        """Test lambda_damping tensor is 3D (batch, 1, 1)."""
        assert sample_state.lambda_damping.ndim == 3
        assert sample_state.lambda_damping.shape[1] == 1
        assert sample_state.lambda_damping.shape[2] == 1

