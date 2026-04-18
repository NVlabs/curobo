# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for MPCSolverResult dataclass."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.solver_mpc_result import MPCSolverResult
from curobo._src.state.state_joint import JointState


@pytest.fixture
def sample_mpc_result():
    """Create a sample MPCSolverResult for testing."""
    batch_size = 2
    dof = 7

    return MPCSolverResult(
        success=torch.tensor([True, False]),
        position_error=torch.rand(batch_size),
        rotation_error=torch.rand(batch_size),
        solve_time=0.01,
        total_time=0.02,
    )


class TestMPCSolverResultDataclass:
    """Test MPCSolverResult dataclass attributes."""

    def test_has_success_field(self):
        """Test MPCSolverResult has success field."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert result.success is not None

    def test_has_next_action_field(self):
        """Test MPCSolverResult has next_action field."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert hasattr(result, 'next_action')

    def test_has_action_sequence_field(self):
        """Test MPCSolverResult has action_sequence field."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert hasattr(result, 'action_sequence')

    def test_has_full_action_sequence_field(self):
        """Test MPCSolverResult has full_action_sequence field."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert hasattr(result, 'full_action_sequence')

    def test_has_robot_state_sequence_field(self):
        """Test MPCSolverResult has robot_state_sequence field."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert hasattr(result, 'robot_state_sequence')

    def test_default_next_action_is_none(self):
        """Test default next_action is None."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert result.next_action is None

    def test_default_action_sequence_is_none(self):
        """Test default action_sequence is None."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert result.action_sequence is None

    def test_default_full_action_sequence_is_none(self):
        """Test default full_action_sequence is None."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert result.full_action_sequence is None

    def test_default_robot_state_sequence_is_none(self):
        """Test default robot_state_sequence is None."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert result.robot_state_sequence is None


class TestMPCSolverResultClone:
    """Test MPCSolverResult.clone method."""

    def test_clone_creates_copy(self, sample_mpc_result):
        """Test clone creates a new object."""
        cloned = sample_mpc_result.clone()
        assert cloned is not sample_mpc_result

    def test_clone_preserves_success(self, sample_mpc_result):
        """Test clone preserves success tensor."""
        cloned = sample_mpc_result.clone()
        assert torch.equal(cloned.success, sample_mpc_result.success)

    def test_clone_preserves_position_error(self, sample_mpc_result):
        """Test clone preserves position_error tensor."""
        cloned = sample_mpc_result.clone()
        assert torch.equal(cloned.position_error, sample_mpc_result.position_error)

    def test_clone_preserves_rotation_error(self, sample_mpc_result):
        """Test clone preserves rotation_error tensor."""
        cloned = sample_mpc_result.clone()
        assert torch.equal(cloned.rotation_error, sample_mpc_result.rotation_error)

    def test_clone_with_next_action(self):
        """Test clone handles next_action correctly."""
        dof = 7
        next_action = JointState.from_position(torch.randn(1, dof))

        result = MPCSolverResult(
            success=torch.tensor([True]),
            next_action=next_action,
        )

        cloned = result.clone()
        assert cloned.next_action is not None

    def test_clone_with_action_sequence(self):
        """Test clone handles action_sequence correctly."""
        horizon = 10
        dof = 7
        action_seq = JointState.from_position(torch.randn(1, horizon, dof))

        result = MPCSolverResult(
            success=torch.tensor([True]),
            action_sequence=action_seq,
        )

        cloned = result.clone()
        assert cloned.action_sequence is not None


class TestMPCSolverResultInheritsFromBaseSolverResult:
    """Test MPCSolverResult inherits from BaseSolverResult."""

    def test_has_position_error(self, sample_mpc_result):
        """Test has position_error from base class."""
        assert hasattr(sample_mpc_result, 'position_error')

    def test_has_rotation_error(self, sample_mpc_result):
        """Test has rotation_error from base class."""
        assert hasattr(sample_mpc_result, 'rotation_error')

    def test_has_goalset_index(self, sample_mpc_result):
        """Test has goalset_index from base class."""
        assert hasattr(sample_mpc_result, 'goalset_index')

    def test_has_solve_time(self, sample_mpc_result):
        """Test has solve_time from base class."""
        assert hasattr(sample_mpc_result, 'solve_time')
        assert sample_mpc_result.solve_time == 0.01

    def test_has_total_time(self, sample_mpc_result):
        """Test has total_time from base class."""
        assert hasattr(sample_mpc_result, 'total_time')
        assert sample_mpc_result.total_time == 0.02

    def test_has_solution(self, sample_mpc_result):
        """Test has solution from base class."""
        assert hasattr(sample_mpc_result, 'solution')

    def test_has_js_solution(self, sample_mpc_result):
        """Test has js_solution from base class."""
        assert hasattr(sample_mpc_result, 'js_solution')

    def test_has_debug_info(self, sample_mpc_result):
        """Test has debug_info from base class."""
        assert hasattr(sample_mpc_result, 'debug_info')


class TestMPCSolverResultDeviceHandling:
    """Test MPCSolverResult device handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensors(self):
        """Test result with CUDA tensors."""
        result = MPCSolverResult(
            success=torch.tensor([True, False], device="cuda:0"),
            position_error=torch.rand(2, device="cuda:0"),
        )
        assert result.success.is_cuda
        assert result.position_error.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_clone_preserves_device(self):
        """Test clone preserves device."""
        result = MPCSolverResult(
            success=torch.tensor([True, False], device="cuda:0"),
            position_error=torch.rand(2, device="cuda:0"),
        )
        cloned = result.clone()
        assert cloned.success.is_cuda


class TestMPCSolverResultWithJointState:
    """Test MPCSolverResult with JointState fields."""

    def test_next_action_is_joint_state(self):
        """Test next_action can be JointState."""
        dof = 7
        next_action = JointState.from_position(torch.randn(1, dof))

        result = MPCSolverResult(
            success=torch.tensor([True]),
            next_action=next_action,
        )

        assert isinstance(result.next_action, JointState)

    def test_action_sequence_is_joint_state(self):
        """Test action_sequence can be JointState."""
        horizon = 10
        dof = 7
        action_seq = JointState.from_position(torch.randn(1, horizon, dof))

        result = MPCSolverResult(
            success=torch.tensor([True]),
            action_sequence=action_seq,
        )

        assert isinstance(result.action_sequence, JointState)

    def test_full_action_sequence_is_joint_state(self):
        """Test full_action_sequence can be JointState."""
        full_horizon = 50
        dof = 7
        full_action_seq = JointState.from_position(torch.randn(1, full_horizon, dof))

        result = MPCSolverResult(
            success=torch.tensor([True]),
            full_action_sequence=full_action_seq,
        )

        assert isinstance(result.full_action_sequence, JointState)

