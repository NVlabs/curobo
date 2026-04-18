# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for SeedIterationStateManager class."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.seed_ik.seed_ik_state import SeedIKState
from curobo._src.solver.seed_ik.seed_iteration_state_manager import SeedIterationStateManager


@pytest.fixture
def state_manager():
    """Create a SeedIterationStateManager for testing."""
    dof = 7
    action_min = torch.zeros(dof) - 2.0
    action_max = torch.zeros(dof) + 2.0

    return SeedIterationStateManager(
        action_min=action_min,
        action_max=action_max,
        rho_min=1e-3,
        lambda_factor=2.0,
        lambda_min=1e-5,
        lambda_max=1e10,
        convergence_position_tolerance=0.005,
        convergence_orientation_tolerance=0.05,
        convergence_joint_limit_weight=1.0,
    )


@pytest.fixture
def sample_current_state():
    """Create a sample current state for testing."""
    batch_size = 4
    dof = 7
    jacobian_rows = 6

    return SeedIKState(
        success=torch.tensor([False, False, False, False]),
        improvement=torch.tensor([False, False, False, False]),
        joint_position=torch.randn(batch_size, dof),
        error_norm=torch.tensor([1.0, 2.0, 1.5, 0.5]),
        jTerror=torch.randn(batch_size, dof),
        jacobian=torch.randn(batch_size, jacobian_rows, dof),
        lambda_damping=torch.ones(batch_size, 1, 1) * 0.1,
        position_errors=torch.tensor([0.01, 0.02, 0.015, 0.001]),
        orientation_errors=torch.tensor([0.1, 0.2, 0.15, 0.01]),
    )


@pytest.fixture
def sample_candidate_state():
    """Create a sample candidate state for testing."""
    batch_size = 4
    dof = 7
    jacobian_rows = 6

    return SeedIKState(
        success=torch.tensor([False, False, False, False]),
        improvement=torch.tensor([False, False, False, False]),
        joint_position=torch.randn(batch_size, dof),
        error_norm=torch.tensor([0.5, 2.5, 1.0, 0.3]),  # Some better, some worse
        jTerror=torch.randn(batch_size, dof),
        jacobian=torch.randn(batch_size, jacobian_rows, dof),
        lambda_damping=torch.ones(batch_size, 1, 1) * 0.1,
        position_errors=torch.tensor([0.005, 0.025, 0.01, 0.0005]),
        orientation_errors=torch.tensor([0.05, 0.25, 0.1, 0.005]),
    )


class TestSeedIterationStateManagerInitialization:
    """Test SeedIterationStateManager initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        dof = 7
        action_min = torch.zeros(dof) - 2.0
        action_max = torch.zeros(dof) + 2.0

        manager = SeedIterationStateManager(
            action_min=action_min,
            action_max=action_max,
            rho_min=1e-3,
            lambda_factor=2.0,
            lambda_min=1e-5,
            lambda_max=1e10,
            convergence_position_tolerance=0.005,
            convergence_orientation_tolerance=0.05,
            convergence_joint_limit_weight=1.0,
        )

        assert manager.rho_min == 1e-3
        assert manager.lambda_factor == 2.0
        assert manager.lambda_min == 1e-5
        assert manager.lambda_max == 1e10

    def test_init_stores_action_bounds(self):
        """Test initialization stores action bounds."""
        dof = 7
        action_min = torch.zeros(dof) - 2.0
        action_max = torch.zeros(dof) + 2.0

        manager = SeedIterationStateManager(
            action_min=action_min,
            action_max=action_max,
            rho_min=1e-3,
            lambda_factor=2.0,
            lambda_min=1e-5,
            lambda_max=1e10,
            convergence_position_tolerance=0.005,
            convergence_orientation_tolerance=0.05,
            convergence_joint_limit_weight=1.0,
        )

        assert torch.equal(manager.action_min, action_min)
        assert torch.equal(manager.action_max, action_max)

    def test_init_stores_convergence_tolerances(self):
        """Test initialization stores convergence tolerances."""
        dof = 7
        action_min = torch.zeros(dof) - 2.0
        action_max = torch.zeros(dof) + 2.0

        manager = SeedIterationStateManager(
            action_min=action_min,
            action_max=action_max,
            rho_min=1e-3,
            lambda_factor=2.0,
            lambda_min=1e-5,
            lambda_max=1e10,
            convergence_position_tolerance=0.005,
            convergence_orientation_tolerance=0.05,
            convergence_joint_limit_weight=1.0,
        )

        assert manager.convergence_position_tolerance == 0.005
        assert manager.convergence_orientation_tolerance == 0.05
        assert manager.convergence_joint_limit_weight == 1.0


class TestSeedIterationStateManagerTrustRegionRatio:
    """Test trust region ratio calculation."""

    def test_calculate_trust_region_ratio_basic(self, state_manager):
        """Test trust region ratio calculation."""
        batch_size = 4
        old_error_norm = torch.tensor([1.0, 2.0, 1.5, 0.5])
        new_error_norm = torch.tensor([0.5, 2.5, 1.0, 0.3])
        predicted_reduction = torch.tensor([[0.5], [0.5], [0.5], [0.2]])

        ratio = state_manager._calculate_trust_region_ratio(
            old_error_norm, new_error_norm, predicted_reduction, batch_size
        )

        assert ratio.shape == (batch_size,)

    def test_calculate_trust_region_ratio_positive_reduction(self, state_manager):
        """Test positive reduction gives positive ratio."""
        old_error_norm = torch.tensor([1.0])
        new_error_norm = torch.tensor([0.5])  # Error reduced
        predicted_reduction = torch.tensor([[0.5]])

        ratio = state_manager._calculate_trust_region_ratio(
            old_error_norm, new_error_norm, predicted_reduction, 1
        )

        assert ratio[0] > 0

    def test_calculate_trust_region_ratio_negative_reduction(self, state_manager):
        """Test negative reduction gives negative ratio."""
        old_error_norm = torch.tensor([0.5])
        new_error_norm = torch.tensor([1.0])  # Error increased
        predicted_reduction = torch.tensor([[0.5]])

        ratio = state_manager._calculate_trust_region_ratio(
            old_error_norm, new_error_norm, predicted_reduction, 1
        )

        assert ratio[0] < 0


class TestSeedIterationStateManagerStepAcceptance:
    """Test step acceptance determination."""

    def test_determine_step_acceptance_accepts_good_ratio(self, state_manager):
        """Test step is accepted when ratio >= rho_min."""
        trust_ratio = torch.tensor([1.0, 0.5, 0.1])  # All > rho_min (1e-3)

        accepted = state_manager._determine_step_acceptance(trust_ratio, 3)

        assert torch.all(accepted)

    def test_determine_step_acceptance_rejects_bad_ratio(self, state_manager):
        """Test step is rejected when ratio < rho_min."""
        trust_ratio = torch.tensor([1e-4, 1e-5, -1.0])  # All < rho_min (1e-3)

        accepted = state_manager._determine_step_acceptance(trust_ratio, 3)

        assert not torch.any(accepted)

    def test_determine_step_acceptance_mixed(self, state_manager):
        """Test mixed acceptance."""
        trust_ratio = torch.tensor([1.0, 1e-5, 0.5, -1.0])

        accepted = state_manager._determine_step_acceptance(trust_ratio, 4)

        expected = torch.tensor([True, False, True, False])
        assert torch.equal(accepted, expected)


class TestSeedIterationStateManagerDampingUpdate:
    """Test damping parameter updates."""

    def test_update_damping_decreases_on_acceptance(self, state_manager):
        """Test damping decreases when step is accepted."""
        batch_size = 2
        current_damping = torch.ones(batch_size, 1, 1) * 0.1
        step_accepted = torch.tensor([True, True])

        updated = state_manager._update_damping_parameter(
            current_damping, step_accepted, batch_size
        )

        # Damping should decrease by lambda_factor (2.0)
        expected = 0.1 / 2.0
        assert torch.allclose(updated, torch.ones_like(updated) * expected)

    def test_update_damping_increases_on_rejection(self, state_manager):
        """Test damping increases when step is rejected."""
        batch_size = 2
        current_damping = torch.ones(batch_size, 1, 1) * 0.1
        step_accepted = torch.tensor([False, False])

        updated = state_manager._update_damping_parameter(
            current_damping, step_accepted, batch_size
        )

        # Damping should increase by lambda_factor (2.0)
        expected = 0.1 * 2.0
        assert torch.allclose(updated, torch.ones_like(updated) * expected)

    def test_update_damping_clamped_to_min(self, state_manager):
        """Test damping is clamped to lambda_min."""
        batch_size = 1
        current_damping = torch.ones(batch_size, 1, 1) * state_manager.lambda_min
        step_accepted = torch.tensor([True])  # Would decrease further

        updated = state_manager._update_damping_parameter(
            current_damping, step_accepted, batch_size
        )

        assert updated[0, 0, 0] >= state_manager.lambda_min

    def test_update_damping_clamped_to_max(self, state_manager):
        """Test damping is clamped to lambda_max."""
        batch_size = 1
        current_damping = torch.ones(batch_size, 1, 1) * state_manager.lambda_max
        step_accepted = torch.tensor([False])  # Would increase further

        updated = state_manager._update_damping_parameter(
            current_damping, step_accepted, batch_size
        )

        assert updated[0, 0, 0] <= state_manager.lambda_max


class TestSeedIterationStateManagerConvergence:
    """Test convergence checking."""

    def test_check_pose_convergence_success(self, state_manager):
        """Test pose convergence when within tolerance."""
        position_errors = torch.tensor([0.001, 0.002])  # < 0.005
        orientation_errors = torch.tensor([0.01, 0.02])  # < 0.05

        converged = state_manager._check_pose_convergence(position_errors, orientation_errors)

        assert torch.all(converged)

    def test_check_pose_convergence_failure_position(self, state_manager):
        """Test pose convergence fails when position error too high."""
        position_errors = torch.tensor([0.01, 0.02])  # > 0.005
        orientation_errors = torch.tensor([0.01, 0.02])  # < 0.05

        converged = state_manager._check_pose_convergence(position_errors, orientation_errors)

        assert not torch.any(converged)

    def test_check_pose_convergence_failure_orientation(self, state_manager):
        """Test pose convergence fails when orientation error too high."""
        position_errors = torch.tensor([0.001, 0.002])  # < 0.005
        orientation_errors = torch.tensor([0.1, 0.2])  # > 0.05

        converged = state_manager._check_pose_convergence(position_errors, orientation_errors)

        assert not torch.any(converged)

    def test_check_joint_limit_satisfaction_within_limits(self, state_manager):
        """Test joint limit satisfaction when within limits."""
        joint_position = torch.zeros(2, 7)  # All zeros, within [-2, 2]

        satisfied = state_manager._check_joint_limit_satisfaction(joint_position)

        assert torch.all(satisfied)

    def test_check_joint_limit_satisfaction_outside_limits(self, state_manager):
        """Test joint limit satisfaction fails when outside limits."""
        joint_position = torch.ones(2, 7) * 3.0  # All 3.0, outside [-2, 2]

        satisfied = state_manager._check_joint_limit_satisfaction(joint_position)

        assert not torch.any(satisfied)


class TestSeedIterationStateManagerUpdateState:
    """Test full state update."""

    def test_update_iteration_state_returns_seed_ik_state(
        self, state_manager, sample_current_state, sample_candidate_state
    ):
        """Test update_iteration_state returns SeedIKState."""
        predicted_reduction = torch.tensor([[0.5], [0.5], [0.5], [0.2]])

        updated = state_manager.update_iteration_state(
            sample_current_state,
            sample_candidate_state,
            predicted_reduction,
            batch_size=4,
        )

        assert isinstance(updated, SeedIKState)

    def test_update_iteration_state_has_all_fields(
        self, state_manager, sample_current_state, sample_candidate_state
    ):
        """Test update_iteration_state result has all fields."""
        predicted_reduction = torch.tensor([[0.5], [0.5], [0.5], [0.2]])

        updated = state_manager.update_iteration_state(
            sample_current_state,
            sample_candidate_state,
            predicted_reduction,
            batch_size=4,
        )

        assert updated.joint_position is not None
        assert updated.lambda_damping is not None
        assert updated.error_norm is not None
        assert updated.success is not None
        assert updated.improvement is not None


class TestSeedIterationStateManagerCUDA:
    """Test SeedIterationStateManager with CUDA tensors."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_trust_region_ratio_cuda(self):
        """Test trust region ratio calculation on CUDA."""
        dof = 7
        action_min = torch.zeros(dof, device="cuda:0") - 2.0
        action_max = torch.zeros(dof, device="cuda:0") + 2.0

        manager = SeedIterationStateManager(
            action_min=action_min,
            action_max=action_max,
            rho_min=1e-3,
            lambda_factor=2.0,
            lambda_min=1e-5,
            lambda_max=1e10,
            convergence_position_tolerance=0.005,
            convergence_orientation_tolerance=0.05,
            convergence_joint_limit_weight=1.0,
        )

        old_error = torch.tensor([1.0], device="cuda:0")
        new_error = torch.tensor([0.5], device="cuda:0")
        predicted = torch.tensor([[0.5]], device="cuda:0")

        ratio = manager._calculate_trust_region_ratio(old_error, new_error, predicted, 1)

        assert ratio.is_cuda

