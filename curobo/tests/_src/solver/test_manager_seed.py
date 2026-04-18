# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for SeedManager class."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.manager_seed import SeedManager
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def cpu_device_cfg():
    """Create a CPU device configuration for testing."""
    return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)


@pytest.fixture
def cuda_device_cfg():
    """Create a CUDA device configuration for testing."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture
def action_bounds(cpu_device_cfg):
    """Create action bounds for 7-DOF robot."""
    dof = 7
    lows = torch.zeros(dof, **cpu_device_cfg.as_torch_dict()) - 2.0
    highs = torch.zeros(dof, **cpu_device_cfg.as_torch_dict()) + 2.0
    return lows, highs


@pytest.fixture
def seed_manager_ik(cpu_device_cfg, action_bounds):
    """Create a SeedManager for IK (action_horizon=1)."""
    lows, highs = action_bounds
    return SeedManager(
        device_cfg=cpu_device_cfg,
        action_dim=7,
        action_bound_lows=lows,
        action_bound_highs=highs,
        random_seed=42,
        action_horizon=1,
    )


@pytest.fixture
def seed_manager_trajopt(cpu_device_cfg, action_bounds):
    """Create a SeedManager for TrajOpt (action_horizon>1)."""
    lows, highs = action_bounds
    return SeedManager(
        device_cfg=cpu_device_cfg,
        action_dim=7,
        action_bound_lows=lows,
        action_bound_highs=highs,
        random_seed=42,
        action_horizon=32,
    )


class TestSeedManagerInitialization:
    """Test SeedManager initialization."""

    def test_init_basic(self, cpu_device_cfg, action_bounds):
        """Test basic initialization."""
        lows, highs = action_bounds
        manager = SeedManager(
            device_cfg=cpu_device_cfg,
            action_dim=7,
            action_bound_lows=lows,
            action_bound_highs=highs,
        )
        assert manager.action_dim == 7
        assert manager.action_horizon == 1

    def test_init_with_action_horizon(self, cpu_device_cfg, action_bounds):
        """Test initialization with action_horizon > 1."""
        lows, highs = action_bounds
        manager = SeedManager(
            device_cfg=cpu_device_cfg,
            action_dim=7,
            action_bound_lows=lows,
            action_bound_highs=highs,
            action_horizon=32,
        )
        assert manager.action_horizon == 32
        assert manager.trajectory_seed_generator is not None

    def test_init_creates_sample_generator(self, seed_manager_ik):
        """Test that sample generator is created."""
        assert seed_manager_ik.action_sample_generator is not None

    def test_init_trajectory_generator_none_for_ik(self, seed_manager_ik):
        """Test trajectory generator is None for IK (horizon=1)."""
        assert seed_manager_ik.trajectory_seed_generator is None

    def test_init_trajectory_generator_exists_for_trajopt(self, seed_manager_trajopt):
        """Test trajectory generator exists for trajopt."""
        assert seed_manager_trajopt.trajectory_seed_generator is not None


class TestSeedManagerPrepareActionSeeds:
    """Test SeedManager.prepare_action_seeds method."""

    def test_prepare_action_seeds_no_seed_config(self, seed_manager_ik):
        """Test preparing seeds without seed_config."""
        num_seeds = 32
        batch_size = 4

        seeds = seed_manager_ik.prepare_action_seeds(batch_size, num_seeds)

        assert seeds.shape == (batch_size * num_seeds, 1, 7)

    def test_prepare_action_seeds_with_exact_seed_config(self, seed_manager_ik, cpu_device_cfg):
        """Test preparing seeds with exact number of seed configs."""
        num_seeds = 4
        batch_size = 2
        dof = 7

        seed_config = torch.randn(batch_size, num_seeds, dof, **cpu_device_cfg.as_torch_dict())
        seeds = seed_manager_ik.prepare_action_seeds(batch_size, num_seeds, seed_config=seed_config)

        assert seeds.shape == (batch_size * num_seeds, 1, dof)

    def test_prepare_action_seeds_with_fewer_seed_config(self, seed_manager_ik, cpu_device_cfg):
        """Test preparing seeds with fewer seed configs than required."""
        num_seeds = 8
        batch_size = 2
        dof = 7
        n_provided = 2

        seed_config = torch.randn(batch_size, n_provided, dof, **cpu_device_cfg.as_torch_dict())
        seeds = seed_manager_ik.prepare_action_seeds(batch_size, num_seeds, seed_config=seed_config)

        # Should fill remaining seeds with random values
        assert seeds.shape == (batch_size * num_seeds, 1, dof)

    def test_prepare_action_seeds_with_more_seed_config(self, seed_manager_ik, cpu_device_cfg):
        """Test preparing seeds with more seed configs than required."""
        num_seeds = 4
        batch_size = 2
        dof = 7
        n_provided = 8

        seed_config = torch.randn(batch_size, n_provided, dof, **cpu_device_cfg.as_torch_dict())
        seeds = seed_manager_ik.prepare_action_seeds(batch_size, num_seeds, seed_config=seed_config)

        # Should only use first num_seeds
        assert seeds.shape == (batch_size * num_seeds, 1, dof)

    def test_prepare_action_seeds_transposed_input(self, seed_manager_ik, cpu_device_cfg):
        """Test preparing seeds with (n, batch, dof) format."""
        num_seeds = 4
        batch_size = 2
        dof = 7

        # Input in (num_seeds, batch, dof) format
        seed_config = torch.randn(num_seeds, batch_size, dof, **cpu_device_cfg.as_torch_dict())
        seeds = seed_manager_ik.prepare_action_seeds(batch_size, num_seeds, seed_config=seed_config)

        assert seeds.shape == (batch_size * num_seeds, 1, dof)

    def test_prepare_action_seeds_single_batch(self, seed_manager_ik):
        """Test preparing seeds for single batch."""
        num_seeds = 16
        batch_size = 1

        seeds = seed_manager_ik.prepare_action_seeds(batch_size, num_seeds)

        assert seeds.shape == (num_seeds, 1, 7)


class TestSeedManagerGenerateRandomActions:
    """Test SeedManager.generate_random_actions method."""

    def test_generate_random_actions_basic(self, seed_manager_ik):
        """Test basic random action generation."""
        num_seeds = 10
        batch = 4

        actions = seed_manager_ik.generate_random_actions(batch, num_seeds)

        assert actions.shape == (batch, num_seeds, 7)

    def test_generate_random_actions_zero_seeds(self, seed_manager_ik):
        """Test with zero seeds requested."""
        actions = seed_manager_ik.generate_random_actions(4, 0)

        assert actions.shape == (4, 1, 7)

    def test_generate_random_actions_within_bounds(self, seed_manager_ik):
        """Test generated actions are within bounds."""
        actions = seed_manager_ik.generate_random_actions(1, 100)

        assert torch.all(actions >= -2.0)
        assert torch.all(actions <= 2.0)


class TestSeedManagerResetSeed:
    """Test SeedManager.reset_seed method."""

    def test_reset_seed_no_error(self, seed_manager_ik):
        """Test reset_seed doesn't raise error."""
        seed_manager_ik.reset_seed()

    def test_reset_seed_changes_sequence(self, seed_manager_ik):
        """Test reset_seed resets the random sequence."""
        # Generate some seeds
        seeds1 = seed_manager_ik.generate_random_actions(1, 10)

        # Reset and generate again
        seed_manager_ik.reset_seed()
        seeds2 = seed_manager_ik.generate_random_actions(1, 10)

        # After reset, should get same sequence (deterministic Halton)
        assert torch.allclose(seeds1, seeds2)


class TestSeedManagerCUDA:
    """Test SeedManager with CUDA tensors."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_generate_random_actions_cuda(self, cuda_device_cfg):
        """Test random action generation on CUDA."""
        dof = 7
        lows = torch.zeros(dof, **cuda_device_cfg.as_torch_dict()) - 2.0
        highs = torch.zeros(dof, **cuda_device_cfg.as_torch_dict()) + 2.0

        manager = SeedManager(
            device_cfg=cuda_device_cfg,
            action_dim=dof,
            action_bound_lows=lows,
            action_bound_highs=highs,
        )

        actions = manager.generate_random_actions(4, 10)

        assert actions.is_cuda
        assert actions.shape == (4, 10, dof)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_prepare_action_seeds_cuda(self, cuda_device_cfg):
        """Test prepare_action_seeds on CUDA."""
        dof = 7
        lows = torch.zeros(dof, **cuda_device_cfg.as_torch_dict()) - 2.0
        highs = torch.zeros(dof, **cuda_device_cfg.as_torch_dict()) + 2.0

        manager = SeedManager(
            device_cfg=cuda_device_cfg,
            action_dim=dof,
            action_bound_lows=lows,
            action_bound_highs=highs,
        )

        seeds = manager.prepare_action_seeds(2, 16)

        assert seeds.is_cuda
        assert seeds.shape == (32, 1, dof)


class TestSeedManagerPrepareTrajectorySeeds:
    """Test SeedManager.prepare_trajectory_seeds method."""

    def test_prepare_trajectory_seeds_constant(self, seed_manager_trajopt, cpu_device_cfg):
        """Test preparing trajectory seeds with constant position."""
        num_seeds = 4
        batch_size = 2
        dof = 7
        horizon = 32

        current_state = JointState.from_position(
            torch.randn(batch_size, dof, **cpu_device_cfg.as_torch_dict())
        )

        seeds = seed_manager_trajopt.prepare_trajectory_seeds(
            batch_size, num_seeds, current_state
        )

        assert seeds.shape == (batch_size * num_seeds, horizon, dof)

    def test_prepare_trajectory_seeds_with_seed_config(self, seed_manager_trajopt, cpu_device_cfg):
        """Test preparing trajectory seeds with target configs."""
        num_seeds = 4
        batch_size = 2
        dof = 7
        horizon = 32

        current_state = JointState.from_position(
            torch.randn(batch_size, dof, **cpu_device_cfg.as_torch_dict())
        )
        seed_config = torch.randn(batch_size, num_seeds, dof, **cpu_device_cfg.as_torch_dict())

        seeds = seed_manager_trajopt.prepare_trajectory_seeds(
            batch_size, num_seeds, current_state, seed_config=seed_config
        )

        assert seeds.shape == (batch_size * num_seeds, horizon, dof)

    def test_prepare_trajectory_seeds_with_seed_traj(self, seed_manager_trajopt, cpu_device_cfg):
        """Test preparing trajectory seeds with full trajectory seeds."""
        num_seeds = 4
        batch_size = 2
        dof = 7
        horizon = 32

        current_state = JointState.from_position(
            torch.randn(batch_size, dof, **cpu_device_cfg.as_torch_dict())
        )
        seed_traj = torch.randn(batch_size, num_seeds, horizon, dof, **cpu_device_cfg.as_torch_dict())

        seeds = seed_manager_trajopt.prepare_trajectory_seeds(
            batch_size, num_seeds, current_state, seed_traj=seed_traj
        )

        assert seeds.shape == (batch_size * num_seeds, horizon, dof)

    def test_prepare_trajectory_seeds_errors_without_generator(self, seed_manager_ik, cpu_device_cfg):
        """Test error when trajectory generator not initialized."""
        current_state = JointState.from_position(
            torch.randn(2, 7, **cpu_device_cfg.as_torch_dict())
        )

        with pytest.raises(ValueError):
            seed_manager_ik.prepare_trajectory_seeds(2, 4, current_state)

