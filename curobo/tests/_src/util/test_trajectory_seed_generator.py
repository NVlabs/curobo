# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg

# Add the project root to the path
from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator


@pytest.fixture
def device_cfg():
    return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)


@pytest.fixture
def action_horizon():
    return 10


@pytest.fixture
def action_dim():
    return 7  # Typical dimension for a 7-DOF robot arm


@pytest.fixture
def generator(device_cfg, action_horizon, action_dim):
    generator = TrajectorySeedGenerator(action_horizon, action_dim, device_cfg)
    return generator


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def num_seeds():
    return 3


class TestInitialization:
    def test_initialization(self, generator, action_horizon):
        """Test that the generator initializes properly."""
        assert generator.action_horizon == action_horizon
        assert hasattr(generator, "_interpolation_weights")
        # Check shape of interpolation weights
        assert generator._interpolation_weights.shape == (1, action_horizon, 2, 1)


class TestInterpolation:
    def test_interpolate_trajectory(self, generator, device_cfg, action_dim, batch_size, num_seeds):
        """Test the internal interpolation method."""
        # Create test data
        start_positions = torch.rand((batch_size, num_seeds, action_dim), device=device_cfg.device)
        goal_positions = torch.rand((batch_size, num_seeds, action_dim), device=device_cfg.device)

        # Call the internal interpolation method
        interpolated = generator._interpolate_trajectory(start_positions, goal_positions)

        # Check shape
        assert interpolated.shape == (batch_size, num_seeds, generator.action_horizon, action_dim)

        # Check that start and end points influence the interpolation correctly
        first_frame = interpolated[:, :, 0, :]
        last_frame = interpolated[:, :, -1, :]

        # First frame should be more influenced by start_positions
        # Last frame should be more influenced by goal_positions
        assert torch.mean(torch.norm(first_frame - start_positions, dim=-1)) < torch.mean(
            torch.norm(first_frame - goal_positions, dim=-1)
        )

        assert torch.mean(torch.norm(last_frame - goal_positions, dim=-1)) < torch.mean(
            torch.norm(last_frame - start_positions, dim=-1)
        )

    def test_generate_interpolated_seeds(
        self, generator, device_cfg, action_dim, batch_size, num_seeds
    ):
        """Test the interpolated seed generation."""
        # Create test data
        start_position = torch.rand((batch_size, action_dim), device=device_cfg.device)
        goal_position = torch.rand((batch_size, num_seeds, action_dim), device=device_cfg.device)

        # Generate seeds
        interpolated_seeds = generator.generate_interpolated_seeds(
            start_position=start_position, goal_position=goal_position, num_seeds=num_seeds
        )

        # Check output shape
        assert interpolated_seeds.shape == (
            batch_size,
            num_seeds,
            generator.action_horizon,
            action_dim,
        )

        # Check that the interpolation works as expected
        # First point should be closer to start, last closer to end
        for b in range(batch_size):
            for s in range(num_seeds):
                # First point should be very close to start
                first_point = interpolated_seeds[b, s, 0]
                last_point = interpolated_seeds[b, s, -1]

                # The first point should be closer to start than the last point is
                start_dist_first = torch.norm(first_point - start_position[b])
                start_dist_last = torch.norm(last_point - start_position[b])
                assert start_dist_first < start_dist_last

                # The last point should be closer to goal than the first point is
                goal_dist_first = torch.norm(first_point - goal_position[b, s])
                goal_dist_last = torch.norm(last_point - goal_position[b, s])
                assert goal_dist_last < goal_dist_first


class TestConstantSeeds:
    def test_generate_constant_seeds(self, generator, device_cfg, action_dim, batch_size, num_seeds):
        """Test the constant seed generation."""
        # Create test data
        constant_position = torch.rand((batch_size, action_dim), device=device_cfg.device)

        # Generate seeds

        constant_seeds = generator.generate_constant_seeds(
            constant_position=constant_position, num_seeds=num_seeds
        )

        # Check output shape
        assert constant_seeds.shape == (batch_size, num_seeds, generator.action_horizon, action_dim)
        # Check that all points in the trajectory are equal to the input
        for b in range(batch_size):
            for s in range(num_seeds):
                # All points should be identical
                for t in range(generator.action_horizon):
                    torch.testing.assert_close(constant_seeds[b, s, t], constant_position[b])


class TestInputValidation:
    @pytest.mark.parametrize("invalid_seeds", [0, -1])
    def test_invalid_num_seeds(
        self, generator, device_cfg, action_dim, batch_size, num_seeds, invalid_seeds
    ):
        """Test validation for invalid number of seeds."""
        valid_start = torch.rand((batch_size, action_dim), device=device_cfg.device)
        valid_goal = torch.rand((batch_size, num_seeds, action_dim), device=device_cfg.device)
        valid_constant = torch.rand((batch_size, num_seeds, action_dim), device=device_cfg.device)

        with pytest.raises(Exception):
            generator.generate_interpolated_seeds(
                num_seeds=invalid_seeds, start_position=valid_start, goal_position=valid_goal
            )

        with pytest.raises(Exception):
            generator.generate_constant_seeds(
                num_seeds=invalid_seeds, constant_position=valid_constant
            )

    def test_invalid_dimensions(self, generator, device_cfg, action_dim, batch_size, num_seeds):
        """Test validation for tensors with invalid dimensions."""
        valid_start = torch.rand((batch_size, action_dim), device=device_cfg.device)
        valid_goal = torch.rand((batch_size, num_seeds, action_dim), device=device_cfg.device)
        valid_constant = torch.rand((batch_size, action_dim), device=device_cfg.device)

        # Test with invalid dimension sizes
        invalid_start_dim = torch.rand((batch_size, action_dim + 1), device=device_cfg.device)
        with pytest.raises(Exception):
            generator.generate_interpolated_seeds(
                num_seeds=num_seeds, start_position=invalid_start_dim, goal_position=valid_goal
            )

        invalid_goal_dim = torch.rand(
            (batch_size, num_seeds, action_dim + 1), device=device_cfg.device
        )
        with pytest.raises(Exception):
            generator.generate_interpolated_seeds(num_seeds, valid_start, invalid_goal_dim)

        invalid_constant_dim = torch.rand(
            (batch_size, action_dim + 1), device=device_cfg.device
        )
        with pytest.raises(Exception):
            generator.generate_constant_seeds(
                num_seeds=num_seeds, constant_position=invalid_constant_dim
            )

        # Test mismatched batch sizes
        invalid_batch_start = torch.rand((batch_size + 1, action_dim), device=device_cfg.device)
        with pytest.raises(Exception):
            generator.generate_interpolated_seeds(
                num_seeds=num_seeds, start_position=invalid_batch_start, goal_position=valid_goal
            )

        # Test mismatched number of seeds
        invalid_num_seeds_goal = torch.rand((batch_size, num_seeds + 1, action_dim), device=device_cfg.device)
        with pytest.raises(Exception):
            generator.generate_interpolated_seeds(num_seeds, valid_start, invalid_num_seeds_goal)

    def test_wrong_tensor_ndim(self, generator, device_cfg, action_dim, batch_size, num_seeds):
        """Test validation for tensors with wrong number of dimensions."""
        valid_start = torch.rand((batch_size, action_dim), device=device_cfg.device)
        valid_goal = torch.rand((batch_size, num_seeds, action_dim), device=device_cfg.device)

        # Test with wrong number of dimensions
        wrong_start_ndim = torch.rand((batch_size, num_seeds, action_dim), device=device_cfg.device)
        with pytest.raises(Exception):
            generator.generate_interpolated_seeds(
                num_seeds=num_seeds, start_position=wrong_start_ndim, goal_position=valid_goal
            )

        wrong_goal_ndim = torch.rand((batch_size, action_dim), device=device_cfg.device)
        with pytest.raises(Exception):
            generator.generate_interpolated_seeds(
                num_seeds=num_seeds, start_position=valid_start, goal_position=wrong_goal_ndim
            )

        wrong_constant_ndim = torch.rand(
            (batch_size, num_seeds, action_dim), device=device_cfg.device
        )
        with pytest.raises(Exception):
            generator.generate_constant_seeds(
                num_seeds=num_seeds, constant_position=wrong_constant_ndim
            )


class TestDecelerationSeeds:
    def test_generate_deceleration_seeds_basic(
        self, generator, device_cfg, action_dim, batch_size, num_seeds
    ):
        """Test basic deceleration seed generation."""
        # Create a current state with velocity
        position = torch.rand((batch_size, action_dim), device=device_cfg.device)
        velocity = torch.rand((batch_size, action_dim), device=device_cfg.device) * 2.0 - 1.0
        acceleration = torch.rand((batch_size, action_dim), device=device_cfg.device) * 0.5
        dt = torch.tensor([0.1], device=device_cfg.device)

        current_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            dt=dt,
            device_cfg=device_cfg,
        )

        # Generate deceleration seeds
        decel_seeds = generator.generate_deceleration_seeds(
            current_state=current_state, num_seeds=num_seeds, deceleration_profile="exponential"
        )

        # Check output shape
        assert decel_seeds.shape == (batch_size, num_seeds, generator.action_horizon, action_dim)

        # Check that trajectory evolves from current position
        for b in range(batch_size):
            first_point = decel_seeds[b, 0, 0]
            # First point should be close to current position
            torch.testing.assert_close(first_point, position[b], rtol=1e-3, atol=1e-3)

    def test_generate_deceleration_seeds_linear_profile(
        self, generator, device_cfg, action_dim, batch_size, num_seeds
    ):
        """Test deceleration with linear profile."""
        position = torch.rand((batch_size, action_dim), device=device_cfg.device)
        velocity = torch.ones((batch_size, action_dim), device=device_cfg.device) * 1.0
        acceleration = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        dt = torch.tensor([0.1], device=device_cfg.device)

        current_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            dt=dt,
            device_cfg=device_cfg,
        )

        decel_seeds = generator.generate_deceleration_seeds(
            current_state=current_state, num_seeds=num_seeds, deceleration_profile="linear"
        )

        # Check output shape
        assert decel_seeds.shape == (batch_size, num_seeds, generator.action_horizon, action_dim)

    def test_generate_deceleration_seeds_smooth_profile(
        self, generator, device_cfg, action_dim, batch_size, num_seeds
    ):
        """Test deceleration with smooth profile."""
        position = torch.rand((batch_size, action_dim), device=device_cfg.device)
        velocity = torch.ones((batch_size, action_dim), device=device_cfg.device) * 1.0
        acceleration = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        dt = torch.tensor([0.1], device=device_cfg.device)

        current_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            dt=dt,
            device_cfg=device_cfg,
        )

        decel_seeds = generator.generate_deceleration_seeds(
            current_state=current_state, num_seeds=num_seeds, deceleration_profile="smooth"
        )

        # Check output shape
        assert decel_seeds.shape == (batch_size, num_seeds, generator.action_horizon, action_dim)

    def test_generate_deceleration_seeds_zero_velocity(
        self, generator, device_cfg, action_dim, batch_size, num_seeds
    ):
        """Test deceleration when velocity is zero."""
        position = torch.rand((batch_size, action_dim), device=device_cfg.device)
        velocity = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        acceleration = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        dt = torch.tensor([0.1], device=device_cfg.device)

        current_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            dt=dt,
            device_cfg=device_cfg,
        )

        decel_seeds = generator.generate_deceleration_seeds(
            current_state=current_state, num_seeds=num_seeds
        )

        # Check output shape
        assert decel_seeds.shape == (batch_size, num_seeds, generator.action_horizon, action_dim)

        # When velocity is zero, position should remain mostly constant
        for b in range(batch_size):
            for s in range(num_seeds):
                # All points should be close to initial position since there's no velocity
                for t in range(generator.action_horizon):
                    torch.testing.assert_close(
                        decel_seeds[b, s, t], position[b], rtol=0.1, atol=0.1
                    )

    def test_generate_deceleration_seeds_no_velocity_attribute(
        self, generator, device_cfg, action_dim, batch_size, num_seeds
    ):
        """Test deceleration when velocity is None."""
        position = torch.rand((batch_size, action_dim), device=device_cfg.device)
        dt = torch.tensor([0.1], device=device_cfg.device)

        current_state = JointState(
            position=position, velocity=None, acceleration=None, dt=dt, device_cfg=device_cfg
        )

        decel_seeds = generator.generate_deceleration_seeds(
            current_state=current_state, num_seeds=num_seeds
        )

        # Check output shape
        assert decel_seeds.shape == (batch_size, num_seeds, generator.action_horizon, action_dim)

    def test_deceleration_invalid_inputs(self, generator, device_cfg, action_dim, batch_size):
        """Test validation for invalid deceleration inputs."""
        position = torch.rand((batch_size, action_dim), device=device_cfg.device)
        dt = torch.tensor([0.1], device=device_cfg.device)

        current_state = JointState(
            position=position, velocity=None, acceleration=None, dt=dt, device_cfg=device_cfg
        )

        # Test invalid num_seeds
        with pytest.raises(Exception):
            generator.generate_deceleration_seeds(current_state=current_state, num_seeds=0)

        with pytest.raises(Exception):
            generator.generate_deceleration_seeds(current_state=current_state, num_seeds=-1)

        # Test invalid state dimensions
        invalid_position = torch.rand(
            (batch_size, 3, action_dim), device=device_cfg.device
        )  # 3D instead of 2D
        invalid_state = JointState(
            position=invalid_position,
            velocity=None,
            acceleration=None,
            dt=dt,
            device_cfg=device_cfg,
        )

        with pytest.raises(Exception):
            generator.generate_deceleration_seeds(current_state=invalid_state, num_seeds=3)

        # Test invalid action_dim
        invalid_action_dim_position = torch.rand(
            (batch_size, action_dim + 1), device=device_cfg.device
        )
        invalid_action_dim_state = JointState(
            position=invalid_action_dim_position,
            velocity=None,
            acceleration=None,
            dt=dt,
            device_cfg=device_cfg,
        )

        with pytest.raises(Exception):
            generator.generate_deceleration_seeds(current_state=invalid_action_dim_state, num_seeds=3)


class TestDecelerationProfiles:
    def test_linear_deceleration_profile(self, generator):
        """Test linear deceleration profile generation."""
        decel_steps = 10
        profile = generator._generate_linear_deceleration_profile(decel_steps)

        # Check shape
        assert profile.shape == (decel_steps,)

        # Check values decrease linearly
        assert profile[0].cpu().item() == pytest.approx(1.0, abs=1e-5)
        assert profile[-1].cpu().item() == pytest.approx(0.0, abs=1e-5)

        # Check monotonically decreasing
        for i in range(len(profile) - 1):
            assert profile[i] >= profile[i + 1]

    def test_linear_deceleration_profile_empty(self, generator):
        """Test linear deceleration profile with zero steps."""
        profile = generator._generate_linear_deceleration_profile(0)
        assert profile.shape == (0,)

    def test_exponential_deceleration_profile(self, generator):
        """Test exponential deceleration profile generation."""
        decel_steps = 10
        profile = generator._generate_exponential_deceleration_profile(decel_steps)

        # Check shape
        assert profile.shape == (decel_steps,)

        # Check monotonically decreasing
        for i in range(len(profile) - 1):
            assert profile[i] >= profile[i + 1]

        # First value should be close to 1.0
        assert profile[0].cpu().item() == pytest.approx(1.0, abs=1e-5)

    def test_exponential_deceleration_profile_empty(self, generator):
        """Test exponential deceleration profile with zero steps."""
        profile = generator._generate_exponential_deceleration_profile(0)
        assert profile.shape == (0,)

    def test_smooth_deceleration_profile(self, generator):
        """Test smooth (cosine-based) deceleration profile generation."""
        decel_steps = 10
        profile = generator._generate_smooth_deceleration_profile(decel_steps)

        # Check shape
        assert profile.shape == (decel_steps,)

        # Check monotonically decreasing
        for i in range(len(profile) - 1):
            assert profile[i] >= profile[i + 1]

        # First value should be close to 1.0
        assert profile[0].cpu().item() == pytest.approx(1.0, abs=1e-5)
        # Last value should be close to 0.0
        assert profile[-1].cpu().item() == pytest.approx(0.0, abs=1e-5)

    def test_smooth_deceleration_profile_empty(self, generator):
        """Test smooth deceleration profile with zero steps."""
        profile = generator._generate_smooth_deceleration_profile(0)
        assert profile.shape == (0,)


class TestAccelerationProfileGeneration:
    def test_deceleration_acceleration_profile_linear(
        self, generator, device_cfg, action_dim, batch_size
    ):
        """Test acceleration profile generation with linear deceleration."""
        current_vel = torch.ones((batch_size, action_dim), device=device_cfg.device)
        current_acc = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        dt = 0.1

        profile = generator._generate_deceleration_acceleration_profile(
            current_vel, current_acc, "linear", dt
        )

        # Check shape
        assert profile.shape == (batch_size, generator.action_horizon, action_dim)

        # Acceleration should oppose velocity (negative when velocity is positive)
        for b in range(batch_size):
            for t in range(generator.action_horizon):
                for d in range(action_dim):
                    if current_vel[b, d] > 1e-6:
                        # Acceleration should be negative to oppose positive velocity
                        assert profile[b, t, d] <= 0.0

    def test_deceleration_acceleration_profile_exponential(
        self, generator, device_cfg, action_dim, batch_size
    ):
        """Test acceleration profile generation with exponential deceleration."""
        current_vel = torch.ones((batch_size, action_dim), device=device_cfg.device)
        current_acc = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        dt = 0.1

        profile = generator._generate_deceleration_acceleration_profile(
            current_vel, current_acc, "exponential", dt
        )

        # Check shape
        assert profile.shape == (batch_size, generator.action_horizon, action_dim)

    def test_deceleration_acceleration_profile_smooth(
        self, generator, device_cfg, action_dim, batch_size
    ):
        """Test acceleration profile generation with smooth deceleration."""
        current_vel = torch.ones((batch_size, action_dim), device=device_cfg.device)
        current_acc = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        dt = 0.1

        profile = generator._generate_deceleration_acceleration_profile(
            current_vel, current_acc, "smooth", dt
        )

        # Check shape
        assert profile.shape == (batch_size, generator.action_horizon, action_dim)

    def test_deceleration_acceleration_profile_unknown(
        self, generator, device_cfg, action_dim, batch_size
    ):
        """Test acceleration profile with unknown deceleration type (should default)."""
        current_vel = torch.ones((batch_size, action_dim), device=device_cfg.device)
        current_acc = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        dt = 0.1

        profile = generator._generate_deceleration_acceleration_profile(
            current_vel, current_acc, "unknown_type", dt
        )

        # Check shape - should still work with default
        assert profile.shape == (batch_size, generator.action_horizon, action_dim)

    def test_deceleration_acceleration_profile_zero_velocity(
        self, generator, device_cfg, action_dim, batch_size
    ):
        """Test acceleration profile when velocity is zero."""
        current_vel = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        current_acc = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        dt = 0.1

        profile = generator._generate_deceleration_acceleration_profile(
            current_vel, current_acc, "linear", dt
        )

        # Check shape
        assert profile.shape == (batch_size, generator.action_horizon, action_dim)

        # Acceleration should be zero when velocity is zero
        torch.testing.assert_close(
            profile, torch.zeros_like(profile), rtol=1e-5, atol=1e-5
        )

    def test_deceleration_acceleration_profile_with_existing_decel(
        self, generator, device_cfg, action_dim, batch_size
    ):
        """Test when current acceleration already opposes velocity."""
        current_vel = torch.ones((batch_size, action_dim), device=device_cfg.device)
        # Current acceleration already opposes velocity
        current_acc = -torch.ones((batch_size, action_dim), device=device_cfg.device) * 5.0
        dt = 0.1

        profile = generator._generate_deceleration_acceleration_profile(
            current_vel, current_acc, "linear", dt
        )

        # Check shape
        assert profile.shape == (batch_size, generator.action_horizon, action_dim)


class TestTrajectoryIntegration:
    def test_integrate_acceleration_to_trajectory(
        self, generator, device_cfg, action_dim, batch_size
    ):
        """Test double integration from acceleration to position."""
        current_pos = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        current_vel = torch.ones((batch_size, action_dim), device=device_cfg.device)
        # Constant deceleration
        acceleration_profile = (
            -torch.ones((batch_size, generator.action_horizon, action_dim), device=device_cfg.device)
            * 0.5
        )
        dt = 0.1

        trajectory = generator._integrate_acceleration_to_trajectory(
            current_pos, current_vel, acceleration_profile, dt
        )

        # Check shape
        assert trajectory.shape == (batch_size, generator.action_horizon, action_dim)

        # First position should be current position
        torch.testing.assert_close(trajectory[:, 0, :], current_pos, rtol=1e-5, atol=1e-5)

        # Trajectory should evolve according to physics
        # With constant deceleration, position should increase but at decreasing rate
        for b in range(batch_size):
            for d in range(action_dim):
                # Position should generally increase (positive initial velocity)
                # but rate of increase should slow down
                if generator.action_horizon > 2:
                    diff1 = trajectory[b, 1, d] - trajectory[b, 0, d]
                    diff2 = trajectory[b, 2, d] - trajectory[b, 1, d]
                    # Later differences should be smaller due to deceleration
                    assert diff2 <= diff1 + 1e-4

    def test_integrate_acceleration_zero_velocity(
        self, generator, device_cfg, action_dim, batch_size
    ):
        """Test integration when starting with zero velocity."""
        current_pos = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        current_vel = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        acceleration_profile = torch.zeros(
            (batch_size, generator.action_horizon, action_dim), device=device_cfg.device
        )
        dt = 0.1

        trajectory = generator._integrate_acceleration_to_trajectory(
            current_pos, current_vel, acceleration_profile, dt
        )

        # Check shape
        assert trajectory.shape == (batch_size, generator.action_horizon, action_dim)

        # Position should remain at current position when no velocity or acceleration
        for t in range(generator.action_horizon):
            torch.testing.assert_close(
                trajectory[:, t, :], current_pos, rtol=1e-5, atol=1e-5
            )

    def test_integrate_acceleration_velocity_reversal(
        self, generator, device_cfg, action_dim, batch_size
    ):
        """Test that velocity reversal is prevented."""
        current_pos = torch.zeros((batch_size, action_dim), device=device_cfg.device)
        current_vel = torch.ones((batch_size, action_dim), device=device_cfg.device) * 0.1
        # Strong deceleration that would cause reversal
        acceleration_profile = (
            -torch.ones((batch_size, generator.action_horizon, action_dim), device=device_cfg.device)
            * 10.0
        )
        dt = 0.1

        trajectory = generator._integrate_acceleration_to_trajectory(
            current_pos, current_vel, acceleration_profile, dt
        )

        # Check shape
        assert trajectory.shape == (batch_size, generator.action_horizon, action_dim)

        # Velocity should be clamped to zero, not reverse
        # This is implicit in the position trajectory - it should stop increasing
        for b in range(batch_size):
            for d in range(action_dim):
                # Find where motion stops
                for t in range(1, generator.action_horizon):
                    pos_diff = trajectory[b, t, d] - trajectory[b, t - 1, d]
                    # Once stopped, should not go backwards
                    if abs(pos_diff.item()) < 1e-5 and t > 1:
                        # Check subsequent positions don't decrease
                        if t + 1 < generator.action_horizon:
                            next_diff = trajectory[b, t + 1, d] - trajectory[b, t, d]
                            assert next_diff.item() >= -1e-5  # Allow small numerical errors
