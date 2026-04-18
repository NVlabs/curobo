# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# CuRobo
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.trajectory_execution_manager import TrajectoryExecutionManager


@pytest.fixture
def device_cfg():
    return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)


@pytest.fixture
def action_dim():
    return 7  # Typical dimension for a 7-DOF robot arm


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def interpolation_steps():
    return 10


@pytest.fixture
def state_trajectory(device_cfg, batch_size, action_dim, interpolation_steps):
    """Create a sample state trajectory for testing."""
    # Generate a simple linear trajectory for testing
    positions = torch.zeros(
        (batch_size, interpolation_steps, action_dim),
        device=device_cfg.device,
        dtype=device_cfg.dtype,
    )

    # Make each step increment by a fixed amount to simulate a trajectory
    for i in range(interpolation_steps):
        positions[:, i, :] = torch.ones_like(positions[:, i, :]) * (i * 0.1)

    velocities = torch.zeros_like(positions)
    accelerations = torch.zeros_like(positions)

    return JointState(positions, velocities, accelerations)


@pytest.fixture
def action_trajectory(device_cfg, batch_size, action_dim):
    """Create a sample action trajectory for testing."""
    # Create action trajectory with one less step than state trajectory
    actions = torch.zeros(
        (batch_size, 1, action_dim), device=device_cfg.device, dtype=device_cfg.dtype
    )
    # Fill with some test values
    actions[:, 0, :] = torch.ones_like(actions[:, 0, :]) * 0.5

    return actions


@pytest.fixture
def action_trajectory_two_steps(device_cfg, batch_size, action_dim):
    """Create a sample action trajectory with two steps for testing."""
    # Create action trajectory with two steps
    actions = torch.zeros(
        (batch_size, 2, action_dim), device=device_cfg.device, dtype=device_cfg.dtype
    )
    # Fill with some test values
    actions[:, 0, :] = torch.ones_like(actions[:, 0, :]) * 0.5
    actions[:, 1, :] = torch.ones_like(actions[:, 1, :]) * 0.7

    return actions


@pytest.fixture
def large_action_trajectory(device_cfg, batch_size, action_dim, interpolation_steps):
    """Create a larger action trajectory for testing."""
    # Create action trajectory with multiple steps
    num_actions = interpolation_steps // 2
    actions = torch.zeros(
        (batch_size, num_actions, action_dim), device=device_cfg.device, dtype=device_cfg.dtype
    )
    # Fill with some test values
    for i in range(num_actions):
        actions[:, i, :] = torch.ones_like(actions[:, i, :]) * (i * 0.2 + 0.5)

    return actions


@pytest.fixture
def execution_manager(interpolation_steps):
    """Create a TrajectoryExecutionManager instance."""
    return TrajectoryExecutionManager(interpolation_steps)


class TestTrajectoryExecutionManager:
    def test_initialization(self, execution_manager, interpolation_steps):
        """Test that the TrajectoryExecutionManager initializes properly."""
        assert execution_manager.interpolation_steps == interpolation_steps
        assert execution_manager._current_joint_state_trajectory is None
        assert execution_manager._current_action_trajectory is None
        assert execution_manager._current_command_idx == 0

    def test_update_state_action_buffers(
        self, execution_manager, state_trajectory, action_trajectory
    ):
        """Test updating state and action buffers."""
        execution_manager.update_state_action_buffers(state_trajectory, action_trajectory)

        # Verify the update
        assert execution_manager._current_joint_state_trajectory is not None
        assert execution_manager._current_action_trajectory is not None
        assert execution_manager._current_command_idx == 0

        # Check that trajectories were copied correctly
        assert torch.allclose(
            execution_manager._current_joint_state_trajectory.position, state_trajectory.position
        )
        assert torch.allclose(execution_manager._current_action_trajectory, action_trajectory)

    def test_get_next_command(self, execution_manager, state_trajectory, action_trajectory):
        """Test getting the next command from the trajectory."""
        metrics = RolloutMetrics()
        execution_manager.update_state_action_metrics_buffers(
            state_trajectory, action_trajectory, metrics
        )

        # Get the first command
        next_command = execution_manager.get_next_command()

        # Verify the command matches the first step in the trajectory
        assert torch.allclose(next_command.position, state_trajectory.position[:, 0, :])
        assert execution_manager._current_command_idx == 1  # Command index should be incremented

        # Get the second command
        next_command = execution_manager.get_next_command()

        # Verify the command matches the second step
        assert torch.allclose(next_command.position, state_trajectory.position[:, 1, :])
        assert execution_manager._current_command_idx == 2

    def test_has_valid_next_command(
        self, execution_manager, state_trajectory, action_trajectory, interpolation_steps
    ):
        """Test checking if the next command is valid."""
        # Before updating buffers, should return False
        assert not execution_manager.has_valid_next_command()

        execution_manager.update_state_action_buffers(state_trajectory, action_trajectory)

        # After updating buffers, should return True
        assert execution_manager.has_valid_next_command()

        # Execute all commands
        for _ in range(interpolation_steps):
            execution_manager.get_next_command()

        # After exhausting all commands, should return False
        assert not execution_manager.has_valid_next_command()

    def test_has_valiaction_dim_buffer(self, execution_manager, state_trajectory, action_trajectory):
        """Test checking if the action buffer is valid."""
        # Before updating buffers, should return False
        assert not execution_manager.has_valiaction_dim_buffer()

        execution_manager.update_state_action_buffers(state_trajectory, action_trajectory)

        # After updating buffers, should return True
        assert execution_manager.has_valiaction_dim_buffer()

    def test_get_action_buffer(self, execution_manager, state_trajectory, action_trajectory):
        """Test getting the action buffer with a single action."""
        execution_manager.update_state_action_buffers(state_trajectory, action_trajectory)

        # Get the action buffer
        action_buffer = execution_manager.get_action_buffer()

        # Verify the action buffer shape is preserved
        assert action_buffer.shape == action_trajectory.shape

        # For single action trajectories, the original action should be preserved
        # since rolling doesn't change anything with a single element
        assert torch.allclose(action_buffer, action_trajectory)

    def test_get_action_buffer_two_steps(
        self, execution_manager, state_trajectory, action_trajectory_two_steps
    ):
        """Test getting the action buffer with two actions."""
        execution_manager.update_state_action_buffers(state_trajectory, action_trajectory_two_steps)

        # Get the action buffer
        action_buffer = execution_manager.get_shifteaction_dim_buffer()

        # Verify the action buffer has the correct shape
        assert action_buffer.shape == action_trajectory_two_steps.shape

        # The action should be rolled with the last action repeated
        assert torch.allclose(action_buffer[..., 0, :], action_trajectory_two_steps[..., 1, :])
        assert torch.allclose(action_buffer[..., 1, :], action_trajectory_two_steps[..., 1, :])

    def test_update_multiple_times(self, execution_manager, state_trajectory, action_trajectory):
        """Test updating the buffers multiple times."""
        # First update
        execution_manager.update_state_action_buffers(state_trajectory, action_trajectory)
        assert execution_manager._current_command_idx == 0

        # Execute a few commands
        execution_manager.get_next_command()
        execution_manager.get_next_command()
        assert execution_manager._current_command_idx == 2

        # Update again with the same trajectory
        execution_manager.update_state_action_buffers(state_trajectory, action_trajectory)

        # Command index should be reset
        assert execution_manager._current_command_idx == 0

        # Can get next command again from the beginning
        next_command = execution_manager.get_next_command()
        assert torch.allclose(next_command.position, state_trajectory.position[:, 0, :])

    def test_get_action_buffer_with_larger_action_trajectory(
        self, execution_manager, state_trajectory, large_action_trajectory
    ):
        """Test getting the action buffer with a larger action trajectory."""
        execution_manager.update_state_action_buffers(state_trajectory, large_action_trajectory)

        # Execute a few commands to advance the action index
        for _ in range(execution_manager.interpolation_steps // 2):
            execution_manager.get_next_command()

        # Get the action buffer
        action_buffer = execution_manager.get_shifteaction_dim_buffer()

        # Verify the action buffer has the correct shape
        assert action_buffer.shape == large_action_trajectory.shape

        # Verify the action buffer is correctly rolled
        # The first action should now be the second action in the original trajectory
        assert torch.allclose(action_buffer[..., 0, :], large_action_trajectory[..., 1, :])

    def test_get_next_command_error(self, execution_manager):
        """Test error case when getting next command without a valid buffer."""
        with pytest.raises(Exception):
            execution_manager.get_next_command()

    def test_get_action_buffer_error(self, execution_manager):
        """Test error case when getting action buffer without a valid buffer."""
        with pytest.raises(Exception):
            execution_manager.get_action_buffer()

    def test_get_shifteaction_dim_buffer_error(self, execution_manager):
        """Test error case when getting shifted action buffer without a valid buffer."""
        with pytest.raises(Exception):
            execution_manager.get_shifteaction_dim_buffer()

    def test_get_current_metrics(self, execution_manager, state_trajectory, action_trajectory):
        """Test getting current metrics."""
        metrics = RolloutMetrics()
        execution_manager.update_state_action_metrics_buffers(
            state_trajectory, action_trajectory, metrics
        )
        result = execution_manager.get_current_metrics()
        assert result == metrics

    def test_update_robot_state_trajectory(self, execution_manager):
        """Test updating robot state trajectory."""
        # CuRobo
        from curobo._src.state.state_robot import RobotState
        from curobo._src.types.device_cfg import DeviceCfg

        robot_state = RobotState(JointState.zeros((7,), DeviceCfg()))
        execution_manager.update_robot_state_trajectory(robot_state)
        assert execution_manager._current_robot_state_trajectory is not None

    def test_get_command_sequence(self, execution_manager, state_trajectory, action_trajectory):
        """Test getting command sequence."""
        execution_manager.update_state_action_buffers(state_trajectory, action_trajectory)
        sequence = execution_manager.get_command_sequence()
        assert sequence is not None
        # Should trim the trajectory - exact shape depends on interpolation_steps
        assert sequence.position.shape[1] <= execution_manager.interpolation_steps * 2

    def test_get_robot_state_sequence(self, execution_manager):
        """Test getting robot state sequence."""
        # CuRobo
        from curobo._src.state.state_robot import RobotState
        from curobo._src.types.device_cfg import DeviceCfg

        robot_state = RobotState(JointState.zeros((7,), DeviceCfg()))
        execution_manager.update_robot_state_trajectory(robot_state)
        result = execution_manager.get_robot_state_sequence()
        assert result is not None

    def test_get_robot_state_sequence_error(self, execution_manager):
        """Test error when getting robot state sequence without updating."""
        with pytest.raises(Exception):
            execution_manager.get_robot_state_sequence()

    def test_has_valid_robot_state_trajectory(self, execution_manager):
        """Test checking if robot state trajectory is valid."""
        # CuRobo
        from curobo._src.state.state_robot import RobotState
        from curobo._src.types.device_cfg import DeviceCfg

        assert not execution_manager.has_valid_robot_state_trajectory()
        robot_state = RobotState(JointState.zeros((7,), DeviceCfg()))
        execution_manager.update_robot_state_trajectory(robot_state)
        assert execution_manager.has_valid_robot_state_trajectory()

    def test_get_shifteaction_dim_buffer_out_of_bounds(
        self, execution_manager, state_trajectory, action_trajectory
    ):
        """Test error when action index is out of bounds."""
        execution_manager.update_state_action_buffers(state_trajectory, action_trajectory)
        # Manually set command idx to trigger out of bounds
        execution_manager._current_command_idx = 1000
        with pytest.raises(Exception):
            execution_manager.get_shifteaction_dim_buffer()

    def test_update_action_buffer(self, execution_manager, action_trajectory):
        """Test updating action buffer directly."""
        execution_manager.update_action_buffer(action_trajectory)
        assert execution_manager._current_action_trajectory is not None
        assert execution_manager._current_command_idx == 0
        assert execution_manager._current_joint_state_trajectory is None
