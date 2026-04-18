# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for MPCSolver class."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.solver_mpc import MPCSolver
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
from curobo._src.solver.solver_mpc_result import MPCSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose


@pytest.fixture(scope="module")
def cuda_device_cfg():
    """Create a CUDA device configuration for testing."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def franka_mpc_config(cuda_device_cfg):
    """Create MPCSolverCfg for Franka robot."""
    config = MPCSolverCfg.create(
        robot="franka.yml",
        device_cfg=cuda_device_cfg,
        use_cuda_graph=False,
    )
    return config


@pytest.fixture(scope="module")
def mpc_solver(franka_mpc_config):
    """Create MPCSolver instance."""
    solver = MPCSolver(franka_mpc_config)
    return solver


@pytest.fixture
def sample_current_state(mpc_solver, cuda_device_cfg):
    """Create a sample current state for testing."""
    default_js = mpc_solver.default_joint_state.clone()
    joint_state = JointState.from_position(
        default_js.position.unsqueeze(0),
        joint_names=mpc_solver.joint_names,
    )
    joint_state.velocity = torch.zeros_like(joint_state.position)
    joint_state.acceleration = torch.zeros_like(joint_state.position)
    return joint_state


class TestMPCSolverCfgCreation:
    """Test MPCSolverCfg creation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_creates_config(self, cuda_device_cfg):
        """Test create creates a valid config."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config is not None
        assert isinstance(config, MPCSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_config_has_optimization_dt(self, franka_mpc_config):
        """Test config has optimization_dt."""
        assert franka_mpc_config.optimization_dt > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_config_has_interpolation_steps(self, franka_mpc_config):
        """Test config has interpolation_steps."""
        assert franka_mpc_config.interpolation_steps >= 1


class TestMPCSolverInitialization:
    """Test MPCSolver initialization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_with_config(self, franka_mpc_config):
        """Test MPCSolver initializes with config."""
        solver = MPCSolver(franka_mpc_config)
        assert solver is not None
        assert solver.config == franka_mpc_config

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_creates_components(self, franka_mpc_config):
        """Test MPCSolver creates necessary components."""
        solver = MPCSolver(franka_mpc_config)

        assert solver.optimizer is not None
        assert solver.metrics_rollout is not None
        assert solver.trajectory_execution_manager is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_mpc_state_flags(self, mpc_solver):
        """Test MPC state flags are initialized."""
        assert mpc_solver._mpc_setup_complete is False
        assert mpc_solver._mpc_initialized is False
        assert mpc_solver._mpc_warm_start_available is False


class TestMPCSolverProperties:
    """Test MPCSolver properties."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_action_dim_property(self, mpc_solver):
        """Test action_dim property returns DOF."""
        assert mpc_solver.action_dim > 0
        # Franka has 7 DOFs
        assert mpc_solver.action_dim == 7

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_action_horizon_property(self, mpc_solver):
        """Test action_horizon property."""
        assert mpc_solver.action_horizon >= 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_kinematics_property(self, mpc_solver):
        """Test kinematics property exists and works."""
        assert mpc_solver.kinematics is not None
        # Verify kinematics can compute forward kinematics
        default_js = mpc_solver.default_joint_state.clone()
        joint_state = JointState.from_position(
            default_js.position.unsqueeze(0),
            joint_names=mpc_solver.joint_names,
        )
        kin_result = mpc_solver.compute_kinematics(joint_state)
        assert kin_result is not None
        assert kin_result.tool_poses is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_joint_position_property(self, mpc_solver):
        """Test default_joint_position property."""
        default_position = mpc_solver.default_joint_position
        assert isinstance(default_position, torch.Tensor)
        assert default_position.shape[-1] == mpc_solver.action_dim

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_joint_state_property(self, mpc_solver):
        """Test default_joint_state property returns JointState."""
        default_js = mpc_solver.default_joint_state
        assert isinstance(default_js, JointState)
        assert default_js.position is not None
        assert default_js.position.shape[-1] == mpc_solver.action_dim

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_joint_names_property(self, mpc_solver):
        """Test joint_names property."""
        names = mpc_solver.joint_names
        assert isinstance(names, list)
        assert len(names) == mpc_solver.action_dim
        # Verify all names are strings
        assert all(isinstance(name, str) for name in names)


class TestMPCSolverSetup:
    """Test MPCSolver setup methods."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_setup(self, franka_mpc_config):
        """Test setup configures solver correctly."""
        solver = MPCSolver(franka_mpc_config)
        default_js = solver.default_joint_state.clone()
        current_state = JointState.from_position(
            default_js.position.unsqueeze(0),
            joint_names=solver.joint_names,
        )
        current_state.velocity = torch.zeros_like(current_state.position)
        current_state.acceleration = torch.zeros_like(current_state.position)

        solver.setup(current_state)

        assert solver._mpc_setup_complete is True
        assert solver.problem_batch_size == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_setup_batch(self, cuda_device_cfg):
        """Test setup configures solver for batch."""
        batch_size = 2
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
            max_batch_size=batch_size,
        )
        solver = MPCSolver(config)

        default_js = solver.default_joint_state.clone()
        default_position = default_js.position.view(1, -1).repeat(batch_size, 1)
        default_velocity = torch.zeros_like(default_position)
        default_acceleration = torch.zeros_like(default_position)

        current_state = JointState(
            position=default_position,
            velocity=default_velocity,
            acceleration=default_acceleration,
            jerk=torch.zeros_like(default_position),
            joint_names=solver.joint_names,
        )

        solver.setup(current_state)

        assert solver._mpc_setup_complete is True
        assert solver.problem_batch_size == batch_size


class TestMPCSolverOptimization:
    """Test MPCSolver optimization methods."""

    @pytest.fixture
    def setup_solver(self, franka_mpc_config):
        """Create and setup solver for optimization tests."""
        solver = MPCSolver(franka_mpc_config)
        default_js = solver.default_joint_state.clone()
        current_state = JointState.from_position(
            default_js.position.unsqueeze(0),
            joint_names=solver.joint_names,
        )
        current_state.velocity = torch.zeros_like(current_state.position)
        current_state.acceleration = torch.zeros_like(current_state.position)
        solver.setup(current_state)
        return solver, current_state

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_optimize_action_sequence_returns_result(self, setup_solver):
        """Test optimize_action_sequence returns valid result."""
        solver, current_state = setup_solver

        result = solver.optimize_action_sequence(current_state)

        assert result is not None
        assert isinstance(result, MPCSolverResult)
        assert result.action_sequence is not None
        assert result.solve_time >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_optimize_action_sequence_returns_valiaction_dims(self, setup_solver):
        """Test optimize_action_sequence returns valid action shapes."""
        solver, current_state = setup_solver

        result = solver.optimize_action_sequence(current_state)

        assert result.action_sequence is not None
        # Action sequence should be a JointState
        assert isinstance(result.action_sequence, JointState)
        # Check position shape: [batch, horizon, dof]
        assert result.action_sequence.position.shape[0] == 1  # batch size
        assert result.action_sequence.position.shape[-1] == solver.action_dim

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_optimize_next_action_returns_result(self, setup_solver):
        """Test optimize_next_action returns valid result."""
        solver, current_state = setup_solver

        result = solver.optimize_next_action(current_state)

        assert result is not None
        assert isinstance(result, MPCSolverResult)
        assert result.next_action is not None
        assert result.solve_time >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_optimize_next_action_returns_single_action(self, setup_solver):
        """Test optimize_next_action returns a single action."""
        solver, current_state = setup_solver

        result = solver.optimize_next_action(current_state)

        assert result.next_action is not None
        assert isinstance(result.next_action, JointState)
        assert result.next_action.position.shape[-1] == solver.action_dim


class TestMPCSolverGoalUpdates:
    """Test MPCSolver goal update methods."""

    @pytest.fixture
    def setup_solver(self, franka_mpc_config):
        """Create and setup solver for goal update tests."""
        solver = MPCSolver(franka_mpc_config)
        default_js = solver.default_joint_state.clone()
        current_state = JointState.from_position(
            default_js.position.unsqueeze(0),
            joint_names=solver.joint_names,
        )
        current_state.velocity = torch.zeros_like(current_state.position)
        current_state.acceleration = torch.zeros_like(current_state.position)
        solver.setup(current_state)
        return solver, current_state

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_goal_tool_poses_from_kin_result(self, setup_solver):
        """Test update_goal_tool_poses from kinematics tool poses."""
        solver, current_state = setup_solver

        # Get current link poses and modify
        kin_result = solver.compute_kinematics(current_state)
        goal_tool_poses = GoalToolPose.from_poses(
            kin_result.tool_poses.to_dict(),
            ordered_tool_frames=solver.tool_frames,
        )

        # Should not raise
        solver.update_goal_tool_poses(goal_tool_poses)

        # Verify we can still optimize
        result = solver.optimize_action_sequence(current_state)
        assert result is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_goal_tool_poses_with_modified_pose(self, setup_solver):
        """Test update_goal_tool_poses with modified target pose."""
        solver, current_state = setup_solver

        # Get current link poses
        kin_result = solver.compute_kinematics(current_state)
        pose_dict = kin_result.tool_poses.to_dict()

        # Modify position of first target link
        target_link = solver.tool_frames[0]
        if target_link in pose_dict:
            # Add small offset to position
            pose_dict[target_link].position = (
                pose_dict[target_link].position + 0.05
            )

        goal_tool_poses = GoalToolPose.from_poses(
            pose_dict, ordered_tool_frames=solver.tool_frames
        )
        solver.update_goal_tool_poses(goal_tool_poses)

        result = solver.optimize_action_sequence(current_state)
        assert result is not None
        # position_error may be None if goal tracking is disabled, just verify result exists
        assert result.success is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_goal_state(self, setup_solver):
        """Test update_goal_state updates the goal state."""
        solver, current_state = setup_solver

        # Create a slightly different goal state
        goal_state = current_state.clone()
        goal_state.position = goal_state.position + 0.1

        solver.update_goal_state(goal_state)

        result = solver.optimize_action_sequence(current_state)
        assert result is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_current_state(self, setup_solver):
        """Test update_current_state updates the current state."""
        solver, current_state = setup_solver

        # Create modified current state
        new_current_state = current_state.clone()
        new_current_state.position = new_current_state.position + 0.05

        solver.update_current_state(new_current_state)

        # Should still be able to optimize
        result = solver.optimize_action_sequence(new_current_state)
        assert result is not None


class TestMPCSolverResetSeed:
    """Test MPCSolver.reset_seed method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_seed_no_error(self, mpc_solver):
        """Test reset_seed doesn't raise error."""
        mpc_solver.reset_seed()


class TestMPCSolverComputeKinematics:
    """Test MPCSolver.compute_kinematics method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compute_kinematics_returns_result(self, mpc_solver, sample_current_state):
        """Test compute_kinematics returns valid result."""
        result = mpc_solver.compute_kinematics(sample_current_state)

        assert result is not None
        assert hasattr(result, 'tool_poses')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compute_kinematics_link_poses_are_valid(self, mpc_solver, sample_current_state):
        """Test compute_kinematics returns valid link poses."""
        result = mpc_solver.compute_kinematics(sample_current_state)

        link_poses_dict = result.tool_poses.to_dict()
        assert len(link_poses_dict) > 0

        # Check that target links are in the result
        for link_name in mpc_solver.tool_frames:
            assert link_name in link_poses_dict
            pose = link_poses_dict[link_name]
            assert isinstance(pose, Pose)
            assert pose.position is not None
            assert pose.quaternion is not None


class TestMPCSolverResult:
    """Test MPCSolverResult dataclass."""

    def test_result_has_next_action(self):
        """Test MPCSolverResult has next_action field."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert hasattr(result, 'next_action')

    def test_result_has_action_sequence(self):
        """Test MPCSolverResult has action_sequence field."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert hasattr(result, 'action_sequence')

    def test_result_has_full_action_sequence(self):
        """Test MPCSolverResult has full_action_sequence field."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert hasattr(result, 'full_action_sequence')

    def test_result_has_robot_state_sequence(self):
        """Test MPCSolverResult has robot_state_sequence field."""
        result = MPCSolverResult(success=torch.tensor([True]))
        assert hasattr(result, 'robot_state_sequence')

    def test_result_clone(self):
        """Test MPCSolverResult clone method."""
        result = MPCSolverResult(
            success=torch.tensor([True, False]),
            position_error=torch.tensor([0.01, 0.02]),
        )
        cloned = result.clone()
        assert cloned is not result
        assert torch.equal(cloned.success, result.success)

    def test_result_clone_preserves_all_fields(self):
        """Test MPCSolverResult clone preserves all fields."""
        result = MPCSolverResult(
            success=torch.tensor([True]),
            position_error=torch.tensor([0.01]),
            rotation_error=torch.tensor([0.02]),
            solve_time=0.015,
        )
        cloned = result.clone()

        assert torch.equal(cloned.success, result.success)
        assert torch.equal(cloned.position_error, result.position_error)
        assert torch.equal(cloned.rotation_error, result.rotation_error)
        assert cloned.solve_time == result.solve_time


class TestMPCSolverTrajectoryExecutionManager:
    """Test MPCSolver trajectory execution manager."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_trajectory_execution_manager(self, mpc_solver):
        """Test solver has trajectory_execution_manager."""
        assert mpc_solver.trajectory_execution_manager is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_trajectory_execution_manager_interpolation_steps(self, mpc_solver):
        """Test trajectory_execution_manager has correct interpolation steps."""
        assert mpc_solver.trajectory_execution_manager.interpolation_steps == (
            mpc_solver.config.interpolation_steps
        )


class TestMPCSolverTargetLinkNames:
    """Test MPCSolver target link names."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_tool_frames(self, mpc_solver):
        """Test solver has tool_frames property."""
        target_links = mpc_solver.tool_frames
        assert isinstance(target_links, list)
        assert len(target_links) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tool_frames_available(self, mpc_solver):
        """Test kinematics tool_frames available."""
        tool_frames = mpc_solver.kinematics.tool_frames

        assert isinstance(tool_frames, list)
        assert len(tool_frames) > 0


class TestMPCSolverResetRobot:
    """Test MPCSolver reset methods."""

    @pytest.fixture
    def setup_solver(self, franka_mpc_config):
        """Create and setup solver for reset tests."""
        solver = MPCSolver(franka_mpc_config)
        default_js = solver.default_joint_state.clone()
        current_state = JointState.from_position(
            default_js.position.unsqueeze(0),
            joint_names=solver.joint_names,
        )
        current_state.velocity = torch.zeros_like(current_state.position)
        current_state.acceleration = torch.zeros_like(current_state.position)
        solver.setup(current_state)
        return solver, current_state

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_robot(self, setup_solver):
        """Test reset_robot resets solver state."""
        solver, current_state = setup_solver

        # Run optimization to change state
        solver.optimize_action_sequence(current_state)

        # Reset robot
        solver.reset_robot(current_state)

        # Verify warm start is reset
        assert solver._mpc_warm_start_available is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_robot_can_optimize_after(self, setup_solver):
        """Test optimization works after reset_robot."""
        solver, current_state = setup_solver

        solver.reset_robot(current_state)

        # Should be able to optimize again
        result = solver.optimize_action_sequence(current_state)
        assert result is not None
        assert result.action_sequence is not None


class TestMPCSolverWarmAndColdStart:
    """Test MPCSolver warm and cold start optimization."""

    @pytest.fixture
    def setup_solver(self, franka_mpc_config):
        """Create and setup solver for warm/cold start tests."""
        solver = MPCSolver(franka_mpc_config)
        default_js = solver.default_joint_state.clone()
        current_state = JointState.from_position(
            default_js.position.unsqueeze(0),
            joint_names=solver.joint_names,
        )
        current_state.velocity = torch.zeros_like(current_state.position)
        current_state.acceleration = torch.zeros_like(current_state.position)
        solver.setup(current_state)
        return solver, current_state

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cold_start_solve(self, setup_solver):
        """Test cold_start_solve runs without error."""
        solver, current_state = setup_solver

        metrics = solver.cold_start_solve(current_state)
        assert metrics is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_warm_start_solve(self, setup_solver):
        """Test warm_start_solve runs without error."""
        solver, current_state = setup_solver

        # Need to run cold start first
        solver.cold_start_solve(current_state)

        metrics = solver.warm_start_solve(current_state)
        assert metrics is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_warm_start_available_after_optimization(self, setup_solver):
        """Test warm start becomes available after optimization."""
        solver, current_state = setup_solver

        assert solver._mpc_warm_start_available is False

        solver.optimize_action_sequence(current_state)

        assert solver._mpc_warm_start_available is True


class TestMPCSolverBatchOptimization:
    """Test MPCSolver batch optimization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_batch_optimization_methods_exist(self, mpc_solver):
        """Test batch optimization related methods exist."""
        assert hasattr(mpc_solver, "setup")
        assert hasattr(mpc_solver, "optimize_action_sequence")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_batch_optimization(self, cuda_device_cfg):
        """Test batch optimization returns correct batch size."""
        batch_size = 2
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
            max_batch_size=batch_size,
        )
        solver = MPCSolver(config)

        default_js = solver.default_joint_state.clone()
        default_position = default_js.position.view(1, -1).repeat(batch_size, 1)
        default_velocity = torch.zeros_like(default_position)
        default_acceleration = torch.zeros_like(default_position)

        current_state = JointState(
            position=default_position,
            velocity=default_velocity,
            acceleration=default_acceleration,
            jerk=torch.zeros_like(default_position),
            joint_names=solver.joint_names,
        )

        solver.setup(current_state)

        result = solver.optimize_action_sequence(current_state)

        assert result is not None
        assert result.success.shape[0] == batch_size
        assert result.action_sequence is not None
        assert result.action_sequence.position.shape[0] == batch_size
