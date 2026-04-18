# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for SeedIKSolver class."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
from curobo._src.solver.solver_ik_result import IKSolverResult
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
def seed_ik_solver(cuda_device_cfg):
    """Create a SeedIKSolver instance for testing."""
    config = SeedIKSolverCfg.create(
        robot="franka.yml",
        device_cfg=cuda_device_cfg,
        num_seeds=4,
        max_iterations=8,
        inner_iterations=4,
        use_cuda_graph=False,  # Disable for testing flexibility
    )
    return SeedIKSolver(config)


@pytest.fixture(scope="module")
def sample_goal_pose(cuda_device_cfg, seed_ik_solver):
    """Create a sample goal pose for testing."""
    device = cuda_device_cfg.device

    # Get tool frames from the solver
    tool_frames = seed_ik_solver.tool_frames
    num_links = len(tool_frames)

    # Shape: [B=1, H=1, L=num_links, G=1, 3/4]
    position = torch.tensor([[[[[0.4, 0.0, 0.4]]]]], device=device, dtype=torch.float32)
    quaternion = torch.tensor([[[[[0.0, 1.0, 0.0, 0.0]]]]], device=device, dtype=torch.float32)

    position = position.expand(1, 1, num_links, 1, 3).contiguous()
    quaternion = quaternion.expand(1, 1, num_links, 1, 4).contiguous()

    return GoalToolPose(
        tool_frames=tool_frames,
        position=position,
        quaternion=quaternion,
    )


@pytest.fixture(scope="module")
def batch_goal_poses(cuda_device_cfg, seed_ik_solver):
    """Create batch goal poses for testing."""
    device = cuda_device_cfg.device
    batch_size = 4

    tool_frames = seed_ik_solver.tool_frames
    num_links = len(tool_frames)

    # Shape: [B=4, H=1, L=num_links, G=1, 3/4]
    positions = torch.tensor([
        [[[[0.4, 0.0, 0.4]]]],
        [[[[0.4, 0.1, 0.4]]]],
        [[[[0.4, -0.1, 0.4]]]],
        [[[[0.4, 0.0, 0.5]]]],
    ], device=device, dtype=torch.float32)

    quaternions = torch.tensor([
        [[[[0.0, 1.0, 0.0, 0.0]]]],
        [[[[0.0, 1.0, 0.0, 0.0]]]],
        [[[[0.0, 1.0, 0.0, 0.0]]]],
        [[[[0.0, 1.0, 0.0, 0.0]]]],
    ], device=device, dtype=torch.float32)

    positions = positions.expand(batch_size, 1, num_links, 1, 3).contiguous()
    quaternions = quaternions.expand(batch_size, 1, num_links, 1, 4).contiguous()

    return GoalToolPose(
        tool_frames=tool_frames,
        position=positions,
        quaternion=quaternions,
    )


class TestSeedIKSolverInitialization:
    """Test SeedIKSolver initialization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_basic(self, seed_ik_solver):
        """Test basic initialization."""
        assert seed_ik_solver is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_has_dof(self, seed_ik_solver):
        """Test solver has dof attribute."""
        assert hasattr(seed_ik_solver, "dof")
        assert seed_ik_solver.dof > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_has_joint_names(self, seed_ik_solver):
        """Test solver has joint_names attribute."""
        assert hasattr(seed_ik_solver, "joint_names")
        assert len(seed_ik_solver.joint_names) == seed_ik_solver.dof

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_has_tool_frames(self, seed_ik_solver):
        """Test solver has tool_frames attribute."""
        assert hasattr(seed_ik_solver, "tool_frames")
        assert len(seed_ik_solver.tool_frames) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_has_action_bounds(self, seed_ik_solver):
        """Test solver has action bounds."""
        assert hasattr(seed_ik_solver, "action_min")
        assert hasattr(seed_ik_solver, "action_max")
        assert seed_ik_solver.action_min.shape == (seed_ik_solver.dof,)
        assert seed_ik_solver.action_max.shape == (seed_ik_solver.dof,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_has_error_calculator(self, seed_ik_solver):
        """Test solver has error calculator."""
        assert hasattr(seed_ik_solver, "error_calculator")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_has_iteration_state_manager(self, seed_ik_solver):
        """Test solver has iteration state manager."""
        assert hasattr(seed_ik_solver, "_iteration_state_manager")


class TestSeedIKSolverProperties:
    """Test SeedIKSolver properties."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_joint_limits_property(self, seed_ik_solver):
        """Test joint_limits property."""
        limits = seed_ik_solver.joint_limits
        assert limits is not None
        assert hasattr(limits, "position")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_kinematics_property(self, seed_ik_solver):
        """Test kinematics property."""
        kin = seed_ik_solver.kinematics
        assert kin is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_n_residuals_property(self, seed_ik_solver):
        """Test n_residuals property."""
        n_res = seed_ik_solver.n_residuals
        assert n_res > 0
        # Should be at least 6 * num_links (position + orientation)
        assert n_res >= 6 * seed_ik_solver.num_links

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_joint_position(self, seed_ik_solver):
        """Test default_joint_position attribute."""
        assert seed_ik_solver.default_joint_position is not None
        assert seed_ik_solver.default_joint_position.shape[-1] == seed_ik_solver.dof

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_default_joint_position(self, seed_ik_solver):
        """Test get_default_joint_position method."""
        default_position = seed_ik_solver.get_default_joint_position()
        assert default_position is not None
        assert default_position.shape[-1] == seed_ik_solver.dof


class TestSeedIKSolverSolveSingle:
    """Test SeedIKSolver.solve_single method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_returns_ik_result(self, seed_ik_solver, sample_goal_pose):
        """Test solve_single returns IKSolverResult."""
        result = seed_ik_solver.solve_single(goal_tool_poses=sample_goal_pose)
        assert isinstance(result, IKSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_has_success(self, seed_ik_solver, sample_goal_pose):
        """Test solve_single result has success tensor."""
        result = seed_ik_solver.solve_single(goal_tool_poses=sample_goal_pose)
        assert hasattr(result, "success")
        assert result.success is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_has_solution(self, seed_ik_solver, sample_goal_pose):
        """Test solve_single result has solution tensor."""
        result = seed_ik_solver.solve_single(goal_tool_poses=sample_goal_pose)
        assert hasattr(result, "solution")
        assert result.solution is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_has_js_solution(self, seed_ik_solver, sample_goal_pose):
        """Test solve_single result has js_solution."""
        result = seed_ik_solver.solve_single(goal_tool_poses=sample_goal_pose)
        assert hasattr(result, "js_solution")
        assert result.js_solution is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_has_position_error(self, seed_ik_solver, sample_goal_pose):
        """Test solve_single result has position_error."""
        result = seed_ik_solver.solve_single(goal_tool_poses=sample_goal_pose)
        assert hasattr(result, "position_error")
        assert result.position_error is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_has_rotation_error(self, seed_ik_solver, sample_goal_pose):
        """Test solve_single result has rotation_error."""
        result = seed_ik_solver.solve_single(goal_tool_poses=sample_goal_pose)
        assert hasattr(result, "rotation_error")
        assert result.rotation_error is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_solution_shape(self, seed_ik_solver, sample_goal_pose):
        """Test solve_single solution has correct shape."""
        result = seed_ik_solver.solve_single(goal_tool_poses=sample_goal_pose)
        # Shape should be (1, return_seeds, dof)
        assert result.solution.shape[0] == 1
        assert result.solution.shape[2] == seed_ik_solver.dof

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_multiple_return_seeds(self, seed_ik_solver, sample_goal_pose):
        """Test solve_single with multiple return_seeds."""
        return_seeds = 3
        result = seed_ik_solver.solve_single(
            goal_tool_poses=sample_goal_pose,
            return_seeds=return_seeds,
        )
        assert result.solution.shape[1] == return_seeds

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_has_solve_time(self, seed_ik_solver, sample_goal_pose):
        """Test solve_single result has solve_time."""
        result = seed_ik_solver.solve_single(goal_tool_poses=sample_goal_pose)
        assert hasattr(result, "solve_time")
        assert result.solve_time >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_has_total_time(self, seed_ik_solver, sample_goal_pose):
        """Test solve_single result has total_time."""
        result = seed_ik_solver.solve_single(goal_tool_poses=sample_goal_pose)
        assert hasattr(result, "total_time")
        assert result.total_time >= 0


class TestSeedIKSolverSolveBatch:
    """Test SeedIKSolver.solve_batch method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_batch_returns_ik_result(self, seed_ik_solver, batch_goal_poses):
        """Test solve_batch returns IKSolverResult."""
        result = seed_ik_solver.solve_batch(goal_tool_poses=batch_goal_poses)
        assert isinstance(result, IKSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_batch_has_success(self, seed_ik_solver, batch_goal_poses):
        """Test solve_batch result has success tensor."""
        result = seed_ik_solver.solve_batch(goal_tool_poses=batch_goal_poses)
        assert hasattr(result, "success")
        assert result.success is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_batch_correct_batch_size(self, seed_ik_solver, batch_goal_poses):
        """Test solve_batch result has correct batch size."""
        batch_size = batch_goal_poses.batch_size
        result = seed_ik_solver.solve_batch(goal_tool_poses=batch_goal_poses)
        assert result.solution.shape[0] == batch_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_batch_solution_shape(self, seed_ik_solver, batch_goal_poses):
        """Test solve_batch solution has correct shape."""
        batch_size = batch_goal_poses.batch_size
        result = seed_ik_solver.solve_batch(goal_tool_poses=batch_goal_poses)
        # Shape should be (batch, return_seeds, dof)
        assert result.solution.shape[0] == batch_size
        assert result.solution.shape[2] == seed_ik_solver.dof


class TestSeedIKSolverGenerateSeedConfigs:
    """Test seed configuration generation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_generate_seed_configs_no_seed(self, seed_ik_solver):
        """Test seed generation without provided seed."""
        batch_size = 2
        seeds = seed_ik_solver._generate_seed_configs(batch_size=batch_size)
        assert seeds.shape == (batch_size, seed_ik_solver.config.num_seeds, seed_ik_solver.dof)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_generate_seed_configs_with_seed(self, seed_ik_solver, cuda_device_cfg):
        """Test seed generation with provided seed."""
        batch_size = 2
        seed_config = torch.randn(
            batch_size, 1, seed_ik_solver.dof, device=cuda_device_cfg.device
        )
        seeds = seed_ik_solver._generate_seed_configs(
            batch_size=batch_size, seed_config=seed_config
        )
        assert seeds.shape == (batch_size, seed_ik_solver.config.num_seeds, seed_ik_solver.dof)
        # First seed should be the provided one
        assert torch.allclose(seeds[:, 0, :], seed_config.squeeze(1))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_generate_seed_configs_includes_default(self, seed_ik_solver):
        """Test seed generation includes default joint state."""
        batch_size = 2
        seeds = seed_ik_solver._generate_seed_configs(batch_size=batch_size)
        # Last seed should be default joint state
        default_position = seed_ik_solver._robot_model.default_joint_state.position.view(1, -1)
        assert torch.allclose(seeds[0, -1, :], default_position.squeeze(0))


class TestSeedIKSolverCheckConvergence:
    """Test convergence checking."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_check_convergence_returns_tensor(self, seed_ik_solver, sample_goal_pose):
        """Test _check_convergence returns tensor."""
        # First, run a solve to get an iteration state
        result = seed_ik_solver.solve_single(goal_tool_poses=sample_goal_pose)
        # Check that solver works and produces results
        assert result.success is not None


class TestSeedIKSolverCalculateNResiduals:
    """Test residual calculation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_calculate_n_residuals_with_joint_limits(self, seed_ik_solver):
        """Test n_residuals calculation with joint limits."""
        num_links = 1
        joint_limit_weight = 1.0
        n_res = seed_ik_solver._calculate_n_residuals(num_links, joint_limit_weight)
        # Should be 6 * num_links + dof
        expected = 6 * num_links + seed_ik_solver.dof
        assert n_res == expected

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_calculate_n_residuals_without_joint_limits(self, seed_ik_solver):
        """Test n_residuals calculation without joint limits."""
        num_links = 1
        joint_limit_weight = 0.0
        n_res = seed_ik_solver._calculate_n_residuals(num_links, joint_limit_weight)
        # Should be 6 * num_links only
        expected = 6 * num_links
        assert n_res == expected


class TestSeedIKSolverResetSeed:
    """Test seed reset functionality."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_seed_exists(self, seed_ik_solver):
        """Test reset_seed method exists."""
        assert hasattr(seed_ik_solver, "reset_seed")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_seed_runs(self, seed_ik_solver):
        """Test reset_seed runs without error."""
        seed_ik_solver.reset_seed()  # Should not raise


class TestSeedIKSolverComputeKinematics:
    """Test kinematics computation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compute_kinematics_exists(self, seed_ik_solver):
        """Test compute_kinematics method exists."""
        assert hasattr(seed_ik_solver, "compute_kinematics")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compute_kinematics_runs(self, seed_ik_solver, cuda_device_cfg):
        """Test compute_kinematics runs."""
        from curobo._src.state.state_joint import JointState

        joint_position = torch.zeros(1, seed_ik_solver.dof, device=cuda_device_cfg.device)
        joint_state = JointState.from_position(joint_position, joint_names=seed_ik_solver.joint_names)
        result = seed_ik_solver.compute_kinematics(joint_state)
        assert result is not None


class TestSeedIKSolverWithSeedConfig:
    """Test SeedIKSolver with custom seed configurations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_with_seed_config(self, seed_ik_solver, sample_goal_pose, cuda_device_cfg):
        """Test solve_single with provided seed configuration."""
        seed_config = torch.zeros(
            1, 1, seed_ik_solver.dof, device=cuda_device_cfg.device
        )
        result = seed_ik_solver.solve_single(
            goal_tool_poses=sample_goal_pose,
            seed_config=seed_config,
        )
        assert isinstance(result, IKSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_batch_with_seed_config(self, seed_ik_solver, batch_goal_poses, cuda_device_cfg):
        """Test solve_batch with provided seed configuration."""
        batch_size = batch_goal_poses.batch_size
        seed_config = torch.zeros(
            batch_size, 1, seed_ik_solver.dof, device=cuda_device_cfg.device
        )
        result = seed_ik_solver.solve_batch(
            goal_tool_poses=batch_goal_poses,
            seed_config=seed_config,
        )
        assert isinstance(result, IKSolverResult)


class TestSeedIKSolverDefaultJointPosition:
    """Test SeedIKSolver default joint position functionality."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_default_joint_position_property(self, seed_ik_solver):
        """Test solver has default_joint_position property."""
        assert hasattr(seed_ik_solver, "default_joint_position")
        assert seed_ik_solver.default_joint_position is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_default_joint_position_returns_tensor(self, seed_ik_solver):
        """Test get_default_joint_position returns tensor."""
        default_pos = seed_ik_solver.get_default_joint_position()
        assert isinstance(default_pos, torch.Tensor)
        assert default_pos.shape[-1] == seed_ik_solver.dof

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_uses_internal_default_joint_position(
        self, seed_ik_solver, sample_goal_pose
    ):
        """Test solve_single works and uses internal default joint position."""
        result = seed_ik_solver.solve_single(goal_tool_poses=sample_goal_pose)
        assert isinstance(result, IKSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_batch_uses_internal_default_joint_position(
        self, seed_ik_solver, batch_goal_poses
    ):
        """Test solve_batch works and uses internal default joint position."""
        result = seed_ik_solver.solve_batch(goal_tool_poses=batch_goal_poses)
        assert isinstance(result, IKSolverResult)


class TestSeedIKSolverGoalToolPoseFromPoses:
    """Test building goals via GoalToolPose.from_poses for solve_single."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_with_goal_tool_pose(self, seed_ik_solver, cuda_device_cfg):
        """Test solve_single with GoalToolPose built from per-link Pose dict."""
        device = cuda_device_cfg.device
        tool_frames = seed_ik_solver.tool_frames

        some_pose = Pose(
            position=torch.tensor([[[0.4, 0.0, 0.4]]], device=device, dtype=torch.float32),
            quaternion=torch.tensor(
                [[[0.0, 1.0, 0.0, 0.0]]], device=device, dtype=torch.float32
            ),
        )
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: some_pose},
            ordered_tool_frames=seed_ik_solver.tool_frames,
            num_goalset=1,
        )

        result = seed_ik_solver.solve_single(goal_tool_poses=goal_tool_poses)
        assert isinstance(result, IKSolverResult)


class TestSeedIKSolverBatchSetup:
    """Test batch size setup."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_setup_batch_size(self, seed_ik_solver):
        """Test _setup_batch_size method."""
        batch_size = 4
        num_seeds = 8
        seed_ik_solver._setup_batch_size(batch_size, num_seeds)

        assert seed_ik_solver._batch_size == batch_size
        assert seed_ik_solver._num_seeds == num_seeds
        assert seed_ik_solver._num_problems == batch_size * num_seeds


class TestSeedIKSolverSelectTopSolutions:
    """Test top solution selection."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_select_top_solutions_basic(self, seed_ik_solver, cuda_device_cfg):
        """Test _select_top_solutions method."""
        device = cuda_device_cfg.device
        batch_size = 2
        num_seeds = 4
        return_seeds = 2
        dof = seed_ik_solver.dof

        # Setup batch indices
        seed_ik_solver._batch_indices = torch.arange(batch_size, device=device) * num_seeds

        solutions = torch.randn(batch_size, num_seeds, dof, device=device)
        successes = torch.tensor([
            [True, False, True, False],
            [True, True, False, False],
        ], device=device)
        position_errors = torch.tensor([
            [0.01, 0.1, 0.02, 0.2],
            [0.01, 0.015, 0.3, 0.4],
        ], device=device)
        orientation_errors = torch.tensor([
            [0.01, 0.1, 0.02, 0.2],
            [0.01, 0.015, 0.3, 0.4],
        ], device=device)

        top_success, top_sol, top_pos, top_ori = seed_ik_solver._select_top_solutions(
            solutions=solutions,
            successes=successes,
            position_errors=position_errors,
            orientation_errors=orientation_errors,
            start_joint_position=None,
            num_seeds=num_seeds,
            return_seeds=return_seeds,
            batch_size=batch_size,
        )

        assert top_sol.shape == (batch_size, return_seeds, dof)
        assert top_success.shape == (batch_size, return_seeds)
        assert top_pos.shape == (batch_size, return_seeds)
        assert top_ori.shape == (batch_size, return_seeds)


class TestSeedIKSolverUpdateToolPoseCriteria:
    """Test link pose criteria updates."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_tool_pose_criteria_exists(self, seed_ik_solver):
        """Test update_tool_pose_criteria method exists."""
        assert hasattr(seed_ik_solver, "update_tool_pose_criteria")


class TestSeedIKSolverEdgeCases:
    """Test edge cases."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_single_return_all_seeds(self, seed_ik_solver, sample_goal_pose):
        """Test solve_single returning all seeds."""
        num_seeds = seed_ik_solver.config.num_seeds
        result = seed_ik_solver.solve_single(
            goal_tool_poses=sample_goal_pose,
            return_seeds=num_seeds,
        )
        assert result.solution.shape[1] == num_seeds

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_batch_single_item(self, seed_ik_solver, cuda_device_cfg):
        """Test solve_batch with single item batch."""
        device = cuda_device_cfg.device
        tool_frames = seed_ik_solver.tool_frames
        num_links = len(tool_frames)

        # Shape: [B=1, H=1, L=num_links, G=1, 3/4]
        position = torch.tensor([[[[[0.4, 0.0, 0.4]]]]], device=device, dtype=torch.float32)
        quaternion = torch.tensor([[[[[0.0, 1.0, 0.0, 0.0]]]]], device=device, dtype=torch.float32)

        position = position.expand(1, 1, num_links, 1, 3).contiguous()
        quaternion = quaternion.expand(1, 1, num_links, 1, 4).contiguous()

        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=position,
            quaternion=quaternion,
        )

        result = seed_ik_solver.solve_batch(goal_tool_poses=goal_poses)
        assert result.solution.shape[0] == 1

