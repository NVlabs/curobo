# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for IKSolver class."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
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
def franka_ik_config(cuda_device_cfg):
    """Create IKSolverCfg for Franka robot (batch_size=1, max_goalset=1)."""
    config = IKSolverCfg.create(
        robot="franka.yml",
        device_cfg=cuda_device_cfg,
        num_seeds=32,
        use_cuda_graph=False,
    )
    config.use_lm_seed = False
    return config


@pytest.fixture(scope="module")
def ik_solver(franka_ik_config):
    """Create IKSolver instance (batch_size=1)."""
    return IKSolver(franka_ik_config)


@pytest.fixture
def sample_goal_pose(cuda_device_cfg):
    """Create a sample goal pose for testing."""
    position = torch.tensor([[0.4, 0.0, 0.4]], **cuda_device_cfg.as_torch_dict())
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
    return Pose(position=position, quaternion=quaternion)


class TestIKSolverCfgCreate:
    """Test IKSolverCfg.create factory method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_creates_config(self, cuda_device_cfg):
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=16,
        )
        assert config is not None
        assert isinstance(config, IKSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_num_seeds(self, cuda_device_cfg):
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=64,
        )
        assert config.num_seeds == 64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_tolerances(self, cuda_device_cfg):
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            position_tolerance=0.01,
            orientation_tolerance=0.1,
        )
        assert config.position_tolerance == 0.01
        assert config.orientation_tolerance == 0.1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_default_lm_ik(self, cuda_device_cfg):
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.use_lm_seed is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_batch_config(self, cuda_device_cfg):
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=8,
            multi_env=True,
            max_goalset=10,
        )
        assert config.max_batch_size == 8
        assert config.multi_env is True
        assert config.max_goalset == 10


class TestIKSolverInitialization:
    """Test IKSolver initialization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_with_config(self, franka_ik_config):
        solver = IKSolver(franka_ik_config)
        assert solver is not None
        assert solver.config == franka_ik_config

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_creates_components(self, franka_ik_config):
        solver = IKSolver(franka_ik_config)
        assert hasattr(solver, 'optimizer')
        assert hasattr(solver, 'metrics_rollout')
        assert hasattr(solver, 'seed_manager')
        assert hasattr(solver, 'goal_registry_manager')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_without_lm_ik(self, cuda_device_cfg):
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        config.use_lm_seed = False
        solver = IKSolver(config)
        assert solver.seed_ik_solver is None


class TestIKSolverProperties:
    """Test IKSolver properties."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_action_dim_property(self, ik_solver):
        assert ik_solver.action_dim > 0
        assert ik_solver.action_dim == 7

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_kinematics_property(self, ik_solver):
        assert hasattr(ik_solver, 'kinematics')
        assert ik_solver.kinematics is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_joint_position_property(self, ik_solver):
        default_position = ik_solver.default_joint_position
        assert isinstance(default_position, torch.Tensor)
        assert default_position.shape[-1] == ik_solver.action_dim

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_joint_state_property(self, ik_solver):
        from curobo._src.state.state_joint import JointState
        default_js = ik_solver.default_joint_state
        assert isinstance(default_js, JointState)
        assert default_js.position is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_joint_names_property(self, ik_solver):
        names = ik_solver.joint_names
        assert isinstance(names, list)
        assert len(names) == ik_solver.action_dim


class TestIKSolverSolvePose:
    """Test IKSolver.solve_pose for single problems (batch_size=1)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_returns_result(self, ik_solver, sample_goal_pose):
        tool_frames = ik_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )
        result = ik_solver.solve_pose(goal_tool_poses=goal_tool_poses, return_seeds=1)
        assert isinstance(result, IKSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_result_fields(self, ik_solver, sample_goal_pose):
        tool_frames = ik_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )
        result = ik_solver.solve_pose(goal_tool_poses=goal_tool_poses)
        assert hasattr(result, 'success')
        assert isinstance(result.success, torch.Tensor)
        assert hasattr(result, 'solution')
        assert result.solution.shape[-1] == ik_solver.action_dim
        assert hasattr(result, 'position_error')
        assert hasattr(result, 'rotation_error')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_multiple_return_seeds(self, ik_solver, sample_goal_pose):
        tool_frames = ik_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )
        result = ik_solver.solve_pose(goal_tool_poses=goal_tool_poses, return_seeds=4)
        assert result.solution.shape[0] == 1
        assert result.solution.shape[1] == 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_with_tool_pose_input(self, ik_solver, cuda_device_cfg):
        tool_frames = ik_solver.kinematics.tool_frames
        num_links = len(tool_frames)
        position = torch.zeros((1, 1, num_links, 1, 3), **cuda_device_cfg.as_torch_dict())
        position[0, 0, 0, 0, :] = torch.tensor([0.4, 0.0, 0.4], **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, 1, num_links, 1, 4), **cuda_device_cfg.as_torch_dict())
        quaternion[0, 0, 0, 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict())
        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position,
            quaternion=quaternion,
        )
        result = ik_solver.solve_pose(goal_tool_poses=tool_pose)
        assert isinstance(result, IKSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_timing(self, ik_solver, sample_goal_pose):
        tool_frames = ik_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )
        result = ik_solver.solve_pose(goal_tool_poses=goal_tool_poses)
        assert result.solve_time >= 0.0
        assert result.total_time >= 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_js_solution(self, ik_solver, sample_goal_pose):
        tool_frames = ik_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )
        result = ik_solver.solve_pose(goal_tool_poses=goal_tool_poses)
        assert result.js_solution is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_run_optimizer_false(self, ik_solver, sample_goal_pose):
        tool_frames = ik_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )
        result = ik_solver.solve_pose(goal_tool_poses=goal_tool_poses, run_optimizer=False)
        assert result is not None


class TestIKSolverSolvePoseBatch:
    """Test IKSolver.solve_pose with batch_size > 1."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_batch(self, cuda_device_cfg):
        batch_size = 4
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=4,
            use_cuda_graph=False,
            max_batch_size=batch_size,
        )
        config.use_lm_seed = False
        solver = IKSolver(config)

        tool_frames = solver.kinematics.tool_frames
        position = torch.tensor(
            [[0.4, 0.0, 0.4]] * batch_size, **cuda_device_cfg.as_torch_dict()
        )
        quaternion = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * batch_size, **cuda_device_cfg.as_torch_dict()
        )
        pose = Pose(position=position, quaternion=quaternion)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = solver.solve_pose(goal_tool_poses=goal_tool_poses)
        assert isinstance(result, IKSolverResult)
        assert result.solution.shape[0] == batch_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_batch_run_optimizer_false(self, cuda_device_cfg):
        batch_size = 2
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=4,
            use_cuda_graph=False,
            max_batch_size=batch_size,
        )
        config.use_lm_seed = False
        solver = IKSolver(config)

        tool_frames = solver.kinematics.tool_frames
        position = torch.tensor(
            [[0.4, 0.0, 0.4]] * batch_size, **cuda_device_cfg.as_torch_dict()
        )
        quaternion = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * batch_size, **cuda_device_cfg.as_torch_dict()
        )
        pose = Pose(position=position, quaternion=quaternion)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = solver.solve_pose(goal_tool_poses=goal_tool_poses, run_optimizer=False)
        assert result is not None


class TestIKSolverSolvePoseBatchEnv:
    """Test IKSolver.solve_pose with multi_env=True."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_multi_env(self, cuda_device_cfg):
        batch_size = 2
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=4,
            use_cuda_graph=False,
            max_batch_size=batch_size,
            multi_env=True,
        )
        config.use_lm_seed = False
        solver = IKSolver(config)

        tool_frames = solver.kinematics.tool_frames
        position = torch.tensor(
            [[0.4, 0.0, 0.4]] * batch_size, **cuda_device_cfg.as_torch_dict()
        )
        quaternion = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * batch_size, **cuda_device_cfg.as_torch_dict()
        )
        pose = Pose(position=position, quaternion=quaternion)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = solver.solve_pose(goal_tool_poses=goal_tool_poses)
        assert isinstance(result, IKSolverResult)


class TestIKSolverVariableBatchSize:
    """Test IKSolver.solve_pose with batch_size < max_batch_size."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_fewer_than_max(self, cuda_device_cfg):
        """Solve 2 problems with max_batch_size=4: padded internally, sliced back."""
        max_batch = 4
        n_actual = 2
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=4,
            use_cuda_graph=False,
            max_batch_size=max_batch,
        )
        config.use_lm_seed = False
        solver = IKSolver(config)

        tool_frames = solver.kinematics.tool_frames
        position = torch.tensor(
            [[0.4, 0.0, 0.4]] * n_actual, **cuda_device_cfg.as_torch_dict()
        )
        quaternion = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * n_actual, **cuda_device_cfg.as_torch_dict()
        )
        pose = Pose(position=position, quaternion=quaternion)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = solver.solve_pose(goal_tool_poses=goal_tool_poses)
        assert isinstance(result, IKSolverResult)
        assert result.success.shape[0] == n_actual
        assert result.solution.shape[0] == n_actual

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_single_problem_ibatch_size_solver(self, cuda_device_cfg):
        """Solve 1 problem with max_batch_size=4."""
        max_batch = 4
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=4,
            use_cuda_graph=False,
            max_batch_size=max_batch,
        )
        config.use_lm_seed = False
        solver = IKSolver(config)

        tool_frames = solver.kinematics.tool_frames
        position = torch.tensor([[0.4, 0.0, 0.4]], **cuda_device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = solver.solve_pose(goal_tool_poses=goal_tool_poses)
        assert result.success.shape[0] == 1
        assert result.solution.shape[0] == 1


class TestIKSolverSolvePoseGoalset:
    """Test IKSolver.solve_pose with goalset (num_goalset > 1)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_goalset(self, cuda_device_cfg):
        num_goalset = 3
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=32,
            use_cuda_graph=False,
            max_goalset=num_goalset,
        )
        config.use_lm_seed = False
        solver = IKSolver(config)

        tool_frames = solver.kinematics.tool_frames
        num_links = len(tool_frames)
        position = torch.zeros((1, 1, num_links, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, 1, num_links, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, 0, 0, i, :] = torch.tensor(
                [0.4 + i * 0.05, 0.0, 0.4], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, 0, 0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )
        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position,
            quaternion=quaternion,
        )
        result = solver.solve_pose(goal_tool_poses=tool_pose)
        assert isinstance(result, IKSolverResult)
        assert hasattr(result, 'goalset_index')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_goalset_with_return_seeds(self, cuda_device_cfg):
        num_goalset = 2
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=32,
            use_cuda_graph=False,
            max_goalset=num_goalset,
        )
        config.use_lm_seed = False
        solver = IKSolver(config)

        tool_frames = solver.kinematics.tool_frames
        num_links = len(tool_frames)
        position = torch.zeros((1, 1, num_links, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, 1, num_links, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, 0, 0, i, :] = torch.tensor(
                [0.4, 0.0, 0.4 + i * 0.05], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, 0, 0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )
        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position,
            quaternion=quaternion,
        )
        result = solver.solve_pose(goal_tool_poses=tool_pose, return_seeds=4)
        assert result.solution.shape[0] == 1
        assert result.solution.shape[1] == 4


class TestIKSolverResetSeed:
    """Test IKSolver.reset_seed method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_seed_no_error(self, ik_solver):
        ik_solver.reset_seed()


class TestIKSolverGetUniqueSolution:
    """Test IKSolver.get_unique_solution method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_unique_solution(self, ik_solver, sample_goal_pose):
        tool_frames = ik_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )
        result = ik_solver.solve_pose(goal_tool_poses=goal_tool_poses, return_seeds=4)
        if result.success.any():
            ik_solver.solution = result.solution.squeeze(0)
            ik_solver.success = result.success.squeeze(0)
            unique_sols = ik_solver.get_unique_solution(roundoff_decimals=1)
            assert isinstance(unique_sols, torch.Tensor)


class TestIKSolverProblemBatchSize:
    """Test IKSolver.problem_batch_size property."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_problem_batch_size_after_solve(self, ik_solver, sample_goal_pose):
        tool_frames = ik_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )
        ik_solver.solve_pose(goal_tool_poses=goal_tool_poses)
        batch_size = ik_solver.problem_batch_size
        assert isinstance(batch_size, int)


class TestIKSolverUpdateWorld:
    """Test IKSolver.update_world method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_world_method_exists(self, ik_solver):
        assert hasattr(ik_solver, 'update_world')
        assert callable(ik_solver.update_world)
