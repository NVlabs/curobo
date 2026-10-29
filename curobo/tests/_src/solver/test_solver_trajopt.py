# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for TrajOptSolver class."""

# Standard Library

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.robot.dynamics.dynamics import Dynamics
from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.solver.processor_trajopt_result import TorchTrajOptResultProcessor
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def cuda_device_cfg():
    """Create a CUDA device configuration for testing."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def franka_trajopt_config(cuda_device_cfg):
    """Create TrajOptSolverCfg for Franka robot."""
    config = TrajOptSolverCfg.create(
        robot="franka.yml",
        device_cfg=cuda_device_cfg,
        num_seeds=4,
        use_cuda_graph=False,
    )
    return config


@pytest.fixture(scope="module")
def trajopt_solver(franka_trajopt_config):
    """Create TrajOptSolver instance."""
    solver = TrajOptSolver(franka_trajopt_config)
    return solver


@pytest.fixture(scope="module")
def franka_trajopt_config_goalset(cuda_device_cfg):
    """TrajOptSolverCfg with room for multi-pose goalsets (num_goalset > 1)."""
    return TrajOptSolverCfg.create(
        robot="franka.yml",
        device_cfg=cuda_device_cfg,
        num_seeds=4,
        use_cuda_graph=False,
        max_goalset=4,
    )


@pytest.fixture(scope="module")
def trajopt_solver_goalset(franka_trajopt_config_goalset):
    """TrajOptSolver instance configured for goalset solves."""
    return TrajOptSolver(franka_trajopt_config_goalset)


@pytest.fixture
def sample_start_state(trajopt_solver, cuda_device_cfg):
    """Create a sample start state for testing."""
    default_js = trajopt_solver.default_joint_state.clone()
    return JointState.from_position(
        default_js.position.unsqueeze(0),
        joint_names=trajopt_solver.joint_names,
    )


@pytest.fixture
def sample_goal_pose(cuda_device_cfg):
    """Create a sample goal pose for testing."""
    position = torch.tensor([[0.4, 0.0, 0.4]], **cuda_device_cfg.as_torch_dict())
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
    return Pose(position=position, quaternion=quaternion)


@pytest.fixture
def sample_start_state_goalset(trajopt_solver_goalset):
    """Start state for goalset tests (matches trajopt_solver_goalset)."""
    default_js = trajopt_solver_goalset.default_joint_state.clone()
    return JointState.from_position(
        default_js.position.unsqueeze(0),
        joint_names=trajopt_solver_goalset.joint_names,
    )


@pytest.mark.parametrize("batched", [False, True])
def test_payload_updates_refresh_metrics_dynamics(
    cuda_device_cfg: DeviceCfg, batched: bool
) -> None:
    """Keep every rollout's torques consistent after loading and unloading a payload."""
    robot = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot["load_dynamics"] = True
    solver = TrajOptSolver(
        TrajOptSolverCfg.create(
            robot=robot, device_cfg=cuda_device_cfg, num_seeds=1, use_cuda_graph=False
        )
    )
    rollouts = solver.core.get_all_rollout_instances() + list(
        solver.core.additional_metrics_rollouts.values()
    )
    assert "interpolated_rollout" in solver.core.additional_metrics_rollouts
    models = [
        model
        for rollout in rollouts
        for model in (rollout.transition_model, rollout.metrics_transition_model)
    ]
    position = solver.default_joint_state.position.reshape(1, 1, -1).clone()
    state = JointState(
        position=position,
        velocity=torch.full_like(position, 0.1),
        acceleration=torch.full_like(position, 0.2),
    )
    for model in models:
        assert model.compute_inverse_dynamics
        model.robot_dynamics.setup_batch_size(1, 1)
    previous_torque = models[0].robot_dynamics.compute_inverse_dynamics(state).clone()
    com = torch.tensor([0.01, 0.02, 0.1], **cuda_device_cfg.as_torch_dict())
    for mass in (1.0, 0.1):
        inertia = torch.tensor(
            [0.001, 0.002, 0.003, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
        ) * mass
        if batched:
            solver.update_links_inertial(
                {"attached_object": {"mass": mass, "com": com, "inertia": inertia}}
            )
        else:
            solver.update_link_inertial("attached_object", mass, com, inertia)
        reference = Dynamics(
            DynamicsCfg(
                kinematics_config=solver.kinematics.config.kinematics_config,
                device_cfg=cuda_device_cfg,
            )
        )
        reference.setup_batch_size(1, 1)
        expected_torque = reference.compute_inverse_dynamics(state).clone()
        assert not torch.allclose(expected_torque, previous_torque)
        for model in models:
            torch.testing.assert_close(
                model.robot_dynamics.compute_inverse_dynamics(state), expected_torque
            )
        previous_torque = expected_torque


class TestTrajOptSolverCfgCreation:
    """Test TrajOptSolverCfg creation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_creates_config(self, cuda_device_cfg):
        """Test create creates a valid config."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=4,
        )
        assert config is not None
        assert isinstance(config, TrajOptSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_num_seeds(self, cuda_device_cfg):
        """Test create sets num_seeds correctly."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=8,
        )
        assert config.num_seeds == 8


class TestTrajOptSolverInitialization:
    """Test TrajOptSolver initialization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_with_config(self, franka_trajopt_config):
        """Test TrajOptSolver initializes with config."""
        solver = TrajOptSolver(franka_trajopt_config)
        assert solver is not None
        assert solver.config == franka_trajopt_config

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_creates_components(self, franka_trajopt_config):
        """Test TrajOptSolver creates necessary components."""
        solver = TrajOptSolver(franka_trajopt_config)

        assert hasattr(solver, 'optimizer')
        assert hasattr(solver, 'metrics_rollout')
        assert hasattr(solver, 'seed_manager')
        assert hasattr(solver, 'goal_registry_manager')
        assert hasattr(solver, '_trajectory_seed_generator')
        assert isinstance(solver.result_processor, TorchTrajOptResultProcessor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_creates_interpolated_rollout(self, franka_trajopt_config):
        """Test TrajOptSolver creates interpolated rollout."""
        solver = TrajOptSolver(franka_trajopt_config)

        assert "interpolated_rollout" in solver.additional_metrics_rollouts


class TestTrajOptSolverProperties:
    """Test TrajOptSolver properties."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_action_dim_property(self, trajopt_solver):
        """Test action_dim property returns DOF."""
        assert trajopt_solver.action_dim > 0
        # Franka has 7 DOFs
        assert trajopt_solver.action_dim == 7

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_action_horizon_property(self, trajopt_solver):
        """Test action_horizon property."""
        assert trajopt_solver.action_horizon > 1  # TrajOpt has multi-step horizon

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_kinematics_property(self, trajopt_solver):
        """Test kinematics property exists."""
        assert hasattr(trajopt_solver, 'kinematics')
        assert trajopt_solver.kinematics is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_joint_position_property(self, trajopt_solver):
        """Test default_joint_position property."""
        default_position = trajopt_solver.default_joint_position
        assert isinstance(default_position, torch.Tensor)
        assert default_position.shape[-1] == trajopt_solver.action_dim

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_joint_state_property(self, trajopt_solver):
        """Test default_joint_state property returns JointState."""
        default_js = trajopt_solver.default_joint_state
        assert isinstance(default_js, JointState)
        assert default_js.position is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_joint_names_property(self, trajopt_solver):
        """Test joint_names property."""
        names = trajopt_solver.joint_names
        assert isinstance(names, list)
        assert len(names) == trajopt_solver.action_dim


class TestTrajOptSolverSolvePose:
    """Test TrajOptSolver.solve_pose method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_method_exists(self, trajopt_solver):
        assert hasattr(trajopt_solver, 'solve_pose')
        assert callable(trajopt_solver.solve_pose)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_returns_result(
        self, trajopt_solver, sample_start_state, sample_goal_pose
    ):
        tool_frames = trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        result = trajopt_solver.solve_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            return_seeds=1,
        )

        assert isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_result_has_success(
        self, trajopt_solver, sample_start_state, sample_goal_pose
    ):
        tool_frames = trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        result = trajopt_solver.solve_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
        )

        assert hasattr(result, 'success')
        assert isinstance(result.success, torch.Tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_result_has_solution(
        self, trajopt_solver, sample_start_state, sample_goal_pose
    ):
        tool_frames = trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        result = trajopt_solver.solve_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
        )

        assert hasattr(result, 'solution')
        if result.solution is not None:
            assert result.solution.ndim == 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_trajectory_dt_update_is_visible_to_all_rollouts(
        self, trajopt_solver, sample_start_state, sample_goal_pose
    ):
        """A central dt write is visible through every rollout's cached goal aliases."""
        tool_frames = trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )
        trajopt_solver.solve_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
        )
        goal_buffer = trajopt_solver.core._goal_buffer
        central_dt = goal_buffer.seed_goal_js.dt
        updated_dt = torch.full_like(central_dt, 0.04)

        trajopt_solver._update_trajectory_dt(updated_dt)

        for rollout in trajopt_solver.get_all_rollout_instances(
            include_optimizer_rollouts=True, include_auxiliary_rollout=False
        ):
            assert rollout._metrics_goal.seed_goal_js.dt is central_dt
            assert rollout._num_particles_goal.seed_goal_js.dt is central_dt
            torch.testing.assert_close(rollout._num_particles_goal.seed_goal_js.dt, updated_dt)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_seed_config_reuses_initial_goal_buffer(
        self, trajopt_solver, sample_start_state, sample_goal_pose, monkeypatch
    ):
        """Initial retiming and optimization share one full goal-buffer preparation."""
        tool_frames = trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )
        seed_config = sample_start_state.position.view(1, 1, -1).repeat(
            1, trajopt_solver.config.num_seeds, 1
        )
        prepare_calls = 0
        original_prepare = trajopt_solver._prepare_goal_buffer

        def count_prepare(*args, **kwargs):
            nonlocal prepare_calls
            prepare_calls += 1
            return original_prepare(*args, **kwargs)

        monkeypatch.setattr(trajopt_solver, "_prepare_goal_buffer", count_prepare)

        trajopt_solver.solve_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            seed_config=seed_config,
        )

        assert prepare_calls == 1


class TestTrajOptSolverProperties2:
    """Test additional TrajOptSolver properties."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_action_horizon_property(self, trajopt_solver):
        """Test action_horizon property returns horizon from rollout."""
        horizon = trajopt_solver.action_horizon
        assert isinstance(horizon, int)
        assert horizon > 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_horizon_property(self, trajopt_solver):
        """Test horizon property."""
        horizon = trajopt_solver.horizon
        assert isinstance(horizon, int)
        assert horizon > 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_opt_dim_property(self, trajopt_solver):
        """Test opt_dim property."""
        opt_dim = trajopt_solver.opt_dim
        assert isinstance(opt_dim, int)
        assert opt_dim > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_interpolation_steps_property(self, trajopt_solver):
        """Test interpolation_steps property."""
        steps = trajopt_solver.interpolation_steps
        assert isinstance(steps, int)
        assert steps == 4


class TestTrajOptSolverResultFromSolve:
    """Test TrajOptSolverResult from solve_pose."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_result_has_timing_info(
        self, trajopt_solver, sample_start_state, sample_goal_pose
    ):
        """Test result has timing information."""
        tool_frames = trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        result = trajopt_solver.solve_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
        )

        assert hasattr(result, 'solve_time')
        assert hasattr(result, 'total_time')
        assert result.solve_time >= 0.0
        assert result.total_time >= 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_result_has_interpolated_trajectory(
        self, trajopt_solver, sample_start_state, sample_goal_pose
    ):
        """Test result has interpolated_trajectory field."""
        tool_frames = trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        result = trajopt_solver.solve_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
        )

        assert hasattr(result, 'interpolated_trajectory')
        full_joint_names = trajopt_solver.kinematics.all_articulated_joint_names
        assert result.interpolated_trajectory.joint_names == full_joint_names
        assert result.interpolated_trajectory.shape[-1] == len(full_joint_names)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_result_has_js_solution(
        self, trajopt_solver, sample_start_state, sample_goal_pose
    ):
        """Test result has js_solution trajectory."""
        tool_frames = trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        result = trajopt_solver.solve_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
        )

        assert hasattr(result, 'js_solution')
        if result.js_solution is not None:
            assert isinstance(result.js_solution, JointState)


class TestTrajOptSolverResetSeed:
    """Test TrajOptSolver.reset_seed method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_seed_no_error(self, trajopt_solver):
        """Test reset_seed doesn't raise error."""
        trajopt_solver.reset_seed()


class TestTrajOptSolverComputeKinematics:
    """Test TrajOptSolver.compute_kinematics method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compute_kinematics_returns_result(self, trajopt_solver, sample_start_state):
        """Test compute_kinematics returns valid result."""
        result = trajopt_solver.compute_kinematics(sample_start_state)

        assert result is not None
        assert hasattr(result, 'tool_poses')


class TestTrajOptSolverWithToolPose:
    """Test TrajOptSolver ToolPose support."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_with_tool_pose_input(
        self, trajopt_solver, sample_start_state, cuda_device_cfg
    ):
        """Test solve_pose accepts ToolPose input."""
        tool_frames = trajopt_solver.kinematics.tool_frames
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

        result = trajopt_solver.solve_pose(
            current_state=sample_start_state,
            goal_tool_poses=tool_pose,
        )
        assert isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tool_frames_available(self, trajopt_solver):
        """Test kinematics tool_frames available for goal creation."""
        tool_frames = trajopt_solver.kinematics.tool_frames

        assert isinstance(tool_frames, list)
        assert len(tool_frames) > 0


class TestTrajOptSolverPrepareTrajectorySeeds:
    """Test TrajOptSolver.prepare_trajectory_seeds method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_prepare_trajectory_seeds_exists(self, trajopt_solver):
        """Test prepare_trajectory_seeds method exists."""
        assert hasattr(trajopt_solver, 'prepare_trajectory_seeds')
        assert callable(trajopt_solver.prepare_trajectory_seeds)


class TestTrajOptSolverEvaluateSeed:
    """Test retiming and dense validation without optimization."""

    @staticmethod
    def _make_cspace_seed(trajopt_solver, sample_start_state):
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.1
        num_seeds = trajopt_solver.config.num_seeds
        seed_config = goal_state.position.view(1, 1, -1).repeat(1, num_seeds, 1)
        seed_traj = trajopt_solver.prepare_trajectory_seeds(
            1,
            num_seeds,
            sample_start_state,
            seed_config=seed_config,
        ).view(1, num_seeds, trajopt_solver.action_horizon, trajopt_solver.action_dim)
        return goal_state, seed_traj

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_seed_valid_mask_rejects_non_graph_lanes(
        self,
        trajopt_solver,
        sample_start_state,
    ):
        """Do not count straight-line fallbacks as graph-planner successes."""
        goal_state, seed_traj = self._make_cspace_seed(trajopt_solver, sample_start_state)
        seed_valid_mask = torch.zeros(
            (1, trajopt_solver.config.num_seeds),
            dtype=torch.bool,
            device=seed_traj.device,
        )

        result = trajopt_solver.evaluate_cspace_seed(
            goal_state,
            sample_start_state,
            seed_traj,
            seed_valid_mask=seed_valid_mask,
        )

        assert not result.success.any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_required_dt_above_maximum_rejects_seed(
        self,
        trajopt_solver,
        sample_start_state,
        monkeypatch,
    ):
        """Reject seeds whose unclamped retiming exceeds the solver limit."""
        goal_state, seed_traj = self._make_cspace_seed(trajopt_solver, sample_start_state)

        def above_maximum_dt(trajectory, epsilon=1e-3, scale_dt=True):
            del epsilon, scale_dt
            return torch.full(
                (trajectory.shape[0],),
                trajopt_solver.config.maximum_trajectory_dt + 0.1,
                device=trajectory.position.device,
                dtype=trajectory.position.dtype,
            )

        monkeypatch.setattr(
            trajopt_solver,
            "_compute_unclamped_trajectory_dt",
            above_maximum_dt,
        )

        result = trajopt_solver.evaluate_cspace_seed(
            goal_state,
            sample_start_state,
            seed_traj,
        )

        assert not result.success.any()


class TestTrajOptSolverResetMethods:
    """Test TrajOptSolver reset methods."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_shape_method_exists(self, trajopt_solver):
        """Test reset_shape method exists."""
        assert hasattr(trajopt_solver, 'reset_shape')
        assert callable(trajopt_solver.reset_shape)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_cuda_graph_method_exists(self, trajopt_solver):
        """Test reset_cuda_graph method exists."""
        assert hasattr(trajopt_solver, 'reset_cuda_graph')
        assert callable(trajopt_solver.reset_cuda_graph)


class TestTrajOptSolverSolvePoseGoalset:
    """Test TrajOptSolver.solve_pose with goalset (num_goalset > 1)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_method_exists(self, trajopt_solver_goalset):
        assert hasattr(trajopt_solver_goalset, 'solve_pose')
        assert callable(trajopt_solver_goalset.solve_pose)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_returns_result(
        self, trajopt_solver_goalset, sample_start_state_goalset, cuda_device_cfg
    ):
        """Test solve_pose returns TrajOptSolverResult."""
        tool_frames = trajopt_solver_goalset.kinematics.tool_frames
        num_links = len(tool_frames)
        num_goalset = 2

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

        result = trajopt_solver_goalset.solve_pose(
            current_state=sample_start_state_goalset,
            goal_tool_poses=tool_pose,
        )

        assert isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_result_has_success(
        self, trajopt_solver_goalset, sample_start_state_goalset, cuda_device_cfg
    ):
        """Test solve_pose result has success field."""
        tool_frames = trajopt_solver_goalset.kinematics.tool_frames
        num_links = len(tool_frames)
        num_goalset = 2

        position = torch.zeros((1, 1, num_links, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, 1, num_links, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        for i in range(num_goalset):
            position[0, 0, 0, i, :] = torch.tensor(
                [0.4, i * 0.1, 0.4], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, 0, 0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position,
            quaternion=quaternion,
        )

        result = trajopt_solver_goalset.solve_pose(
            current_state=sample_start_state_goalset,
            goal_tool_poses=tool_pose,
        )

        assert hasattr(result, 'success')
        assert isinstance(result.success, torch.Tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_with_dict_input(
        self, trajopt_solver_goalset, sample_start_state_goalset, sample_goal_pose
    ):
        """Test solve_pose accepts dict input."""
        tool_frames = trajopt_solver_goalset.kinematics.tool_frames
        # Single goalset pose via dict
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        result = trajopt_solver_goalset.solve_pose(
            current_state=sample_start_state_goalset,
            goal_tool_poses=goal_tool_poses,
        )

        assert isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_result_has_timing(
        self, trajopt_solver_goalset, sample_start_state_goalset, cuda_device_cfg
    ):
        """Test solve_pose result has timing information."""
        tool_frames = trajopt_solver_goalset.kinematics.tool_frames
        num_links = len(tool_frames)
        num_goalset = 2

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

        result = trajopt_solver_goalset.solve_pose(
            current_state=sample_start_state_goalset,
            goal_tool_poses=tool_pose,
        )

        assert hasattr(result, 'solve_time')
        assert hasattr(result, 'total_time')
        assert result.solve_time >= 0.0
        assert result.total_time >= 0.0


class TestTrajOptSolverSolveCspace:
    """Test TrajOptSolver.solve_cspace method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_cspace_method_exists(self, trajopt_solver):
        assert hasattr(trajopt_solver, 'solve_cspace')
        assert callable(trajopt_solver.solve_cspace)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_interpolation_buffer_error_is_actionable(self, cuda_device_cfg) -> None:
        """Test an undersized interpolation buffer reports both remedies."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=1,
            interpolation_buffer_size=1,
            use_cuda_graph=False,
        )
        solver = TrajOptSolver(config)
        current_state = JointState.from_position(
            solver.default_joint_state.position.unsqueeze(0),
            joint_names=solver.joint_names,
        )
        goal_state = current_state.clone()
        goal_state.position[..., 0] += 0.1

        with pytest.raises(ValueError) as error:
            solver.solve_cspace(goal_state=goal_state, current_state=current_state)

        message = str(error.value)
        assert "interpolation_buffer_size=1" in message
        assert "larger interpolation_buffer_size or interpolation_dt" in message
        assert "significantly increase GPU memory usage" in message

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_cspace_uses_implicit_seed_goal(
        self, trajopt_solver, sample_start_state, monkeypatch
    ):
        """solve_cspace should not populate goal_js; seed_goal_js pins the endpoint."""
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.1
        captured = {}

        def fake_solve_impl(**kwargs):
            captured.update(kwargs)
            solve_state = kwargs["solve_state"]
            return TrajOptSolverResult(
                success=torch.ones(
                    (solve_state.batch_size, kwargs["return_seeds"]),
                    device=sample_start_state.device,
                    dtype=torch.bool,
                )
            )

        monkeypatch.setattr(trajopt_solver, "_solve_impl", fake_solve_impl)

        trajopt_solver.solve_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
        )

        assert captured["use_implicit_goal"] is True
        assert "goal_state" not in captured

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_cspace_returns_result(self, trajopt_solver, sample_start_state):
        """Test solve_cspace returns TrajOptSolverResult."""
        # Create goal state slightly different from start
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.1  # Move first joint

        result = trajopt_solver.solve_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
        )

        assert isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_cspace_result_has_success(self, trajopt_solver, sample_start_state):
        """Test solve_cspace result has success field."""
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.1

        result = trajopt_solver.solve_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
        )

        assert hasattr(result, 'success')
        assert isinstance(result.success, torch.Tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_cspace_result_has_solution(self, trajopt_solver, sample_start_state):
        """Test solve_cspace result has solution field."""
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.1

        result = trajopt_solver.solve_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
        )

        assert hasattr(result, 'solution')
        if result.solution is not None:
            # Shape should be (batch, return_seeds, horizon, dof)
            assert result.solution.ndim == 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_cspace_result_has_timing(self, trajopt_solver, sample_start_state):
        """Test solve_cspace result has timing information."""
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.1

        result = trajopt_solver.solve_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
        )

        assert hasattr(result, 'solve_time')
        assert hasattr(result, 'total_time')
        assert result.solve_time >= 0.0
        assert result.total_time >= 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_cspace_result_has_js_solution(self, trajopt_solver, sample_start_state):
        """Test solve_cspace result has js_solution trajectory."""
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.1

        result = trajopt_solver.solve_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
        )

        assert hasattr(result, 'js_solution')
        if result.js_solution is not None:
            assert isinstance(result.js_solution, JointState)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_cspace_same_start_goal(self, trajopt_solver, sample_start_state):
        """Test solve_cspace when goal equals start."""
        goal_state = sample_start_state.clone()

        result = trajopt_solver.solve_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
        )

        assert isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_cspace_with_return_seeds(self, trajopt_solver, sample_start_state):
        """Test solve_cspace with return_seeds parameter."""
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.1

        result = trajopt_solver.solve_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
            return_seeds=2,
        )

        assert isinstance(result, TrajOptSolverResult)
        # Result should have shape with return_seeds dimension
        if result.success is not None:
            assert result.success.shape[-1] == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_cspace_multiple_joints(self, trajopt_solver, sample_start_state):
        """Test solve_cspace with multiple joint changes."""
        goal_state = sample_start_state.clone()
        # Move multiple joints
        goal_state.position[..., 0] += 0.1
        goal_state.position[..., 2] += 0.15
        goal_state.position[..., 4] -= 0.1

        result = trajopt_solver.solve_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
        )

        assert isinstance(result, TrajOptSolverResult)
