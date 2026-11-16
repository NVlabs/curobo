# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for IKSolver class."""

# Third Party
import pytest
import torch

# CuRobo
import curobo._src.solver.solver_ik as solver_ik_module
from curobo._src.robot.kinematics.kinematics import KinematicsState
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_ik_result import IKSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_multi_link_ik_uses_additional_lm_seeds(self, cuda_device_cfg) -> None:
        """Test two-link IK increases the LM seed count."""
        config = IKSolverCfg.create(
            robot="dual_ur10e.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=32,
            seed_solver_num_seeds=32,
            use_cuda_graph=False,
        )

        solver = IKSolver(config)

        assert len(solver.kinematics.tool_frames) == 2
        assert solver.seed_ik_solver.config.num_seeds == 128
        assert solver.seed_ik_solver.config.max_iterations == 20


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


class TestIKSolverGetResult:
    """Test selection and assembly of IK results."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_result_reuses_feasibility_and_selects_complete_state(
        self,
        ik_solver: IKSolver,
        cuda_device_cfg: DeviceCfg,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reduce cached metrics and gather complete multi-link solution state."""
        problem_count = 4
        num_seeds = 4
        return_seeds = 2
        num_links = 2
        num_spheres = 3
        action_dim = ik_solver.action_dim
        tensor_args = cuda_device_cfg.as_torch_dict()

        opt_result = torch.arange(
            problem_count * action_dim,
            **tensor_args,
        ).view(problem_count, 1, action_dim)
        position_values = torch.tensor(
            [
                [[0.001, 0.004]],
                [[0.006, 0.001]],
                [[0.002, 0.003]],
                [[0.001, 0.002]],
            ],
            **tensor_args,
        )
        orientation_values = torch.tensor(
            [
                [[0.01, 0.02]],
                [[0.01, 0.01]],
                [[0.04, 0.01]],
                [[0.06, 0.01]],
            ],
            **tensor_args,
        )
        cspace_values = torch.tensor([[0.1], [0.0], [0.0], [0.0]], **tensor_args)
        goalset_values = torch.tensor(
            [[[10, 11]], [[20, 21]], [[30, 31]], [[40, 41]]],
            device=cuda_device_cfg.device,
            dtype=torch.int32,
        )
        convergence = CostCollection(
            names=[
                "start_cspace_dist_tolerance",
                "tool_pose_position_tolerance",
                "tool_pose_orientation_tolerance",
                "tool_pose_goalset_index",
            ],
            values=[
                cspace_values,
                position_values,
                orientation_values,
                goalset_values,
            ],
        )
        costs_and_constraints = CostsAndConstraints()

        def fail_feasibility_recomputation(*args: object, **kwargs: object) -> None:
            pytest.fail("cached RolloutMetrics.feasible should be used")

        monkeypatch.setattr(
            costs_and_constraints,
            "get_feasible",
            fail_feasibility_recomputation,
        )

        tool_position = torch.arange(
            problem_count * num_links * 3,
            **tensor_args,
        ).view(problem_count, 1, num_links, 3)
        tool_quaternion = torch.arange(
            problem_count * num_links * 4,
            **tensor_args,
        ).view(problem_count, 1, num_links, 4)
        robot_spheres = torch.arange(
            problem_count * num_spheres * 4,
            **tensor_args,
        ).view(problem_count, 1, num_spheres, 4)
        robot_com = torch.arange(problem_count * 4, **tensor_args).view(
            problem_count, 1, 4
        )
        state = RobotState(
            joint_state=JointState.from_position(opt_result, joint_names=ik_solver.joint_names),
            cuda_robot_model_state=KinematicsState(
                tool_poses=ToolPose(
                    tool_frames=["tool_a", "tool_b"],
                    position=tool_position,
                    quaternion=tool_quaternion,
                ),
                robot_spheres=robot_spheres,
                robot_com=robot_com,
            ),
        )
        metrics = RolloutMetrics(
            costs_and_constraints=costs_and_constraints,
            feasible=torch.ones(
                (problem_count, 1),
                device=cuda_device_cfg.device,
                dtype=torch.bool,
            ),
            convergence=convergence,
            state=state,
        )

        torch_result = ik_solver._get_result_torch(
            opt_result,
            metrics,
            num_seeds=num_seeds,
            return_seeds=return_seeds,
        )
        result = ik_solver._get_result(
            opt_result,
            metrics,
            num_seeds=num_seeds,
            return_seeds=return_seeds,
        )

        for field in (
            "success",
            "feasible",
            "solution",
            "position_error",
            "rotation_error",
            "goalset_index",
        ):
            torch.testing.assert_close(getattr(result, field), getattr(torch_result, field))
        torch.testing.assert_close(
            result.debug_info["seed_idx"], torch_result.debug_info["seed_idx"]
        )
        torch.testing.assert_close(
            result.solution_state.tool_poses.position,
            torch_result.solution_state.tool_poses.position,
        )
        torch.testing.assert_close(
            result.solution_state.tool_poses.quaternion,
            torch_result.solution_state.tool_poses.quaternion,
        )
        torch.testing.assert_close(
            result.solution_state.robot_spheres,
            torch_result.solution_state.robot_spheres,
        )
        torch.testing.assert_close(
            result.solution_state.cuda_robot_model_state.robot_com,
            torch_result.solution_state.cuda_robot_model_state.robot_com,
        )
        torch.testing.assert_close(result.js_solution.position, torch_result.js_solution.position)
        torch.testing.assert_close(result.js_solution.velocity, torch_result.js_solution.velocity)
        torch.testing.assert_close(
            result.js_solution.acceleration, torch_result.js_solution.acceleration
        )
        torch.testing.assert_close(result.js_solution.jerk, torch_result.js_solution.jerk)

        solution_ptr = result.solution.data_ptr()
        result.solution_state.joint_state.velocity.fill_(1.0)
        result.js_solution.velocity.fill_(1.0)
        result = ik_solver._get_result(
            opt_result,
            metrics,
            num_seeds=num_seeds,
            return_seeds=return_seeds,
        )
        assert result.solution.data_ptr() == solution_ptr
        assert torch.count_nonzero(result.solution_state.joint_state.velocity) == 0
        assert torch.count_nonzero(result.js_solution.velocity) == 0

        expected_indices = torch.tensor([0, 2], device=cuda_device_cfg.device)
        torch.testing.assert_close(
            result.solution[0],
            opt_result[:, 0].index_select(0, expected_indices),
        )
        torch.testing.assert_close(
            result.position_error,
            torch.tensor([[0.004, 0.003]], **tensor_args),
        )
        torch.testing.assert_close(
            result.rotation_error,
            torch.tensor([[0.02, 0.04]], **tensor_args),
        )
        torch.testing.assert_close(
            result.goalset_index[0],
            goalset_values[:, 0].index_select(0, expected_indices),
        )
        torch.testing.assert_close(
            result.solution_state.robot_spheres[0],
            robot_spheres[:, 0].index_select(0, expected_indices),
        )
        assert result.solution_state.robot_spheres.shape == (
            1,
            return_seeds,
            num_spheres,
            4,
        )
        assert result.js_solution.position.shape[-1] == len(
            ik_solver.auxiliary_rollout.transition_model.robot_model.all_articulated_joint_names
        )
        assert torch.count_nonzero(result.js_solution.velocity) == 0
        assert torch.count_nonzero(result.js_solution.acceleration) == 0
        assert torch.count_nonzero(result.js_solution.jerk) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_result_cuda_graph_replays_with_updated_metrics(
        self,
        cuda_device_cfg: DeviceCfg,
    ) -> None:
        """Replay one captured result graph against updated rollout buffers."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=32,
            use_cuda_graph=True,
        )
        config.use_lm_seed = False
        solver = IKSolver(config)
        tool_frames = solver.kinematics.tool_frames

        def solve_at_height(height: float) -> IKSolverResult:
            pose = Pose(
                position=torch.tensor(
                    [[0.4, 0.0, height]], **cuda_device_cfg.as_torch_dict()
                ),
                quaternion=torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict()
                ),
            )
            goal = GoalToolPose.from_poses(
                {tool_frames[0]: pose.unsqueeze(1)},
                ordered_tool_frames=tool_frames,
            )
            return solver.solve_pose(goal_tool_poses=goal, return_seeds=1)

        first_result = solve_at_height(0.4)
        first_solution = first_result.solution.clone()
        result_graph = solver._result_processor._result_graph
        assert solver._result_processor.is_cuda_graph_initialized

        second_result = solve_at_height(0.5)
        assert solver._result_processor._result_graph is result_graph
        assert second_result.success[0, 0]
        assert not torch.equal(second_result.solution, first_solution)

        solver._result_processor.reset_cuda_graph()
        assert not solver._result_processor.is_cuda_graph_initialized


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
    def test_solve_pose_does_not_double_count_lm_solve_time(
        self,
        ik_solver: IKSolver,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keep LM and main optimizer times separate from enclosing total time."""

        class FakeCudaEventTimer:
            def start(self) -> "FakeCudaEventTimer":
                return self

            def stop(self) -> float:
                return 5.0

        class FakeSeedIKSolver:
            def solve_seed_configs(self, **kwargs: object) -> IKSolverResult:
                return seed_result

        current_state = ik_solver.default_joint_state.clone().unsqueeze(0)
        goal_tool_poses = ik_solver.compute_kinematics(current_state).tool_poses.as_goal()
        seed_position = current_state.position.view(1, 1, -1).expand(
            1,
            ik_solver.config.num_seeds,
            -1,
        ).clone()
        seed_result = IKSolverResult(
            success=torch.ones(
                (1, ik_solver.config.num_seeds),
                device=current_state.device,
                dtype=torch.bool,
            ),
            js_solution=JointState.from_position(
                seed_position,
                joint_names=ik_solver.joint_names,
            ),
            solve_time=1.25,
            total_time=2.0,
        )
        main_result = IKSolverResult(
            success=torch.ones((1, 1), device=current_state.device, dtype=torch.bool),
            solve_time=2.5,
            total_time=4.0,
        )

        def fake_solve_impl(**kwargs: object) -> IKSolverResult:
            return main_result

        monkeypatch.setattr(solver_ik_module, "CudaEventTimer", FakeCudaEventTimer)
        monkeypatch.setattr(ik_solver.config, "use_lm_seed", True)
        monkeypatch.setattr(ik_solver, "seed_ik_solver", FakeSeedIKSolver())
        monkeypatch.setattr(ik_solver, "_solve_impl", fake_solve_impl)

        result = ik_solver.solve_pose(
            goal_tool_poses,
            current_state=current_state,
        )

        assert result.total_time == pytest.approx(5.0)
        assert result.solve_time == pytest.approx(3.75)

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_defers_optimizer_rollout_update(
        self,
        ik_solver: IKSolver,
        sample_goal_pose: Pose,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Skip optimizer rollout propagation when seed validation returns immediately."""
        update_calls = 0
        update_rollout_params = ik_solver.optimizer.update_rollout_params

        def count_update_calls(goal: object) -> None:
            nonlocal update_calls
            update_calls += 1
            update_rollout_params(goal)

        monkeypatch.setattr(
            ik_solver.optimizer,
            "update_rollout_params",
            count_update_calls,
        )
        tool_frames = ik_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        ik_solver.solve_pose(goal_tool_poses=goal_tool_poses, run_optimizer=False)

        assert update_calls == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_solve_pose_updates_optimizer_rollouts_before_fallback(
        self,
        ik_solver: IKSolver,
        sample_goal_pose: Pose,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Propagate rollout goals before running the fallback optimizer."""
        update_calls = 0
        update_rollout_params = ik_solver.optimizer.update_rollout_params

        def count_update_calls(goal: object) -> None:
            nonlocal update_calls
            update_calls += 1
            update_rollout_params(goal)

        monkeypatch.setattr(ik_solver.config, "exit_early", False)
        monkeypatch.setattr(
            ik_solver.optimizer,
            "update_rollout_params",
            count_update_calls,
        )
        tool_frames = ik_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        ik_solver.solve_pose(goal_tool_poses=goal_tool_poses)

        assert update_calls == 1


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_seed_restores_lm_and_main_seed_streams(
        self, cuda_device_cfg: DeviceCfg
    ) -> None:
        """Restore the first sequential slice in both IK seed generators."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=4,
            seed_solver_num_seeds=4,
            use_cuda_graph=False,
        )
        solver = IKSolver(config)

        solver.reset_seed()
        lm_first = solver.seed_ik_solver._generate_seed_configs(batch_size=1).clone()
        main_first = solver.prepare_action_seeds(batch_size=1, num_seeds=4).clone()

        lm_next = solver.seed_ik_solver._generate_seed_configs(batch_size=1)
        main_next = solver.prepare_action_seeds(batch_size=1, num_seeds=4)
        assert not torch.equal(lm_next, lm_first)
        assert not torch.equal(main_next, main_first)

        solver.reset_seed()
        lm_reset = solver.seed_ik_solver._generate_seed_configs(batch_size=1)
        main_reset = solver.prepare_action_seeds(batch_size=1, num_seeds=4)

        torch.testing.assert_close(lm_reset, lm_first, rtol=0.0, atol=0.0)
        torch.testing.assert_close(main_reset, main_first, rtol=0.0, atol=0.0)


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
