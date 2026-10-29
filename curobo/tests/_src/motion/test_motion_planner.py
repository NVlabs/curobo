# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for MotionPlanner class."""

# Standard Library
from dataclasses import replace
from inspect import signature
from types import SimpleNamespace
from typing import Any, Callable

# Third Party
import pytest
import torch

from curobo._src.cost.tool_pose_criteria import PoseErrorFrame, ToolPoseCriteria
from curobo._src.motion.motion_planner import MotionPlanner, _fit_linear_seed_trajectories
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.motion.planner_grasp import (
    GraspPhasePlanner,
    get_result_goal_configuration,
)
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.trajectory.trajectory_seed_fitter import (
    BSplineTimeOptimalSeedFitter,
)
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose


def test_default_refinement_budget() -> None:
    """Default to 50 initial iterations and one 25-iteration graph-sized refinement."""
    from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
    from curobo._src.solver.solver_trajopt import TrajOptSolver
    from curobo._src.util.config_io import join_path, load_yaml
    from curobo.content import get_task_configs_path

    config = load_yaml(join_path(get_task_configs_path(), "trajopt/lbfgs_bspline_trajopt.yml"))
    assert config["optimizer"]["num_iters"] == 50
    assert config["optimizer"]["inner_iters"] == 25
    assert config["optimizer"]["fixed_iters"] is True
    for method in (
        MotionPlanner.plan_pose,
        BatchMotionPlanner.plan_pose,
        MotionPlanner.plan_cspace,
        BatchMotionPlanner.plan_cspace,
        TrajOptSolver.solve_pose,
        TrajOptSolver.solve_cspace,
    ):
        assert signature(method).parameters["finetune_attempts"].default == 1
    for method in (
        BatchMotionPlanner.plan_pose,
        TrajOptSolver.solve_pose,
        TrajOptSolver.solve_cspace,
    ):
        params = signature(method).parameters
        assert params["time_optimal_iters"].default == 25
        assert params["finetune_iters"].default == 25
        assert params["finetune_dt_scale"].default == 0.6
    for method in (TrajOptSolver.solve_pose, TrajOptSolver.solve_cspace):
        assert signature(method).parameters["initial_dt_scale"].default == 1.25


@pytest.fixture(scope="module")
def cuda_device_cfg():
    """Create a CUDA device configuration for testing."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def motion_planner_config(cuda_device_cfg):
    """Create MotionPlannerCfg for Franka robot."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        device_cfg=cuda_device_cfg,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
        max_goalset=4,
    )
    return config


@pytest.fixture(scope="module")
def motion_planner_config_with_scene(cuda_device_cfg):
    """Create MotionPlannerCfg with scene collision for Franka robot."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
        device_cfg=cuda_device_cfg,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
        max_goalset=4,
    )
    return config


@pytest.fixture(scope="module")
def motion_planner(motion_planner_config):
    """Create MotionPlanner instance."""
    return MotionPlanner(motion_planner_config)


@pytest.fixture(scope="module")
def motion_planner_with_scene(motion_planner_config_with_scene):
    """Create MotionPlanner instance with scene collision."""
    return MotionPlanner(motion_planner_config_with_scene)


@pytest.fixture
def sample_start_state(motion_planner):
    """Create a sample start state from default joint position."""
    default_js = motion_planner.default_joint_state.clone()
    return JointState.from_position(
        default_js.position.unsqueeze(0),
        joint_names=motion_planner.joint_names,
    )


@pytest.fixture
def sample_goal_pose(cuda_device_cfg):
    """Create a sample goal pose for testing."""
    position = torch.tensor([[0.4, 0.0, 0.4]], **cuda_device_cfg.as_torch_dict())
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
    return Pose(position=position, quaternion=quaternion)


class TestMotionPlannerClassDefinition:
    """Test MotionPlanner class definition."""

    def test_class_exists(self):
        """Test MotionPlanner class exists."""
        assert MotionPlanner is not None

    def test_init_method_exists(self):
        """Test __init__ method exists."""
        assert hasattr(MotionPlanner, "__init__")

    def test_plan_pose_method_exists(self):
        """Test plan_pose method exists."""
        assert hasattr(MotionPlanner, "plan_pose")

    def test_plan_cspace_method_exists(self):
        """Test plan_cspace method exists."""
        assert hasattr(MotionPlanner, "plan_cspace")

    def test_plan_grasp_method_exists(self):
        """Test plan_grasp method exists."""
        assert hasattr(MotionPlanner, "plan_grasp")

    def test_warmup_method_exists(self):
        """Test warmup method exists."""
        assert hasattr(MotionPlanner, "warmup")

    def test_compute_kinematics_method_exists(self):
        """Test compute_kinematics method exists."""
        assert hasattr(MotionPlanner, "compute_kinematics")


class TestMotionPlannerInitialization:
    """Test MotionPlanner initialization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_with_config(self, motion_planner_config):
        """Test MotionPlanner initializes with config."""
        planner = MotionPlanner(motion_planner_config)
        assert planner is not None
        assert planner.config == motion_planner_config

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_creates_ik_solver(self, motion_planner):
        """Test MotionPlanner creates IK solver."""
        assert hasattr(motion_planner, "ik_solver")
        assert motion_planner.ik_solver is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_creates_trajopt_solver(self, motion_planner):
        """Test MotionPlanner creates TrajOpt solver."""
        assert hasattr(motion_planner, "trajopt_solver")
        assert motion_planner.trajopt_solver is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_creates_graph_planner(self, motion_planner):
        """Test MotionPlanner creates graph planner."""
        assert hasattr(motion_planner, "graph_planner")
        assert motion_planner.graph_planner is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_creates_uniform_time_seed_fitter(self, motion_planner):
        """Use the fixed-iteration IRLS fitter for B-spline seeds."""
        assert isinstance(motion_planner.graph_seed_fitter, BSplineTimeOptimalSeedFitter)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_without_scene_no_collision_checker(self, motion_planner):
        """Test MotionPlanner without scene has no scene collision checker."""
        assert motion_planner.scene_collision_checker is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_with_scene_creates_collision_checker(self, motion_planner_with_scene):
        """Test MotionPlanner with scene creates collision checker."""
        assert motion_planner_with_scene.scene_collision_checker is not None


class TestMotionPlannerProperties:
    """Test MotionPlanner properties."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_attachment_manager_property(self, motion_planner: MotionPlanner) -> None:
        """Test the exposed attachment manager updates shared solver spheres."""
        manager = motion_planner.attachment_manager
        assert manager is motion_planner.trajopt_solver.core.attachment_manager
        assert (
            manager.kinematics_params
            is motion_planner.ik_solver.core.attachment_manager.kinematics_params
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_joint_names_property(self, motion_planner):
        """Test joint_names property returns list of names."""
        names = motion_planner.joint_names
        assert isinstance(names, list)
        assert len(names) > 0
        # Franka has 7 DOF
        assert len(names) == 7

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tool_frames_property(self, motion_planner):
        """Test tool_frames property returns list."""
        names = motion_planner.tool_frames
        assert isinstance(names, list)
        assert len(names) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_joint_state_property(self, motion_planner):
        """Test default_joint_state property returns JointState."""
        default_js = motion_planner.default_joint_state
        assert isinstance(default_js, JointState)
        assert default_js.position is not None
        assert default_js.position.shape[-1] == 7  # Franka DOF

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_kinematics_property(self, motion_planner):
        """Test kinematics property exists."""
        assert hasattr(motion_planner, "kinematics")
        assert motion_planner.kinematics is not None


class TestMotionPlannerComputeKinematics:
    """Test MotionPlanner.compute_kinematics method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compute_kinematics_returns_result(self, motion_planner, sample_start_state):
        """Test compute_kinematics returns valid result."""
        result = motion_planner.compute_kinematics(sample_start_state)

        assert result is not None
        assert hasattr(result, "tool_poses")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compute_kinematics_link_poses_has_position(self, motion_planner, sample_start_state):
        """Test compute_kinematics link_poses has position."""
        result = motion_planner.compute_kinematics(sample_start_state)

        # link_poses should have position data
        link_poses_dict = result.tool_poses.to_dict()
        for link_name, pose in link_poses_dict.items():
            assert pose.position is not None


class TestMotionPlannerWarmup:
    """Test MotionPlanner.warmup method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_warmup_no_error(self, motion_planner):
        """Test warmup doesn't raise error."""
        # Should not raise
        result = motion_planner.warmup(
            enable_graph=True,
            num_warmup_iterations=2,
        )
        assert result is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_warmup_without_graph(self, motion_planner):
        """Test warmup without graph planner."""
        result = motion_planner.warmup(
            enable_graph=False,
            num_warmup_iterations=2,
        )
        assert result is True


class TestMotionPlannerPlanSinglePose:
    """Test MotionPlanner.plan_pose method."""

    def test_linear_fitter_preserves_moving_start_state(self) -> None:
        """Forward velocity and acceleration to every fitted IK lane."""

        class CapturingFitter:
            config = SimpleNamespace(rollout_horizon=5)

            def __init__(self) -> None:
                self.start_state: JointState | None = None

            def fit(
                self,
                target_position: torch.Tensor,
                start_state: JointState,
                goal_position: torch.Tensor,
            ) -> torch.Tensor:
                self.start_state = start_state
                return target_position[:, :4].clone()

        start_state = JointState(
            position=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            velocity=torch.tensor([[0.2, -0.3]], dtype=torch.float32),
            acceleration=torch.tensor([[0.4, -0.5]], dtype=torch.float32),
            jerk=torch.zeros((1, 2), dtype=torch.float32),
        )
        goals = torch.tensor(
            [[[1.0, 2.0], [2.0, 3.0]]],
            dtype=torch.float32,
        )
        fitter = CapturingFitter()

        fitted = _fit_linear_seed_trajectories(
            start_state,
            goals,
            action_horizon=4,
            seed_fitter=fitter,
        )

        assert fitted.shape == (1, 2, 4, 2)
        assert fitter.start_state is not None
        torch.testing.assert_close(
            fitter.start_state.velocity,
            start_state.velocity.repeat_interleave(2, dim=0),
        )
        torch.testing.assert_close(
            fitter.start_state.acceleration,
            start_state.acceleration.repeat_interleave(2, dim=0),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_returns_result(self, motion_planner, sample_start_state, sample_goal_pose):
        """Test plan_pose returns TrajOptSolverResult."""
        # Get tool frame and create goal
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose},
            ordered_tool_frames=tool_frames,
        )

        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            max_attempts=1,
        )

        # Result may be None if planning fails, or TrajOptSolverResult
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_result_has_success(
        self, motion_planner, sample_start_state, sample_goal_pose
    ):
        """Test plan_pose result has success field."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose},
            ordered_tool_frames=tool_frames,
        )

        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            max_attempts=2,
        )

        if result is not None:
            assert hasattr(result, "success")
            assert isinstance(result.success, torch.Tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_result_has_timing(
        self, motion_planner, sample_start_state, sample_goal_pose
    ):
        """Test plan_pose result has timing information."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose},
            ordered_tool_frames=tool_frames,
        )

        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            max_attempts=2,
        )

        if result is not None:
            assert hasattr(result, "total_time")
            assert hasattr(result, "solve_time")
            assert result.total_time >= 0.0
            assert result.solve_time >= 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_with_goal_tool_pose(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_pose accepts GoalToolPose input."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_links = len(tool_frames)

        # 5D GoalToolPose: [B, H=1, L, G=1, 3/4]
        position = torch.zeros((1, 1, num_links, 1, 3), **cuda_device_cfg.as_torch_dict())
        position[0, 0, 0, 0, :] = torch.tensor([0.4, 0.0, 0.4], **cuda_device_cfg.as_torch_dict())

        quaternion = torch.zeros((1, 1, num_links, 1, 4), **cuda_device_cfg.as_torch_dict())
        quaternion[0, 0, 0, 0, :] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
        )

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position,
            quaternion=quaternion,
        )

        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_pose,
            max_attempts=1,
        )

        # Should not raise, result may be None or TrajOptSolverResult
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_with_implicit_goal(
        self, motion_planner, sample_start_state, sample_goal_pose
    ):
        """Test plan_pose with use_implicit_goal=True."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose},
            ordered_tool_frames=tool_frames,
        )

        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            use_implicit_goal=True,
            max_attempts=1,
        )

        assert result is None or isinstance(result, TrajOptSolverResult)


class TestMotionPlannerResetSeed:
    """Test MotionPlanner.reset_seed method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_seed_no_error(self, motion_planner):
        """Test reset_seed doesn't raise error."""
        # Should not raise
        motion_planner.reset_seed()


class TestMotionPlannerWithSceneCollision:
    """Test MotionPlanner with scene collision."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_with_scene_collision(self, motion_planner_with_scene, sample_goal_pose):
        """Test planning with scene collision works."""
        default_js = motion_planner_with_scene.default_joint_state.clone()
        start_state = JointState.from_position(
            default_js.position.unsqueeze(0),
            joint_names=motion_planner_with_scene.joint_names,
        )

        tool_frames = motion_planner_with_scene.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose},
            ordered_tool_frames=tool_frames,
        )

        result = motion_planner_with_scene.plan_pose(
            current_state=start_state,
            goal_tool_poses=goal_tool_poses,
            max_attempts=1,
        )

        # Should not raise, result may be None or TrajOptSolverResult
        assert result is None or isinstance(result, TrajOptSolverResult)


class TestMotionPlannerPlanGrasp:
    """Test MotionPlanner.plan_grasp method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_returns_grasp_plan_result(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_grasp returns GraspPlanResult."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        # Create goalset of grasp poses: (batch=1, num_goalset, dim)
        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4 + i * 0.05, 0.0, 0.3], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        # 5D GoalToolPose: [B, H=1, L=1, G, 3/4]
        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
            plan_approach_to_grasp=False,
            plan_grasp_to_lift=False,
        )

        assert isinstance(result, GraspPlanResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_result_has_status(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_grasp result has status field."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 1

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        position[0, 0, :] = torch.tensor([0.4, 0.0, 0.3], **cuda_device_cfg.as_torch_dict())
        quaternion[0, 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict())

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
            plan_approach_to_grasp=False,
            plan_grasp_to_lift=False,
        )

        # Result should have status field (may be None if successful)
        assert hasattr(result, "status")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_result_has_goalset_result(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_grasp result has goalset_result field."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4, i * 0.1, 0.35], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
            plan_approach_to_grasp=False,
            plan_grasp_to_lift=False,
        )

        # Result should have goalset_result from plan_pose
        assert hasattr(result, "goalset_result")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_with_custom_approach_offset(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_grasp with custom approach offset."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 1

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        position[0, 0, :] = torch.tensor([0.4, 0.0, 0.35], **cuda_device_cfg.as_torch_dict())
        quaternion[0, 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict())

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
            grasp_approach_axis="z",
            grasp_approach_offset=-0.1,
            plan_approach_to_grasp=False,
            plan_grasp_to_lift=False,
        )

        assert isinstance(result, GraspPlanResult)


class TestMotionPlannerGraspUpdateCache:
    """Test fixed-address grasp criteria and collision preset reuse."""

    @staticmethod
    def _make_phase_planner(
        criteria_updates: list[dict[str, ToolPoseCriteria]],
        collision_updates: list[tuple[str, tuple[str, ...]]],
        *,
        constrained_rrt_planner: object | None = None,
        ik_solver: object | None = None,
        plan_pose: Callable[..., object] | None = None,
    ) -> GraspPhasePlanner:
        """Create one CPU-only grasp-phase planner for cache tests."""
        config = SimpleNamespace(
            device_cfg=DeviceCfg(device=torch.device("cpu")),
            trajopt_solver_config=SimpleNamespace(use_cuda_graph=False),
            ik_solver_config=SimpleNamespace(
                position_tolerance=0.005,
                orientation_tolerance=0.05,
                multi_env=False,
            ),
        )
        return GraspPhasePlanner(
            config=config,
            batch_size=1,
            kinematics=SimpleNamespace(),
            ik_solver=ik_solver or SimpleNamespace(),
            trajopt_solver=SimpleNamespace(config=SimpleNamespace(num_seeds=1)),
            constrained_rrt_planner=constrained_rrt_planner,
            constrained_rrt_seed_fitter=None,
            plan_pose=plan_pose or (lambda *_args, **_kwargs: None),
            update_tool_pose_criteria=lambda criteria: criteria_updates.append(criteria),
            disable_link_collision=lambda links: collision_updates.append(
                ("disable", tuple(links))
            ),
            enable_link_collision=lambda links: collision_updates.append(("enable", tuple(links))),
        )

    def test_grasp_updates_reuse_fixed_preset_executors(self) -> None:
        """Reuse one executor for each criteria and collision preset."""
        criteria_updates: list[dict[str, ToolPoseCriteria]] = []
        collision_updates: list[tuple[str, tuple[str, ...]]] = []
        phase_planner = self._make_phase_planner(criteria_updates, collision_updates)

        phase_planner.activate_tool_pose_criteria(
            ["ee_link"],
            axis="x",
            in_tool_frame=False,
        )
        criteria_key = ("tool_pose", ("ee_link",), "x", False)
        criteria_executor = phase_planner._update_executors[criteria_key]
        criteria = criteria_updates[-1]["ee_link"]
        criteria_update_count = len(criteria_updates)
        phase_planner.activate_tool_pose_criteria(
            ["ee_link"],
            axis="x",
            in_tool_frame=False,
        )

        assert phase_planner._update_executors[criteria_key] is criteria_executor
        assert len(criteria_updates) == criteria_update_count
        assert criteria_updates[-1]["ee_link"] is criteria
        torch.testing.assert_close(
            criteria.path_position_projector,
            torch.diag(torch.tensor([0.0, 1.0, 1.0])),
        )
        assert criteria.error_frame.item() == PoseErrorFrame.WORLD
        torch.testing.assert_close(
            criteria.path_tolerance,
            torch.tensor([0.005, 0.05]),
        )

        phase_planner.activate_collision(["ee_link"], enable=False)
        collision_key = ("collision", ("ee_link",), False)
        collision_executor = phase_planner._update_executors[collision_key]
        phase_planner.activate_collision(["ee_link"], enable=False)

        assert phase_planner._update_executors[collision_key] is collision_executor
        assert collision_updates[-1] == ("disable", ("ee_link",))
        phase_planner.destroy()

    def test_grasp_linear_phase_keeps_trajopt_fallback(self) -> None:
        """Fall back to TrajOpt after a projected-RRT attempt fails."""
        endpoint_calls: list[bool] = []
        rrt_calls: list[bool] = []
        criteria_calls: list[dict[str, ToolPoseCriteria]] = []
        collision_calls: list[tuple[str, tuple[str, ...]]] = []
        trajopt_result = SimpleNamespace(success=torch.tensor([True]))

        def solve_goalset(*_args: object, **_kwargs: object) -> SimpleNamespace:
            endpoint_calls.append(True)
            return SimpleNamespace(
                solution=torch.zeros(1, 1, 7),
                success=torch.tensor([[True]]),
            )

        phase_planner = self._make_phase_planner(
            criteria_calls,
            collision_calls,
            constrained_rrt_planner=object(),
            ik_solver=SimpleNamespace(solve_goalset=solve_goalset),
            plan_pose=lambda *_args, **_kwargs: trajopt_result,
        )

        def plan_rrt(*_args: object) -> None:
            rrt_calls.append(True)
            return None

        phase_planner._plan_rrt_phase = plan_rrt
        result = phase_planner.plan_linear_phase(
            SimpleNamespace(tool_frames=["ee_link"]),
            SimpleNamespace(),
            axis="z",
            in_tool_frame=True,
            use_rrt=True,
            finetune_attempts=0,
        )

        assert result is trajopt_result
        assert len(endpoint_calls) == 1
        assert len(rrt_calls) == 1
        assert ("tool_pose", ("ee_link",), "z", True) in phase_planner._update_executors
        assert ("tool_pose", ("ee_link",), None, True) in phase_planner._update_executors
        phase_planner.destroy()


class TestMotionPlannerUpdateMethods:
    """Test MotionPlanner update methods exist."""

    def test_update_world_method_exists(self):
        """Test update_world method exists."""
        assert hasattr(MotionPlanner, "update_world")

    def test_clear_scene_cache_method_exists(self):
        """Test clear_scene_cache method exists."""
        assert hasattr(MotionPlanner, "clear_scene_cache")

    def test_enable_link_collision_method_exists(self):
        """Test enable_link_collision method exists."""
        assert hasattr(MotionPlanner, "enable_link_collision")
        assert callable(MotionPlanner.enable_link_collision)

    def test_disable_link_collision_method_exists(self):
        """Test disable_link_collision method exists."""
        assert hasattr(MotionPlanner, "disable_link_collision")
        assert callable(MotionPlanner.disable_link_collision)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_enable_disable_link_collision(self, motion_planner):
        """Test enable/disable link collision methods work."""
        # Get a link name from the target links
        link_names = motion_planner.tool_frames
        if len(link_names) > 0:
            # Should not raise
            motion_planner.disable_link_collision([link_names[0]])
            motion_planner.enable_link_collision([link_names[0]])

    def test_update_link_inertial_method_exists(self):
        """Test update_link_inertial method exists."""
        assert hasattr(MotionPlanner, "update_link_inertial")

    def test_update_links_inertial_method_exists(self):
        """Test update_links_inertial method exists."""
        assert hasattr(MotionPlanner, "update_links_inertial")


class TestMotionPlannerPlanSingleGoalset:
    """Test MotionPlanner.plan_pose method."""

    def test_plan_pose_reuses_ik_once_for_graph_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reuse direct-attempt IK for RRT, then refresh it after RRT failure."""

        class CapturingIkSolver:
            def __init__(self) -> None:
                self.calls = 0

            def solve_pose(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
                self.calls += 1
                return SimpleNamespace(
                    solution=torch.ones((1, 2, 2), dtype=torch.float32),
                    success=torch.ones((1, 2), dtype=torch.bool),
                    total_time=0.1,
                    solve_time=0.05,
                )

        class CapturingTrajoptSolver:
            def __init__(self) -> None:
                self.config = SimpleNamespace(num_seeds=2)
                self.action_horizon = 4
                self.calls = 0

            def solve_pose(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
                self.calls += 1
                return SimpleNamespace(
                    success=torch.tensor([[self.calls == 3]], dtype=torch.bool),
                    total_time=0.2,
                    solve_time=0.15,
                    debug_info={},
                    trajectory_source=None,
                )

        planner = object.__new__(MotionPlanner)
        planner._destroyed = True
        planner._warming_up = False
        planner.config = SimpleNamespace()
        planner.ik_solver = CapturingIkSolver()
        planner.trajopt_solver = CapturingTrajoptSolver()
        planner.graph_planner = object()
        planner.graph_seed_fitter = None
        graph_calls = []

        def graph_seed(*args: Any, **kwargs: Any) -> torch.Tensor:
            graph_calls.append(None)
            return torch.ones((1, 2, 4, 2), dtype=torch.float32)

        planner._get_graph_seed_trajectories = graph_seed
        monkeypatch.setattr(
            "curobo._src.motion.motion_planner._fit_linear_seed_trajectories",
            lambda *args, **kwargs: torch.zeros((1, 2, 4, 2), dtype=torch.float32),
        )
        monkeypatch.setattr(
            "curobo._src.motion.motion_planner._should_evaluate_seed_before_trajopt",
            lambda *args, **kwargs: False,
        )

        result = planner._plan_pose_single(
            object(),
            JointState.from_position(torch.zeros((1, 2), dtype=torch.float32)),
            use_implicit_goal=True,
            max_attempts=3,
            enable_graph_attempt=1,
        )

        assert result is not None
        assert bool(result.success.item())
        assert planner.ik_solver.calls == 2
        assert planner.trajopt_solver.calls == 3
        assert len(graph_calls) == 2
        assert planner.last_planning_attempts == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_method_callable(self, motion_planner):
        """Test plan_pose method is callable."""
        assert hasattr(motion_planner, "plan_pose")
        assert callable(motion_planner.plan_pose)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_with_dict_input(self, motion_planner, sample_start_state, sample_goal_pose):
        """Test plan_pose with GoalToolPose built via from_poses."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose},
            ordered_tool_frames=tool_frames,
        )

        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            max_attempts=1,
        )

        # Result may be None or TrajOptSolverResult
        assert result is None or isinstance(result, TrajOptSolverResult)


class TestMotionPlannerPlanCspace:
    """Test MotionPlanner.plan_cspace method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_method_callable(self, motion_planner):
        """Test plan_cspace method is callable."""
        assert hasattr(motion_planner, "plan_cspace")
        assert callable(motion_planner.plan_cspace)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_returns_result(self, motion_planner, sample_start_state):
        """Test plan_cspace returns TrajOptSolverResult."""
        # Create a goal state slightly different from start
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.1  # Move first joint

        result = motion_planner.plan_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
            max_attempts=1,
        )

        # Result may be None if planning fails, or TrajOptSolverResult
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_returns_valid_fitted_seed_without_trajopt(
        self,
        motion_planner_config,
    ):
        """Return a retimed, densely validated linear fit when refinement is optional."""
        config = replace(motion_planner_config, always_run_trajopt=False)
        with MotionPlanner(config) as planner:
            start_state = planner.default_joint_state.clone().unsqueeze(0)
            goal_state = start_state.clone()
            goal_state.position[..., 0] += 0.1

            result = planner.plan_cspace(
                current_state=start_state,
                goal_state=goal_state,
                max_attempts=1,
                enable_graph_attempt=1,
            )

        assert result is not None
        assert bool(result.success.item())
        assert result.trajectory_source == "linear_fit"
        assert result.debug_info["trajectory_source"] == "linear_fit"
        assert result.solve_time == 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_result_has_success(self, motion_planner, sample_start_state):
        """Test plan_cspace result has success field."""
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.1

        result = motion_planner.plan_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
            max_attempts=2,
        )

        if result is not None:
            assert hasattr(result, "success")
            assert isinstance(result.success, torch.Tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_result_has_timing(self, motion_planner, sample_start_state):
        """Test plan_cspace result has timing information."""
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.1

        result = motion_planner.plan_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
            max_attempts=2,
        )

        if result is not None:
            assert hasattr(result, "total_time")
            assert hasattr(result, "solve_time")
            assert result.total_time >= 0.0
            assert result.solve_time >= 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_with_graph_planner(self, motion_planner, sample_start_state):
        """Test plan_cspace with graph planner enabled."""
        goal_state = sample_start_state.clone()
        goal_state.position[..., 0] += 0.2  # Larger movement

        result = motion_planner.plan_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
            max_attempts=3,
            enable_graph_attempt=1,  # Enable graph on 2nd attempt
        )

        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_same_start_and_goal(self, motion_planner, sample_start_state):
        """Test plan_cspace when goal is same as start."""
        goal_state = sample_start_state.clone()

        result = motion_planner.plan_cspace(
            current_state=sample_start_state,
            goal_state=goal_state,
            max_attempts=1,
        )

        # Should succeed easily when start == goal
        assert result is None or isinstance(result, TrajOptSolverResult)


class TestMotionPlannerUpdateWorldAndCache:
    """Test MotionPlanner update_world and clear_scene_cache with scene collision."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_world_with_scene(self, motion_planner_with_scene):
        """Test update_world updates collision checker."""
        from curobo._src.geom.types import Cuboid, SceneCfg

        # Clear existing scene first to make room in cache
        motion_planner_with_scene.clear_scene_cache()

        # Create a new scene with a cuboid
        cuboid = Cuboid(
            name="test_box",
            pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
            dims=[0.1, 0.1, 0.1],
        )
        new_scene = SceneCfg(cuboid=[cuboid])

        # Should not raise
        motion_planner_with_scene.update_world(new_scene)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_clear_scene_cache_with_scene(self, motion_planner_with_scene):
        """Test clear_scene_cache clears collision cache."""
        # Should not raise
        motion_planner_with_scene.clear_scene_cache()


class TestMotionPlannerUpdateInertial:
    """Test MotionPlanner inertial update methods.

    Note: These methods require inverse dynamics to be configured.
    Standard configs don't have dynamics, so we test method existence
    and signature rather than execution.
    """

    def test_update_link_inertial_method_signature(self):
        """Test update_link_inertial method signature."""
        import inspect

        sig = inspect.signature(MotionPlanner.update_link_inertial)
        params = list(sig.parameters.keys())
        assert "link_name" in params
        assert "mass" in params
        assert "com" in params
        assert "inertia" in params

    def test_update_links_inertial_method_signature(self):
        """Test update_links_inertial method signature."""
        import inspect

        sig = inspect.signature(MotionPlanner.update_links_inertial)
        params = list(sig.parameters.keys())
        assert "link_properties" in params

    def test_updates_are_propagated_to_rrt_planners(self):
        """Keep IK, RRT, constrained RRT, and TrajOpt inertial models aligned."""

        class CapturingComponent:
            def __init__(self) -> None:
                self.single_calls = []
                self.batch_calls = []

            def update_link_inertial(self, link_name, mass, com, inertia) -> None:
                self.single_calls.append((link_name, mass, com, inertia))

            def update_links_inertial(self, link_properties) -> None:
                self.batch_calls.append(link_properties)

        planner = object.__new__(MotionPlanner)
        planner._destroyed = True
        planner.config = SimpleNamespace(graph_planner_algorithm="rrt")
        planner.ik_solver = CapturingComponent()
        planner.graph_planner = CapturingComponent()
        planner.constrained_rrt_planner = CapturingComponent()
        planner.trajopt_solver = CapturingComponent()

        planner.update_link_inertial("attached_object", mass=3.0)
        properties = {"attached_object": {"mass": 2.0}}
        planner.update_links_inertial(properties)

        for component in (
            planner.ik_solver,
            planner.graph_planner,
            planner.constrained_rrt_planner,
            planner.trajopt_solver,
        ):
            assert component.single_calls == [("attached_object", 3.0, None, None)]
            assert component.batch_calls == [properties]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_link_inertial_raises_without_dynamics(self, motion_planner):
        """Test update_link_inertial raises error without dynamics config."""
        link_names = motion_planner.tool_frames
        if len(link_names) > 0:
            link_name = link_names[0]
            # Should raise ValueError because dynamics is None
            with pytest.raises(ValueError, match="inverse dynamics"):
                motion_planner.update_link_inertial(link_name, mass=2.0)


class TestMotionPlannerMultipleAttempts:
    """Test MotionPlanner with multiple planning attempts."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_multiple_attempts_uses_graph(
        self, motion_planner, sample_start_state, sample_goal_pose
    ):
        """Test plan_pose with multiple attempts triggers graph planner."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose},
            ordered_tool_frames=tool_frames,
        )

        # Use max_attempts > enable_graph_attempt to trigger graph planning
        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            max_attempts=3,
            enable_graph_attempt=1,  # Enable graph on 2nd attempt
        )

        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_with_difficult_goal(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test planning with a difficult goal that may need multiple attempts."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames

        # Create a goal that might be harder to reach
        position = torch.tensor([[0.6, 0.3, 0.2]], **cuda_device_cfg.as_torch_dict())
        quaternion = torch.tensor([[0.707, 0.707, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
        goal_pose = Pose(position=position, quaternion=quaternion)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal_pose},
            ordered_tool_frames=tool_frames,
        )

        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            max_attempts=3,
        )

        # May succeed or fail, but should not raise
        assert result is None or isinstance(result, TrajOptSolverResult)


class TestMotionPlannerGoalsetWarmupAndPlanning:
    """Test warmup with num_goalset and subsequent single/goalset planning.

    Verifies that warming up with a large num_goalset properly allocates buffers
    so that later calls with smaller num_goalset (including num_goalset=1 for
    plan_pose) work via padding without shape mismatch errors.
    """

    @pytest.fixture(scope="class")
    def goalset_planner(self, cuda_device_cfg):
        """Create a fresh MotionPlanner with max_goalset=4 and warmup."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_ik_seeds=16,
            num_trajopt_seeds=2,
            use_cuda_graph=True,
            max_goalset=4,
        )
        planner = MotionPlanner(config)
        planner.warmup(enable_graph=True, num_warmup_iterations=2)
        return planner

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_warmup_with_goalset_succeeds(self, cuda_device_cfg):
        """Test warmup with max_goalset > 1 completes without error."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_ik_seeds=16,
            num_trajopt_seeds=2,
            use_cuda_graph=False,
            max_goalset=4,
        )
        planner = MotionPlanner(config)
        result = planner.warmup(enable_graph=True, num_warmup_iterations=2)
        assert result is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_single_goal_after_goalset_warmup(
        self,
        goalset_planner,
        cuda_device_cfg,
    ):
        """Test plan_pose (num_goalset=1) works after goalset warmup."""
        planner = goalset_planner
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0),
            joint_names=planner.joint_names,
        )
        tool_frames = planner.trajopt_solver.kinematics.tool_frames
        position = torch.tensor([[0.4, 0.0, 0.4]], **cuda_device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: Pose(position=position, quaternion=quaternion)},
            ordered_tool_frames=tool_frames,
        )

        result = planner.plan_pose(
            current_state=start,
            goal_tool_poses=goal_tool_poses,
            max_attempts=1,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_after_goalset_warmup(self, goalset_planner, cuda_device_cfg):
        """Test plan_pose works after goalset warmup."""
        planner = goalset_planner
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0),
            joint_names=planner.joint_names,
        )
        tool_frames = planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 3

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4 + i * 0.05, 0.0, 0.4], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = planner.plan_pose(
            current_state=start,
            goal_tool_poses=goal_tool_pose,
            max_attempts=1,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_then_goalset_then_pose(self, goalset_planner, cuda_device_cfg):
        """Test alternating between plan_pose and plan_pose after warmup."""
        planner = goalset_planner
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0),
            joint_names=planner.joint_names,
        )
        tool_frames = planner.trajopt_solver.kinematics.tool_frames

        # plan_pose (num_goalset=1)
        pos1 = torch.tensor([[0.4, 0.0, 0.4]], **cuda_device_cfg.as_torch_dict())
        quat1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
        goal_tool_poses1 = GoalToolPose.from_poses(
            {tool_frames[0]: Pose(position=pos1, quaternion=quat1)},
            ordered_tool_frames=tool_frames,
        )
        r1 = planner.plan_pose(
            current_state=start,
            goal_tool_poses=goal_tool_poses1,
            max_attempts=1,
        )
        assert r1 is None or isinstance(r1, TrajOptSolverResult)

        # plan_pose (num_goalset=2) via GoalToolPose
        pos2 = torch.zeros((1, 2, 3), **cuda_device_cfg.as_torch_dict())
        quat2 = torch.zeros((1, 2, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(2):
            pos2[0, i, :] = torch.tensor([0.4, 0.05 * i, 0.4], **cuda_device_cfg.as_torch_dict())
            quat2[0, i, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict())
        goal_tool_pose2 = GoalToolPose(
            tool_frames=tool_frames,
            position=pos2.unsqueeze(1).unsqueeze(1),
            quaternion=quat2.unsqueeze(1).unsqueeze(1),
        )
        r2 = planner.plan_pose(
            current_state=start, goal_tool_poses=goal_tool_pose2, max_attempts=1
        )
        assert r2 is None or isinstance(r2, TrajOptSolverResult)

        # plan_pose again (num_goalset=1)
        r3 = planner.plan_pose(
            current_state=start,
            goal_tool_poses=goal_tool_poses1,
            max_attempts=1,
        )
        assert r3 is None or isinstance(r3, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_then_cspace_after_warmup(self, goalset_planner, cuda_device_cfg):
        """Test warmup -> plan_pose -> plan_cspace with CUDA graphs enabled."""
        planner = goalset_planner
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0),
            joint_names=planner.joint_names,
        )
        tool_frames = planner.trajopt_solver.kinematics.tool_frames

        pose_goal = Pose(
            position=torch.tensor([[0.4, 0.0, 0.4]], **cuda_device_cfg.as_torch_dict()),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict()),
        )
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: pose_goal},
            ordered_tool_frames=tool_frames,
        )

        pose_result = planner.plan_pose(
            current_state=start,
            goal_tool_poses=goal_tool_poses,
            max_attempts=1,
        )
        assert pose_result is None or isinstance(pose_result, TrajOptSolverResult)

        cspace_goal = start.clone()
        cspace_goal.position[..., 0] += 0.1
        cspace_result = planner.plan_cspace(
            current_state=start,
            goal_state=cspace_goal,
            max_attempts=1,
        )
        assert cspace_result is None or isinstance(cspace_result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_then_pose_after_warmup(self, goalset_planner, cuda_device_cfg):
        """Test warmup -> plan_cspace -> plan_pose with CUDA graphs enabled."""
        planner = goalset_planner
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0),
            joint_names=planner.joint_names,
        )
        tool_frames = planner.trajopt_solver.kinematics.tool_frames

        cspace_goal = start.clone()
        cspace_goal.position[..., 0] += 0.1
        cspace_result = planner.plan_cspace(
            current_state=start,
            goal_state=cspace_goal,
            max_attempts=1,
        )
        assert cspace_result is None or isinstance(cspace_result, TrajOptSolverResult)

        pose_goal = Pose(
            position=torch.tensor([[0.4, 0.0, 0.4]], **cuda_device_cfg.as_torch_dict()),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict()),
        )
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: pose_goal},
            ordered_tool_frames=tool_frames,
        )
        pose_result = planner.plan_pose(
            current_state=start,
            goal_tool_poses=goal_tool_poses,
            max_attempts=1,
        )
        assert pose_result is None or isinstance(pose_result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_after_goalset_warmup(self, goalset_planner, cuda_device_cfg):
        """Test plan_grasp works after goalset warmup.

        plan_grasp internally calls plan_pose (num_goalset>1) then plan_pose
        (num_goalset=1) for approach/grasp/lift, exercising both padding paths.
        """
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        planner = goalset_planner
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0),
            joint_names=planner.joint_names,
        )
        tool_frames = planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 3

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4 + i * 0.02, 0.0, 0.35], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = planner.plan_grasp(
            current_state=start,
            grasp_poses=goal_tool_pose,
            grasp_approach_axis="z",
            grasp_approach_offset=-0.1,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=False,
        )

        assert isinstance(result, GraspPlanResult)
        assert hasattr(result, "status")


class TestMotionPlannerIKFailurePaths:
    """Test MotionPlanner IK failure paths (lines 130, 135-136)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_ik_failure_unreachable_goal(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test planning with unreachable goal that causes IK failure (line 130)."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames

        # Create an unreachable goal (far outside workspace)
        position = torch.tensor([[5.0, 5.0, 5.0]], **cuda_device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
        unreachable_pose = Pose(position=position, quaternion=quaternion)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: unreachable_pose},
            ordered_tool_frames=tool_frames,
        )

        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            max_attempts=2,
            enable_graph_attempt=10,  # Don't trigger graph
        )

        # Should return None or unsuccessful result due to IK failure
        assert result is None or not result.success.any()
        assert motion_planner.last_graph_calls == 0
        assert motion_planner.last_graph_time == 0.0
        assert not motion_planner.last_graph_success
        assert motion_planner.last_graph_call_telemetry == []


class TestMotionPlannerGraphPlannerPath:
    """Test MotionPlanner graph planner path (lines 139-163)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_triggers_graph_on_first_attempt(
        self, motion_planner, sample_start_state, sample_goal_pose
    ):
        """Test graph planner is triggered when enable_graph_attempt=0."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: sample_goal_pose},
            ordered_tool_frames=tool_frames,
        )

        # Set enable_graph_attempt=0 to trigger graph on first attempt (line 138)
        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            max_attempts=2,
            enable_graph_attempt=0,  # Use graph from first attempt
        )

        # May succeed or fail
        assert result is None or isinstance(result, TrajOptSolverResult)
        assert motion_planner.last_graph_calls >= 1
        assert motion_planner.last_graph_time >= 0.0
        telemetry = motion_planner.last_graph_call_telemetry
        assert len(telemetry) == motion_planner.last_graph_calls
        assert sum(record.solve_time for record in telemetry) == pytest.approx(
            motion_planner.last_graph_time
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_graph_failure_continues(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test graph failure continues to next attempt (line 163)."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames

        # Goal at edge of workspace may cause graph to fail
        position = torch.tensor([[0.7, 0.5, 0.1]], **cuda_device_cfg.as_torch_dict())
        quaternion = torch.tensor([[0.5, 0.5, 0.5, 0.5]], **cuda_device_cfg.as_torch_dict())
        edge_pose = Pose(position=position, quaternion=quaternion)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: edge_pose},
            ordered_tool_frames=tool_frames,
        )

        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            max_attempts=3,
            enable_graph_attempt=0,
        )

        assert result is None or isinstance(result, TrajOptSolverResult)


class TestMotionPlannerGoalsetFailure:
    """Test MotionPlanner goalset failure path (line 200)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_returns_none_unreachable(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_pose returns None for unreachable goalset (line 200)."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        # Create unreachable goalset
        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [10.0 + i, 10.0, 10.0], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_pose,
            max_attempts=1,
        )

        # Should return None due to IK failure
        assert result is None


class TestMotionPlannerPlanGraspAdvanced:
    """Test MotionPlanner.plan_grasp advanced paths."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_with_dict_input(self, motion_planner, sample_start_state, cuda_device_cfg):
        """Test plan_grasp with GoalToolPose from from_poses (num_goalset=1)."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames

        position = torch.tensor([[0.4, 0.0, 0.35]], **cuda_device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())

        grasp_pose = Pose(position=position, quaternion=quaternion)
        grasp_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: grasp_pose},
            ordered_tool_frames=tool_frames,
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=grasp_tool_poses,
            plan_approach_to_grasp=False,
            plan_grasp_to_lift=False,
        )

        assert isinstance(result, GraspPlanResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_goalset_returns_none(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_grasp when goalset planning returns None (lines 261-263)."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        # Create unreachable grasp poses
        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        for i in range(num_goalset):
            position[0, i, :] = torch.tensor([10.0, 10.0, 10.0], **cuda_device_cfg.as_torch_dict())
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
        )

        assert isinstance(result, GraspPlanResult)
        assert "None" in result.status or "reachable" in result.status.lower()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_offset_in_world_frame(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_grasp with offset in world frame (not tool frame)."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4, i * 0.05, 0.35], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
            grasp_approach_in_tool_frame=False,
            plan_approach_to_grasp=False,
            plan_grasp_to_lift=False,
        )

        assert isinstance(result, GraspPlanResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_with_full_approach_planning(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_grasp with plan_approach_to_grasp=True."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4 + i * 0.02, 0.0, 0.35], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=False,
        )

        assert isinstance(result, GraspPlanResult)


class TestMotionPlannerPlanGraspFullWorkflow:
    """Test plan_grasp full workflow including approach, grasp, and lift."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_full_workflow_with_lift(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_grasp with full workflow including lift."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4 + i * 0.02, 0.0, 0.35], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
            grasp_approach_axis="z",
            grasp_approach_offset=-0.1,
            grasp_lift_axis="z",
            grasp_lift_offset=0.1,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=True,
        )

        assert isinstance(result, GraspPlanResult)
        assert motion_planner.last_graph_calls == 0
        assert result.grasp_trajectory.control_space in ControlSpace.bspline_types()
        assert result.lift_trajectory.control_space in ControlSpace.bspline_types()
        action_horizon = motion_planner.trajopt_solver.action_horizon
        assert result.grasp_trajectory.knot.shape[-2] == action_horizon
        assert result.lift_trajectory.knot.shape[-2] == action_horizon
        assert result.grasp_interpolated_trajectory.position.ndim == 4
        assert result.lift_interpolated_trajectory.position.ndim == 4
        assert result.grasp_interpolated_last_tstep.shape == (1, 1)
        assert result.lift_interpolated_last_tstep.shape == (1, 1)
        assert result.grasp_interpolated_last_tstep.item() > 1
        assert result.lift_interpolated_last_tstep.item() > 1
        approach_end = result.approach_interpolated_trajectory.position[
            0,
            0,
            result.approach_interpolated_last_tstep.item() - 1,
        ]
        grasp_start = result.grasp_interpolated_trajectory.position[0, 0, 0]
        grasp_end = result.grasp_interpolated_trajectory.position[
            0,
            0,
            result.grasp_interpolated_last_tstep.item() - 1,
        ]
        lift_start = result.lift_interpolated_trajectory.position[0, 0, 0]
        torch.testing.assert_close(approach_end, grasp_start)
        torch.testing.assert_close(grasp_end, lift_start)
        grasp_state = JointState.from_position(
            grasp_end.unsqueeze(0),
            joint_names=result.grasp_interpolated_trajectory.joint_names,
        )
        actual_grasp = motion_planner.compute_kinematics(
            motion_planner.kinematics.get_active_js(grasp_state)
        ).tool_poses.get_link_pose(tool_frames[0])
        selected_index = int(result.goalset_index.item())
        expected_grasp = torch.tensor(
            [[0.4 + selected_index * 0.02, 0.0, 0.35]],
            **cuda_device_cfg.as_torch_dict(),
        )
        assert torch.linalg.vector_norm(actual_grasp.position - expected_grasp) < 0.005

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_defaults_to_ik_goalset_selection(
        self,
        motion_planner,
        sample_start_state,
        cuda_device_cfg,
        monkeypatch,
    ) -> None:
        """Use IK-only goal selection before the full approach pose plan."""
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 3
        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        position[..., 0] = 0.5
        position[0, :, 1] = torch.linspace(
            -0.15,
            0.15,
            num_goalset,
            **cuda_device_cfg.as_torch_dict(),
        )
        position[..., 2] = 0.3
        quaternion[..., 0] = 1.0
        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )
        original_plan_pose = motion_planner.plan_pose
        plan_pose_calls: list[bool] = []

        def count_plan_pose_calls(
            *args: object,
            **kwargs: object,
        ) -> TrajOptSolverResult | None:
            plan_pose_calls.append(True)
            return original_plan_pose(*args, **kwargs)

        monkeypatch.setattr(motion_planner, "plan_pose", count_plan_pose_calls)

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
            grasp_approach_axis="z",
            grasp_approach_offset=0.1,
            grasp_lift_axis="z",
            grasp_lift_offset=0.1,
            plan_approach_to_grasp=False,
            plan_grasp_to_lift=False,
        )

        assert result.success.any()
        assert result.goalset_result.solution.shape[1] == 1
        assert len(plan_pose_calls) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_trajopt_phase_passes(
        self,
        motion_planner,
        sample_start_state,
        cuda_device_cfg,
        monkeypatch,
    ) -> None:
        """Use one TrajOpt pass by default and opt into the second pass."""
        tool_frames = motion_planner.tool_frames
        position = torch.tensor([[0.5, 0.0, 0.3]], **cuda_device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
        grasp_tool_pose = GoalToolPose.from_poses(
            {tool_frames[0]: Pose(position=position, quaternion=quaternion)},
            ordered_tool_frames=tool_frames,
        )
        original_plan_pose = motion_planner.plan_pose
        plan_pose_calls: list[dict[str, object]] = []

        def count_plan_pose_calls(
            *args: object,
            **kwargs: object,
        ) -> TrajOptSolverResult | None:
            plan_pose_calls.append(kwargs)
            return original_plan_pose(*args, **kwargs)

        def reject_rrt(*args: object, **kwargs: object) -> None:
            raise AssertionError("projected RRT must be bypassed on the explicit TrajOpt path")

        monkeypatch.setattr(motion_planner, "plan_pose", count_plan_pose_calls)
        monkeypatch.setattr(motion_planner._grasp_phase_planner, "_plan_rrt_phase", reject_rrt)

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=grasp_tool_pose,
            grasp_approach_offset=0.1,
            grasp_lift_offset=0.1,
            use_trajopt_for_goalset_selection=True,
        )

        assert result.success.any()
        assert len(plan_pose_calls) == 4
        assert [call.get("finetune_attempts", 1) for call in plan_pose_calls] == [
            1,
            1,
            0,
            0,
        ]

        plan_pose_calls.clear()
        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=grasp_tool_pose,
            grasp_approach_offset=0.1,
            grasp_lift_offset=0.1,
            use_trajopt_for_goalset_selection=True,
            use_double_pass_trajopt_for_constrained_phases=True,
        )

        assert result.success.any()
        assert len(plan_pose_calls) == 4
        assert [call.get("finetune_attempts", 1) for call in plan_pose_calls] == [
            1,
            1,
            1,
            1,
        ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_rejects_conflicting_constrained_phase_flags(
        self,
        motion_planner,
        sample_start_state,
    ) -> None:
        """Reject simultaneous RRT and double-pass TrajOpt selection."""
        grasp_poses = motion_planner.compute_kinematics(sample_start_state).tool_poses.as_goal()
        with pytest.raises(ValueError, match="mutually exclusive"):
            motion_planner.plan_grasp(
                current_state=sample_start_state,
                grasp_poses=grasp_poses,
                use_rrt_for_constrained_phases=True,
                use_double_pass_trajopt_for_constrained_phases=True,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_reuses_selected_grasp_configuration(
        self,
        motion_planner,
        sample_start_state,
        cuda_device_cfg,
        monkeypatch,
    ) -> None:
        """Connect the approach endpoint to the grasp configuration already selected."""
        tool_frames = motion_planner.tool_frames
        position = torch.tensor([[0.5, 0.0, 0.3]], **cuda_device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
        grasp_tool_pose = GoalToolPose.from_poses(
            {tool_frames[0]: Pose(position=position, quaternion=quaternion)},
            ordered_tool_frames=tool_frames,
        )
        original_rrt_plan = motion_planner._grasp_phase_planner._plan_rrt_phase
        rrt_starts: list[torch.Tensor] = []
        rrt_goals: list[torch.Tensor] = []

        def record_rrt_plan(
            goal_tool_poses: GoalToolPose,
            current_state: JointState,
            goal_configuration: torch.Tensor,
            goal_valid: torch.Tensor,
            axis: str,
            in_tool_frame: bool,
        ) -> object:
            rrt_starts.append(current_state.position.clone())
            rrt_goals.append(goal_configuration.clone())
            return original_rrt_plan(
                goal_tool_poses,
                current_state,
                goal_configuration,
                goal_valid,
                axis,
                in_tool_frame,
            )

        monkeypatch.setattr(
            motion_planner._grasp_phase_planner,
            "_plan_rrt_phase",
            record_rrt_plan,
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=grasp_tool_pose,
            grasp_approach_offset=0.1,
            plan_grasp_to_lift=False,
            use_rrt_for_constrained_phases=True,
        )

        assert result.success.any()
        assert len(rrt_starts) == 1
        expected_goal = get_result_goal_configuration(
            result.goalset_result,
            motion_planner.kinematics,
        )
        assert torch.allclose(expected_goal, rrt_goals[0])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_with_custom_axes(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_grasp with different approach and lift axes."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 1

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        position[0, 0, :] = torch.tensor([0.4, 0.0, 0.35], **cuda_device_cfg.as_torch_dict())
        quaternion[0, 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict())

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
            grasp_approach_axis="x",
            grasp_approach_offset=-0.1,
            grasp_lift_axis="z",
            grasp_lift_offset=0.1,
            plan_approach_to_grasp=False,
            plan_grasp_to_lift=False,
        )

        assert isinstance(result, GraspPlanResult)


class TestMotionPlannerKinematicsProperty:
    """Test MotionPlanner.kinematics property."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_kinematics_property_returns_kinematics(self, motion_planner):
        """Test kinematics property returns kinematics object."""
        kinematics = motion_planner.kinematics
        assert kinematics is not None
        assert hasattr(kinematics, "tool_frames")


class TestMotionPlannerPartialIKSuccess:
    """Test MotionPlanner partial IK success path (lines 135-136)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_partial_ik_success(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test planning when some IK seeds succeed but not all.

        Uses a borderline goal that may cause partial IK success.
        """
        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames

        # Goal at edge of workspace - may have partial IK success
        position = torch.tensor([[0.5, 0.25, 0.15]], **cuda_device_cfg.as_torch_dict())
        quaternion = torch.tensor([[0.707, 0.0, 0.707, 0.0]], **cuda_device_cfg.as_torch_dict())
        edge_pose = Pose(position=position, quaternion=quaternion)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: edge_pose},
            ordered_tool_frames=tool_frames,
        )

        result = motion_planner.plan_pose(
            current_state=sample_start_state,
            goal_tool_poses=goal_tool_poses,
            max_attempts=1,
            enable_graph_attempt=10,  # Disable graph to focus on IK path
        )

        assert result is None or isinstance(result, TrajOptSolverResult)


class TestMotionPlannerPlanGraspFailurePaths:
    """Test plan_grasp failure paths (lines 270-271, 304-305)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_goalset_success_false(
        self, motion_planner, sample_start_state, cuda_device_cfg
    ):
        """Test plan_grasp when goalset result has success=False (lines 270-271).

        This requires IK to return results but with success=False.
        """
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        # Create poses that are borderline - may get result but not succeed
        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        for i in range(num_goalset):
            # Borderline reachable positions
            position[0, i, :] = torch.tensor(
                [0.8, 0.5 + i * 0.1, 0.05], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, i, :] = torch.tensor(
                [0.0, 1.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
        )

        assert isinstance(result, GraspPlanResult)
        # Result should have status set if failed
        if not result.success.any() if result.success is not None else True:
            assert result.status is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_approach_fails(self, motion_planner, sample_start_state, cuda_device_cfg):
        """Test plan_grasp when approach planning fails.

        Goalset succeeds but approach position is hard to reach.
        """
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        tool_frames = motion_planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        # Reachable grasp poses
        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())

        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4, i * 0.02, 0.35], **cuda_device_cfg.as_torch_dict()
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
            )

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position.unsqueeze(1).unsqueeze(1),
            quaternion=quaternion.unsqueeze(1).unsqueeze(1),
        )

        # Use a very large approach offset that may be unreachable
        result = motion_planner.plan_grasp(
            current_state=sample_start_state,
            grasp_poses=goal_tool_pose,
            grasp_approach_axis="z",
            grasp_approach_offset=-1.5,
            plan_grasp_to_lift=False,
        )

        assert isinstance(result, GraspPlanResult)
