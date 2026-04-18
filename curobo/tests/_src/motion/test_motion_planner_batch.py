# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for BatchMotionPlanner class."""

import pytest
import torch

from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose


@pytest.fixture(scope="module")
def cuda_device_cfg():
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


class TestBatchMotionPlannerInit:
    """Test BatchMotionPlanner initialization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_with_batch_config(self, cuda_device_cfg):
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=4,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)
        assert planner.batch_size == 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_graph_planner_when_shared_env(self, cuda_device_cfg):
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=4,
            multi_env=False,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)
        assert planner.graph_planner is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_no_graph_planner_when_multi_env(self, cuda_device_cfg):
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=4,
            multi_env=True,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)
        assert planner.graph_planner is None


class TestBatchMotionPlannerProperties:
    """Test BatchMotionPlanner properties."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_joint_names(self, cuda_device_cfg):
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=2,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)
        assert isinstance(planner.joint_names, list)
        assert len(planner.joint_names) == 7

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tool_frames(self, cuda_device_cfg):
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=2,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)
        assert isinstance(planner.tool_frames, list)
        assert len(planner.tool_frames) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_joint_state(self, cuda_device_cfg):
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=2,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)
        joint_state = planner.default_joint_state
        assert isinstance(joint_state, JointState)
        assert joint_state.position is not None


class TestBatchMotionPlannerPlanPose:
    """Test BatchMotionPlanner.plan_pose."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_returns_result(self, cuda_device_cfg):
        batch_size = 4
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=batch_size,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)

        start = planner.default_joint_state.position.unsqueeze(0).repeat(batch_size, 1)
        current_states = JointState.from_position(start, joint_names=planner.joint_names)

        tool_frames = planner.tool_frames
        position = torch.tensor([[0.4, 0.0, 0.4]], **cuda_device_cfg.as_torch_dict())
        position = position.repeat(batch_size, 1)
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
        quaternion = quaternion.repeat(batch_size, 1)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: Pose(position=position, quaternion=quaternion).unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        result = planner.plan_pose(goal_tool_poses=goal_tool_poses, current_state=current_states)
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_result_batch_dim(self, cuda_device_cfg):
        batch_size = 4
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=batch_size,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)

        start = planner.default_joint_state.position.unsqueeze(0).repeat(batch_size, 1)
        current_states = JointState.from_position(start, joint_names=planner.joint_names)

        tool_frames = planner.tool_frames
        position = torch.tensor([[0.4, 0.0, 0.4]], **cuda_device_cfg.as_torch_dict())
        position = position.repeat(batch_size, 1)
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
        quaternion = quaternion.repeat(batch_size, 1)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: Pose(position=position, quaternion=quaternion).unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        result = planner.plan_pose(goal_tool_poses=goal_tool_poses, current_state=current_states)
        if result is not None:
            assert result.success.shape[0] == batch_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_with_max_attempts(self, cuda_device_cfg):
        batch_size = 2
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=batch_size,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)

        start = planner.default_joint_state.position.unsqueeze(0).repeat(batch_size, 1)
        current_states = JointState.from_position(start, joint_names=planner.joint_names)

        tool_frames = planner.tool_frames
        position = torch.tensor([[0.4, 0.0, 0.4]], **cuda_device_cfg.as_torch_dict())
        position = position.repeat(batch_size, 1)
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **cuda_device_cfg.as_torch_dict())
        quaternion = quaternion.repeat(batch_size, 1)
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: Pose(position=position, quaternion=quaternion).unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        result = planner.plan_pose(
            goal_tool_poses=goal_tool_poses,
            current_state=current_states,
            max_attempts=3,
            success_ratio=0.5,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)


class TestBatchMotionPlannerPlanCspace:
    """Test BatchMotionPlanner.plan_cspace."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_returns_result(self, cuda_device_cfg):
        batch_size = 4
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=batch_size,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)

        start = planner.default_joint_state.position.unsqueeze(0).repeat(batch_size, 1)
        current_states = JointState.from_position(start, joint_names=planner.joint_names)

        goal_states = current_states.clone()
        goal_states.position[..., 0] += 0.1

        result = planner.plan_cspace(goal_states, current_states)
        assert result is not None
        assert isinstance(result, TrajOptSolverResult)
        assert result.success.shape[0] == batch_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_with_max_attempts(self, cuda_device_cfg):
        batch_size = 2
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=batch_size,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)

        start = planner.default_joint_state.position.unsqueeze(0).repeat(batch_size, 1)
        current_states = JointState.from_position(start, joint_names=planner.joint_names)

        goal_states = current_states.clone()
        goal_states.position[..., 0] += 0.1

        result = planner.plan_cspace(
            goal_states, current_states, max_attempts=2, success_ratio=1.0,
        )
        assert result is not None
        assert isinstance(result, TrajOptSolverResult)


class TestBatchMotionPlannerWarmup:
    """Test BatchMotionPlanner.warmup."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_warmup_succeeds(self, cuda_device_cfg):
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=2,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)
        result = planner.warmup(enable_graph=False, num_warmup_iterations=2)
        assert result is True


class TestBatchMotionPlannerPlanGrasp:
    """Test BatchMotionPlanner.plan_grasp."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_returns_result(self, cuda_device_cfg):
        batch_size = 2
        n_grasps = 3
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=batch_size,
            max_goalset=n_grasps,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)

        start = planner.default_joint_state.position.unsqueeze(0).repeat(batch_size, 1)
        current_states = JointState.from_position(start, joint_names=planner.joint_names)

        tool_frames = planner.tool_frames
        num_links = len(tool_frames)
        positions = torch.zeros(batch_size, num_links, n_grasps, 3, **cuda_device_cfg.as_torch_dict())
        quaternions = torch.zeros(batch_size, num_links, n_grasps, 4, **cuda_device_cfg.as_torch_dict())
        for i in range(n_grasps):
            positions[:, 0, i, :] = torch.tensor([0.4 + i * 0.05, 0.0, 0.4], **cuda_device_cfg.as_torch_dict())
            quaternions[:, 0, i, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict())

        from curobo._src.motion.motion_planner_result import GraspPlanResult

        grasp_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=positions[:, None],
            quaternion=quaternions[:, None],
        )
        result = planner.plan_grasp(grasp_poses, current_states)
        assert isinstance(result, GraspPlanResult)
        assert result.success.shape[0] == batch_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_approach_only(self, cuda_device_cfg):
        batch_size = 2
        n_grasps = 2
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=batch_size,
            max_goalset=n_grasps,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)

        start = planner.default_joint_state.position.unsqueeze(0).repeat(batch_size, 1)
        current_states = JointState.from_position(start, joint_names=planner.joint_names)

        tool_frames = planner.tool_frames
        num_links = len(tool_frames)
        positions = torch.zeros(batch_size, num_links, n_grasps, 3, **cuda_device_cfg.as_torch_dict())
        quaternions = torch.zeros(batch_size, num_links, n_grasps, 4, **cuda_device_cfg.as_torch_dict())
        for i in range(n_grasps):
            positions[:, 0, i, :] = torch.tensor([0.4, i * 0.1, 0.4], **cuda_device_cfg.as_torch_dict())
            quaternions[:, 0, i, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict())

        from curobo._src.motion.motion_planner_result import GraspPlanResult

        grasp_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=positions[:, None],
            quaternion=quaternions[:, None],
        )
        result = planner.plan_grasp(
            grasp_poses, current_states,
            plan_approach_to_grasp=False, plan_grasp_to_lift=False,
        )
        assert isinstance(result, GraspPlanResult)
        assert result.approach_success.shape[0] == batch_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_has_per_problem_success(self, cuda_device_cfg):
        batch_size = 4
        n_grasps = 2
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=batch_size,
            max_goalset=n_grasps,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)

        start = planner.default_joint_state.position.unsqueeze(0).repeat(batch_size, 1)
        current_states = JointState.from_position(start, joint_names=planner.joint_names)

        tool_frames = planner.tool_frames
        num_links = len(tool_frames)
        positions = torch.zeros(batch_size, num_links, n_grasps, 3, **cuda_device_cfg.as_torch_dict())
        quaternions = torch.zeros(batch_size, num_links, n_grasps, 4, **cuda_device_cfg.as_torch_dict())
        for i in range(n_grasps):
            positions[:, 0, i, :] = torch.tensor([0.4 + i * 0.05, 0.0, 0.4], **cuda_device_cfg.as_torch_dict())
            quaternions[:, 0, i, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict())

        from curobo._src.motion.motion_planner_result import GraspPlanResult

        grasp_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=positions[:, None],
            quaternion=quaternions[:, None],
        )
        result = planner.plan_grasp(grasp_poses, current_states)
        assert isinstance(result, GraspPlanResult)
        assert result.success.dtype == torch.bool
        assert result.success.shape[0] == batch_size
        assert result.approach_success.shape[0] == batch_size
        assert result.grasp_success.shape[0] == batch_size
        assert result.lift_success.shape[0] == batch_size


class TestBatchMotionPlannerResetSeed:
    """Test BatchMotionPlanner.reset_seed."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_seed_no_error(self, cuda_device_cfg):
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=2,
            use_cuda_graph=False,
        )
        planner = BatchMotionPlanner(config)
        planner.reset_seed()
