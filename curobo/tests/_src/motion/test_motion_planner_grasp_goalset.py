# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Test plan_grasp with varying goalset sizes after warmup with large num_goalset."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.motion.motion_planner_result import GraspPlanResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


def _make_grasp_tool_pose(planner, n_grasps, device_cfg):
    """Build a ToolPose with *n_grasps* grasp candidates spread around the workspace."""
    tool_frames = planner.trajopt_solver.kinematics.tool_frames
    td = device_cfg.as_torch_dict()

    position = torch.zeros((1, n_grasps, 3), **td)
    quaternion = torch.zeros((1, n_grasps, 4), **td)

    for i in range(n_grasps):
        t = i / max(n_grasps - 1, 1)
        position[0, i, :] = torch.tensor(
            [0.35 + 0.15 * t, -0.1 + 0.2 * t, 0.25 + 0.1 * t], **td
        )
        quaternion[0, i, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], **td)

    return GoalToolPose(
        tool_frames=tool_frames,
        position=position[:, None, None],
        quaternion=quaternion[:, None, None],
    )


@pytest.fixture(scope="module")
def cuda_device_cfg():
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def grasp_planner(cuda_device_cfg):
    """Create a MotionPlanner with max_goalset=100 and cuda_graph warmup."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        device_cfg=cuda_device_cfg,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=True,
        max_goalset=100,
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    return planner


class TestPlanGraspVaryingGoalsetSize:
    """Warmup with 100 goalset, then plan_grasp with 40 and 80 grasps."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_40_grasps(self, grasp_planner, cuda_device_cfg):
        """plan_grasp with 40 grasps after warmup with 100 goalset."""
        start = JointState.from_position(
            grasp_planner.default_joint_state.position.unsqueeze(0),
            joint_names=grasp_planner.joint_names,
        )
        tool_pose = _make_grasp_tool_pose(grasp_planner, 40, cuda_device_cfg)

        result = grasp_planner.plan_grasp(
            current_state=start,
            grasp_poses=tool_pose,
            grasp_approach_offset=-0.1,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=True,
        )

        assert isinstance(result, GraspPlanResult)
        assert hasattr(result, "status")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_80_grasps(self, grasp_planner, cuda_device_cfg):
        """plan_grasp with 80 grasps after the 40-grasp call."""
        start = JointState.from_position(
            grasp_planner.default_joint_state.position.unsqueeze(0),
            joint_names=grasp_planner.joint_names,
        )
        tool_pose = _make_grasp_tool_pose(grasp_planner, 80, cuda_device_cfg)

        result = grasp_planner.plan_grasp(
            current_state=start,
            grasp_poses=tool_pose,
            grasp_approach_offset=-0.1,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=True,
        )

        assert isinstance(result, GraspPlanResult)
        assert hasattr(result, "status")
