# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for GraspPlanResult dataclass."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.motion.motion_planner_result import GraspPlanResult
from curobo._src.state.state_joint import JointState


class TestGraspPlanResultDataclass:
    """Test GraspPlanResult dataclass attributes."""

    def test_default_success_is_none(self):
        """Test default success is None."""
        result = GraspPlanResult()
        assert result.success is None

    def test_default_grasp_trajectory_is_none(self):
        """Test default grasp_trajectory is None."""
        result = GraspPlanResult()
        assert result.grasp_trajectory is None

    def test_default_grasp_trajectory_dt_is_none(self):
        """Test default grasp_trajectory_dt is None."""
        result = GraspPlanResult()
        assert result.grasp_trajectory_dt is None

    def test_default_grasp_interpolated_trajectory_is_none(self):
        """Test default grasp_interpolated_trajectory is None."""
        result = GraspPlanResult()
        assert result.grasp_interpolated_trajectory is None

    def test_default_approach_trajectory_is_none(self):
        """Test default approach_trajectory is None."""
        result = GraspPlanResult()
        assert result.approach_trajectory is None

    def test_default_approach_trajectory_dt_is_none(self):
        """Test default approach_trajectory_dt is None."""
        result = GraspPlanResult()
        assert result.approach_trajectory_dt is None

    def test_default_approach_interpolated_trajectory_is_none(self):
        """Test default approach_interpolated_trajectory is None."""
        result = GraspPlanResult()
        assert result.approach_interpolated_trajectory is None

    def test_default_lift_trajectory_is_none(self):
        """Test default lift_trajectory is None."""
        result = GraspPlanResult()
        assert result.lift_trajectory is None

    def test_default_lift_trajectory_dt_is_none(self):
        """Test default lift_trajectory_dt is None."""
        result = GraspPlanResult()
        assert result.lift_trajectory_dt is None

    def test_default_lift_interpolated_trajectory_is_none(self):
        """Test default lift_interpolated_trajectory is None."""
        result = GraspPlanResult()
        assert result.lift_interpolated_trajectory is None

    def test_default_status_is_none(self):
        """Test default status is None."""
        result = GraspPlanResult()
        assert result.status is None

    def test_default_planning_time_is_zero(self):
        """Test default planning_time is 0.0."""
        result = GraspPlanResult()
        assert result.planning_time == 0.0

    def test_default_goalset_index_is_none(self):
        """Test default goalset_index is None."""
        result = GraspPlanResult()
        assert result.goalset_index is None


class TestGraspPlanResultWithValues:
    """Test GraspPlanResult with actual values."""

    def test_with_success_tensor(self):
        """Test with success tensor."""
        success = torch.tensor([True])
        result = GraspPlanResult(success=success)
        assert torch.equal(result.success, success)

    def test_with_grasp_trajectory(self):
        """Test with grasp_trajectory."""
        trajectory = JointState.from_position(
            torch.randn(1, 10, 7),
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
        )
        result = GraspPlanResult(grasp_trajectory=trajectory)
        assert result.grasp_trajectory is trajectory

    def test_with_grasp_trajectory_dt(self):
        """Test with grasp_trajectory_dt."""
        dt = torch.tensor([0.02])
        result = GraspPlanResult(grasp_trajectory_dt=dt)
        assert torch.equal(result.grasp_trajectory_dt, dt)

    def test_with_status_string(self):
        """Test with status string."""
        result = GraspPlanResult(status="Success")
        assert result.status == "Success"

    def test_with_planning_time(self):
        """Test with planning_time."""
        result = GraspPlanResult(planning_time=1.5)
        assert result.planning_time == 1.5

    def test_with_goalset_index(self):
        """Test with goalset_index."""
        goalset_index = torch.tensor([2])
        result = GraspPlanResult(goalset_index=goalset_index)
        assert torch.equal(result.goalset_index, goalset_index)


class TestGraspPlanResultCompleteResult:
    """Test GraspPlanResult with complete result."""

    def test_complete_successful_result(self):
        """Test complete successful result."""
        result = GraspPlanResult(
            success=torch.tensor([True]),
            approach_trajectory=JointState.from_position(
                torch.randn(1, 10, 7),
                joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
            ),
            approach_trajectory_dt=torch.tensor([0.02]),
            grasp_trajectory=JointState.from_position(
                torch.randn(1, 10, 7),
                joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
            ),
            grasp_trajectory_dt=torch.tensor([0.02]),
            lift_trajectory=JointState.from_position(
                torch.randn(1, 5, 7),
                joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
            ),
            lift_trajectory_dt=torch.tensor([0.02]),
            status="Planning successful",
            planning_time=0.5,
            goalset_index=torch.tensor([0]),
        )

        assert result.success[0].item() == True
        assert result.approach_trajectory is not None
        assert result.grasp_trajectory is not None
        assert result.lift_trajectory is not None
        assert result.status == "Planning successful"
        assert result.planning_time == 0.5

    def test_complete_failed_result(self):
        """Test complete failed result."""
        result = GraspPlanResult(
            success=torch.tensor([False]),
            status="No valid grasp found",
            planning_time=2.0,
        )

        assert result.success[0].item() == False
        assert result.grasp_trajectory is None
        assert result.status == "No valid grasp found"
        assert result.planning_time == 2.0


class TestGraspPlanResultDeviceHandling:
    """Test GraspPlanResult device handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_success_tensor(self):
        """Test with CUDA success tensor."""
        success = torch.tensor([True], device="cuda:0")
        result = GraspPlanResult(success=success)
        assert result.success.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_goalset_index(self):
        """Test with CUDA goalset_index tensor."""
        goalset_index = torch.tensor([0], device="cuda:0")
        result = GraspPlanResult(goalset_index=goalset_index)
        assert result.goalset_index.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_dt_tensor(self):
        """Test with CUDA dt tensor."""
        dt = torch.tensor([0.02], device="cuda:0")
        result = GraspPlanResult(grasp_trajectory_dt=dt)
        assert result.grasp_trajectory_dt.is_cuda

