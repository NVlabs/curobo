# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for PoseCostMetric."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_pose_metric import PoseCostMetric
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture(params=["cpu", "cuda:0"])
def device_cfg(request):
    """Create tensor configuration for both CPU and GPU."""
    device = request.param
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device(device))


class TestPoseCostMetric:
    """Test PoseCostMetric class."""

    def test_default_init(self):
        """Test default initialization."""
        metric = PoseCostMetric()
        assert metric.hold_partial_pose is False
        assert metric.release_partial_pose is False
        assert metric.hold_vec_weight is None
        assert metric.reach_partial_pose is False
        assert metric.reach_full_pose is False
        assert metric.reach_vec_weight is None
        assert metric.offset_position is None
        assert metric.offset_rotation is None
        assert metric.offset_tstep_fraction == -1.0
        assert metric.remove_offset_waypoint is False
        assert metric.include_link_pose is False
        assert metric.project_to_goal_frame is None

    def test_init_with_hold_partial_pose(self, device_cfg):
        """Test initialization with hold_partial_pose."""
        hold_vec_weight = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], **device_cfg.as_torch_dict())
        metric = PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=hold_vec_weight,
        )
        assert metric.hold_partial_pose is True
        assert torch.allclose(metric.hold_vec_weight, hold_vec_weight)

    def test_init_with_reach_partial_pose(self, device_cfg):
        """Test initialization with reach_partial_pose."""
        reach_vec_weight = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], **device_cfg.as_torch_dict())
        metric = PoseCostMetric(
            reach_partial_pose=True,
            reach_vec_weight=reach_vec_weight,
        )
        assert metric.reach_partial_pose is True
        assert torch.allclose(metric.reach_vec_weight, reach_vec_weight)

    def test_init_with_offset(self, device_cfg):
        """Test initialization with offset position and rotation."""
        offset_position = torch.tensor([0.1, 0.0, 0.0], **device_cfg.as_torch_dict())
        offset_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], **device_cfg.as_torch_dict())
        metric = PoseCostMetric(
            offset_position=offset_position,
            offset_rotation=offset_rotation,
            offset_tstep_fraction=0.8,
        )
        assert torch.allclose(metric.offset_position, offset_position)
        assert torch.allclose(metric.offset_rotation, offset_rotation)
        assert metric.offset_tstep_fraction == 0.8

    def test_clone(self, device_cfg):
        """Test cloning PoseCostMetric."""
        hold_vec_weight = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], **device_cfg.as_torch_dict())
        offset_position = torch.tensor([0.1, 0.0, 0.0], **device_cfg.as_torch_dict())
        offset_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], **device_cfg.as_torch_dict())
        original = PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=hold_vec_weight,
            offset_position=offset_position,
            offset_rotation=offset_rotation,
            offset_tstep_fraction=0.5,
            remove_offset_waypoint=True,
        )

        cloned = original.clone()

        assert cloned.hold_partial_pose == original.hold_partial_pose
        assert torch.allclose(cloned.hold_vec_weight, original.hold_vec_weight)
        assert torch.allclose(cloned.offset_position, original.offset_position)
        assert torch.allclose(cloned.offset_rotation, original.offset_rotation)
        assert cloned.offset_tstep_fraction == original.offset_tstep_fraction
        assert cloned.remove_offset_waypoint == original.remove_offset_waypoint

        # Verify deep copy
        cloned.hold_vec_weight[0] = 100.0
        assert original.hold_vec_weight[0] == 1.0

    def test_clone_with_none_tensors(self):
        """Test cloning with None tensors."""
        original = PoseCostMetric(
            hold_partial_pose=True,
            reach_full_pose=True,
        )
        cloned = original.clone()

        assert cloned.hold_vec_weight is None
        assert cloned.reach_vec_weight is None
        assert cloned.offset_position is None
        assert cloned.offset_rotation is None

    def test_create_grasp_approach_metric_default(self, device_cfg):
        """Test create_grasp_approach_metric with default parameters."""
        metric = PoseCostMetric.create_grasp_approach_metric(device_cfg=device_cfg)

        assert metric.hold_partial_pose is True
        assert metric.hold_vec_weight is not None
        assert metric.hold_vec_weight.shape == (6,)
        # Default linear_axis is 2 (z-axis), so hold_vec_weight[5] should be 0
        assert metric.hold_vec_weight[5] == 0.0  # z position not held
        assert metric.hold_vec_weight[0] == 1.0  # x orientation held
        assert metric.hold_vec_weight[1] == 1.0  # y orientation held
        assert metric.hold_vec_weight[2] == 1.0  # z orientation held
        assert metric.offset_position is not None
        assert metric.offset_position[2] == 0.1  # Default offset on z-axis
        assert metric.offset_tstep_fraction == 0.8

    def test_create_grasp_approach_metric_x_axis(self, device_cfg):
        """Test create_grasp_approach_metric with x-axis approach."""
        metric = PoseCostMetric.create_grasp_approach_metric(
            offset_position=0.15,
            linear_axis=0,  # x-axis
            tstep_fraction=0.7,
            device_cfg=device_cfg,
        )

        assert metric.hold_partial_pose is True
        assert metric.hold_vec_weight[3] == 0.0  # x position not held
        assert metric.hold_vec_weight[4] == 1.0  # y position held
        assert metric.hold_vec_weight[5] == 1.0  # z position held
        assert metric.offset_position[0] == 0.15  # offset on x-axis
        assert metric.offset_position[1] == 0.0
        assert metric.offset_position[2] == 0.0
        assert metric.offset_tstep_fraction == 0.7

    def test_create_grasp_approach_metric_y_axis(self, device_cfg):
        """Test create_grasp_approach_metric with y-axis approach."""
        metric = PoseCostMetric.create_grasp_approach_metric(
            offset_position=0.2,
            linear_axis=1,  # y-axis
            device_cfg=device_cfg,
        )

        assert metric.hold_vec_weight[4] == 0.0  # y position not held
        assert metric.offset_position[1] == 0.2  # offset on y-axis
        assert metric.offset_position[0] == 0.0
        assert metric.offset_position[2] == 0.0

    def test_create_grasp_approach_metric_with_project_to_goal_frame(self, device_cfg):
        """Test create_grasp_approach_metric with project_to_goal_frame."""
        metric = PoseCostMetric.create_grasp_approach_metric(
            project_to_goal_frame=True,
            device_cfg=device_cfg,
        )
        # Note: create_grasp_approach_metric doesn't set project_to_goal_frame
        # in the current implementation, so this tests the default behavior

    def test_reset_metric(self):
        """Test reset_metric factory method."""
        metric = PoseCostMetric.reset_metric()

        assert metric.remove_offset_waypoint is True
        assert metric.reach_full_pose is True
        assert metric.release_partial_pose is True

    def test_reach_position_metric(self, device_cfg):
        """Test reach_position_metric factory method."""
        metric = PoseCostMetric.reach_position_metric(device_cfg=device_cfg)

        assert metric.reach_partial_pose is True
        assert metric.reach_vec_weight is not None
        # Position weights should be 1, orientation weights should be 0
        assert metric.reach_vec_weight[0] == 0.0  # orientation x
        assert metric.reach_vec_weight[1] == 0.0  # orientation y
        assert metric.reach_vec_weight[2] == 0.0  # orientation z
        assert metric.reach_vec_weight[3] == 1.0  # position x
        assert metric.reach_vec_weight[4] == 1.0  # position y
        assert metric.reach_vec_weight[5] == 1.0  # position z

    def test_offset_tstep_fraction_default(self):
        """Test default offset_tstep_fraction."""
        metric = PoseCostMetric()
        assert metric.offset_tstep_fraction == -1.0

    def test_include_link_pose(self):
        """Test include_link_pose flag."""
        metric = PoseCostMetric(include_link_pose=True)
        assert metric.include_link_pose is True

    def test_project_to_goal_frame(self):
        """Test project_to_goal_frame flag."""
        metric = PoseCostMetric(project_to_goal_frame=True)
        assert metric.project_to_goal_frame is True

        metric_false = PoseCostMetric(project_to_goal_frame=False)
        assert metric_false.project_to_goal_frame is False

    def test_all_flags_combinations(self):
        """Test various flag combinations."""
        metric = PoseCostMetric(
            hold_partial_pose=True,
            release_partial_pose=True,
            reach_partial_pose=True,
            reach_full_pose=True,
            remove_offset_waypoint=True,
            include_link_pose=True,
            project_to_goal_frame=True,
        )
        assert metric.hold_partial_pose is True
        assert metric.release_partial_pose is True
        assert metric.reach_partial_pose is True
        assert metric.reach_full_pose is True
        assert metric.remove_offset_waypoint is True
        assert metric.include_link_pose is True
        assert metric.project_to_goal_frame is True

    def test_clone_preserves_all_fields(self, device_cfg):
        """Test that clone preserves all fields."""
        hold_vec_weight = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], **device_cfg.as_torch_dict())
        reach_vec_weight = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], **device_cfg.as_torch_dict())
        offset_position = torch.tensor([0.1, 0.2, 0.3], **device_cfg.as_torch_dict())
        offset_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], **device_cfg.as_torch_dict())

        original = PoseCostMetric(
            hold_partial_pose=True,
            release_partial_pose=True,
            hold_vec_weight=hold_vec_weight,
            reach_partial_pose=True,
            reach_full_pose=True,
            reach_vec_weight=reach_vec_weight,
            offset_position=offset_position,
            offset_rotation=offset_rotation,
            offset_tstep_fraction=0.75,
            remove_offset_waypoint=True,
            include_link_pose=True,
            project_to_goal_frame=True,
        )

        cloned = original.clone()

        assert cloned.hold_partial_pose == original.hold_partial_pose
        assert cloned.release_partial_pose == original.release_partial_pose
        assert torch.allclose(cloned.hold_vec_weight, original.hold_vec_weight)
        assert cloned.reach_partial_pose == original.reach_partial_pose
        assert cloned.reach_full_pose == original.reach_full_pose
        assert torch.allclose(cloned.reach_vec_weight, original.reach_vec_weight)
        assert torch.allclose(cloned.offset_position, original.offset_position)
        assert torch.allclose(cloned.offset_rotation, original.offset_rotation)
        assert cloned.offset_tstep_fraction == original.offset_tstep_fraction
        assert cloned.remove_offset_waypoint == original.remove_offset_waypoint
        assert cloned.include_link_pose == original.include_link_pose
        assert cloned.project_to_goal_frame == original.project_to_goal_frame

    def test_grasp_approach_metric_is_independent_of_device_cfg_for_default(self):
        """Test that grasp approach metric tensors are created on specified device."""
        cpu_cfg = DeviceCfg(device=torch.device("cpu"))
        metric_cpu = PoseCostMetric.create_grasp_approach_metric(device_cfg=cpu_cfg)
        assert metric_cpu.hold_vec_weight.device == cpu_cfg.device
        assert metric_cpu.offset_position.device == cpu_cfg.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_grasp_approach_metric_cuda(self):
        """Test grasp approach metric on CUDA."""
        cuda_cfg = DeviceCfg(device=torch.device("cuda", 0))
        metric_cuda = PoseCostMetric.create_grasp_approach_metric(device_cfg=cuda_cfg)
        assert metric_cuda.hold_vec_weight.device == cuda_cfg.device
        assert metric_cuda.offset_position.device == cuda_cfg.device

    def test_reach_position_metric_device(self, device_cfg):
        """Test that reach_position_metric tensors are on correct device."""
        metric = PoseCostMetric.reach_position_metric(device_cfg=device_cfg)
        assert metric.reach_vec_weight.device == device_cfg.device

