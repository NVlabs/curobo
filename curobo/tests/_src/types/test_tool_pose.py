# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for ToolPose and GoalToolPose."""

# Third Party
import pytest
import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose

# CuRobo
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


@pytest.fixture(params=["cpu", "cuda:0"])
def device_cfg(request):
    """Create tensor configuration for both CPU and GPU."""
    device = request.param
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device(device))


class TestToolPose:
    """Test ToolPose class: 4D FK output [batch, horizon, num_links, 3/4]."""

    def test_initialization(self, device_cfg):
        """Test ToolPose initialization with 4D tensors."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert tp.tool_frames == tool_frames
        assert tp.position.shape == (2, 1, 3, 3)
        assert tp.quaternion.shape == (2, 1, 3, 4)

    def test_initialization_rejects_non_4d(self, device_cfg):
        """Test that __post_init__ rejects non-4D tensors."""
        tool_frames = ["link1", "link2"]
        positions_3d = torch.rand(2, 2, 3, **device_cfg.as_torch_dict())
        quaternions_3d = torch.rand(2, 2, 4, **device_cfg.as_torch_dict())

        with pytest.raises(Exception):
            ToolPose(tool_frames=tool_frames, position=positions_3d, quaternion=quaternions_3d)

    def test_initialization_rejects_link_mismatch(self, device_cfg):
        """Test that __post_init__ rejects mismatched num_links and tool_frames."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, **device_cfg.as_torch_dict())

        with pytest.raises(Exception):
            ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

    def test_get_link_pose(self, device_cfg):
        """Test getting a specific link pose returns 2D [B*H, 3/4]."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        link1_pose = tp.get_link_pose("link1")
        assert link1_pose.position.shape == (2, 3)
        assert link1_pose.quaternion.shape == (2, 4)
        assert torch.allclose(link1_pose.position, positions[:, :, 0, :].reshape(-1, 3))

    def test_get_link_pose_with_contiguous(self, device_cfg):
        """Test getting link pose with contiguous flag."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        link_pose = tp.get_link_pose("link1", make_contiguous=True)
        assert link_pose.position.is_contiguous()
        assert link_pose.quaternion.is_contiguous()

    def test_get_link_pose_invalid_name(self, device_cfg):
        """Test getting link pose with invalid name raises error."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        with pytest.raises(Exception):
            tp.get_link_pose("invalid_link")

    def test_to_dict(self, device_cfg):
        """Test converting to dictionary of 2D Poses."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        pose_dict = tp.to_dict()
        assert len(pose_dict) == 3
        assert "link1" in pose_dict
        assert "link2" in pose_dict
        assert "link3" in pose_dict
        assert isinstance(pose_dict["link1"], Pose)

    def test_to_dict_non_contiguous(self, device_cfg):
        """Test converting to dictionary without making contiguous."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        pose_dict = tp.to_dict(make_contiguous=False)
        assert len(pose_dict) == 2

    def test_clone(self, device_cfg):
        """Test cloning ToolPose produces an independent copy."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        tp_clone = tp.clone()

        assert tp_clone.tool_frames == tp.tool_frames
        assert tp_clone.tool_frames is not tp.tool_frames
        assert torch.allclose(tp_clone.position, tp.position)

        tp_clone.position[0, 0, 0, 0] = 999.0
        assert tp.position[0, 0, 0, 0] != 999.0

    def test_detach(self, device_cfg):
        """Test detaching gradients from ToolPose."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict(), requires_grad=True)
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict(), requires_grad=True)

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        tp_detached = tp.detach()

        assert not tp_detached.position.requires_grad
        assert not tp_detached.quaternion.requires_grad

    def test_len(self, device_cfg):
        """Test length of ToolPose equals number of links."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert len(tp) == 3

    def test_getitem_with_string(self, device_cfg):
        """Test indexing with string returns 2D Pose for that link."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        link_pose = tp["link2"]
        assert isinstance(link_pose, Pose)
        assert torch.allclose(
            link_pose.position, positions[:, :, 1, :].reshape(-1, 3)
        )

    def test_getitem_with_int(self, device_cfg):
        """Test indexing with integer returns batch-sliced ToolPose."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(5, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(5, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        tp_indexed = tp[2]
        assert isinstance(tp_indexed, ToolPose)
        assert tp_indexed.tool_frames == tool_frames
        assert tp_indexed.batch_size == 1

    def test_getitem_with_slice(self, device_cfg):
        """Test indexing with slice returns batch-sliced ToolPose."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(5, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(5, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        tp_sliced = tp[1:3]
        assert isinstance(tp_sliced, ToolPose)
        assert tp_sliced.tool_frames == tool_frames
        assert tp_sliced.position.shape[0] == 2

    def test_position_field(self, device_cfg):
        """Test position is a direct tensor field."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert torch.allclose(tp.position, positions)

    def test_quaternion_field(self, device_cfg):
        """Test quaternion is a direct tensor field."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())
        quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert torch.allclose(tp.quaternion, quaternions, atol=1e-5)

    def test_position_setter(self, device_cfg):
        """Test setting position."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        new_positions = torch.ones(2, 1, 2, 3, **device_cfg.as_torch_dict())
        tp.position = new_positions

        assert torch.allclose(tp.position, new_positions)

    def test_quaternion_setter(self, device_cfg):
        """Test setting quaternion."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        new_quaternions = torch.zeros(2, 1, 2, 4, **device_cfg.as_torch_dict())
        new_quaternions[..., 0] = 1.0
        tp.quaternion = new_quaternions

        assert torch.allclose(tp.quaternion, new_quaternions)

    def test_shape_property(self, device_cfg):
        """Test shape property returns full 4D shape."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(5, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(5, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert tp.shape == (5, 1, 3, 3)

    def test_ndim_property(self, device_cfg):
        """Test ndim property is always 4."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(5, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(5, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert tp.ndim == 4

    def test_copy_(self, device_cfg):
        """Test copying from another ToolPose."""
        link_names1 = ["link1", "link2"]
        positions1 = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions1 = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())
        tp1 = ToolPose(tool_frames=link_names1, position=positions1, quaternion=quaternions1)

        link_names2 = ["link3", "link4"]
        positions2 = torch.ones(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions2 = torch.ones(2, 1, 2, 4, **device_cfg.as_torch_dict())
        quaternions2 = quaternions2 / torch.linalg.norm(quaternions2, dim=-1, keepdim=True)
        tp2 = ToolPose(tool_frames=link_names2, position=positions2, quaternion=quaternions2)

        tp1.copy_(tp2)

        assert tp1.tool_frames == link_names2
        assert torch.allclose(tp1.position, positions2)

    def test_requires_grad_(self, device_cfg):
        """Test setting requires_grad."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        tp.requires_grad_(True)

        assert tp.position.requires_grad
        assert tp.quaternion.requires_grad

    def test_requires_grad_false(self, device_cfg):
        """Test disabling requires_grad."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        tp.requires_grad_(True)
        assert tp.position.requires_grad

        tp.requires_grad_(False)

        assert not tp.position.requires_grad
        assert not tp.quaternion.requires_grad

    def test_get_link_pose_preserves_name(self, device_cfg):
        """Test that get_link_pose preserves link name."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        link_pose = tp.get_link_pose("link1")

        assert link_pose.name == "link1"

    def test_n_horizon_property(self, device_cfg):
        """Test horizon property."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 10, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 10, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert tp.horizon == 10

    def test_n_links_property(self, device_cfg):
        """Test num_links property."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert tp.num_links == 3

    def test_contiguous(self, device_cfg):
        """Test contiguous returns contiguous tensors."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        tp_contig = tp.contiguous()

        assert tp_contig.position.is_contiguous()
        assert tp_contig.quaternion.is_contiguous()

    def test_reorder_links(self, device_cfg):
        """Test reordering links."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        new_order = ["link3", "link1", "link2"]
        tp_reordered = tp.reorder_links(new_order)

        assert tp_reordered.tool_frames == new_order

    def test_reorder_links_same_order_returns_self(self, device_cfg):
        """Test that reorder_links with same order returns self."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        tp_reordered = tp.reorder_links(tool_frames)

        assert tp_reordered is tp

    def test_reorder_links_invalid_subset_raises_error(self, device_cfg):
        """Test that reorder_links with invalid subset raises error."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        with pytest.raises(Exception):
            tp.reorder_links(["link1", "link2", "link3"])

    def test_reorder_links_preserves_data(self, device_cfg):
        """Test that reorder_links preserves pose data correctly."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        link1_pose_before = tp.get_link_pose("link1")

        new_order = ["link3", "link1", "link2"]
        tp_reordered = tp.reorder_links(new_order)

        link1_pose_after = tp_reordered.get_link_pose("link1")

        assert torch.allclose(link1_pose_before.position, link1_pose_after.position)

    def test_as_goal(self, device_cfg):
        """Test as_goal converts 4D ToolPose to 5D GoalToolPose with num_goalset=1."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        gtp = tp.as_goal()

        assert isinstance(gtp, GoalToolPose)
        assert gtp.batch_size == 2
        assert gtp.num_goalset == 1
        assert gtp.num_links == 2
        assert gtp.tool_frames == tool_frames
        assert torch.allclose(gtp.position.squeeze(3), positions)
        assert torch.allclose(gtp.quaternion.squeeze(3), quaternions)

    def test_as_goal_with_reorder(self, device_cfg):
        """Test as_goal with link reordering."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, **device_cfg.as_torch_dict())

        tp = ToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        gtp = tp.as_goal(ordered_tool_frames=["link3", "link1"])

        assert gtp.tool_frames == ["link3", "link1"]
        assert gtp.num_links == 2
        assert gtp.num_goalset == 1


class TestGoalToolPose:
    """Test GoalToolPose class: 5D goal specification [batch, horizon, num_links, num_goalset, 3/4]."""

    def test_initialization(self, device_cfg):
        """Test GoalToolPose initialization with 5D tensors."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert gtp.tool_frames == tool_frames
        assert gtp.batch_size == 2
        assert gtp.num_goalset == 3

    def test_initialization_rejects_non_5d(self, device_cfg):
        """Test that __post_init__ rejects non-5D tensors."""
        tool_frames = ["link1", "link2"]
        positions_4d = torch.rand(2, 1, 2, 3, **device_cfg.as_torch_dict())
        quaternions_4d = torch.rand(2, 1, 2, 4, **device_cfg.as_torch_dict())

        with pytest.raises(Exception):
            GoalToolPose(
                tool_frames=tool_frames, position=positions_4d, quaternion=quaternions_4d
            )

    def test_initialization_rejects_link_mismatch(self, device_cfg):
        """Test that __post_init__ rejects mismatched num_links and tool_frames."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 3, 1, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 1, 4, **device_cfg.as_torch_dict())

        with pytest.raises(Exception):
            GoalToolPose(
                tool_frames=tool_frames, position=positions, quaternion=quaternions
            )

    def test_n_goalset_property(self, device_cfg):
        """Test num_goalset property."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert gtp.num_goalset == 3

    def test_batch_size_property(self, device_cfg):
        """Test batch_size property."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(5, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(5, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert gtp.batch_size == 5

    def test_get_link_pose(self, device_cfg):
        """Test getting a specific link pose flattens [B*H*G, 3/4]."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 4, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        link1_pose = gtp.get_link_pose("link1")
        assert link1_pose.position.shape == (8, 3)
        assert link1_pose.quaternion.shape == (8, 4)

    def test_get_link_pose_invalid_name(self, device_cfg):
        """Test getting link pose with invalid name raises error."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        with pytest.raises(Exception):
            gtp.get_link_pose("invalid_link")

    def test_get_link_pose_with_contiguous(self, device_cfg):
        """Test getting link pose with contiguous flag."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        link_pose = gtp.get_link_pose("link1", make_contiguous=True)
        assert link_pose.position.is_contiguous()
        assert link_pose.quaternion.is_contiguous()

    def test_clone(self, device_cfg):
        """Test cloning GoalToolPose produces an independent copy."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        gtp_clone = gtp.clone()

        assert isinstance(gtp_clone, GoalToolPose)
        assert gtp_clone.tool_frames == gtp.tool_frames
        assert gtp_clone.tool_frames is not gtp.tool_frames
        assert torch.allclose(gtp_clone.position, gtp.position)

        gtp_clone.position[0, 0, 0, 0, 0] = 999.0
        assert gtp.position[0, 0, 0, 0, 0] != 999.0

    def test_from_poses(self, device_cfg):
        """Test from_poses builds GoalToolPose from 2D Poses (num_goalset=1)."""
        tool_frames = ["link1", "link2", "link3"]
        pose_dict = {}
        for link_name in tool_frames:
            position = torch.rand(2, 3, **device_cfg.as_torch_dict())
            quaternion = torch.rand(2, 4, **device_cfg.as_torch_dict())
            pose_dict[link_name] = Pose(
                position=position, quaternion=quaternion, normalize_rotation=True
            )

        gtp = GoalToolPose.from_poses(pose_dict)

        assert gtp.tool_frames == tool_frames
        assert gtp.batch_size == 2
        assert gtp.num_goalset == 1

    def test_from_poses_with_ordered_link_names(self, device_cfg):
        """Test from_poses with custom ordering."""
        pose_dict = {}
        for link_name in ["link3", "link1", "link2"]:
            position = torch.rand(2, 3, **device_cfg.as_torch_dict())
            quaternion = torch.rand(2, 4, **device_cfg.as_torch_dict())
            pose_dict[link_name] = Pose(
                position=position, quaternion=quaternion, normalize_rotation=True
            )

        ordered_names = ["link1", "link2", "link3"]
        gtp = GoalToolPose.from_poses(pose_dict, ordered_tool_frames=ordered_names)

        assert gtp.tool_frames == ordered_names

    def test_from_poses_empty_raises_error(self, device_cfg):
        """Test that from_poses with empty dict raises error."""
        with pytest.raises(Exception):
            GoalToolPose.from_poses({})

    def test_from_poses_missing_ordered_links_raises_error(self, device_cfg):
        """Test that from_poses with missing ordered links raises error."""
        pose_dict = {}
        for link_name in ["link1", "link2"]:
            position = torch.rand(2, 3, **device_cfg.as_torch_dict())
            quaternion = torch.rand(2, 4, **device_cfg.as_torch_dict())
            pose_dict[link_name] = Pose(
                position=position, quaternion=quaternion, normalize_rotation=True
            )

        ordered_names = ["link1", "link2", "link3"]
        with pytest.raises(Exception):
            GoalToolPose.from_poses(pose_dict, ordered_tool_frames=ordered_names)

    def test_from_poses_preserves_pose_data(self, device_cfg):
        """Test that from_poses preserves pose data correctly."""
        tool_frames = ["link1", "link2"]
        pose_dict = {}
        positions = {}
        for link_name in tool_frames:
            position = torch.rand(2, 3, **device_cfg.as_torch_dict())
            quaternion = torch.rand(2, 4, **device_cfg.as_torch_dict())
            positions[link_name] = position.clone()
            pose_dict[link_name] = Pose(
                position=position, quaternion=quaternion, normalize_rotation=True
            )

        gtp = GoalToolPose.from_poses(pose_dict)

        link1_pose = gtp.get_link_pose("link1")
        assert torch.allclose(link1_pose.position, positions["link1"])

    def test_from_poses_with_goalset(self, device_cfg):
        """Test from_poses with num_goalset > 1."""
        tool_frames = ["link1", "link2"]
        num_goalset = 3
        batch = 2
        pose_dict = {}
        for link_name in tool_frames:
            position = torch.rand(batch * num_goalset, 3, **device_cfg.as_torch_dict())
            quaternion = torch.rand(batch * num_goalset, 4, **device_cfg.as_torch_dict())
            pose_dict[link_name] = Pose(
                position=position, quaternion=quaternion, normalize_rotation=True
            )

        gtp = GoalToolPose.from_poses(pose_dict, num_goalset=num_goalset)

        assert gtp.batch_size == batch
        assert gtp.num_goalset == num_goalset
        assert gtp.position.shape == (batch, 1, len(tool_frames), num_goalset, 3)

    def test_reorder_links(self, device_cfg):
        """Test reordering links."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 4, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        new_order = ["link3", "link1", "link2"]
        gtp_reordered = gtp.reorder_links(new_order)

        assert gtp_reordered.tool_frames == new_order

    def test_reorder_links_same_order_returns_self(self, device_cfg):
        """Test that reorder_links with same order returns self."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 4, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        gtp_reordered = gtp.reorder_links(tool_frames)

        assert gtp_reordered is gtp

    def test_reorder_links_invalid_subset_raises_error(self, device_cfg):
        """Test that reorder_links with invalid subset raises error."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        with pytest.raises(Exception):
            gtp.reorder_links(["link1", "link2", "link3"])

    def test_reorder_links_preserves_data(self, device_cfg):
        """Test that reorder_links preserves pose data correctly."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 4, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        link1_pose_before = gtp.get_link_pose("link1")

        new_order = ["link3", "link1", "link2"]
        gtp_reordered = gtp.reorder_links(new_order)

        link1_pose_after = gtp_reordered.get_link_pose("link1")

        assert torch.allclose(link1_pose_before.position, link1_pose_after.position)

    def test_to_dict(self, device_cfg):
        """Test that to_dict returns per-link 2D Poses."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        pose_dict = gtp.to_dict()
        assert len(pose_dict) == 2
        assert "link1" in pose_dict
        assert "link2" in pose_dict
        assert isinstance(pose_dict["link1"], Pose)

    def test_detach(self, device_cfg):
        """Test detaching gradients from GoalToolPose."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict(), requires_grad=True)
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict(), requires_grad=True)

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        gtp_detached = gtp.detach()

        assert not gtp_detached.position.requires_grad
        assert not gtp_detached.quaternion.requires_grad

    def test_getitem_with_string(self, device_cfg):
        """Test indexing with string returns 2D Pose for that link."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 4, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        link_pose = gtp["link2"]
        assert isinstance(link_pose, Pose)

    def test_getitem_with_int(self, device_cfg):
        """Test indexing with integer returns batch-sliced GoalToolPose."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(5, 1, 3, 4, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(5, 1, 3, 4, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        gtp_indexed = gtp[2]
        assert isinstance(gtp_indexed, GoalToolPose)
        assert gtp_indexed.tool_frames == tool_frames
        assert gtp_indexed.batch_size == 1

    def test_len(self, device_cfg):
        """Test length equals number of links."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 4, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert len(gtp) == 3

    def test_shape_property(self, device_cfg):
        """Test shape property returns full 5D shape."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(5, 1, 3, 4, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(5, 1, 3, 4, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert gtp.shape == (5, 1, 3, 4, 3)

    def test_ndim_property(self, device_cfg):
        """Test ndim property is always 5."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(5, 1, 3, 4, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(5, 1, 3, 4, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert gtp.ndim == 5

    def test_n_horizon_property(self, device_cfg):
        """Test horizon property."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 10, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 10, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert gtp.horizon == 10

    def test_n_links_property(self, device_cfg):
        """Test num_links property."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(2, 1, 3, 4, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 3, 4, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert gtp.num_links == 3

    def test_device_property(self, device_cfg):
        """Test device property returns tensor device."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        assert gtp.device == positions.device

    def test_copy_(self, device_cfg):
        """Test in-place copy from another GoalToolPose."""
        tool_frames1 = ["link1", "link2"]
        positions1 = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions1 = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())
        gtp1 = GoalToolPose(tool_frames=tool_frames1, position=positions1, quaternion=quaternions1)

        tool_frames2 = ["link3", "link4"]
        positions2 = torch.ones(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions2 = torch.ones(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())
        quaternions2 = quaternions2 / torch.linalg.norm(quaternions2, dim=-1, keepdim=True)
        gtp2 = GoalToolPose(tool_frames=tool_frames2, position=positions2, quaternion=quaternions2)

        gtp1.copy_(gtp2)

        assert gtp1.tool_frames == tool_frames2
        assert torch.allclose(gtp1.position, positions2)

    def test_requires_grad_(self, device_cfg):
        """Test enabling requires_grad."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        gtp.requires_grad_(True)

        assert gtp.position.requires_grad
        assert gtp.quaternion.requires_grad

    def test_requires_grad_false(self, device_cfg):
        """Test disabling requires_grad."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        gtp.requires_grad_(True)
        assert gtp.position.requires_grad

        gtp.requires_grad_(False)

        assert not gtp.position.requires_grad
        assert not gtp.quaternion.requires_grad

    def test_getitem_with_slice(self, device_cfg):
        """Test indexing with slice returns batch-sliced GoalToolPose."""
        tool_frames = ["link1", "link2", "link3"]
        positions = torch.rand(5, 1, 3, 4, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(5, 1, 3, 4, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        gtp_sliced = gtp[1:3]
        assert isinstance(gtp_sliced, GoalToolPose)
        assert gtp_sliced.tool_frames == tool_frames
        assert gtp_sliced.position.shape[0] == 2
        assert torch.allclose(gtp_sliced.position, positions[1:3])

    def test_getitem_with_tensor(self, device_cfg):
        """Test indexing with boolean mask returns filtered GoalToolPose."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(4, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(4, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)

        mask = torch.tensor([True, False, True, False])
        gtp_filtered = gtp[mask]
        assert isinstance(gtp_filtered, GoalToolPose)
        assert gtp_filtered.position.shape[0] == 2

    def test_get_link_pose_preserves_name(self, device_cfg):
        """Test that get_link_pose sets the name field on returned Pose."""
        tool_frames = ["link1", "link2"]
        positions = torch.rand(2, 1, 2, 3, 3, **device_cfg.as_torch_dict())
        quaternions = torch.rand(2, 1, 2, 3, 4, **device_cfg.as_torch_dict())

        gtp = GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)
        link_pose = gtp.get_link_pose("link1")

        assert link_pose.name == "link1"

    def test_from_poses_to_dict_round_trip(self, device_cfg):
        """Test that from_poses -> to_dict -> from_poses preserves data."""
        tool_frames = ["link1", "link2"]
        pose_dict = {}
        for link_name in tool_frames:
            position = torch.rand(2, 3, **device_cfg.as_torch_dict())
            quaternion = torch.rand(2, 4, **device_cfg.as_torch_dict())
            quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True)
            pose_dict[link_name] = Pose(
                position=position, quaternion=quaternion, normalize_rotation=False,
            )

        gtp1 = GoalToolPose.from_poses(pose_dict, ordered_tool_frames=tool_frames)
        roundtrip_dict = gtp1.to_dict()
        gtp2 = GoalToolPose.from_poses(roundtrip_dict, ordered_tool_frames=tool_frames)

        assert torch.allclose(gtp1.position, gtp2.position, atol=1e-6)
        assert torch.allclose(gtp1.quaternion, gtp2.quaternion, atol=1e-6)
