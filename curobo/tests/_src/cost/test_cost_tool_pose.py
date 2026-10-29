# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for ToolPoseCost and ToolPoseCostCfg."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.tool_pose_criteria import PoseErrorFrame, ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.warp import init_warp


@pytest.fixture(scope="module", autouse=True)
def setup_warp():
    """Initialize warp before running any tests in this module."""
    init_warp(quiet=True)


@pytest.fixture(params=["cuda:0"])
def device_cfg(request):
    """Create tensor configuration for GPU only (warp requires CUDA)."""
    device = request.param
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device(device))


class TestToolPoseCostCfg:
    """Test ToolPoseCostCfg class."""

    def test_default_init(self, device_cfg):
        """Test default initialization."""
        cfg = ToolPoseCostCfg(weight=[1.0, 1.0], device_cfg=device_cfg)
        assert cfg.class_type == ToolPoseCost
        assert cfg.tool_frames is None
        assert isinstance(cfg.default_tool_pose_criteria, ToolPoseCriteria)
        assert not hasattr(cfg, "use_generalized_pose_metric")

    def test_init_with_tool_frames(self, device_cfg):
        """Test initialization with tool_frames."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        assert cfg.tool_frames == tool_frames
        assert cfg.num_links == 2
        assert "ee_link" in cfg.tool_pose_criteria
        assert "gripper_link" in cfg.tool_pose_criteria

    def test_set_tool_frames(self, device_cfg):
        """Test set_tool_frames method."""
        cfg = ToolPoseCostCfg(weight=[1.0, 1.0], device_cfg=device_cfg)
        tool_frames = ["link1", "link2", "link3"]
        cfg.set_tool_frames(tool_frames)
        assert cfg.tool_frames == tool_frames
        assert cfg.num_links == 3

    def test_clone(self, device_cfg):
        """Test clone method."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=tool_frames,
            use_lie_group=True,
            use_grad_input=True,
            device_cfg=device_cfg,
        )
        cloned = cfg.clone()
        assert cloned is not cfg
        assert cloned.tool_frames == cfg.tool_frames
        assert cloned.tool_frames is not cfg.tool_frames
        assert cloned.use_grad_input is True
        assert not hasattr(cloned, "use_generalized_pose_metric")
        assert "use_lie_group" not in cloned.__dict__
        assert torch.allclose(cloned.weight, cfg.weight)

    def test_use_lie_group_is_compatibility_only(self, device_cfg):
        """Test that the removed rotation switch is accepted but not stored."""
        cfg = ToolPoseCostCfg(weight=[1.0, 1.0], use_lie_group=True, device_cfg=device_cfg)
        assert "use_lie_group" not in cfg.__dict__
        assert not hasattr(ToolPoseCostCfg, "rotation_method")


class TestToolPoseCost:
    """Test ToolPoseCost class."""

    def test_init(self, device_cfg):
        """Test ToolPoseCost initialization."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)
        assert cost is not None
        assert cost.config == cfg
        assert cost.num_links == 1
        assert cost.tool_frames == tool_frames
        assert not hasattr(cost, "use_generalized_pose_metric")

    def test_quadratic_pose_metrics_use_stable_device_storage(self, device_cfg):
        """Criteria updates refresh persistent quadratic-metric tensors in place."""
        cost = ToolPoseCost(
            ToolPoseCostCfg(
                weight=[1.0, 1.0],
                tool_frames=["ee_link"],
                device_cfg=device_cfg,
            )
        )
        position_metric_pointer = cost._position_metric.data_ptr()
        rotation_metric_pointer = cost._rotation_metric.data_ptr()
        tolerance_pointer = cost._pose_tolerance.data_ptr()
        pose_active_pointer = cost._pose_active.data_ptr()
        torch.testing.assert_close(cost._position_metric[0], torch.eye(3, device=device_cfg.device))
        torch.testing.assert_close(cost._rotation_metric[0], torch.eye(3, device=device_cfg.device))
        torch.testing.assert_close(cost._position_metric[1], torch.zeros((3, 3), device=device_cfg.device))
        torch.testing.assert_close(cost._rotation_metric[1], torch.zeros((3, 3), device=device_cfg.device))
        torch.testing.assert_close(
            cost._pose_active,
            torch.tensor([1, 0], device=device_cfg.device, dtype=torch.uint8),
        )

        direction = torch.tensor([1.0, -2.0, 0.5], **device_cfg.as_torch_dict())
        direction /= torch.linalg.vector_norm(direction)
        projector = torch.eye(3, **device_cfg.as_torch_dict()) - torch.outer(
            direction, direction
        )
        position_scale = torch.tensor([2.0, 0.5, 1.5], **device_cfg.as_torch_dict())
        rotation_scale = torch.tensor([0.75, 1.25, 2.0], **device_cfg.as_torch_dict())
        updated_criteria = ToolPoseCriteria(
            terminal_position_projector=projector,
            terminal_rotation_projector=projector,
            terminal_position_axis_scale=position_scale,
            terminal_rotation_axis_scale=rotation_scale,
            terminal_tolerance=[0.1, 0.2],
            error_frame=PoseErrorFrame.GOAL,
            device_cfg=device_cfg,
        )
        cost.update_tool_pose_criteria({"ee_link": updated_criteria})
        scaled_position_projector = position_scale.unsqueeze(-1) * projector
        scaled_rotation_projector = rotation_scale.unsqueeze(-1) * projector
        expected_position_metric = scaled_position_projector.T @ scaled_position_projector
        expected_rotation_metric = scaled_rotation_projector.T @ scaled_rotation_projector
        torch.testing.assert_close(cost._position_metric[0], expected_position_metric)
        torch.testing.assert_close(cost._rotation_metric[0], expected_rotation_metric)
        torch.testing.assert_close(
            cost._pose_tolerance[0],
            torch.tensor([0.1, 0.2], **device_cfg.as_torch_dict()),
        )
        assert cost._position_metric.data_ptr() == position_metric_pointer
        assert cost._rotation_metric.data_ptr() == rotation_metric_pointer
        assert cost._pose_tolerance.data_ptr() == tolerance_pointer
        assert cost._pose_active.data_ptr() == pose_active_pointer
        torch.testing.assert_close(
            cost._pose_active,
            torch.tensor([1, 0], device=device_cfg.device, dtype=torch.uint8),
        )

    def test_quadratic_pose_metric_updates_after_cuda_graph_capture(self, device_cfg):
        """A captured kernel reads updated quadratic metrics without recapture."""
        tool_frames = ["ee_link"]
        cost = ToolPoseCost(
            ToolPoseCostCfg(
                weight=[1.4, 0.9],
                tool_frames=tool_frames,
                default_tool_pose_criteria=ToolPoseCriteria.track_pose(
                    non_terminal_scale=1.0,
                    device_cfg=device_cfg,
                ),
                device_cfg=device_cfg,
            )
        )
        batch_size = 4
        horizon = 16
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)
        current_position = torch.randn(
            (batch_size, horizon, 1, 3), **device_cfg.as_torch_dict()
        )
        current_quaternion = torch.randn(
            (batch_size, horizon, 1, 4), **device_cfg.as_torch_dict()
        )
        current_quaternion /= torch.linalg.vector_norm(
            current_quaternion, dim=-1, keepdim=True
        )
        goal_position = torch.randn((1, 1, 1, 1, 3), **device_cfg.as_torch_dict())
        goal_quaternion = torch.randn((1, 1, 1, 1, 4), **device_cfg.as_torch_dict())
        goal_quaternion /= torch.linalg.vector_norm(goal_quaternion, dim=-1, keepdim=True)
        current_pose = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )
        goal_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )
        goal_index = torch.zeros(
            (batch_size, 1), device=device_cfg.device, dtype=torch.int32
        )

        def evaluate():
            return cost.forward(
                current_tool_poses=current_pose,
                goal_tool_poses=goal_pose,
                idxs_goal=goal_index,
            )[0]

        evaluate()
        torch.cuda.synchronize(device_cfg.device)
        capture_stream = torch.cuda.Stream(device=device_cfg.device)
        capture_stream.wait_stream(torch.cuda.current_stream(device_cfg.device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            captured_cost = evaluate()
        torch.cuda.current_stream(device_cfg.device).wait_stream(capture_stream)

        graph.replay()
        original_cost = captured_cost.clone()
        cost.update_tool_pose_criteria(
            {
                "ee_link": ToolPoseCriteria(
                    terminal_position_axis_scale=[2.0, 1.0, 0.5],
                    terminal_rotation_axis_scale=[0.5, 1.5, 2.0],
                    device_cfg=device_cfg,
                )
            }
        )
        graph.replay()
        updated_graph_cost = captured_cost.clone()
        updated_eager_cost = evaluate().clone()

        assert not torch.allclose(original_cost, updated_graph_cost)
        torch.testing.assert_close(
            updated_graph_cost, updated_eager_cost, atol=2.0e-5, rtol=2.0e-5
        )
        assert torch.count_nonzero(original_cost[:, :-1]).item() > 0
        torch.testing.assert_close(
            updated_graph_cost[:, :-1],
            torch.zeros_like(updated_graph_cost[:, :-1]),
        )

        cost.update_tool_pose_criteria(
            {"ee_link": ToolPoseCriteria.disabled(device_cfg=device_cfg)}
        )
        graph.replay()
        disabled_graph_cost = captured_cost.clone()
        disabled_eager_cost = evaluate().clone()
        torch.testing.assert_close(disabled_graph_cost, torch.zeros_like(disabled_graph_cost))
        torch.testing.assert_close(disabled_graph_cost, disabled_eager_cost)

    def test_setup_batch_tensors(self, device_cfg):
        """Test setup_batch_tensors."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)
        cost.setup_batch_tensors(batch_size=4, horizon=10)

        num_links = len(tool_frames)
        assert cost._batch_size == 4
        assert cost._horizon == 10
        assert cost._out_distance.shape == (4, 10, 2 * num_links)
        assert cost._out_position_distance.shape == (4, 10, num_links)
        assert cost._out_rotation_distance.shape == (4, 10, num_links)
        assert cost._out_position_gradient.shape == (4, 10, num_links, 3)
        assert cost._out_rotation_gradient.shape == (4, 10, num_links, 4)

    def test_forward_single_link(self, device_cfg):
        """Test forward with a single link."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)
        batch_size = 4
        horizon = 10
        num_links = 1
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create current poses (shape: batch, horizon, num_links, 3/4)
        current_position = torch.zeros(
            (batch_size, horizon, num_links, 3), **device_cfg.as_torch_dict()
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4), **device_cfg.as_torch_dict()
        )
        current_quaternion[..., 0] = 1.0
        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Create goal poses (shape: B, H, num_links, num_goalset, 3/4)
        goal_position = torch.zeros((1, 1, num_links, 1, 3), **device_cfg.as_torch_dict())
        goal_quaternion = torch.zeros((1, 1, num_links, 1, 4), **device_cfg.as_torch_dict())
        goal_quaternion[..., 0] = 1.0
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros((batch_size, 1), device=device_cfg.device, dtype=torch.int32)

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        assert cost_val.shape == (batch_size, horizon, 2 * num_links)
        assert linear_dist.shape == (batch_size, horizon, num_links)
        assert angular_dist.shape == (batch_size, horizon, num_links)

    def test_forward_multiple_links(self, device_cfg):
        """Test forward with multiple links."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)
        batch_size = 4
        horizon = 10
        num_links = 2
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create current poses (shape: batch, horizon, num_links, 3/4)
        current_position = torch.zeros(
            (batch_size, horizon, num_links, 3), **device_cfg.as_torch_dict()
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4), **device_cfg.as_torch_dict()
        )
        current_quaternion[..., 0] = 1.0
        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Create goal poses (shape: B, H, num_links, num_goalset, 3/4)
        goal_position = torch.zeros((1, 1, num_links, 1, 3), **device_cfg.as_torch_dict())
        goal_quaternion = torch.zeros((1, 1, num_links, 1, 4), **device_cfg.as_torch_dict())
        goal_quaternion[..., 0] = 1.0
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros((batch_size, 1), device=device_cfg.device, dtype=torch.int32)

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        assert cost_val.shape == (batch_size, horizon, 2 * num_links)
        assert linear_dist.shape == (batch_size, horizon, num_links)
        assert angular_dist.shape == (batch_size, horizon, num_links)

    def test_forward_missing_poses_raises_error(self, device_cfg):
        """Test that missing poses raises error."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)
        cost.setup_batch_tensors(batch_size=4, horizon=10)

        with pytest.raises(Exception):
            cost.forward(
                current_tool_poses=None, goal_tool_poses=None, idxs_goal=None
            )

    def test_forward_rejects_consistently_reordered_tool_frames(self, device_cfg):
        """Criteria remain aligned to the configured tool-frame order."""
        configured_frames = ["ee_link", "gripper_link"]
        runtime_frames = list(reversed(configured_frames))
        cost = ToolPoseCost(
            ToolPoseCostCfg(
                weight=[1.0, 1.0],
                tool_frames=configured_frames,
                device_cfg=device_cfg,
            )
        )
        cost.setup_batch_tensors(batch_size=1, horizon=1)
        current_quaternion = torch.zeros((1, 1, 2, 4), **device_cfg.as_torch_dict())
        current_quaternion[..., 0] = 1.0
        goal_quaternion = torch.zeros((1, 1, 2, 1, 4), **device_cfg.as_torch_dict())
        goal_quaternion[..., 0] = 1.0

        with pytest.raises(ValueError, match="configured tool_frames"):
            cost.forward(
                current_tool_poses=ToolPose(
                    tool_frames=runtime_frames,
                    position=torch.zeros((1, 1, 2, 3), **device_cfg.as_torch_dict()),
                    quaternion=current_quaternion,
                ),
                goal_tool_poses=GoalToolPose(
                    tool_frames=runtime_frames,
                    position=torch.zeros((1, 1, 2, 1, 3), **device_cfg.as_torch_dict()),
                    quaternion=goal_quaternion,
                ),
                idxs_goal=torch.zeros((1, 1), dtype=torch.int32, device=device_cfg.device),
            )

    def test_update_tool_pose_criteria(self, device_cfg):
        """Test update_tool_pose_criteria method."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)

        # Create new criteria
        new_criteria = {
            "ee_link": ToolPoseCriteria(
                device_cfg=device_cfg,
                terminal_position_axis_scale=[2.0, 2.0, 2.0],
            ),
            "gripper_link": ToolPoseCriteria(
                device_cfg=device_cfg,
                terminal_rotation_axis_scale=[0.5, 0.5, 0.5],
            ),
        }

        # Update should not raise
        cost.update_tool_pose_criteria(new_criteria)

    def test_common_pose_metric(self, device_cfg):
        """Test the single common pose metric with a position offset."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=tool_frames,
            device_cfg=device_cfg,
        )
        cost = ToolPoseCost(cfg)
        batch_size = 4
        horizon = 10
        num_links = 1
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create poses with offset (shape: batch, horizon, num_links, 3/4)
        current_position = torch.ones(
            (batch_size, horizon, num_links, 3), **device_cfg.as_torch_dict()
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4), **device_cfg.as_torch_dict()
        )
        current_quaternion[..., 0] = 1.0
        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Goal poses (shape: B, H, num_links, num_goalset, 3/4)
        goal_position = torch.zeros((1, 1, num_links, 1, 3), **device_cfg.as_torch_dict())
        goal_quaternion = torch.zeros((1, 1, num_links, 1, 4), **device_cfg.as_torch_dict())
        goal_quaternion[..., 0] = 1.0
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros((batch_size, 1), device=device_cfg.device, dtype=torch.int32)

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        assert cost_val.shape == (batch_size, horizon, 2 * num_links)
        # Terminal cost should be non-zero due to position offset
        assert torch.any(cost_val[:, -1, :] > 0)


class TestToolPoseCostGradients:
    """Test gradient computation for ToolPoseCost."""

    @pytest.fixture
    def gradient_device_cfg(self):
        """Create tensor configuration for gradient tests (requires CUDA)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return DeviceCfg(device=torch.device("cuda:0"))

    def test_tool_pose_cost_gradient_position(self, gradient_device_cfg):
        """Test that gradients flow correctly through position inputs."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=tool_frames,
            use_grad_input=True,
            device_cfg=gradient_device_cfg,
        )
        cost = ToolPoseCost(cfg)
        batch_size = 2
        horizon = 3
        num_links = 1
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create current poses with requires_grad
        current_position = torch.randn(
            (batch_size, horizon, num_links, 3),
            dtype=torch.float32,
            device=gradient_device_cfg.device,
            requires_grad=True,
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4),
            dtype=torch.float32,
            device=gradient_device_cfg.device,
        )
        current_quaternion[..., 0] = 1.0
        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Create goal poses (fixed, 5D: B, H, L, G, 3/4)
        goal_position = torch.zeros(
            (1, 1, num_links, 1, 3), dtype=torch.float32, device=gradient_device_cfg.device
        )
        goal_quaternion = torch.zeros(
            (1, 1, num_links, 1, 4), dtype=torch.float32, device=gradient_device_cfg.device
        )
        goal_quaternion[..., 0] = 1.0
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros(
            (batch_size, 1), device=gradient_device_cfg.device, dtype=torch.int32
        )

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        # Sum the cost and backpropagate
        loss = cost_val.sum()
        loss.backward()

        # Check that gradients exist
        assert current_position.grad is not None
        assert current_position.grad.shape == current_position.shape

    def test_tool_pose_cost_gradient_quaternion(self, gradient_device_cfg):
        """Test that gradients flow correctly through quaternion inputs."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=tool_frames,
            use_grad_input=True,
            device_cfg=gradient_device_cfg,
        )
        cost = ToolPoseCost(cfg)
        batch_size = 2
        horizon = 3
        num_links = 1
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create current poses with requires_grad on quaternion
        current_position = torch.randn(
            (batch_size, horizon, num_links, 3),
            dtype=torch.float32,
            device=gradient_device_cfg.device,
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4),
            dtype=torch.float32,
            device=gradient_device_cfg.device,
            requires_grad=True,
        )
        # Manually set w=1 for identity quaternion (detach and modify)
        with torch.no_grad():
            current_quaternion.data[..., 0] = 1.0

        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Create goal poses with rotation offset (5D: B, H, L, G, 3/4)
        goal_position = torch.zeros(
            (1, 1, num_links, 1, 3), dtype=torch.float32, device=gradient_device_cfg.device
        )
        # Small rotation around z-axis
        goal_quaternion = torch.tensor(
            [[[[[0.995, 0.0, 0.0, 0.1]]]]],
            dtype=torch.float32,
            device=gradient_device_cfg.device,
        )
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros(
            (batch_size, 1), device=gradient_device_cfg.device, dtype=torch.int32
        )

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        # Sum the cost and backpropagate
        loss = cost_val.sum()
        loss.backward()

        # Check that gradients exist for quaternion
        assert current_quaternion.grad is not None
        assert current_quaternion.grad.shape == current_quaternion.shape

    def test_tool_pose_cost_multiple_links_gradient(self, gradient_device_cfg):
        """Test gradients with multiple tool frames."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=tool_frames,
            use_grad_input=True,
            device_cfg=gradient_device_cfg,
        )
        cost = ToolPoseCost(cfg)
        batch_size = 2
        horizon = 3
        num_links = 2
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create current poses with requires_grad
        current_position = torch.randn(
            (batch_size, horizon, num_links, 3),
            dtype=torch.float32,
            device=gradient_device_cfg.device,
            requires_grad=True,
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4),
            dtype=torch.float32,
            device=gradient_device_cfg.device,
            requires_grad=True,
        )
        with torch.no_grad():
            current_quaternion.data[..., 0] = 1.0

        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Create goal poses (fixed, 5D: B, H, L, G, 3/4)
        goal_position = torch.zeros(
            (1, 1, num_links, 1, 3), dtype=torch.float32, device=gradient_device_cfg.device
        )
        goal_quaternion = torch.zeros(
            (1, 1, num_links, 1, 4), dtype=torch.float32, device=gradient_device_cfg.device
        )
        goal_quaternion[..., 0] = 1.0
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros(
            (batch_size, 1), device=gradient_device_cfg.device, dtype=torch.int32
        )

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        # Sum the cost and backpropagate
        loss = cost_val.sum()
        loss.backward()

        # Check that gradients exist for both position and quaternion
        assert current_position.grad is not None
        assert current_quaternion.grad is not None
        assert current_position.grad.shape == current_position.shape
        assert current_quaternion.grad.shape == current_quaternion.shape
