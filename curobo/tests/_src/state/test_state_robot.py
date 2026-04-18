# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for RobotState."""

# Standard Library
from typing import List

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg


class TestRobotState:
    """Test RobotState class."""

    @pytest.fixture
    def joint_names(self) -> List[str]:
        """Get sample joint names."""
        return ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

    @pytest.fixture
    def device_cfg(self) -> DeviceCfg:
        """Get tensor device configuration."""
        return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)

    @pytest.fixture
    def sample_joint_state(self, joint_names, device_cfg) -> JointState:
        """Create sample joint state."""
        position = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        acceleration = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        jerk = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        return JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            joint_names=joint_names,
            device_cfg=device_cfg,
        )

    @pytest.fixture
    def sample_robot_state(self, sample_joint_state) -> RobotState:
        """Create sample robot state."""
        return RobotState(joint_state=sample_joint_state)

    def test_initialization_basic(self, sample_joint_state):
        """Test basic RobotState initialization."""
        rs = RobotState(joint_state=sample_joint_state)
        assert rs.joint_state is not None
        assert torch.allclose(rs.joint_state.position, sample_joint_state.position)

    def test_initialization_with_joint_torque(self, sample_joint_state, device_cfg):
        """Test RobotState initialization with joint torque."""
        joint_torque = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        rs = RobotState(joint_state=sample_joint_state, joint_torque=joint_torque)

        assert rs.joint_torque is not None
        assert torch.allclose(rs.joint_torque, joint_torque)

    def test_data_ptr(self, sample_robot_state, sample_joint_state):
        """Test data_ptr method."""
        ptr = sample_robot_state.data_ptr()
        assert ptr == sample_joint_state.position.data_ptr()

    def test_len(self, sample_robot_state, sample_joint_state):
        """Test length of RobotState."""
        assert len(sample_robot_state) == len(sample_joint_state)

    def test_detach(self, sample_robot_state):
        """Test detaching RobotState from computation graph."""
        detached = sample_robot_state.detach()
        assert not detached.joint_state.position.requires_grad
        assert not detached.joint_state.velocity.requires_grad

    def test_detach_with_joint_torque(self, sample_joint_state, device_cfg):
        """Test detaching RobotState with joint torque."""
        joint_torque = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        rs = RobotState(joint_state=sample_joint_state, joint_torque=joint_torque)

        detached = rs.detach()
        assert not detached.joint_torque.requires_grad

    def test_getitem_int(self, device_cfg, joint_names):
        """Test indexing RobotState with integer."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)
        rs = RobotState(joint_state=joint_state)

        rs_indexed = rs[0]
        assert rs_indexed.joint_state.position.shape == (7,)
        assert torch.allclose(rs_indexed.joint_state.position, position[0])

    def test_getitem_tensor(self, device_cfg, joint_names):
        """Test indexing RobotState with tensor."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)
        joint_torque = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        rs = RobotState(joint_state=joint_state, joint_torque=joint_torque)

        idx = torch.tensor([0, 2, 4], device=device_cfg.device, dtype=torch.long)
        rs_indexed = rs[idx]
        assert rs_indexed.joint_state.position.shape == (3, 7)
        assert torch.allclose(rs_indexed.joint_state.position, position[idx])
        assert torch.allclose(rs_indexed.joint_torque, joint_torque[idx])

    def test_clone(self, sample_robot_state):
        """Test cloning RobotState."""
        cloned = sample_robot_state.clone()

        assert torch.allclose(
            cloned.joint_state.position, sample_robot_state.joint_state.position
        )
        assert torch.allclose(
            cloned.joint_state.velocity, sample_robot_state.joint_state.velocity
        )

        # Verify they are different objects
        assert cloned.joint_state.position.data_ptr() != sample_robot_state.joint_state.position.data_ptr()

    def test_clone_with_joint_torque(self, sample_joint_state, device_cfg):
        """Test cloning RobotState with joint torque."""
        joint_torque = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        rs = RobotState(joint_state=sample_joint_state, joint_torque=joint_torque)

        cloned = rs.clone()
        assert torch.allclose(cloned.joint_torque, joint_torque)
        assert cloned.joint_torque.data_ptr() != joint_torque.data_ptr()

    def test_copy_(self, device_cfg, joint_names):
        """Test copy_ method."""
        position1 = torch.ones(7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        rs1 = RobotState(joint_state=js1)
        rs2 = RobotState(joint_state=js2)

        rs1.copy_(rs2)
        assert torch.allclose(rs1.joint_state.position, position2)

    def test_copy_with_joint_torque(self, device_cfg, joint_names):
        """Test copy_ with joint torque."""
        position1 = torch.ones(7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)
        torque1 = torch.ones(7, device=device_cfg.device, dtype=device_cfg.dtype)
        torque2 = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        rs1 = RobotState(joint_state=js1, joint_torque=torque1)
        rs2 = RobotState(joint_state=js2, joint_torque=torque2)

        rs1.copy_(rs2)
        assert torch.allclose(rs1.joint_torque, torque2)

    def test_copy_only_index(self, device_cfg, joint_names):
        """Test copy_only_index method."""
        position1 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype) * 2.0

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        rs1 = RobotState(joint_state=js1)
        rs2 = RobotState(joint_state=js2)

        rs1.copy_only_index(rs2, 0)
        assert torch.allclose(rs1.joint_state.position[0], position2[0])

    def test_copy_only_index_with_torque(self, device_cfg, joint_names):
        """Test copy_only_index with joint torque."""
        position1 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype) * 2.0
        torque1 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        torque2 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype) * 3.0

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        rs1 = RobotState(joint_state=js1, joint_torque=torque1)
        rs2 = RobotState(joint_state=js2, joint_torque=torque2)

        rs1.copy_only_index(rs2, 0)
        assert torch.allclose(rs1.joint_torque[0], torque2[0])

    def test_copy_at_batch_seed_indices(self, device_cfg, joint_names):
        """Test copy_at_batch_seed_indices method."""
        position1 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype) * 2.0

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        rs1 = RobotState(joint_state=js1)
        rs2 = RobotState(joint_state=js2)

        batch_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)
        seed_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)

        rs1.copy_at_batch_seed_indices(rs2, batch_idx, seed_idx)
        assert torch.allclose(rs1.joint_state.position[0, 0], position2[0, 0])

    def test_copy_at_batch_seed_indices_with_torque(self, device_cfg, joint_names):
        """Test copy_at_batch_seed_indices with joint torque."""
        position1 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype) * 2.0
        torque1 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        torque2 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype) * 3.0

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        rs1 = RobotState(joint_state=js1, joint_torque=torque1)
        rs2 = RobotState(joint_state=js2, joint_torque=torque2)

        batch_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)
        seed_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)

        rs1.copy_at_batch_seed_indices(rs2, batch_idx, seed_idx)
        assert torch.allclose(rs1.joint_torque[0, 0], torque2[0, 0])

    def test_link_names_property_none(self, sample_robot_state):
        """Test tool_frames property when None."""
        tool_frames = sample_robot_state.tool_frames
        assert tool_frames == []

    def test_robot_state_is_sequence(self, sample_robot_state):
        """Test that RobotState is a Sequence."""
        from collections.abc import Sequence as ABCSequence

        assert isinstance(sample_robot_state, ABCSequence)

    def test_get_link_pose_raises_error_when_none(self, sample_robot_state):
        """Test get_link_pose raises error when link_poses is None."""
        with pytest.raises(Exception, match="Link poses are not set"):
            sample_robot_state.get_link_pose("link1")

    def test_robot_spheres_property_none(self, sample_robot_state):
        """Test robot_spheres property when cuda_robot_model_state is None."""
        assert sample_robot_state.robot_spheres is None

    def test_link_poses_property_none(self, sample_robot_state):
        """Test link_poses property when cuda_robot_model_state is None."""
        assert sample_robot_state.tool_poses is None

