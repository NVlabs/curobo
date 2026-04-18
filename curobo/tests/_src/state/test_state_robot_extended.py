# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for RobotState with KinematicsState integration."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def robot_model():
    """Create a robot model for testing."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])
    return Kinematics(cfg, compute_jacobian=True, compute_com=True)


@pytest.fixture(scope="module")
def device_cfg():
    """Get tensor device configuration."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return DeviceCfg(device=device, dtype=torch.float32)


@pytest.fixture
def robot_state_with_model(robot_model, device_cfg):
    """Create a RobotState with KinematicsState."""
    q_test = torch.tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], device=device_cfg.device, dtype=device_cfg.dtype
    ).view(1, -1)

    cuda_state = robot_model.compute_kinematics(
        JointState.from_position(q_test, joint_names=robot_model.joint_names)
    )
    joint_state = JointState.from_position(q_test.squeeze(0), joint_names=robot_model.joint_names)

    return RobotState(joint_state=joint_state, cuda_robot_model_state=cuda_state)


@pytest.fixture
def robot_state_batch_with_model(robot_model, device_cfg):
    """Create a batch RobotState with KinematicsState."""
    batch_size = 5
    q_test = torch.randn(
        batch_size, 7, device=device_cfg.device, dtype=device_cfg.dtype
    )

    cuda_state = robot_model.compute_kinematics(
        JointState.from_position(q_test, joint_names=robot_model.joint_names)
    )
    joint_state = JointState.from_position(q_test, joint_names=robot_model.joint_names)

    return RobotState(joint_state=joint_state, cuda_robot_model_state=cuda_state)


class TestRobotStateWithCudaModel:
    """Test RobotState with KinematicsState integration."""

    pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

    def test_robot_spheres_property(self, robot_state_with_model):
        """Test robot_spheres property with actual KinematicsState."""
        spheres = robot_state_with_model.robot_spheres
        assert spheres is not None
        assert isinstance(spheres, torch.Tensor)
        assert spheres.shape[-1] == 4  # x, y, z, r format

    def test_link_poses_property(self, robot_state_with_model):
        """Test link_poses property with actual KinematicsState."""
        link_poses = robot_state_with_model.tool_poses
        assert link_poses is not None
        assert hasattr(link_poses, 'position')
        assert hasattr(link_poses, 'quaternion')

    def test_link_names_property_with_poses(self, robot_state_with_model):
        """Test tool_frames property when link_poses exists."""
        tool_frames = robot_state_with_model.tool_frames
        assert isinstance(tool_frames, list)
        assert len(tool_frames) > 0
        assert "panda_hand" in tool_frames

    def test_get_link_pose_success(self, robot_state_with_model):
        """Test get_link_pose with valid link name."""
        link_pose = robot_state_with_model.get_link_pose("panda_hand")
        assert link_pose is not None
        assert hasattr(link_pose, 'position')
        assert hasattr(link_pose, 'quaternion')

    def test_detach_with_cuda_model_state(self, robot_state_with_model):
        """Test detach with KinematicsState."""
        detached = robot_state_with_model.detach()
        assert detached.cuda_robot_model_state is not None
        assert not detached.joint_state.position.requires_grad

    def test_clone_with_cuda_model_state(self, robot_state_with_model):
        """Test clone with KinematicsState."""
        cloned = robot_state_with_model.clone()
        assert cloned.cuda_robot_model_state is not None
        assert cloned.robot_spheres is not None
        # Verify different objects
        assert cloned.robot_spheres.data_ptr() != robot_state_with_model.robot_spheres.data_ptr()

    def test_getitem_with_cuda_model_state(self, robot_state_batch_with_model):
        """Test indexing with KinematicsState using tensor index."""
        idx = torch.tensor([0, 2], device=robot_state_batch_with_model.joint_state.device)
        indexed = robot_state_batch_with_model[idx]
        assert indexed.cuda_robot_model_state is not None
        assert indexed.robot_spheres is not None
        assert indexed.robot_spheres.shape[0] == 2

    def test_copy_with_cuda_model_state(self, robot_model, device_cfg):
        """Test copy_ with KinematicsState."""
        # Create two different states
        q1 = torch.tensor(
            [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], device=device_cfg.device, dtype=device_cfg.dtype
        ).view(1, 1, -1)
        q2 = torch.tensor(
            [0.5, -0.8, 0.3, -1.5, 0.2, 0.8, 0.1], device=device_cfg.device, dtype=device_cfg.dtype
        ).view(1, 1, -1)

        cuda_state1 = robot_model.compute_kinematics(
            JointState.from_position(q1, joint_names=robot_model.joint_names)
        )
        cuda_state2 = robot_model.compute_kinematics(
            JointState.from_position(q2, joint_names=robot_model.joint_names)
        )

        js1 = JointState.from_position(q1.squeeze(0), joint_names=robot_model.joint_names)
        js2 = JointState.from_position(q2.squeeze(0), joint_names=robot_model.joint_names)

        rs1 = RobotState(joint_state=js1, cuda_robot_model_state=cuda_state1)
        rs2 = RobotState(joint_state=js2, cuda_robot_model_state=cuda_state2)

        # Copy rs2 into rs1
        rs1.copy_(rs2)

        # Verify the copy happened
        assert torch.allclose(rs1.joint_state.position, q2.squeeze(0))

    def test_copy_at_batch_seed_indices_with_spheres(self, robot_model, device_cfg):
        """Test copy_at_batch_seed_indices with robot_spheres."""
        batch_size = 5
        num_seeds = 3

        q1 = torch.randn(batch_size, num_seeds, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        q2 = torch.randn(batch_size, num_seeds, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        cuda_state1 = robot_model.compute_kinematics(
            JointState.from_position(q1, joint_names=robot_model.joint_names)
        )
        cuda_state2 = robot_model.compute_kinematics(
            JointState.from_position(q2, joint_names=robot_model.joint_names)
        )

        js1 = JointState.from_position(q1, joint_names=robot_model.joint_names)
        js2 = JointState.from_position(q2, joint_names=robot_model.joint_names)

        rs1 = RobotState(joint_state=js1, cuda_robot_model_state=cuda_state1)
        rs2 = RobotState(joint_state=js2, cuda_robot_model_state=cuda_state2)

        batch_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)
        seed_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)

        rs1.copy_at_batch_seed_indices(rs2, batch_idx, seed_idx)

        assert torch.allclose(rs1.robot_spheres[0, 0], rs2.robot_spheres[0, 0])
        assert torch.allclose(rs1.tool_poses.position[0, 0], rs2.tool_poses.position[0, 0])

    def test_copy_at_batch_seed_indices_with_link_poses(self, robot_model, device_cfg):
        """Test copy_at_batch_seed_indices specifically for link_poses branch."""
        batch_size = 3
        num_seeds = 2

        q1 = torch.randn(batch_size, num_seeds, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        q2 = torch.randn(batch_size, num_seeds, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        cuda_state1 = robot_model.compute_kinematics(
            JointState.from_position(q1, joint_names=robot_model.joint_names)
        )
        cuda_state2 = robot_model.compute_kinematics(
            JointState.from_position(q2, joint_names=robot_model.joint_names)
        )

        js1 = JointState.from_position(q1, joint_names=robot_model.joint_names)
        js2 = JointState.from_position(q2, joint_names=robot_model.joint_names)

        rs1 = RobotState(joint_state=js1, cuda_robot_model_state=cuda_state1)
        rs2 = RobotState(joint_state=js2, cuda_robot_model_state=cuda_state2)

        batch_idx = torch.tensor([1], device=device_cfg.device, dtype=torch.long)
        seed_idx = torch.tensor([0], device=device_cfg.device, dtype=torch.long)

        rs1.copy_at_batch_seed_indices(rs2, batch_idx, seed_idx)

        # Verify link poses were copied
        assert torch.allclose(rs1.tool_poses.position[1, 0], rs2.tool_poses.position[1, 0])
        assert torch.allclose(rs1.tool_poses.quaternion[1, 0], rs2.tool_poses.quaternion[1, 0])

    def test_copy_only_index_with_spheres(self, robot_model, device_cfg):
        """Test copy_only_index with robot_spheres."""
        batch_size = 10

        q1 = torch.randn(batch_size, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        q2 = torch.randn(batch_size, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        cuda_state1 = robot_model.compute_kinematics(
            JointState.from_position(q1, joint_names=robot_model.joint_names)
        )
        cuda_state2 = robot_model.compute_kinematics(
            JointState.from_position(q2, joint_names=robot_model.joint_names)
        )

        js1 = JointState.from_position(q1, joint_names=robot_model.joint_names)
        js2 = JointState.from_position(q2, joint_names=robot_model.joint_names)

        rs1 = RobotState(joint_state=js1, cuda_robot_model_state=cuda_state1)
        rs2 = RobotState(joint_state=js2, cuda_robot_model_state=cuda_state2)

        # Copy index 0 from rs2 to rs1
        rs1.copy_only_index(rs2, 0)

        # Verify spheres were copied
        assert torch.allclose(rs1.robot_spheres[0], rs2.robot_spheres[0])

    def test_copy_only_index_with_link_poses(self, robot_model, device_cfg):
        """Test copy_only_index specifically for link_poses branch."""
        batch_size = 5

        q1 = torch.randn(batch_size, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        q2 = torch.randn(batch_size, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        cuda_state1 = robot_model.compute_kinematics(
            JointState.from_position(q1, joint_names=robot_model.joint_names)
        )
        cuda_state2 = robot_model.compute_kinematics(
            JointState.from_position(q2, joint_names=robot_model.joint_names)
        )

        js1 = JointState.from_position(q1, joint_names=robot_model.joint_names)
        js2 = JointState.from_position(q2, joint_names=robot_model.joint_names)

        rs1 = RobotState(joint_state=js1, cuda_robot_model_state=cuda_state1)
        rs2 = RobotState(joint_state=js2, cuda_robot_model_state=cuda_state2)

        # Copy index 2
        rs1.copy_only_index(rs2, 2)

        # Verify link poses were copied
        assert torch.allclose(rs1.tool_poses.position[2], rs2.tool_poses.position[2])
        assert torch.allclose(rs1.tool_poses.quaternion[2], rs2.tool_poses.quaternion[2])

    def test_robot_state_with_joint_torque(self, robot_model, device_cfg):
        """Test RobotState with joint torque."""
        q_test = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_torque = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        cuda_state = robot_model.compute_kinematics(
            JointState.from_position(q_test, joint_names=robot_model.joint_names)
        )
        joint_state = JointState.from_position(q_test, joint_names=robot_model.joint_names)

        rs = RobotState(
            joint_state=joint_state,
            joint_torque=joint_torque,
            cuda_robot_model_state=cuda_state
        )

        assert rs.joint_torque is not None
        assert torch.allclose(rs.joint_torque, joint_torque)

    def test_copy_at_batch_seed_indices_with_joint_torque(self, robot_model, device_cfg):
        """Test copy_at_batch_seed_indices with joint_torque."""
        batch_size = 3
        num_seeds = 2

        q1 = torch.randn(batch_size, num_seeds, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        q2 = torch.randn(batch_size, num_seeds, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        torque1 = torch.randn(batch_size, num_seeds, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        torque2 = torch.randn(batch_size, num_seeds, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        cuda_state1 = robot_model.compute_kinematics(
            JointState.from_position(q1, joint_names=robot_model.joint_names)
        )
        cuda_state2 = robot_model.compute_kinematics(
            JointState.from_position(q2, joint_names=robot_model.joint_names)
        )

        js1 = JointState.from_position(q1, joint_names=robot_model.joint_names)
        js2 = JointState.from_position(q2, joint_names=robot_model.joint_names)

        rs1 = RobotState(joint_state=js1, joint_torque=torque1, cuda_robot_model_state=cuda_state1)
        rs2 = RobotState(joint_state=js2, joint_torque=torque2, cuda_robot_model_state=cuda_state2)

        batch_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)
        seed_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)

        rs1.copy_at_batch_seed_indices(rs2, batch_idx, seed_idx)

        # Verify joint torque was copied
        assert torch.allclose(rs1.joint_torque[0, 0], torque2[0, 0])
        assert torch.allclose(rs1.joint_torque[1, 1], torque2[1, 1])

    def test_copy_only_index_with_joint_torque(self, robot_model, device_cfg):
        """Test copy_only_index with joint_torque."""
        batch_size = 5

        q1 = torch.randn(batch_size, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        q2 = torch.randn(batch_size, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        torque1 = torch.randn(batch_size, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        torque2 = torch.randn(batch_size, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        cuda_state1 = robot_model.compute_kinematics(
            JointState.from_position(q1, joint_names=robot_model.joint_names)
        )
        cuda_state2 = robot_model.compute_kinematics(
            JointState.from_position(q2, joint_names=robot_model.joint_names)
        )

        js1 = JointState.from_position(q1, joint_names=robot_model.joint_names)
        js2 = JointState.from_position(q2, joint_names=robot_model.joint_names)

        rs1 = RobotState(joint_state=js1, joint_torque=torque1, cuda_robot_model_state=cuda_state1)
        rs2 = RobotState(joint_state=js2, joint_torque=torque2, cuda_robot_model_state=cuda_state2)

        rs1.copy_only_index(rs2, 3)

        # Verify joint torque was copied
        assert torch.allclose(rs1.joint_torque[3], torque2[3])

