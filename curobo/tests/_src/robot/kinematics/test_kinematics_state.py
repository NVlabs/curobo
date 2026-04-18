# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for KinematicsState."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def robot_state(cuda_device_cfg):
    """Create a sample robot state for testing."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])
    robot_model = Kinematics(cfg, compute_jacobian=True, compute_com=True)

    q_test = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(cuda_device_cfg.as_torch_dict())
    ).view(1, 1, -1)

    state = robot_model.compute_kinematics(
        JointState.from_position(q_test, joint_names=robot_model.joint_names)
    )
    return state


def test_cuda_robot_model_state_clone(robot_state):
    """Test that clone() creates an independent copy of the state."""
    cloned_state = robot_state.clone()

    # Verify all fields are cloned
    assert cloned_state is not robot_state
    assert cloned_state.robot_spheres is not robot_state.robot_spheres
    assert torch.equal(cloned_state.robot_spheres, robot_state.robot_spheres)

    # Verify that modifying clone doesn't affect original
    original_spheres = robot_state.robot_spheres.clone()
    cloned_state.robot_spheres[:] = 0.0
    assert torch.equal(robot_state.robot_spheres, original_spheres)

    # Test jacobian cloning if present
    if robot_state.tool_jacobians is not None:
        assert cloned_state.tool_jacobians is not robot_state.tool_jacobians
        assert torch.equal(cloned_state.tool_jacobians, robot_state.tool_jacobians)

    # Test link_poses cloning if present
    if robot_state.tool_poses is not None:
        assert cloned_state.tool_poses is not robot_state.tool_poses

    # Test robot_com cloning if present
    if robot_state.robot_com is not None:
        assert cloned_state.robot_com is not robot_state.robot_com
        assert torch.equal(cloned_state.robot_com, robot_state.robot_com)


def test_cuda_robot_model_state_detach(robot_state):
    """Test that detach() removes gradient tracking."""
    # Create a state with gradients
    device_cfg = DeviceCfg()
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])
    robot_model = Kinematics(cfg, compute_jacobian=True, compute_com=True)

    q_test = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(device_cfg.as_torch_dict())
    ).view(1, 1, -1).requires_grad_(True)

    state = robot_model.compute_kinematics(
        JointState.from_position(q_test, joint_names=robot_model.joint_names)
    )

    # Detach the state
    detached_state = state.detach()

    # Verify tensors are detached
    assert not detached_state.robot_spheres.requires_grad

    if detached_state.tool_jacobians is not None:
        assert not detached_state.tool_jacobians.requires_grad

    if detached_state.robot_com is not None:
        assert not detached_state.robot_com.requires_grad


def test_cuda_robot_model_state_trajectory_shape(robot_state):
    """Test that forward returns [batch, horizon, ...] tensor shapes."""
    device_cfg = DeviceCfg()
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])
    robot_model = Kinematics(cfg, compute_jacobian=True, compute_com=True)

    batch_size = 4
    horizon = 10

    q_test = torch.randn(batch_size, horizon, 7, **(device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(
        JointState.from_position(q_test, joint_names=robot_model.joint_names)
    )

    assert state.robot_spheres.ndim == 4
    assert state.robot_spheres.shape[0] == batch_size
    assert state.robot_spheres.shape[1] == horizon

    if state.tool_jacobians is not None:
        assert state.tool_jacobians.shape[0] == batch_size
        assert state.tool_jacobians.shape[1] == horizon

    if state.robot_com is not None:
        assert state.robot_com.shape[0] == batch_size
        assert state.robot_com.shape[1] == horizon


def test_cuda_robot_model_state_copy_(robot_state):
    """Test in-place copy operation."""
    # Test that copy_ method exists and is callable
    # Note: copy_ requires all fields to be non-None, so we test with a full state
    other_state = robot_state.clone()

    # Modify other_state
    other_state.robot_spheres[:] = other_state.robot_spheres + 0.1

    # Store original for comparison
    original_spheres = robot_state.robot_spheres.clone()

    # Copy other_state into robot_state
    robot_state.copy_(other_state)

    # Verify copy worked
    assert torch.allclose(robot_state.robot_spheres, other_state.robot_spheres, atol=1e-6)
    assert not torch.allclose(robot_state.robot_spheres, original_spheres, atol=1e-6)


def test_cuda_robot_model_state_link_names(robot_state):
    """Test tool_frames property."""
    tool_frames = robot_state.tool_frames

    assert isinstance(tool_frames, list)
    assert len(tool_frames) > 0
    assert all(isinstance(name, str) for name in tool_frames)


def test_cuda_robot_model_state_get_link_spheres(robot_state):
    """Test get_link_spheres method."""
    spheres = robot_state.get_link_spheres()

    assert isinstance(spheres, torch.Tensor)
    assert len(spheres.shape) == 4  # [batch, horizon, num_spheres, 4]
    assert spheres.shape[-1] == 4  # x, y, z, radius


def test_cuda_robot_model_state_empty_creation():
    """Test creating an empty state."""
    state = KinematicsState()

    assert state.tool_poses is None
    assert state.tool_jacobians is None
    assert state.robot_spheres is None
    assert state.robot_com is None
    assert state.robot_collision_geometry is None
    assert state.tool_frames == []


def test_cuda_robot_model_state_partial_creation():
    """Test creating a state with only some fields."""
    device_cfg = DeviceCfg()

    # Create state with only robot_spheres [batch, horizon, num_spheres, 4]
    spheres = torch.randn(2, 1, 10, 4, **(device_cfg.as_torch_dict()))
    state = KinematicsState(robot_spheres=spheres)

    assert state.robot_spheres is not None
    assert torch.equal(state.robot_spheres, spheres)
    assert state.tool_poses is None
    assert state.tool_jacobians is None


@pytest.mark.parametrize("batch_size", [1, 5, 20])
def test_cuda_robot_model_state_different_batch_sizes(batch_size, cuda_device_cfg):
    """Test state creation with different batch sizes."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])
    robot_model = Kinematics(cfg, compute_jacobian=True, compute_com=True)

    q_test = torch.randn(batch_size, 1, 7, **(cuda_device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(
        JointState.from_position(q_test, joint_names=robot_model.joint_names)
    )

    # Verify batch and horizon dimensions
    assert state.robot_spheres.shape[0] == batch_size
    assert state.robot_spheres.shape[1] == 1
    if state.tool_jacobians is not None:
        assert state.tool_jacobians.shape[0] == batch_size
        assert state.tool_jacobians.shape[1] == 1
    if state.robot_com is not None:
        assert state.robot_com.shape[0] == batch_size
        assert state.robot_com.shape[1] == 1

