# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.kinematics.kinematics_reducer import KinematicsReducer
from curobo._src.state.state_joint import JointState
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def franka_config(cuda_device_cfg):
    """Load Franka robot configuration for testing."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_data_dict(data_dict=robot_data,
    device_cfg=cuda_device_cfg)
    return cfg


def test_dof_reduction_basic_functionality(franka_config):
    """Test basic DOF reduction functionality with Franka robot."""
    original_config = franka_config.kinematics_config

    # Test reducing to just end effector
    desired_links = ["panda_hand"]
    reduced_config = KinematicsReducer.reduce_dof(
        original_config, desired_links, remove_collision_spheres=True
    )

    # Verify basic properties
    assert reduced_config is not None
    assert len(reduced_config.joint_names) <= len(original_config.joint_names)
    assert len(reduced_config.tool_frames) <= len(original_config.tool_frames)

    # Verify that the desired link is still present
    assert "panda_hand" in reduced_config.tool_frames

    print(f"Original DOF: {len(original_config.joint_names)}")
    print(f"Reduced DOF: {len(reduced_config.joint_names)}")
    print(f"Original links: {len(original_config.tool_frames)}")
    print(f"Reduced links: {len(reduced_config.tool_frames)}")


def test_dof_reduction_preserves_kinematic_chain(franka_config):
    """Test that DOF reduction preserves necessary kinematic chain elements."""
    original_config = franka_config.kinematics_config

    # Reduce to end effector
    desired_links = ["panda_hand"]
    reduced_config = KinematicsReducer.reduce_dof(original_config, desired_links)

    # Verify kinematic structure integrity
    assert reduced_config.joint_names is not None
    assert reduced_config.tool_frames is not None
    assert len(reduced_config.joint_names) > 0
    assert len(reduced_config.tool_frames) > 0

    # Verify tensor shapes are consistent
    n_joints = len(reduced_config.joint_names)
    num_links = len(reduced_config.tool_frames)

    # These tensors map to the full kinematic structure (all links in chain)
    # not just the stored links, so they may be larger than num_links
    assert reduced_config.joint_map.shape[0] >= num_links
    assert reduced_config.joint_map_type.shape[0] >= num_links
    assert reduced_config.link_map.shape[0] >= num_links

    # Store link map should match the number of stored links
    assert reduced_config.tool_frame_map.shape[0] == num_links

    # Verify cspace config dimensions
    if reduced_config.cspace is not None:
        if reduced_config.cspace.default_joint_position is not None:
            assert reduced_config.cspace.default_joint_position.shape[0] == n_joints
        if reduced_config.cspace.null_space_weight is not None:
            assert reduced_config.cspace.null_space_weight.shape[0] == n_joints
        if reduced_config.cspace.cspace_distance_weight is not None:
            assert reduced_config.cspace.cspace_distance_weight.shape[0] == n_joints

    # Verify joint limits dimensions
    if reduced_config.joint_limits is not None:
        assert reduced_config.joint_limits.position.shape[1] == n_joints  # shape is [2, n_joints]


def test_lock_jointstate_contains_removed_joints(franka_config):
    """Test that lock_jointstate contains all removed joints."""
    original_config = franka_config.kinematics_config

    # Get original joint count
    original_joint_count = len(original_config.joint_names)

    # Reduce DOF
    desired_links = ["panda_hand"]
    reduced_config = KinematicsReducer.reduce_dof(original_config, desired_links)

    reduced_joint_count = len(reduced_config.joint_names)

    # Check that some joints were removed (DOF was actually reduced)
    if reduced_joint_count < original_joint_count:
        # lock_jointstate should contain the removed joints
        assert reduced_config.lock_jointstate is not None

        removed_joint_count = original_joint_count - reduced_joint_count
        lock_joint_count = len(reduced_config.lock_jointstate.joint_names)

        # All removed joints should be in lock state
        assert lock_joint_count >= removed_joint_count

        # Verify lock state has proper dimensions
        assert reduced_config.lock_jointstate.position.shape[0] == lock_joint_count

        print(f"Removed joints: {removed_joint_count}")
        print(f"Lock joints: {lock_joint_count}")
        print(f"Lock joint names: {reduced_config.lock_jointstate.joint_names}")

        # Verify that all locked joints are unique
        unique_lock_joints = set(reduced_config.lock_jointstate.joint_names)
        assert len(unique_lock_joints) == len(reduced_config.lock_jointstate.joint_names), (
            f"Duplicate joints found in lock state: {reduced_config.lock_jointstate.joint_names}"
        )


def test_joint_state_reconstruction(cuda_device_cfg):
    """Test reconstruction of full joint state from reduced optimization result."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "dual_ur10e.yml"))
    cfg = KinematicsCfg.from_data_dict(data_dict=robot_data, device_cfg=cuda_device_cfg)


    original_config = cfg.kinematics_config

    # Reduce DOF
    desired_links = ["tool1"]
    reduced_config = KinematicsReducer.reduce_dof(original_config, desired_links)

    # Skip test if no DOF reduction occurred
    if len(reduced_config.joint_names) == len(original_config.joint_names):
        pytest.skip("No DOF reduction occurred for this configuration")

    # Create a mock reduced joint state (simulating optimization result)
    n_reduced_joints = len(reduced_config.joint_names)
    mock_positions = torch.zeros(n_reduced_joints, dtype=original_config.device_cfg.dtype)

    reduced_joint_state = JointState(
        position=mock_positions, joint_names=reduced_config.joint_names
    )

    # Reconstruct full joint state
    full_joint_state = KinematicsReducer.reconstruct_joint_state(
        reduced_joint_state, reduced_config.lock_jointstate, original_config.joint_names
    )

    # Verify reconstruction
    assert full_joint_state is not None
    assert len(full_joint_state.joint_names) == len(original_config.joint_names)
    assert full_joint_state.position.shape[0] == len(original_config.joint_names)

    # Verify all original joints are represented
    for joint_name in original_config.joint_names:
        assert joint_name in full_joint_state.joint_names



def test_collision_spheres_removal(franka_config):
    """Test that collision spheres are properly removed during DOF reduction."""
    original_config = franka_config.kinematics_config

    # Test with collision sphere removal enabled
    reduced_config = KinematicsReducer.reduce_dof(
        original_config, ["panda_hand"], remove_collision_spheres=True
    )

    # Verify collision spheres are removed/zeroed
    if original_config.link_spheres is not None and reduced_config.link_spheres is not None:
        assert reduced_config.link_spheres.numel() <= original_config.link_spheres.numel()

    # The reduced config should have minimal collision data
    assert (
        reduced_config.total_spheres == 0
        or reduced_config.total_spheres < original_config.total_spheres
    )


def test_multiple_desired_links(franka_config):
    """Test DOF reduction with multiple desired links."""
    original_config = franka_config.kinematics_config

    # Test with multiple links
    desired_links = ["panda_hand", "panda_link7"]
    reduced_config = KinematicsReducer.reduce_dof(original_config, desired_links)

    # Verify both desired links are preserved
    for link_name in desired_links:
        if link_name in original_config.tool_frames:  # Only check if link exists in original
            assert link_name in reduced_config.tool_frames

    # Verify configuration is valid
    assert reduced_config.joint_names is not None
    assert len(reduced_config.joint_names) > 0


def test_error_handling_invalid_links(franka_config):
    """Test error handling when invalid link names are provided."""
    original_config = franka_config.kinematics_config

    # Test with completely invalid link names
    with pytest.raises(ValueError, match="None of the desired links"):
        KinematicsReducer.reduce_dof(original_config, ["nonexistent_link_123"])


def test_configuration_consistency(franka_config):
    """Test that the reduced configuration maintains internal consistency."""
    original_config = franka_config.kinematics_config

    reduced_config = KinematicsReducer.reduce_dof(original_config, ["panda_hand"])

    # Test tensor device consistency
    assert reduced_config.joint_map.device == original_config.joint_map.device
    assert reduced_config.link_map.device == original_config.link_map.device

    # Test tensor dtype consistency
    assert reduced_config.joint_map.dtype == original_config.joint_map.dtype
    assert reduced_config.link_map.dtype == original_config.link_map.dtype

    # Test name mapping consistency
    if reduced_config.link_name_to_idx_map is not None:
        # Check that link names map to correct indices in the global link space
        for link_name, global_idx in reduced_config.link_name_to_idx_map.items():
            assert isinstance(global_idx, int)
            assert global_idx >= 0


def test_g1_config(cuda_device_cfg):
    """Test G1 robot configuration."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "unitree_g1.yml"))
    cfg = KinematicsCfg.from_data_dict(data_dict=robot_data, device_cfg=cuda_device_cfg)

    full_kinematics = Kinematics(cfg)
    # get default joint position:
    default_joint_position = full_kinematics.default_joint_state.clone()
    kin_state = full_kinematics.compute_kinematics(default_joint_position).clone()
    # print(kin_state)
    reduced_config = KinematicsReducer.reduce_dof(
        cfg.kinematics_config, cfg.kinematics_config.tool_frames
    )

    cfg.kinematics_config = reduced_config
    reduced_kinematics = Kinematics(cfg)
    default_joint_position = reduced_kinematics.default_joint_state.clone()
    reduced_kin_state = reduced_kinematics.compute_kinematics(default_joint_position).clone()
    # print(reduced_kin_state)

    assert kin_state.tool_poses.tool_frames == reduced_kin_state.tool_poses.tool_frames
    assert torch.allclose(kin_state.tool_poses.position, reduced_kin_state.tool_poses.position)
    assert torch.allclose(kin_state.tool_poses.quaternion, reduced_kin_state.tool_poses.quaternion)

