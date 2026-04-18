# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Extended unit tests for Kinematics covering edge cases and missing paths."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def franka_cfg(cuda_device_cfg):
    """Load Franka robot configuration for testing."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])
    return cfg

@pytest.fixture(scope="module")
def franka_mesh_cfg(cuda_device_cfg):
    """Load Franka robot configuration for testing."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, tool_frames=robot_data["robot_cfg"]["kinematics"]["mesh_link_names"])
    return cfg



def test_cuda_robot_model_single_joint_position(franka_cfg):
    """Test that [1, 1, dof] joint configuration works."""
    robot_model = Kinematics(franka_cfg)

    q = torch.zeros(1, 1, 7, device=franka_cfg.device_cfg.device, dtype=franka_cfg.device_cfg.dtype)

    state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    assert state.robot_spheres.shape[0] == 1


def test_cuda_robot_model_batch_size_update(franka_cfg):
    """Test dynamic batch size updates."""
    robot_model = Kinematics(franka_cfg)

    # Test with different batch sizes
    batch_sizes = [1, 5, 10, 20]

    for batch_size in batch_sizes:
        q = torch.randn(batch_size, 1, 7, **(franka_cfg.device_cfg.as_torch_dict()))
        state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

        assert state.robot_spheres.shape[0] == batch_size
        assert robot_model._batch == batch_size


def test_cuda_robot_model_property_getters(franka_cfg):
    """Test all property getter methods."""
    robot_model = Kinematics(franka_cfg)

    # Test tool_frames property
    tool_frames = robot_model.tool_frames
    assert isinstance(tool_frames, list)
    assert len(tool_frames) > 0

    # Test joint_names property
    joint_names = robot_model.joint_names
    assert isinstance(joint_names, list)
    assert len(joint_names) == 7  # Franka has 7 DOF

    # Test get_dof method
    dof = robot_model.get_dof()
    assert dof == 7

    # Test kinematics_config property
    kin_config = robot_model.kinematics_config
    assert kin_config is not None

    # Test default_joint_position property
    default_joint_position = robot_model.default_joint_position
    assert isinstance(default_joint_position, torch.Tensor)
    assert default_joint_position.shape[0] == 7


def test_cuda_robot_model_compute_flags(franka_cfg):
    """Test different compute flag combinations."""
    # Test with all flags enabled
    robot_with_all = Kinematics(
        franka_cfg,
        compute_jacobian=True,
        compute_spheres=True,
        compute_com=True
    )
    q = torch.zeros(1, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    state = robot_with_all.compute_kinematics(JointState.from_position(q, joint_names=robot_with_all.joint_names))

    # Verify that when flags are enabled, we get results
    assert state.robot_spheres is not None
    assert state.robot_com is not None
    # Note: Jacobian computation may vary based on internal implementation

    # Test basic computation works
    assert not torch.any(torch.isnan(state.robot_spheres))
    assert not torch.any(torch.isnan(state.robot_com))


def test_cuda_robot_model_forward_alias(franka_cfg):
    """Test that forward() is an alias for get_state()."""
    robot_model = Kinematics(franka_cfg)

    q = torch.zeros(1, 7, **(franka_cfg.device_cfg.as_torch_dict()))

    state1 = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))
    state2 = robot_model.compute_kinematics(JointState.from_position(q.clone(), joint_names=robot_model.joint_names))

    # Results should be identical
    assert torch.allclose(state1.robot_spheres, state2.robot_spheres, atol=1e-5)


def test_cuda_robot_model_gradients_enabled(franka_cfg):
    """Test forward kinematics with gradient tracking."""
    robot_model = Kinematics(franka_cfg)

    q = torch.zeros(1, 7, **(franka_cfg.device_cfg.as_torch_dict()), requires_grad=True)

    state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    # Compute a loss and backpropagate
    loss = state.robot_spheres[..., :3].sum()
    loss.backward()

    # Gradients should be computed
    assert q.grad is not None
    assert not torch.all(q.grad == 0)


def test_cuda_robot_model_default_joint_position(franka_cfg):
    """Test default joint position."""
    robot_model = Kinematics(franka_cfg)

    default_position = robot_model.default_joint_position

    assert isinstance(default_position, torch.Tensor)
    assert default_position.shape[0] == robot_model.get_dof()
    assert default_position.device == franka_cfg.device_cfg.device


def test_cuda_robot_model_cspace(franka_cfg):
    """Test configuration space properties."""
    robot_model = Kinematics(franka_cfg)

    # Test joint limits
    joint_limits = robot_model.get_joint_limits()
    assert joint_limits is not None

    # Test cspace configuration
    cspace = robot_model.kinematics_config.cspace
    if cspace is not None:
        assert hasattr(cspace, 'joint_names')


def test_cuda_robot_model_self_collision_config(franka_cfg):
    """Test self-collision configuration."""
    robot_model = Kinematics(franka_cfg)

    # Check if self-collision configuration exists and is retrievable
    self_collision_cfg = robot_model.get_self_collision_config()

    # May be None if not configured, otherwise should have basic attributes
    if self_collision_cfg is not None:
        # Check that it has expected attributes for collision checking
        assert hasattr(self_collision_cfg, 'num_spheres') or hasattr(self_collision_cfg, 'collision_pairs')


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_cuda_robot_model_different_batch_sizes(franka_cfg, batch_size):
    """Test kinematics computation with different batch sizes."""
    robot_model = Kinematics(franka_cfg, compute_jacobian=True)

    q = torch.randn(batch_size, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    # Verify batch dimensions
    assert state.robot_spheres.shape[0] == batch_size
    if state.tool_jacobians is not None:
        assert state.tool_jacobians.shape[0] == batch_size


def test_cuda_robot_model_zero_configuration(franka_cfg):
    """Test kinematics at zero configuration."""
    robot_model = Kinematics(franka_cfg)

    q_zero = torch.zeros(1, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(JointState.from_position(q_zero, joint_names=robot_model.joint_names))

    # Should produce valid output
    assert state.robot_spheres is not None
    assert not torch.any(torch.isnan(state.robot_spheres))


def test_cuda_robot_model_random_configurations(franka_cfg):
    """Test kinematics with random valid configurations."""
    robot_model = Kinematics(franka_cfg)

    # Generate random configurations within joint limits
    joint_limits = robot_model.get_joint_limits()

    q_rand = torch.rand(5, 7, **(franka_cfg.device_cfg.as_torch_dict()))

    if joint_limits is not None and joint_limits.position is not None:
        # Scale to joint limits
        q_min = joint_limits.position[0]
        q_max = joint_limits.position[1]
        q_rand = q_min + (q_max - q_min) * q_rand

    state = robot_model.compute_kinematics(JointState.from_position(q_rand, joint_names=robot_model.joint_names))

    # Should produce valid output without NaNs
    assert not torch.any(torch.isnan(state.robot_spheres))


def test_cuda_robot_model_link_names_consistency(franka_cfg):
    """Test that link names are consistent across state."""
    robot_model = Kinematics(franka_cfg)

    model_link_names = robot_model.tool_frames

    q = torch.zeros(1, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    state_link_names = state.tool_frames

    assert model_link_names == state_link_names


def test_cuda_robot_model_spheres_format(franka_cfg):
    """Test that robot spheres are in correct format [x, y, z, radius]."""
    robot_model = Kinematics(franka_cfg, compute_spheres=True)

    q = torch.zeros(1, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    spheres = state.robot_spheres

    # Check shape: [batch, horizon, num_spheres, 4]
    assert len(spheres.shape) == 4
    assert spheres.shape[-1] == 4

    # Check that radii are positive (or negative for disabled spheres)
    radii = spheres[..., 3]
    assert torch.all((radii > 0) | (radii < 0))  # Either positive or negative (disabled)


def test_cuda_robot_model_link_pose_access(franka_cfg):
    """Test accessing link poses."""
    robot_model = Kinematics(franka_cfg)

    q = torch.zeros(1, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    # Access link poses
    if state.tool_poses is not None:
        for link_name in state.tool_frames:
            link_pose = state.tool_poses.get_link_pose(link_name)
            assert link_pose is not None
            assert link_pose.position is not None
            assert link_pose.quaternion is not None


def test_cuda_robot_model_jacobian_shape(franka_cfg):
    """Test that Jacobian has correct shape."""
    robot_model = Kinematics(franka_cfg, compute_jacobian=True)

    batch_size = 5
    q = torch.randn(batch_size, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    if state.tool_jacobians is not None:
        # Shape should be [batch_size, horizon, num_links, 6, num_dof]
        assert len(state.tool_jacobians.shape) == 5
        assert state.tool_jacobians.shape[0] == batch_size
        assert state.tool_jacobians.shape[-2] == 6  # 3 linear + 3 angular


def test_cuda_robot_model_com_computation(franka_cfg):
    """Test center of mass computation."""
    robot_model = Kinematics(franka_cfg, compute_com=True)

    q = torch.zeros(1, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    if state.robot_com is not None:
        # Shape should be [batch_size, 4] - xyz=global CoM, w=total mass
        assert state.robot_com.shape[-1] == 4

        # Mass should be positive
        total_mass = state.robot_com[..., 3]
        assert torch.all(total_mass > 0)


@pytest.mark.parametrize(
    "robot_file,ee_link",
    [("franka.yml", "panda_hand"), ("ur10e.yml", "tool0")]
)
def test_cuda_robot_model_different_robots(robot_file, ee_link, cuda_device_cfg):
    """Test model creation with different robot configurations."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), robot_file))

    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, [ee_link])
    robot_model = Kinematics(cfg)

    # Should be able to compute kinematics
    dof = robot_model.get_dof()
    q = torch.zeros(1, dof, **(cuda_device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    assert state.robot_spheres is not None


def test_cuda_robot_model_update_batch_size_force(franka_cfg):
    """Test force update of batch size."""
    robot_model = Kinematics(franka_cfg)

    # Initial computation
    q1 = torch.zeros(5, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    robot_model.compute_kinematics(JointState.from_position(q1, joint_names=robot_model.joint_names))

    assert robot_model._batch == 5

    # Update with same batch size but force update
    robot_model.update_batch_size(5, 1, force_update=True, reset_buffers=True)

    # Should still work
    state = robot_model.compute_kinematics(JointState.from_position(q1, joint_names=robot_model.joint_names))
    assert state.robot_spheres.shape[0] == 5


def test_cuda_robot_model_cuda_device_cfg(franka_cfg):
    """Test that model respects tensor device configuration."""
    robot_model = Kinematics(franka_cfg)

    assert robot_model.device_cfg == franka_cfg.device_cfg

    # Create input on correct device
    q = torch.zeros(1, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    # Output should be on same device
    assert state.robot_spheres.device == franka_cfg.device_cfg.device


def test_cuda_robot_model_invalid_batch_size_zero(franka_cfg):
    """Test error handling for batch_size == 0."""
    robot_model = Kinematics(franka_cfg)

    with pytest.raises(ValueError, match="batch and horizon must be > 0"):
        robot_model.update_batch_size(0, 1)


def test_cuda_robot_model_invalid_joint_dims_3d(franka_cfg):
    """Test that 3D input [batch, horizon, dof] is accepted and 4D is rejected."""
    robot_model = Kinematics(franka_cfg)

    q_3d = torch.zeros(2, 3, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    state = robot_model.compute_kinematics(JointState.from_position(q_3d, joint_names=robot_model.joint_names))
    assert state is not None

    q_4d = torch.zeros(2, 3, 1, 7, **(franka_cfg.device_cfg.as_torch_dict()))
    with pytest.raises(ValueError, match="joint_position must be"):
        robot_model.compute_kinematics(JointState.from_position(q_4d, joint_names=robot_model.joint_names))


def test_cuda_robot_model_invalid_dof_mismatch(franka_cfg):
    """Test error handling for DOF mismatch."""
    robot_model = Kinematics(franka_cfg)

    # Create tensor with wrong DOF
    q_wrong = torch.zeros(1, 5, **(franka_cfg.device_cfg.as_torch_dict()))  # Should be 7

    with pytest.raises(ValueError, match="q should have dof"):
        robot_model.compute_kinematics(JointState.from_position(q_wrong, joint_names=robot_model.joint_names))


def test_cuda_robot_model_base_link_property(franka_cfg):
    """Test base_link property."""
    robot_model = Kinematics(franka_cfg)

    base_link = robot_model.base_link
    assert isinstance(base_link, str)
    assert len(base_link) > 0


def test_cuda_robot_model_robot_spheres_property(franka_cfg):
    """Test robot_spheres property."""
    robot_model = Kinematics(franka_cfg)

    spheres = robot_model.robot_spheres
    assert spheres is not None
    assert isinstance(spheres, torch.Tensor)


def test_cuda_robot_model_all_articulated_joint_names(franka_cfg):
    """Test all_articulated_joint_names property."""
    robot_model = Kinematics(franka_cfg)

    joint_names = robot_model.all_articulated_joint_names
    assert isinstance(joint_names, list)
    assert len(joint_names) > 0


def test_cuda_robot_model_total_spheres(franka_cfg):
    """Test total_spheres property."""
    robot_model = Kinematics(franka_cfg)

    total = robot_model.total_spheres
    assert isinstance(total, int)
    assert total > 0


def test_cuda_robot_model_lock_jointstate(franka_cfg):
    """Test lock_jointstate property."""
    robot_model = Kinematics(franka_cfg)

    lock_js = robot_model.lock_jointstate
    # May be None if no locked joints
    if lock_js is not None:
        assert hasattr(lock_js, 'position')


def test_cuda_robot_model_default_joint_state(franka_cfg):
    """Test default_joint_state method."""
    robot_model = Kinematics(franka_cfg)

    default_joint_state = robot_model.default_joint_state
    assert default_joint_state is not None
    assert hasattr(default_joint_state, 'position')
    assert default_joint_state.position.shape[0] == robot_model.get_dof()


def test_cuda_robot_model_compute_kinematics_from_joint_state(franka_cfg, cuda_device_cfg):
    """Test compute_kinematics with JointState input."""
    from curobo._src.state.state_joint import JointState

    robot_model = Kinematics(franka_cfg)

    q = torch.zeros(7, **(cuda_device_cfg.as_torch_dict()))
    joint_state = JointState.from_position(q)

    state = robot_model.compute_kinematics(joint_state)
    assert state is not None
    assert state.robot_spheres is not None


def test_cuda_robot_model_get_link_poses(franka_cfg, cuda_device_cfg):
    """Test get_link_poses method."""
    robot_model = Kinematics(franka_cfg)

    q = torch.zeros(1, 7, **(cuda_device_cfg.as_torch_dict()))
    tool_frames = robot_model.tool_frames[:2]  # Get first 2 links

    poses = robot_model.get_link_poses(q, tool_frames)
    assert poses is not None
    assert poses.position is not None


def test_cuda_robot_model_get_robot_as_spheres(franka_cfg, cuda_device_cfg):
    """Test get_robot_as_spheres method."""
    robot_model = Kinematics(franka_cfg)

    q = torch.zeros(1, 7, **(cuda_device_cfg.as_torch_dict()))

    # Test with filter_valid=True
    spheres = robot_model.get_robot_as_spheres(q, filter_valid=True)
    assert isinstance(spheres, list)

    # Test with filter_valid=False
    spheres_all = robot_model.get_robot_as_spheres(q, filter_valid=False)
    assert isinstance(spheres_all, list)


def test_cuda_robot_model_dof_property(franka_cfg):
    """Test dof property (alias for get_dof)."""
    robot_model = Kinematics(franka_cfg)

    assert robot_model.dof == robot_model.get_dof()
    assert robot_model.dof == 7  # Franka has 7 DOF


def test_cuda_robot_model_get_full_js(franka_cfg, cuda_device_cfg):
    """Test get_full_js method."""
    from curobo._src.state.state_joint import JointState

    robot_model = Kinematics(franka_cfg)

    # Create joint state for active joints with joint names
    q = torch.zeros(7, **(cuda_device_cfg.as_torch_dict()))
    joint_state = JointState.from_position(q, joint_names=robot_model.joint_names)

    # Get full joint state including locked joints
    full_js = robot_model.get_full_js(joint_state)

    assert full_js is not None
    assert full_js.position is not None
    # Full joint state should include locked joints
    assert full_js.position.shape[0] >= joint_state.position.shape[0]


def test_cuda_robot_model_get_mimic_js(franka_cfg, cuda_device_cfg):
    """Test get_mimic_js method."""
    from curobo._src.state.state_joint import JointState

    robot_model = Kinematics(franka_cfg)

    q = torch.zeros(7, **(cuda_device_cfg.as_torch_dict()))
    joint_state = JointState.from_position(q, joint_names=robot_model.joint_names)

    mimic_js = robot_model.get_mimic_js(joint_state)

    # Franka doesn't have mimic joints, so should return None
    # This tests the None return path
    assert mimic_js is None


def test_cuda_robot_model_get_active_js(franka_cfg, cuda_device_cfg):
    """Test get_active_js method."""
    from curobo._src.state.state_joint import JointState

    robot_model = Kinematics(franka_cfg)

    # Get full joint state
    q_active = torch.zeros(7, **(cuda_device_cfg.as_torch_dict()))
    js_active = JointState.from_position(q_active, joint_names=robot_model.joint_names)
    full_js = robot_model.get_full_js(js_active)

    # Extract active joints back
    active_js = robot_model.get_active_js(full_js)

    assert active_js is not None
    assert active_js.position.shape == js_active.position.shape


def test_cuda_robot_model_compute_kinematics(franka_cfg, cuda_device_cfg):
    """Test compute_kinematics method (deprecated)."""
    from curobo._src.state.state_joint import JointState

    robot_model = Kinematics(franka_cfg)

    q = torch.zeros(7, **(cuda_device_cfg.as_torch_dict()))
    joint_state = JointState.from_position(q)

    state = robot_model.compute_kinematics(joint_state)

    assert state is not None
    assert state.robot_spheres is not None


def test_cuda_robot_model_compute_kinematics_from_joint_position(franka_cfg, cuda_device_cfg):
    """Test get_state with raw tensor input."""
    robot_model = Kinematics(franka_cfg)

    q = torch.zeros(1, 7, **(cuda_device_cfg.as_torch_dict()))

    state = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    assert state is not None
    assert state.robot_spheres is not None


def test_cuda_robot_model_get_link_transform(franka_cfg):
    """Test get_link_transform method."""
    robot_model = Kinematics(franka_cfg)

    link_name = robot_model.tool_frames[0]
    transform = robot_model.get_link_transform(link_name)

    assert transform is not None
    assert hasattr(transform, 'position')
    assert hasattr(transform, 'quaternion')


def test_cuda_robot_model_get_all_link_transforms(franka_cfg):
    """Test get_all_link_transforms method."""
    robot_model = Kinematics(franka_cfg)

    transforms = robot_model.get_all_link_transforms()

    assert transforms is not None
    assert hasattr(transforms, 'position')
    assert hasattr(transforms, 'quaternion')


def test_cuda_robot_model_update_kinematics_config(franka_cfg):
    """Test update_kinematics_config method."""
    robot_model = Kinematics(franka_cfg)

    # Get current config
    old_config = robot_model.kinematics_config

    # Update with same config (should work)
    robot_model.update_kinematics_config(old_config)

    # Config should be updated
    assert robot_model.kinematics_config is not None


def test_cuda_robot_model_get_robot_link_meshes(franka_cfg):
    """Test get_robot_link_meshes method."""
    robot_model = Kinematics(franka_cfg)

    # This requires kinematics_parser to be set
    if robot_model.config.kinematics_parser is not None:
        meshes = robot_model.get_robot_link_meshes()
        assert isinstance(meshes, list)


def test_cuda_robot_model_get_robot_as_mesh(franka_mesh_cfg, cuda_device_cfg):
    """Test get_robot_as_mesh method."""
    robot_model = Kinematics(franka_mesh_cfg)

    # This requires kinematics_parser to be set
    if robot_model.config.kinematics_parser is not None:
        q = torch.zeros(1, 7, **(cuda_device_cfg.as_torch_dict()))
        meshes = robot_model.get_robot_as_mesh(q)
        assert isinstance(meshes, list)


def test_cuda_robot_model_get_link_mesh(franka_cfg):
    """Test get_link_mesh method."""
    robot_model = Kinematics(franka_cfg)

    # This requires kinematics_parser to be set
    if robot_model.config.kinematics_parser is not None:
        link_name = robot_model.tool_frames[0]
        mesh = robot_model.get_link_mesh(link_name)
        assert mesh is not None


def test_cuda_robot_model_compute_kinematics_consistency(franka_cfg, cuda_device_cfg):
    """Test compute_kinematics and get_state produce the same result."""
    from curobo._src.state.state_joint import JointState

    robot_model = Kinematics(franka_cfg)

    q = torch.zeros(7, **(cuda_device_cfg.as_torch_dict()))
    joint_state = JointState.from_position(q, joint_names=robot_model.joint_names)

    state1 = robot_model.compute_kinematics(joint_state)
    state2 = robot_model.compute_kinematics(JointState.from_position(q, joint_names=robot_model.joint_names))

    assert torch.allclose(state1.robot_spheres, state2.robot_spheres, atol=1e-5)
