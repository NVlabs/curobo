# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Integration tests for KinematicsParams using KinematicsLoader."""

# Third Party
import pytest
import torch

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

# CuRobo
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="class")
def franka_kinematics_cfg(cuda_device_cfg):
    """Load Franka robot and get kinematics config."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])
    return cfg.kinematics_config


class TestKinematicsParamsProperties:
    """Test KinematicsParams properties."""

    def test_kinematics_device_cfg_all_link_names(self, franka_kinematics_cfg):
        """Test all_link_names property."""
        all_links = franka_kinematics_cfg.all_link_names

        assert isinstance(all_links, list)
        assert len(all_links) > 0
        assert all(isinstance(name, str) for name in all_links)

    def test_kinematics_device_cfg_num_pose_links(self, franka_kinematics_cfg):
        """Test num_pose_links property."""
        num = franka_kinematics_cfg.num_pose_links

        assert isinstance(num, int)
        assert num > 0

    def test_kinematics_device_cfg_num_links(self, franka_kinematics_cfg):
        """Test num_links property."""
        num = franka_kinematics_cfg.num_links

        assert isinstance(num, int)
        assert num > 0

    def test_kinematics_device_cfg_num_spheres(self, franka_kinematics_cfg):
        """Test num_spheres property."""
        num = franka_kinematics_cfg.num_spheres

        assert isinstance(num, int)
        assert num > 0

    def test_kinematics_device_cfg_num_dof(self, franka_kinematics_cfg):
        """Test num_dof property."""
        num = franka_kinematics_cfg.num_dof

        assert isinstance(num, int)
        assert num == 7  # Franka has 7 DOF


class TestKinematicsParamsSphereOperations:
    """Test sphere-related methods of KinematicsParams."""

    def test_get_sphere_index_from_link_name(self, franka_kinematics_cfg):
        """Test getting sphere indices for a link."""
        link_name = "panda_hand"

        sphere_indices = franka_kinematics_cfg.get_sphere_index_from_link_name(link_name)

        assert sphere_indices is not None
        assert isinstance(sphere_indices, torch.Tensor)
        assert len(sphere_indices) > 0

    def test_get_link_spheres(self, franka_kinematics_cfg):
        """Test getting spheres for a link."""
        link_name = "panda_hand"

        spheres = franka_kinematics_cfg.get_link_spheres(link_name)

        assert spheres is not None
        assert isinstance(spheres, torch.Tensor)
        assert spheres.shape[-1] == 4  # x, y, z, radius

    def test_get_reference_link_spheres(self, franka_kinematics_cfg):
        """Test getting reference spheres for a link."""
        link_name = "panda_hand"

        spheres = franka_kinematics_cfg.get_reference_link_spheres(link_name)

        assert spheres is not None
        assert isinstance(spheres, torch.Tensor)
        assert spheres.shape[-1] == 4  # x, y, z, radius

    def test_update_link_spheres(self, franka_kinematics_cfg, cuda_device_cfg):
        """Test updating spheres for a link."""
        link_name = "panda_hand"

        # Get current spheres
        original_spheres = franka_kinematics_cfg.get_link_spheres(link_name).clone()

        # Create new spheres
        num_spheres = original_spheres.shape[0]
        new_spheres = torch.randn(num_spheres, 4, **(cuda_device_cfg.as_torch_dict()))
        new_spheres[:, 3] = 0.05  # Set radius

        # Update
        franka_kinematics_cfg.update_link_spheres(link_name, new_spheres)

        # Verify update
        updated_spheres = franka_kinematics_cfg.get_link_spheres(link_name)
        assert torch.allclose(updated_spheres, new_spheres, atol=1e-5)

        # Restore original
        franka_kinematics_cfg.update_link_spheres(link_name, original_spheres)

    def test_get_number_of_spheres(self, franka_kinematics_cfg):
        """Test getting number of spheres for a link."""
        link_name = "panda_hand"

        num_spheres = franka_kinematics_cfg.get_number_of_spheres(link_name)

        assert isinstance(num_spheres, int)
        assert num_spheres > 0

    def test_disable_enable_link_spheres(self, franka_kinematics_cfg):
        """Test disabling and enabling link spheres."""
        link_name = "panda_link5"

        # Get original spheres
        original_spheres = franka_kinematics_cfg.get_link_spheres(link_name).clone()

        # Disable spheres
        franka_kinematics_cfg.disable_link_spheres(link_name)
        disabled_spheres = franka_kinematics_cfg.get_link_spheres(link_name)

        # Radii should be negative when disabled
        assert torch.all(disabled_spheres[:, 3] < 0)

        # Enable spheres
        franka_kinematics_cfg.enable_link_spheres(link_name)
        enabled_spheres = franka_kinematics_cfg.get_link_spheres(link_name)

        # Should match original
        assert torch.allclose(enabled_spheres, original_spheres, atol=1e-5)

    def test_update_disable_link_spheres(self, franka_kinematics_cfg, cuda_device_cfg):
        """Test updating and disabling object link spheres."""
        link_name = "attached_object"

        # Get current sphere count for the link
        current_spheres = franka_kinematics_cfg.get_link_spheres(link_name)
        num_link_spheres = current_spheres.shape[0]

        # Create object spheres matching the link's sphere count
        object_spheres = torch.zeros(num_link_spheres, 4, **(cuda_device_cfg.as_torch_dict()))
        object_spheres[:, 3] = 0.02  # radius
        object_spheres[:, 0] = torch.linspace(0, 0.1, num_link_spheres)  # x positions

        franka_kinematics_cfg.update_link_spheres(link_name, object_spheres)

        # Verify update
        attached_spheres = franka_kinematics_cfg.get_link_spheres(link_name)
        assert attached_spheres.shape[0] == object_spheres.shape[0]

        franka_kinematics_cfg.disable_link_spheres(link_name)

        # Verify disable (spheres should have negative radii)
        detached_spheres = franka_kinematics_cfg.get_link_spheres(link_name)
        assert torch.all(detached_spheres[:, 3] < 0)


class TestKinematicsParamsLinkOperations:
    """Test link-related methods of KinematicsParams."""

    def test_get_link_masses_com(self, franka_kinematics_cfg):
        """Test getting link masses and COM."""
        link_name = "panda_link5"

        masses_com = franka_kinematics_cfg.get_link_masses_com(link_name)

        assert masses_com is not None
        assert isinstance(masses_com, torch.Tensor)
        assert masses_com.shape[-1] == 4  # [com_x, com_y, com_z, mass]

    def test_update_link_mass(self, franka_kinematics_cfg):
        """Test updating link mass."""
        link_name = "panda_link5"

        # Get original mass
        original_masses_com = franka_kinematics_cfg.get_link_masses_com(link_name).clone()
        original_mass = original_masses_com[3].item()

        # Update mass
        new_mass = 2.5
        franka_kinematics_cfg.update_link_mass(link_name, new_mass)

        # Verify update
        updated_masses_com = franka_kinematics_cfg.get_link_masses_com(link_name)
        assert abs(updated_masses_com[3].item() - new_mass) < 1e-5

        # Restore original
        franka_kinematics_cfg.update_link_mass(link_name, original_mass)

    def test_update_link_com(self, franka_kinematics_cfg, cuda_device_cfg):
        """Test updating link center of mass."""
        link_name = "panda_link5"

        # Get original COM
        original_masses_com = franka_kinematics_cfg.get_link_masses_com(link_name).clone()
        original_com = original_masses_com[:3].clone()

        # Update COM
        new_com = torch.tensor([0.1, 0.2, 0.3], **(cuda_device_cfg.as_torch_dict()))
        franka_kinematics_cfg.update_link_com(link_name, new_com)

        # Verify update
        updated_masses_com = franka_kinematics_cfg.get_link_masses_com(link_name)
        assert torch.allclose(updated_masses_com[:3], new_com, atol=1e-5)

        # Restore original
        franka_kinematics_cfg.update_link_com(link_name, original_com)

    def test_update_link_inertia(self, franka_kinematics_cfg, cuda_device_cfg):
        """Test updating link inertia."""
        link_name = "panda_link5"

        # Get original inertia
        original_inertia = franka_kinematics_cfg.get_link_inertia(link_name).clone()

        # Update inertia
        new_inertia = torch.tensor([0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003],
                                   **(cuda_device_cfg.as_torch_dict()))
        franka_kinematics_cfg.update_link_inertia(link_name, new_inertia)

        # Verify update
        updated_inertia = franka_kinematics_cfg.get_link_inertia(link_name)
        assert torch.allclose(updated_inertia, new_inertia, atol=1e-5)

        # Restore original
        franka_kinematics_cfg.update_link_inertia(link_name, original_inertia)

    def test_get_link_inertia(self, franka_kinematics_cfg):
        """Test getting link inertia."""
        link_name = "panda_link5"

        inertia = franka_kinematics_cfg.get_link_inertia(link_name)

        assert inertia is not None
        assert isinstance(inertia, torch.Tensor)
        assert inertia.shape[-1] == 6  # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]


class TestKinematicsParamsMethods:
    """Test general methods of KinematicsParams."""

    def test_kinematics_device_cfg_clone(self, franka_kinematics_cfg):
        """Test cloning KinematicsParams."""
        cloned = franka_kinematics_cfg.clone()

        assert cloned is not franka_kinematics_cfg
        assert cloned.joint_names == franka_kinematics_cfg.joint_names
        assert cloned.tool_frames == franka_kinematics_cfg.tool_frames
        assert torch.equal(cloned.link_spheres, franka_kinematics_cfg.link_spheres)

    def test_kinematics_device_cfg_make_contiguous(self, franka_kinematics_cfg):
        """Test make_contiguous method."""
        # Should make all tensors contiguous
        franka_kinematics_cfg.make_contiguous()

        # Verify tensors are contiguous
        assert franka_kinematics_cfg.fixed_transforms.is_contiguous()
        assert franka_kinematics_cfg.link_map.is_contiguous()
        assert franka_kinematics_cfg.joint_map.is_contiguous()

    def test_kinematics_device_cfg_copy_(self, franka_kinematics_cfg, cuda_device_cfg):
        """Test copy_ method."""
        # Create a minimal config to copy into
        robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
        cfg2 = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])
        kin_cfg2 = cfg2.kinematics_config

        # Save original values
        original_spheres = kin_cfg2.link_spheres.clone()

        # Copy franka config into cfg2
        kin_cfg2.copy_(franka_kinematics_cfg)

        # Verify copy
        assert torch.equal(kin_cfg2.link_spheres, franka_kinematics_cfg.link_spheres)

    def test_kinematics_device_cfg_get_robot_collision_geometry(self, franka_kinematics_cfg):
        """Test get_robot_collision_geometry method."""
        collision_geom = franka_kinematics_cfg.get_robot_collision_geometry()

        assert collision_geom is not None
        assert hasattr(collision_geom, 'num_links')
        assert hasattr(collision_geom, 'link_sphere_idx_map')

    def test_kinematics_device_cfg_load_cspace_cfg_from_kinematics(self, franka_kinematics_cfg):
        """Test loading CSpaceParams from kinematics."""
        cspace = franka_kinematics_cfg.load_cspace_cfg_from_kinematics()

        # May return None if cspace is already set in kinematics_config
        if cspace is not None:
            assert cspace.joint_names == franka_kinematics_cfg.joint_names
            assert cspace.default_joint_position is not None
        else:
            # If None, cspace should already exist
            assert franka_kinematics_cfg.cspace is not None


@pytest.mark.parametrize("robot_file,link_name", [
    ("franka.yml", "panda_hand"),
    ("ur10e.yml", "tool0"),
])
class TestKinematicsParamsMultiRobot:
    """Test KinematicsParams with different robots."""

    def test_kinematics_cfg_basic_properties(self, robot_file, link_name, cuda_device_cfg):
        """Test basic properties work for different robots."""
        robot_data = load_yaml(join_path(get_robot_configs_path(), robot_file))
        cfg = KinematicsCfg.from_robot_yaml_file(robot_data, [link_name])
        kin_cfg = cfg.kinematics_config

        assert kin_cfg.num_dof > 0
        assert kin_cfg.num_links > 0
        assert kin_cfg.num_spheres > 0
        assert len(kin_cfg.all_link_names) > 0

    def test_kinematics_cfg_sphere_operations(self, robot_file, link_name, cuda_device_cfg):
        """Test sphere operations work for different robots."""
        robot_data = load_yaml(join_path(get_robot_configs_path(), robot_file))
        cfg = KinematicsCfg.from_robot_yaml_file(robot_data, [link_name])
        kin_cfg = cfg.kinematics_config

        # Get spheres for the ee link
        spheres = kin_cfg.get_link_spheres(link_name)
        assert spheres is not None
        assert spheres.shape[-1] == 4

        # Get number of spheres
        num = kin_cfg.get_number_of_spheres(link_name)
        assert num == spheres.shape[0]


class TestKinematicsParamsExport:
    """Test export functionality of KinematicsParams."""

    def test_export_to_urdf_basic(self, franka_kinematics_cfg, tmp_path):
        """Test basic URDF export functionality."""
        # Create temporary output path
        output_path = str(tmp_path / "test_robot.urdf")

        # Export to URDF
        result = franka_kinematics_cfg.export_to_urdf(
            output_path=output_path,
            robot_name="test_robot",
        )

        # Check that file was created
        assert result is not None
        import os
        assert os.path.exists(output_path)

    def test_export_to_urdf_with_spheres(self, franka_kinematics_cfg, tmp_path):
        """Test URDF export with collision spheres."""
        output_path = str(tmp_path / "test_robot_spheres.urdf")

        result = franka_kinematics_cfg.export_to_urdf(
            output_path=output_path,
            robot_name="test_robot_spheres",
            include_spheres=True,
        )

        assert result is not None
        import os
        assert os.path.exists(output_path)

        # Verify file has content
        with open(output_path, 'r') as f:
            content = f.read()
            assert len(content) > 0
            assert "robot" in content.lower()

    def test_export_to_urdf_without_spheres(self, franka_kinematics_cfg, tmp_path):
        """Test URDF export without collision spheres."""
        output_path = str(tmp_path / "test_robot_no_spheres.urdf")

        result = franka_kinematics_cfg.export_to_urdf(
            output_path=output_path,
            robot_name="test_robot_no_spheres",
            include_spheres=False,
        )

        assert result is not None
        import os
        assert os.path.exists(output_path)

    def test_export_to_urdf_returns_urdf_object(self, franka_kinematics_cfg, tmp_path):
        """Test that export_to_urdf returns a URDF object."""
        output_path = str(tmp_path / "test_robot_obj.urdf")

        urdf_obj = franka_kinematics_cfg.export_to_urdf(
            output_path=output_path,
            robot_name="test_robot",
        )

        # Should return a yourdfpy URDF object
        assert urdf_obj is not None
        # Verify it's a URDF object
        assert type(urdf_obj).__name__ == 'URDF'


class TestKinematicsParamsEdgeCases:
    """Test edge cases and special functionality."""

    def test_make_contiguous_with_non_contiguous_tensors(self, franka_kinematics_cfg):
        """Test make_contiguous on non-contiguous tensors."""
        # Clone to avoid modifying the fixture
        import copy
        cfg = copy.deepcopy(franka_kinematics_cfg)

        # Store original
        original_transforms = cfg.fixed_transforms.clone()

        # Make it non-contiguous by selecting every other element and back
        cfg.fixed_transforms = cfg.fixed_transforms[::2]  # Slice makes non-contiguous
        if len(cfg.fixed_transforms) > 0:
            # Only test if we have elements
            may_not_be_contiguous = not cfg.fixed_transforms.is_contiguous()

            # Call make_contiguous
            cfg.make_contiguous()

            # After make_contiguous, should be contiguous
            assert cfg.fixed_transforms.is_contiguous()

    def test_load_cspace_cfg_from_kinematics(self, franka_kinematics_cfg):
        """Test loading cspace config from kinematics."""
        import copy
        cfg = copy.deepcopy(franka_kinematics_cfg)

        # Remove cspace
        cfg.cspace = None

        # Load cspace from kinematics
        cfg.load_cspace_cfg_from_kinematics()

        # Should have cspace now
        assert cfg.cspace is not None
        assert cfg.cspace.joint_names is not None
        assert len(cfg.cspace.joint_names) == cfg.num_dof

    def test_reference_link_spheres_auto_creation(self, franka_kinematics_cfg):
        """Test that reference_link_spheres is auto-created from link_spheres."""
        # When link_spheres exists and reference is None, reference should be cloned
        assert franka_kinematics_cfg.link_spheres is not None
        assert franka_kinematics_cfg.reference_link_spheres is not None

        # They should have the same values but be different objects
        assert torch.allclose(
            franka_kinematics_cfg.link_spheres,
            franka_kinematics_cfg.reference_link_spheres,
        )

