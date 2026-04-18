# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for cuda_robot_model types module."""

# Standard Library
import numpy as np

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.robot.types.joint_types import JointType
from curobo._src.robot.types.link_params import LinkParams


class TestJointType:
    """Tests for JointType enum."""

    def test_joint_type_enum_values(self):
        """Test that JointType enum has all expected values."""
        assert JointType.FIXED.value == -1
        assert JointType.X_PRISM.value == 0
        assert JointType.Y_PRISM.value == 1
        assert JointType.Z_PRISM.value == 2
        assert JointType.X_ROT.value == 3
        assert JointType.Y_ROT.value == 4
        assert JointType.Z_ROT.value == 5
        assert JointType.X_PRISM_NEG.value == 6
        assert JointType.Y_PRISM_NEG.value == 7
        assert JointType.Z_PRISM_NEG.value == 8
        assert JointType.X_ROT_NEG.value == 9
        assert JointType.Y_ROT_NEG.value == 10
        assert JointType.Z_ROT_NEG.value == 11

    def test_joint_type_name_access(self):
        """Test accessing JointType by name."""
        assert JointType["FIXED"] == JointType.FIXED
        assert JointType["X_ROT"] == JointType.X_ROT
        assert JointType["Z_PRISM"] == JointType.Z_PRISM

    def test_joint_type_value_access(self):
        """Test accessing JointType by value."""
        assert JointType(-1) == JointType.FIXED
        assert JointType(3) == JointType.X_ROT
        assert JointType(5) == JointType.Z_ROT

    def test_joint_type_comparison(self):
        """Test JointType equality and comparison."""
        assert JointType.FIXED == JointType.FIXED
        assert JointType.X_ROT != JointType.Y_ROT
        assert JointType.FIXED.value < JointType.X_PRISM.value


class TestLinkParams:
    """Tests for LinkParams dataclass."""

    @pytest.fixture
    def basic_link_params(self):
        """Create basic LinkParams for testing."""
        fixed_transform = np.eye(4)[:3, :]
        return LinkParams(
            link_name="test_link",
            joint_name="test_joint",
            joint_type=JointType.Z_ROT,
            fixed_transform=fixed_transform,
            parent_link_name="base_link",
        )

    def test_link_params_creation(self, basic_link_params):
        """Test basic LinkParams creation."""
        assert basic_link_params.link_name == "test_link"
        assert basic_link_params.joint_name == "test_joint"
        assert basic_link_params.joint_type == JointType.Z_ROT
        assert basic_link_params.parent_link_name == "base_link"

    def test_link_params_default_values(self, basic_link_params):
        """Test LinkParams default values."""
        assert basic_link_params.joint_velocity_limits == [-2.0, 2.0]
        assert basic_link_params.joint_offset == [1.0, 0.0]
        assert basic_link_params.joint_effort_limit == [10000.0]
        assert basic_link_params.link_mass == 0.01
        assert np.array_equal(basic_link_params.link_com, np.array([0.0, 0.0, 0.0]))

    def test_link_params_fixed_transform_shape(self):
        """Test that fixed_transform must be shape (3, 4)."""
        with pytest.raises(ValueError):
            LinkParams(
                link_name="test_link",
                joint_name="test_joint",
                joint_type=JointType.FIXED,
                fixed_transform=np.eye(4),  # Wrong shape (4, 4)
            )

    def test_link_params_get_link_com_and_mass(self, basic_link_params):
        """Test get_link_com_and_mass method."""
        com_mass = basic_link_params.get_link_com_and_mass()
        assert com_mass.shape == (4,)
        assert np.array_equal(com_mass[:3], basic_link_params.link_com)
        assert com_mass[3] == basic_link_params.link_mass

    def test_link_params_create(self):
        """Test creating LinkParams from dictionary."""
        dict_data = {
            "link_name": "test_link",
            "joint_name": "test_joint",
            "joint_type": "Z_ROT",
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],  # pose format
            "parent_link_name": "base_link",
        }

        link_params = LinkParams.create(dict_data)
        assert link_params.link_name == "test_link"
        assert link_params.joint_type == JointType.Z_ROT
        assert link_params.fixed_transform.shape == (3, 4)

    def test_link_params_custom_values(self):
        """Test LinkParams with custom values."""
        fixed_transform = np.eye(4)[:3, :]
        link_params = LinkParams(
            link_name="custom_link",
            joint_name="custom_joint",
            joint_type=JointType.X_PRISM,
            fixed_transform=fixed_transform,
            joint_velocity_limits=[-5.0, 5.0],
            joint_effort_limit=5000.0,
            link_mass=0.5,
        )

        assert link_params.joint_velocity_limits == [-5.0, 5.0]
        assert link_params.joint_effort_limit == 5000.0
        assert link_params.link_mass == 0.5


class TestJointLimits:
    """Tests for JointLimits dataclass."""

    @pytest.fixture
    def basic_joint_limits(self, cuda_device_cfg):
        """Create basic JointLimits for testing."""
        joint_names = ["joint1", "joint2", "joint3"]
        position = torch.tensor([[-3.14, -2.0, -1.5], [3.14, 2.0, 1.5]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]], **(cuda_device_cfg.as_torch_dict()))

        return JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            device_cfg=cuda_device_cfg,
        )

    def test_joint_limits_creation(self, basic_joint_limits):
        """Test basic JointLimits creation."""
        assert len(basic_joint_limits.joint_names) == 3
        assert basic_joint_limits.position.shape == (2, 3)
        assert basic_joint_limits.velocity.shape == (2, 3)
        assert basic_joint_limits.acceleration.shape == (2, 3)
        assert basic_joint_limits.jerk.shape == (2, 3)

    def test_joint_limits_shape_validation(self, cuda_device_cfg):
        """Test that JointLimits validates tensor shapes."""
        joint_names = ["joint1", "joint2"]
        position = torch.tensor([[-3.14, -2.0], [3.14, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))  # Wrong shape
        acceleration = torch.tensor([[-10.0, -10.0], [10.0, 10.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-100.0, -100.0], [100.0, 100.0]], **(cuda_device_cfg.as_torch_dict()))

        with pytest.raises(ValueError):
            JointLimits(
                joint_names=joint_names,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                device_cfg=cuda_device_cfg,
            )

    def test_joint_limits_lower_greater_than_upper(self, cuda_device_cfg):
        """Test that lower limits must be less than upper limits."""
        joint_names = ["joint1"]
        position = torch.tensor([[3.14], [-3.14]], **(cuda_device_cfg.as_torch_dict()))  # Inverted
        velocity = torch.tensor([[-2.0], [2.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-10.0], [10.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-100.0], [100.0]], **(cuda_device_cfg.as_torch_dict()))

        with pytest.raises(ValueError):
            JointLimits(
                joint_names=joint_names,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                device_cfg=cuda_device_cfg,
            )

    def test_joint_limits_with_effort(self, cuda_device_cfg):
        """Test JointLimits with optional effort limits."""
        joint_names = ["joint1", "joint2"]
        position = torch.tensor([[-3.14, -2.0], [3.14, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-10.0, -10.0], [10.0, 10.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-100.0, -100.0], [100.0, 100.0]], **(cuda_device_cfg.as_torch_dict()))
        effort = torch.tensor([[-50.0, -50.0], [50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))

        joint_limits = JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            effort=effort,
            device_cfg=cuda_device_cfg,
        )

        assert joint_limits.effort is not None
        assert joint_limits.effort.shape == (2, 2)

    @pytest.mark.parametrize("dof", [1, 3, 7, 10])
    def test_joint_limits_different_dof(self, dof, cuda_device_cfg):
        """Test JointLimits with different DOF values."""
        joint_names = [f"joint{i}" for i in range(dof)]
        position = torch.rand(2, dof, **(cuda_device_cfg.as_torch_dict())) * 2 - 1
        position[0] = -torch.abs(position[0])  # Ensure lower < upper
        position[1] = torch.abs(position[1])
        velocity = torch.ones(2, dof, **(cuda_device_cfg.as_torch_dict()))
        velocity[0] = -velocity[0]
        acceleration = torch.ones(2, dof, **(cuda_device_cfg.as_torch_dict())) * 10
        acceleration[0] = -acceleration[0]
        jerk = torch.ones(2, dof, **(cuda_device_cfg.as_torch_dict())) * 100
        jerk[0] = -jerk[0]

        joint_limits = JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            device_cfg=cuda_device_cfg,
        )

        assert len(joint_limits.joint_names) == dof
        assert joint_limits.position.shape == (2, dof)


class TestCSpaceParams:
    """Tests for CSpaceParams dataclass."""

    @pytest.fixture
    def basic_cspace_cfg(self, cuda_device_cfg):
        """Create basic CSpaceParams for testing."""
        joint_names = ["joint1", "joint2", "joint3"]
        return CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.zeros(3, **(cuda_device_cfg.as_torch_dict())),
            null_space_weight=torch.ones(3, **(cuda_device_cfg.as_torch_dict())),
            cspace_distance_weight=torch.ones(3, **(cuda_device_cfg.as_torch_dict())),
            device_cfg=cuda_device_cfg,
        )

    def test_cspace_cfg_creation(self, basic_cspace_cfg):
        """Test basic CSpaceParams creation."""
        assert len(basic_cspace_cfg.joint_names) == 3
        assert basic_cspace_cfg.default_joint_position.shape == (3,)
        assert basic_cspace_cfg.null_space_weight.shape == (3,)
        assert basic_cspace_cfg.cspace_distance_weight.shape == (3,)

    def test_cspace_cfg_inplace_reindex(self, cuda_device_cfg):
        """Test inplace_reindex method."""
        joint_names = ["joint1", "joint2", "joint3", "joint4"]
        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.tensor([0.0, 1.0, 2.0, 3.0], **(cuda_device_cfg.as_torch_dict())),
            null_space_weight=torch.ones(4, **(cuda_device_cfg.as_torch_dict())),
            cspace_distance_weight=torch.ones(4, **(cuda_device_cfg.as_torch_dict())),
            device_cfg=cuda_device_cfg,
        )

        # Reindex to a subset
        new_joint_names = ["joint2", "joint4"]
        cspace.inplace_reindex(new_joint_names)

        assert len(cspace.joint_names) == 2
        assert cspace.default_joint_position.shape == (2,)
        # Check that values are correctly reindexed
        assert torch.allclose(cspace.default_joint_position, torch.tensor([1.0, 3.0], **(cuda_device_cfg.as_torch_dict())))

    def test_cspace_cfg_with_max_acceleration(self, cuda_device_cfg):
        """Test CSpaceParams with max_acceleration parameter."""
        joint_names = ["joint1", "joint2"]
        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.zeros(2, **(cuda_device_cfg.as_torch_dict())),
            max_acceleration=15.0,
            device_cfg=cuda_device_cfg,
        )

        expected = torch.tensor([15.0, 15.0], **(cuda_device_cfg.as_torch_dict())).to(cspace.max_acceleration.dtype)
        assert torch.allclose(cspace.max_acceleration, expected)

    def test_cspace_cfg_with_max_jerk(self, cuda_device_cfg):
        """Test CSpaceParams with max_jerk parameter."""
        joint_names = ["joint1", "joint2", "joint3"]
        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.zeros(3, **(cuda_device_cfg.as_torch_dict())),
            max_jerk=500.0,
            device_cfg=cuda_device_cfg,
        )

        assert cspace.max_jerk is not None
        assert cspace.max_jerk.shape[0] == 3


class TestKinematicsParams:
    """Tests for KinematicsParams dataclass.

    Note: KinematicsParams requires many complex tensors and is typically
    created by KinematicsLoader. Testing it directly requires extensive setup.
    It's better tested through integration tests with Kinematics.
    """

    @pytest.mark.skip(reason="KinematicsParams requires extensive setup, tested via integration")
    def test_kinematics_device_cfg_placeholder(self):
        """Placeholder test - KinematicsParams tested via Kinematics."""
        pass


class TestIntegration:
    """Integration tests for types module."""

    def test_link_params_with_all_joint_types(self):
        """Test LinkParams creation with all JointType values."""
        fixed_transform = np.eye(4)[:3, :]

        for joint_type in JointType:
            link_params = LinkParams(
                link_name=f"link_{joint_type.name}",
                joint_name=f"joint_{joint_type.name}",
                joint_type=joint_type,
                fixed_transform=fixed_transform,
            )
            assert link_params.joint_type == joint_type

    def test_joint_limits_and_cspace_consistency(self, cuda_device_cfg):
        """Test that JointLimits and CSpaceParams have consistent joint_names."""
        joint_names = ["joint1", "joint2", "joint3"]

        # Create JointLimits
        position = torch.tensor([[-3.14, -2.0, -1.5], [3.14, 2.0, 1.5]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]], **(cuda_device_cfg.as_torch_dict()))

        joint_limits = JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            device_cfg=cuda_device_cfg,
        )

        # Create CSpaceParams
        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.zeros(3, **(cuda_device_cfg.as_torch_dict())),
            device_cfg=cuda_device_cfg,
        )

        # Verify consistency
        assert joint_limits.joint_names == cspace.joint_names


class TestCSpaceParamsAdvanced:
    """Advanced tests for CSpaceParams."""

    def test_cspace_cfg_velocity_scale(self, cuda_device_cfg):
        """Test CSpaceParams with velocity_scale."""
        joint_names = ["joint1", "joint2"]
        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.zeros(2, **(cuda_device_cfg.as_torch_dict())),
            velocity_scale=torch.ones(2, **(cuda_device_cfg.as_torch_dict())) * 0.5,
            device_cfg=cuda_device_cfg,
        )

        assert cspace.velocity_scale is not None
        assert torch.allclose(cspace.velocity_scale, torch.tensor([0.5, 0.5], **(cuda_device_cfg.as_torch_dict())))

    def test_cspace_cfg_acceleration_scale(self, cuda_device_cfg):
        """Test CSpaceParams with acceleration_scale."""
        joint_names = ["joint1", "joint2", "joint3"]
        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.zeros(3, **(cuda_device_cfg.as_torch_dict())),
            acceleration_scale=torch.ones(3, **(cuda_device_cfg.as_torch_dict())) * 2.0,
            device_cfg=cuda_device_cfg,
        )

        assert cspace.acceleration_scale is not None
        assert cspace.acceleration_scale.shape[0] == 3

    def test_cspace_cfg_jerk_scale(self, cuda_device_cfg):
        """Test CSpaceParams with jerk_scale."""
        joint_names = ["joint1"]
        jerk_scale_tensor = torch.ones(1, **(cuda_device_cfg.as_torch_dict())) * 1.5
        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.zeros(1, **(cuda_device_cfg.as_torch_dict())),
            jerk_scale=jerk_scale_tensor,
            device_cfg=cuda_device_cfg,
        )

        assert cspace.jerk_scale is not None
        assert len(cspace.jerk_scale.shape) > 0

    def test_cspace_cfg_null_space_maximum_distance(self, cuda_device_cfg):
        """Test CSpaceParams with null_space_maximum_distance."""
        joint_names = ["joint1", "joint2"]
        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.zeros(2, **(cuda_device_cfg.as_torch_dict())),
            null_space_maximum_distance=torch.ones(2, **(cuda_device_cfg.as_torch_dict())) * 0.1,
            device_cfg=cuda_device_cfg,
        )

        assert cspace.null_space_maximum_distance is not None
        assert cspace.null_space_maximum_distance.shape[0] == 2

    def test_cspace_cfg_position_limit_clip(self, cuda_device_cfg):
        """Test CSpaceParams with position_limit_clip."""
        joint_names = ["joint1", "joint2", "joint3"]
        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.zeros(3, **(cuda_device_cfg.as_torch_dict())),
            position_limit_clip=0.05,
            device_cfg=cuda_device_cfg,
        )

        assert cspace.position_limit_clip == 0.05


class TestJointLimitsAdvanced:
    """Advanced tests for JointLimits."""

    def test_joint_limits_numpy_input(self, cuda_device_cfg):
        """Test JointLimits with numpy arrays as input."""
        joint_names = ["joint1", "joint2"]
        position_np = np.array([[-3.14, -2.0], [3.14, 2.0]])

        position = torch.from_numpy(position_np).to(**cuda_device_cfg.as_torch_dict())
        velocity = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-10.0, -10.0], [10.0, 10.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-100.0, -100.0], [100.0, 100.0]], **(cuda_device_cfg.as_torch_dict()))

        joint_limits = JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            device_cfg=cuda_device_cfg,
        )

        assert joint_limits.position.shape == (2, 2)

    def test_joint_limits_device_transfer(self, cuda_device_cfg):
        """Test that JointLimits tensors are on correct device."""
        joint_names = ["joint1", "joint2", "joint3"]
        position = torch.tensor([[-3.14, -2.0, -1.5], [3.14, 2.0, 1.5]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]], **(cuda_device_cfg.as_torch_dict()))

        joint_limits = JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            device_cfg=cuda_device_cfg,
        )

        # All tensors should be on the specified device
        assert joint_limits.position.device == cuda_device_cfg.device
        assert joint_limits.velocity.device == cuda_device_cfg.device
        assert joint_limits.acceleration.device == cuda_device_cfg.device
        assert joint_limits.jerk.device == cuda_device_cfg.device


class TestLinkParamsAdvanced:
    """Advanced tests for LinkParams."""

    def test_link_params_mimic_joint(self):
        """Test LinkParams with mimic joint."""
        fixed_transform = np.eye(4)[:3, :]
        link_params = LinkParams(
            link_name="mimic_link",
            joint_name="mimic_joint",
            joint_type=JointType.Z_ROT,
            fixed_transform=fixed_transform,
            mimic_joint_name="parent_joint",
        )

        assert link_params.mimic_joint_name == "parent_joint"

    def test_link_params_joint_axis(self):
        """Test LinkParams with custom joint axis."""
        fixed_transform = np.eye(4)[:3, :]
        joint_axis = np.array([0.0, 0.0, 1.0])

        link_params = LinkParams(
            link_name="test_link",
            joint_name="test_joint",
            joint_type=JointType.Z_ROT,
            fixed_transform=fixed_transform,
            joint_axis=joint_axis,
        )

        assert np.array_equal(link_params.joint_axis, joint_axis)

    def test_link_params_custom_inertia(self):
        """Test LinkParams with custom inertia values."""
        fixed_transform = np.eye(4)[:3, :]
        link_inertia = np.array([0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003])

        link_params = LinkParams(
            link_name="heavy_link",
            joint_name="heavy_joint",
            joint_type=JointType.X_PRISM,
            fixed_transform=fixed_transform,
            link_inertia=link_inertia,
        )

        assert np.array_equal(link_params.link_inertia, link_inertia)

    def test_link_params_joint_id(self):
        """Test LinkParams with joint_id."""
        fixed_transform = np.eye(4)[:3, :]
        link_params = LinkParams(
            link_name="test_link",
            joint_name="test_joint",
            joint_type=JointType.Z_ROT,
            fixed_transform=fixed_transform,
            joint_id=5,
        )

        assert link_params.joint_id == 5

    def test_link_params_invalid_com_shape(self):
        """Test LinkParams with invalid COM shape raises error."""
        fixed_transform = np.eye(4)[:3, :]
        link_params = LinkParams(
            link_name="test_link",
            joint_name="test_joint",
            joint_type=JointType.Z_ROT,
            fixed_transform=fixed_transform,
            link_com=np.array([0.0, 0.0]),  # Wrong shape - should be (3,)
        )

        # get_link_com_and_mass should raise error for wrong shape
        with pytest.raises(ValueError, match="link_com shape does not match"):
            link_params.get_link_com_and_mass()


class TestJointLimitsMethods:
    """Tests for JointLimits methods."""

    def test_joint_limits_from_data_dict(self, cuda_device_cfg):
        """Test creating JointLimits from dictionary."""
        data = {
            "joint_names": ["j1", "j2", "j3"],
            "position": [[-3.0, -2.0, -1.0], [3.0, 2.0, 1.0]],
            "velocity": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            "acceleration": [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]],
            "jerk": [[-50.0, -50.0, -50.0], [50.0, 50.0, 50.0]],
        }

        joint_limits = JointLimits.from_data_dict(data, cuda_device_cfg)

        assert joint_limits.joint_names == ["j1", "j2", "j3"]
        assert joint_limits.position.shape == (2, 3)
        assert joint_limits.effort is None

    def test_joint_limits_from_data_dict_with_effort(self, cuda_device_cfg):
        """Test creating JointLimits from dictionary with effort."""
        data = {
            "joint_names": ["j1", "j2"],
            "position": [[-3.0, -2.0], [3.0, 2.0]],
            "velocity": [[-1.0, -1.0], [1.0, 1.0]],
            "acceleration": [[-5.0, -5.0], [5.0, 5.0]],
            "jerk": [[-50.0, -50.0], [50.0, 50.0]],
            "effort": [[-100.0, -80.0], [100.0, 80.0]],
        }

        joint_limits = JointLimits.from_data_dict(data, cuda_device_cfg)

        assert joint_limits.effort is not None
        assert joint_limits.effort.shape == (2, 2)

    def test_joint_limits_clone(self, cuda_device_cfg):
        """Test cloning JointLimits."""
        joint_names = ["j1", "j2"]
        position = torch.tensor([[-3.0, -2.0], [3.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-50.0, -50.0], [50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))

        joint_limits = JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            device_cfg=cuda_device_cfg,
        )

        cloned = joint_limits.clone()

        assert cloned is not joint_limits
        assert cloned.joint_names == joint_limits.joint_names
        assert torch.equal(cloned.position, joint_limits.position)
        assert torch.equal(cloned.velocity, joint_limits.velocity)

        # Verify independence
        cloned.position[0, 0] = -999.0
        assert joint_limits.position[0, 0] != -999.0

    def test_joint_limits_copy_(self, cuda_device_cfg):
        """Test in-place copy of JointLimits."""
        joint_names = ["j1", "j2"]
        position1 = torch.tensor([[-3.0, -2.0], [3.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity1 = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration1 = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk1 = torch.tensor([[-50.0, -50.0], [50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))

        jl1 = JointLimits(joint_names, position1, velocity1, acceleration1, jerk1, device_cfg=cuda_device_cfg)

        position2 = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity2 = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], **(cuda_device_cfg.as_torch_dict()))
        acceleration2 = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk2 = torch.tensor([[-20.0, -20.0], [20.0, 20.0]], **(cuda_device_cfg.as_torch_dict()))

        jl2 = JointLimits(joint_names, position2, velocity2, acceleration2, jerk2, device_cfg=cuda_device_cfg)

        # Copy jl2 into jl1
        jl1.copy_(jl2)

        assert torch.equal(jl1.position, jl2.position)
        assert torch.equal(jl1.velocity, jl2.velocity)

    def test_joint_limits_validate_shape(self, cuda_device_cfg):
        """Test validate_shape method."""
        joint_names = ["j1", "j2", "j3"]
        position = torch.tensor([[-3.0, -2.0, -1.0], [3.0, 2.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-50.0, -50.0, -50.0], [50.0, 50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))

        joint_limits = JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            device_cfg=cuda_device_cfg,
        )

        # Should not raise error for correct DOF
        joint_limits.validate_shape(3, check_effort=False)

    def test_joint_limits_validate_shape_error(self, cuda_device_cfg):
        """Test validate_shape raises error for wrong DOF."""
        joint_names = ["j1", "j2", "j3"]
        position = torch.tensor([[-3.0, -2.0, -1.0], [3.0, 2.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-50.0, -50.0, -50.0], [50.0, 50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))

        joint_limits = JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            device_cfg=cuda_device_cfg,
        )

        # Should raise error for wrong DOF
        with pytest.raises(ValueError):
            joint_limits.validate_shape(5, check_effort=False)

    def test_joint_limits_position_properties(self, cuda_device_cfg):
        """Test position_lower_limits and position_upper_limits properties."""
        joint_names = ["j1", "j2"]
        position = torch.tensor([[-3.0, -2.0], [3.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-50.0, -50.0], [50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))

        joint_limits = JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            device_cfg=cuda_device_cfg,
        )

        lower = joint_limits.position_lower_limits
        upper = joint_limits.position_upper_limits

        assert torch.equal(lower, torch.tensor([-3.0, -2.0], **(cuda_device_cfg.as_torch_dict())))
        assert torch.equal(upper, torch.tensor([3.0, 2.0], **(cuda_device_cfg.as_torch_dict())))

    def test_joint_limits_inverted_velocity_limits(self, cuda_device_cfg):
        """Test that inverted velocity limits raise error."""
        joint_names = ["j1"]
        position = torch.tensor([[-3.0], [3.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[2.0], [-2.0]], **(cuda_device_cfg.as_torch_dict()))  # Inverted
        acceleration = torch.tensor([[-5.0], [5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-50.0], [50.0]], **(cuda_device_cfg.as_torch_dict()))

        with pytest.raises(ValueError, match="lower velocity limits must be less than upper"):
            JointLimits(
                joint_names=joint_names,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                device_cfg=cuda_device_cfg,
            )

    def test_joint_limits_inverted_acceleration_limits(self, cuda_device_cfg):
        """Test that inverted acceleration limits raise error."""
        joint_names = ["j1"]
        position = torch.tensor([[-3.0], [3.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-2.0], [2.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[5.0], [-5.0]], **(cuda_device_cfg.as_torch_dict()))  # Inverted
        jerk = torch.tensor([[-50.0], [50.0]], **(cuda_device_cfg.as_torch_dict()))

        with pytest.raises(ValueError, match="lower acceleration limits must be less than upper"):
            JointLimits(
                joint_names=joint_names,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                device_cfg=cuda_device_cfg,
            )

    def test_joint_limits_inverted_jerk_limits(self, cuda_device_cfg):
        """Test that inverted jerk limits raise error."""
        joint_names = ["j1"]
        position = torch.tensor([[-3.0], [3.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-2.0], [2.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-5.0], [5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[50.0], [-50.0]], **(cuda_device_cfg.as_torch_dict()))  # Inverted

        with pytest.raises(ValueError, match="lower jerk limits must be less than upper"):
            JointLimits(
                joint_names=joint_names,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                device_cfg=cuda_device_cfg,
            )

    def test_joint_limits_inverted_effort_limits(self, cuda_device_cfg):
        """Test that inverted effort limits raise error."""
        joint_names = ["j1"]
        position = torch.tensor([[-3.0], [3.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-2.0], [2.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-5.0], [5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-50.0], [50.0]], **(cuda_device_cfg.as_torch_dict()))
        effort = torch.tensor([[100.0], [-100.0]], **(cuda_device_cfg.as_torch_dict()))  # Inverted

        with pytest.raises(ValueError, match="lower effort limits must be less than upper"):
            JointLimits(
                joint_names=joint_names,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                effort=effort,
                device_cfg=cuda_device_cfg,
            )

    def test_joint_limits_validate_shape_with_effort(self, cuda_device_cfg):
        """Test validate_shape with effort checking."""
        joint_names = ["j1", "j2"]
        position = torch.tensor([[-3.0, -2.0], [3.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-50.0, -50.0], [50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))
        effort = torch.tensor([[-100.0, -100.0], [100.0, 100.0]], **(cuda_device_cfg.as_torch_dict()))

        joint_limits = JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            effort=effort,
            device_cfg=cuda_device_cfg,
        )

        # Should pass with correct DOF and effort checking
        joint_limits.validate_shape(2, check_effort=True)


class TestCSpaceParamsMethods:
    """Tests for CSpaceParams methods."""

    def test_cspace_cfg_clone(self, cuda_device_cfg):
        """Test cloning CSpaceParams."""
        joint_names = ["joint1", "joint2"]
        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.tensor([0.5, 1.0], **(cuda_device_cfg.as_torch_dict())),
            null_space_weight=torch.ones(2, **(cuda_device_cfg.as_torch_dict())),
            device_cfg=cuda_device_cfg,
        )

        cloned = cspace.clone()

        assert cloned is not cspace
        assert cloned.joint_names == cspace.joint_names
        assert torch.equal(cloned.default_joint_position, cspace.default_joint_position)

        # Verify independence
        cloned.default_joint_position[0] = 999.0
        assert cspace.default_joint_position[0] != 999.0

    def test_cspace_cfg_copy_(self, cuda_device_cfg):
        """Test in-place copy of CSpaceParams."""
        joint_names = ["joint1", "joint2"]
        cspace1 = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.zeros(2, **(cuda_device_cfg.as_torch_dict())),
            device_cfg=cuda_device_cfg,
        )

        cspace2 = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.ones(2, **(cuda_device_cfg.as_torch_dict())),
            device_cfg=cuda_device_cfg,
        )

        cspace1.copy_(cspace2)

        assert torch.equal(cspace1.default_joint_position, cspace2.default_joint_position)

    def test_cspace_cfg_scale_joint_limits(self, cuda_device_cfg):
        """Test scaling joint limits with CSpaceParams scales."""
        joint_names = ["joint1", "joint2"]
        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=torch.zeros(2, **(cuda_device_cfg.as_torch_dict())),
            velocity_scale=torch.ones(2, **(cuda_device_cfg.as_torch_dict())) * 0.5,
            acceleration_scale=torch.ones(2, **(cuda_device_cfg.as_torch_dict())) * 0.8,
            jerk_scale=torch.ones(2, **(cuda_device_cfg.as_torch_dict())) * 0.9,
            device_cfg=cuda_device_cfg,
        )

        # Create joint limits
        position = torch.tensor([[-3.0, -2.0], [3.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-10.0, -10.0], [10.0, 10.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-100.0, -100.0], [100.0, 100.0]], **(cuda_device_cfg.as_torch_dict()))

        joint_limits = JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            device_cfg=cuda_device_cfg,
        )

        scaled = cspace.scale_joint_limits(joint_limits)

        # Velocity should be scaled by 0.5
        assert torch.allclose(scaled.velocity, velocity * 0.5)
        # Acceleration should be scaled by 0.8
        assert torch.allclose(scaled.acceleration, acceleration * 0.8)
        # Jerk should be scaled by 0.9
        assert torch.allclose(scaled.jerk, jerk * 0.9)

    def test_cspace_cfg_load_from_joint_limits(self, cuda_device_cfg):
        """Test loading CSpaceParams from joint limits."""
        joint_names = ["joint1", "joint2", "joint3"]
        joint_upper = torch.tensor([3.0, 2.0, 1.5], **(cuda_device_cfg.as_torch_dict()))
        joint_lower = torch.tensor([-3.0, -2.0, -1.5], **(cuda_device_cfg.as_torch_dict()))

        cspace = CSpaceParams.load_from_joint_limits(
            joint_position_upper=joint_upper,
            joint_position_lower=joint_lower,
            joint_names=joint_names,
            device_cfg=cuda_device_cfg,
        )

        assert cspace is not None
        assert cspace.joint_names == joint_names
        assert cspace.default_joint_position is not None
        # Retract config should be in the middle of limits
        expected = (joint_upper + joint_lower) / 2.0
        assert torch.allclose(cspace.default_joint_position, expected)

    def test_cspace_cfg_validation_cspace_distance_weight_mismatch(self, cuda_device_cfg):
        """Test CSpaceParams raises error for mismatched cspace_distance_weight shape."""
        joint_names = ["joint1", "joint2"]

        with pytest.raises(ValueError, match="cspace_distance_weight shape"):
            CSpaceParams(
                joint_names=joint_names,
                default_joint_position=torch.zeros(2, **(cuda_device_cfg.as_torch_dict())),
                cspace_distance_weight=torch.ones(5, **(cuda_device_cfg.as_torch_dict())),  # Wrong size
                device_cfg=cuda_device_cfg,
            )

    def test_cspace_cfg_validation_null_space_weight_mismatch(self, cuda_device_cfg):
        """Test CSpaceParams raises error for mismatched null_space_weight shape."""
        joint_names = ["joint1", "joint2"]

        with pytest.raises(ValueError, match="null_space_weight shape"):
            CSpaceParams(
                joint_names=joint_names,
                default_joint_position=torch.zeros(2, **(cuda_device_cfg.as_torch_dict())),
                null_space_weight=torch.ones(3, **(cuda_device_cfg.as_torch_dict())),  # Wrong size
                device_cfg=cuda_device_cfg,
            )

    def test_cspace_cfg_validation_null_space_maximum_distance_mismatch(self, cuda_device_cfg):
        """Test CSpaceParams raises error for mismatched null_space_maximum_distance shape."""
        joint_names = ["joint1", "joint2", "joint3"]

        with pytest.raises(ValueError, match="null_space_maximum_distance shape"):
            CSpaceParams(
                joint_names=joint_names,
                default_joint_position=torch.zeros(3, **(cuda_device_cfg.as_torch_dict())),
                null_space_maximum_distance=torch.ones(2, **(cuda_device_cfg.as_torch_dict())),  # Wrong size
                device_cfg=cuda_device_cfg,
            )

    def test_cspace_cfg_list_input_conversion(self, cuda_device_cfg):
        """Test that CSpaceParams converts list inputs to tensors."""
        joint_names = ["joint1", "joint2"]

        cspace = CSpaceParams(
            joint_names=joint_names,
            default_joint_position=[0.5, 1.0],  # List input
            position_limit_clip=[0.1, 0.2],  # List input
            device_cfg=cuda_device_cfg,
        )

        # Should be converted to tensors
        assert isinstance(cspace.default_joint_position, torch.Tensor)
        assert isinstance(cspace.position_limit_clip, torch.Tensor)

