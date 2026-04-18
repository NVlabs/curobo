# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Validation tests for JointLimits to cover all error paths."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.robot.types.joint_limits import JointLimits


class TestJointLimitsShapeValidation:
    """Test all shape validation error paths in JointLimits.__post_init__."""

    def test_joint_limits_position_wrong_shape_rows(self, cuda_device_cfg):
        """Test error when position has wrong number of rows (not 2)."""
        joint_names = ["j1", "j2"]
        # Wrong: 3 rows instead of 2
        position = torch.tensor([[-3.0, -2.0], [3.0, 2.0], [0.0, 0.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-50.0, -50.0], [50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))

        with pytest.raises(ValueError, match="position shape does not match dof"):
            JointLimits(
                joint_names=joint_names,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                device_cfg=cuda_device_cfg,
            )

    def test_joint_limits_velocity_wrong_cols(self, cuda_device_cfg):
        """Test error when velocity has wrong number of columns."""
        joint_names = ["j1", "j2"]
        position = torch.tensor([[-3.0, -2.0], [3.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        # Wrong: 3 columns instead of 2
        velocity = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-50.0, -50.0], [50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))

        with pytest.raises(ValueError, match="velocity shape does not match dof"):
            JointLimits(
                joint_names=joint_names,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                device_cfg=cuda_device_cfg,
            )

    def test_joint_limits_acceleration_wrong_shape(self, cuda_device_cfg):
        """Test error when acceleration has wrong shape."""
        joint_names = ["j1", "j2"]
        position = torch.tensor([[-3.0, -2.0], [3.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        # Wrong: 1 column instead of 2
        acceleration = torch.tensor([[-5.0], [5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-50.0, -50.0], [50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))

        with pytest.raises(ValueError, match="acceleration shape does not match dof"):
            JointLimits(
                joint_names=joint_names,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                device_cfg=cuda_device_cfg,
            )

    def test_joint_limits_jerk_wrong_shape(self, cuda_device_cfg):
        """Test error when jerk has wrong shape."""
        joint_names = ["j1", "j2"]
        position = torch.tensor([[-3.0, -2.0], [3.0, 2.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], **(cuda_device_cfg.as_torch_dict()))
        # Wrong: 3 columns instead of 2
        jerk = torch.tensor([[-50.0, -50.0, -50.0], [50.0, 50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))

        with pytest.raises(ValueError, match="jerk shape does not match dof"):
            JointLimits(
                joint_names=joint_names,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                device_cfg=cuda_device_cfg,
            )


class TestJointLimitsValidateShapeMethod:
    """Test validate_shape() method error paths."""

    @pytest.fixture
    def basic_joint_limits(self, cuda_device_cfg):
        """Create basic JointLimits for testing."""
        joint_names = ["j1", "j2", "j3"]
        position = torch.tensor([[-3.0, -2.0, -1.0], [3.0, 2.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        velocity = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], **(cuda_device_cfg.as_torch_dict()))
        acceleration = torch.tensor([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]], **(cuda_device_cfg.as_torch_dict()))
        jerk = torch.tensor([[-50.0, -50.0, -50.0], [50.0, 50.0, 50.0]], **(cuda_device_cfg.as_torch_dict()))
        effort = torch.tensor([[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]], **(cuda_device_cfg.as_torch_dict()))

        return JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            effort=effort,
            device_cfg=cuda_device_cfg,
        )

    def test_validate_shape_velocity_mismatch(self, basic_joint_limits):
        """Test validate_shape raises error when velocity shape doesn't match DOF."""
        # basic_joint_limits has dof=3, call validate with dof=2
        # This will pass position check (line 134-135) but fail on velocity (line 136-137)
        with pytest.raises(ValueError, match="velocity shape does not match dof"):
            basic_joint_limits.validate_shape(2, check_effort=False)

    def test_validate_shape_acceleration_mismatch(self, basic_joint_limits):
        """Test validate_shape raises error when acceleration shape doesn't match DOF."""
        # Will pass position and velocity but fail on acceleration (line 138-141)
        with pytest.raises(ValueError, match="acceleration shape does not match dof"):
            basic_joint_limits.validate_shape(2, check_effort=False)

    def test_validate_shape_jerk_mismatch(self, basic_joint_limits):
        """Test validate_shape raises error when jerk shape doesn't match DOF."""
        # Will pass position, velocity, acceleration but fail on jerk (line 142-143)
        with pytest.raises(ValueError, match="jerk shape does not match dof"):
            basic_joint_limits.validate_shape(2, check_effort=False)

    def test_validate_shape_effort_mismatch(self, basic_joint_limits):
        """Test validate_shape raises error when effort shape doesn't match DOF with check_effort=True."""
        # Will pass all checks until effort with check_effort=True (line 144-145)
        with pytest.raises(ValueError, match="effort shape does not match dof"):
            basic_joint_limits.validate_shape(2, check_effort=True)

    def test_validate_shape_passes_correct_dof(self, basic_joint_limits):
        """Test validate_shape passes when DOF matches."""
        # Should not raise error
        basic_joint_limits.validate_shape(3, check_effort=True)
        # If we get here, test passed
        assert True

