# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Comprehensive tests for transform functions."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.geom.transform import (
    batch_transform_points,
    quaternion_rate_to_axis_angle_rate,
    transform_points,
)


@pytest.fixture
def device_cfg_dict():
    """Get tensor config as dict."""
    return {"device": torch.device("cuda"), "dtype": torch.float32}


class TestQuaternionRateConversion:
    """Test quaternion_rate_to_axis_angle_rate function."""

    def test_basic_conversion(self, device_cfg_dict):
        """Test basic quaternion rate conversion (lines 42-65)."""
        # Identity quaternion
        current_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], **device_cfg_dict)

        # Small rate
        quat_rate = torch.tensor([0.0, 0.1, 0.0, 0.0], **device_cfg_dict)

        omega = quaternion_rate_to_axis_angle_rate(quat_rate, current_quat)

        assert omega.shape == (3,)
        # Should produce angular velocity
        assert omega.abs().sum() > 0

    def test_batch_conversion(self, device_cfg_dict):
        """Test batched quaternion rate conversion."""
        # Batch of quaternions
        current_quat = torch.randn(10, 4, **device_cfg_dict)
        quat_rate = torch.randn(10, 4, **device_cfg_dict)

        omega = quaternion_rate_to_axis_angle_rate(quat_rate, current_quat)

        assert omega.shape == (10, 3)

    def test_zero_rate(self, device_cfg_dict):
        """Test with zero rate."""
        current_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], **device_cfg_dict)
        quat_rate = torch.zeros(4, **device_cfg_dict)

        omega = quaternion_rate_to_axis_angle_rate(quat_rate, current_quat)

        # Zero rate should give zero angular velocity
        assert torch.allclose(omega, torch.zeros(3, **device_cfg_dict))


class TestTransformPoints:
    """Test transform_points function."""

    def test_basic_transform(self, device_cfg_dict):
        """Test basic point transformation."""
        position = torch.tensor([1.0, 2.0, 3.0], **device_cfg_dict)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], **device_cfg_dict)  # Identity rotation
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], **device_cfg_dict)

        transformed = transform_points(position, quaternion, points)

        assert transformed.shape == (2, 3)
        # Identity rotation + translation
        expected = points + position
        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_transform_with_rotation(self, device_cfg_dict):
        """Test transformation with rotation."""
        position = torch.zeros(3, **device_cfg_dict)
        # 90 degree rotation around Z axis: [cos(45°), 0, 0, sin(45°)]
        quaternion = torch.tensor([0.7071, 0.0, 0.0, 0.7071], **device_cfg_dict)
        points = torch.tensor([[1.0, 0.0, 0.0]], **device_cfg_dict)

        transformed = transform_points(position, quaternion, points)

        assert transformed.shape == (1, 3)
        # Point should be rotated
        assert not torch.allclose(transformed, points)

    def test_transform_with_preallocated_outputs(self, device_cfg_dict):
        """Test with preallocated output tensors."""
        position = torch.tensor([1.0, 0.0, 0.0], **device_cfg_dict)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], **device_cfg_dict)
        points = torch.tensor([[0.0, 0.0, 0.0]], **device_cfg_dict)

        # Preallocate outputs
        out_points = torch.zeros(1, 3, **device_cfg_dict)
        out_gp = torch.zeros(1, 3, **device_cfg_dict)
        out_gq = torch.zeros(1, 4, **device_cfg_dict)
        out_gpt = torch.zeros(1, 3, **device_cfg_dict)

        transformed = transform_points(
            position, quaternion, points,
            out_points=out_points,
            out_gp=out_gp,
            out_gq=out_gq,
            out_gpt=out_gpt
        )

        assert transformed.shape == (1, 3)
        # Check that transformation is correct
        assert torch.allclose(transformed, position.unsqueeze(0) + points, atol=1e-5)


class TestBatchTransformPoints:
    """Test batch_transform_points function."""

    def test_basic_batch_transform(self, device_cfg_dict):
        """Test basic batch transformation."""
        batch_size = 5
        num_points = 10

        position = torch.randn(batch_size, 3, **device_cfg_dict)
        quaternion = torch.randn(batch_size, 4, **device_cfg_dict)
        quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)  # Normalize
        points = torch.randn(batch_size, num_points, 3, **device_cfg_dict)

        transformed = batch_transform_points(position, quaternion, points)

        assert transformed.shape == (batch_size, num_points, 3)

    def test_batch_transform_with_outputs(self, device_cfg_dict):
        """Test batch transform with preallocated outputs (lines 140-148)."""
        batch_size = 3
        num_points = 5

        position = torch.zeros(batch_size, 3, **device_cfg_dict)
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch_size, **device_cfg_dict)
        points = torch.randn(batch_size, num_points, 3, **device_cfg_dict)

        # Preallocate
        out_points = torch.zeros(batch_size, num_points, 3, **device_cfg_dict)
        out_gp = torch.zeros(batch_size, num_points, 3, **device_cfg_dict)
        out_gq = torch.zeros(batch_size, num_points, 4, **device_cfg_dict)
        out_gpt = torch.zeros(batch_size, num_points, 3, **device_cfg_dict)

        transformed = batch_transform_points(
            position, quaternion, points,
            out_points=out_points,
            out_gp=out_gp,
            out_gq=out_gq,
            out_gpt=out_gpt
        )

        # Check shape and values
        assert transformed.shape == (batch_size, num_points, 3)
        # Identity transform should preserve points
        assert torch.allclose(transformed, points, atol=1e-5)

    def test_batch_identity_transform(self, device_cfg_dict):
        """Test batch identity transformation."""
        batch_size = 2
        position = torch.zeros(batch_size, 3, **device_cfg_dict)
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch_size, **device_cfg_dict)
        points = torch.randn(batch_size, 5, 3, **device_cfg_dict)

        transformed = batch_transform_points(position, quaternion, points)

        # Identity transform should preserve points
        assert torch.allclose(transformed, points, atol=1e-5)


class TestTransformEdgeCases:
    """Test edge cases in transformations."""

    def test_single_point_transform(self, device_cfg_dict):
        """Test transforming single point."""
        position = torch.tensor([1.0, 2.0, 3.0], **device_cfg_dict)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], **device_cfg_dict)
        points = torch.tensor([[0.0, 0.0, 0.0]], **device_cfg_dict)

        transformed = transform_points(position, quaternion, points)

        assert transformed.shape == (1, 3)

    def test_many_points_transform(self, device_cfg_dict):
        """Test transforming many points."""
        position = torch.zeros(3, **device_cfg_dict)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], **device_cfg_dict)
        points = torch.randn(1000, 3, **device_cfg_dict)

        transformed = transform_points(position, quaternion, points)

        assert transformed.shape == (1000, 3)

    def test_batch_single_point(self, device_cfg_dict):
        """Test batch transform with single point per batch."""
        batch_size = 10
        position = torch.randn(batch_size, 3, **device_cfg_dict)
        quaternion = torch.randn(batch_size, 4, **device_cfg_dict)
        quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
        points = torch.randn(batch_size, 1, 3, **device_cfg_dict)

        transformed = batch_transform_points(position, quaternion, points)

        assert transformed.shape == (batch_size, 1, 3)

