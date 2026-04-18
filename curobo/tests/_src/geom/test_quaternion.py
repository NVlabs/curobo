# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import pytest
import torch
from torch.autograd import gradcheck

# CuRobo
from curobo._src.geom.quaternion import (
    angular_distance_axis_angle,
    angular_distance_phi3,
    normalize_quaternion,
    quat_multiply,
)


@pytest.fixture(autouse=True)
def disable_donated_buffers_for_gradcheck():
    """Disable donated buffers for gradcheck compatibility with torch.compile.

    torch.compile may use donated buffers optimization which is incompatible with
    gradcheck (which requires create_graph=True internally).
    """
    import torch._functorch.config as functorch_config

    with functorch_config.patch(donated_buffer=False):
        yield


pytestmark = pytest.mark.parametrize(
    "device",
    ["cuda", "cpu"],
)


def test_quat_multiply(device):
    """Test quaternion multiplication"""
    # Create two test quaternions (identity quaternions)
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    q2 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    q_res = torch.zeros_like(q1)

    # Test multiplication
    result = quat_multiply(q1, q2, q_res)

    # Result should be identity quaternion
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    assert torch.allclose(result, expected)


def test_angular_distance_phi3(device):
    """Test angular distance calculation"""
    # Create two identical quaternions
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    q2 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    # Distance between identical quaternions should be 0
    distance = angular_distance_phi3(q1, q2)
    assert float(distance) == pytest.approx(0.0, abs=1e-5)

    # Test with 90-degree rotation around X axis
    q3 = torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=device)
    distance = angular_distance_phi3(q1, q3)
    assert float(distance) == pytest.approx(0.5, abs=1e-3)  # Should be 0.5 for 90-degree rotation


def test_normalize_quaternion(device):
    """Test quaternion normalization"""
    # Test with non-normalized quaternion
    q = torch.tensor([2.0, 1.0, 1.0, 1.0], device=device)
    normalized = normalize_quaternion(q)

    # Check if normalized quaternion has unit length
    norm = torch.linalg.norm(normalized)
    assert float(norm) == pytest.approx(1.0, abs=1e-5)


def test_batch_operations(device):
    """Test quaternion operations with batched inputs"""
    # Create batch of quaternions
    q_batch1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]], device=device)
    q_batch2 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)

    # Test batch normalization
    normalized = normalize_quaternion(q_batch1)
    norms = torch.linalg.norm(normalized, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms))

    # Test batch angular distance
    distances = angular_distance_phi3(q_batch1, q_batch2)
    assert distances.shape[0] == 2
    assert float(distances[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(distances[1]) == pytest.approx(0.5, abs=1e-3)


def test_angular_distance_gradcheck(device):
    """Test gradient accuracy for angular_distance_phi3 using gradcheck."""
    # Create test quaternions with double precision
    q1 = torch.tensor([[1.0, 0.1, 0.2, 0.3]], device=device, dtype=torch.double, requires_grad=True)
    q2 = torch.tensor([[0.9, 0.2, 0.1, 0.4]], device=device, dtype=torch.double, requires_grad=True)

    # Normalize inputs
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    # Verify gradient computation
    assert gradcheck(angular_distance_phi3, (q1, q2), eps=1e-6, atol=1e-4)


def test_quat_multiply_gradcheck(device):
    """Test gradient accuracy for quaternion multiplication using gradcheck."""
    # Create test quaternions with double precision
    q1 = torch.tensor([[1.0, 0.1, 0.2, 0.3]], device=device, dtype=torch.double, requires_grad=True)
    q2 = torch.tensor([[0.9, 0.2, 0.1, 0.4]], device=device, dtype=torch.double, requires_grad=True)

    # Normalize inputs
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    def quat_multiply_wrapper(q1, q2):
        q_res = torch.zeros_like(q1)
        return quat_multiply(q1, q2, q_res)

    # Verify gradient computation
    assert gradcheck(quat_multiply_wrapper, (q1, q2), eps=1e-6, atol=1e-4)


def test_normalize_quaternion_gradcheck(device):
    """Test gradient accuracy for quaternion normalization using gradcheck."""
    # Create a test quaternion with double precision
    q = torch.tensor([[2.0, 1.0, 1.0, 1.0]], device=device, dtype=torch.double, requires_grad=True)

    # Verify gradient computation
    assert gradcheck(normalize_quaternion, (q,), eps=1e-6, atol=1e-4)


def test_axis_angle_error(device):
    """Test axis-angle error calculation"""
    # Create two identical quaternions (identity quaternion)
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    q2 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    # Error between identical quaternions should be 0
    error = angular_distance_axis_angle(q1, q2)
    assert float(error) == pytest.approx(0.0, abs=1e-5)

    # Test with 90-degree rotation around X axis
    q3 = torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=device)
    error = angular_distance_axis_angle(q1, q3)
    assert float(error) == pytest.approx(1.5708, abs=1e-4)  # pi/2 (90 degrees)

    # Test with 180-degree rotation around Z axis
    q4 = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    error = angular_distance_axis_angle(q1, q4)
    assert float(error) == pytest.approx(3.1416, abs=1e-4)  # pi (180 degrees)


def test_axis_angle_error_gradcheck(device):
    """Test gradient accuracy for axis_angle_error using gradcheck."""
    # Create test quaternions with double precision for better numerical accuracy
    q1 = torch.tensor([[1.0, 0.1, 0.2, 0.3]], device=device, dtype=torch.double, requires_grad=True)
    q2 = torch.tensor([[0.9, 0.2, 0.1, 0.4]], device=device, dtype=torch.double, requires_grad=True)

    # Normalize inputs to ensure they're valid quaternions
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    # Verify gradient computation
    assert gradcheck(angular_distance_axis_angle, (q1, q2), eps=1e-6, atol=1e-4)
