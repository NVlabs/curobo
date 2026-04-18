# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


# Third Party
import numpy as np
import pytest
import torch

# CuRobo
from curobo._src.geom.transform import batch_transform_points
from curobo._src.types.pose import Pose


def get_test_pose(cpu_cuda_device_cfg):
    position = torch.tensor([1.0, 2.0, 3.0], device=cpu_cuda_device_cfg.device)
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=cpu_cuda_device_cfg.device)  # Identity rotation
    pose = Pose(position, quaternion)

    return pose


def test_initialization(cpu_cuda_device_cfg):
    """Test pose initialization"""
    test_pose = get_test_pose(cpu_cuda_device_cfg)
    assert test_pose.position.shape == torch.tensor([[1.0, 2.0, 3.0]]).shape
    assert test_pose.quaternion.shape == torch.tensor([[1.0, 0.0, 0.0, 0.0]]).shape


def test_get_matrix(cpu_cuda_device_cfg):
    """Test conversion to transformation matrix"""
    test_pose = get_test_pose(cpu_cuda_device_cfg)

    matrix = test_pose.get_matrix()
    expected_shape = torch.tensor(np.eye(4), device=test_pose.position.device).unsqueeze(0).shape
    assert matrix.shape == expected_shape

    # Verify position in translation part of matrix
    assert torch.allclose(
        matrix[0, :3, 3], torch.tensor([1.0, 2.0, 3.0], device=test_pose.position.device)
    )

    # Verify identity rotation
    assert torch.allclose(matrix[0, :3, :3], torch.eye(3, device=test_pose.position.device))


def test_pose_inverse(cpu_cuda_device_cfg):
    """Test pose inversion"""
    test_pose = get_test_pose(cpu_cuda_device_cfg)

    inverted_pose = test_pose.inverse()
    inverted_matrix = inverted_pose.get_matrix()
    original_matrix = test_pose.get_matrix()

    identity = torch.bmm(original_matrix, inverted_matrix)
    identity = identity.view(4, 4)

    assert torch.allclose(identity, torch.eye(4, device=test_pose.position.device))


def test_pose_multiplication(cpu_cuda_device_cfg):
    """Test pose multiplication"""
    # Create another pose
    test_pose = get_test_pose(cpu_cuda_device_cfg)

    position2 = torch.tensor([4.0, 5.0, 6.0], device=test_pose.position.device)
    quaternion2 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=test_pose.position.device)
    pose2 = Pose(position=position2, quaternion=quaternion2)

    # Multiply poses
    multiplied_pose = test_pose.multiply(pose2)
    multiplied_matrix = multiplied_pose.get_matrix()

    expected_matrix = torch.tensor(
        [[1, 0, 0, 5], [0, 1, 0, 7], [0, 0, 1, 9], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=test_pose.position.device,
    )

    assert torch.allclose(multiplied_matrix, expected_matrix.view(-1, 4, 4))


def test_transform_points(cpu_cuda_device_cfg):
    """Test point transformation"""
    test_pose = get_test_pose(cpu_cuda_device_cfg)

    points = torch.tensor(
        [[0.0, 0.0, 0.0]], device=test_pose.position.device
    )  # Test transform origin
    transformed = test_pose.transform_points(points)

    # Should be same as position since origin is transformed
    assert torch.allclose(transformed, test_pose.position)


def test_angular_distance(cpu_cuda_device_cfg):
    """Test angular distance calculation"""
    test_pose = get_test_pose(cpu_cuda_device_cfg)

    # Create another identity pose
    pose2 = Pose(test_pose.position, test_pose.quaternion)
    distance = test_pose.angular_distance(pose2)

    # Angular distance should be zero for identical poses
    assert float(distance) == pytest.approx(0.0, abs=1e-5)


def test_pose_transform_point(cpu_cuda_device_cfg):
    new_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0], cpu_cuda_device_cfg)

    new_pose.position.requires_grad = True
    new_pose.quaternion.requires_grad = True

    points = torch.zeros((3, 3), device=cpu_cuda_device_cfg.device, dtype=cpu_cuda_device_cfg.dtype)
    points[:, 0] = 0.1
    points[2, 0] = -0.5

    out_pt = new_pose.transform_point(points)

    loss = torch.mean(out_pt)
    loss.backward()
    assert torch.norm(new_pose.position.grad) > 0.0
    assert torch.norm(new_pose.quaternion.grad) > 0.0


def test_pose_transform_point_grad(cpu_cuda_device_cfg):
    new_pose = Pose.from_list([10.0, 0, 0.1, 1.0, 0, 0, 0], cpu_cuda_device_cfg)
    new_pose.position.requires_grad = True
    new_pose.quaternion.requires_grad = True

    points = torch.zeros((1, 1, 3), device=cpu_cuda_device_cfg.device, dtype=cpu_cuda_device_cfg.dtype) + 10.0

    # buffers:
    out_points = torch.zeros(
        (points.shape[0], points.shape[1], 3), device=points.device, dtype=points.dtype
    )
    out_gp = torch.zeros((new_pose.position.shape[0], 3), device=cpu_cuda_device_cfg.device)
    out_gq = torch.zeros((new_pose.position.shape[0], 4), device=cpu_cuda_device_cfg.device)
    out_gpt = torch.zeros((points.shape[0], points.shape[1], 3), device=cpu_cuda_device_cfg.device)

    torch.autograd.gradcheck(
        batch_transform_points,
        (new_pose.position, new_pose.quaternion, points, out_points, out_gp, out_gq, out_gpt),
        eps=1e-6,
        atol=1.0,
        # nondet_tol=100.0,
    )
