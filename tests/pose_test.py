#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Third Party
import torch

# CuRobo
from curobo.geom.transform import batch_transform_points, transform_points
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose


def test_pose_transform_point():
    tensor_args = TensorDeviceType()
    new_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0], tensor_args)

    new_pose.position.requires_grad = True
    new_pose.quaternion.requires_grad = True

    points = torch.zeros((3, 3), device=tensor_args.device, dtype=tensor_args.dtype)
    points[:, 0] = 0.1
    points[2, 0] = -0.5

    out_pt = new_pose.transform_point(points)

    loss = torch.sum(out_pt)
    loss.backward()
    assert torch.norm(new_pose.position.grad) > 0.0
    assert torch.norm(new_pose.quaternion.grad) > 0.0


def test_pose_transform_point_grad():
    tensor_args = TensorDeviceType()
    new_pose = Pose.from_list([10.0, 0, 0.1, 1.0, 0, 0, 0], tensor_args)
    new_pose.position.requires_grad = True
    new_pose.quaternion.requires_grad = True

    points = torch.zeros((1, 1, 3), device=tensor_args.device, dtype=tensor_args.dtype) + 10.0

    # buffers:
    out_points = torch.zeros(
        (points.shape[0], points.shape[1], 3), device=points.device, dtype=points.dtype
    )
    out_gp = torch.zeros((new_pose.position.shape[0], 3), device=tensor_args.device)
    out_gq = torch.zeros((new_pose.position.shape[0], 4), device=tensor_args.device)
    out_gpt = torch.zeros((points.shape[0], points.shape[1], 3), device=tensor_args.device)

    torch.autograd.gradcheck(
        batch_transform_points,
        (new_pose.position, new_pose.quaternion, points, out_points, out_gp, out_gq, out_gpt),
        eps=1e-6,
        atol=1.0,
        # nondet_tol=100.0,
    )


# test_pose_transform_point()
# test_pose_transform_point_grad()
