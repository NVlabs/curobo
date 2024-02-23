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


@torch.jit.script
def project_depth_to_pointcloud(depth_image: torch.Tensor, intrinsics_matrix: torch.Tensor):
    """Projects numpy depth image to point cloud.

    Args:
      np_depth_image: numpy array float, shape (h, w).
        intrinsics array: numpy array float, 3x3 intrinsics matrix.

    Returns:
      array of float (h, w, 3)
    """
    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    cx = intrinsics_matrix[0, 2]
    cy = intrinsics_matrix[1, 2]
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    input_x = torch.arange(width, dtype=torch.float32, device=depth_image.device)
    input_y = torch.arange(height, dtype=torch.float32, device=depth_image.device)
    input_x, input_y = torch.meshgrid(input_x, input_y, indexing="ij")
    input_x, input_y = input_x.T, input_y.T

    input_z = depth_image
    output_x = (input_x * input_z - cx * input_z) / fx
    output_y = (input_y * input_z - cy * input_z) / fy
    raw_pc = torch.stack([output_x, output_y, input_z], -1)

    return raw_pc


@torch.jit.script
def get_projection_rays(height: int, width: int, intrinsics_matrix: torch.Tensor):
    """Projects numpy depth image to point cloud.

    Args:
      np_depth_image: numpy array float, shape (h, w).
        intrinsics array: numpy array float, 3x3 intrinsics matrix.

    Returns:
      array of float (h, w, 3)
    """
    fx = intrinsics_matrix[:, 0, 0]
    fy = intrinsics_matrix[:, 1, 1]
    cx = intrinsics_matrix[:, 0, 2]
    cy = intrinsics_matrix[:, 1, 2]

    input_x = torch.arange(width, dtype=torch.float32, device=intrinsics_matrix.device)
    input_y = torch.arange(height, dtype=torch.float32, device=intrinsics_matrix.device)
    input_x, input_y = torch.meshgrid(input_x, input_y, indexing="ij")

    input_x, input_y = input_x.T, input_y.T

    input_x = input_x.unsqueeze(0).repeat(intrinsics_matrix.shape[0], 1, 1)
    input_y = input_y.unsqueeze(0).repeat(intrinsics_matrix.shape[0], 1, 1)

    input_z = torch.ones(
        (intrinsics_matrix.shape[0], height, width),
        device=intrinsics_matrix.device,
        dtype=torch.float32,
    )

    output_x = (input_x - cx) / fx
    output_y = (input_y - cy) / fy

    rays = torch.stack([output_x, output_y, input_z], -1).reshape(
        intrinsics_matrix.shape[0], width * height, 3
    )
    rays = rays * (1.0 / 1000.0)
    return rays


@torch.jit.script
def project_pointcloud_to_depth(
    pointcloud: torch.Tensor,
    output_image: torch.Tensor,
):
    """Projects pointcloud to depth image

    Args:
      np_depth_image: numpy array float, shape (h, w).
        intrinsics array: numpy array float, 3x3 intrinsics matrix.

    Returns:
      array of float (h, w)
    """
    width, height = output_image.shape

    output_image = output_image.view(-1)
    output_image[:] = pointcloud[:, 2]
    output_image = output_image.view(width, height)
    return output_image


@torch.jit.script
def project_depth_using_rays(
    depth_image: torch.Tensor, rays: torch.Tensor, filter_origin: bool = False
):
    if filter_origin:
        depth_image = torch.where(depth_image < 0.01, 0, depth_image)

    depth_image = depth_image.view(depth_image.shape[0], -1, 1).contiguous()
    points = depth_image * rays
    return points
