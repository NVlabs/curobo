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
"""Computer Vision functions, including projection between depth and pointclouds."""

# Third Party
import torch

# CuRobo
from curobo.util.torch_utils import get_torch_jit_decorator


@get_torch_jit_decorator()
def project_depth_to_pointcloud(
    depth_image: torch.Tensor,
    intrinsics_matrix: torch.Tensor,
) -> torch.Tensor:
    """Projects depth image to point cloud.

    Args:
      depth_image: torch tensor of shape (b, h, w).
        intrinsics array: torch tensor for intrinsics matrix of shape (b, 3, 3).

    Returns:
      torch tensor of shape (b, h, w, 3)
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


@get_torch_jit_decorator()
def get_projection_rays(
    height: int,
    width: int,
    intrinsics_matrix: torch.Tensor,
    depth_to_meter: float = 0.001,
) -> torch.Tensor:
    """Get projection rays for a image size and batch of intrinsics matrices.

    Args:
        height: Height of the images.
        width: Width of the images.
        intrinsics_matrix: Batch of intrinsics matrices of shape (b, 3, 3).
        depth_to_meter: Scaling factor to convert depth to meters.

    Returns:
        torch.Tensor: Projection rays of shape (b, height * width, 3).
    """
    fx = intrinsics_matrix[:, 0:1, 0:1]
    fy = intrinsics_matrix[:, 1:2, 1:2]
    cx = intrinsics_matrix[:, 0:1, 2:3]
    cy = intrinsics_matrix[:, 1:2, 2:3]

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
    rays = rays * depth_to_meter
    return rays


@get_torch_jit_decorator()
def project_pointcloud_to_depth(
    pointcloud: torch.Tensor,
    output_image: torch.Tensor,
) -> torch.Tensor:
    """Projects pointcloud to depth image based on indices.

    Args:
        pointcloud: PointCloud of shape (b, h, w, 3).
        output_image: Image of shape (b, h, w).

    Returns:
        torch.Tensor: Depth image of shape (b, h, w).
    """
    width, height = output_image.shape

    output_image = output_image.view(-1)
    output_image[:] = pointcloud[:, 2]
    output_image = output_image.view(width, height)
    return output_image


@get_torch_jit_decorator()
def project_depth_using_rays(
    depth_image: torch.Tensor,
    rays: torch.Tensor,
    filter_origin: bool = False,
    depth_threshold: float = 0.01,
) -> torch.Tensor:
    """Project depth image to pointcloud using projection rays.

    Projection rays can be calculated using :func:`~curobo.geom.cv.get_projection_rays` function.

    Args:
        depth_image: Dpepth image of shape (b, h, w).
        rays: Projection rays of shape (b, h * w, 3).
        filter_origin: Remove points with depth less than depth_threshold.
        depth_threshold: Threshold to filter points.

    Returns:
        Pointcloud of shape (b, h * w, 3).
    """
    if filter_origin:
        depth_image = torch.where(depth_image < depth_threshold, 0, depth_image)

    depth_image = depth_image.view(depth_image.shape[0], -1, 1).contiguous()
    points = depth_image * rays
    return points
