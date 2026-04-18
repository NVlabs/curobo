# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional

# Third Party
import torch
from torch.profiler import record_function

# CuRobo
from curobo._src.geom.cv import (
    extract_depth_from_structured_pointcloud,
    get_projection_rays,
    project_depth_using_rays,
)
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise


@dataclass
class CameraObservation:
    name: str = "camera_image"
    #: rgb image format is BxHxWxchannels
    rgb_image: Optional[torch.Tensor] = None
    depth_image: Optional[torch.Tensor] = None
    image_segmentation: Optional[torch.Tensor] = None
    projection_matrix: Optional[torch.Tensor] = None
    projection_rays: Optional[torch.Tensor] = None
    resolution: Optional[List[int]] = None
    pose: Optional[Pose] = None
    #: intrinsics_matrix: Batch of intrinsics matrices of shape (b, 3, 3).
    #: contains the following fields:
    #: [[fx, 0, cx],
    #: [0, fy, cy],
    #: [0, 0, 1]]
    intrinsics: Optional[torch.Tensor] = None
    timestamp: Optional[torch.Tensor] = None
    depth_to_meter: float = 0.001

    def filter_depth(self, distance: float = 0.01):
        if self.depth_image is None:
            log_and_raise("depth_image is None, cannot filter depth")
        self.depth_image = torch.where(self.depth_image < distance, 0, self.depth_image)

    @property
    def shape(self):
        if self.rgb_image is None:
            log_and_raise("rgb_image is None, cannot get shape")
        return self.rgb_image.shape

    @record_function("camera/copy_")
    def copy_(self, new_data: CameraObservation):
        if self.rgb_image is not None:
            self.rgb_image.copy_(new_data.rgb_image)
        if self.depth_image is not None:
            self.depth_image.copy_(new_data.depth_image)
        if self.image_segmentation is not None:
            self.image_segmentation.copy_(new_data.image_segmentation)
        if self.projection_matrix is not None:
            self.projection_matrix.copy_(new_data.projection_matrix)
        if self.projection_rays is not None:
            self.projection_rays.copy_(new_data.projection_rays)
        if self.pose is not None:
            self.pose.copy_(new_data.pose)
        if self.timestamp is not None:
            self.timestamp.copy_(new_data.timestamp)
        self.depth_to_meter = new_data.depth_to_meter
        self.resolution = new_data.resolution

    @record_function("camera/clone")
    def clone(self):
        return CameraObservation(
            depth_image=self.depth_image.clone() if self.depth_image is not None else None,
            rgb_image=self.rgb_image.clone() if self.rgb_image is not None else None,
            intrinsics=self.intrinsics.clone() if self.intrinsics is not None else None,
            resolution=self.resolution,
            pose=self.pose.clone() if self.pose is not None else None,
            timestamp=self.timestamp.clone() if self.timestamp is not None else None,
            image_segmentation=(
                self.image_segmentation.clone() if self.image_segmentation is not None else None
            ),
            projection_matrix=(
                self.projection_matrix.clone() if self.projection_matrix is not None else None
            ),
            projection_rays=(
                self.projection_rays.clone() if self.projection_rays is not None else None
            ),
            name=self.name,
            depth_to_meter=self.depth_to_meter,
        )

    def to(self, device: torch.device):
        self.rgb_image = self.rgb_image.to(device=device) if self.rgb_image is not None else None
        self.depth_image = (
            self.depth_image.to(device=device) if self.depth_image is not None else None
        )

        self.image_segmentation = (
            self.image_segmentation.to(device=device)
            if self.image_segmentation is not None
            else None
        )
        self.projection_matrix = (
            self.projection_matrix.to(device=device) if self.projection_matrix is not None else None
        )
        self.projection_rays = (
            self.projection_rays.to(device=device) if self.projection_rays is not None else None
        )
        self.intrinsics = self.intrinsics.to(device=device) if self.intrinsics is not None else None
        self.timestamp = self.timestamp.to(device=device) if self.timestamp is not None else None
        self.pose = self.pose.to(device=device) if self.pose is not None else None

        return self

    def get_pointcloud(self, project_to_pose: bool = False):
        if self.depth_image is None:
            log_and_raise("depth_image is None, cannot generate pointcloud")
        if self.projection_rays is None:
            self.update_projection_rays()
        depth_image = self.depth_image
        if len(self.depth_image.shape) == 2:
            depth_image = self.depth_image.unsqueeze(0)
        point_cloud = project_depth_using_rays(depth_image, self.projection_rays)

        if project_to_pose and self.pose is not None:
            point_cloud = self.pose.batch_transform_points(point_cloud)

        return point_cloud

    def extract_depth_from_structured_pointcloud(
        self, pointcloud, output_image: Optional[torch.Tensor] = None
    ):
        """Extract depth image from structured pointcloud.

        This function assumes the pointcloud maintains the spatial grid structure
        [batch, height, width, 3] where Z-axis represents depth values.

        **Important:** This only works for structured pointclouds in camera frame where
        Z-axis is aligned with depth. If the pointcloud has been transformed to world
        frame (e.g., via self.pose), this will NOT work correctly.

        Args:
            pointcloud: Structured pointcloud in camera frame. Shape: [h, w, 3] or [b, h, w, 3].
            output_image: Optional pre-allocated output tensor. Shape: [h, w] or [b, h, w].

        Returns:
            Depth image extracted from Z-coordinates. Shape: [b, h, w].
        """
        # Ensure pointcloud has batch dimension
        if len(pointcloud.shape) == 3:
            # (h, w, 3) -> (1, h, w, 3)
            pointcloud = pointcloud.unsqueeze(0)

        if output_image is None:
            # Extract shape from pointcloud: [b, h, w, 3] -> [b, h, w]
            batch_size, height, width = pointcloud.shape[0], pointcloud.shape[1], pointcloud.shape[2]
            output_image = torch.zeros(
                (batch_size, height, width),
                dtype=pointcloud.dtype,
                device=pointcloud.device,
            )
        elif len(output_image.shape) == 2:
            # Add batch dimension if not present
            output_image = output_image.unsqueeze(0)

        depth_image = extract_depth_from_structured_pointcloud(
            pointcloud, output_image=output_image
        )
        return depth_image

    def update_projection_rays(self):
        if self.depth_image is None:
            log_and_raise("depth_image is None, cannot update projection rays")
        if self.intrinsics is None:
            log_and_raise("intrinsics is None, cannot update projection rays")
        intrinsics = self.intrinsics
        if len(intrinsics.shape) == 2:
            intrinsics = intrinsics.unsqueeze(0)
        project_rays = get_projection_rays(
            self.depth_image.shape[-2], self.depth_image.shape[-1], intrinsics, depth_to_meter=self.depth_to_meter
        )

        if self.projection_rays is None:
            self.projection_rays = project_rays

        self.projection_rays.copy_(project_rays)

    def stack(self, new_observation: CameraObservation, dim: int = 0):
        rgb_image = (
            torch.stack((self.rgb_image, new_observation.rgb_image), dim=dim)
            if self.rgb_image is not None
            else None
        )
        depth_image = (
            torch.stack((self.depth_image, new_observation.depth_image), dim=dim)
            if self.depth_image is not None
            else None
        )
        image_segmentation = (
            torch.stack((self.image_segmentation, new_observation.image_segmentation), dim=dim)
            if self.image_segmentation is not None
            else None
        )
        projection_matrix = (
            torch.stack((self.projection_matrix, new_observation.projection_matrix), dim=dim)
            if self.projection_matrix is not None
            else None
        )
        projection_rays = (
            torch.stack((self.projection_rays, new_observation.projection_rays), dim=dim)
            if self.projection_rays is not None
            else None
        )
        resolution = self.resolution

        pose = self.pose.stack(new_observation.pose) if self.pose is not None else None

        intrinsics = (
            torch.stack((self.intrinsics, new_observation.intrinsics), dim=dim)
            if self.intrinsics is not None
            else None
        )
        timestamp = (
            torch.stack((self.timestamp, new_observation.timestamp), dim=dim)
            if self.timestamp is not None
            else None
        )
        return CameraObservation(
            name=self.name,
            rgb_image=rgb_image,
            depth_image=depth_image,
            image_segmentation=image_segmentation,
            projection_matrix=projection_matrix,
            projection_rays=projection_rays,
            resolution=resolution,
            pose=pose,
            intrinsics=intrinsics,
            timestamp=timestamp,
        )
    def save_to_file(self, file_path: str):
        data = {
            "rgb_image": self.rgb_image,
            "depth_image": self.depth_image,
            "intrinsics": self.intrinsics,
            "pose": self.pose.tolist(),
            "timestamp": self.timestamp,
            "depth_to_meter": self.depth_to_meter,
        }
        torch.save(data, file_path)
