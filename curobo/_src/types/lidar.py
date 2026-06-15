# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.profiler import record_function

from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise


@dataclass
class LidarObservation:
    """Structured LiDAR range-image observation.

    The mapper expects range, RGB, and optional feature tensors to already be
    registered in the LiDAR pixel frame. Range values are Euclidean distance
    from the LiDAR origin, not camera z-depth.
    """

    name: str = "lidar_range_image"
    #: Range image in meters. Shape ``(num_lidars, H, W)`` float32.
    range_image: Optional[torch.Tensor] = None
    #: RGB image aligned to ``range_image``. Shape ``(num_lidars, H, W, 3)`` uint8.
    rgb_image: Optional[torch.Tensor] = None
    #: Optional neural feature grid registered to the LiDAR image frame. Shape
    #: ``(num_lidars, feature_H, feature_W, feature_dim)`` float16,
    #: channels-last with stride 1 on the channel dim.
    feature_grid: Optional[torch.Tensor] = None
    #: LiDAR poses in the map/world frame, leading dimension ``num_lidars``.
    pose: Optional[Pose] = None
    #: Per-LiDAR valid range bounds in meters. Shape ``(num_lidars, 2)`` with
    #: columns ``[min_m, max_m]``.
    valid_range_m: Optional[torch.Tensor] = None
    #: Per-LiDAR elevation bounds in radians. Shape ``(num_lidars, 2)`` with
    #: columns ``[min_elevation_rad, max_elevation_rad]``. For planar scans
    #: with ``H == 1``, both entries must be equal.
    elevation_range_rad: Optional[torch.Tensor] = None
    timestamp: Optional[torch.Tensor] = None

    @property
    def shape(self):
        if self.range_image is None:
            log_and_raise("range_image is None, cannot get shape")
        return self.range_image.shape

    @record_function("lidar/copy_")
    def copy_(self, new_data: LidarObservation):
        if self.range_image is not None:
            self.range_image.copy_(new_data.range_image)
        if self.rgb_image is not None and new_data.rgb_image is not None:
            self.rgb_image.copy_(new_data.rgb_image)
        if self.feature_grid is not None and new_data.feature_grid is not None:
            self.feature_grid.copy_(new_data.feature_grid)
        if self.pose is not None:
            self.pose.copy_(new_data.pose)
        if self.valid_range_m is not None:
            self.valid_range_m.copy_(new_data.valid_range_m)
        if self.elevation_range_rad is not None:
            self.elevation_range_rad.copy_(new_data.elevation_range_rad)
        if self.timestamp is not None and new_data.timestamp is not None:
            self.timestamp.copy_(new_data.timestamp)
        return self

    @record_function("lidar/clone")
    def clone(self):
        return LidarObservation(
            name=self.name,
            range_image=self.range_image.clone() if self.range_image is not None else None,
            rgb_image=self.rgb_image.clone() if self.rgb_image is not None else None,
            feature_grid=(
                self.feature_grid.clone() if self.feature_grid is not None else None
            ),
            pose=self.pose.clone() if self.pose is not None else None,
            valid_range_m=(
                self.valid_range_m.clone() if self.valid_range_m is not None else None
            ),
            elevation_range_rad=(
                self.elevation_range_rad.clone()
                if self.elevation_range_rad is not None
                else None
            ),
            timestamp=self.timestamp.clone() if self.timestamp is not None else None,
        )

    def to(self, device: torch.device):
        self.range_image = (
            self.range_image.to(device=device) if self.range_image is not None else None
        )
        self.rgb_image = self.rgb_image.to(device=device) if self.rgb_image is not None else None
        self.feature_grid = (
            self.feature_grid.to(device=device) if self.feature_grid is not None else None
        )
        self.pose = self.pose.to(device=device) if self.pose is not None else None
        self.valid_range_m = (
            self.valid_range_m.to(device=device) if self.valid_range_m is not None else None
        )
        self.elevation_range_rad = (
            self.elevation_range_rad.to(device=device)
            if self.elevation_range_rad is not None
            else None
        )
        self.timestamp = (
            self.timestamp.to(device=device) if self.timestamp is not None else None
        )
        return self
