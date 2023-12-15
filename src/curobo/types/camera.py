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
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional

# Third Party
import torch

# CuRobo
from curobo.types.math import Pose
from curobo.util.logger import log_error, log_warn


@dataclass
class CameraObservation:
    name: str = "camera_image"
    #: rgb image format is BxHxWxchannels
    rgb_image: Optional[torch.Tensor] = None
    depth_image: Optional[torch.Tensor] = None
    image_segmentation: Optional[torch.Tensor] = None
    projection_matrix: Optional[torch.Tensor] = None
    resolution: Optional[List[int]] = None
    pose: Optional[Pose] = None
    intrinsics: Optional[torch.Tensor] = None
    timestamp: float = 0.0

    @property
    def shape(self):
        return self.rgb_image.shape

    def copy_(self, new_data: CameraObservation):
        self.rgb_image.copy_(new_data.rgb_image)
        self.depth_image.copy_(new_data.depth_image)
        self.image_segmentation.copy_(new_data.image_segmentation)
        self.projection_matrix.copy_(new_data.projection_matrix)
        self.resolution = new_data.resolution

    def to(self, device: torch.device):
        if self.rgb_image is not None:
            self.rgb_image = self.rgb_image.to(device=device)
        if self.depth_image is not None:
            self.depth_image = self.depth_image.to(device=device)
        return self
