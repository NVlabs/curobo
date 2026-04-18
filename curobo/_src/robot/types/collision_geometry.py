# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RobotCollisionGeometry:
    #: Maps each sphere to the link it belongs to. Shape is [num_spheres]. Dtype is int32.
    link_sphere_idx_map: torch.Tensor

    #: Number of links. Dtype is int32.
    num_links: int

    def clone(self) -> RobotCollisionGeometry:
        return RobotCollisionGeometry(
            link_sphere_idx_map=self.link_sphere_idx_map.clone(),
            num_links=self.num_links,
        )

    def copy_(self, other: RobotCollisionGeometry):
        self.link_sphere_idx_map.copy_(other.link_sphere_idx_map)

    def detach(self) -> RobotCollisionGeometry:
        return RobotCollisionGeometry(
            link_sphere_idx_map=self.link_sphere_idx_map.detach(),
            num_links=self.num_links,
        )
