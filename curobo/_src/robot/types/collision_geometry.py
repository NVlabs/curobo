# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch

from curobo._src.util.logging import log_and_raise


@dataclass
class RobotCollisionGeometry:
    #: Maps each sphere to the link it belongs to. Shape is [num_spheres]. Dtype is int32.
    link_sphere_idx_map: torch.Tensor

    #: Number of links. Dtype is int32.
    num_links: int

    #: Shared world-collision masks [num_envs, num_spheres], owned by KinematicsParams.
    sphere_collision_mask: torch.Tensor

    def clone(self) -> RobotCollisionGeometry:
        return RobotCollisionGeometry(
            link_sphere_idx_map=self.link_sphere_idx_map,
            num_links=self.num_links,
            sphere_collision_mask=self.sphere_collision_mask,
        )

    def copy_(self, other: RobotCollisionGeometry):
        """Keep model-owned references stable when copying per-query state."""
        if (
            other is None
            or self.link_sphere_idx_map is not other.link_sphere_idx_map
            or self.sphere_collision_mask is not other.sphere_collision_mask
            or self.num_links != other.num_links
        ):
            log_and_raise("cannot copy kinematics states from different collision geometry owners")
        return self

    def detach(self) -> RobotCollisionGeometry:
        return RobotCollisionGeometry(
            link_sphere_idx_map=self.link_sphere_idx_map,
            num_links=self.num_links,
            sphere_collision_mask=self.sphere_collision_mask,
        )
