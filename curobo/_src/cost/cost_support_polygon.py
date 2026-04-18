# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from curobo._src.cost.cost_support_polygon_cfg import CostSupportPolygonCfg

from curobo._src.cost.cost_base import BaseCost
from curobo._src.geom.convex_polygon_helper import ConvexPolygon2DHelper
from curobo._src.util.logging import log_and_raise


class CostSupportPolygon(BaseCost):
    def __init__(self, config: CostSupportPolygonCfg):
        super().__init__(config)
        # Initialize the convex polygon helper
        self._polygon_helper = ConvexPolygon2DHelper(config.device_cfg)

    def build_convex_hull(self, vertices: torch.Tensor, padding: Optional[float] = None):
        """Build a convex hull from a set of vertices.

        Args:
            vertices: The vertices of the convex hull. Shape is [batch, n_vertices, 2]
            padding: Optional padding to apply to the convex hull

        Returns:
            The convex hull. Shape is [n_vertices, 2]
        """
        self._polygon_helper.build_convex_hull(vertices, padding)

    def forward(self, robot_com: torch.Tensor, robot_spheres: torch.Tensor) -> torch.Tensor:
        """Compute the support polygon from the robot_spheres.

        Args:
            robot_com: The center of mass of the robot. Shape is [batch_size, horizon, 3]
            robot_spheres: The spheres of the robot. Shape is [batch_size, horizon, num_spheres, 4]

        Returns:
            The cost of the support polygon. Shape is [batch_size, horizon]
        """
        foot_spheres = robot_spheres[
            :, :, self.config.foot_sphere_indices, :2
        ].detach()  # only use x, y

        if self._polygon_helper._cached_convex_hulls is None:
            self.build_convex_hull(foot_spheres[:, 0, :, :], padding=0.05)

        # Project CoM to 2D (x, y plane)
        com_2d = robot_com[:, :, :2]  # Shape: [batch_size, horizon, 2]

        # Compute support polygon cost vectorized across all batch and horizon
        cost = self._compute_support_polygon_cost_vectorized(com_2d)

        # Apply cost weight
        cost = cost * self._weight.squeeze()

        return cost

    def _compute_support_polygon_cost_vectorized(self, com_pos: torch.Tensor) -> torch.Tensor:
        """Vectorized computation of support polygon cost using cached convex hull.

        Args:
            com_pos: Center of mass positions. Shape: [batch_size, horizon, 2]

        Returns:
            Cost values. Shape: [batch_size, horizon]
        """
        batch_size, horizon, _ = com_pos.shape

        # Get convex hull (computed once and cached)
        if self._polygon_helper._cached_convex_hulls is None:
            log_and_raise("No convex hull cached, call build_convex_hull first")
            return torch.zeros(batch_size, horizon, device=com_pos.device, dtype=com_pos.dtype)

        # Create batch indices - all points use the same hulls
        batch_indices = torch.arange(batch_size, device=com_pos.device)

        # Reshape CoM for batch processing: [batch_size, horizon, 1, 2]
        com_reshaped = com_pos.unsqueeze(2)

        # Compute signed distances using helper (negative inside, positive outside)
        signed_distances = self._polygon_helper.compute_point_hull_distance(
            com_reshaped, batch_indices
        )
        distances = signed_distances.squeeze(-1)  # Remove the single point dimension

        # Apply cost function
        inside_cost_weight = getattr(self.config, "inside_cost_weight", 0.0)
        if inside_cost_weight > 0:
            # Define desired margin distance
            margin_target = 0.1  # 10cm desired margin when inside

            # When outside: use full distance cost
            # When inside: linear cost that decreases as we get further from boundary
            is_inside = distances < 0
            inside_cost = inside_cost_weight * torch.clamp(
                margin_target + distances, min=0.0
            )  # distances is negative inside

            cost = torch.where(is_inside, inside_cost, distances)
        else:
            # Original behavior: no cost when inside
            cost = torch.clamp(distances, min=0.0)  # Only positive distances (outside hull)

        return cost
