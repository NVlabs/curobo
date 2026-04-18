# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Optional

import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_warn


class ConvexPolygon2DHelper:
    def __init__(self, device_cfg: DeviceCfg = DeviceCfg()):
        self._cached_convex_hulls = None
        self._tensor_args = device_cfg

    def build_convex_hull(self, vertices: torch.Tensor, padding: Optional[float] = None):
        """Build a convex hull from a set of vertices.

        Args:
            vertices: The vertices of the convex hull. Shape is [batch, n_vertices, 2]
            padding: Optional padding to apply to the convex hull

        Returns:
            The convex hull vertices. Cached internally for later use.
        """
        self._cached_convex_hulls = self._compute_convex_hull_vectorized(vertices.detach(), padding)

    def compute_point_hull_distance(
        self, points: torch.Tensor, idxs_batch: torch.Tensor
    ) -> torch.Tensor:
        """Compute the distance from a set of points to a set of convex hulls.

        Args:
            points: The points to compute the distance to. Shape is [batch, horizon, n_points, 2]
            idxs_batch: The indices of the batch to compute the distance. Shape is [batch]. For
            each index in points (b_idx, : ,:), we compute with respect to the hull in
            self._cached_convex_hulls[idxs_batch[b_idx]]

        Returns:
            The distance from the points to the hulls. Shape is [batch, horizon, n_points]
        """
        # positive distance outside the hull, negative distance inside the hull
        if self._cached_convex_hulls is None:
            log_warn("No convex hull cached, call build_convex_hull first")
            return torch.zeros_like(points[..., 0])

        batch_size, horizon, n_points, _ = points.shape

        # Get the appropriate hulls based on idxs_batch
        selected_hulls = self._cached_convex_hulls[idxs_batch]  # [batch, n_vertices, 2]

        # Use efficient vectorized computation without massive tensor expansion
        is_inside = self._point_in_convex_hull_optimized(points, selected_hulls)
        distances = self._distance_to_convex_hull_optimized(points, selected_hulls)

        # Apply sign convention: negative inside, positive outside
        signed_distances = torch.where(is_inside, -distances, distances)

        return signed_distances

    def _compute_convex_hull_vectorized(
        self, vertices: torch.Tensor, padding: Optional[float] = None
    ) -> torch.Tensor:
        """Compute convex hull for all batches using vectorized operations.
        Assumes that vertices are the same shape for all batches.

        Args:
            vertices: Vertex positions. Shape: [batch_size, n_pts, 2]
            padding: Optional padding radius to expand the hull

        Returns:
            Convex hull vertices. Shape: [batch_size, max_hull_vertices, 2]
        """
        batch_size, n_pts, _ = vertices.shape
        device = vertices.device
        dtype = vertices.dtype

        # Handle degenerate cases
        if n_pts < 3:
            log_warn("Not enough points to form a stable polygon")
            return vertices

        # Compute convex hull for each batch
        hulls = []
        max_hull_size = 0

        for i in range(batch_size):
            hull = self._compute_convex_hull_single(vertices[i], padding)
            hulls.append(hull)
            max_hull_size = max(max_hull_size, hull.shape[0])

        # Stack and pad hulls
        padded_hulls = torch.zeros(batch_size, max_hull_size, 2, device=device, dtype=dtype)

        for i, hull in enumerate(hulls):
            n_vertices = hull.shape[0]
            padded_hulls[i, :n_vertices] = hull
            # Pad with last vertex if needed
            if n_vertices < max_hull_size:
                padded_hulls[i, n_vertices:] = hull[-1].expand(max_hull_size - n_vertices, -1)

        return padded_hulls

    def _compute_convex_hull_single(
        self, points: torch.Tensor, padding: Optional[float] = None
    ) -> torch.Tensor:
        """Compute convex hull for a single set of points using pure PyTorch (no numpy).

        Args:
            points: Points. Shape: [n_points, 2]
            padding: Optional padding radius to expand the hull

        Returns:
            Convex hull vertices. Shape: [n_hull_vertices, 2]
        """
        n_points = points.shape[0]

        if n_points < 3:
            return points

        # Find the bottom-most point (lowest y, then leftmost x)
        y_coords = points[:, 1]
        x_coords = points[:, 0]

        # Find minimum y
        min_y = torch.min(y_coords)
        y_mask = torch.abs(y_coords - min_y) < 1e-8

        # Among points with minimum y, find minimum x
        x_coords_at_min_y = torch.where(y_mask, x_coords, torch.inf)
        min_x_idx = torch.argmin(x_coords_at_min_y)

        start_point = points[min_x_idx]

        # Remove start point and sort others by polar angle
        other_indices = torch.arange(n_points, device=points.device)
        other_mask = other_indices != min_x_idx
        other_points = points[other_mask]

        if other_points.shape[0] == 0:
            return points[:1]  # Only one point

        # Compute polar angles relative to start point
        vectors = other_points - start_point.unsqueeze(0)
        angles = torch.atan2(vectors[:, 1], vectors[:, 0])
        distances = torch.norm(vectors, dim=1)

        # Sort by angle, then by distance for tie-breaking
        sort_keys = angles * 1e6 + distances  # Scale angles to dominate
        sorted_indices = torch.argsort(sort_keys)
        sorted_points = other_points[sorted_indices]

        # Graham scan using PyTorch operations
        hull_list = [start_point]

        for point in sorted_points:
            # Remove points that would create right turns
            while len(hull_list) >= 2:
                p1 = hull_list[-2]
                p2 = hull_list[-1]
                p3 = point

                # Cross product to determine turn direction
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

                if cross <= 1e-10:  # Right turn or collinear
                    hull_list.pop()
                else:
                    break

            hull_list.append(point)

        # Apply padding if specified and we have more than one point
        if padding is not None and padding > 0 and len(hull_list) > 1:
            hull_tensor = torch.stack(hull_list)
            hull_list = self._apply_padding_to_hull(hull_tensor, padding)

        # Convert list back to tensor
        if len(hull_list) == 1:
            return hull_list[0].unsqueeze(0)
        else:
            return torch.stack(hull_list)

    def _apply_padding_to_hull(self, hull_tensor: torch.Tensor, pad_radius: float) -> list:
        """Apply padding to a convex hull by expanding it outward from the centroid.

        Args:
            hull_tensor: Hull vertices. Shape: [n_vertices, 2]
            pad_radius: Padding radius

        Returns:
            List of padded vertices
        """
        # Calculate centroid of the convex hull
        centroid = torch.mean(hull_tensor, dim=0)

        # For each vertex, calculate vector from centroid and extend outward
        padded_vertices = []
        for vertex in hull_tensor:
            # Vector from centroid to vertex
            direction = vertex - centroid
            direction_norm = torch.norm(direction)

            # Handle case where vertex is at centroid (shouldn't happen in practice)
            if direction_norm > 1e-8:
                # Normalize direction and scale by padding
                direction_unit = direction / direction_norm
                padded_vertex = vertex + direction_unit * pad_radius
            else:
                padded_vertex = vertex

            padded_vertices.append(padded_vertex)

        return padded_vertices

    def _point_in_convex_hull_optimized(
        self, points: torch.Tensor, hulls: torch.Tensor
    ) -> torch.Tensor:
        """Optimized point-in-convex-polygon test using efficient broadcasting.

        Args:
            points: Points to test. Shape: [batch, horizon, n_points, 2]
            hulls: Convex hull vertices. Shape: [batch, n_vertices, 2]

        Returns:
            Boolean tensor indicating if points are inside. Shape: [batch, horizon, n_points]
        """
        batch_size, horizon, n_points, _ = points.shape
        n_vertices = hulls.shape[1]

        # Reshape for efficient broadcasting
        # points: [batch, horizon, n_points, 1, 2]
        # hulls: [batch, 1, 1, n_vertices, 2]
        points_exp = points.unsqueeze(3)  # [batch, horizon, n_points, 1, 2]
        hulls_exp = hulls.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, n_vertices, 2]

        # Get hull edges using efficient indexing
        hull_curr = hulls_exp  # [batch, 1, 1, n_vertices, 2]
        hull_next = torch.roll(hulls_exp, shifts=-1, dims=3)  # [batch, 1, 1, n_vertices, 2]

        # Vectorized edge and point vectors
        edge_vecs = hull_next - hull_curr  # [batch, 1, 1, n_vertices, 2]
        point_vecs = points_exp - hull_curr  # [batch, horizon, n_points, n_vertices, 2]

        # Cross product computation
        cross_products = (
            edge_vecs[..., 0] * point_vecs[..., 1] - edge_vecs[..., 1] * point_vecs[..., 0]
        )  # [batch, horizon, n_points, n_vertices]

        # Check sign consistency
        eps = 1e-8
        all_positive = torch.all(cross_products >= -eps, dim=3)  # [batch, horizon, n_points]
        all_negative = torch.all(cross_products <= eps, dim=3)  # [batch, horizon, n_points]

        return all_positive | all_negative

    def _distance_to_convex_hull_optimized(
        self, points: torch.Tensor, hulls: torch.Tensor
    ) -> torch.Tensor:
        """Optimized distance computation using efficient broadcasting.

        Args:
            points: Points. Shape: [batch, horizon, n_points, 2]
            hulls: Convex hull vertices. Shape: [batch, n_vertices, 2]

        Returns:
            Distances. Shape: [batch, horizon, n_points]
        """
        batch_size, horizon, n_points, _ = points.shape
        n_vertices = hulls.shape[1]

        # Efficient broadcasting setup
        hulls_exp = hulls.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, n_vertices, 2]
        edge_start = hulls_exp  # [batch, 1, 1, n_vertices, 2]
        edge_end = torch.roll(hulls_exp, shifts=-1, dims=3)  # [batch, 1, 1, n_vertices, 2]

        # Compute distance from all points to all edges efficiently
        edge_distances = self._point_to_line_distance_optimized(
            points,  # [batch, horizon, n_points, 2]
            edge_start,  # [batch, 1, 1, n_vertices, 2]
            edge_end,  # [batch, 1, 1, n_vertices, 2]
        )  # [batch, horizon, n_points, n_vertices]

        # Use smooth minimum for better gradients
        smoothing = 0.01
        weights = torch.softmax(
            -edge_distances / smoothing, dim=3
        )  # [batch, horizon, n_points, n_vertices]
        smooth_min_dist = (weights * edge_distances).sum(dim=3)  # [batch, horizon, n_points]

        return smooth_min_dist

    def _point_to_line_distance_optimized(
        self, points: torch.Tensor, line_starts: torch.Tensor, line_ends: torch.Tensor
    ) -> torch.Tensor:
        """Optimized vectorized distance from points to line segments.

        Args:
            points: Points. Shape: [batch, horizon, n_points, 2]
            line_starts: Line segment start points. Shape: [batch, 1, 1, n_lines, 2]
            line_ends: Line segment end points. Shape: [batch, 1, 1, n_lines, 2]

        Returns:
            Distances from points to line segments. Shape: [batch, horizon, n_points, n_lines]
        """
        # Efficient broadcasting without massive memory expansion
        line_vectors = line_ends - line_starts  # [batch, 1, 1, n_lines, 2]
        line_lengths_sq = torch.sum(
            line_vectors * line_vectors, dim=-1, keepdim=True
        )  # [batch, 1, 1, n_lines, 1]
        line_lengths_sq = torch.clamp(line_lengths_sq, min=1e-8)

        # Broadcasting-friendly computation
        points_exp = points.unsqueeze(3)  # [batch, horizon, n_points, 1, 2]
        point_vectors = points_exp - line_starts  # [batch, horizon, n_points, n_lines, 2]

        # Project point vectors onto line vectors
        projections = torch.sum(
            point_vectors * line_vectors, dim=-1, keepdim=True
        )  # [batch, horizon, n_points, n_lines, 1]
        t = projections / line_lengths_sq  # [batch, horizon, n_points, n_lines, 1]
        t_clamped = torch.clamp(t, 0.0, 1.0)

        # Find closest points on line segments
        closest_points = (
            line_starts + t_clamped * line_vectors
        )  # [batch, horizon, n_points, n_lines, 2]

        # Calculate distances
        distances = torch.norm(
            points_exp - closest_points, dim=-1
        )  # [batch, horizon, n_points, n_lines]

        return distances
