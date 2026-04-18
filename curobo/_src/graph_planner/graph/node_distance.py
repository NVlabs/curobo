# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

# Standard Library
from typing import Tuple

# Third Party
import torch

# CuRobo
from curobo._src.util.tensor_util import stable_topk
from curobo._src.util.torch_util import get_torch_jit_decorator
from curobo._src.types.device_cfg import DeviceCfg


class DistanceNeighborCalculator:
    """Class for handling distance calculations, nearest neighbors, and unique node operations.

    This class encapsulates functionality related to:
    - Calculating weighted distances between configurations
    - Finding nearest neighbors in configuration space
    - Identifying unique nodes based on distance thresholds
    """

    def __init__(self, action_dim: int, cspace_distance_weight: torch.Tensor, device_cfg: DeviceCfg):
        """Initialize the DistanceNeighborCalculator.

        Args:
            action_dim: Dimensionality of the action space (robot DOF)
            cspace_distance_weight: Weights for each dimension in distance calculations
            device_cfg: Tensor arguments specifying device and dtype
        """
        self.action_dim = action_dim
        self.cspace_distance_weight = cspace_distance_weight
        self.device_cfg = device_cfg

    def calculate_weighted_distance(
        self, pt: torch.Tensor, batch_pts: torch.Tensor
    ) -> torch.Tensor:
        """Calculate weighted distance between a point and batch of points.

        Args:
            pt: Single point or batch of points (B, DOF)
            batch_pts: Batch of points to calculate distance to (B, DOF)

        Returns:
            Weighted distances between points
        """
        return self.jit_calculate_weighted_distance(pt, batch_pts, self.cspace_distance_weight)

    def find_nearest_neighbors(
        self, new_nodes: torch.Tensor, existing_nodes: torch.Tensor, neighbors_per_node: int
    ) -> torch.Tensor:
        """Find nearest neighbors for each new node from existing nodes.

        Args:
            new_nodes: New nodes to find neighbors for (B, DOF+1)
            existing_nodes: Existing nodes to search in (N, DOF+1)
            neighbors_per_node: Number of neighbors to find per node

        Returns:
            Nearest neighbors for each new node (B, K, DOF+1)
        """
        nearest_nodes = self.jit_find_nearest_neighbors(
            new_nodes,
            existing_nodes,
            self.cspace_distance_weight,
            neighbors_per_node,
            self.action_dim,
        )
        return nearest_nodes

    def get_unique_nodes(
        self, nodes: torch.Tensor, similarity_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find unique nodes based on weighted distance similarity.

        Args:
            nodes: Nodes to find unique elements from (B, DOF)
            similarity_threshold: Threshold for considering nodes similar

        Returns:
            Tuple of (unique_nodes, inverse_indices)
        """
        return self.jit_get_unique_nodes(
            nodes, self.cspace_distance_weight, similarity_threshold, self.action_dim
        )

    def get_unique_nodes_zero_distance(
        self, nodes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find unique nodes with zero distance threshold.

        Args:
            nodes: Nodes to find unique elements from (B, DOF)

        Returns:
            Tuple of (unique_nodes, inverse_indices)
        """
        return self.jit_get_unique_nodes_zero_distance(
            nodes, self.cspace_distance_weight, self.action_dim
        )

    @staticmethod
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
    def jit_calculate_weighted_distance(
        pt: torch.Tensor, batch_pts: torch.Tensor, distance_weight: torch.Tensor
    ) -> torch.Tensor:
        """JIT-compiled function to calculate weighted distance.

        Args:
            pt: Point(s) to calculate distance from
            batch_pts: Batch of points to calculate distance to
            distance_weight: Weight for each dimension

        Returns:
            Weighted distances
        """
        vec = batch_pts - pt
        dist = torch.norm(vec * distance_weight, dim=-1)
        return dist

    @staticmethod
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
    def jit_find_nearest_neighbors(
        new_nodes: torch.Tensor,
        existing_nodes: torch.Tensor,
        distance_weight: torch.Tensor,
        neighbors_per_node: int,
        action_dim: int,
    ) -> torch.Tensor:
        """JIT-compiled function to find nearest neighbors.

        Args:
            new_nodes: New nodes to find neighbors for
            existing_nodes: Existing nodes to search in
            distance_weight: Weight for each dimension
            neighbors_per_node: Number of neighbors to find per node
            action_dim: Dimensionality of action space

        Returns:
            Nearest neighbors for each new node
        """
        batch_pts = new_nodes[:, :action_dim] * distance_weight
        pt = existing_nodes[:, :action_dim] * distance_weight
        dist = torch.cdist(batch_pts, pt, p=2.0)
        _, idx = stable_topk(dist, k=neighbors_per_node, largest=False, dim=-1)
        return existing_nodes[idx]

    @staticmethod
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
    def jit_get_unique_nodes(
        nodes: torch.Tensor,
        distance_weight: torch.Tensor,
        node_similarity_threshold: float,
        action_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """JIT-compiled function to find unique nodes based on distance threshold.

        This can consume a lot of memory for large number of nodes.

        Args:
            nodes: Nodes to find unique elements from
            distance_weight: Weight for each dimension
            node_similarity_threshold: Threshold for considering nodes similar
            action_dim: Dimensionality of action space

        Returns:
            Tuple of (unique_nodes, inverse_indices)
        """
        node_positions = nodes[:, :action_dim] * distance_weight
        dist_node = torch.cdist(node_positions, node_positions, p=2.0)

        similar_nodes_flag = dist_node <= node_similarity_threshold
        dist_node[similar_nodes_flag] = 0.0
        _, idx = torch.min(dist_node, dim=-1)

        unique_idx, inverse_idx = torch.unique(idx, return_inverse=True)

        unique_nodes = nodes[unique_idx]
        return unique_nodes, inverse_idx

    @staticmethod
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
    def jit_get_unique_nodes_zero_distance(
        nodes: torch.Tensor, distance_weight: torch.Tensor, action_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """JIT-compiled function to find unique nodes with zero distance threshold.

        Args:
            nodes: Nodes to find unique elements from
            distance_weight: Weight for each dimension
            action_dim: Dimensionality of action space

        Returns:
            Tuple of (unique_nodes, inverse_indices)
        """
        vec = nodes[:, :action_dim].unsqueeze(1) - nodes[:, :action_dim]
        dist_node = torch.norm(vec * distance_weight, dim=-1)
        _, idx = torch.min(dist_node, dim=-1)
        unique_idx, inverse_idx = torch.unique(idx, return_inverse=True)
        unique_nodes = nodes[unique_idx]
        return unique_nodes, inverse_idx
