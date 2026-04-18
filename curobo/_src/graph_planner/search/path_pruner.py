# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

# Standard Library
from typing import Callable, List, Optional, Tuple

# Third Party
import torch

# CuRobo
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


class PathPruner:
    """Class responsible for pruning paths in the PRM graph planner.

    This class handles path pruning by shortcutting from each node to other nodes in the path.
    """

    def __init__(self, config: PRMGraphPlannerCfg, device_cfg: Optional[DeviceCfg] = None):
        """Initialize the path pruner.

        Args:
            config: Configuration for the PRM planner
            device_cfg: Tensor device and type arguments. If None, uses config.device_cfg
        """
        self.config = config
        self.device_cfg = config.device_cfg if device_cfg is None else device_cfg

        # These will be set through set_dependencies method
        self.action_dim = None
        self.cspace_distance_weight = None
        self._preallocated_node_buffer = None
        self._steer_and_register_edges_fn = None
        self._find_path_for_index_pairs_fn = None

    def set_dependencies(
        self,
        action_dim: int,
        cspace_distance_weight: torch.Tensor,
        preallocated_node_buffer: torch.Tensor,
        steer_and_register_edges_fn: Callable,
        find_path_for_index_pairs_fn: Callable,
    ):
        """Set the dependencies needed for path pruning.

        Args:
            action_dim: Dimensionality of the action space (robot DOF)
            cspace_distance_weight: Weight vector for C-space distance calculations
            preallocated_node_buffer: Buffer containing roadmap nodes
            steer_and_register_edges_fn: Function to steer between nodes and register edges
            find_path_for_index_pairs_fn: Function to find paths between node indices
        """
        self.action_dim = action_dim
        self.cspace_distance_weight = cspace_distance_weight
        self._preallocated_node_buffer = preallocated_node_buffer
        self._steer_and_register_edges_fn = steer_and_register_edges_fn
        self._find_path_for_index_pairs_fn = find_path_for_index_pairs_fn

        # Check that all dependencies are properly set
        if None in [
            self.action_dim,
            self.cspace_distance_weight,
            self._preallocated_node_buffer,
            self._steer_and_register_edges_fn,
            self._find_path_for_index_pairs_fn,
        ]:
            log_and_raise("Not all dependencies were properly set for PathPruner")

    # @profiler.record_function("path_pruner/prune_path_with_shortcuts")
    def prune_path_with_shortcuts(
        self, paths: List[List[int]], start_idx: List[int], goal_idx: List[int]
    ) -> Tuple[List[List[int]], List[float]]:
        """Prunes paths by attempting to connect distant nodes with shortcuts.

        This method tries to connect all pairs of nodes in the path directly, potentially
        bypassing intermediate nodes if a direct connection is feasible.

        Args:
            paths: List of paths, where each path is a list of node indices
            start_idx: List of start node indices
            goal_idx: List of goal node indices

        Returns:
            Tuple of:
                - Pruned paths (list of lists of node indices)
                - Path lengths (list of floats)
        """
        edge_set = self._prepare_edges_for_shortcuts(paths)

        # Register edges in one batch operation
        self._steer_and_register_edges_fn(edge_set[:, 0], edge_set[:, 1], add_exact_node=False)

        # Find shortest paths with updated graph
        s_path, c_max = self._find_path_for_index_pairs_fn(start_idx, goal_idx, return_length=True)
        return s_path, c_max

    # @profiler.record_function("path_pruner/prepare_edges_for_shortcuts")
    def _prepare_edges_for_shortcuts(self, paths: List[List[int]]) -> torch.Tensor:
        """Prepare all possible edge pairs for shortcuts from the given paths.

        Args:
            paths: List of paths, where each path is a list of node indices

        Returns:
            Tensor of shape (total_edges, 2, action_dim+1) containing all possible edge pairs
        """
        # Count total edge pairs to allocate memory once
        total_edges = 0
        for path in paths:
            path_len = len(path)
            total_edges += (path_len * (path_len + 1)) // 2

        # Pre-allocate the edge tensor with exact size needed
        edge_set = torch.zeros(
            (total_edges, 2, self.action_dim + 1),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )

        # Fill the edge set efficiently
        edge_idx = 0
        for k in range(len(paths)):
            path_indices = paths[k]
            path_len = len(path_indices)

            # Gather node configurations in one operation instead of individual lookups
            path_nodes = self._preallocated_node_buffer[path_indices]

            # Vectorized approach for creating edge pairs
            for i in range(path_len):
                # Number of edges from this node
                num_edges = path_len - i

                # Set source nodes (repeated for each destination)
                edge_set[edge_idx : edge_idx + num_edges, 0] = path_nodes[i]

                # Set destination nodes (all nodes from i to end)
                edge_set[edge_idx : edge_idx + num_edges, 1] = path_nodes[i:path_len]

                edge_idx += num_edges
        return edge_set
