# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo._src.state.state_robot import RobotState
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


@dataclass
class ConnectedGraph:
    """Data class stores information about the created graph. Useful for debugging."""

    #: Shape is (B, N, DOF)
    nodes: torch.Tensor

    #: Shape is (num_edges, 2, DOF) where each edge is a dof configuration
    edges: torch.Tensor

    #: Shape is (num_edges,3) where each column is [start_node_idx, end_node_idx, edge_distance]
    connectivity: torch.Tensor
    robot_state_nodes: Optional[RobotState] = None
    shortest_path_lengths: Optional[torch.Tensor] = None

    def set_shortest_path_lengths(self, shortest_path_lengths: torch.Tensor):
        self.shortest_path_lengths = shortest_path_lengths

    def get_node_distance(self):
        if self.shortest_path_lengths is not None:
            min_l = min(self.nodes.shape[0], self.shortest_path_lengths.shape[0])
            return torch.cat(
                (self.nodes[:min_l], self.shortest_path_lengths[:min_l].unsqueeze(1)), dim=-1
            )
        else:
            return None


@get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
def jit_add_nodes_to_buffer(
    preallocated_node_buffer: torch.Tensor,
    new_nodes: torch.Tensor,
    used_node_count: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    number_of_nodes = new_nodes.shape[0]
    preallocated_node_buffer[used_node_count : used_node_count + number_of_nodes, :action_dim] = (
        new_nodes
    )
    preallocated_node_buffer[used_node_count : used_node_count + number_of_nodes, action_dim] = (
        torch.arange(
            used_node_count,
            used_node_count + number_of_nodes,
            device=device,
            dtype=dtype,
        )
    )
    return preallocated_node_buffer


@get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
def jit_add_all_nodes_to_buffer(
    all_nodes, preallocated_node_buffer, used_node_count: int, action_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    preallocated_node_buffer = jit_add_nodes_to_buffer(
        preallocated_node_buffer,
        all_nodes,
        used_node_count,
        action_dim,
        all_nodes.device,
        all_nodes.dtype,
    )

    new_nodes_in_roadmap = preallocated_node_buffer[
        used_node_count : used_node_count + all_nodes.shape[0], :
    ]

    used_node_count = used_node_count + all_nodes.shape[0]

    return preallocated_node_buffer, new_nodes_in_roadmap, used_node_count


class GraphNodeManager:
    """Manages the graph nodes storage and operations for a PRM motion planner."""

    def __init__(
        self,
        config,
        distance_calculator=None,
        graph_path_finder=None,
        auxiliary_rollout=None,
        device_cfg=None,
    ):
        self.config = config
        self.device_cfg = device_cfg
        self.distance_calculator = distance_calculator
        self.graph_path_finder = graph_path_finder
        self.auxiliary_rollout = auxiliary_rollout

        # Initialize buffers
        self._node_idx_padding_buffer = torch.as_tensor(
            [0.0], device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )

        self._preallocated_idx_buffer = torch.arange(
            0,
            self.config.steer_buffer_size,
            device=self.device_cfg.device,
            dtype=torch.int64,
        )
        self._preallocated_node_buffer = torch.zeros(
            (self.config.max_nodes, self.action_dim + 1),  # action_dim, node_index
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )

        self._used_node_count = 0

    # @profiler.record_function("base_graph_planner/add_nodes_to_buffer")
    def add_nodes_to_buffer(self, new_nodes):
        number_of_nodes = new_nodes.shape[0]
        if self._used_node_count + number_of_nodes >= self._preallocated_node_buffer.shape[0]:
            log_and_raise(
                f"Path memory buffer is too small: buffer size={self._preallocated_node_buffer.shape[0]}, "
                f"required={self._used_node_count + number_of_nodes}"
            )

        self._preallocated_node_buffer = jit_add_nodes_to_buffer(
            self._preallocated_node_buffer,
            new_nodes,
            self._used_node_count,
            self.action_dim,
            self.device_cfg.device,
            self.device_cfg.dtype,
        )
        self._used_node_count = self._used_node_count + number_of_nodes

    def get_connected_graph(self):
        if self._used_node_count == 0:
            return None
        self.graph_path_finder.update_graph()
        edge_list = self.graph_path_finder.get_edges()
        edges = torch.as_tensor(
            edge_list, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )

        # find start and end points for these edges:
        start_pts = self._preallocated_node_buffer[edges[:, 0].long(), : self.action_dim].unsqueeze(1)
        end_pts = self._preallocated_node_buffer[edges[:, 1].long(), : self.action_dim].unsqueeze(1)

        # first check the start and end points:
        node_edges = torch.cat((start_pts, end_pts), dim=1)

        nodes = self.valid_node_buffer[:, : self.action_dim]
        robot_state_nodes = self.auxiliary_rollout.compute_state_from_action(nodes.unsqueeze(1))

        return ConnectedGraph(
            nodes=nodes, edges=node_edges, connectivity=edges, robot_state_nodes=robot_state_nodes
        )

    # @profiler.record_function("base_graph_planner/_register_nodes_and_connections")
    def register_nodes_and_connections(
        self, node_set: torch.Tensor, start_nodes: torch.Tensor, add_exact_node=False
    ):
        """Register nodes and their connections to the roadmap.

        Args:
            node_set (torch.Tensor): (B, DOF+1) Tensor of node configurations with indices
            start_nodes (torch.Tensor): (B, DOF+1) Tensor of start nodes with indices
            add_exact_node (bool, optional): Whether to add the exact nodes. Defaults to False.
        """
        nodes_in_roadmap = self.add_nodes_to_roadmap(
            nodes=node_set[:, : self.action_dim], add_exact_node=add_exact_node
        )
        # now connect start nodes to new nodes:
        edge_list = []
        edge_distance = (
            self.distance_calculator.calculate_weighted_distance(
                start_nodes[:, : self.action_dim], nodes_in_roadmap[:, : self.action_dim]
            )
            .to(device="cpu")
            .tolist()
        )
        start_idx_list = start_nodes[:, self.action_dim].to(device="cpu", dtype=torch.int64).tolist()
        goal_idx_list = (
            nodes_in_roadmap[:, self.action_dim].to(device="cpu", dtype=torch.int64).tolist()
        )
        edge_list = [
            [start_idx_list[x], goal_idx_list[x], edge_distance[x]]
            for x in range(nodes_in_roadmap.shape[0])
        ]
        self.graph_path_finder.add_edges(edge_list)
        return True

    # @profiler.record_function("base_graph_planner/add_nodes_to_roadmap")
    def add_nodes_to_roadmap(
        self, nodes: torch.Tensor, add_exact_node: bool = False
    ) -> torch.Tensor:
        """Add nodes to roadmap and return the node set.

        Args:
            nodes: (B, action_dim)
            add_exact_node: (bool)

        Returns:
            node_set: (B, action_dim + 1)
        """
        # Check for duplicates in new nodes:
        if len(nodes.shape) != 2:
            log_and_raise("nodes must be a 2D tensor")
        if nodes.shape[1] != self.action_dim:
            log_and_raise(f"nodes must have the action_dim columns: {nodes.shape[1]} != {self.action_dim}")
        if self._used_node_count == 0:
            log_and_raise("used_node_count must not be 0")
        similarity_threshold = self.config.cspace_similarity_threshold
        if add_exact_node:
            similarity_threshold = 0.0

        unique_nodes, n_inv = self.distance_calculator.get_unique_nodes(nodes, similarity_threshold)

        dist, closest_node_in_roadmap_idx = torch.min(
            self.distance_calculator.calculate_weighted_distance(
                unique_nodes[:, : self.action_dim].unsqueeze(1),
                self.valid_node_buffer[:, : self.action_dim],
            ),
            dim=-1,
        )

        node_exists_in_roadmap = dist <= similarity_threshold
        unique_nodes_to_add = unique_nodes[~node_exists_in_roadmap]

        if (
            self._preallocated_node_buffer.shape[0]
            <= self._used_node_count + unique_nodes_to_add.shape[0]
        ):
            log_and_raise(
                "reached max_nodes in graph, reduce graph attempts or increase max_nodes "
                + f"{self._preallocated_node_buffer.shape}, {self._used_node_count}, {unique_nodes_to_add.shape}"
            )
        if unique_nodes.shape[-1] != self.action_dim:
            log_and_raise("unique_nodes must have action_dim columns")
        if unique_nodes_to_add.shape[-1] != self.action_dim:
            log_and_raise("unique_nodes_to_add must have action_dim columns")

        self._preallocated_node_buffer, node_set, self._used_node_count = self._jit_add_new_nodes(
            unique_nodes,
            unique_nodes_to_add,
            node_exists_in_roadmap,
            self._preallocated_node_buffer,
            closest_node_in_roadmap_idx,
            self._used_node_count,
            self.action_dim,
            self._node_idx_padding_buffer,
        )

        node_set = node_set[n_inv]
        return node_set

    # @profiler.record_function("base_graph_planner/add_exact_nodes_to_roadmap")
    def add_initial_exact_nodes_to_roadmap(self, nodes: torch.Tensor) -> torch.Tensor:
        """Add nodes to roadmap and return the node set.

        Args:
            nodes: (B, action_dim)

        Returns:
            node_set: (B, action_dim + 1)
        """
        # Check for duplicates in new nodes:
        if len(nodes.shape) != 2:
            log_and_raise("nodes must be a 2D tensor")
        if nodes.shape[1] != self.action_dim:
            log_and_raise(f"nodes must have the action_dim columns: {nodes.shape[1]} != {self.action_dim}")
        if self._used_node_count != 0:
            log_and_raise("used_node_count must be 0")

        unique_nodes, n_inv = self.distance_calculator.get_unique_nodes_zero_distance(nodes)

        self._preallocated_node_buffer, node_set, self._used_node_count = (
            jit_add_all_nodes_to_buffer(
                unique_nodes,
                self._preallocated_node_buffer,
                self._used_node_count,
                self.action_dim,
            )
        )

        node_set = node_set[n_inv]
        return node_set

    def get_nodes_in_path(self, path_list: List[Union[List[int], None]]):
        paths = []
        for i in range(len(path_list)):
            if path_list[i] is None:
                paths.append(None)
                continue
            paths.append(self._preallocated_node_buffer[path_list[i], : self.action_dim])
        return paths

    def reset_buffer(self):
        self.graph_path_finder.reset_graph()
        self._preallocated_node_buffer *= 0.0
        self._used_node_count = 0
        self._default_joint_position_feasible = None

    def reset_graph_path_finder(self):
        self.graph_path_finder.reset_graph()

    @property
    def action_dim(self) -> int:
        """Dimensionality of the action space (robot degrees of freedom)."""
        return self.auxiliary_rollout.action_dim

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._used_node_count

    @staticmethod
    @profiler.record_function("base_graph_planner/jit_add_new_nodes")
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
    def _jit_add_new_nodes(
        unique_nodes,
        unique_nodes_to_add,
        node_exists_in_roadmap,
        preallocated_node_buffer,
        closest_node_in_roadmap_idx,
        used_node_count: int,
        action_dim: int,
        node_idx_padding_buffer: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        preallocated_node_buffer, new_nodes_in_roadmap, used_node_count = (
            jit_add_all_nodes_to_buffer(
                unique_nodes_to_add,
                preallocated_node_buffer,
                used_node_count,
                action_dim,
            )
        )

        # new_nodes_in_roadmap does not contain the nodes that already exist in the roadmap
        #

        # existing_node_in_all_nodes_idx = closest_node_in_roadmap_idx[node_exists_in_roadmap]

        # get nodes from allocated buffer
        # existing_nodes_in_roadmap = preallocated_node_buffer[existing_node_in_all_nodes_idx,:]

        unique_nodes_with_idx = torch.cat(
            [unique_nodes, node_idx_padding_buffer.repeat(unique_nodes.shape[0], 1)],
            dim=-1,
        )  # make it action_dim + 1
        # Create a mask tensor once (avoid using ~ operator which creates a new tensor)
        exists_mask = node_exists_in_roadmap
        not_exists_mask = ~node_exists_in_roadmap

        # Use masked_scatter_ for in-place updates (avoids creating intermediate tensors)
        if exists_mask.any():
            closest_idx = closest_node_in_roadmap_idx[exists_mask].to(dtype=unique_nodes.dtype)

            unique_nodes_with_idx[:, action_dim].masked_scatter_(exists_mask, closest_idx)

        if not_exists_mask.any():
            unique_nodes_with_idx[:, action_dim].masked_scatter_(
                not_exists_mask, new_nodes_in_roadmap[:, action_dim]
            )

        return preallocated_node_buffer, unique_nodes_with_idx, used_node_count

    @property
    def node_idx_padding_buffer(self):
        return self._node_idx_padding_buffer

    @property
    def preallocated_node_buffer(self):
        return self._preallocated_node_buffer

    @property
    def valid_node_buffer(self):
        return self._preallocated_node_buffer[: self.n_nodes, :]
