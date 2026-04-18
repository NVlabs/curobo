# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

# Standard Library
from typing import Optional, Tuple

# Third Party
import torch

# CuRobo
from curobo._src.graph_planner.graph.connector_linear import LinearConnector
from curobo._src.graph_planner.graph.node_distance import DistanceNeighborCalculator
from curobo._src.graph_planner.graph.node_manager import GraphNodeManager
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.torch_util import get_profiler_decorator, get_torch_jit_decorator


class GraphConstructor:
    """Handles the construction and maintenance of the PRM graph structure.

    This class encapsulates the functionality related to building and maintaining
    the graph structure for PRM motion planning, including node addition, edge creation,
    and special handling for terminal (start/goal) nodes.
    """

    def __init__(
        self,
        config: PRMGraphPlannerCfg,
        linear_connector: LinearConnector,
        distance_calculator: DistanceNeighborCalculator,
        node_manager: GraphNodeManager,
        action_dim: int,
        check_feasibility_fn,
        device_cfg,
    ):
        """Initialize the graph constructor.

        Args:
            config: Configuration for the graph planner
            device_cfg: Tensor arguments for device/dtype
            linear_connector: Component for linear connection between nodes
            distance_calculator: Component for calculating distances and neighbors
            node_manager: Component for managing graph nodes
            action_dim: Dimensionality of the action space
            check_feasibility_fn: Function to check if configurations are feasible
        """
        self.config = config
        self.device_cfg = device_cfg
        self.linear_connector = linear_connector
        self.distance_calculator = distance_calculator
        self.node_manager = node_manager
        self.action_dim = action_dim
        self.check_feasibility_fn = check_feasibility_fn

        # State for default joint position nodes
        self._default_joint_position_feasible = None
        self._default_node_in_roadmap = None

    @get_profiler_decorator("graph_constructor/steer_and_register_edges")
    def steer_and_register_edges(
        self,
        start_nodes: torch.Tensor,
        goal_nodes: torch.Tensor,
        add_exact_node=False,
    ):
        """Connect nodes from start to goal where both are batched.

        Args:
            start_nodes: Starting configurations with indices, shape (B, action_dim+1)
            goal_nodes: Goal configurations with indices, shape (B, action_dim+1)
            add_exact_node: Whether to add the exact goal node
        """
        if start_nodes.shape[0] != goal_nodes.shape[0]:
            log_and_raise("start_nodes and goal_nodes must have the same batch size")

        if start_nodes.shape[-1] != self.action_dim + 1:
            log_and_raise("start_nodes must have action_dim + 1 columns")
        if goal_nodes.shape[-1] != self.action_dim + 1:
            log_and_raise("goal_nodes must have action_dim + 1 columns")

        steer_nodes = self.linear_connector.steer_until_infeasible(
            start_nodes,
            goal_nodes,
        )
        self.node_manager.register_nodes_and_connections(
            steer_nodes, start_nodes, add_exact_node=add_exact_node
        )

    @get_profiler_decorator("graph_constructor/connect_nodes")
    def connect_nodes(
        self,
        new_nodes: torch.Tensor,
        add_exact_node=False,
        neighbors_per_node=10,
    ):
        """Connect new nodes to the existing graph.

        Args:
            new_nodes: New configurations to add, shape (B, action_dim) or (B, action_dim+1)
            add_exact_node: Whether to add the exact nodes
            neighbors_per_node: Number of neighbors to connect to
        """
        # Connect the batch to the existing graph
        if new_nodes.shape[0] == 0:
            log_info("no valid configuration found")
            return

        # Add index column if needed
        if new_nodes.shape[-1] != self.action_dim + 1:
            new_nodes = torch.cat(
                (
                    new_nodes,
                    self.node_manager.node_idx_padding_buffer.repeat(new_nodes.shape[0], 1),
                ),
                dim=-1,
            )

        # Find nearest neighbors in the existing roadmap
        existing_nodes_in_roadmap = self.distance_calculator.find_nearest_neighbors(
            new_nodes,
            self.node_manager.valid_node_buffer,
            neighbors_per_node=min(neighbors_per_node, self.node_manager.n_nodes),
        )

        # Reshape for batch processing
        goal_nodes = (
            new_nodes.unsqueeze(1)
            .repeat(1, existing_nodes_in_roadmap.shape[1], 1)
            .reshape(new_nodes.shape[0] * existing_nodes_in_roadmap.shape[1], self.action_dim + 1)
        )
        start_nodes = existing_nodes_in_roadmap.reshape(
            new_nodes.shape[0] * existing_nodes_in_roadmap.shape[1], self.action_dim + 1
        )

        # Connect the nodes
        self.steer_and_register_edges(start_nodes, goal_nodes, add_exact_node=add_exact_node)

    @get_profiler_decorator("graph_constructor/initialize_default_node")
    def initialize_default_node(
        self, default_joint_state: JointState
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """Initialize the default joint position node if needed.

        Args:
            default_joint_state: The default joint state to use

        Returns:
            Tuple of (default_node_in_roadmap, default_joint_position_feasible)
        """
        if self.config.use_default_position_heuristic and self._default_joint_position_feasible is None:
            default_feasible = self.check_feasibility_fn(
                default_joint_state.position.view(1, self.action_dim)
            )
            self._default_joint_position_feasible = default_feasible.item()
            if self._default_joint_position_feasible:
                default_node = default_joint_state.position.view(1, self.action_dim).clone()
                self._default_node_in_roadmap = (
                    self.node_manager.add_initial_exact_nodes_to_roadmap(default_node)
                )

        return self._default_node_in_roadmap, self._default_joint_position_feasible

    @get_profiler_decorator("graph_constructor/initialize_terminal_graph_connections")
    def initialize_terminal_graph_connections(
        self,
        x_init_batch: torch.Tensor,
        x_goal_batch: torch.Tensor,
        default_joint_state: JointState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize connections for terminal nodes (start and goal).

        Args:
            x_init_batch: Batch of start configurations, shape (B, action_dim)
            x_goal_batch: Batch of goal configurations, shape (B, action_dim)
            default_joint_state: Default joint state to use

        Returns:
            Tuple of (start_nodes_in_roadmap, goal_nodes_in_roadmap)
        """
        if len(x_init_batch.shape) != 2:
            log_and_raise("x_init_batch must be a 2D tensor")
        if len(x_goal_batch.shape) != 2:
            log_and_raise("x_goal_batch must be a 2D tensor")
        if x_init_batch.shape[0] != x_goal_batch.shape[0]:
            log_and_raise("x_init_batch and x_goal_batch must have the same batch size")
        if x_init_batch.shape[1] != self.action_dim:
            log_and_raise("x_init_batch must have the same number of actions as the model")
        if x_goal_batch.shape[1] != self.action_dim:
            log_and_raise("x_goal_batch must have the same number of actions as the model")

        # Initialize default node if configured
        self.initialize_default_node(default_joint_state)

        # Add start and goal nodes to the roadmap
        batch_size = x_init_batch.shape[0]

        start_goal_nodes = torch.cat((x_init_batch, x_goal_batch), dim=0)

        if self.node_manager.n_nodes == 0:
            start_goal_nodes_in_roadmap = self.node_manager.add_initial_exact_nodes_to_roadmap(
                start_goal_nodes
            )
        else:
            start_goal_nodes_in_roadmap = self.node_manager.add_nodes_to_roadmap(
                start_goal_nodes, add_exact_node=True
            )

        # Reshape to (batch_size, 2, action_dim+1) for easier indexing
        start_goal_nodes_in_roadmap = start_goal_nodes_in_roadmap.view(
            2, batch_size, self.action_dim + 1
        )

        start_nodes_in_roadmap = start_goal_nodes_in_roadmap[0, :, :]
        goal_nodes_in_roadmap = start_goal_nodes_in_roadmap[1, :, :]

        # Process terminal nodes and establish initial connections
        start_steer_nodes, goal_steer_nodes = self._preprocess_terminal_nodes(
            start_nodes_in_roadmap, goal_nodes_in_roadmap
        )

        # Steer from start to goal and goal to start
        self.steer_and_register_edges(start_steer_nodes, goal_steer_nodes, add_exact_node=False)

        # Connect terminal nodes to nearest nodes if configured
        if self.config.connect_terminal_nodes_with_nearest:
            nodes = torch.cat(
                (
                    x_init_batch,
                    x_goal_batch,
                )
            )

            self.connect_nodes(
                nodes,
                neighbors_per_node=self.config.neighbors_per_node,
                add_exact_node=False,
            )

        return start_nodes_in_roadmap, goal_nodes_in_roadmap

    @get_profiler_decorator("graph_constructor/preprocess_terminal_nodes")
    @get_torch_jit_decorator(dynamic=True, only_valid_for_compile=True, slow_to_compile=True)
    def _preprocess_terminal_nodes(
        self,
        start_nodes_in_roadmap: torch.Tensor,
        goal_nodes_in_roadmap: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess start and goal nodes for connection.

        Args:
            start_nodes_in_roadmap: Start nodes in the roadmap, shape (B, action_dim+1)
            goal_nodes_in_roadmap: Goal nodes in the roadmap, shape (B, action_dim+1)

        Returns:
            Tuple of (start_steer_nodes, goal_steer_nodes)
        """
        # Create bidirectional connections between start and goal
        batch_size = start_nodes_in_roadmap.shape[0]
        start_steer_nodes = torch.cat(
            (
                start_nodes_in_roadmap,
                goal_nodes_in_roadmap,
            ),
            dim=0,
        )
        goal_steer_nodes = torch.cat(
            (
                goal_nodes_in_roadmap,
                start_nodes_in_roadmap,
            ),
            dim=0,
        )

        # Handle default node if available and feasible
        if self.config.use_default_position_heuristic and self._default_joint_position_feasible:
            default_node_in_roadmap = self._default_node_in_roadmap
            default_node_repeat = default_node_in_roadmap.view(1, self.action_dim + 1).repeat(
                batch_size, 1
            )

            # Create connections to/from default node
            default_start_steer_nodes = torch.cat(
                (
                    start_nodes_in_roadmap,
                    default_node_repeat,
                    goal_nodes_in_roadmap,
                    default_node_repeat,
                ),
                dim=0,
            )
            default_goal_steer_nodes = torch.cat(
                (
                    default_node_repeat,
                    start_nodes_in_roadmap,
                    default_node_repeat,
                    goal_nodes_in_roadmap,
                ),
                dim=0,
            )

            # Combine all connections
            start_steer_nodes = torch.cat((start_steer_nodes, default_start_steer_nodes), dim=0)
            goal_steer_nodes = torch.cat((goal_steer_nodes, default_goal_steer_nodes), dim=0)

        return start_steer_nodes, goal_steer_nodes

    def reset(self):
        """Reset the graph constructor state."""
        self._default_joint_position_feasible = None
        self._default_node_in_roadmap = None
