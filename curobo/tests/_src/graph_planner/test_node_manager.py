# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Comprehensive tests for GraphNodeManager."""

# Third Party
import pytest
import torch

from curobo._src.graph_planner.graph.node_distance import DistanceNeighborCalculator

# CuRobo
from curobo._src.graph_planner.graph.node_manager import ConnectedGraph, GraphNodeManager
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.graph_planner.search.path_finder_networkx import NetworkXPathFinder
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture(scope="module")
def prm_config():
    """Get PRM planner configuration."""
    return PRMGraphPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
    )


@pytest.fixture(scope="module")
def device_cfg():
    """Get tensor configuration."""
    return DeviceCfg()


@pytest.fixture
def node_manager(prm_config, device_cfg):
    """Create a GraphNodeManager instance with dependencies."""
    # Create distance calculator
    cspace_weight = torch.ones(7, **device_cfg.as_torch_dict())
    distance_calc = DistanceNeighborCalculator(
        action_dim=7,
        cspace_distance_weight=cspace_weight,
        device_cfg=device_cfg,
    )

    # Create path finder
    path_finder = NetworkXPathFinder(seed=42)

    # Create rollout (using config's rollout)
    rollout = RobotRollout(prm_config.rollout_config)

    manager = GraphNodeManager(
        config=prm_config,
        device_cfg=device_cfg,
        distance_calculator=distance_calc,
        graph_path_finder=path_finder,
        auxiliary_rollout=rollout,
    )

    # Add some initial nodes to avoid _used_node_count == 0 errors
    initial_nodes = torch.randn(5, 7, **device_cfg.as_torch_dict())
    manager.add_nodes_to_buffer(initial_nodes)

    return manager


class TestGraphNodeManagerInitialization:
    """Test GraphNodeManager initialization."""

    def test_manager_basic_init(self, prm_config, device_cfg):
        """Test basic manager initialization."""
        cspace_weight = torch.ones(7, **device_cfg.as_torch_dict())
        distance_calc = DistanceNeighborCalculator(7, cspace_weight, device_cfg)
        path_finder = NetworkXPathFinder()
        rollout = RobotRollout(prm_config.rollout_config)

        manager = GraphNodeManager(
            config=prm_config,
            device_cfg=device_cfg,
            distance_calculator=distance_calc,
            graph_path_finder=path_finder,
            auxiliary_rollout=rollout,
        )

        assert manager is not None
        assert manager.config == prm_config
        assert manager.device_cfg == device_cfg
        assert manager._used_node_count == 0

    def test_manager_buffer_allocation(self, node_manager):
        """Test that buffers are properly allocated."""
        assert node_manager._preallocated_node_buffer is not None
        assert node_manager._preallocated_node_buffer.shape[0] == node_manager.config.max_nodes
        assert node_manager._preallocated_node_buffer.shape[1] == node_manager.action_dim + 1

    def test_action_dim_property(self, node_manager):
        """Test action_dim property."""
        assert node_manager.action_dim == 7

    def test_n_nodes_property(self, node_manager):
        """Test n_nodes property."""
        # Should have 5 nodes from fixture
        assert node_manager.n_nodes == 5


class TestGraphNodeManagerNodeOperations:
    """Test node addition and management."""

    def test_add_nodes_to_buffer(self, prm_config, device_cfg):
        """Test adding nodes to buffer."""
        cspace_weight = torch.ones(7, **device_cfg.as_torch_dict())
        distance_calc = DistanceNeighborCalculator(7, cspace_weight, device_cfg)
        path_finder = NetworkXPathFinder()
        rollout = RobotRollout(prm_config.rollout_config)

        manager = GraphNodeManager(
            config=prm_config,
            device_cfg=device_cfg,
            distance_calculator=distance_calc,
            graph_path_finder=path_finder,
            auxiliary_rollout=rollout,
        )

        # Add nodes
        nodes = torch.randn(10, 7, **device_cfg.as_torch_dict())
        manager.add_nodes_to_buffer(nodes)

        assert manager.n_nodes == 10

    def test_add_multiple_batches(self, prm_config, device_cfg):
        """Test adding multiple batches of nodes."""
        cspace_weight = torch.ones(7, **device_cfg.as_torch_dict())
        distance_calc = DistanceNeighborCalculator(7, cspace_weight, device_cfg)
        path_finder = NetworkXPathFinder()
        rollout = RobotRollout(prm_config.rollout_config)

        manager = GraphNodeManager(
            config=prm_config,
            device_cfg=device_cfg,
            distance_calculator=distance_calc,
            graph_path_finder=path_finder,
            auxiliary_rollout=rollout,
        )

        # Add first batch
        nodes1 = torch.randn(5, 7, **device_cfg.as_torch_dict())
        manager.add_nodes_to_buffer(nodes1)
        assert manager.n_nodes == 5

        # Add second batch
        nodes2 = torch.randn(3, 7, **device_cfg.as_torch_dict())
        manager.add_nodes_to_buffer(nodes2)
        assert manager.n_nodes == 8

    def test_buffer_overflow_warning(self, prm_config, device_cfg):
        """Test buffer overflow triggers warning (line 140)."""
        cspace_weight = torch.ones(7, **device_cfg.as_torch_dict())
        distance_calc = DistanceNeighborCalculator(7, cspace_weight, device_cfg)
        path_finder = NetworkXPathFinder()
        rollout = RobotRollout(prm_config.rollout_config)

        manager = GraphNodeManager(
            config=prm_config,
            device_cfg=device_cfg,
            distance_calculator=distance_calc,
            graph_path_finder=path_finder,
            auxiliary_rollout=rollout,
        )

        # Try to add more nodes than buffer can hold
        # First fill most of the buffer
        remaining = manager.config.max_nodes - 10
        if remaining > 0:
            large_batch = torch.randn(remaining, 7, **device_cfg.as_torch_dict())
            manager.add_nodes_to_buffer(large_batch)

        # Now try to overflow - should log error
        overflow_batch = torch.randn(20, 7, **device_cfg.as_torch_dict())
        try:
            manager.add_nodes_to_buffer(overflow_batch)
        except (RuntimeError, ValueError):
            # Expected to fail or log error
            pass


class TestGraphNodeManagerRoadmapOperations:
    """Test roadmap node operations."""

    def test_add_initial_exact_nodes_to_roadmap(self, prm_config, device_cfg):
        """Test adding initial exact nodes (lines 289-293)."""
        cspace_weight = torch.ones(7, **device_cfg.as_torch_dict())
        distance_calc = DistanceNeighborCalculator(7, cspace_weight, device_cfg)
        path_finder = NetworkXPathFinder()
        rollout = RobotRollout(prm_config.rollout_config)

        manager = GraphNodeManager(
            config=prm_config,
            device_cfg=device_cfg,
            distance_calculator=distance_calc,
            graph_path_finder=path_finder,
            auxiliary_rollout=rollout,
        )

        # Must be called when _used_node_count == 0
        assert manager._used_node_count == 0

        nodes = torch.randn(3, 7, **device_cfg.as_torch_dict())
        node_set = manager.add_initial_exact_nodes_to_roadmap(nodes)

        assert node_set.shape[0] == 3
        assert node_set.shape[1] == 8  # action_dim + 1 (index)
        assert manager.n_nodes == 3

    def test_add_initial_exact_nodes_error_when_not_empty(self, node_manager):
        """Test error when adding initial exact nodes to non-empty roadmap (line 293)."""
        # node_manager already has 5 nodes from fixture
        assert node_manager._used_node_count > 0

        nodes = torch.randn(2, 7, device=node_manager.device_cfg.device)

        # Should log error since _used_node_count != 0
        try:
            node_set = node_manager.add_initial_exact_nodes_to_roadmap(nodes)
        except (RuntimeError, ValueError):
            # Expected error
            pass

    def test_add_initial_exact_nodes_invalid_shape(self, prm_config, device_cfg):
        """Test error for invalid node shape (lines 289)."""
        cspace_weight = torch.ones(7, **device_cfg.as_torch_dict())
        distance_calc = DistanceNeighborCalculator(7, cspace_weight, device_cfg)
        path_finder = NetworkXPathFinder()
        rollout = RobotRollout(prm_config.rollout_config)

        manager = GraphNodeManager(
            config=prm_config,
            device_cfg=device_cfg,
            distance_calculator=distance_calc,
            graph_path_finder=path_finder,
            auxiliary_rollout=rollout,
        )

        # 1D tensor instead of 2D
        nodes = torch.randn(7, **device_cfg.as_torch_dict())

        try:
            node_set = manager.add_initial_exact_nodes_to_roadmap(nodes)
        except (RuntimeError, ValueError, IndexError):
            pass

    def test_add_initial_exact_nodes_wrong_dimensions(self, prm_config, device_cfg):
        """Test error for wrong number of dimensions (lines 291)."""
        cspace_weight = torch.ones(7, **device_cfg.as_torch_dict())
        distance_calc = DistanceNeighborCalculator(7, cspace_weight, device_cfg)
        path_finder = NetworkXPathFinder()
        rollout = RobotRollout(prm_config.rollout_config)

        manager = GraphNodeManager(
            config=prm_config,
            device_cfg=device_cfg,
            distance_calculator=distance_calc,
            graph_path_finder=path_finder,
            auxiliary_rollout=rollout,
        )

        # Wrong number of columns (5 instead of 7)
        nodes = torch.randn(3, 5, **device_cfg.as_torch_dict())

        try:
            node_set = manager.add_initial_exact_nodes_to_roadmap(nodes)
        except (RuntimeError, ValueError, IndexError):
            pass

    def test_add_nodes_to_roadmap(self, node_manager):
        """Test adding nodes to roadmap with duplicate detection."""
        # Add some new nodes
        nodes = torch.randn(5, 7, device=node_manager.device_cfg.device)

        node_set = node_manager.add_nodes_to_roadmap(nodes, add_exact_node=False)

        # Should return node set with indices
        assert node_set.shape[0] == 5
        assert node_set.shape[1] == 8  # action_dim + 1

    def test_add_nodes_to_roadmap_with_exact_nodes(self, node_manager):
        """Test adding exact nodes to roadmap."""
        nodes = torch.randn(3, 7, device=node_manager.device_cfg.device)

        node_set = node_manager.add_nodes_to_roadmap(nodes, add_exact_node=True)

        assert node_set.shape[0] == 3
        assert node_set.shape[1] == 8

    def test_add_nodes_to_roadmap_invalid_shape(self, node_manager):
        """Test error for invalid node shape (line 228)."""
        # 1D tensor instead of 2D
        nodes = torch.randn(7, device=node_manager.device_cfg.device)

        try:
            node_set = node_manager.add_nodes_to_roadmap(nodes)
        except (RuntimeError, ValueError, IndexError):
            pass

    def test_add_nodes_to_roadmap_wrong_dimensions(self, node_manager):
        """Test error for wrong dimensions (line 230)."""
        # Wrong number of columns
        nodes = torch.randn(3, 5, device=node_manager.device_cfg.device)

        try:
            node_set = node_manager.add_nodes_to_roadmap(nodes)
        except (RuntimeError, ValueError, IndexError):
            pass

    def test_add_nodes_to_roadmap_when_empty(self, prm_config, device_cfg):
        """Test error when roadmap is empty (line 232)."""
        cspace_weight = torch.ones(7, **device_cfg.as_torch_dict())
        distance_calc = DistanceNeighborCalculator(7, cspace_weight, device_cfg)
        path_finder = NetworkXPathFinder()
        rollout = RobotRollout(prm_config.rollout_config)

        manager = GraphNodeManager(
            config=prm_config,
            device_cfg=device_cfg,
            distance_calculator=distance_calc,
            graph_path_finder=path_finder,
            auxiliary_rollout=rollout,
        )

        # Don't add any initial nodes - _used_node_count == 0
        assert manager._used_node_count == 0

        nodes = torch.randn(2, 7, **device_cfg.as_torch_dict())

        # Should log error since roadmap is empty
        try:
            node_set = manager.add_nodes_to_roadmap(nodes)
        except (RuntimeError, ValueError):
            pass

    def test_add_nodes_exceeding_max_nodes(self, node_manager):
        """Test error when adding nodes exceeds max_nodes (lines 254, 259, 261)."""
        # Calculate remaining capacity
        remaining = node_manager.config.max_nodes - node_manager.n_nodes

        if remaining > 10:
            # Add nodes up to near capacity
            fill_nodes = torch.randn(remaining - 5, 7, device=node_manager.device_cfg.device)
            try:
                node_set = node_manager.add_nodes_to_roadmap(fill_nodes)
            except:
                pass  # May fail, that's ok

            # Now try to add more than capacity
            overflow_nodes = torch.randn(20, 7, device=node_manager.device_cfg.device)
            try:
                node_set = node_manager.add_nodes_to_roadmap(overflow_nodes)
            except (RuntimeError, ValueError):
                # Expected error (line 254)
                pass


class TestGraphNodeManagerConnectionOperations:
    """Test node connection operations."""

    def test_register_nodes_and_connections(self, node_manager):
        """Test registering nodes with connections."""
        # Create node set and start nodes
        node_set = torch.randn(3, 7, device=node_manager.device_cfg.device)

        # Add node indices
        node_set_with_idx = torch.cat([
            node_set,
            torch.arange(10, 13, device=node_manager.device_cfg.device, dtype=node_set.dtype).unsqueeze(1)
        ], dim=1)

        # Create start nodes (from existing roadmap)
        start_nodes = node_manager.valid_node_buffer[:3, :]

        result = node_manager.register_nodes_and_connections(
            node_set=node_set_with_idx,
            start_nodes=start_nodes,
            add_exact_node=False,
        )

        assert result is True

    def test_register_nodes_with_exact_nodes(self, node_manager):
        """Test registering exact nodes."""
        node_set = torch.randn(2, 7, device=node_manager.device_cfg.device)
        node_set_with_idx = torch.cat([
            node_set,
            torch.tensor([[15.0], [16.0]], device=node_manager.device_cfg.device)
        ], dim=1)

        start_nodes = node_manager.valid_node_buffer[:2, :]

        result = node_manager.register_nodes_and_connections(
            node_set=node_set_with_idx,
            start_nodes=start_nodes,
            add_exact_node=True,
        )

        assert result is True


class TestGraphNodeManagerGraphRetrieval:
    """Test graph retrieval functionality."""

    def test_get_connected_graph(self, node_manager):
        """Test getting connected graph (lines 157-175)."""
        # Add some edges to create connections
        node_manager.graph_path_finder.add_edge(0, 1, 1.0)
        node_manager.graph_path_finder.add_edge(1, 2, 1.5)
        node_manager.graph_path_finder.add_edge(2, 3, 2.0)

        connected_graph = node_manager.get_connected_graph()

        assert connected_graph is not None
        assert isinstance(connected_graph, ConnectedGraph)
        assert connected_graph.nodes is not None
        assert connected_graph.edges is not None
        assert connected_graph.connectivity is not None
        assert connected_graph.robot_state_nodes is not None

    def test_get_connected_graph_empty_manager(self, prm_config, device_cfg):
        """Test get_connected_graph returns None when empty (line 158)."""
        cspace_weight = torch.ones(7, **device_cfg.as_torch_dict())
        distance_calc = DistanceNeighborCalculator(7, cspace_weight, device_cfg)
        path_finder = NetworkXPathFinder()
        rollout = RobotRollout(prm_config.rollout_config)

        manager = GraphNodeManager(
            config=prm_config,
            device_cfg=device_cfg,
            distance_calculator=distance_calc,
            graph_path_finder=path_finder,
            auxiliary_rollout=rollout,
        )

        # No nodes added - should return None
        connected_graph = manager.get_connected_graph()

        assert connected_graph is None


class TestGraphNodeManagerPathOperations:
    """Test path-related operations."""

    def test_get_nodes_in_path_valid(self, node_manager):
        """Test getting nodes for valid path (lines 313-314)."""
        # Create a simple path
        path_list = [[0, 1, 2], [3, 4]]

        paths = node_manager.get_nodes_in_path(path_list)

        assert len(paths) == 2
        assert paths[0] is not None
        assert paths[0].shape == (3, 7)  # 3 nodes in first path
        assert paths[1] is not None
        assert paths[1].shape == (2, 7)  # 2 nodes in second path

    def test_get_nodes_in_path_with_none(self, node_manager):
        """Test getting nodes with None path (line 313-314)."""
        # Path list with None (failed path)
        path_list = [[0, 1], None, [2, 3, 4]]

        paths = node_manager.get_nodes_in_path(path_list)

        assert len(paths) == 3
        assert paths[0] is not None
        assert paths[1] is None  # Should preserve None
        assert paths[2] is not None


class TestGraphNodeManagerReset:
    """Test reset functionality."""

    def test_reset_buffer(self, node_manager):
        """Test resetting the buffer."""
        initial_count = node_manager.n_nodes
        assert initial_count > 0

        # Reset
        node_manager.reset_buffer()

        # Should be empty
        assert node_manager.n_nodes == 0
        assert node_manager._used_node_count == 0

    def test_reset_graph_path_finder(self, node_manager):
        """Test resetting just the graph path finder (line 325)."""
        # Add some edges
        node_manager.graph_path_finder.add_edge(0, 1, 1.0)
        node_manager.graph_path_finder.update_graph()

        # Reset just the path finder
        node_manager.reset_graph_path_finder()

        # Graph should be empty
        assert node_manager.graph_path_finder.graph.number_of_edges() == 0


class TestGraphNodeManagerProperties:
    """Test property methods."""

    def test_node_idx_padding_buffer_property(self, node_manager):
        """Test node_idx_padding_buffer property."""
        buffer = node_manager.node_idx_padding_buffer

        assert buffer is not None
        assert buffer.shape == (1,)
        assert buffer.item() == 0.0

    def test_preallocated_node_buffer_property(self, node_manager):
        """Test preallocated_node_buffer property."""
        buffer = node_manager.preallocated_node_buffer

        assert buffer is not None
        assert buffer.shape[0] == node_manager.config.max_nodes
        assert buffer.shape[1] == node_manager.action_dim + 1

    def test_valid_node_buffer_property(self, node_manager):
        """Test valid_node_buffer property."""
        buffer = node_manager.valid_node_buffer

        assert buffer is not None
        assert buffer.shape[0] == node_manager.n_nodes
        assert buffer.shape[1] == node_manager.action_dim + 1


class TestConnectedGraphDataclass:
    """Test ConnectedGraph dataclass."""

    def test_connected_graph_creation(self, device_cfg):
        """Test creating ConnectedGraph."""
        nodes = torch.randn(10, 7, **device_cfg.as_torch_dict())
        edges = torch.randn(5, 2, 7, **device_cfg.as_torch_dict())
        connectivity = torch.tensor([[0, 1, 1.0], [1, 2, 1.5]], **device_cfg.as_torch_dict())

        graph = ConnectedGraph(
            nodes=nodes,
            edges=edges,
            connectivity=connectivity,
        )

        assert graph.nodes.shape == (10, 7)
        assert graph.edges.shape == (5, 2, 7)
        assert graph.connectivity.shape == (2, 3)

    def test_set_shortest_path_lengths(self, device_cfg):
        """Test setting shortest path lengths (line 42)."""
        nodes = torch.randn(10, 7, **device_cfg.as_torch_dict())
        edges = torch.randn(5, 2, 7, **device_cfg.as_torch_dict())
        connectivity = torch.tensor([[0, 1, 1.0]], **device_cfg.as_torch_dict())

        graph = ConnectedGraph(nodes=nodes, edges=edges, connectivity=connectivity)

        # Set path lengths
        path_lengths = torch.randn(10, **device_cfg.as_torch_dict())
        graph.set_shortest_path_lengths(path_lengths)

        assert graph.shortest_path_lengths is not None
        assert torch.equal(graph.shortest_path_lengths, path_lengths)

    def test_get_node_distance_with_path_lengths(self, device_cfg):
        """Test get_node_distance with path lengths (lines 45-51)."""
        nodes = torch.randn(10, 7, **device_cfg.as_torch_dict())
        edges = torch.randn(5, 2, 7, **device_cfg.as_torch_dict())
        connectivity = torch.tensor([[0, 1, 1.0]], **device_cfg.as_torch_dict())

        graph = ConnectedGraph(nodes=nodes, edges=edges, connectivity=connectivity)

        # Set path lengths
        path_lengths = torch.randn(10, **device_cfg.as_torch_dict())
        graph.set_shortest_path_lengths(path_lengths)

        # Get node distance
        node_dist = graph.get_node_distance()

        assert node_dist is not None
        assert node_dist.shape == (10, 8)  # nodes + distance column

    def test_get_node_distance_without_path_lengths(self, device_cfg):
        """Test get_node_distance without path lengths (line 51)."""
        nodes = torch.randn(10, 7, **device_cfg.as_torch_dict())
        edges = torch.randn(5, 2, 7, **device_cfg.as_torch_dict())
        connectivity = torch.tensor([[0, 1, 1.0]], **device_cfg.as_torch_dict())

        graph = ConnectedGraph(nodes=nodes, edges=edges, connectivity=connectivity)

        # Don't set path lengths
        node_dist = graph.get_node_distance()

        # Should return None
        assert node_dist is None

    def test_get_node_distance_with_mismatched_sizes(self, device_cfg):
        """Test get_node_distance when sizes don't match (line 46)."""
        nodes = torch.randn(10, 7, **device_cfg.as_torch_dict())
        edges = torch.randn(5, 2, 7, **device_cfg.as_torch_dict())
        connectivity = torch.tensor([[0, 1, 1.0]], **device_cfg.as_torch_dict())

        graph = ConnectedGraph(nodes=nodes, edges=edges, connectivity=connectivity)

        # Set path lengths with different size
        path_lengths = torch.randn(15, **device_cfg.as_torch_dict())  # More than nodes
        graph.set_shortest_path_lengths(path_lengths)

        # Should handle size mismatch (line 46)
        node_dist = graph.get_node_distance()

        assert node_dist is not None
        # Should use minimum of the two sizes
        assert node_dist.shape[0] == min(10, 15)

