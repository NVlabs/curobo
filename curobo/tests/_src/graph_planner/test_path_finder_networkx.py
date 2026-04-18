# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Comprehensive tests for NetworkXPathFinder."""

# Third Party
import pytest

# CuRobo
from curobo._src.graph_planner.search.path_finder_networkx import NetworkXPathFinder


@pytest.fixture
def path_finder():
    """Create a NetworkXPathFinder instance."""
    return NetworkXPathFinder(seed=42)


class TestNetworkXPathFinderInitialization:
    """Test NetworkXPathFinder initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        finder = NetworkXPathFinder()

        assert finder is not None
        assert finder.graph is not None
        assert finder.node_list == []
        assert finder.edge_list == []

    def test_init_with_seed(self):
        """Test initialization with specific seed."""
        finder = NetworkXPathFinder(seed=123)

        assert finder.seed == 123
        assert finder.graph is not None

    def test_reset_seed(self, path_finder):
        """Test resetting seed (lines 44-45)."""
        path_finder.reset_seed()

        # Seed should be reset in numpy and random
        # Verify by generating random numbers
        import random

        import numpy as np

        val1_np = np.random.rand()
        val1_rand = random.random()

        # Reset again
        path_finder.reset_seed()

        val2_np = np.random.rand()
        val2_rand = random.random()

        # Should produce same values
        assert val1_np == val2_np
        assert val1_rand == val2_rand


class TestNetworkXPathFinderNodeManagement:
    """Test node management functionality."""

    def test_add_node(self, path_finder):
        """Test adding a single node."""
        path_finder.add_node(0)

        assert 0 in path_finder.node_list
        assert len(path_finder.node_list) == 1

    def test_add_multiple_nodes_individually(self, path_finder):
        """Test adding multiple nodes one by one."""
        for i in range(5):
            path_finder.add_node(i)

        assert len(path_finder.node_list) == 5
        assert set(path_finder.node_list) == {0, 1, 2, 3, 4}

    def test_add_nodes_batch(self, path_finder):
        """Test adding nodes in batch (lines 56-57)."""
        nodes = [10, 11, 12, 13, 14]
        path_finder.add_nodes(nodes)

        assert len(path_finder.node_list) == 5
        assert all(node in path_finder.node_list for node in nodes)


class TestNetworkXPathFinderEdgeManagement:
    """Test edge management functionality."""

    def test_add_edge(self, path_finder):
        """Test adding a single edge (lines 60-61)."""
        path_finder.add_edge(0, 1, 1.5)

        assert len(path_finder.edge_list) == 1
        assert path_finder.edge_list[0] == [0, 1, 1.5]

    def test_add_edges_batch(self, path_finder):
        """Test adding edges in batch."""
        edges = [[0, 1, 1.0], [1, 2, 2.0], [2, 3, 1.5]]
        path_finder.add_edges(edges)

        assert len(path_finder.edge_list) == 3

    def test_update_graph_with_edges(self, path_finder):
        """Test updating graph with edges."""
        path_finder.add_node(0)
        path_finder.add_node(1)
        path_finder.add_edge(0, 1, 1.0)

        path_finder.update_graph()

        # After update, lists should be cleared
        assert len(path_finder.node_list) == 0
        assert len(path_finder.edge_list) == 0

        # Graph should have the edge
        assert path_finder.graph.has_edge(0, 1)

    def test_update_graph_empty_lists(self, path_finder):
        """Test updating graph with empty lists."""
        # Should not crash with empty lists
        path_finder.update_graph()

        assert len(path_finder.node_list) == 0
        assert len(path_finder.edge_list) == 0


class TestNetworkXPathFinderGraphOperations:
    """Test graph operations."""

    def test_get_edges(self, path_finder):
        """Test getting edges from graph (lines 74-75)."""
        # Build a simple graph
        path_finder.add_node(0)
        path_finder.add_node(1)
        path_finder.add_node(2)
        path_finder.add_edge(0, 1, 1.0)
        path_finder.add_edge(1, 2, 2.0)
        path_finder.update_graph()

        edges = path_finder.get_edges()

        assert len(edges) == 2
        assert all(len(edge) == 3 for edge in edges)  # (start, end, weight)

    def test_reset_graph(self, path_finder):
        """Test resetting the graph."""
        # Build a graph
        path_finder.add_node(0)
        path_finder.add_node(1)
        path_finder.add_edge(0, 1, 1.0)
        path_finder.update_graph()

        # Reset
        path_finder.reset_graph()

        # Graph should be empty
        assert path_finder.graph.number_of_nodes() == 0
        assert path_finder.graph.number_of_edges() == 0
        assert path_finder.node_list == []
        assert path_finder.edge_list == []


class TestNetworkXPathFinderPathSearch:
    """Test path finding functionality."""

    def test_path_exists_true(self, path_finder):
        """Test path_exists returns True when path exists."""
        # Build a simple path: 0 -> 1 -> 2
        path_finder.add_nodes([0, 1, 2])
        path_finder.add_edges([[0, 1, 1.0], [1, 2, 1.0]])

        exists = path_finder.path_exists(0, 2)

        assert exists is True

    def test_path_exists_false_disconnected(self, path_finder):
        """Test path_exists returns False for disconnected nodes."""
        # Build disconnected nodes
        path_finder.add_nodes([0, 1, 2, 3])
        path_finder.add_edge(0, 1, 1.0)
        path_finder.add_edge(2, 3, 1.0)
        # 0-1 is disconnected from 2-3

        exists = path_finder.path_exists(0, 3)

        assert exists is False

    def test_path_exists_node_not_in_graph(self, path_finder):
        """Test path_exists returns False when node doesn't exist (lines 84)."""
        path_finder.add_nodes([0, 1])
        path_finder.add_edge(0, 1, 1.0)

        # Node 5 doesn't exist
        exists = path_finder.path_exists(0, 5)

        assert exists is False

    def test_get_shortest_path_basic(self, path_finder):
        """Test getting shortest path."""
        # Build a simple graph
        path_finder.add_nodes([0, 1, 2, 3])
        path_finder.add_edges([
            [0, 1, 1.0],
            [1, 2, 1.0],
            [2, 3, 1.0],
        ])

        path = path_finder.get_shortest_path(0, 3)

        assert path == [0, 1, 2, 3]

    def test_get_shortest_path_with_length(self, path_finder):
        """Test getting shortest path with length (lines 92-93)."""
        path_finder.add_nodes([0, 1, 2])
        path_finder.add_edges([
            [0, 1, 2.0],
            [1, 2, 3.0],
        ])

        path, length = path_finder.get_shortest_path(0, 2, return_length=True)

        assert path == [0, 1, 2]
        assert length == 5.0  # 2.0 + 3.0

    def test_get_shortest_path_multiple_routes(self, path_finder):
        """Test shortest path selection with multiple routes."""
        # Create graph with two paths
        path_finder.add_nodes([0, 1, 2, 3])
        path_finder.add_edges([
            [0, 1, 1.0],
            [1, 3, 1.0],  # Direct: total = 2.0
            [0, 2, 5.0],
            [2, 3, 5.0],  # Indirect: total = 10.0
        ])

        path = path_finder.get_shortest_path(0, 3)

        # Should take the shorter path through node 1
        assert path == [0, 1, 3]

    def test_get_path_lengths(self, path_finder):
        """Test getting path lengths from goal (lines 98-116)."""
        # Build a graph
        path_finder.add_nodes([0, 1, 2, 3])
        path_finder.add_edges([
            [0, 1, 1.0],
            [1, 2, 2.0],
            [2, 3, 3.0],
        ])

        lengths = path_finder.get_path_lengths(goal_node_idx=3)

        # Should return list of lengths from each node to goal
        assert isinstance(lengths, list)
        assert len(lengths) >= 4
        assert lengths[3] == 0.0  # Goal to itself
        assert lengths[2] == 3.0  # Node 2 to goal
        assert lengths[1] == 5.0  # Node 1 to goal (2+3)
        assert lengths[0] == 6.0  # Node 0 to goal (1+2+3)

    def test_get_path_lengths_with_disconnected_nodes(self, path_finder):
        """Test get_path_lengths with disconnected nodes."""
        # Build graph with disconnected component
        path_finder.add_nodes([0, 1, 2, 3, 5])
        path_finder.add_edges([
            [0, 1, 1.0],
            [1, 2, 1.0],
            [2, 3, 1.0],
            # Node 5 is disconnected
        ])

        lengths = path_finder.get_path_lengths(goal_node_idx=3)

        # Should return list with length based on max node index + 1
        assert isinstance(lengths, list)
        assert len(lengths) >= 4
        # Connected nodes should have valid lengths
        assert lengths[3] == 0.0
        assert lengths[2] == 1.0
        # Disconnected node should have -1.0
        if len(lengths) > 5:
            assert lengths[5] == -1.0


class TestNetworkXPathFinderEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_graph_operations(self, path_finder):
        """Test operations on empty graph."""
        # path_exists on empty graph
        exists = path_finder.path_exists(0, 1)
        assert exists is False

        # get_edges on empty graph
        edges = path_finder.get_edges()
        assert edges == []

    def test_single_node_graph(self, path_finder):
        """Test graph with single node."""
        path_finder.add_node(0)
        path_finder.update_graph()

        # Path from node to itself
        exists = path_finder.path_exists(0, 0)
        assert exists is True

    def test_add_duplicate_nodes(self, path_finder):
        """Test adding duplicate nodes."""
        path_finder.add_node(0)
        path_finder.add_node(0)  # Duplicate
        path_finder.update_graph()

        # NetworkX should handle duplicates gracefully
        assert path_finder.graph.number_of_nodes() == 1

    def test_add_duplicate_edges(self, path_finder):
        """Test adding duplicate edges with different weights."""
        path_finder.add_nodes([0, 1])
        path_finder.add_edge(0, 1, 1.0)
        path_finder.add_edge(0, 1, 2.0)  # Duplicate with different weight
        path_finder.update_graph()

        # Should have the edge (NetworkX may update weight)
        assert path_finder.graph.has_edge(0, 1)

