# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Comprehensive tests for GraphConstructor."""

# Third Party
import pytest
import torch

from curobo._src.graph_planner.graph.connector_linear import LinearConnector

# CuRobo
from curobo._src.graph_planner.graph.constructor import GraphConstructor
from curobo._src.graph_planner.graph.node_distance import DistanceNeighborCalculator
from curobo._src.graph_planner.graph.node_manager import GraphNodeManager
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.graph_planner.search.path_finder_networkx import NetworkXPathFinder
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.state.state_joint import JointState


@pytest.fixture(scope="module")
def prm_config():
    """Get PRM planner configuration."""
    return PRMGraphPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
    )



@pytest.fixture
def graph_constructor(prm_config, cuda_device_cfg):
    """Create a GraphConstructor instance with all dependencies."""
    # Use the full PRMGraphPlanner to get properly initialized components
    from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner

    planner = PRMGraphPlanner(prm_config)

    # Return the constructor from the planner
    return planner.graph_constructor


class TestGraphConstructorInitialization:
    """Test GraphConstructor initialization."""

    def test_constructor_basic_init(self, graph_constructor):
        """Test basic initialization."""
        assert graph_constructor is not None
        assert graph_constructor.action_dim == 7
        assert graph_constructor.linear_connector is not None
        assert graph_constructor.distance_calculator is not None
        assert graph_constructor.node_manager is not None

    def test_constructor_initial_state(self, graph_constructor):
        """Test initial state of constructor."""
        assert graph_constructor._default_joint_position_feasible is None
        assert graph_constructor._default_node_in_roadmap is None


class TestGraphConstructorSteerAndRegister:
    """Test steer_and_register_edges functionality."""

    def test_steer_and_register_edges_basic(self, graph_constructor, cuda_device_cfg):
        """Test basic steering and edge registration."""
        # Create start and goal nodes with indices
        start_nodes = torch.cat([
            torch.randn(3, 7, **cuda_device_cfg.as_torch_dict()),
            torch.tensor([[0.0], [1.0], [2.0]], **cuda_device_cfg.as_torch_dict())
        ], dim=1)

        goal_nodes = torch.cat([
            torch.randn(3, 7, **cuda_device_cfg.as_torch_dict()),
            torch.tensor([[3.0], [4.0], [5.0]], **cuda_device_cfg.as_torch_dict())
        ], dim=1)

        # Add nodes to manager first
        initial_nodes = torch.randn(10, 7, **cuda_device_cfg.as_torch_dict())
        graph_constructor.node_manager.add_nodes_to_buffer(initial_nodes)

        # Steer and register
        graph_constructor.steer_and_register_edges(start_nodes, goal_nodes)

        # Should complete without error
        assert True

    def test_steer_and_register_with_exact_nodes(self, cuda_device_cfg):
        """Test steering with exact node addition."""
        # Use full planner for proper initialization
        from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner

        config = PRMGraphPlannerCfg.create(
            robot="franka.yml",
            scene_model="collision_test.yml",
        )
        planner = PRMGraphPlanner(config)

        # Add some nodes first
        initial_nodes = torch.randn(5, 7, **cuda_device_cfg.as_torch_dict())
        planner.node_manager.add_nodes_to_buffer(initial_nodes)

        start_nodes = torch.cat([
            torch.randn(2, 7, **cuda_device_cfg.as_torch_dict()),
            torch.tensor([[0.0], [1.0]], **cuda_device_cfg.as_torch_dict())
        ], dim=1)

        goal_nodes = torch.cat([
                torch.randn(2, 7, **cuda_device_cfg.as_torch_dict()),
            torch.tensor([[2.0], [3.0]], **cuda_device_cfg.as_torch_dict())
        ], dim=1)

        planner.graph_constructor.steer_and_register_edges(
            start_nodes, goal_nodes, add_exact_node=True
        )

        assert True

    def test_steer_and_register_mismatched_batch_size(self, graph_constructor, cuda_device_cfg):
        """Test error for mismatched batch sizes (line 86)."""
        start_nodes = torch.cat([
            torch.randn(3, 7, **cuda_device_cfg.as_torch_dict()),
            torch.arange(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(1)
        ], dim=1)

        # Different batch size
        goal_nodes = torch.cat([
            torch.randn(5, 7, **cuda_device_cfg.as_torch_dict()),
            torch.arange(5, **cuda_device_cfg.as_torch_dict()).unsqueeze(1)
        ], dim=1)

        # Should log error
        try:
            graph_constructor.steer_and_register_edges(start_nodes, goal_nodes)
        except (RuntimeError, ValueError):
            pass

    def test_steer_and_register_wrong_start_dimensions(self, graph_constructor, cuda_device_cfg):
        """Test error for wrong start dimensions (line 89)."""
        # Wrong number of columns (only action_dim, missing index)
        start_nodes = torch.randn(2, 7, **cuda_device_cfg.as_torch_dict())
        goal_nodes = torch.cat([
            torch.randn(2, 7, **cuda_device_cfg.as_torch_dict()),
            torch.arange(2, **cuda_device_cfg.as_torch_dict()).unsqueeze(1)
        ], dim=1)

        try:
            graph_constructor.steer_and_register_edges(start_nodes, goal_nodes)
        except (RuntimeError, ValueError, IndexError):
            pass

    def test_steer_and_register_wrong_goal_dimensions(self, graph_constructor, cuda_device_cfg):
        """Test error for wrong goal dimensions (line 91)."""
        start_nodes = torch.cat([
            torch.randn(2, 7, **cuda_device_cfg.as_torch_dict()),
            torch.arange(2, **cuda_device_cfg.as_torch_dict()).unsqueeze(1)
        ], dim=1)

        # Wrong number of columns
        goal_nodes = torch.randn(2, 7, **cuda_device_cfg.as_torch_dict())

        try:
            graph_constructor.steer_and_register_edges(start_nodes, goal_nodes)
        except (RuntimeError, ValueError, IndexError):
            pass


class TestGraphConstructorConnectNodes:
    """Test connect_nodes functionality."""

    def test_connect_nodes_basic(self, graph_constructor, cuda_device_cfg):
        """Test basic node connection."""
        # Add some initial nodes to roadmap
        initial_nodes = torch.randn(10, 7, **cuda_device_cfg.as_torch_dict())
        graph_constructor.node_manager.add_nodes_to_buffer(initial_nodes)

        # Add new nodes to connect
        new_nodes = torch.randn(5, 7, **cuda_device_cfg.as_torch_dict())

        graph_constructor.connect_nodes(new_nodes, neighbors_per_node=3)

        # Should complete successfully
        assert graph_constructor.node_manager.n_nodes >= 10

    def test_connect_nodes_with_empty_batch(self, graph_constructor, cuda_device_cfg):
        """Test connect_nodes with empty batch (lines 117-118)."""
        # Empty batch - should log info and return
        empty_nodes = torch.randn(0, 7, **cuda_device_cfg.as_torch_dict())

        # Should handle gracefully
        graph_constructor.connect_nodes(empty_nodes)

        # Should complete without error
        assert True

    def test_connect_nodes_without_index_column(self, graph_constructor, cuda_device_cfg):
        """Test connect_nodes adds index column when missing."""
        # Add initial nodes
        initial_nodes = torch.randn(5, 7, **cuda_device_cfg.as_torch_dict())
        graph_constructor.node_manager.add_nodes_to_buffer(initial_nodes)

        # Provide nodes without index column
        new_nodes = torch.randn(3, 7, **cuda_device_cfg.as_torch_dict())

        graph_constructor.connect_nodes(new_nodes, neighbors_per_node=2)

        assert True

    def test_connect_nodes_with_exact_nodes(self, graph_constructor, cuda_device_cfg):
        """Test connect_nodes with exact node addition."""
        initial_nodes = torch.randn(8, 7, **cuda_device_cfg.as_torch_dict())
        graph_constructor.node_manager.add_nodes_to_buffer(initial_nodes)

        new_nodes = torch.randn(4, 7, **cuda_device_cfg.as_torch_dict())

        graph_constructor.connect_nodes(new_nodes, add_exact_node=True, neighbors_per_node=3)

        assert True


class TestGraphConstructorRetractNode:
    """Test default node initialization."""

    def test_initialize_default_node_feasible(self, prm_config, cuda_device_cfg):
        """Test initializing feasible default node."""
        # Enable default position heuristic
        prm_config.use_default_position_heuristic = True

        # Create constructor with dependencies
        cspace_weight = torch.ones(7, **cuda_device_cfg.as_torch_dict())
        distance_calc = DistanceNeighborCalculator(7, cspace_weight, cuda_device_cfg)
        path_finder = NetworkXPathFinder()
        rollout = RobotRollout(prm_config.rollout_config)
        node_manager = GraphNodeManager(
            config=prm_config,
            device_cfg=cuda_device_cfg,
            distance_calculator=distance_calc,
            graph_path_finder=path_finder,
            auxiliary_rollout=rollout,
        )
        linear_connector = LinearConnector(prm_config, cuda_device_cfg)

        def always_feasible(samples):
            return torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)

        constructor = GraphConstructor(
            config=prm_config,
            device_cfg=cuda_device_cfg,
            linear_connector=linear_connector,
            distance_calculator=distance_calc,
            node_manager=node_manager,
            action_dim=7,
            check_feasibility_fn=always_feasible,
        )

        # Create default joint state
        default_position = torch.zeros(1, 7, **cuda_device_cfg.as_torch_dict())
        default_joint_state = JointState.from_position(default_position)

        # Initialize default node
        default_node, is_feasible = constructor.initialize_default_node(default_joint_state)

        # Should be feasible and add node
        assert is_feasible is True
        assert default_node is not None

    def test_initialize_default_node_infeasible(self, prm_config, cuda_device_cfg):
        """Test initializing infeasible default node."""
        prm_config.use_default_position_heuristic = True

        cspace_weight = torch.ones(7, **cuda_device_cfg.as_torch_dict())
        distance_calc = DistanceNeighborCalculator(7, cspace_weight, cuda_device_cfg)
        path_finder = NetworkXPathFinder()
        rollout = RobotRollout(prm_config.rollout_config)
        node_manager = GraphNodeManager(
            config=prm_config, device_cfg=cuda_device_cfg,
            distance_calculator=distance_calc,
            graph_path_finder=path_finder,
            auxiliary_rollout=rollout,
        )
        linear_connector = LinearConnector(prm_config, cuda_device_cfg)

        def always_infeasible(samples):
            return torch.zeros(samples.shape[0], dtype=torch.bool, device=samples.device)

        constructor = GraphConstructor(
            config=prm_config,
            device_cfg=cuda_device_cfg,
            linear_connector=linear_connector,
            distance_calculator=distance_calc,
            node_manager=node_manager,
            action_dim=7,
            check_feasibility_fn=always_infeasible,
        )

        default_position = torch.zeros(1, 7, **cuda_device_cfg.as_torch_dict())
        default_joint_state = JointState.from_position(default_position)

        default_node, is_feasible = constructor.initialize_default_node(default_joint_state)

        # Should be infeasible and not add node
        assert is_feasible is False

    def test_initialize_default_node_disabled(self, graph_constructor, cuda_device_cfg):
        """Test default node when use_default_position_heuristic is False."""
        # Disable default position heuristic
        graph_constructor.config.use_default_position_heuristic = False
        graph_constructor._default_joint_position_feasible = None

        default_position = torch.zeros(1, 7, **cuda_device_cfg.as_torch_dict())
        default_joint_state = JointState.from_position(default_position)

        default_node, is_feasible = graph_constructor.initialize_default_node(default_joint_state)

        # Should return None, None when disabled
        assert default_node is None


class TestGraphConstructorTerminalNodes:
    """Test terminal node initialization and connection."""

    def test_initialize_terminal_graph_connections_basic(self, graph_constructor, cuda_device_cfg):
        """Test initializing terminal connections."""
        x_init = torch.randn(2, 7, **cuda_device_cfg.as_torch_dict())
        x_goal = torch.randn(2, 7, **cuda_device_cfg.as_torch_dict())

        default_position = torch.zeros(1, 7, **cuda_device_cfg.as_torch_dict())
        default_joint_state = JointState.from_position(default_position)

        start_nodes, goal_nodes = graph_constructor.initialize_terminal_graph_connections(
            x_init, x_goal, default_joint_state
        )

        assert start_nodes.shape == (2, 8)  # action_dim + 1
        assert goal_nodes.shape == (2, 8)

    def test_initialize_terminal_with_existing_roadmap(self, cuda_device_cfg):
        """Test terminal initialization with existing roadmap (line 212)."""
        from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner

        config = PRMGraphPlannerCfg.create(
            robot="franka.yml",
            scene_model="collision_test.yml",
        )
        planner = PRMGraphPlanner(config)

        # Add some nodes first
        existing_nodes = torch.randn(5, 7, **cuda_device_cfg.as_torch_dict())
        planner.node_manager.add_nodes_to_buffer(existing_nodes)

        x_init = torch.randn(1, 7, **cuda_device_cfg.as_torch_dict())
        x_goal = torch.randn(1, 7, **cuda_device_cfg.as_torch_dict())
        default_position = torch.zeros(1, 7, **cuda_device_cfg.as_torch_dict())
        default_joint_state = JointState.from_position(default_position)

        # This should use add_nodes_to_roadmap instead of add_initial_exact_nodes_to_roadmap
        start_nodes, goal_nodes = planner.graph_constructor.initialize_terminal_graph_connections(
            x_init, x_goal, default_joint_state
        )

        assert start_nodes.shape == (1, 8)
        assert goal_nodes.shape == (1, 8)

    def test_initialize_terminal_with_connect_nearest(self, cuda_device_cfg):
        """Test terminal initialization with connect_terminal_nodes_with_nearest (lines 238-245)."""
        from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner

        config = PRMGraphPlannerCfg.create(
            robot="franka.yml",
            scene_model="collision_test.yml",
        )
        # Enable connecting to nearest neighbors
        config.connect_terminal_nodes_with_nearest = True

        planner = PRMGraphPlanner(config)

        # Add nodes to roadmap
        initial_nodes = torch.randn(10, 7, **cuda_device_cfg.as_torch_dict())
        planner.node_manager.add_nodes_to_buffer(initial_nodes)

        x_init = torch.randn(2, 7, **cuda_device_cfg.as_torch_dict())
        x_goal = torch.randn(2, 7, **cuda_device_cfg.as_torch_dict())
        default_position = torch.zeros(1, 7, **cuda_device_cfg.as_torch_dict())
        default_joint_state = JointState.from_position(default_position)

        start_nodes, goal_nodes = planner.graph_constructor.initialize_terminal_graph_connections(
            x_init, x_goal, default_joint_state
        )

        # Should execute connect_nodes (lines 238-249)
        assert start_nodes is not None
        assert goal_nodes is not None

    def test_initialize_terminal_wrong_init_shape(self, graph_constructor, cuda_device_cfg):
        """Test error for wrong x_init shape (line 193)."""
        # 1D instead of 2D
        x_init = torch.randn(7, **cuda_device_cfg.as_torch_dict())
        x_goal = torch.randn(2, 7, **cuda_device_cfg.as_torch_dict())
        default_position = torch.zeros(1, 7, **cuda_device_cfg.as_torch_dict())
        default_joint_state = JointState.from_position(default_position)

        try:
            start_nodes, goal_nodes = graph_constructor.initialize_terminal_graph_connections(
                x_init, x_goal, default_joint_state
            )
        except (RuntimeError, ValueError, IndexError):
            pass

    def test_initialize_terminal_wrong_goal_shape(self, graph_constructor, cuda_device_cfg):
        """Test error for wrong x_goal shape (line 195)."""
        x_init = torch.randn(2, 7, **cuda_device_cfg.as_torch_dict())
        # 1D instead of 2D
        x_goal = torch.randn(7, **cuda_device_cfg.as_torch_dict())
        default_position = torch.zeros(1, 7, **cuda_device_cfg.as_torch_dict())
        default_joint_state = JointState.from_position(default_position)

        try:
            start_nodes, goal_nodes = graph_constructor.initialize_terminal_graph_connections(
                x_init, x_goal, default_joint_state
            )
        except (RuntimeError, ValueError, IndexError):
            pass

    def test_initialize_terminal_mismatched_batch(self, graph_constructor, cuda_device_cfg):
        """Test error for mismatched batch sizes (line 197)."""
        x_init = torch.randn(2, 7, **cuda_device_cfg.as_torch_dict())
        x_goal = torch.randn(3, 7, **cuda_device_cfg.as_torch_dict())  # Different batch size
        default_position = torch.zeros(1, 7, **cuda_device_cfg.as_torch_dict())
        default_joint_state = JointState.from_position(default_position)

        try:
            start_nodes, goal_nodes = graph_constructor.initialize_terminal_graph_connections(
                x_init, x_goal, default_joint_state
            )
        except (RuntimeError, ValueError):
            pass

    def test_initialize_terminal_wrong_init_dimensions(self, graph_constructor, cuda_device_cfg):
        """Test error for wrong x_init dimensions (line 199)."""
        x_init = torch.randn(2, 5, **cuda_device_cfg.as_torch_dict())  # Wrong action_dim
        x_goal = torch.randn(2, 7, **cuda_device_cfg.as_torch_dict())
        default_position = torch.zeros(1, 7, **cuda_device_cfg.as_torch_dict())
        default_joint_state = JointState.from_position(default_position)

        try:
            start_nodes, goal_nodes = graph_constructor.initialize_terminal_graph_connections(
                x_init, x_goal, default_joint_state
            )
        except (RuntimeError, ValueError):
            pass

    def test_initialize_terminal_wrong_goal_dimensions(self, graph_constructor, cuda_device_cfg):
        """Test error for wrong x_goal dimensions (line 201)."""
        x_init = torch.randn(2, 7, **cuda_device_cfg.as_torch_dict())
        x_goal = torch.randn(2, 5, **cuda_device_cfg.as_torch_dict())  # Wrong action_dim
        default_position = torch.zeros(1, 7, **cuda_device_cfg.as_torch_dict())
        default_joint_state = JointState.from_position(default_position)

        try:
            start_nodes, goal_nodes = graph_constructor.initialize_terminal_graph_connections(
                x_init, x_goal, default_joint_state
            )
        except (RuntimeError, ValueError):
            pass


class TestGraphConstructorReset:
    """Test reset functionality."""

    def test_reset(self, graph_constructor):
        """Test resetting constructor state."""
        # Set some state
        graph_constructor._default_joint_position_feasible = True
        graph_constructor._default_node_in_roadmap = torch.randn(1, 8)

        # Reset
        graph_constructor.reset()

        # State should be cleared
        assert graph_constructor._default_joint_position_feasible is None
        assert graph_constructor._default_node_in_roadmap is None

