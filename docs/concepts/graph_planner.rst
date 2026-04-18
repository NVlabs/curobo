Graph Planner
==============

Overview
---------

The CuRobo Graph Planner implements a Probabilistic Roadmap (PRM) based motion planning algorithm that leverages GPU acceleration for high-performance robot motion planning. It constructs a graph-based representation of the robot's configuration space and finds feasible paths between start and goal configurations.


.. graphviz::

   digraph architecture {
       rankdir=TB;
       edge [color = "#2B4162", fontsize=10];
       node [shape="box", style="rounded, filled", fontsize=12, color="#cccccc"];

       PRMGraphPlanner [label="PRMGraphPlanner\n(Main Coordinator)", color="#76b900", fontcolor="white"];

       GraphConstructor [label="GraphConstructor\n(Graph Building)", color="#558c8c", fontcolor="white"];
       NodeManager [label="GraphNodeManager\n(Node Storage)", color="#558c8c", fontcolor="white"];
       NodeSampling [label="NodeSamplingStrategy\n(Configuration Sampling)", color="#558c8c", fontcolor="white"];
       DistanceCalculator [label="DistanceNeighborCalculator\n(C-Space Metrics)", color="#558c8c", fontcolor="white"];
       LinearConnector [label="LinearConnector\n(Edge Creation)", color="#558c8c", fontcolor="white"];
       PathFinder [label="NetworkXPathFinder\n(Graph Search)", color="#558c8c", fontcolor="white"];
       PathPruner [label="PathPruner\n(Path Pruning)", color="#558c8c", fontcolor="white"];

       PRMGraphPlanner -> GraphConstructor;
       PRMGraphPlanner -> NodeManager;
       PRMGraphPlanner -> NodeSampling;
       PRMGraphPlanner -> PathFinder;
       PRMGraphPlanner -> PathPruner;

       GraphConstructor -> NodeManager;
       GraphConstructor -> LinearConnector;
       GraphConstructor -> DistanceCalculator;

       NodeManager -> PathFinder;

       subgraph cluster_components {
           style="rounded";
           color="#558c8c";
           label="Core Components";
           fontsize=12;
           GraphConstructor; NodeManager; NodeSampling;
           LinearConnector; DistanceCalculator;
       }

       subgraph cluster_search {
           style="rounded";
           color="#2B4162";
           label="Path Search & Pruning";
           fontsize=12;
           PathFinder; PathPruner;
       }
   }

Key Components
--------------

:class:`~curobo._src.graph_planner.graph_planner_prm.PRMGraphPlanner` (Main Coordinator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~curobo._src.graph_planner.graph_planner_prm.PRMGraphPlanner` class serves as the primary entry point and coordinates all planning activities:

- Initializes and manages all subcomponents
- Exposes the main path finding interface
- Handles roadmap extension and maintenance
- Integrates with CuRobo's rollout-based feasibility checking

:class:`~curobo._src.graph_planner.graph.constructor.GraphConstructor`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~curobo._src.graph_planner.graph.constructor.GraphConstructor` handles building and maintaining the graph structure:

- Manages node addition and edge creation
- Coordinates connection of terminal nodes (start/goal)
- Handles special cases like retraction nodes
- Directs the steering process between configurations

:class:`~curobo._src.graph_planner.graph.node_sampling_strategy.NodeSamplingStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This component specializes in generating feasible robot configurations:

- Provides uniform sampling within joint limits
- Implements informed sampling using ellipsoids (similar to BIT*)
- Supports multiple ellipsoid projection methods (SVD, Householder, approximate)
- Filters out collision configurations
- Uses Halton sequences for efficient low-discrepancy sampling

:class:`~curobo._src.graph_planner.graph.node_manager.GraphNodeManager`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~curobo._src.graph_planner.graph.node_manager.GraphNodeManager` handles the storage and organization of graph nodes:

- Maintains buffers of valid nodes and their connections
- Tracks node indices and their relationships
- Manages node registration and edge creation
- Provides efficient access to graph structure

:class:`~curobo._src.graph_planner.graph.connector_linear.LinearConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~curobo._src.graph_planner.graph.connector_linear.LinearConnector` creates edges between nodes by:

- Implementing local steering between configurations
- Checking feasibility of connections
- Generating intermediate configurations along edges
- Enforcing joint limits during steering

:class:`~curobo._src.graph_planner.graph.node_distance.DistanceNeighborCalculator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This component computes distances in configuration space:

- Uses weighted Euclidean distance in joint space
- Finds nearest neighbors efficiently
- Supports custom distance metrics with joint weights

:class:`~curobo._src.graph_planner.search.path_finder_networkx.NetworkXPathFinder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~curobo._src.graph_planner.search.path_finder_networkx.NetworkXPathFinder` uses NetworkX for graph search:

- Wraps NetworkX for path finding on the roadmap
- Provides shortest path computation using Dijkstra's algorithm
- Checks path existence between nodes
- Computes path lengths between nodes

:class:`~curobo._src.graph_planner.search.path_pruner.PathPruner`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~curobo._src.graph_planner.search.path_pruner.PathPruner` optimizes the found paths:

- Simplifies paths by removing unnecessary waypoints
- Shortens paths by attempting direct connections
- Preserves path feasibility during optimization

Planning Workflow
-----------------

The planning process follows these main steps:

1. **Initialization**: Configure components and prepare internal buffers
2. **Graph Construction**:
   - Add start and goal nodes to the roadmap
   - Sample feasible configurations using the sampling strategy
   - Connect configurations using the linear connector
   - Register nodes and edges in the graph
3. **Path Finding**:
   - Search for a path using NetworkX
   - If no path exists, extend the roadmap with more samples
   - Use informed sampling to focus on promising regions
4. **Path Optimization**:
   - Prune unnecessary waypoints
   - Attempt shortcutting to simplify the path
5. **Interpolation**:
   - Generate a smooth, densely sampled trajectory from the path

GPU Acceleration
----------------

The implementation leverages GPU acceleration through:

- TorchScript JIT compilation for performance-critical functions
- CUDA graphs for optimized, repeated execution of the same operations
- Batched operations for efficient parallel computation
- Tensor-based storage for graph nodes and connections

Configuration
-------------

The planner is highly configurable through the :class:`~curobo._src.graph_planner.graph_planner_prm_cfg.PRMGraphPlannerCfg` class:

- Memory limits for node storage
- Sampling parameters
- Search iterations and heuristics
- Path optimization settings
- Buffer sizes for batched operations

Performance Considerations
--------------------------

Several strategies improve performance:

- Profiling decorators track component performance
- Batched collision checking minimizes CPU-GPU transfers
- Optimized nearest neighbor search
- JIT-compiled transformation functions
- Multiple ellipsoid projection methods with different performance characteristics

Key Parameters
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``max_nodes``
     - Maximum number of nodes in the graph
   * - ``feasibility_buffer_size``
     - Maximum points to check for feasibility per batch
   * - ``steer_buffer_size``
     - Maximum points allowed between two nodes when steering
   * - ``exploration_radius``
     - Maximum radius to sample around the linear path
   * - ``new_nodes_per_iteration``
     - Number of nodes to sample per iteration
   * - ``max_path_finding_iterations``
     - Maximum iterations to find a path
   * - ``min_finetune_iterations``
     - Minimum iterations for path optimization
   * - ``use_default_position_heuristic``
     - Whether to connect through a retracted configuration
   * - ``neighbors_per_node``
     - Number of nearest neighbors to connect to

Usage Example
-------------

.. code-block:: python

   # Import required modules
   from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
   from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner
   import torch

   # Initialize the graph planner with a robot and world configuration
   robot_file = "franka.yml"
   world_file = "collision_thin_walls.yml"

   base_graph_planner_config = PRMGraphPlannerCfg.create(
       robot=robot_file,
       scene_model=world_file,
   )

   planner = PRMGraphPlanner(base_graph_planner_config)

   # Optionally warm up the planner for better performance
   planner.warmup()

   # Generate feasible start and goal configurations
   samples = planner.sampling_strategy.generate_feasible_action_samples(2)
   x_start = samples[0:1, :]  # Start configuration
   x_goal = samples[1:2, :]   # Goal configuration

   # Find a path between start and goal
   result = planner.find_path(
       x_start=x_start,
       x_goal=x_goal,
   )

   # Extract the trajectory if planning was successful
   if result.success[0]:
       path = result.plan[0]
       # Use the path for robot execution

Integration with CuRobo
-----------------------

The graph planner integrates with the broader CuRobo framework:

- Uses CuRobo's collision checking for feasibility testing
- Leverages the robot model for kinematics
- Relies on rollout-based feasibility evaluation
- Integrates with existing trajectory representations

Implementation Details
----------------------

Memory Management
~~~~~~~~~~~~~~~~~

The implementation carefully manages memory through:

- Pre-allocated buffers for nodes and edges
- Configurable buffer sizes
- Efficient tensor operations to minimize allocations
- Reuse of computation buffers when possible

Informed Sampling
~~~~~~~~~~~~~~~~~

The planner implements informed sampling strategies to focus computation in promising regions:

- Initial uniform sampling to explore the space
- Informed ellipsoidal sampling that focuses around the direct path
- Progressive radius reduction to refine the solution

Path Optimization
~~~~~~~~~~~~~~~~~

Once a path is found, it undergoes several optimization steps:

- Shortcutting to remove unnecessary waypoints
- Collision checking of optimized segments
- Smoothing to improve trajectory quality
- Dense interpolation to ensure smooth execution

Use Cases
---------

The PRM planner is particularly effective for:

- High-dimensional configuration spaces
- Problems requiring a reusable roadmap
- Multi-query planning scenarios
- Applications where preprocessing can amortize online planning costs
