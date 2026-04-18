# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union

# CuRobo
from curobo._src.geom.collision import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.config_io import join_path, resolve_config
from curobo.content import (
    get_robot_configs_path,
    get_scene_configs_path,
    get_task_configs_path,
)


@dataclass
class PRMGraphPlannerCfg:
    #: Maximum number of nodes in the graph
    max_nodes: int

    #: Maximum number of points to check for feasibility per call. When query is larger than this,
    #: it is split into multiple calls.
    feasibility_buffer_size: int

    #: Maximum number of points allowed to steer between two nodes
    steer_buffer_size: int

    #: Maximum radius to sample around the linear path between start and goal.
    #: Similar to c_max in BIT*.
    exploration_radius: float

    #: Number of nodes to sample per iteration
    new_nodes_per_iteration: int

    #: Maximum number of iterations to find a path
    max_path_finding_iterations: int

    #: Minimum number of iterations to finetune path
    min_finetune_iterations: int

    #: Whether to connect the start and goal through the default joint position as a heuristic.
    use_default_position_heuristic: bool

    #: Threshold for similarity in cspace distance.
    cspace_similarity_threshold: float

    #: Node sample rejection ratio. Should be greater than 1.
    sample_rejection_ratio: int

    #: Number of nearest neighbors to steer from each sampled node to
    #: connect to existing nodes in graph.
    neighbors_per_node: int

    #: Configuration for the rollout function.
    rollout_config: RobotRolloutCfg

    #: Seed for the node sampler.
    sampler_seed: int

    #: Number of samples to store in the node sampler buffer.
    sampler_buffer_size: int

    #: Whether to use cuda graph.
    use_cuda_graph_for_rollout: bool

    #: Whether to connect start and goal nodes with nearest other start and goal nodes in graph
    #: during initially adding nodes to the graph. Set this to True if performance is poor.
    connect_terminal_nodes_with_nearest: bool

    #: Exploration radius growth factor. Should be greater than 1.
    exploration_radius_growth_factor: float

    #: Nearest neighbors growth factor. Should be greater than 1.
    neighbors_per_node_growth_factor: float

    #: New nodes per iteration growth factor. Should be greater than 1.
    new_nodes_per_iteration_growth_factor: float

    #: Ellipsoid projection method to use. Choose from "svd", "householder", "approximate"
    ellipsoid_projection_method: str = "householder"

    #: Configuration for the world collision checker.
    scene_collision_cfg: Optional[SceneCollisionCfg] = None

    #: Tensor device configuration..
    device_cfg: DeviceCfg = DeviceCfg()

    graph_path_finder_seed: int = 42

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        graph_planner_config: Union[str, Dict[str, Any]] = "graph_planner/exact_graph_planner.yml",
        rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "graph_planner/transition_graph_planner.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        use_cuda_graph_for_rollout: bool = True,
        transition_model_config_instance_type: Type[
            RobotStateTransitionCfg
        ] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        graph_path_finder_seed: int = 42,
    ) -> "PRMGraphPlannerCfg":
        """Create PRMGraphPlannerCfg from flexible inputs.

        Each config parameter accepts either a file path (str), a dictionary,
        or an already-constructed config object.

        Args:
            robot: Robot configuration - path to YAML, dict, or RobotCfg object.
            graph_planner_config: Graph planner config - path or dict.
            rollout: Rollout config - path or dict.
            transition_model: Transition model config - path or dict.
            scene_model: Optional scene model config - path or dict.
            collision_cache: Cache configuration for collision checking.
            self_collision_check: Whether to check self-collision.
            device_cfg: Device configuration.
            use_cuda_graph_for_rollout: Whether to use CUDA graphs for rollout.
            transition_model_config_instance_type: Transition model config class.
            cost_manager_config_instance_type: Cost manager config class.
            graph_path_finder_seed: Seed for graph path finder.

        Returns:
            Configured PRMGraphPlannerCfg instance.
        """
        # Resolve robot config
        robot_resolved = resolve_config(join_path(get_robot_configs_path(), robot))
        if isinstance(robot_resolved, dict):
            robot_config = RobotCfg.create(robot_resolved, device_cfg)
        else:
            robot_config = robot_resolved

        # Resolve scene model
        if scene_model is not None:
            scene_model_dict = resolve_config(join_path(get_scene_configs_path(), scene_model))
        else:
            scene_model_dict = None

        # Resolve other configs
        graph_planner_config_dict = resolve_config(
            join_path(get_task_configs_path(), graph_planner_config)
        )
        rollout_config_dict = resolve_config(join_path(get_task_configs_path(), rollout))
        transition_model_dict = resolve_config(
            join_path(get_task_configs_path(), transition_model)
        )

        # Create scene collision config
        scene_collision_cfg = None
        if scene_model_dict is not None:
            # Parse scene config
            if isinstance(scene_model_dict, list):
                scene_cfg_obj = [SceneCfg.create(x) for x in scene_model_dict]
            elif isinstance(scene_model_dict, dict):
                scene_cfg_obj = SceneCfg.create(scene_model_dict)
            else:
                scene_cfg_obj = scene_model_dict

            scene_collision_cfg = SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=scene_cfg_obj,
                cache=collision_cache,
            )

        # Process rollout config
        rollout_dict = rollout_config_dict["rollout"]
        rollout_dict["transition_model_cfg"] = transition_model_dict["transition_model_cfg"]

        rollout_config = RobotRolloutCfg.create_with_component_types(
            data_dict=rollout_dict,
            robot_cfg=robot_config,
            device_cfg=device_cfg,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
        )

        # Extract graph planner parameters
        graph_params = graph_planner_config_dict["graph_planner"]

        return PRMGraphPlannerCfg(
            **graph_params,
            device_cfg=device_cfg,
            scene_collision_cfg=scene_collision_cfg,
            use_cuda_graph_for_rollout=use_cuda_graph_for_rollout,
            rollout_config=rollout_config,
            graph_path_finder_seed=graph_path_finder_seed,
        )
