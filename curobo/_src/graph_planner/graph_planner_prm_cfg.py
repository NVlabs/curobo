# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union

# CuRobo
from curobo._src.graph_planner.config_graph_planner_resources import GraphPlannerResourcesCfg
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.config_io import join_path, resolve_config
from curobo.content import get_task_configs_path

_RESOURCE_PARAMETER_NAMES = {
    "feasibility_buffer_size",
    "steer_buffer_size",
    "cspace_similarity_threshold",
    "sample_rejection_ratio",
    "sampler_seed",
}


@dataclass
class PRMGraphPlannerCfg:
    #: Runtime resources shared with other graph-planning algorithms.
    resources_config: GraphPlannerResourcesCfg

    #: Maximum number of nodes in the graph
    max_nodes: int

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

    #: Number of nearest neighbors to steer from each sampled node to
    #: connect to existing nodes in graph.
    neighbors_per_node: int

    #: Number of samples to store in the node sampler buffer.
    sampler_buffer_size: int

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

    graph_path_finder_seed: int = 42

    @property
    def feasibility_buffer_size(self) -> int:
        return self.resources_config.feasibility_buffer_size

    @property
    def steer_buffer_size(self) -> int:
        return self.resources_config.steer_buffer_size

    @property
    def cspace_similarity_threshold(self) -> float:
        return self.resources_config.cspace_similarity_threshold

    @property
    def sample_rejection_ratio(self) -> int:
        return self.resources_config.sample_rejection_ratio

    @property
    def rollout_config(self):
        return self.resources_config.rollout_config

    @property
    def sampler_seed(self) -> int:
        return self.resources_config.sampler_seed

    @property
    def use_cuda_graph_for_rollout(self) -> bool:
        return self.resources_config.use_cuda_graph_for_rollout

    @use_cuda_graph_for_rollout.setter
    def use_cuda_graph_for_rollout(self, value: bool) -> None:
        self.resources_config.use_cuda_graph_for_rollout = value

    @property
    def scene_collision_cfg(self):
        return self.resources_config.scene_collision_cfg

    @property
    def device_cfg(self) -> DeviceCfg:
        return self.resources_config.device_cfg

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        graph_planner_config: Union[str, Dict[str, Any]] = "graph_planner/exact_graph_planner.yml",
        resources_config: Union[str, Dict[str, Any]] = "graph_planner/resources.yml",
        rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = (
            "graph_planner/transition_graph_planner.yml"
        ),
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
        graph_planner_config_dict = resolve_config(
            join_path(get_task_configs_path(), graph_planner_config)
        )
        graph_params = copy.deepcopy(graph_planner_config_dict["graph_planner"])
        resource_overrides = {
            name: graph_params.pop(name)
            for name in tuple(graph_params)
            if name in _RESOURCE_PARAMETER_NAMES
        }
        resources = GraphPlannerResourcesCfg.create(
            robot=robot,
            resources_config=resources_config,
            rollout=rollout,
            transition_model=transition_model,
            scene_model=scene_model,
            collision_cache=collision_cache,
            device_cfg=device_cfg,
            use_cuda_graph_for_rollout=use_cuda_graph_for_rollout,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
            parameter_overrides=resource_overrides,
        )

        return PRMGraphPlannerCfg(
            resources_config=resources,
            **graph_params,
            graph_path_finder_seed=graph_path_finder_seed,
        )
