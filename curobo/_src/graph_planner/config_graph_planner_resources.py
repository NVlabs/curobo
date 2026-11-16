# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shared runtime-resource configuration for graph planners."""

from __future__ import annotations

# Standard Library
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union

# CuRobo
from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.config_io import join_path, resolve_config
from curobo.content import get_robot_configs_path, get_scene_configs_path, get_task_configs_path


@dataclass
class GraphPlannerResourcesCfg:
    """Configuration for robot, collision, sampling, and steering resources."""

    feasibility_buffer_size: int
    steer_buffer_size: int
    cspace_similarity_threshold: float
    sample_rejection_ratio: int
    rollout_config: RobotRolloutCfg
    sampler_seed: int
    use_cuda_graph_for_rollout: bool
    scene_collision_cfg: Optional[SceneCollisionCfg] = None
    device_cfg: DeviceCfg = DeviceCfg()

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        resources_config: Union[str, Dict[str, Any]] = "graph_planner/resources.yml",
        rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = (
            "graph_planner/transition_graph_planner.yml"
        ),
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        use_cuda_graph_for_rollout: bool = True,
        transition_model_config_instance_type: Type[
            RobotStateTransitionCfg
        ] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        parameter_overrides: Optional[Dict[str, Any]] = None,
    ) -> "GraphPlannerResourcesCfg":
        """Create graph-planner resources without selecting a planning algorithm."""
        robot_resolved = resolve_config(join_path(get_robot_configs_path(), robot))
        robot_config = (
            RobotCfg.create(robot_resolved, device_cfg)
            if isinstance(robot_resolved, dict)
            else robot_resolved
        )

        scene_collision_cfg = None
        if scene_model is not None:
            scene_model_data = resolve_config(join_path(get_scene_configs_path(), scene_model))
            if isinstance(scene_model_data, list):
                scene_cfg = [SceneCfg.create(value) for value in scene_model_data]
            elif isinstance(scene_model_data, dict):
                scene_cfg = SceneCfg.create(scene_model_data)
            else:
                scene_cfg = scene_model_data
            scene_collision_cfg = SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=scene_cfg,
                cache=collision_cache,
            )

        resources_data = resolve_config(join_path(get_task_configs_path(), resources_config))
        resource_params = copy.deepcopy(
            resources_data.get("graph_planner_resources", resources_data)
        )
        if parameter_overrides is not None:
            resource_params.update(parameter_overrides)

        rollout_data = resolve_config(join_path(get_task_configs_path(), rollout))
        transition_data = resolve_config(join_path(get_task_configs_path(), transition_model))
        rollout_dict = copy.deepcopy(rollout_data["rollout"])
        rollout_dict["transition_model_cfg"] = transition_data["transition_model_cfg"]
        rollout_config = RobotRolloutCfg.create_with_component_types(
            data_dict=rollout_dict,
            robot_cfg=robot_config,
            device_cfg=device_cfg,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
        )

        return GraphPlannerResourcesCfg(
            **resource_params,
            rollout_config=rollout_config,
            scene_collision_cfg=scene_collision_cfg,
            device_cfg=device_cfg,
            use_cuda_graph_for_rollout=use_cuda_graph_for_rollout,
        )
