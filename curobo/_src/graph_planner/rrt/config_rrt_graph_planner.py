# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Complete configuration for constructing the RRT graph planner."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union

# CuRobo
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.graph_planner.config_graph_planner_resources import GraphPlannerResourcesCfg
from curobo._src.graph_planner.rrt.planner_bi_rrt import BiRRTConfig
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg


@dataclass
class RRTGraphPlannerCfg:
    """Compose shared runtime resources with bidirectional-RRT tuning."""

    resources_config: GraphPlannerResourcesCfg
    planner_config: BiRRTConfig

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        planner: Optional[Union[BiRRTConfig, Dict[str, Any]]] = None,
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
    ) -> "RRTGraphPlannerCfg":
        """Create geometric RRT resources sharing robot parameters without computing torques."""
        if isinstance(planner, dict):
            planner = BiRRTConfig(**planner)
        elif planner is not None and not isinstance(planner, BiRRTConfig):
            raise TypeError("planner must be a dict, BiRRTConfig, or None")
        resolved_resources = GraphPlannerResourcesCfg.create(
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
        )
        # RRT searches geometric paths; TrajOpt validates their time-parameterized dynamics.
        resolved_resources.rollout_config.transition_model_cfg.compute_inverse_dynamics = False
        constraint_cfg = resolved_resources.rollout_config.constraint_cfg
        if constraint_cfg is None:
            constraint_cfg = cost_manager_config_instance_type()
            resolved_resources.rollout_config.constraint_cfg = constraint_cfg
        if constraint_cfg.scene_collision_cfg is not None:
            constraint_cfg.scene_collision_cfg.convert_to_binary = False
        if constraint_cfg.self_collision_cfg is not None:
            constraint_cfg.self_collision_cfg.convert_to_binary = False
        if constraint_cfg.tool_pose_cfg is None:
            constraint_cfg.tool_pose_cfg = ToolPoseCostCfg(
                weight=[1.0, 1.0],
                default_tool_pose_criteria=ToolPoseCriteria.disabled(device_cfg=device_cfg),
                device_cfg=device_cfg,
            )
        return RRTGraphPlannerCfg(
            resources_config=resolved_resources,
            planner_config=planner or BiRRTConfig(),
        )
