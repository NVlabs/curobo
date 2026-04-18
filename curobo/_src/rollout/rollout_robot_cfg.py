# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclass for RobotRollout (device, costs, transitions, collision)."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Union

# CuRobo
from curobo._src.geom.collision import SceneCollisionCfg
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.logging import log_and_raise


@dataclass
class RobotRolloutCfg:
    """Configuration for :class:`RobotRollout`.

    Groups device settings, cost-manager configs (costs, constraints,
    convergence), the state-transition model, and the scene-collision model
    needed to initialize a robot rollout.
    """

    #: Device and dtype for the rollout class.
    device_cfg: DeviceCfg

    #: Whether to sum costs across the horizon of the action sequence.
    sum_horizon: bool = False

    #: Seed for the random number generator.
    sampler_seed: int = 1312

    #: Type used to instantiate cost manager configs.
    cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg

    #: Type used to instantiate transition model configs.
    transition_model_config_instance_type: Type[RobotStateTransitionCfg] = (
        RobotStateTransitionCfg
    )

    #: Configuration for the robot state transition model.
    transition_model_cfg: Optional[RobotStateTransitionCfg] = None

    #: Configuration for costs.
    cost_cfg: Optional[RobotCostManagerCfg] = None

    #: Configuration for constraints.
    constraint_cfg: Optional[RobotCostManagerCfg] = None

    #: Configuration for hybrid costs and constraints.
    hybrid_cost_constraint_cfg: Optional[RobotCostManagerCfg] = None

    #: Configuration for convergence costs.
    convergence_cfg: Optional[RobotCostManagerCfg] = None

    #: World collision model.
    scene_collision_cfg: Optional[SceneCollisionCfg] = None

    def __post_init__(self):
        if self.transition_model_cfg is not None:
            if self.transition_model_config_instance_type != type(self.transition_model_cfg):
                log_and_raise(
                    f"transition_model_config_instance_type"
                    f" {self.transition_model_config_instance_type} must match type of"
                    f" transition_model_cfg {type(self.transition_model_cfg)}"
                )
        for field_name in [
            "cost_cfg",
            "constraint_cfg",
            "hybrid_cost_constraint_cfg",
            "convergence_cfg",
        ]:
            cfg = getattr(self, field_name)
            if cfg is not None:
                if self.cost_manager_config_instance_type != type(cfg):
                    log_and_raise(
                        f"cost_manager_config_instance_type"
                        f" {self.cost_manager_config_instance_type} must match type of"
                        f" {field_name} {type(cfg)}"
                    )

    @classmethod
    def create_with_component_types(
        cls,
        data_dict: Dict,
        robot_cfg: Union[Dict, RobotCfg],
        device_cfg: DeviceCfg = DeviceCfg(),
        transition_model_config_instance_type: Type[RobotStateTransitionCfg] = (
            RobotStateTransitionCfg
        ),
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
    ):
        """Create a RobotRolloutCfg from a dictionary with the specified component types."""
        if "transition_model_cfg" in data_dict and data_dict["transition_model_cfg"] is not None:
            data_dict["transition_model_cfg"] = transition_model_config_instance_type.create(
                data_dict["transition_model_cfg"], robot_cfg, device_cfg
            )

        for config_key in [
            "cost_cfg",
            "constraint_cfg",
            "hybrid_cost_constraint_cfg",
            "convergence_cfg",
        ]:
            if config_key in data_dict and data_dict[config_key] is not None:
                data_dict[config_key] = cost_manager_config_instance_type.create(
                    data_dict[config_key], device_cfg=device_cfg
                )

        return cls(
            **data_dict,
            device_cfg=device_cfg,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
        )

    def get_cost_manager_configs(
        self, include_constraint_cfg: bool = True
    ) -> List[RobotCostManagerCfg]:
        manager_configs = []
        for m_cfg in [
            self.cost_cfg,
            self.constraint_cfg if include_constraint_cfg else None,
            self.hybrid_cost_constraint_cfg,
            self.convergence_cfg,
        ]:
            if m_cfg is not None:
                manager_configs.append(m_cfg)
        return manager_configs
