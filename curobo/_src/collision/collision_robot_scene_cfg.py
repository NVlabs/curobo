# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Configuration for robot-scene collision checking."""

# Standard Library
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Third Party
import torch

from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
from curobo._src.cost.cost_cspace_position import PositionCSpaceCost
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.geom.collision import (
    SceneCollision,
    SceneCollisionCfg,
    create_collision_checker,
)
from curobo._src.geom.types import SceneCfg

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.sampling import SampleBuffer
from curobo._src.util.warp import init_warp
from curobo._src.util_file import (
    get_robot_configs_path,
    get_scene_configs_path,
    join_path,
    load_yaml,
)


@dataclass
class RobotSceneCollisionCfg:
    """Configuration for robot-scene collision checking.

    This dataclass holds all components needed for differentiable collision checking
    between a robot and the environment.

    Attributes:
        kinematics: Robot kinematics model for forward kinematics.
        sampler: Sample buffer for generating random configurations.
        bound_scale: Scaling factor for joint bounds.
        cspace_cost: Cost function for joint position limit violations.
        self_collision_cost: Cost function for self-collision.
        collision_cost: Cost function for scene collision.
        collision_constraint: Constraint function for scene collision.
        scene_model: Scene collision checker.
        rejection_ratio: Ratio of samples to generate for rejection sampling.
        device_cfg: Tensor device configuration.
        contact_distance: Distance threshold for contact detection.
    """

    kinematics: Kinematics
    sampler: SampleBuffer
    bound_scale: torch.Tensor
    cspace_cost: PositionCSpaceCost
    self_collision_cost: Optional[SelfCollisionCost] = None
    collision_cost: Optional[SceneCollisionCost] = None
    collision_constraint: Optional[SceneCollisionCost] = None
    scene_model: Optional[SceneCollision] = None
    rejection_ratio: int = 10
    device_cfg: DeviceCfg = DeviceCfg()
    contact_distance: float = 0.0

    @staticmethod
    def load_from_config(
        robot_config: Union[RobotCfg, str] = "franka.yml",
        scene_model: Union[None, str, Dict, SceneCfg, List[SceneCfg], List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_envs: int = 1,
        n_meshes: int = 50,
        n_cuboids: int = 50,
        collision_activation_distance: float = 0.2,
        self_collision_activation_distance: float = 0.0,
        max_collision_distance: float = 1.0,
        scene_collision_checker: Optional[SceneCollision] = None,
        pose_weight: List[float] = [1, 1, 1, 1],
    ) -> "RobotSceneCollisionCfg":
        """Load configuration from robot and scene config files.

        Args:
            robot_config: Robot configuration file or RobotCfg object.
            scene_model: Scene configuration for collision checking.
            device_cfg: Tensor device configuration.
            num_envs: Number of parallel environments.
            n_meshes: Maximum number of mesh objects in scene cache.
            n_cuboids: Maximum number of cuboid objects in scene cache.
            collision_activation_distance: Distance at which collision cost activates.
            self_collision_activation_distance: Distance for self-collision activation.
            max_collision_distance: Maximum distance for collision queries.
            scene_collision_checker: Pre-built scene collision checker.
            pose_weight: Weights for pose cost (unused in collision-only mode).

        Returns:
            Configured RobotSceneCollisionCfg instance.
        """
        init_warp(device_cfg=device_cfg)
        scene_collision_cost = self_collision_cost = scene_collision_constraint = None

        if isinstance(robot_config, str):
            robot_config = load_yaml(join_path(get_robot_configs_path(), robot_config))[
                "robot_cfg"
            ]
        if isinstance(robot_config, Dict):
            if "robot_cfg" in robot_config:
                robot_config = robot_config["robot_cfg"]
            robot_config = RobotCfg.create(robot_config, device_cfg)
        kinematics = Kinematics(robot_config.kinematics)

        if isinstance(scene_model, str):
            scene_model = load_yaml(join_path(get_scene_configs_path(), scene_model))
        if isinstance(scene_model, List):
            if isinstance(scene_model[0], str):
                scene_model = [
                    load_yaml(join_path(get_scene_configs_path(), x)) for x in scene_model
                ]
        if scene_collision_checker is None and scene_model is not None:
            # Parse scene config
            if isinstance(scene_model, list) and len(scene_model) > 0:
                if isinstance(scene_model[0], dict):
                    scene_cfg_obj = [SceneCfg.create(x) for x in scene_model]
                else:
                    scene_cfg_obj = scene_model
            elif isinstance(scene_model, dict):
                scene_cfg_obj = SceneCfg.create(scene_model)
            else:
                scene_cfg_obj = scene_model

            world_cfg = SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=scene_cfg_obj,
                num_envs=num_envs,
                max_distance=max_collision_distance,
                cache={"mesh": n_meshes, "primitive": n_cuboids},
            )
            scene_collision_checker = create_collision_checker(world_cfg)

        if scene_collision_checker is not None:
            collision_cost_config = SceneCollisionCostCfg(
                weight=device_cfg.to_device([1.0]),
                device_cfg=device_cfg,
                use_grad_input=False,
                activation_distance=collision_activation_distance,
                _scene_collision_checker=scene_collision_checker,
            )
            scene_collision_cost = SceneCollisionCost(collision_cost_config)
            collision_constraint_config = SceneCollisionCostCfg(
                weight=device_cfg.to_device([1.0]),
                device_cfg=device_cfg,
                use_grad_input=True,
                activation_distance=0.0,
            )
            collision_constraint_config.scene_collision_checker = scene_collision_checker
            scene_collision_constraint = SceneCollisionCost(collision_constraint_config)

        self_collision_config = SelfCollisionCostCfg(
            weight=device_cfg.to_device([1.0]),
            device_cfg=device_cfg,
            use_grad_input=True,
            self_collision_kin_config=kinematics.get_self_collision_config(),
        )

        self_collision_cost = SelfCollisionCost(self_collision_config)
        cspace_config = CSpaceCostCfg(
            device_cfg.to_device([1.0, 0.0]),
            device_cfg,
            use_grad_input=True,
            cost_type=CSpaceCostType.POSITION,
            activation_distance=[0.0, 0.0],
        )
        cspace_config.set_bounds(kinematics.get_joint_limits(), teleport_mode=True)
        cspace_config.update_dof(kinematics.get_dof())

        cspace_cost = PositionCSpaceCost(cspace_config)
        sample_gen = SampleBuffer.create_halton_sample_buffer(
            ndims=kinematics.get_dof(),
            device_cfg=device_cfg,
            store_buffer=2000,
            up_bounds=kinematics.get_joint_limits().position[1],
            low_bounds=kinematics.get_joint_limits().position[0],
            seed=123,
        )

        bound_scale = (
            kinematics.get_joint_limits().position[1]
            - kinematics.get_joint_limits().position[0]
        ).unsqueeze(0) / 2.0
        dist_threshold = 0.0
        if collision_activation_distance > 0.0:
            dist_threshold = (
                (0.5 / collision_activation_distance)
                * collision_activation_distance
                * collision_activation_distance
            )

        return RobotSceneCollisionCfg(
            kinematics=kinematics,
            sampler=sample_gen,
            bound_scale=bound_scale,
            cspace_cost=cspace_cost,
            self_collision_cost=self_collision_cost,
            collision_cost=scene_collision_cost,
            collision_constraint=scene_collision_constraint,
            scene_model=scene_collision_checker,
            device_cfg=device_cfg,
            contact_distance=dist_threshold,
        )

