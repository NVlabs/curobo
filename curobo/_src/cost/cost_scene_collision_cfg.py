# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional, Type, Union

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.geom.collision import SceneCollision
from curobo._src.util.logging import log_and_raise


@dataclass
class SceneCollisionCostCfg(BaseCostCfg):
    """Create Collision Cost Configuration."""

    class_type: Type[SceneCollisionCost] = SceneCollisionCost

    #: Sweep for collisions between timesteps in a trajectory.
    use_sweep: bool = False
    use_sweep_kernel: bool = False

    #: Speed metric scales the collision distance by sphere velocity (similar to CHOMP Planner
    #: ICRA'09). This prevents the optimizer from speeding through obstacles to minimize cost and
    #: instead encourages the robot to move around the obstacle.
    use_speed_metric: bool = False

    #: The distance outside collision at which to activate the cost. Having a non-zero value enables
    #: the robot to move slowly when within this distance to an obstacle. This enables our
    #: post optimization interpolation to not hit any obstacles.
    activation_distance: Union[torch.Tensor, float] = 0.0

    #: Setting this flag to true will sum the distance across spheres of the robot.
    #: Setting to False will only take the max distance
    sum_distance: bool = True

    num_spheres: int = 0

    _num_scene_collision_checkers: int = 0

    #: SceneCollision instance to use for distance queries.
    _scene_collision_checker: Optional[SceneCollision] = None

    def __post_init__(self):
        if isinstance(self.activation_distance, float):
            self.activation_distance = self.device_cfg.to_device([float(self.activation_distance)])
        if self._scene_collision_checker is not None:
            self.scene_collision_checker = self._scene_collision_checker

        return super().__post_init__()


    def update_num_spheres(self, num_spheres: int):
        self.num_spheres = num_spheres

    def update_num_scene_collision_checkers(self, num_scene_collision_checkers: int):
        self._num_scene_collision_checkers = num_scene_collision_checkers

    @property
    def scene_collision_checker(self) -> Optional[SceneCollision]:
        return self._scene_collision_checker

    @scene_collision_checker.setter
    def scene_collision_checker(self, scene_collision_checker: SceneCollision):
        if not isinstance(scene_collision_checker, SceneCollision):
            log_and_raise(
                "scene_collision_checker must be an instance of SceneCollision, got {}".format(
                    type(scene_collision_checker)
                )
            )
        self._scene_collision_checker = scene_collision_checker
        self.update_num_scene_collision_checkers(
            self.scene_collision_checker.get_num_scene_collision_checkers()
        )
