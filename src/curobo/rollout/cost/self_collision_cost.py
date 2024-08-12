#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
from dataclasses import dataclass
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo.cuda_robot_model.types import SelfCollisionKinematicsConfig
from curobo.curobolib.geom import SelfCollisionDistance

# Local Folder
from .cost_base import CostBase, CostConfig


@dataclass
class SelfCollisionCostConfig(CostConfig):
    self_collision_kin_config: Optional[SelfCollisionKinematicsConfig] = None

    def __post_init__(self):
        return super().__post_init__()


class SelfCollisionCost(CostBase, SelfCollisionCostConfig):
    def __init__(self, config: SelfCollisionCostConfig):
        SelfCollisionCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self._batch_size = None

    def update_batch_size(self, robot_spheres):
        # Assuming n stays constant
        # TODO: use collision buffer here?

        if self._batch_size is None or self._batch_size != robot_spheres.shape:
            b, h, n, k = robot_spheres.shape
            self._out_distance = torch.zeros(
                (b, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            self._out_vec = torch.zeros(
                (b, h, n, k), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            self._batch_size = robot_spheres.shape
            self._sparse_sphere_idx = torch.zeros(
                (b, h, n), device=self.tensor_args.device, dtype=torch.uint8
            )

    def forward(self, robot_spheres):
        self.update_batch_size(robot_spheres)

        dist = SelfCollisionDistance.apply(
            self._out_distance,
            self._out_vec,
            self._sparse_sphere_idx,
            robot_spheres,
            self.self_collision_kin_config.offset,
            self.weight,
            self.self_collision_kin_config.collision_matrix,
            self.self_collision_kin_config.thread_location,
            self.self_collision_kin_config.thread_max,
            self.self_collision_kin_config.checks_per_thread,
            self.self_collision_kin_config.experimental_kernel,
            self.return_loss,
        )

        if self.classify:
            dist = torch.where(dist > 0, dist + 1.0, dist)

        return dist
