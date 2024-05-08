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
# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import WorldCollisionConfig, WorldPrimitiveCollision
from curobo.geom.types import WorldConfig
from curobo.rollout.cost.primitive_collision_cost import (
    PrimitiveCollisionCost,
    PrimitiveCollisionCostConfig,
)
from curobo.types.base import TensorDeviceType
from curobo.util_file import get_world_configs_path, join_path, load_yaml


def test_primitive_collision_cost():
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))

    world_cfg = WorldConfig.from_dict(data_dict)
    coll_cfg = WorldPrimitiveCollision(
        WorldCollisionConfig(world_model=world_cfg, tensor_args=TensorDeviceType())
    )

    cost_cfg = PrimitiveCollisionCostConfig(
        weight=1.0,
        tensor_args=tensor_args,
        world_coll_checker=coll_cfg,
        use_sweep=False,
        classify=False,
    )
    cost = PrimitiveCollisionCost(cost_cfg)
    q_spheres = torch.as_tensor(
        [[0.1, 0.0, 0.0, 0.2], [10.0, 0.0, 0.0, 0.1]], **(tensor_args.as_torch_dict())
    ).view(-1, 1, 1, 4)
    c = cost.forward(q_spheres).flatten()
    assert c[0] > 0.0 and c[1] == 0.0
