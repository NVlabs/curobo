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
import pytest

# CuRobo
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.util_file import get_world_configs_path, join_path, load_yaml


def test_mesh_world():
    world_file = "collision_test.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    mesh_world_cfg = world_cfg.get_mesh_world()
    assert len(mesh_world_cfg.mesh) == len(world_cfg.cuboid)
    obb_world_cfg = mesh_world_cfg.get_obb_world()
    assert len(obb_world_cfg.cuboid) == len(mesh_world_cfg.mesh)


@pytest.mark.parametrize(
    "sphere_fit_type",
    [e.value for e in SphereFitType],
)
def test_bounding_volume(sphere_fit_type):
    world_file = "collision_test.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    obs = world_cfg.objects[-1]
    spheres = obs.get_bounding_spheres(100, 0.01, sphere_fit_type)
    assert len(spheres) > 0
