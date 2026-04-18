# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import pytest

# CuRobo
from curobo._src.geom.sphere_fit import SphereFitType
from curobo._src.geom.types import SceneCfg
from curobo._src.util_file import get_scene_configs_path, join_path, load_yaml


def test_mesh_world():
    world_file = "collision_test.yml"
    data_dict = load_yaml(join_path(get_scene_configs_path(), world_file))
    scene_cfg = SceneCfg.create(data_dict)
    mesh_world_cfg = scene_cfg.get_mesh_world()
    assert len(mesh_world_cfg.mesh) == len(scene_cfg.cuboid)
    obb_world_cfg = mesh_world_cfg.get_obb_world()
    assert len(obb_world_cfg.cuboid) == len(mesh_world_cfg.mesh)


@pytest.mark.parametrize(
    "sphere_fit_type",
    [e for e in SphereFitType],
)
def test_bounding_volume(sphere_fit_type):
    world_file = "collision_test.yml"
    data_dict = load_yaml(join_path(get_scene_configs_path(), world_file))
    scene_cfg = SceneCfg.create(data_dict)
    if len(scene_cfg.objects) == 0:
        pytest.skip("No objects in scene")
    obs = scene_cfg.objects[-1]
    spheres = obs.get_bounding_spheres(100, 0.01, sphere_fit_type)
    assert len(spheres) > 0
