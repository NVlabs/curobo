# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene representation module.

This module provides classes for building and managing scene representations
with various obstacle types.

Example:
    ```python
    from curobo.scene import Scene, Cuboid, Mesh, Sphere

    # Build scene programmatically
    scene = Scene(
        cuboid=[
            Cuboid(name="table", dims=[1.0, 1.0, 0.1], pose=[0, 0, 0.5, 1, 0, 0, 0]),
            Cuboid(name="wall", dims=[0.1, 2.0, 2.0], pose=[1.0, 0, 1.0, 1, 0, 0, 0]),
        ],
        sphere=[
            Sphere(name="ball", radius=0.1, pose=[0.5, 0.5, 1.0, 1, 0, 0, 0])
        ],
        mesh=[
            Mesh(name="object", file_path="model.obj", pose=[0, 0, 0.6, 1, 0, 0, 0])
        ]
    )

    # Or load from file/dict
    scene = Scene.create(scene_dict)

    # Modify scene
    scene.add_obstacle(Sphere(name="new_ball", radius=0.2, pose=[...]))
    scene.remove_obstacle("old_ball")
    ```
"""

from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import (
    Capsule,
    Cuboid,
    Cylinder,
    Mesh,
    Obstacle,
    Sphere,
    VoxelGrid,
)
from curobo._src.geom.types import SceneCfg as Scene

__all__ = [
    "Scene",
    "SceneData",
    "Obstacle",
    "Cuboid",
    "Sphere",
    "Capsule",
    "Cylinder",
    "Mesh",
    "VoxelGrid",
]
