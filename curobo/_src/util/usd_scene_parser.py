# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""USD Scene Parser for reading obstacles from USD stages.

This module provides the UsdSceneParser class for parsing obstacle geometry
from USD stages. It is the read-only counterpart to UsdWriter.
"""

# Standard Library
from typing import List, Optional

# Third Party
import numpy as np
import torch

# CuRobo
from curobo._src.geom.types import (
    Capsule,
    Cuboid,
    Cylinder,
    Mesh,
    SceneCfg,
    Sphere,
)
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_warn
from curobo._src.util.usd_util import get_prim_world_pose

try:
    # Third Party
    from pxr import Usd, UsdGeom
except ImportError:
    raise ImportError(
        "usd-core failed to import, install with pip install usd-core"
        + " NOTE: Do not install this if using with ISAAC SIM."
    )


def get_cylinder_attrs(prim, cache=None, transform=None) -> Cylinder:
    """Extract Cylinder obstacle from a USD Cylinder prim.

    Args:
        prim: The USD prim to extract from.
        cache: UsdGeom.XformCache for transform computation.
        transform: Optional additional transformation matrix to apply.

    Returns:
        Cylinder obstacle instance.
    """
    size = prim.GetAttribute("size").Get()
    if size is None:
        size = 1.0
    height = prim.GetAttribute("height").Get() * size
    radius = prim.GetAttribute("radius").Get() * size

    mat, t_scale = get_prim_world_pose(cache, prim)

    if transform is not None:
        mat = transform @ mat
    # compute position and orientation on cuda:
    tensor_mat = torch.as_tensor(mat, device=torch.device("cuda", 0))
    pose = Pose.from_matrix(tensor_mat).tolist()
    return Cylinder(name=str(prim.GetPath()), pose=pose, height=height, radius=radius)


def get_capsule_attrs(prim, cache=None, transform=None) -> Capsule:
    """Extract Capsule obstacle from a USD Capsule prim.

    Args:
        prim: The USD prim to extract from.
        cache: UsdGeom.XformCache for transform computation.
        transform: Optional additional transformation matrix to apply.

    Returns:
        Capsule obstacle instance.
    """
    size = prim.GetAttribute("size").Get()
    if size is None:
        size = 1.0
    height = prim.GetAttribute("height").Get() * size
    radius = prim.GetAttribute("radius").Get() * size

    mat, t_scale = get_prim_world_pose(cache, prim)
    base = [0, 0, -height / 2]
    tip = [0, 0, height / 2]

    if transform is not None:
        mat = transform @ mat
    # compute position and orientation on cuda:
    tensor_mat = torch.as_tensor(mat, device=torch.device("cuda", 0))
    pose = Pose.from_matrix(tensor_mat).tolist()
    return Capsule(name=str(prim.GetPath()), pose=pose, base=base, tip=tip, radius=radius)


def get_cube_attrs(prim, cache=None, transform=None) -> Cuboid:
    """Extract Cuboid obstacle from a USD Cube prim.

    Args:
        prim: The USD prim to extract from.
        cache: UsdGeom.XformCache for transform computation.
        transform: Optional additional transformation matrix to apply.

    Returns:
        Cuboid obstacle instance.

    Raises:
        ValueError: If cube dimensions are zero or negative.
    """
    # read cube size:
    size = prim.GetAttribute("size").Get()
    if size is None:
        size = 1.0
    dims = list(prim.GetAttribute("xformOp:scale").Get())
    # scale is 0.5 -> length of 1 will become 0.5,
    dims = [d * size for d in dims]
    if any([x <= 0 for x in dims]):
        raise ValueError("Negative or zero dimension")
    mat, t_scale = get_prim_world_pose(cache, prim)

    if transform is not None:
        mat = transform @ mat
    # compute position and orientation on cuda:
    tensor_mat = torch.as_tensor(mat, device=torch.device("cuda", 0))
    pose = Pose.from_matrix(tensor_mat).tolist()
    return Cuboid(name=str(prim.GetPath()), pose=pose, dims=dims)


def get_sphere_attrs(prim, cache=None, transform=None) -> Sphere:
    """Extract Sphere obstacle from a USD Sphere prim.

    Args:
        prim: The USD prim to extract from.
        cache: UsdGeom.XformCache for transform computation.
        transform: Optional additional transformation matrix to apply.

    Returns:
        Sphere obstacle instance.

    Raises:
        ValueError: If sphere radius is zero or negative.
    """
    size = prim.GetAttribute("size").Get()
    if size is None:
        size = 1.0
    radius = prim.GetAttribute("radius").Get()
    scale = prim.GetAttribute("xformOp:scale").Get()
    if scale is not None:
        radius = radius * max(list(scale)) * size

    if radius <= 0:
        raise ValueError("Negative or zero radius")
    mat, t_scale = get_prim_world_pose(cache, prim)

    if transform is not None:
        mat = transform @ mat
    # compute position and orientation on cuda:
    tensor_mat = torch.as_tensor(mat, device=torch.device("cuda", 0))
    pose = Pose.from_matrix(tensor_mat).tolist()

    return Sphere(name=str(prim.GetPath()), pose=pose, radius=radius, position=pose[:3])


def get_mesh_attrs(prim, cache=None, transform=None) -> Optional[Mesh]:
    """Extract Mesh obstacle from a USD Mesh prim.

    Args:
        prim: The USD prim to extract from.
        cache: UsdGeom.XformCache for transform computation.
        transform: Optional additional transformation matrix to apply.

    Returns:
        Mesh obstacle instance, or None if mesh data is invalid.
    """
    points = list(prim.GetAttribute("points").Get())
    points = [np.ravel(x) for x in points]

    faces = list(prim.GetAttribute("faceVertexIndices").Get())

    face_count = list(prim.GetAttribute("faceVertexCounts").Get())
    # assume faces are 3:
    if len(faces) / 3 != len(face_count):
        log_warn(
            "Mesh faces "
            + str(len(faces) / 3)
            + " are not matching faceVertexCounts "
            + str(len(face_count))
        )
        return None
    faces = np.array(faces).reshape(len(face_count), 3).tolist()
    if prim.GetAttribute("xformOp:scale").IsValid():
        scale = list(prim.GetAttribute("xformOp:scale").Get())
    else:
        scale = [1.0, 1.0, 1.0]
    size = prim.GetAttribute("size").Get()
    if size is None:
        size = 1
    scale = [s * size for s in scale]

    mat, t_scale = get_prim_world_pose(cache, prim)
    # also get any world scale:
    scale = t_scale

    if transform is not None:
        mat = transform @ mat
    # compute position and orientation on cuda:
    tensor_mat = torch.as_tensor(mat, device=torch.device("cuda", 0))
    pose = Pose.from_matrix(tensor_mat).tolist()

    m = Mesh(
        name=str(prim.GetPath()),
        pose=pose,
        vertices=points,
        faces=faces,
        scale=scale,
    )

    return m


class UsdSceneParser:
    """Parser for reading obstacle geometry from USD stages.

    This class provides functionality to load and parse obstacle data from
    USD files or stages. It supports extracting cuboids, spheres, cylinders,
    capsules, and meshes.

    Example:
        >>> parser = UsdSceneParser()
        >>> parser.load_stage_from_file("scene.usd")
        >>> obstacles = parser.get_obstacles_from_stage()
    """

    def __init__(self) -> None:
        """Initialize the UsdSceneParser."""
        self.stage = None
        self._xform_cache = UsdGeom.XformCache()

    def load_stage_from_file(self, file_path: str):
        """Load a USD stage from a file.

        Args:
            file_path: Path to the USD file.
        """
        self.stage = Usd.Stage.Open(file_path)

    def load_stage(self, stage: Usd.Stage):
        """Load a USD stage object directly.

        Args:
            stage: The USD stage to load.
        """
        self.stage = stage

    def get_pose(self, prim_path: str, timecode: float = 0.0, inverse: bool = False) -> np.matrix:
        """Get the world pose of a prim.

        Args:
            prim_path: Path to the prim.
            timecode: Time at which to evaluate the pose. Defaults to 0.0.
            inverse: If True, return the inverse transform. Defaults to False.

        Returns:
            4x4 transformation matrix as numpy array.
        """
        self._xform_cache.SetTime(timecode)
        reference_prim = self.stage.GetPrimAtPath(prim_path)
        r_T_w, _ = get_prim_world_pose(self._xform_cache, reference_prim, inverse=inverse)
        return r_T_w

    def get_obstacles_from_stage(
        self,
        only_paths: Optional[List[str]] = None,
        ignore_paths: Optional[List[str]] = None,
        only_substring: Optional[List[str]] = None,
        ignore_substring: Optional[List[str]] = None,
        reference_prim_path: Optional[str] = None,
        timecode: float = 0,
    ) -> SceneCfg:
        """Extract obstacles from the loaded USD stage.

        Traverses the USD stage and extracts geometric primitives as obstacle
        objects. Supports filtering by path and substring matching.

        Args:
            only_paths: If provided, only include prims whose paths start with
                one of these prefixes.
            ignore_paths: If provided, exclude prims whose paths start with
                one of these prefixes.
            only_substring: If provided, only include prims whose paths contain
                one of these substrings.
            ignore_substring: If provided, exclude prims whose paths contain
                one of these substrings.
            reference_prim_path: If provided, transform all obstacles relative
                to this prim's coordinate frame.
            timecode: Time at which to evaluate transforms. Defaults to 0.

        Returns:
            SceneCfg containing all extracted obstacles.
        """
        obstacles = {"cuboid": [], "sphere": None, "mesh": None, "cylinder": None, "capsule": None}
        r_T_w = None
        self._xform_cache.Clear()
        self._xform_cache.SetTime(timecode)
        if reference_prim_path is not None:
            reference_prim = self.stage.GetPrimAtPath(reference_prim_path)
            r_T_w, _ = get_prim_world_pose(self._xform_cache, reference_prim, inverse=True)
        all_items = self.stage.Traverse()
        for x in all_items:
            if only_paths is not None:
                if not any([str(x.GetPath()).startswith(k) for k in only_paths]):
                    continue
            if ignore_paths is not None:
                if any([str(x.GetPath()).startswith(k) for k in ignore_paths]):
                    continue
            if only_substring is not None:
                if not any([k in str(x.GetPath()) for k in only_substring]):
                    continue
            if ignore_substring is not None:
                if any([k in str(x.GetPath()) for k in ignore_substring]):
                    continue
            if x.IsA(UsdGeom.Cube):
                if obstacles["cuboid"] is None:
                    obstacles["cuboid"] = []
                cube = get_cube_attrs(x, cache=self._xform_cache, transform=r_T_w)
                obstacles["cuboid"].append(cube)
            elif x.IsA(UsdGeom.Sphere):
                if obstacles["sphere"] is None:
                    obstacles["sphere"] = []
                obstacles["sphere"].append(
                    get_sphere_attrs(x, cache=self._xform_cache, transform=r_T_w)
                )
            elif x.IsA(UsdGeom.Mesh):
                if obstacles["mesh"] is None:
                    obstacles["mesh"] = []
                m_data = get_mesh_attrs(x, cache=self._xform_cache, transform=r_T_w)
                if m_data is not None:
                    obstacles["mesh"].append(m_data)
            elif x.IsA(UsdGeom.Cylinder):
                if obstacles["cylinder"] is None:
                    obstacles["cylinder"] = []
                cube = get_cylinder_attrs(x, cache=self._xform_cache, transform=r_T_w)
                obstacles["cylinder"].append(cube)
            elif x.IsA(UsdGeom.Capsule):
                if obstacles["capsule"] is None:
                    obstacles["capsule"] = []
                cap = get_capsule_attrs(x, cache=self._xform_cache, transform=r_T_w)
                obstacles["capsule"].append(cap)
        scene_model = SceneCfg(**obstacles)
        return scene_model

