# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Common USD utilities shared by UsdWriter and UsdSceneParser.

This module contains helper functions for USD path manipulation, transform handling,
and geometry attribute setting/getting that are used by multiple USD-related modules.
"""

# Standard Library
from typing import List

# Third Party
from curobo._src.types.device_cfg import DeviceCfg

# CuRobo
from curobo._src.types.pose import Pose

try:
    # Third Party
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    raise ImportError(
        "usd-core failed to import, install with pip install usd-core"
        + " NOTE: Do not install this if using with ISAAC SIM."
    )


def join_usd_path(path1: str, path2: str) -> str:
    """Join two USD scene graph paths.

    USD paths are hierarchical like "/world/robot/link1" and need special handling.
    Unlike filesystem paths, a leading "/" in path2 doesn't make it absolute in USD context.

    Args:
        path1: Parent USD path (e.g., "/world").
        path2: Child USD path (e.g., "/obstacles" or "obstacles").

    Returns:
        str: Joined USD path (e.g., "/world/obstacles").
    """
    # Strip trailing separator from path1
    if path1 and path1.endswith("/"):
        path1 = path1.rstrip("/")

    # Strip leading separator from path2 for USD paths
    if path2 and path2.startswith("/"):
        path2 = path2.lstrip("/")

    # Join with /
    if path1 and path2:
        return f"{path1}/{path2}"
    elif path2:
        return f"/{path2}"
    else:
        return path1


def set_prim_translate(prim, translation):
    """Set translation on a USD prim.

    Args:
        prim: The USD prim to set translation on.
        translation: Translation as [x, y, z] list or tuple.
    """
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(translation))


def set_prim_transform(
    prim, pose: List[float], scale: List[float] = [1, 1, 1], use_float: bool = False
):
    """Set full transform (translation, orientation, scale) on a USD prim.

    Args:
        prim: The USD prim to set transform on.
        pose: Pose as [x, y, z, qw, qx, qy, qz] list.
        scale: Scale as [sx, sy, sz] list. Defaults to [1, 1, 1].
        use_float: Use float precision instead of double. Defaults to False.
    """
    if not prim.GetAttribute("xformOp:translate").IsValid():
        UsdGeom.Xformable(prim).AddTranslateOp(UsdGeom.XformOp.PrecisionFloat)

    if prim.GetAttribute("xformOp:orient").IsValid():
        if isinstance(prim.GetAttribute("xformOp:orient").Get(), Gf.Quatf):
            use_float = True
    else:
        UsdGeom.Xformable(prim).AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
        use_float = True

    if not prim.GetAttribute("xformOp:scale").IsValid():
        UsdGeom.Xformable(prim).AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
    quat = pose[3:]

    if use_float:
        position = Gf.Vec3f(pose[:3])
        q = Gf.Quatf(quat[0], quat[1:])
        dims = Gf.Vec3f(scale)

    else:
        position = Gf.Vec3d(pose[:3])
        q = Gf.Quatd(quat[0], quat[1:])
        dims = Gf.Vec3d(scale)

    prim.GetAttribute("xformOp:translate").Set(position)
    prim.GetAttribute("xformOp:orient").Set(q)
    prim.GetAttribute("xformOp:scale").Set(dims)


def get_prim_world_pose(cache: UsdGeom.XformCache, prim: Usd.Prim, inverse: bool = False):
    """Get world pose from a USD prim.

    Args:
        cache: XformCache for computing world transforms.
        prim: The USD prim to get pose from.
        inverse: If True, return the inverse transform. Defaults to False.

    Returns:
        Tuple of (4x4 transformation matrix as numpy array, scale as list).
    """
    world_transform: Gf.Matrix4d = cache.GetLocalToWorldTransform(prim)
    # get scale:
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    scale = list(scale)
    t_mat = world_transform.RemoveScaleShear()
    if inverse:
        t_mat = t_mat.GetInverse()

    translation: Gf.Vec3d = t_mat.ExtractTranslation()
    rotation: Gf.Rotation = t_mat.ExtractRotation()
    q = rotation.GetQuaternion()
    orientation = [q.GetReal()] + list(q.GetImaginary())
    t_mat = (
        Pose.from_list(list(translation) + orientation, DeviceCfg())
        .get_matrix()
        .view(4, 4)
        .cpu()
        .numpy()
    )

    return t_mat, scale


def get_transform(pose):
    """Create a Gf.Matrix4d from a pose.

    Args:
        pose: Pose as [x, y, z, qw, qx, qy, qz] list.

    Returns:
        Gf.Matrix4d transformation matrix.
    """
    position = Gf.Vec3d(pose[:3])
    quat = pose[3:]
    rotation = Gf.Rotation(Gf.Quatf(quat[0], quat[1:]))

    mat_pose = Gf.Matrix4d()

    mat_pose.SetTransform(rotation, position)
    return mat_pose


def get_position_quat(pose, use_float: bool = True):
    """Convert pose to position and quaternion with specified precision.

    Args:
        pose: Pose as [x, y, z, qw, qx, qy, qz] list.
        use_float: Use float precision if True, else double. Defaults to True.

    Returns:
        Tuple of (position as Gf.Vec3f/d, quaternion as Gf.Quatf/d).
    """
    quat = pose[3:]

    if use_float:
        position = Gf.Vec3f(pose[:3])

        quat = Gf.Quatf(quat[0], quat[1:])

    else:
        position = Gf.Vec3d(pose[:3])

        quat = Gf.Quatd(quat[0], quat[1:])
    return position, quat


def create_stage(
    name: str = "curobo_stage.usd",
    base_frame: str = "/world",
):
    """Create a new USD stage with standard settings.

    Args:
        name: File path for the stage. Defaults to "curobo_stage.usd".
        base_frame: Root prim path. Defaults to "/world".

    Returns:
        Usd.Stage: The created stage.
    """
    stage = Usd.Stage.CreateNew(name)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    xform = stage.DefinePrim(base_frame, "Xform")
    stage.SetDefaultPrim(xform)
    return stage

