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
import math
from typing import Dict, List, Optional, Union, Tuple

# Third Party
import numpy as np
import torch
from tqdm import tqdm

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.types import (
    Capsule,
    Cuboid,
    Cylinder,
    Material,
    Mesh,
    Obstacle,
    Sphere,
    WorldConfig,
)
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util_file import (
    file_exists,
    get_assets_path,
    get_filename,
    get_files_from_dir,
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGenResult

try:
    # Third Party
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    raise ImportError(
        "usd-core failed to import, install with pip install usd-core"
        + " NOTE: Do not install this if using with ISAAC SIM."
    )

# Quad mesh triangulation warp kernel
try:
    import warp as wp  # type: ignore

    _WARP_AVAILABLE = True

    @wp.kernel
    def _triangulate_quads_kernel(
        verts: wp.array1d(dtype=wp.vec3),
        quads: wp.array1d(dtype=wp.vec4i),
        tris_out: wp.array1d(dtype=wp.vec3i),
    ):
        ind = wp.tid()
        ind_out = 2 * ind  # each quad as 2 triangles in the output buffer
        quad_inds = quads[ind]
        q0, q1, q2, q3 = quad_inds[0], quad_inds[1], quad_inds[2], quad_inds[3]
        v0 = verts[q0]
        v1 = verts[q1]
        v2 = verts[q2]
        v3 = verts[q3]

        # Two possible triangulations
        nA1 = wp.normalize(wp.cross(v1 - v0, v2 - v0))
        nA2 = wp.normalize(wp.cross(v2 - v0, v3 - v0))

        nB1 = wp.normalize(wp.cross(v3 - v1, v0 - v1))
        nB2 = wp.normalize(wp.cross(v2 - v1, v3 - v1))

        if wp.dot(nA1, nA2) > wp.dot(nB1, nB2):
            tris_out[ind_out] = wp.vec3i(q0, q1, q2)
            tris_out[ind_out + 1] = wp.vec3i(q0, q2, q3)
        else:
            tris_out[ind_out] = wp.vec3i(q1, q3, q0)
            tris_out[ind_out + 1] = wp.vec3i(q1, q2, q3)

except ImportError:  # pragma: no cover – Warp not available
    _WARP_AVAILABLE = False


def set_prim_translate(prim, translation):
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(translation))


def set_prim_transform(
    prim, pose: List[float], scale: List[float] = [1, 1, 1], use_float: bool = False
):
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

    # get scale:


def get_prim_world_pose(cache: UsdGeom.XformCache, prim: Usd.Prim, inverse: bool = False):
    world_transform: Gf.Matrix4d = cache.GetLocalToWorldTransform(prim)
    # get scale:
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    scale = list(scale)
    t_mat = world_transform.RemoveScaleShear()
    if inverse:
        t_mat = t_mat.GetInverse()

    # mat = np.zeros((4,4))
    # mat[:,:] = t_mat
    translation: Gf.Vec3d = t_mat.ExtractTranslation()
    rotation: Gf.Rotation = t_mat.ExtractRotation()
    q = rotation.GetQuaternion()
    orientation = [q.GetReal()] + list(q.GetImaginary())
    t_mat = (
        Pose.from_list(list(translation) + orientation, TensorDeviceType())
        .get_matrix()
        .view(4, 4)
        .cpu()
        .numpy()
    )

    return t_mat, scale


def get_transform(pose):
    position = Gf.Vec3d(pose[:3])
    quat = pose[3:]
    rotation = Gf.Rotation(Gf.Quatf(quat[0], quat[1:]))

    mat_pose = Gf.Matrix4d()

    mat_pose.SetTransform(rotation, position)
    return mat_pose


def get_position_quat(pose, use_float: bool = True):
    quat = pose[3:]

    if use_float:
        position = Gf.Vec3f(pose[:3])

        quat = Gf.Quatf(quat[0], quat[1:])

    else:
        position = Gf.Vec3d(pose[:3])

        quat = Gf.Quatd(quat[0], quat[1:])
    return position, quat


def set_geom_mesh_attrs(mesh_geom: UsdGeom.Mesh, obs: Mesh, timestep=None):
    verts, faces = obs.get_mesh_data()
    mesh_geom.CreatePointsAttr(verts)
    mesh_geom.CreateFaceVertexCountsAttr([3 for _ in range(len(faces))])
    mesh_geom.CreateFaceVertexIndicesAttr(np.ravel(faces).tolist())
    mesh_geom.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    if obs.vertex_colors is not None:
        primvarsapi = UsdGeom.PrimvarsAPI(mesh_geom)
        primvar = primvarsapi.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3f, interpolation="faceVarying"
        )
        scale = 1.0
        # color needs to be in range of 0-1. Hence converting if the color is in [0,255]
        if max(np.ravel(obs.vertex_colors) > 1.0):
            scale = 255.0
        primvar.Set([Gf.Vec3f(x[0] / scale, x[1] / scale, x[2] / scale) for x in obs.vertex_colors])

    # low = np.min(verts, axis=0)
    # high = np.max(verts, axis=0)
    # mesh_geom.CreateExtentAttr([low, high])
    pose = obs.pose
    position = Gf.Vec3d(pose[:3])
    quat = pose[3:]
    q = Gf.Quatf(quat[0], quat[1:])

    # rotation = Gf.Rotation(Gf.Quatf(quat[0], quat[1:]))

    # mat_pose = Gf.Matrix4d()
    # mat_pose.SetTransform(rotation, position)
    # size = 1.0
    # mesh_geom.CreateSizeAttr(size)
    if timestep is not None:
        # UsdGeom.Xformable(mesh_geom).AddTransformOp().Set(time=timestep, value=mat_pose)
        a = UsdGeom.Xformable(mesh_geom)  #
        a.AddTranslateOp().Set(time=timestep, value=position)
        a.AddOrientOp().Set(time=timestep, value=q)
    else:
        a = UsdGeom.Xformable(mesh_geom)  #
        a.AddTranslateOp().Set(position)
        a.AddOrientOp().Set(q)

        # UsdGeom.Xformable(mesh_geom).AddTransformOp().Set(mat_pose)


def set_geom_cube_attrs(
    cube_geom: UsdGeom.Cube, dims: List[float], pose: List[float], timestep=None
):
    dims = Gf.Vec3d(np.ravel(dims).tolist())
    position = Gf.Vec3d(pose[:3])
    quat = pose[3:]
    q = Gf.Quatf(quat[0], quat[1:])
    # rotation = Gf.Rotation(q)

    # mat_pose = Gf.Matrix4d()
    # mat_pose.SetTransform(rotation, position)
    # mat = mat_pose
    # mat_scale = Gf.Matrix4d()
    # mat_scale.SetScale(dims)
    # mat = mat_scale * mat_pose
    size = 1.0
    cube_geom.CreateSizeAttr(size)
    # create scale:

    a = UsdGeom.Xformable(cube_geom)  #
    a.AddTranslateOp().Set(position)
    a.AddOrientOp().Set(q)

    # a.AddTransformOp().Set(mat)
    # scale will set the length to the given value
    a.AddScaleOp().Set(dims)


def set_geom_cylinder_attrs(
    cube_geom: UsdGeom.Cylinder, radius, height, pose: List[float], timestep=None
):
    position = Gf.Vec3d(pose[:3])
    quat = pose[3:]
    q = Gf.Quatf(quat[0], quat[1:])

    # create scale:
    cube_geom.CreateRadiusAttr(radius)
    cube_geom.CreateHeightAttr(height)
    a = UsdGeom.Xformable(cube_geom)  #
    a.AddTranslateOp().Set(position)
    a.AddOrientOp().Set(q)


def set_geom_sphere_attrs(
    sphere_geom: UsdGeom.Sphere, radius: float, pose: List[float], timestep=None
):
    position = Gf.Vec3d(pose[:3])
    quat = pose[3:]
    q = Gf.Quatf(quat[0], quat[1:])

    a = UsdGeom.Xformable(sphere_geom)  #
    a.AddTranslateOp().Set(position)
    a.AddOrientOp().Set(q)

    sphere_geom.CreateRadiusAttr(float(radius))


def set_cylinder_attrs(prim: UsdGeom.Cylinder, radius: float, height: float, pose, color=[]):
    # set size to 1:
    position = Gf.Vec3d(np.ravel(pose.xyz).tolist())
    quat = pose.so3.wxyz
    rotation = Gf.Rotation(Gf.Quatf(quat[0], quat[1:]))

    mat_pose = Gf.Matrix4d()
    mat_pose.SetTransform(rotation, position)

    prim.GetAttribute("height").Set(height)
    prim.GetAttribute("radius").Set(radius)

    UsdGeom.Xformable(prim).AddTransformOp().Set(mat_pose)


def get_cylinder_attrs(prim, cache=None, transform=None) -> Cylinder:
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


def get_capsule_attrs(prim, cache=None, transform=None) -> Cylinder:
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
    # read cube size:
    size = prim.GetAttribute("size").Get()
    if size is None:
        size = 1.0
    dims = list(prim.GetAttribute("xformOp:scale").Get())
    # scale is 0.5 -> length of 1 will become 0.5,
    dims = [d * size for d in dims]
    if any([x <= 0 for x in dims]):
        raise ValueError("Negative or zero dimension")
    # dims = [x*2 for x in dims]
    mat, t_scale = get_prim_world_pose(cache, prim)

    if transform is not None:
        mat = transform @ mat
    # compute position and orientation on cuda:
    tensor_mat = torch.as_tensor(mat, device=torch.device("cuda", 0))
    pose = Pose.from_matrix(tensor_mat).tolist()
    return Cuboid(name=str(prim.GetPath()), pose=pose, dims=dims)


def get_sphere_attrs(prim, cache=None, transform=None) -> Sphere:
    # read cube information
    # scale = prim.GetAttribute("size").Get()
    size = prim.GetAttribute("size").Get()
    if size is None:
        size = 1.0
    radius = prim.GetAttribute("radius").Get()
    scale = prim.GetAttribute("xformOp:scale").Get()
    if scale is not None:
        radius = radius * max(list(scale)) * size

    if radius <= 0:
        raise ValueError("Negative or zero radius")
    # dims = [x*2 for x in dims]
    mat, t_scale = get_prim_world_pose(cache, prim)
    # position = list(prim.GetAttribute("xformOp:translate").Get())
    # q = prim.GetAttribute("xformOp:orient").Get()
    # orientation = [q.GetReal()] + list(q.GetImaginary())

    if transform is not None:
        mat = transform @ mat
    # compute position and orientation on cuda:
    tensor_mat = torch.as_tensor(mat, device=torch.device("cuda", 0))
    pose = Pose.from_matrix(tensor_mat).tolist()

    return Sphere(name=str(prim.GetPath()), pose=pose, radius=radius, position=pose[:3])


def triangulate_mesh_faces(
    vertices: List[List[float]],
    faces: List[int],
    face_count: List[int],
) -> Tuple[List[int], List[int]]:
    # Triangulate mesh faces. Returns a flattened index buffer and a face-count list (all 3s).
    if not faces or not face_count:
        raise ValueError("Empty face data provided")

    expected_vertices = sum(face_count)
    if len(faces) != expected_vertices:
        raise ValueError(
            f"Face data inconsistent: {len(faces)} vertices but face_count sums to {expected_vertices}"
        )

    if any(c < 3 for c in face_count):
        raise ValueError("Found a face with < 3 vertices")

    tri_count = sum(1 for c in face_count if c == 3)
    quad_count = sum(1 for c in face_count if c == 4)
    other_count = len(face_count) - tri_count - quad_count

    log_info(f"[Triangulation] triangles={tri_count}, quads={quad_count}, others={other_count}")

    if other_count == 0 and quad_count == 0:
        return faces, face_count

    new_faces: List[int] = []
    quad_list: List[List[int]] = []

    face_idx = 0
    for count in face_count:
        if count == 3:
            new_faces.extend(faces[face_idx : face_idx + 3])
        elif count == 4:
            quad_list.append(faces[face_idx : face_idx + 4])
        else:
            v0 = faces[face_idx]
            for i in range(1, count - 1):
                new_faces.extend([v0, faces[face_idx + i], faces[face_idx + i + 1]])
        face_idx += count

    if quad_list:
        if _WARP_AVAILABLE:
            log_info(
                f"[Triangulation] Using Warp GPU kernel on {len(quad_list)} quads",
            )

            import numpy as _np

            verts_np = _np.asarray(vertices, dtype=_np.float32)
            quads_np = _np.asarray(quad_list, dtype=_np.int32)

            verts_wp = wp.array(verts_np, dtype=wp.vec3, device="cuda")
            quads_wp = wp.array(quads_np, dtype=wp.vec4i, device="cuda")
            tris_wp = wp.empty(shape=(quads_wp.shape[0] * 2,), dtype=wp.vec3i, device="cuda")

            wp.launch(
                _triangulate_quads_kernel,
                dim=quads_wp.shape[0],
                inputs=[verts_wp, quads_wp],
                outputs=[tris_wp],
            )

            tris_np = tris_wp.numpy().astype(_np.int64).reshape(-1, 3)
            new_faces.extend(tris_np.flatten().tolist())
        else:
            log_warn(
                f"[Triangulation] Warp not available, CPU splitting {len(quad_list)} quads",
            )
            # CPU fallback – split along fixed diagonal (v0-v2)
            for q in quad_list:
                new_faces.extend([q[0], q[1], q[2], q[0], q[2], q[3]])

    triangle_count = len(new_faces) // 3
    return new_faces, [3] * triangle_count


def get_mesh_attrs(prim, cache=None, transform=None) -> Mesh:
    # read cube information
    # scale = prim.GetAttribute("size").Get()
    points = list(prim.GetAttribute("points").Get())
    points = [np.ravel(x) for x in points]
    # points = np.ndarray(points)

    faces = list(prim.GetAttribute("faceVertexIndices").Get())

    face_count = list(prim.GetAttribute("faceVertexCounts").Get())
    # assume faces are 3:
    if len(faces) / 3 != len(face_count):
        log_warn(f"Mesh {prim.GetPath()} has non-triangular faces, triangulating...")
        try:
            faces, face_count = triangulate_mesh_faces(points, faces, face_count)
            log_info(f"Triangulated mesh {prim.GetPath()} with {len(faces)} faces")
        except ValueError as e:
            log_error(f"Failed to triangulate mesh {prim.GetPath()}: {e}")
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
    # position = list(prim.GetAttribute("xformOp:translate").Get())
    # q = prim.GetAttribute("xformOp:orient").Get()
    # orientation = [q.GetReal()] + list(q.GetImaginary())

    if transform is not None:
        mat = transform @ mat
    # compute position and orientation on cuda:
    tensor_mat = torch.as_tensor(mat, device=torch.device("cuda", 0))
    pose = Pose.from_matrix(tensor_mat).tolist()

    #

    m = Mesh(
        name=str(prim.GetPath()),
        pose=pose,
        vertices=points,
        faces=faces,
        scale=scale,
    )
    # print(len(m.vertices), max(m.faces))

    return m


def create_stage(
    name: str = "curobo_stage.usd",
    base_frame: str = "/world",
):
    stage = Usd.Stage.CreateNew(name)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    xform = stage.DefinePrim(base_frame, "Xform")
    stage.SetDefaultPrim(xform)
    return stage


class UsdHelper:
    def __init__(self, use_float=True) -> None:
        self.stage = None
        self.dt = None
        self._use_float = use_float
        self._xform_cache = UsdGeom.XformCache()

    def create_stage(
        self,
        name: str = "curobo_stage.usd",
        base_frame: str = "/world",
        timesteps: Optional[int] = None,
        dt=0.02,
        interpolation_steps: float = 1,
    ):
        # print("name", name)
        self.stage = Usd.Stage.CreateNew(name)
        UsdGeom.SetStageUpAxis(self.stage, "Z")
        UsdGeom.SetStageMetersPerUnit(self.stage, 1)
        UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1)
        xform = self.stage.DefinePrim(base_frame, "Xform")
        self.stage.SetDefaultPrim(xform)
        self.dt = dt
        self.interpolation_steps = interpolation_steps
        if timesteps is not None:
            self.stage.SetStartTimeCode(1)
            self.stage.SetEndTimeCode(timesteps * self.interpolation_steps)
            self.stage.SetTimeCodesPerSecond((1.0 / self.dt))
            # print(1.0 / self)

    def add_subroot(self, root="/world", sub_root="/obstacles", pose: Optional[Pose] = None):
        xform = self.stage.DefinePrim(join_path(root, sub_root), "Xform")
        if pose is not None:
            set_prim_transform(xform, pose.tolist(), use_float=self._use_float)

    def load_stage_from_file(self, file_path: str):
        self.stage = Usd.Stage.Open(file_path)

    def load_stage(self, stage: Usd.Stage):
        self.stage = stage

    def get_pose(self, prim_path: str, timecode: float = 0.0, inverse: bool = False) -> np.matrix:
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
    ) -> WorldConfig:
        # read obstacles from usd by iterating through all prims:
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
        world_model = WorldConfig(**obstacles)
        return world_model

    def add_world_to_stage(
        self,
        obstacles: WorldConfig,
        base_frame: str = "/world",
        obstacles_frame: str = "obstacles",
        base_t_obstacle_pose: Optional[Pose] = None,
        timestep: Optional[float] = None,
    ):
        # iterate through every obstacle type and create prims:

        self.add_subroot(base_frame, obstacles_frame, base_t_obstacle_pose)
        full_path = join_path(base_frame, obstacles_frame)
        prim_path = [
            self.get_prim_from_obstacle(o, full_path, timestep=timestep) for o in obstacles.objects
        ]
        return prim_path

    def get_prim_from_obstacle(
        self, obstacle: Obstacle, base_frame: str = "/world/obstacles", timestep=None
    ):

        if isinstance(obstacle, Cuboid):
            return self.add_cuboid_to_stage(obstacle, base_frame, timestep=timestep)
        elif isinstance(obstacle, Mesh):
            return self.add_mesh_to_stage(obstacle, base_frame, timestep=timestep)
        elif isinstance(obstacle, Sphere):
            return self.add_sphere_to_stage(obstacle, base_frame, timestep=timestep)
        elif isinstance(obstacle, Cylinder):
            return self.add_cylinder_to_stage(obstacle, base_frame, timestep=timestep)

        else:
            raise NotImplementedError

    def add_cuboid_to_stage(
        self,
        obstacle: Cuboid,
        base_frame: str = "/world/obstacles",
        timestep=None,
        enable_physics: bool = False,
    ):
        root_path = join_path(base_frame, obstacle.name)
        obj_geom = UsdGeom.Cube.Define(self.stage, root_path)
        obj_prim = self.stage.GetPrimAtPath(root_path)

        set_geom_cube_attrs(obj_geom, obstacle.dims, obstacle.pose, timestep=timestep)
        obj_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool, custom=False)
        obj_prim.GetAttribute("physics:rigidBodyEnabled").Set(enable_physics)

        if obstacle.color is not None:
            self.add_material(
                "material_" + obstacle.name, root_path, obstacle.color, obj_prim, obstacle.material
            )
        return root_path

    def add_cylinder_to_stage(
        self,
        obstacle: Cylinder,
        base_frame: str = "/world/obstacles",
        timestep=None,
        enable_physics: bool = False,
    ):
        root_path = join_path(base_frame, obstacle.name)
        obj_geom = UsdGeom.Cylinder.Define(self.stage, root_path)
        obj_prim = self.stage.GetPrimAtPath(root_path)

        set_geom_cylinder_attrs(
            obj_geom, obstacle.radius, obstacle.height, obstacle.pose, timestep=timestep
        )
        obj_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool, custom=False)
        obj_prim.GetAttribute("physics:rigidBodyEnabled").Set(enable_physics)

        if obstacle.color is not None:
            self.add_material(
                "material_" + obstacle.name, root_path, obstacle.color, obj_prim, obstacle.material
            )
        return root_path

    def add_sphere_to_stage(
        self,
        obstacle: Sphere,
        base_frame: str = "/world/obstacles",
        timestep=None,
        enable_physics: bool = False,
    ):
        root_path = join_path(base_frame, obstacle.name)
        obj_geom = UsdGeom.Sphere.Define(self.stage, root_path)
        obj_prim = self.stage.GetPrimAtPath(root_path)
        if obstacle.pose is None:
            obstacle.pose = obstacle.position + [1, 0, 0, 0]
        set_geom_sphere_attrs(obj_geom, obstacle.radius, obstacle.pose, timestep=timestep)
        obj_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool, custom=False)
        obj_prim.GetAttribute("physics:rigidBodyEnabled").Set(enable_physics)

        if obstacle.color is not None:
            self.add_material(
                "material_" + obstacle.name, root_path, obstacle.color, obj_prim, obstacle.material
            )
        return root_path

    def add_mesh_to_stage(
        self,
        obstacle: Mesh,
        base_frame: str = "/world/obstacles",
        timestep=None,
        enable_physics: bool = False,
    ):
        root_path = join_path(base_frame, obstacle.name)
        obj_geom = UsdGeom.Mesh.Define(self.stage, root_path)
        obj_prim = self.stage.GetPrimAtPath(root_path)
        # obstacle.update_material() # This does not get the correct materials
        set_geom_mesh_attrs(obj_geom, obstacle, timestep=timestep)

        obj_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool, custom=False)
        obj_prim.GetAttribute("physics:rigidBodyEnabled").Set(enable_physics)

        if obstacle.color is not None:
            self.add_material(
                "material_" + obstacle.name, root_path, obstacle.color, obj_prim, obstacle.material
            )

        return root_path

    def get_obstacle_from_prim(self, prim_path: str) -> Obstacle:
        pass

    def write_stage_to_file(self, file_path: str, flatten: bool = False):
        if flatten:
            usd_str = self.stage.Flatten().ExportToString()
        else:
            usd_str = self.stage.GetRootLayer().ExportToString()
        with open(file_path, "w") as f:
            f.write(usd_str)

    def create_animation(
        self,
        robot_world_cfg: WorldConfig,
        pose: Pose,
        base_frame="/world",
        robot_frame="/robot",
        dt: float = 0.02,
    ):
        """Create animation, given meshes and pose

        Args:
            prim_names: _description_
            pose: [ timesteps, n_meshes, pose]
            dt: _description_. Defaults to 0.02.
        """
        prim_names = self.add_world_to_stage(
            robot_world_cfg, base_frame=base_frame, obstacles_frame=robot_frame, timestep=0
        )
        for i, i_val in enumerate(prim_names):
            curr_prim = self.stage.GetPrimAtPath(i_val)
            form = UsdGeom.Xformable(curr_prim).GetOrderedXformOps()
            if len(form) < 2:
                log_warn("Pose transformation not found" + i_val)
                continue

            pos_form = form[0]
            quat_form = form[1]
            use_float = True  # default is float
            for t in range(pose.batch):
                c_p, c_q = get_position_quat(pose.get_index(t, i).tolist(), use_float)
                pos_form.Set(time=t * self.interpolation_steps, value=c_p)
                quat_form.Set(time=t * self.interpolation_steps, value=c_q)
                # c_t = get_transform(pose.get_index(t, i).tolist())
                # form.Set(time=t * self.interpolation_steps, value=c_t)

    def create_obstacle_animation(
        self,
        obstacles: List[List[Obstacle]],
        base_frame: str = "/world",
        obstacles_frame: str = "robot_base",
    ):
        # add obstacles to stage:
        prim_paths = self.add_world_to_stage(
            WorldConfig(objects=obstacles[0]),
            base_frame=base_frame,
            obstacles_frame=obstacles_frame,
        )

        #
        for t in range(len(obstacles)):
            current_obs = obstacles[t]
            for j in range(len(current_obs)):
                obs = current_obs[j]
                obs_name = join_path(join_path(base_frame, obstacles_frame), obs.name)
                if obs_name not in prim_paths:
                    log_warn("Obstacle not found")
                    continue
                #
                prim = self.stage.GetPrimAtPath(obs_name)
                form = UsdGeom.Xformable(prim).GetOrderedXformOps()

                pos_form = form[0]
                c_p = Gf.Vec3d(obs.position)
                pos_form.Set(time=t * self.interpolation_steps, value=c_p)

    def create_linkpose_robot_animation(
        self,
        robot_usd_path: str,
        link_names: List[str],
        joint_names: List[str],
        pose: Pose,
        robot_base_frame="/world/robot",
        local_asset_path="assets/",
        write_robot_usd_path="assets/",
        robot_asset_prim_path="/panda",
    ):
        """Create animation, given meshes and pose

        Args:
            prim_names: _description_
            pose: [ timesteps, n_meshes, pose]
            dt: _description_. Defaults to 0.02.
        """
        link_prims, joint_prims = self.load_robot_usd(
            robot_usd_path,
            link_names,
            joint_names,
            robot_base_frame=robot_base_frame,
            local_asset_path=local_asset_path,
            write_asset_path=write_robot_usd_path,
            robot_asset_prim_path=robot_asset_prim_path,
        )

        for i, i_val in enumerate(link_names):
            if i_val not in link_prims:
                log_warn("Link not found in usd: " + i_val)
                continue
            form = UsdGeom.Xformable(link_prims[i_val]).GetOrderedXformOps()
            if len(form) < 2:
                log_warn("Pose transformation not found" + i_val)
                continue

            pos_form = form[0]
            quat_form = form[1]
            use_float = False
            if link_prims[i_val].GetAttribute("xformOp:orient").IsValid():
                if isinstance(link_prims[i_val].GetAttribute("xformOp:orient").Get(), Gf.Quatf):
                    use_float = True

            for t in range(pose.batch):
                c_p, c_q = get_position_quat(pose.get_index(t, i).tolist(), use_float)
                pos_form.Set(time=t * self.interpolation_steps, value=c_p)
                quat_form.Set(time=t * self.interpolation_steps, value=c_q)

    def add_material(
        self,
        material_name: str,
        object_path: str,
        color: List[float],
        obj_prim: Usd.Prim,
        material: Material = Material(),
    ):
        mat_path = join_path(object_path, material_name)
        material_usd = UsdShade.Material.Define(self.stage, mat_path)
        pbrShader = UsdShade.Shader.Define(self.stage, join_path(mat_path, "PbrShader"))
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(material.roughness)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(material.metallic)
        pbrShader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[:3]))
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[:3]))
        pbrShader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[:3]))

        pbrShader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(color[3])
        material_usd.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")
        obj_prim.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(obj_prim).Bind(material_usd)
        return material_usd

    def save(self):
        self.stage.Save()

    @staticmethod
    def write_trajectory_animation(
        robot_model_file: str,
        world_model: WorldConfig,
        q_start: JointState,
        q_traj: JointState,
        dt: float = 0.02,
        save_path: str = "out.usd",
        tensor_args: TensorDeviceType = TensorDeviceType(),
        interpolation_steps: float = 1.0,
        robot_base_frame="robot",
        base_frame="/world",
        kin_model: Optional[CudaRobotModel] = None,
        visualize_robot_spheres: bool = True,
        robot_color: Optional[List[float]] = None,
        flatten_usd: bool = False,
        goal_pose: Optional[Pose] = None,
        goal_color: Optional[List[float]] = None,
    ):
        if kin_model is None:
            config_file = load_yaml(join_path(get_robot_configs_path(), robot_model_file))
            if "robot_cfg" not in config_file:
                config_file["robot_cfg"] = config_file
            config_file["robot_cfg"]["kinematics"]["load_link_names_with_mesh"] = True
            robot_cfg = CudaRobotModelConfig.from_data_dict(
                config_file["robot_cfg"]["kinematics"], tensor_args=tensor_args
            )
            kin_model = CudaRobotModel(robot_cfg)
        m = kin_model.get_robot_link_meshes()
        offsets = [x.pose for x in m]
        robot_mesh_model = WorldConfig(mesh=m)
        if robot_color is not None:
            robot_mesh_model.add_color(robot_color)
            robot_mesh_model.add_material(Material(metallic=0.4))
        if goal_pose is not None:
            kin_model.link_names
            if kin_model.ee_link in kin_model.kinematics_config.mesh_link_names:
                index = kin_model.kinematics_config.mesh_link_names.index(kin_model.ee_link)
                gripper_mesh = m[index]
            if len(goal_pose.shape) == 1:
                goal_pose = goal_pose.unsqueeze(0)
            if len(goal_pose.shape) == 2:
                goal_pose = goal_pose.unsqueeze(0)
            for i in range(goal_pose.n_goalset):
                g = goal_pose.get_index(0, i).to_list()
                world_model.add_obstacle(
                    Mesh(
                        file_path=gripper_mesh.file_path,
                        pose=g,
                        name="goal_idx_" + str(i),
                        color=goal_color,
                    )
                )
        usd_helper = UsdHelper()
        usd_helper.create_stage(
            save_path,
            timesteps=q_traj.position.shape[0],
            dt=dt,
            interpolation_steps=interpolation_steps,
            base_frame=base_frame,
        )
        if world_model is not None:
            usd_helper.add_world_to_stage(world_model, base_frame=base_frame)

        animation_links = kin_model.kinematics_config.mesh_link_names
        animation_poses = kin_model.get_link_poses(q_traj.position.contiguous(), animation_links)
        # add offsets for visual mesh:
        for i, ival in enumerate(offsets):
            offset_pose = Pose.from_list(ival)
            new_pose = Pose(
                animation_poses.position[:, i, :], animation_poses.quaternion[:, i, :]
            ).multiply(offset_pose)
            animation_poses.position[:, i, :] = new_pose.position
            animation_poses.quaternion[:, i, :] = new_pose.quaternion

        robot_base_frame = join_path(base_frame, robot_base_frame)

        usd_helper.create_animation(
            robot_mesh_model, animation_poses, base_frame, robot_frame=robot_base_frame
        )
        if visualize_robot_spheres:
            # visualize robot spheres:
            sphere_traj = kin_model.get_robot_as_spheres(q_traj.position)
            # change color:
            for s in sphere_traj:
                for k in s:
                    k.color = [0, 0.27, 0.27, 1.0]
            usd_helper.create_obstacle_animation(
                sphere_traj, base_frame=base_frame, obstacles_frame="curobo/robot_collision"
            )
        usd_helper.write_stage_to_file(save_path, flatten=flatten_usd)

    @staticmethod
    def load_robot(
        robot_model_file: str,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> CudaRobotModel:
        config_file = load_yaml(join_path(get_robot_configs_path(), robot_model_file))
        if "robot_cfg" in config_file:
            config_file = config_file["robot_cfg"]
        config_file["kinematics"]["load_link_names_with_mesh"] = True
        # config_file["robot_cfg"]["kinematics"]["use_usd_kinematics"] = False

        robot_cfg = CudaRobotModelConfig.from_data_dict(
            config_file["kinematics"], tensor_args=tensor_args
        )

        kin_model = CudaRobotModel(robot_cfg)
        return kin_model

    @staticmethod
    def write_trajectory_animation_with_robot_usd(
        robot_model_file: str,
        world_model: Union[WorldConfig, None],
        q_start: JointState,
        q_traj: JointState,
        dt: float = 0.02,
        save_path: str = "out.usd",
        tensor_args: TensorDeviceType = TensorDeviceType(),
        interpolation_steps: float = 1.0,
        write_robot_usd_path: str = "assets/",
        robot_base_frame: str = "robot",
        robot_usd_local_reference: str = "assets/",
        base_frame="/world",
        kin_model: Optional[CudaRobotModel] = None,
        visualize_robot_spheres: bool = True,
        robot_asset_prim_path=None,
        robot_color: Optional[List[float]] = None,
        flatten_usd: bool = False,
        goal_pose: Optional[Pose] = None,
        goal_color: Optional[List[float]] = None,
    ):
        usd_exists = False
        # if usd file doesn't exist, fall back to urdf animation script

        if kin_model is None:
            robot_model_file = load_yaml(join_path(get_robot_configs_path(), robot_model_file))
            if "robot_cfg" in robot_model_file:
                robot_model_file = robot_model_file["robot_cfg"]
            robot_model_file["kinematics"]["load_link_names_with_mesh"] = True
            robot_model_file["kinematics"]["use_usd_kinematics"] = True
            if "usd_path" in robot_model_file["kinematics"]:

                usd_exists = file_exists(
                    join_path(get_assets_path(), robot_model_file["kinematics"]["usd_path"])
                )
            else:
                usd_exists = False
        else:
            if kin_model.generator_config.usd_path is not None:
                usd_exists = file_exists(kin_model.generator_config.usd_path)
        if robot_color is not None:
            log_warn(
                "robot_color is not supported when using robot from usd, "
                + "using urdf mode instead to write usd file"
            )
            usd_exists = False
        if not usd_exists:
            log_info("robot usd not found, using urdf animation instead")
            robot_model_file["kinematics"]["use_usd_kinematics"] = False
            return UsdHelper.write_trajectory_animation(
                robot_model_file,
                world_model,
                q_start,
                q_traj,
                dt,
                save_path,
                tensor_args,
                interpolation_steps,
                robot_base_frame=robot_base_frame,
                base_frame=base_frame,
                kin_model=kin_model,
                visualize_robot_spheres=visualize_robot_spheres,
                robot_color=robot_color,
                flatten_usd=flatten_usd,
                goal_pose=goal_pose,
                goal_color=goal_color,
            )
        if kin_model is None:
            robot_cfg = CudaRobotModelConfig.from_data_dict(
                robot_model_file["kinematics"], tensor_args=tensor_args
            )

            kin_model = CudaRobotModel(robot_cfg)

        if robot_asset_prim_path is None:
            robot_asset_prim_path = kin_model.kinematics_parser.robot_prim_root

        robot_base_frame = join_path(base_frame, robot_base_frame)

        robot_usd_path = kin_model.generator_config.usd_path

        usd_helper = UsdHelper()
        usd_helper.create_stage(
            save_path,
            timesteps=q_traj.position.shape[0],
            dt=dt,
            interpolation_steps=interpolation_steps,
            base_frame=base_frame,
        )
        if world_model is not None:
            usd_helper.add_world_to_stage(world_model, base_frame=base_frame)

        animation_links = kin_model.kinematics_config.mesh_link_names
        animation_poses = kin_model.get_link_poses(q_traj.position, animation_links)

        usd_helper.create_linkpose_robot_animation(
            robot_usd_path,
            animation_links,
            kin_model.joint_names,
            animation_poses,
            local_asset_path=robot_usd_local_reference,
            write_robot_usd_path=write_robot_usd_path,
            robot_base_frame=robot_base_frame,
            robot_asset_prim_path=robot_asset_prim_path,
        )
        if visualize_robot_spheres:
            # visualize robot spheres:
            sphere_traj = kin_model.get_robot_as_spheres(q_traj.position)
            # change color:
            for s in sphere_traj:
                for k in s:
                    k.color = [0, 0.27, 0.27, 1.0]
            usd_helper.create_obstacle_animation(
                sphere_traj, base_frame=base_frame, obstacles_frame="curobo/robot_collision"
            )
        usd_helper.write_stage_to_file(save_path, flatten=flatten_usd)

    @staticmethod
    def create_grid_usd(
        usds_path: Union[str, List[str]],
        save_path: str,
        base_frame: str,
        max_envs: int,
        max_timecode: float,
        x_space: float,
        y_space: float,
        x_per_row: int,
        local_asset_path: str,
        dt: float = 0.02,
        interpolation_steps: int = 1,
        prefix_string: Optional[str] = None,
        flatten_usd: bool = False,
    ):
        # create stage:
        usd_helper = UsdHelper()
        usd_helper.create_stage(
            save_path,
            timesteps=max_timecode,
            dt=dt,
            interpolation_steps=interpolation_steps,
            base_frame=base_frame,
        )

        # read all usds:
        if isinstance(usds_path, list):
            files = usds_path
        else:
            files = get_files_from_dir(usds_path, [".usda", ".usd"], prefix_string)
        # get count and clamp to max:
        n_envs = min(len(files), max_envs)
        # create grid
        # :
        count_x = x_per_row
        count_y = int(np.ceil((n_envs) / x_per_row))
        x_set = np.linspace(0, x_space * count_x, count_x)
        y_set = np.linspace(0, y_space * count_y, count_y)
        xv, yv = np.meshgrid(x_set, y_set)
        xv = np.ravel(xv)
        yv = np.ravel(yv)

        # define prim + add reference:

        for i in range(n_envs):
            world_usd_path = files[i]
            env_base_frame = (
                base_frame + "/grid_" + get_filename(world_usd_path, remove_extension=True)
            )
            prim = usd_helper.stage.DefinePrim(env_base_frame, "Xform")
            set_prim_transform(prim, [xv[i], yv[i], 0, 1, 0, 0, 0])
            ref = prim.GetReferences()
            ref.AddReference(assetPath=join_path(local_asset_path, get_filename(world_usd_path)))

        # write usd to disk:

        usd_helper.write_stage_to_file(save_path, flatten=flatten_usd)

    def load_robot_usd(
        self,
        robot_usd_path: str,
        link_names: List[str],
        joint_names: List[str],
        robot_base_frame="/world/robot",
        write_asset_path="assets/",
        local_asset_path="assets/",
        robot_asset_prim_path="/panda",
    ):
        # copy robot prim and it's derivatives to a seperate usd:
        robot_usd_name = get_filename(robot_usd_path)

        out_path = join_path(write_asset_path, robot_usd_name)
        out_local_path = join_path(local_asset_path, robot_usd_name)
        if not file_exists(out_path) or not file_exists(out_local_path):
            robot_stage = Usd.Stage.Open(robot_usd_path)  # .Flatten()  # .Flatten()
            # set pose to zero for root prim:

            prim = robot_stage.GetPrimAtPath(robot_asset_prim_path)
            if not prim.IsValid():
                log_error(
                    "robot prim is not valid : " + robot_asset_prim_path + " " + robot_usd_path
                )
            set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0])
            robot_stage = robot_stage.Flatten()
            robot_stage.Export(out_path)
            robot_stage.Export(out_local_path)

        # create a base prim:
        prim = self.stage.DefinePrim(robot_base_frame)
        ref = prim.GetReferences()

        ref.AddReference(
            assetPath=join_path(local_asset_path, robot_usd_name), primPath=robot_asset_prim_path
        )
        link_prims, joint_prims = self.get_robot_prims(link_names, joint_names, robot_base_frame)
        return link_prims, joint_prims

    def get_robot_prims(
        self, link_names: List[str], joint_names: List[str], robot_base_path: str = "/world/robot"
    ):
        all_prims = [x for x in self.stage.Traverse()]
        joint_prims = {}
        link_prims = {}
        for j_idx, j in enumerate(joint_names):
            for k in range(len(all_prims)):
                current_prim = all_prims[k]
                prim_path = current_prim.GetPath().pathString
                if robot_base_path in prim_path and j in prim_path:
                    joint_prims[j] = current_prim
                    current_prim.GetAttribute("physics:jointEnabled").Set(False)
        for j_idx, j in enumerate(link_names):
            for k in range(len(all_prims)):
                current_prim = all_prims[k]
                prim_path = current_prim.GetPath().pathString
                if (
                    robot_base_path in prim_path
                    and j in prim_path
                    and "geometry" not in prim_path
                    and "joint" not in prim_path
                    and current_prim.GetTypeName() == "Xform"
                ):
                    link_prims[j] = current_prim

                    # stat = current_prim.GetAttribute("physics:rigidBodyEnabled")
                    current_prim.GetAttribute("physics:rigidBodyEnabled").Set(False)

        return link_prims, joint_prims

    def update_robot_joint_state(self, joint_prims: List[Usd.Prim], js: JointState, timestep: int):
        for j_idx, j in enumerate(js.joint_names):
            if timestep is not None:
                joint_prims[j].GetAttribute("drive:angular:physics:targetPosition").Set(
                    time=timestep, value=np.degrees(js.position[..., j_idx].item())
                )
            else:
                joint_prims[j].GetAttribute("drive:angular:physics:targetPosition").Set(
                    value=np.degrees(js.position[..., j_idx].item())
                )

    @staticmethod
    def write_motion_gen_log(
        result: MotionGenResult,
        robot_file: Union[str, RobotConfig],
        world_config: Union[None, WorldConfig],
        start_state: JointState,
        goal_pose: Pose,
        save_prefix: str = "log",
        write_ik: bool = False,
        write_trajopt: bool = False,
        write_graph: bool = False,
        goal_object: Optional[Obstacle] = None,
        overlay_ik: bool = False,
        overlay_trajopt: bool = False,
        visualize_robot_spheres: bool = True,
        link_spheres: Optional[torch.Tensor] = None,
        grid_space: float = 1.0,
        write_robot_usd_path="assets/",
        robot_asset_prim_path="/panda",
        fps: int = 24,
        link_poses: Optional[Dict[str, Pose]] = None,
        flatten_usd: bool = False,
    ):
        if goal_object is None:
            log_warn("Using franka gripper as goal object")
            goal_object = Mesh(
                name="target_gripper",
                file_path=join_path(
                    get_assets_path(),
                    "robot/franka_description/meshes/visual/hand.dae",
                ),
                color=[0.0, 0.8, 0.1, 1.0],
                pose=goal_pose.to_list(),
            )

        if goal_object is not None:
            goal_object.pose = np.ravel(goal_pose.tolist()).tolist()
            world_config = world_config.clone()
            world_config.add_obstacle(goal_object)
        if link_poses is not None:
            link_goals = []
            for k in link_poses.keys():
                link_goals.append(
                    Mesh(
                        name="target_" + k,
                        file_path=join_path(
                            get_assets_path(),
                            "robot/franka_description/meshes/visual/hand.dae",
                        ),
                        color=[0.0, 0.8, 0.1, 1.0],
                        pose=link_poses[k].to_list(),
                    )
                )
                world_config.add_obstacle(link_goals[-1])
        kin_model = UsdHelper.load_robot(robot_file)
        if link_spheres is not None:
            kin_model.kinematics_config.link_spheres = link_spheres
        if write_graph:
            log_error("Logging graph planner is not supported")
        if write_ik:
            x_space = y_space = grid_space
            if overlay_ik:
                x_space = y_space = 0.0

            ik_iter_steps = result.debug_info["ik_result"].debug_info["solver"]["steps"]
            # convert ik_iter to a trajectory:
            ik_iter_steps = torch.cat(
                [torch.cat(ik_iter_steps[i], dim=1) for i in range(len(ik_iter_steps))], dim=1
            )
            vis_traj = ik_iter_steps
            num_seeds, n_iters, dof = vis_traj.shape
            usd_paths = []
            for j in tqdm(range(num_seeds)):
                current_traj = vis_traj[j].view(-1, dof)[:, :]  # we remove last timestep

                usd_paths.append(save_prefix + "_ik_seed_" + str(j) + ".usd")
                UsdHelper.write_trajectory_animation_with_robot_usd(
                    robot_file,
                    world_config,
                    start_state,
                    JointState.from_position(current_traj),
                    dt=(1 / fps),
                    save_path=usd_paths[-1],
                    base_frame="/world_base_" + str(j),
                    kin_model=kin_model,
                    visualize_robot_spheres=visualize_robot_spheres,
                    write_robot_usd_path=write_robot_usd_path,
                    robot_asset_prim_path=robot_asset_prim_path,
                    flatten_usd=flatten_usd,
                )

            UsdHelper.create_grid_usd(
                usd_paths,
                save_prefix + "_grid_ik.usd",
                base_frame="/world",
                max_envs=len(usd_paths),
                max_timecode=n_iters,
                x_space=x_space,
                y_space=y_space,
                x_per_row=int(math.floor(math.sqrt(len(usd_paths)))),
                local_asset_path="",
                dt=(1.0 / fps),
            )
        if write_trajopt:
            if "trajopt_result" not in result.debug_info:
                log_warn("Trajopt result was not found in debug information")
                return
            trajectory_iter_steps = result.debug_info["trajopt_result"].debug_info["solver"][
                "steps"
            ]
            vis_traj = []
            for i in range(len(trajectory_iter_steps)):
                vis_traj += trajectory_iter_steps[i]

            full_traj = torch.cat(vis_traj, dim=0)
            num_seeds, h, dof = vis_traj[-1].shape
            n, _, _ = full_traj.shape  # this will have iterations
            full_traj = full_traj.view(-1, num_seeds, h, dof)
            n_steps = full_traj.shape[0]

            full_traj = full_traj.transpose(0, 1).contiguous()  # n_seeds, n_steps, h, dof
            n1, n2, _, _ = full_traj.shape

            full_traj = torch.cat(
                (start_state.position.view(1, 1, 1, -1).repeat(n1, n2, 1, 1), full_traj), dim=-2
            )
            usd_paths = []
            finetune_usd_paths = []
            for j in tqdm(range(num_seeds)):
                current_traj = full_traj[j].view(-1, dof)  # we remove last timestep
                # add start state to current trajectory since it's not in the optimization:
                usd_paths.append(save_prefix + "_trajopt_seed_" + str(j) + ".usd")
                UsdHelper.write_trajectory_animation_with_robot_usd(
                    robot_file,
                    world_config,
                    start_state,
                    JointState.from_position(current_traj),
                    dt=(1.0 / fps),
                    save_path=usd_paths[-1],
                    base_frame="/world_base_" + str(j),
                    kin_model=kin_model,
                    visualize_robot_spheres=visualize_robot_spheres,
                    write_robot_usd_path=write_robot_usd_path,
                    robot_asset_prim_path=robot_asset_prim_path,
                    flatten_usd=flatten_usd,
                )
            # add finetuning step:

            if "finetune_trajopt_result" in result.debug_info and True:
                finetune_iter_steps = result.debug_info["finetune_trajopt_result"].debug_info[
                    "solver"
                ]["steps"]

                vis_traj = []
                if finetune_iter_steps is not None:
                    for i in range(len(finetune_iter_steps)):
                        vis_traj += finetune_iter_steps[i]
                full_traj = torch.cat(vis_traj, dim=0)
                num_seeds, h, dof = vis_traj[-1].shape
                n, _, _ = full_traj.shape  # this will have iterations
                # print(full_traj.shape)
                full_traj = full_traj.view(-1, num_seeds, h, dof)
                n1, n2, _, _ = full_traj.shape

                full_traj = torch.cat(
                    (start_state.position.view(1, 1, 1, -1).repeat(n1, n2, 1, 1), full_traj), dim=-2
                )

                # n_steps = full_traj.shape[0]

                full_traj = full_traj.transpose(0, 1).contiguous()  # n_seeds, n_steps, h, dof
                for j in tqdm(range(num_seeds)):
                    current_traj = full_traj[j][-1].view(-1, dof)

                    finetune_usd_paths.append(save_prefix + "_finetune_seed_" + str(j) + ".usd")
                    UsdHelper.write_trajectory_animation_with_robot_usd(
                        robot_file,
                        world_config,
                        start_state,
                        JointState.from_position(current_traj),
                        dt=(1.0 / fps),
                        save_path=finetune_usd_paths[-1],
                        base_frame="/world_base_" + str(j),
                        kin_model=kin_model,
                        visualize_robot_spheres=visualize_robot_spheres,
                        write_robot_usd_path=write_robot_usd_path,
                        robot_asset_prim_path=robot_asset_prim_path,
                        flatten_usd=flatten_usd,
                    )
            x_space = y_space = grid_space
            if overlay_trajopt:
                x_space = y_space = 0.0
            if len(finetune_usd_paths) == 1:
                usd_paths.append(finetune_usd_paths[0])
            UsdHelper.create_grid_usd(
                usd_paths,
                save_prefix + "_grid_trajopt.usd",
                base_frame="/world",
                max_envs=len(usd_paths),
                max_timecode=n_steps * h,
                x_space=x_space,
                y_space=y_space,
                x_per_row=int(math.floor(math.sqrt(len(usd_paths)))),
                local_asset_path="",
                dt=(1.0 / fps),
            )
            if False and len(finetune_usd_paths) > 1:
                UsdHelper.create_grid_usd(
                    finetune_usd_paths,
                    save_prefix + "_grid_finetune_trajopt.usd",
                    base_frame="/world",
                    max_envs=len(finetune_usd_paths),
                    max_timecode=n_steps * h,
                    x_space=x_space,
                    y_space=y_space,
                    x_per_row=int(math.floor(math.sqrt(len(finetune_usd_paths)))),
                    local_asset_path="",
                    dt=(1.0 / fps),
                )
