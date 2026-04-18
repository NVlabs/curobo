# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Geometry types are defined in this module. See :ref:`scene_collision` for more information."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third Party
import numpy as np
import torch
import trimesh
import trimesh.scene

# CuRobo
from curobo._src.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo._src.types.camera import CameraObservation
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util_file import get_assets_path, join_path


@dataclass
class Material:
    """Material properties of an obstacle, useful for rendering."""

    metallic: float = 0.0
    roughness: float = 0.4


@dataclass
class Obstacle:
    """Base class for all obstacles."""

    #: Unique name of obstacle.
    name: str

    #: Pose of obstacle as a list with format [x y z qw qx qy qz]
    pose: Optional[List[float]] = None

    #: Scale obsctacle. This is only implemented for :class:`Mesh` and :class:`PointCloud`
    #: obstacles.
    scale: Optional[List[float]] = None

    #: Color of obstacle to use in visualization.
    color: Optional[List[float]] = None

    #: Texture name for the obstacle.
    texture_id: Optional[str] = None

    #: Texture to apply to obstacle in visualization.
    texture: Optional[str] = None

    #: Material properties to apply in visualization.
    material: Material = field(default_factory=Material)

    #: Device and floating point precision to use for tensors.
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)

    def get_trimesh_mesh(
        self,
        process: bool = True,
        process_color: bool = True,
        transform_with_pose: bool = False,
    ) -> trimesh.Trimesh:
        """Create a trimesh instance from the obstacle representation.

        Args:
            process (bool, optional): process when loading from file. Defaults to True.

        Raises:
            NotImplementedError: requires implementation in derived class.

        Returns:
            trimesh.Trimesh: instance of obstacle as a trimesh.
        """
        raise NotImplementedError

    def save_as_mesh(self, file_path: str, transform_with_pose: bool = False):
        """Save obstacle as a mesh file.

        Args:
            file_path: Path to save mesh file.
            transform_with_pose: Transform obstacle with pose before saving.
        """
        mesh_scene = self.get_trimesh_mesh(transform_with_pose=transform_with_pose)


        mesh_scene.export(file_path)

    def get_cuboid(self) -> Cuboid:
        """Get oriented bounding box of obstacle (OBB).

        Returns:
            Cuboid: returns obstacle as a cuboid.
        """
        # create a trimesh object:
        m = self.get_trimesh_mesh()
        # compute bounding box:
        dims = m.bounding_box_oriented.primitive.extents
        dims = dims
        offset = m.bounding_box_oriented.primitive.transform
        new_pose = self.pose

        base_pose = Pose.from_list(self.pose)
        offset_pose = Pose.from_matrix(self.device_cfg.to_device(offset))
        new_base_pose = base_pose.multiply(offset_pose)
        new_pose = new_base_pose.tolist()

        if self.color is None:
            # get average color:
            if isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
                m.visual = m.visual.to_color()
            if isinstance(m.visual, trimesh.visual.color.ColorVisuals):
                self.color = (np.ravel(m.visual.main_color) / 255.0).tolist()
        return Cuboid(
            name=self.name,
            pose=new_pose,
            dims=dims.tolist(),
            color=self.color,
            texture=self.texture,
            material=self.material,
            device_cfg=self.device_cfg,
        )

    def get_mesh(self, process: bool = True) -> Mesh:
        """Get obstacle as a mesh.

        Args:
            process (bool, optional): process mesh from file. Defaults to True.

        Returns:
            Mesh: obstacle as a mesh.
        """
        # load sphere as a mesh in trimesh:
        m = self.get_trimesh_mesh(process=process)
        if isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
            m.visual = m.visual.to_color()

        return Mesh(
            name=self.name,
            vertices=m.vertices,
            faces=m.faces,
            pose=self.pose,
            color=self.color,
            texture_id=self.texture_id,
            texture=self.texture,
            material=self.material,
            vertex_colors=m.visual.vertex_colors,
            face_colors=m.visual.face_colors,
            device_cfg=self.device_cfg,
        )

    def get_transform_matrix(self) -> np.ndarray:
        """Get homogenous transformation matrix from pose.

        Returns:
           np.ndarray : transformation matrix.
        """
        # convert pose to matrix:
        mat = trimesh.scene.transforms.kwargs_to_matrix(
            translation=self.pose[:3], quaternion=self.pose[3:]
        )
        return mat

    def get_sphere(self, n: int = 1) -> Sphere:
        """Compute a sphere that fits in the volume of the object.

        Args:
            n: number of spheres
        Returns:
            spheres
        """
        obb = self.get_cuboid()

        # from obb, compute the number of spheres

        # fit a sphere of radius equal to the smallest dim
        r = min(obb.dims)
        sph = Sphere(
            name="m_sphere",
            pose=obb.pose,
            position=obb.pose[:3],
            radius=r,
            device_cfg=self.device_cfg,
        )

        return sph

    def get_bounding_spheres(
        self,
        num_spheres: Optional[int] = None,
        surface_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.MORPHIT,
        pre_transform_pose: Optional[Pose] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> List[Sphere]:
        """Compute spheres that approximate the volume of the object.

        Args:
            num_spheres: Number of spheres.  ``None`` for automatic.
            surface_radius: Radius added to surface-sampled spheres.
            fit_type: Sphere fit algorithm.
            pre_transform_pose: Optional pose applied before the object pose.
            device_cfg: Device configuration.

        Returns:
            List of :class:`Sphere` instances.
        """
        mesh = self.get_trimesh_mesh()
        fit_result = fit_spheres_to_mesh(
            mesh, num_spheres, surface_radius=surface_radius, fit_type=fit_type,
        )
        pts = fit_result.centers
        n_radius = fit_result.radii

        obj_pose = Pose.from_list(self.pose, device_cfg)
        # transform object:

        # transform points:
        if pre_transform_pose is not None:
            obj_pose = pre_transform_pose.multiply(obj_pose)  # convert object pose to another frame

        if pts is None or len(pts) == 0:
            log_warn("spheres could not be fit!, adding one sphere at origin")
            centroid = mesh.centroid
            pts = torch.tensor([[centroid[0], centroid[1], centroid[2]]],
                               dtype=torch.float32, device=device_cfg.device)
            n_radius = torch.tensor([0.02], dtype=torch.float32, device=device_cfg.device)
            obj_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0], device_cfg)

        points_cuda = device_cfg.to_device(pts)
        pts = obj_pose.transform_points(points_cuda).cpu().view(-1, 3)
        n_radius_cpu = n_radius.cpu() if isinstance(n_radius, torch.Tensor) else n_radius

        new_spheres = [
            Sphere(
                name="sph_" + str(i),
                pose=[pts[i, 0].item(), pts[i, 1].item(), pts[i, 2].item(), 1, 0, 0, 0],
                radius=float(n_radius_cpu[i]),
            )
            for i in range(pts.shape[0])
        ]

        return new_spheres


@dataclass
class Cuboid(Obstacle):
    """Represent obstacle as a cuboid."""

    #: Dimensions of cuboid in meters [x_length, y_length, z_length].
    dims: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        """Post initialization checks if pose was set."""
        if self.pose is None:
            log_and_raise("Cuboid Obstacle requires Pose")

    def get_trimesh_mesh(
        self, process: bool = True, process_color: bool = True, transform_with_pose: bool = False
    ) -> trimesh.Trimesh:
        """Create a trimesh instance from the obstacle representation.

        Args:
            process: Flag is not used.
            process_color: Flag is not used.

        Returns:
            trimesh.Trimesh: Instance of obstacle as a trimesh.
        """
        m = trimesh.creation.box(extents=self.dims)
        if self.color is not None:
            color_visual = trimesh.visual.color.ColorVisuals(
                mesh=m, face_colors=self.color, vertex_colors=self.color
            )
            m.visual = color_visual
        if transform_with_pose:
            m.apply_transform(self.get_transform_matrix())
        return m


@dataclass
class Capsule(Obstacle):
    """Represent obstacle as a capsule."""

    #: Radius of capsule in meters.
    radius: float = 0.0

    #: Base of capsule in meters [x, y, z].
    base: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    #: Tip of capsule in meters [x, y, z].
    tip: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def get_trimesh_mesh(
        self,
        process: bool = True,
        process_color: bool = True,
        transform_with_pose: bool = False,
    ) -> trimesh.Trimesh:
        """Create a trimesh instance from the obstacle representation.

        Args:
            process: Flag is not used.
            process_color: Flag is not used.

        Returns:
            trimesh.Trimesh: Instance of obstacle as a trimesh.
        """
        height = self.tip[2] - self.base[2]
        if (
            height < 0
            or self.tip[0] != 0
            or self.tip[1] != 0
            or self.base[0] != 0
            or self.base[1] != 0
        ):
            log_and_raise(
                "Capsule to Mesh is only supported when base and tip are at xy origin with a positive height"
            )
        m = trimesh.creation.capsule(radius=self.radius, height=self.tip[2] - self.base[2])
        if self.color is not None:
            color_visual = trimesh.visual.color.ColorVisuals(
                mesh=m, face_colors=self.color, vertex_colors=self.color
            )
            m.visual = color_visual
        if transform_with_pose:
            m.apply_transform(self.get_transform_matrix())
        return m


@dataclass
class Cylinder(Obstacle):
    """Obstacle represented as a cylinder."""

    #: Radius of cylinder in meters.
    radius: float = 0.0

    #: Height of cylinder in meters.
    height: float = 0.0

    def get_trimesh_mesh(
        self, process: bool = True, process_color: bool = True, transform_with_pose: bool = False
    ) -> trimesh.Trimesh:
        """Create a trimesh instance from the obstacle representation.

        Args:
            process: Flag is not used.
            process_color: Flag is not used.

        Returns:
            trimesh.Trimesh: Instance of obstacle as a trimesh.
        """
        m = trimesh.creation.cylinder(radius=self.radius, height=self.height)
        if self.color is not None:
            color_visual = trimesh.visual.color.ColorVisuals(
                mesh=m, face_colors=self.color, vertex_colors=self.color
            )
            m.visual = color_visual
        if transform_with_pose:
            m.apply_transform(self.get_transform_matrix())
        return m


@dataclass
class Sphere(Obstacle):
    """Obstacle represented as a sphere."""

    #: Radius of sphere in meters.
    radius: float = 0.0

    #: Position is deprecated, use pose instead
    position: Optional[List[float]] = None

    def __post_init__(self):
        """Post initialization checks if position was set, logs warning to use pose instead."""
        if self.position is not None:
            self.pose = self.position + [1, 0, 0, 0]
            log_warn("Sphere.position is deprecated, use Sphere.pose instead")
        if self.pose is not None:
            self.position = self.pose[:3]

    def get_trimesh_mesh(
        self, process: bool = True, process_color: bool = True, transform_with_pose: bool = False
    ) -> trimesh.Trimesh:
        """Create a trimesh instance from the obstacle representation.

        Args:
            process: Flag is not used.
            process_color: Flag is not used.

        Returns:
            trimesh.Trimesh: Instance of obstacle as a trimesh.
        """
        m = trimesh.creation.icosphere(radius=self.radius)
        if self.color is not None:
            color_visual = trimesh.visual.color.ColorVisuals(
                mesh=m, face_colors=self.color, vertex_colors=self.color
            )
            m.visual = color_visual
        if transform_with_pose:
            m.apply_transform(self.get_transform_matrix())
        return m

    def get_cuboid(self) -> Cuboid:
        """Get oriented bounding box of obstacle (OBB).

        Returns:
            Cuboid: returns obstacle as a cuboid.
        """
        # create a trimesh object:
        m = self.get_trimesh_mesh()
        # compute bounding box:
        dims = m.bounding_box_oriented.primitive.extents
        dims = dims
        offset = m.bounding_box_oriented.primitive.transform
        new_pose = self.pose

        base_pose = Pose.from_list(self.pose)
        offset_pose = Pose.from_matrix(self.device_cfg.to_device(offset))
        new_base_pose = base_pose.multiply(offset_pose)
        new_pose = new_base_pose.tolist()

        # since a sphere is symmetrical, we set the orientation to identity
        new_pose[3:] = [1, 0, 0, 0]

        if self.color is None:
            # get average color:
            if isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
                m.visual = m.visual.to_color()
            if isinstance(m.visual, trimesh.visual.color.ColorVisuals):
                self.color = (np.ravel(m.visual.main_color) / 255.0).tolist()
        return Cuboid(
            name=self.name,
            pose=new_pose,
            dims=dims.tolist(),
            color=self.color,
            texture=self.texture,
            material=self.material,
            device_cfg=self.device_cfg,
        )


@dataclass
class Mesh(Obstacle):
    """Obstacle represented as mesh."""

    #: Path to mesh file.
    file_path: Optional[str] = None

    #: Full mesh as a string, loading from this is not implemented yet.
    file_string: Optional[str] = None

    #: Path to urdf file, does not support loading from this yet.
    urdf_path: Optional[str] = None

    #: Vertices of mesh.
    vertices: Optional[List[List[float]]] = None

    #: Faces of mesh.
    faces: Optional[List[int]] = None

    #: Vertex colors of mesh. Should be float in range of [0, 1].
    vertex_colors: Optional[List[List[float]]] = None

    #: Vertex normals of mesh.
    vertex_normals: Optional[List[List[float]]] = None

    #: Face colors of mesh. Should be float in range of [0, 1].
    face_colors: Optional[List[List[float]]] = None

    def __post_init__(self):
        """Post initialization adds absolute path to file_path and scales vertices."""
        if self.file_path is not None:
            self.file_path = join_path(get_assets_path(), self.file_path)
        if self.urdf_path is not None:
            self.urdf_path = join_path(get_assets_path(), self.urdf_path)
        if self.scale is not None and self.vertices is not None:
            self.vertices = np.ravel(self.scale) * self.vertices
            self.scale = None

    def get_trimesh_mesh(
        self, process: bool = True, process_color: bool = True, transform_with_pose: bool = False
    ) -> trimesh.Trimesh:
        """Create a trimesh instance from the obstacle representation.

        Args:
            process: process flag passed to :class:`trimesh.load`.
            process_color: if True, load mesh visual from texture.

        Returns:
            trimesh.Trimesh: Instance of obstacle as a trimesh.
        """
        # load mesh from filepath or verts and faces:
        if self.file_path is not None:
            m = trimesh.load(self.file_path, process=process, force="mesh")
            if isinstance(m, trimesh.Scene):
                m = m.dump(concatenate=True)
            if process_color and isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
                try:
                    m.visual = m.visual.to_color()
                except Exception as e:
                    log_warn("Could not convert texture to color: " + str(e))
            if self.scale is not None:
                m.vertices = np.ravel(self.scale) * m.vertices
        else:
            if self.vertices is not None and isinstance(self.vertices, torch.Tensor):
                self = self.to_cpu()

            m = trimesh.Trimesh(
                self.vertices,
                self.faces,
                vertex_colors=self.vertex_colors,
                vertex_normals=self.vertex_normals,
                face_colors=self.face_colors,
            )
        if transform_with_pose:
            m.apply_transform(self.get_transform_matrix())

        return m

    def to_cpu(self):
        """Convert the mesh to CPU memory."""
        if self.vertices is not None:
            self.vertices = self.vertices.cpu()
        if self.faces is not None:
            self.faces = self.faces.cpu()
        if self.vertex_normals is not None:
            self.vertex_normals = self.vertex_normals.cpu()
        if self.face_colors is not None:
            self.face_colors = self.face_colors.cpu()
        if self.vertex_colors is not None:
            self.vertex_colors = self.vertex_colors.cpu()
        return self

    def to_gpu(self):
        """Convert the mesh to GPU memory."""
        if self.vertex_colors is not None:
            self.vertex_colors = self.vertex_colors.cpu()
        if self.vertex_normals is not None:
            self.vertex_normals = self.vertex_normals.cpu()
        if self.face_colors is not None:
            self.face_colors = self.face_colors.cpu()
        return self

    def update_material(self):
        """Load material into vertex_colors and face_colors."""
        if (
            self.color is None
            and self.vertex_colors is None
            and self.face_colors is None
            and self.file_path is not None
        ):
            # try to load material:
            m = trimesh.load(self.file_path, process=False, force="mesh")
            if isinstance(m, trimesh.Scene):
                m = m.dump(concatenate=True)
            if isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
                m.visual = m.visual.to_color()
            if isinstance(m.visual, trimesh.visual.color.ColorVisuals):
                if isinstance(m.visual.vertex_colors[0], np.ndarray):
                    self.vertex_colors = m.visual.vertex_colors
                    self.face_colors = m.visual.face_colors
                else:
                    self.vertex_colors = [m.visual.vertex_colors for x in range(len(m.vertices))]

    def get_mesh_data(self, process: bool = True) -> Tuple[List[List[float]], List[int]]:
        """Get vertices and faces of mesh.

        Args:
            process: process flag passed to :class:`trimesh.load`.

        Returns:
            Tuple[List[List[float]], List[int]]: vertices and faces of mesh.
        """
        verts = faces = None
        if self.file_path is not None:
            m = self.get_trimesh_mesh(process=process)
            verts = m.vertices.view(np.ndarray)
            faces = m.faces
        elif self.vertices is not None and self.faces is not None:
            verts = self.vertices
            faces = self.faces
        else:
            ValueError("No Mesh object found")

        return verts, faces

    @staticmethod
    def from_pointcloud(
        pointcloud: np.ndarray,
        pitch: float = 0.02,
        name="world_pc",
        pose: List[float] = [0, 0, 0, 1, 0, 0, 0],
        filter_close_points: float = 0.0,
    ):
        """Create a mesh from a pointcloud via voxelized surface extraction.

        Voxelizes the point cloud into a 3D grid and emits quad faces on
        voxel boundaries (occupied vs. empty), producing a watertight mesh
        without requiring scikit-image or marching cubes.

        Args:
            pointcloud: Input pointcloud of shape [n_pts, 3].
            pitch: Voxel size in meters.
            name: Name to assign to created mesh.
            pose: Pose to assign to created mesh.
            filter_close_points: Filter points closer than this distance to the
                origin. Threshold is in meters and should be positive.

        Returns:
            Mesh: Mesh created from pointcloud.
        """
        if filter_close_points > 0.0:
            dist = np.linalg.norm(pointcloud, axis=-1)
            pointcloud = pointcloud[dist > filter_close_points]

        if len(pointcloud) == 0:
            return Mesh(name, pose=pose, vertices=[[0, 0, 0]], faces=[0, 0, 0])

        pts = np.asarray(pointcloud, dtype=np.float64)
        origin = pts.min(axis=0) - pitch
        ijk = np.floor((pts - origin) / pitch).astype(np.int64)

        grid_shape = ijk.max(axis=0) + 3  # +2 pad so boundary is always empty
        occupied = np.zeros(grid_shape, dtype=bool)
        occupied[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

        # For each axis direction, find faces between occupied and empty voxels
        # Face normals: +x, -x, +y, -y, +z, -z
        face_templates = np.array([
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],  # -x face
            [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],  # +x face
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],  # -y face
            [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],  # +y face
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # -z face
            [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],  # +z face
        ], dtype=np.float64)

        verts_list = []
        faces_list = []
        vi = 0  # running vertex index

        axis_shifts = [(0, 1, 0, 1), (0, -1, 0, 1), (1, 1, 2, 3), (1, -1, 2, 3),
                        (2, 1, 4, 5), (2, -1, 4, 5)]

        for axis, direction, neg_tpl, pos_tpl in axis_shifts:
            shifted = np.roll(occupied, -direction, axis=axis)
            if direction > 0:
                boundary = occupied & ~shifted
                tpl = face_templates[pos_tpl]
            else:
                boundary = occupied & ~shifted
                tpl = face_templates[neg_tpl]

            coords = np.argwhere(boundary)
            if len(coords) == 0:
                continue

            quad_verts = coords[:, None, :] + tpl[None, :, :]
            quad_verts = quad_verts.reshape(-1, 3).astype(np.float64) * pitch + origin
            n = len(coords)
            idx = np.arange(n) * 4 + vi
            tri_faces = np.column_stack([
                idx, idx + 1, idx + 2,
                idx, idx + 2, idx + 3,
            ]).reshape(-1, 3)

            verts_list.append(quad_verts)
            faces_list.append(tri_faces)
            vi += n * 4

        if not verts_list:
            return Mesh(name, pose=pose, vertices=[[0, 0, 0]], faces=[0, 0, 0])

        all_verts = np.concatenate(verts_list, axis=0)
        all_faces = np.concatenate(faces_list, axis=0)

        return Mesh(name, pose=pose, vertices=all_verts.tolist(), faces=all_faces.flatten().tolist())


@dataclass
class PointCloud(Obstacle):
    """Obstacle represented as a pointcloud."""

    #: Points of pointcloud.
    points: Union[torch.Tensor, np.ndarray, List[List[float]]] = None

    #: Features of pointcloud.
    points_features: Union[torch.Tensor, np.ndarray, List[List[float]], None] = None

    def __post_init__(self):
        """Post initialization scales points if scale is set."""
        if self.scale is not None and self.points is not None:
            self.points = np.ravel(self.scale) * self.points
            self.scale = None

    def get_trimesh_mesh(
        self,
        process: bool = True,
        process_color: bool = True,
        transform_with_pose: bool = False,
    ) -> trimesh.Trimesh:
        """Create a trimesh instance from the obstacle representation.

        Args:
            process: Not used.
            process_color: Not used.

        Returns:
            trimesh.Trimesh: Instance of obstacle as a trimesh.
        """
        points = self.points
        if isinstance(points, torch.Tensor):
            points = points.view(-1, 3).cpu().numpy()
        if isinstance(points, list):
            points = np.ndarray(points)

        mesh = Mesh.from_pointcloud(points, pose=self.pose)
        return mesh.get_trimesh_mesh(transform_with_pose=transform_with_pose)

    def get_mesh_data(self, process: bool = True) -> Tuple[List[List[float]], List[int]]:
        """Get mesh data from pointcloud.

        Args:
            process: process flag passed to :class:`trimesh.load`.

        Returns:
            verts, faces: vertices and faces of mesh.
        """
        verts = faces = None
        m = self.get_trimesh_mesh(process=process)
        verts = m.vertices.view(np.ndarray)
        faces = m.faces
        return verts, faces

    @staticmethod
    def from_camera_observation(
        camera_obs: CameraObservation,
        name: str = "pc_obstacle",
        pose: Optional[List[float]] = None,
    ) -> PointCloud:
        """Create a pointcloud from a camera observation.

        Args:
            camera_obs: Input camera observation.
            name: Name to assign to created pointcloud.
            pose: Pose to assign to created pointcloud.

        Returns:
            PointCloud: Pointcloud created from camera observation.
        """
        return PointCloud(name=name, pose=pose, points=camera_obs.get_pointcloud())


@dataclass
class VoxelGrid(Obstacle):
    """VoxelGrid representation of an obstacle. Requires voxel to contain ESDF."""

    #: Dimensions of voxel grid in meters [x_length, y_length, z_length].
    dims: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    #: Voxel size in meters.
    voxel_size: float = 0.02  # meters

    #: Feature tensor of voxel grid, typically ESDF.
    feature_tensor: Optional[torch.Tensor] = None

    #: XYZR tensor of voxel grid.
    xyzr_tensor: Optional[torch.Tensor] = None

    #: Data type of feature tensor.
    feature_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        """Post initialization checks."""
        if self.feature_tensor is not None:
            self.feature_dtype = self.feature_tensor.dtype

    def get_grid_shape(self) -> Tuple[List[int], List[float], List[float]]:
        """Get shape of voxel grid."""
        bounds = self.dims

        grid_shape = [bounds[0], bounds[1], bounds[2]]

        grid_shape = [round(x / self.voxel_size) for x in grid_shape]

        low = [-bounds[0] / 2, -bounds[1] / 2, -bounds[2] / 2]
        high = [bounds[0] / 2, bounds[1] / 2, bounds[2] / 2]

        return grid_shape, low, high

    def create_xyzr_tensor(
        self, transform_to_origin: bool = False, device_cfg: DeviceCfg = DeviceCfg()
    ) -> torch.Tensor:
        """Create XYZR tensor of voxel grid.

        Args:
            transform_to_origin: Transform points to origin.
            device_cfg: Device and floating point precision to use for tensors.

        Returns:
            xyzr_tensor: XYZR tensor of voxel grid with r referring to voxel size.
        """
        trange, low, high = self.get_grid_shape()
        inv_voxel_size = 1.0 / self.voxel_size
        x = torch.linspace(1, trange[0], trange[0], device=device_cfg.device) - round(
            (0.5 * self.dims[0]) * inv_voxel_size
        )
        y = torch.linspace(1, trange[1], trange[1], device=device_cfg.device) - round(
            (0.5 * self.dims[1]) * inv_voxel_size
        )
        z = torch.linspace(1, trange[2], trange[2], device=device_cfg.device) - round(
            (0.5 * self.dims[2]) * inv_voxel_size
        )
        x = x * self.voxel_size - 0.5 * self.voxel_size
        y = y * self.voxel_size - 0.5 * self.voxel_size
        z = z * self.voxel_size - 0.5 * self.voxel_size
        w, l, h = x.shape[0], y.shape[0], z.shape[0]
        xyz = (
            torch.stack(torch.meshgrid(x, y, z, indexing="ij")).permute((1, 2, 3, 0)).reshape(-1, 3)
        )

        if transform_to_origin:
            pose = Pose.from_list(self.pose, device_cfg=device_cfg)
            xyz = pose.transform_points(xyz.contiguous())
        r = torch.zeros_like(xyz[:, 0:1])
        xyzr = torch.cat([xyz, r], dim=1)

        return xyzr

    def get_occupied_voxels(self, feature_threshold: Optional[float] = None) -> torch.Tensor:
        """Get occupied voxels from voxel grid.

        Args:
            feature_threshold: esdf value threshold to consider as occupied.

        Returns:
            occupied voxels as a tensor of shape [occupied_voxels].
        """
        if feature_threshold is None:
            feature_threshold = -0.5 * self.voxel_size
        if self.xyzr_tensor is None or self.feature_tensor is None:
            log_and_raise("Feature tensor or xyzr tensor is empty")
        xyzr = self.xyzr_tensor.clone()
        xyzr[:, 3] = self.feature_tensor
        occupied = xyzr[self.feature_tensor > feature_threshold]
        return occupied

    def clone(self) -> VoxelGrid:
        """Clone data of voxel grid."""
        return VoxelGrid(
            name=self.name,
            pose=self.pose.copy(),
            dims=self.dims.copy(),
            feature_tensor=(
                self.feature_tensor.clone() if self.feature_tensor is not None else None
            ),
            xyzr_tensor=self.xyzr_tensor.clone() if self.xyzr_tensor is not None else None,
            feature_dtype=self.feature_dtype,
            voxel_size=self.voxel_size,
        )


@dataclass
class SceneCfg(Sequence):
    """Representation of World for use in CuRobo."""

    #: List of Sphere obstacles.
    sphere: Optional[List[Sphere]] = None

    #: List of Cuboid obstacles.
    cuboid: Optional[List[Cuboid]] = None

    #: List of Capsule obstacles.
    capsule: Optional[List[Capsule]] = None

    #: List of Cylinder obstacles.
    cylinder: Optional[List[Cylinder]] = None

    #: List of Mesh obstacles.
    mesh: Optional[List[Mesh]] = None

    #: List of ESDF voxel grid obstacles.
    voxel: Optional[List[VoxelGrid]] = None

    #: List of all obstacles in world.
    objects: Optional[List[Obstacle]] = None

    def __post_init__(self):
        """Post initialization checks, also creates a list of all obstacles."""
        # create objects list:
        if self.sphere is None:
            self.sphere = []
        if self.cuboid is None:
            self.cuboid = []
        if self.capsule is None:
            self.capsule = []
        if self.mesh is None:
            self.mesh = []
        if self.cylinder is None:
            self.cylinder = []
        if self.voxel is None:
            self.voxel = []
        if self.objects is None:
            self.objects = (
                self.sphere
                + self.cuboid
                + self.capsule
                + self.mesh
                + self.cylinder
                + self.voxel
            )

    def __len__(self) -> int:
        """Get number of obstacles."""
        return len(self.objects)

    def __getitem__(self, idx: int) -> Obstacle:
        """Get obstacle at index."""
        return self.objects[idx]

    def clone(self) -> SceneCfg:
        """Clone world configuration."""
        return SceneCfg(
            cuboid=self.cuboid.copy() if self.cuboid is not None else None,
            sphere=self.sphere.copy() if self.sphere is not None else None,
            mesh=self.mesh.copy() if self.mesh is not None else None,
            capsule=self.capsule.copy() if self.capsule is not None else None,
            cylinder=self.cylinder.copy() if self.cylinder is not None else None,
            voxel=self.voxel.copy() if self.voxel is not None else None,
        )

    @staticmethod
    def create(data_dict: Dict[str, Any]) -> SceneCfg:
        """Create SceneCfg from a dictionary.

        Args:
            data_dict: World configuration dictionary.

        Returns:
            Configured SceneCfg instance.
        """
        cuboid = None
        sphere = None
        capsule = None
        mesh = None
        cylinder = None
        voxel = None
        # load yaml:
        if "cuboid" in data_dict.keys():
            cuboid = [Cuboid(name=x, **data_dict["cuboid"][x]) for x in data_dict["cuboid"]]
        if "sphere" in data_dict.keys():
            sphere = [Sphere(name=x, **data_dict["sphere"][x]) for x in data_dict["sphere"]]
        if "mesh" in data_dict.keys():
            mesh = [Mesh(name=x, **data_dict["mesh"][x]) for x in data_dict["mesh"]]
        if "capsule" in data_dict.keys():
            capsule = [Capsule(name=x, **data_dict["capsule"][x]) for x in data_dict["capsule"]]
        if "cylinder" in data_dict.keys():
            cylinder = [Cylinder(name=x, **data_dict["cylinder"][x]) for x in data_dict["cylinder"]]
        if "voxel" in data_dict.keys():
            voxel = [VoxelGrid(name=x, **data_dict["voxel"][x]) for x in data_dict["voxel"]]

        return SceneCfg(
            cuboid=cuboid,
            sphere=sphere,
            capsule=capsule,
            cylinder=cylinder,
            mesh=mesh,
            voxel=voxel,
        )

    # load world config as obbs: convert all types to obbs
    @staticmethod
    def create_obb_world(current_world: SceneCfg) -> SceneCfg:
        """Approximate all obstcales to oriented bounding boxes."""
        sphere_obb = []
        capsule_obb = []
        cylinder_obb = []
        mesh_obb = []
        cuboid_obb = current_world.cuboid

        if current_world.capsule is not None and len(current_world.capsule) > 0:
            capsule_obb = [x.get_cuboid() for x in current_world.capsule]
        if current_world.sphere is not None and len(current_world.sphere) > 0:
            sphere_obb = [x.get_cuboid() for x in current_world.sphere]
        if current_world.cylinder is not None and len(current_world.cylinder) > 0:
            cylinder_obb = [x.get_cuboid() for x in current_world.cylinder]

        if current_world.mesh is not None and len(current_world.mesh) > 0:
            mesh_obb = [x.get_cuboid() for x in current_world.mesh]

        if current_world.voxel is not None and len(current_world.voxel) > 0:
            log_and_raise("VoxelGrid cannot be converted to obb world")

        return SceneCfg(
            cuboid=cuboid_obb + sphere_obb + capsule_obb + cylinder_obb + mesh_obb
        )

    @staticmethod
    def create_mesh_scene(current_world: SceneCfg, process: bool = False) -> SceneCfg:
        """Convert all obstacles to meshes. Does not convert :class:`VoxelGrid`.

        Args:
            current_world: Current world configuration.
            process: process flag passed to :class:`trimesh.load`.

        Returns:
            SceneCfg: World configuration with all obstacles converted to meshes.
        """
        sphere_obb = []
        capsule_obb = []
        cuboid_obb = []
        cylinder_obb = []
        if current_world.capsule is not None and len(current_world.capsule) > 0:
            capsule_obb = [x.get_mesh(process=process) for x in current_world.capsule]

        if current_world.sphere is not None and len(current_world.sphere) > 0:
            sphere_obb = [x.get_mesh(process=process) for x in current_world.sphere]

        if current_world.cuboid is not None and len(current_world.cuboid) > 0:
            cuboid_obb = [x.get_mesh(process=process) for x in current_world.cuboid]

        if current_world.cylinder is not None and len(current_world.cylinder) > 0:
            cylinder_obb = [x.get_mesh(process=process) for x in current_world.cylinder]
        if current_world.voxel is not None and len(current_world.voxel) > 0:
            log_and_raise("VoxelGrid cannot be converted to mesh world")

        return SceneCfg(
            mesh=current_world.mesh
            + sphere_obb
            + capsule_obb
            + cuboid_obb
            + cylinder_obb
        )

    @staticmethod
    def create_collision_support_world(
        current_world: SceneCfg, process: bool = True
    ) -> SceneCfg:
        """Converts all obstacles to only supported collision types.

        Args:
            current_world: Current world configuration.
            process: process flag passed to :class:`trimesh.load`.

        Returns:
            SceneCfg: World configuration with all obstacles converted to supported collision
                types.
        """
        sphere_obb = []
        capsule_obb = []
        cuboid_obb = []
        cylinder_obb = []
        if current_world.capsule is not None and len(current_world.capsule) > 0:
            capsule_obb = [x.get_mesh(process=process) for x in current_world.capsule]

        if current_world.sphere is not None and len(current_world.sphere) > 0:
            sphere_obb = [x.get_mesh(process=process) for x in current_world.sphere]

        if current_world.cuboid is not None and len(current_world.cuboid) > 0:
            cuboid_obb = current_world.cuboid

        if current_world.cylinder is not None and len(current_world.cylinder) > 0:
            cylinder_obb = [x.get_mesh(process=process) for x in current_world.cylinder]

        return SceneCfg(
            mesh=current_world.mesh + sphere_obb + capsule_obb + cylinder_obb,
            cuboid=cuboid_obb,
            voxel=current_world.voxel,
        )

    @staticmethod
    def get_scene_graph(
        current_world: SceneCfg, process_color: bool = True
    ) -> trimesh.scene.scene.Scene:
        """Get trimesh scene graph of world.

        Args:
            current_world: Current world configuration.
            process_color: Load color of meshes.

        Returns:
            trimesh.scene.scene.Scene: Scene graph of world.
        """
        m_world = SceneCfg.create_mesh_scene(current_world)
        mesh_scene = trimesh.scene.scene.Scene(base_frame="world_origin")
        mesh_list = m_world
        for m in mesh_list:
            mesh_scene.add_geometry(
                m.get_trimesh_mesh(process_color=process_color),
                geom_name=m.name,
                parent_node_name="world_origin",
                transform=m.get_transform_matrix(),
            )

        return mesh_scene

    @staticmethod
    def create_merged_mesh_world(
        current_world: SceneCfg, process: bool = True, process_color: bool = True
    ) -> SceneCfg:
        """Merge all obstacles in the world to a single mesh.

        Args:
            current_world: Current world configuration.
            process: process flag passed to :class:`trimesh.load`.
            process_color: Load color of meshes.

        Returns:
            SceneCfg: World configuration with a single merged mesh as obstacle.
        """
        mesh_scene = SceneCfg.get_scene_graph(current_world, process_color=process_color)
        mesh_scene = mesh_scene.dump(concatenate=True)
        new_mesh = Mesh(
            vertices=mesh_scene.vertices.view(np.ndarray),
            faces=mesh_scene.faces,
            name="merged_mesh",
            pose=[0, 0, 0, 1, 0, 0, 0],
        )
        return SceneCfg(mesh=[new_mesh])

    def get_obb_world(self) -> SceneCfg:
        """Get world with all obstacles as oriented bounding boxes."""
        return SceneCfg.create_obb_world(self)

    def get_mesh_world(self, merge_meshes: bool = False, process: bool = False) -> SceneCfg:
        """Get world with all obstacles as meshes."""
        if merge_meshes:
            return SceneCfg.create_merged_mesh_world(self, process=process)
        else:
            return SceneCfg.create_mesh_scene(self, process=process)

    def get_collision_check_world(self, mesh_process: bool = False) -> SceneCfg:
        """Get world with all obstacles converted to supported collision types."""
        return SceneCfg.create_collision_support_world(self, process=mesh_process)

    def save_scene_as_mesh(
        self, file_path: str, save_as_scene_graph=False, process_color: bool = True
    ):
        """Save world as a mesh file.

        Args:
            file_path: Path to save mesh file.
            save_as_scene_graph: Save as scene graph.
            process_color: Load color of meshes.
        """
        mesh_scene = SceneCfg.get_scene_graph(self, process_color=process_color)
        if save_as_scene_graph:
            mesh_scene = mesh_scene.dump(concatenate=True)

        mesh_scene.export(file_path)

    def get_cache_dict(self) -> Dict[str, int]:
        """Computes the number of obstacles in each type."""
        cache = {"obb": len(self.cuboid), "mesh": len(self.mesh)}
        return cache

    def add_obstacle(self, obstacle: Obstacle):
        """Add obstacle to world.

        Args:
            obstacle: Obstacle to add to world.
        """
        if isinstance(obstacle, Mesh):
            self.mesh.append(obstacle)
        elif isinstance(obstacle, Cuboid):
            self.cuboid.append(obstacle)
        elif isinstance(obstacle, Sphere):
            self.sphere.append(obstacle)
        elif isinstance(obstacle, Cylinder):
            self.cylinder.append(obstacle)
        elif isinstance(obstacle, Capsule):
            self.capsule.append(obstacle)
        elif isinstance(obstacle, VoxelGrid):
            self.voxel.append(obstacle)
        else:
            ValueError("Obstacle type not supported")
        self.objects.append(obstacle)

    def randomize_color(self, r=[0, 1], g=[0, 1], b=[0, 1]):
        """Randomize color of objects within the given range

        Args:
            r: range of red color.
            g: range of green color.
            b: range of blue color.
        """
        n = len(self.objects)
        r_l = np.random.uniform(r[0], r[1], n)
        g_l = np.random.uniform(g[0], g[1], n)
        b_l = np.random.uniform(b[0], b[1], n)
        for i, i_val in enumerate(self.objects):
            i_val.color = [r_l[i], g_l[i], b_l[i], 1.0]

    def add_color(self, rgba=[0.0, 0.0, 0.0, 1.0]):
        """Update color of obstacles.

        Args:
            rgba: red, green, blue, alpha values.
        """
        for i, i_val in enumerate(self.objects):
            i_val.color = rgba

    def add_material(self, material=Material()):
        """Add material to all obstacles.

        Args:
            material: material to add to obstacles.
        """
        for i, i_val in enumerate(self.objects):
            i_val.material = material

    def get_obstacle(self, name: str) -> Union[None, Obstacle]:
        """Get obstacle by name.

        Args:
            name: Name of obstacle.

        Returns:
            Obstacle with given name. If not found, returns None.
        """
        for i in self.objects:
            if i.name == name:
                return i
        return None

    def remove_obstacle(self, name: str):
        """Remove obstacle by name.

        Args:
            name: Name of obstacle to remove.
        """
        for i in range(len(self.objects)):
            if self.objects[i].name == name:
                del self.objects[i]
                return

    def remove_absolute_paths(self):
        """Remove absolute paths from file paths in obstacles. May not work on Windows."""
        for obj in self.objects:
            if obj.name.startswith("/"):
                obj.name = obj.name[1:]


def tensor_sphere(
    pt: Union[List[float], np.array, torch.Tensor],
    radius: float,
    tensor: Optional[torch.Tensor] = None,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> torch.Tensor:
    """Tensor representation of a sphere.

    Args:
        pt: Input point.
        radius: Radius of sphere.
        tensor: Tensor to update. If None, creates a new tensor.
        device_cfg: Device and floating point precision to use for tensors.

    Returns:
        tensor: Tensor representation of sphere.
    """
    if tensor is None:
        tensor = torch.zeros(4, device=device_cfg.device, dtype=device_cfg.dtype)
    tensor[:3] = torch.as_tensor(pt, device=device_cfg.device, dtype=device_cfg.dtype)
    tensor[3] = radius
    return tensor


def tensor_capsule(
    base: Union[List[float], torch.Tensor, np.array],
    tip: Union[List[float], torch.Tensor, np.array],
    radius: float,
    tensor: Optional[torch.Tensor] = None,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> torch.Tensor:
    """Tensor representation of a capsule.

    Args:
        base: Base of capsule.
        tip: Tip of capsule.
        radius: radius of capsule.
        tensor: Tensor to update. If None, creates a new tensor.
        device_cfg: Device and floating point precision to use for tensors.

    Returns:
        torch.Tensor: Tensor representation of capsule.
    """
    if tensor is None:
        tensor = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)
    tensor[:3] = torch.as_tensor(base, device=device_cfg.device, dtype=device_cfg.dtype)
    tensor[3:6] = torch.as_tensor(tip, device=device_cfg.device, dtype=device_cfg.dtype)
    tensor[6] = radius
    return tensor


def tensor_cube(
    pose: List[float], dims: List[float], device_cfg: DeviceCfg = DeviceCfg()
) -> List[torch.Tensor, torch.Tensor]:
    """Tensor representation of a cube.

    Args:
        pose: x,y,z, qw, qx, qy, qz.
        dims: length, width, height in meters. Frame is at the center of the cube.
        device_cfg: Device and floating point precision to use for tensors.

    Returns:
        List[torch.Tensor, torch.Tensor]: Tensor representation of cube, first tensor is
            dimensions and second tensor is inverse of pose.
    """
    w_T_b = Pose.from_list(pose, device_cfg=device_cfg)
    b_T_w = w_T_b.inverse()
    dims_t = torch.tensor(
        [dims[0], dims[1], dims[2]], device=device_cfg.device, dtype=device_cfg.dtype
    )
    cube = [dims_t, b_T_w.get_pose_vector()]
    return cube


def batch_tensor_cube(
    pose: List[List[float]],
    dims: List[List[float]],
    device_cfg: DeviceCfg = DeviceCfg(),
) -> List[torch.Tensor]:
    """Tensor representation of a batch of cubes
    Args:
        pose : Poses of the cubes in x,y,z, qw,qx,qy,qz.
        dims : Dimensions of the cubes. Frame is at the center of the cube.
        device_cfg: Device and floating point precision to use for tensors.

    Returns:
        List[torch.Tensor]: Tensor representation of cubes, first tensor is dimensions and
            second tensor is inverse of poses.
    """
    w_T_b = Pose.from_batch_list(pose, device_cfg=device_cfg)
    b_T_w = w_T_b.inverse()
    dims_t = torch.as_tensor(np.array(dims), device=device_cfg.device, dtype=device_cfg.dtype)
    cube = [dims_t, b_T_w.get_pose_vector()]
    return cube
