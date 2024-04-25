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
from __future__ import annotations

# Standard Library
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

# Third Party
import numpy as np
import torch
import trimesh

# CuRobo
from curobo.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.util.logger import log_error, log_warn
from curobo.util_file import get_assets_path, join_path


@dataclass
class Material:
    metallic: float = 0.0
    roughness: float = 0.4


@dataclass
class Obstacle:
    """Base class for all obstacles."""

    #: Unique name of obstacle.
    name: str

    #: Pose of obstacle as a list with format [x y z qw qx qy qz]
    pose: Optional[List[float]] = None

    #: NOTE: implement scaling for all obstacle types.
    scale: Optional[List[float]] = None

    #: Color of obstacle to use in visualization.
    color: Optional[List[float]] = None

    #: texture to apply to obstacle in visualization.
    texture_id: Optional[str] = None

    #: texture to apply to obstacle in visualization.
    texture: Optional[str] = None

    #: material properties to apply in visualization.
    material: Material = field(default_factory=Material)

    tensor_args: TensorDeviceType = field(default_factory=TensorDeviceType)

    def get_trimesh_mesh(self, process: bool = True, process_color: bool = True) -> trimesh.Trimesh:
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
        mesh_scene = self.get_trimesh_mesh()
        if transform_with_pose:
            mesh_scene.apply_transform(self.get_transform_matrix())

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
        offset_pose = Pose.from_matrix(self.tensor_args.to_device(offset))
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
            tensor_args=self.tensor_args,
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
            tensor_args=self.tensor_args,
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
            tensor_args=self.tensor_args,
        )

        return sph

    def get_bounding_spheres(
        self,
        n_spheres: int = 1,
        surface_sphere_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        pre_transform_pose: Optional[Pose] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> List[Sphere]:
        """Compute n spheres that fits in the volume of the object.

        Args:
            n: number of spheres
        Returns:
            spheres
        """
        mesh = self.get_trimesh_mesh()
        pts, n_radius = fit_spheres_to_mesh(
            mesh, n_spheres, surface_sphere_radius, fit_type, voxelize_method=voxelize_method
        )

        obj_pose = Pose.from_list(self.pose, tensor_args)
        # transform object:

        # transform points:
        if pre_transform_pose is not None:
            obj_pose = pre_transform_pose.multiply(obj_pose)  # convert object pose to another frame

        if pts is None or len(pts) == 0:
            log_warn("spheres could not be fit!, adding one sphere at origin")
            pts = np.zeros((1, 3))
            pts[0, :] = mesh.centroid
            n_radius = [0.02]
            obj_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0], tensor_args)

        points_cuda = tensor_args.to_device(pts)
        pts = obj_pose.transform_points(points_cuda).cpu().view(-1, 3).numpy()

        new_spheres = [
            Sphere(
                name="sph_" + str(i),
                pose=[pts[i, 0], pts[i, 1], pts[i, 2], 1, 0, 0, 0],
                radius=n_radius[i],
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
        if self.pose is None:
            log_error("Cuboid Obstacle requires Pose")

    def get_trimesh_mesh(self, process: bool = True, process_color: bool = True):
        m = trimesh.creation.box(extents=self.dims)
        if self.color is not None:
            color_visual = trimesh.visual.color.ColorVisuals(
                mesh=m, face_colors=self.color, vertex_colors=self.color
            )
            m.visual = color_visual
        return m


@dataclass
class Capsule(Obstacle):
    radius: float = 0.0
    base: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    tip: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def get_trimesh_mesh(self, process: bool = True, process_color: bool = True):
        height = self.tip[2] - self.base[2]
        if (
            height < 0
            or self.tip[0] != 0
            or self.tip[1] != 0
            or self.base[0] != 0
            or self.base[1] != 0
        ):
            log_error(
                "Capsule to Mesh is only supported when base and tip are at xy origin with a positive height"
            )
        m = trimesh.creation.capsule(radius=self.radius, height=self.tip[2] - self.base[2])
        if self.color is not None:
            color_visual = trimesh.visual.color.ColorVisuals(
                mesh=m, face_colors=self.color, vertex_colors=self.color
            )
            m.visual = color_visual
        return m


@dataclass
class Cylinder(Obstacle):
    radius: float = 0.0
    height: float = 0.0

    def get_trimesh_mesh(self, process: bool = True, process_color: bool = True):
        m = trimesh.creation.cylinder(radius=self.radius, height=self.height)
        if self.color is not None:
            color_visual = trimesh.visual.color.ColorVisuals(
                mesh=m, face_colors=self.color, vertex_colors=self.color
            )
            m.visual = color_visual
        return m


@dataclass
class Sphere(Obstacle):
    radius: float = 0.0

    #: position is deprecated, use pose instead
    position: Optional[List[float]] = None

    def __post_init__(self):
        if self.position is not None:
            self.pose = self.position + [1, 0, 0, 0]
            log_warn("Sphere.position is deprecated, use Sphere.pose instead")
        if self.pose is not None:
            self.position = self.pose[:3]

    def get_trimesh_mesh(self, process: bool = True, process_color: bool = True):
        m = trimesh.creation.icosphere(radius=self.radius)
        if self.color is not None:
            color_visual = trimesh.visual.color.ColorVisuals(
                mesh=m, face_colors=self.color, vertex_colors=self.color
            )
            m.visual = color_visual
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
        offset_pose = Pose.from_matrix(self.tensor_args.to_device(offset))
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
            tensor_args=self.tensor_args,
        )


@dataclass
class Mesh(Obstacle):
    file_path: Optional[str] = None
    file_string: Optional[str] = (
        None  # storing full mesh as a string, loading from this is not implemented yet.
    )
    urdf_path: Optional[str] = None  # useful for visualization in isaac gym.
    vertices: Optional[List[List[float]]] = None
    faces: Optional[List[int]] = None
    vertex_colors: Optional[List[List[float]]] = None
    vertex_normals: Optional[List[List[float]]] = None
    face_colors: Optional[List[List[float]]] = None

    def __post_init__(self):
        if self.file_path is not None:
            self.file_path = join_path(get_assets_path(), self.file_path)
        if self.urdf_path is not None:
            self.urdf_path = join_path(get_assets_path(), self.urdf_path)
        if self.scale is not None and self.vertices is not None:
            self.vertices = np.ravel(self.scale) * self.vertices
            self.scale = None

    def get_trimesh_mesh(self, process: bool = True, process_color: bool = True):
        # load mesh from filepath or verts and faces:
        if self.file_path is not None:
            m = trimesh.load(self.file_path, process=process, force="mesh")
            if isinstance(m, trimesh.Scene):
                m = m.dump(concatenate=True)
            if process_color and isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
                m.visual = m.visual.to_color()
        else:
            m = trimesh.Trimesh(
                self.vertices,
                self.faces,
                vertex_colors=self.vertex_colors,
                vertex_normals=self.vertex_normals,
                face_colors=self.face_colors,
            )
        if self.scale is not None:
            m.vertices = np.ravel(self.scale) * m.vertices
            self.scale = None
        return m

    def update_material(self):
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

    def get_mesh_data(self, process: bool = True):
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
        if filter_close_points > 0.0:
            # remove points that are closer than given threshold
            dist = np.linalg.norm(pointcloud, axis=-1)
            pointcloud = pointcloud[dist > filter_close_points]

        pc = trimesh.PointCloud(pointcloud)
        scene_mesh = trimesh.voxel.ops.points_to_marching_cubes(pc.vertices, pitch=pitch)
        return Mesh(name, pose=pose, vertices=scene_mesh.vertices, faces=scene_mesh.faces)


@dataclass
class BloxMap(Obstacle):
    map_path: Optional[str] = None
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    voxel_size: float = 0.02
    #: integrator type to use in nvblox. Options: ["tsdf", "occupancy"]
    integrator_type: str = "tsdf"
    mesh_file_path: Optional[str] = None
    mapper_instance: Any = None
    mesh: Optional[Mesh] = None

    def __post_init__(self):
        if self.map_path is not None:
            self.map_path = join_path(get_assets_path(), self.map_path)
        if self.mesh_file_path is not None:
            self.mesh = Mesh(
                name=self.name + "_mesh", file_path=self.mesh_file_path, pose=self.pose
            )

    def get_trimesh_mesh(self, process: bool = True, process_color: bool = True):
        if self.mesh is not None:
            return self.mesh.get_trimesh_mesh(process)
        else:
            log_warn("no mesh found for obstacle: " + self.name)
            return None


@dataclass
class PointCloud(Obstacle):
    points: Union[torch.Tensor, np.ndarray, List[List[float]]] = None
    points_features: Union[torch.Tensor, np.ndarray, List[List[float]], None] = None

    def __post_init__(self):
        if self.scale is not None and self.points is not None:
            self.points = np.ravel(self.scale) * self.points
            self.scale = None

    def get_trimesh_mesh(self, process: bool = True, process_color: bool = True):
        points = self.points
        if isinstance(points, torch.Tensor):
            points = points.view(-1, 3).cpu().numpy()
        if isinstance(points, list):
            points = np.ndarray(points)

        mesh = Mesh.from_pointcloud(points, pose=self.pose)
        return mesh.get_trimesh_mesh()

    def get_mesh_data(self, process: bool = True):
        verts = faces = None
        m = self.get_trimesh_mesh(process=process)
        verts = m.vertices.view(np.ndarray)
        faces = m.faces
        return verts, faces

    @staticmethod
    def from_camera_observation(
        camera_obs: CameraObservation, name: str = "pc_obstacle", pose: Optional[List[float]] = None
    ):
        return PointCloud(name=name, pose=pose, points=camera_obs.get_pointcloud())

    def get_bounding_spheres(
        self,
        n_spheres: int = 1,
        surface_sphere_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        pre_transform_pose: Optional[Pose] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> List[Sphere]:
        """Compute n spheres that fits in the volume of the object.

        Args:
            n: number of spheres
        Returns:
            spheres
        """
        # sample points in pointcloud:

        # mesh = self.get_trimesh_mesh()
        # pts, n_radius = fit_spheres_to_mesh(
        #     mesh, n_spheres, surface_sphere_radius, fit_type, voxelize_method=voxelize_method
        # )

        obj_pose = Pose.from_list(self.pose, tensor_args)
        # transform object:

        # transform points:
        if pre_transform_pose is not None:
            obj_pose = pre_transform_pose.multiply(obj_pose)  # convert object pose to another frame

        if pts is None or len(pts) == 0:
            log_warn("spheres could not be fit!, adding one sphere at origin")
            pts = np.zeros((1, 3))
            pts[0, :] = mesh.centroid
            n_radius = [0.02]
            obj_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0], tensor_args)

        points_cuda = tensor_args.to_device(pts)
        pts = obj_pose.transform_points(points_cuda).cpu().view(-1, 3).numpy()

        new_spheres = [
            Sphere(
                name="sph_" + str(i),
                pose=[pts[i, 0], pts[i, 1], pts[i, 2], 1, 0, 0, 0],
                radius=n_radius[i],
            )
            for i in range(pts.shape[0])
        ]

        return new_spheres


@dataclass
class VoxelGrid(Obstacle):
    dims: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    voxel_size: float = 0.02  # meters
    feature_tensor: Optional[torch.Tensor] = None
    xyzr_tensor: Optional[torch.Tensor] = None
    feature_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.feature_tensor is not None:
            self.feature_dtype = self.feature_tensor.dtype

    def get_grid_shape(self):
        bounds = self.dims
        low = [-bounds[0] / 2, -bounds[1] / 2, -bounds[2] / 2]
        high = [bounds[0] / 2, bounds[1] / 2, bounds[2] / 2]
        grid_shape = [
            1 + int(high[i] / self.voxel_size) - (int(low[i] / self.voxel_size))
            for i in range(len(low))
        ]
        return grid_shape, low, high

    def create_xyzr_tensor(
        self, transform_to_origin: bool = False, tensor_args: TensorDeviceType = TensorDeviceType()
    ):
        trange, low, high = self.get_grid_shape()

        x = torch.linspace(low[0], high[0], trange[0], device=tensor_args.device)
        y = torch.linspace(low[1], high[1], trange[1], device=tensor_args.device)
        z = torch.linspace(low[2], high[2], trange[2], device=tensor_args.device)
        w, l, h = x.shape[0], y.shape[0], z.shape[0]
        xyz = (
            torch.stack(torch.meshgrid(x, y, z, indexing="ij")).permute((1, 2, 3, 0)).reshape(-1, 3)
        )

        if transform_to_origin:
            pose = Pose.from_list(self.pose, tensor_args=tensor_args)
            xyz = pose.transform_points(xyz.contiguous())
        r = torch.zeros_like(xyz[:, 0:1]) + (self.voxel_size * 0.5)
        xyzr = torch.cat([xyz, r], dim=1)

        return xyzr

    def get_occupied_voxels(self, feature_threshold: Optional[float] = None):
        if feature_threshold is None:
            feature_threshold = -0.5 * self.voxel_size
        if self.xyzr_tensor is None or self.feature_tensor is None:
            log_error("Feature tensor or xyzr tensor is empty")
        xyzr = self.xyzr_tensor.clone()
        xyzr[:, 3] = self.feature_tensor
        occupied = xyzr[self.feature_tensor > feature_threshold]
        return occupied

    def clone(self):
        return VoxelGrid(
            name=self.name,
            pose=self.pose.copy(),
            dims=self.dims.copy(),
            feature_tensor=self.feature_tensor.clone() if self.feature_tensor is not None else None,
            xyzr_tensor=self.xyzr_tensor.clone() if self.xyzr_tensor is not None else None,
            feature_dtype=self.feature_dtype,
            voxel_size=self.voxel_size,
        )


@dataclass
class WorldConfig(Sequence):
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

    #: BloxMap obstacle.
    blox: Optional[List[BloxMap]] = None

    voxel: Optional[List[VoxelGrid]] = None

    #: List of all obstacles in world.
    objects: Optional[List[Obstacle]] = None

    def __post_init__(self):
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
        if self.blox is None:
            self.blox = []
        if self.voxel is None:
            self.voxel = []
        if self.objects is None:
            self.objects = (
                self.sphere
                + self.cuboid
                + self.capsule
                + self.mesh
                + self.cylinder
                + self.blox
                + self.voxel
            )

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        return self.objects[idx]

    def clone(self):
        return WorldConfig(
            cuboid=self.cuboid.copy() if self.cuboid is not None else None,
            sphere=self.sphere.copy() if self.sphere is not None else None,
            mesh=self.mesh.copy() if self.mesh is not None else None,
            capsule=self.capsule.copy() if self.capsule is not None else None,
            cylinder=self.cylinder.copy() if self.cylinder is not None else None,
            blox=self.blox.copy() if self.blox is not None else None,
            voxel=self.voxel.copy() if self.voxel is not None else None,
        )

    @staticmethod
    def from_dict(data_dict: Dict[str, Any]) -> WorldConfig:
        cuboid = None
        sphere = None
        capsule = None
        mesh = None
        blox = None
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
        if "blox" in data_dict.keys():
            blox = [BloxMap(name=x, **data_dict["blox"][x]) for x in data_dict["blox"]]
        if "voxel" in data_dict.keys():
            voxel = [VoxelGrid(name=x, **data_dict["voxel"][x]) for x in data_dict["voxel"]]

        return WorldConfig(
            cuboid=cuboid,
            sphere=sphere,
            capsule=capsule,
            cylinder=cylinder,
            mesh=mesh,
            blox=blox,
            voxel=voxel,
        )

    # load world config as obbs: convert all types to obbs
    @staticmethod
    def create_obb_world(current_world: WorldConfig):
        sphere_obb = []
        capsule_obb = []
        cylinder_obb = []
        mesh_obb = []
        blox_obb = []
        cuboid_obb = current_world.cuboid

        if current_world.capsule is not None and len(current_world.capsule) > 0:
            capsule_obb = [x.get_cuboid() for x in current_world.capsule]
        if current_world.sphere is not None and len(current_world.sphere) > 0:
            sphere_obb = [x.get_cuboid() for x in current_world.sphere]
        if current_world.cylinder is not None and len(current_world.cylinder) > 0:
            cylinder_obb = [x.get_cuboid() for x in current_world.cylinder]
        if current_world.blox is not None and len(current_world.blox) > 0:
            for i in range(len(current_world.blox)):
                if current_world.blox[i].mesh is not None:
                    blox_obb.append(current_world.blox[i].get_cuboid())

        if current_world.mesh is not None and len(current_world.mesh) > 0:
            mesh_obb = [x.get_cuboid() for x in current_world.mesh]

        if current_world.voxel is not None and len(current_world.voxel) > 0:
            log_error("VoxelGrid cannot be converted to obb world")

        return WorldConfig(
            cuboid=cuboid_obb + sphere_obb + capsule_obb + cylinder_obb + mesh_obb + blox_obb
        )

    @staticmethod
    def create_mesh_world(current_world: WorldConfig, process: bool = False):
        sphere_obb = []
        capsule_obb = []
        cuboid_obb = []
        cylinder_obb = []
        blox_obb = []
        if current_world.capsule is not None and len(current_world.capsule) > 0:
            capsule_obb = [x.get_mesh(process=process) for x in current_world.capsule]

        if current_world.sphere is not None and len(current_world.sphere) > 0:
            sphere_obb = [x.get_mesh(process=process) for x in current_world.sphere]

        if current_world.cuboid is not None and len(current_world.cuboid) > 0:
            cuboid_obb = [x.get_mesh(process=process) for x in current_world.cuboid]

        if current_world.cylinder is not None and len(current_world.cylinder) > 0:
            cylinder_obb = [x.get_mesh(process=process) for x in current_world.cylinder]
        if current_world.blox is not None and len(current_world.blox) > 0:
            for i in range(len(current_world.blox)):
                if current_world.blox[i].mesh is not None:
                    blox_obb.append(current_world.blox[i].get_mesh(process=process))
        if current_world.voxel is not None and len(current_world.voxel) > 0:
            log_error("VoxelGrid cannot be converted to mesh world")

        return WorldConfig(
            mesh=current_world.mesh
            + sphere_obb
            + capsule_obb
            + cuboid_obb
            + cylinder_obb
            + blox_obb
        )

    @staticmethod
    def create_collision_support_world(current_world: WorldConfig, process: bool = True):
        sphere_obb = []
        capsule_obb = []
        cuboid_obb = []
        cylinder_obb = []
        blox_obb = []
        if current_world.capsule is not None and len(current_world.capsule) > 0:
            capsule_obb = [x.get_mesh(process=process) for x in current_world.capsule]

        if current_world.sphere is not None and len(current_world.sphere) > 0:
            sphere_obb = [x.get_mesh(process=process) for x in current_world.sphere]

        if current_world.cuboid is not None and len(current_world.cuboid) > 0:
            cuboid_obb = current_world.cuboid

        if current_world.cylinder is not None and len(current_world.cylinder) > 0:
            cylinder_obb = [x.get_mesh(process=process) for x in current_world.cylinder]
        if current_world.blox is not None and len(current_world.blox) > 0:
            for i in range(len(current_world.blox)):
                if current_world.blox[i].mesh is not None:
                    blox_obb.append(current_world.blox[i].get_mesh(process=process))

        return WorldConfig(
            mesh=current_world.mesh + sphere_obb + capsule_obb + cylinder_obb + blox_obb,
            cuboid=cuboid_obb,
            voxel=current_world.voxel,
        )

    @staticmethod
    def get_scene_graph(current_world: WorldConfig, process_color: bool = True):
        m_world = WorldConfig.create_mesh_world(current_world)
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
        current_world: WorldConfig, process: bool = True, process_color: bool = True
    ):
        mesh_scene = WorldConfig.get_scene_graph(current_world, process_color=process_color)
        mesh_scene = mesh_scene.dump(concatenate=True)
        new_mesh = Mesh(
            vertices=mesh_scene.vertices.view(np.ndarray),
            faces=mesh_scene.faces,
            name="merged_mesh",
            pose=[0, 0, 0, 1, 0, 0, 0],
        )
        return WorldConfig(mesh=[new_mesh])

    def get_obb_world(self):
        return WorldConfig.create_obb_world(self)

    def get_mesh_world(self, merge_meshes: bool = False, process: bool = False):
        if merge_meshes:
            return WorldConfig.create_merged_mesh_world(self, process=process)
        else:
            return WorldConfig.create_mesh_world(self, process=process)

    def get_collision_check_world(self, mesh_process: bool = False):
        return WorldConfig.create_collision_support_world(self, process=mesh_process)

    def save_world_as_mesh(
        self, file_path: str, save_as_scene_graph=False, process_color: bool = True
    ):
        mesh_scene = WorldConfig.get_scene_graph(self, process_color=process_color)
        if save_as_scene_graph:
            mesh_scene = mesh_scene.dump(concatenate=True)

        mesh_scene.export(file_path)

    def get_cache_dict(self) -> Dict[str, int]:
        """Computes the number of obstacles in each type

        Returns:
            _description_
        """
        cache = {"obb": len(self.cuboid), "mesh": len(self.mesh)}
        return cache

    def add_obstacle(self, obstacle: Obstacle):
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
            r: _description_. Defaults to [0,1].
            g: _description_. Defaults to [0,1].
            b: _description_. Defaults to [0,1].

        Returns:
            _description_
        """
        n = len(self.objects)
        r_l = np.random.uniform(r[0], r[1], n)
        g_l = np.random.uniform(g[0], g[1], n)
        b_l = np.random.uniform(b[0], b[1], n)
        for i, i_val in enumerate(self.objects):
            i_val.color = [r_l[i], g_l[i], b_l[i], 1.0]

    def add_color(self, rgba=[0.0, 0.0, 0.0, 1.0]):
        for i, i_val in enumerate(self.objects):
            i_val.color = rgba

    def add_material(self, material=Material()):
        for i, i_val in enumerate(self.objects):
            i_val.material = material

    def get_obstacle(self, name: str) -> Union[None, Obstacle]:
        for i in self.objects:
            if i.name == name:
                return i
        return None

    def remove_obstacle(self, name: str):
        for i in range(len(self.objects)):
            if self.objects[i].name == name:
                del self.objects[i]
                return

    def remove_absolute_paths(self) -> WorldConfig:
        for obj in self.objects:
            if obj.name.startswith("/"):
                obj.name = obj.name[1:]


def tensor_sphere(pt, radius, tensor=None, tensor_args=TensorDeviceType()):
    if tensor is None:
        tensor = torch.empty(4, device=tensor_args.device, dtype=tensor_args.dtype)
    tensor[:3] = torch.as_tensor(pt, device=tensor_args.device, dtype=tensor_args.dtype)
    tensor[3] = radius
    return tensor


def tensor_capsule(base, tip, radius, tensor=None, tensor_args=TensorDeviceType()):
    if tensor is None:
        tensor = torch.empty(7, device=tensor_args.device, dtype=tensor_args.dtype)
    tensor[:3] = torch.as_tensor(base, device=tensor_args.device, dtype=tensor_args.dtype)
    tensor[3:6] = torch.as_tensor(tip, device=tensor_args.device, dtype=tensor_args.dtype)
    tensor[6] = radius
    return tensor


def tensor_cube(pose, dims, tensor_args=TensorDeviceType()):
    """

    Args:
        pose (_type_): x,y,z, qw,qx,qy,qz
        dims (_type_): _description_
        tensor_args (_type_, optional): _description_. Defaults to TensorDeviceType().

    Returns:
        _type_: _description_
    """
    w_T_b = Pose.from_list(pose, tensor_args=tensor_args)
    b_T_w = w_T_b.inverse()
    dims_t = torch.tensor(
        [dims[0], dims[1], dims[2]], device=tensor_args.device, dtype=tensor_args.dtype
    )
    cube = [dims_t, b_T_w.get_pose_vector()]
    return cube


def batch_tensor_cube(pose, dims, tensor_args=TensorDeviceType()):
    """

    Args:
        pose (_type_): x,y,z, qw,qx,qy,qz
        dims (_type_): _description_
        tensor_args (_type_, optional): _description_. Defaults to TensorDeviceType().

    Returns:
        _type_: _description_
    """
    w_T_b = Pose.from_batch_list(pose, tensor_args=tensor_args)
    b_T_w = w_T_b.inverse()
    dims_t = torch.as_tensor(np.array(dims), device=tensor_args.device, dtype=tensor_args.dtype)
    cube = [dims_t, b_T_w.get_pose_vector()]
    return cube
