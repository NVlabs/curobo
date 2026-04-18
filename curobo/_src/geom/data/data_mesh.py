# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""GPU data storage and SDF functions for mesh obstacles with Warp BVH acceleration.

This module provides:
- MeshData: Python dataclass owning GPU tensors and Warp mesh cache
- MeshDataWarp: Warp struct for kernel access
- is_obs_enabled, load_obstacle_transform, compute_local_sdf,
  compute_local_sdf_with_grad: Generic kernel overloads
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Third Party
import numpy as np
import torch
import warp as wp

# CuRobo
from curobo._src.geom.data.helper_pose import (
    get_obs_idx,
    load_transform_from_inv_pose,
)
from curobo._src.geom.types import Mesh, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.warp import init_warp, warp_support_bvh_constructor_type

# =============================================================================
# Warp Mesh Cache Helper
# =============================================================================


@dataclass(frozen=True)
class WarpMeshCache:
    """Cached Warp mesh data with BVH acceleration structure.

    This is a frozen dataclass to ensure mesh data is not accidentally modified
    after loading to GPU.
    """

    #: Name of the mesh.
    name: str

    #: Mesh ID, created by Warp once mesh is loaded to device.
    mesh_id: int

    #: Vertices of the mesh as Warp array.
    vertices: wp.array

    #: Faces of the mesh as Warp array.
    faces: wp.array

    #: Warp mesh instance with BVH acceleration structure.
    mesh: wp.Mesh

    def get_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get axis-aligned bounding box of the mesh."""
        torch_verts = wp.to_torch(self.mesh.points)
        min_vert = torch.min(torch_verts, dim=0).values
        max_vert = torch.max(torch_verts, dim=0).values
        return min_vert, max_vert


# =============================================================================
# Python Dataclass
# =============================================================================


@dataclass
class MeshData:
    """GPU tensor storage for mesh obstacles with Warp BVH acceleration.

    This dataclass owns the GPU tensor buffers for mesh obstacles and provides
    methods for adding, updating, and querying meshes. It can be shared between
    multiple consumers (collision checker, TSDF integrator, visualization) to
    avoid duplicate GPU allocations and ensure synchronized pose updates.

    The Warp mesh cache is shared across environments - if the same mesh geometry
    appears in multiple environments, it's loaded to GPU once and referenced by ID.

    Tensor layouts:
        - mesh_ids: (num_envs, max_n) - int64 Warp mesh handles
        - dims: (num_envs, max_n, 4) - bounding box dimensions [x, y, z, pad]
        - inv_pose: (num_envs, max_n, 8) - inverse pose [x, y, z, qw, qx, qy, qz, pad]
        - enable: (num_envs, max_n) - uint8 flags
        - count: (num_envs,) - int32 active counts
    """

    #: Warp mesh IDs for BVH queries.
    mesh_ids: torch.Tensor

    #: Mesh bounding box dimensions [x, y, z, pad] in local frame.
    dims: torch.Tensor

    #: Inverse pose [x, y, z, qw, qx, qy, qz] with padding.
    inv_pose: torch.Tensor

    #: Enable flag per mesh. 1 = active, 0 = disabled.
    enable: torch.Tensor

    #: Number of active meshes per environment.
    count: torch.Tensor

    #: Name lookup per environment. names[env_idx][mesh_idx] = name.
    names: List[List[Optional[str]]]

    #: Shared Warp mesh cache. Maps mesh name to WarpMeshCache.
    wp_cache: Dict[str, WarpMeshCache]

    #: Maximum number of meshes per environment.
    max_n: int

    #: Number of environments.
    num_envs: int

    #: Device configuration.
    device_cfg: DeviceCfg

    #: Maximum query distance for mesh SDF computation.
    max_dist: float = 0.1

    #: Warp device for mesh loading.
    _wp_device: wp.context.Device = field(repr=False, default=None)

    def __post_init__(self):
        """Initialize Warp device after dataclass creation."""
        if self._wp_device is None:
            object.__setattr__(
                self, "_wp_device", wp.torch.device_from_torch(self.device_cfg.device)
            )

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def create_cache(
        cls,
        max_n: int,
        num_envs: int,
        device_cfg: DeviceCfg,
        max_dist: float = 0.1,
    ) -> MeshData:
        """Create an empty cache for mesh obstacles.

        Args:
            max_n: Maximum number of meshes per environment.
            num_envs: Number of environments.
            device_cfg: Device and dtype configuration.
            max_dist: Maximum query distance for mesh SDF.

        Returns:
            Empty MeshData with allocated GPU buffers.
        """
        init_warp()

        mesh_ids = torch.zeros((num_envs, max_n), device=device_cfg.device, dtype=torch.int64)
        dims = torch.zeros(
            (num_envs, max_n, 4),
            dtype=device_cfg.dtype,
            device=device_cfg.device,
        )
        inv_pose = torch.zeros(
            (num_envs, max_n, 8),
            dtype=device_cfg.dtype,
            device=device_cfg.device,
        )
        inv_pose[..., 3] = 1.0  # Identity quaternion (qw=1)

        enable = torch.zeros((num_envs, max_n), dtype=torch.uint8, device=device_cfg.device)
        count = torch.zeros((num_envs,), device=device_cfg.device, dtype=torch.int32)
        names = [[None for _ in range(max_n)] for _ in range(num_envs)]
        wp_cache: Dict[str, WarpMeshCache] = {}

        return cls(
            mesh_ids=mesh_ids,
            dims=dims,
            inv_pose=inv_pose,
            enable=enable,
            count=count,
            names=names,
            wp_cache=wp_cache,
            max_n=max_n,
            num_envs=num_envs,
            device_cfg=device_cfg,
            max_dist=max_dist,
        )

    @classmethod
    def from_scene_cfg(
        cls,
        scene_cfg: SceneCfg,
        device_cfg: DeviceCfg,
        env_idx: int = 0,
        num_envs: int = 1,
        max_n: Optional[int] = None,
        max_dist: float = 0.1,
    ) -> MeshData:
        """Create MeshData from a scene configuration.

        Args:
            scene_cfg: Scene configuration containing mesh obstacles.
            device_cfg: Device and dtype configuration.
            env_idx: Environment index to load the meshes into.
            num_envs: Number of environments.
            max_n: Maximum meshes to allocate. If None, uses len(scene_cfg.mesh).
            max_dist: Maximum query distance for mesh SDF.

        Returns:
            MeshData populated with meshes from the scene config.
        """
        meshes = scene_cfg.mesh
        num_meshes = len(meshes) if meshes else 0

        if max_n is None:
            max_n = max(num_meshes, 1)

        instance = cls.create_cache(max_n, num_envs, device_cfg, max_dist)

        if num_meshes > 0:
            instance.load_batch(meshes, env_idx)

        return instance

    @classmethod
    def from_batch_scene_cfg(
        cls,
        scene_cfg_list: List[SceneCfg],
        device_cfg: DeviceCfg,
        max_n: Optional[int] = None,
        max_dist: float = 0.1,
    ) -> MeshData:
        """Create MeshData from a list of scene configurations.

        Each scene config is loaded into a separate environment. Mesh geometry
        is shared across environments when the same mesh name appears.

        Args:
            scene_cfg_list: List of scene configurations, one per environment.
            device_cfg: Device and dtype configuration.
            max_n: Maximum meshes per environment. If None, uses max across scenes.
            max_dist: Maximum query distance for mesh SDF.

        Returns:
            MeshData populated with meshes from all scene configs.
        """
        num_envs = len(scene_cfg_list)
        mesh_counts = [len(cfg.mesh) if cfg.mesh else 0 for cfg in scene_cfg_list]

        if max_n is None:
            max_n = max(mesh_counts) if mesh_counts else 1

        instance = cls.create_cache(max_n, num_envs, device_cfg, max_dist)

        for env_idx, scene_cfg in enumerate(scene_cfg_list):
            if scene_cfg.mesh:
                instance.load_batch(scene_cfg.mesh, env_idx)

        return instance

    # -------------------------------------------------------------------------
    # Load / Add Methods
    # -------------------------------------------------------------------------

    def _load_mesh_to_warp(self, mesh: Mesh) -> WarpMeshCache:
        """Load a cuRobo mesh into Warp with BVH acceleration."""
        verts, faces = mesh.get_mesh_data()
        v = wp.array(verts, dtype=wp.vec3, device=self._wp_device)
        f = wp.array(np.ravel(faces), dtype=int, device=self._wp_device)

        if warp_support_bvh_constructor_type():
            new_mesh = wp.Mesh(points=v, indices=f, bvh_constructor="sah")
        else:
            new_mesh = wp.Mesh(points=v, indices=f)

        return WarpMeshCache(mesh.name, new_mesh.id, v, f, new_mesh)

    def _load_mesh_into_cache(self, mesh: Mesh) -> WarpMeshCache:
        """Load a mesh into the Warp cache, reusing existing if already loaded."""
        if mesh.name not in self.wp_cache:
            self.wp_cache[mesh.name] = self._load_mesh_to_warp(mesh)
        else:
            log_warn(f"Mesh already in cache, reusing existing instance: {mesh.name}")
        return self.wp_cache[mesh.name]

    def load_batch(self, meshes: List[Mesh], env_idx: int) -> None:
        """Load a batch of meshes into an environment, replacing existing ones.

        Args:
            meshes: List of Mesh objects to load.
            env_idx: Environment index to load into.
        """
        num_meshes = len(meshes)
        if num_meshes > self.max_n:
            log_and_raise(f"Cannot load {num_meshes} meshes, max cache size is {self.max_n}")

        if num_meshes == 0:
            self.enable[env_idx, :] = 0
            self.count[env_idx] = 0
            return

        name_list = []
        pose_list = []
        id_list = torch.zeros((num_meshes,), device=self.device_cfg.device, dtype=torch.int64)
        dims_list = torch.zeros(
            (num_meshes, 3), device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )

        for i, mesh in enumerate(meshes):
            mesh_data = self._load_mesh_into_cache(mesh)
            pose_list.append(mesh.pose)
            id_list[i] = mesh_data.mesh_id
            name_list.append(mesh_data.name)
            # Compute bounding box dimensions from mesh bounds
            lower, upper = mesh_data.get_bounds()
            dims_list[i] = upper - lower

        pose_buffer = Pose.from_batch_list(pose_list, self.device_cfg)
        inv_pose_buffer = pose_buffer.inverse()

        self.mesh_ids[env_idx, :num_meshes] = id_list
        self.dims[env_idx, :num_meshes, :3] = dims_list
        self.inv_pose[env_idx, :num_meshes, :7] = inv_pose_buffer.get_pose_vector()
        self.enable[env_idx, :num_meshes] = 1
        self.enable[env_idx, num_meshes:] = 0
        self.names[env_idx][:num_meshes] = name_list
        self.count[env_idx] = num_meshes

    def add(self, mesh: Mesh, env_idx: int = 0) -> int:
        """Add a single mesh to an environment.

        Args:
            mesh: Mesh object to add.
            env_idx: Environment index to add to.

        Returns:
            Index of the added mesh in the environment.

        Raises:
            RuntimeError: If cache is full.
        """
        current_count = int(self.count[env_idx].item())
        if current_count >= self.max_n:
            log_and_raise(f"Cannot add mesh, cache is full ({self.max_n} meshes)")

        mesh_data = self._load_mesh_into_cache(mesh)
        w_obj_pose = Pose.from_list(mesh.pose, self.device_cfg)
        obj_w_pose = w_obj_pose.inverse()

        # Compute bounding box dimensions from mesh bounds
        lower, upper = mesh_data.get_bounds()
        mesh_dims = upper - lower

        self.mesh_ids[env_idx, current_count] = mesh_data.mesh_id
        self.dims[env_idx, current_count, :3] = mesh_dims
        self.inv_pose[env_idx, current_count, :7] = obj_w_pose.get_pose_vector()
        self.enable[env_idx, current_count] = 1
        self.names[env_idx][current_count] = mesh_data.name
        self.count[env_idx] += 1

        return current_count

    # -------------------------------------------------------------------------
    # Update Methods
    # -------------------------------------------------------------------------

    def update_pose(
        self,
        name: str,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        env_idx: int = 0,
    ) -> None:
        """Update the pose of an existing mesh.

        Args:
            name: Name of the mesh to update.
            w_obj_pose: New pose in world frame.
            obj_w_pose: New inverse pose. Used if w_obj_pose is None.
            env_idx: Environment index.

        Raises:
            RuntimeError: If mesh with given name not found.
        """
        if w_obj_pose is None and obj_w_pose is None:
            log_and_raise("Either w_obj_pose or obj_w_pose must be provided")

        idx = self.get_idx(name, env_idx)

        if w_obj_pose is not None:
            obj_w_pose = w_obj_pose.inverse()

        self.inv_pose[env_idx, idx, :7] = obj_w_pose.get_pose_vector()

    def update_from_warp_id(
        self,
        warp_mesh_id: int,
        name: str,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        env_idx: int = 0,
        mesh_idx: Optional[int] = None,
    ) -> None:
        """Update or add a mesh using a Warp mesh ID directly.

        This is useful when mesh geometry is created externally (e.g., from depth images).

        Args:
            warp_mesh_id: Warp mesh ID from wp.Mesh.id.
            name: Name for the mesh.
            w_obj_pose: Pose in world frame.
            obj_w_pose: Inverse pose. Used if w_obj_pose is None.
            env_idx: Environment index.
            mesh_idx: Index to update. If None, uses name lookup or adds new.
        """
        if w_obj_pose is None and obj_w_pose is None:
            log_and_raise("Either w_obj_pose or obj_w_pose must be provided")

        if w_obj_pose is not None:
            obj_w_pose = w_obj_pose.inverse()

        if mesh_idx is None:
            if name in self.names[env_idx]:
                mesh_idx = self.names[env_idx].index(name)
            else:
                mesh_idx = int(self.count[env_idx].item())
                if mesh_idx >= self.max_n:
                    log_and_raise("Cannot add mesh, cache is full")
                self.count[env_idx] = mesh_idx + 1

        self.mesh_ids[env_idx, mesh_idx] = warp_mesh_id
        self.inv_pose[env_idx, mesh_idx, :7] = obj_w_pose.get_pose_vector()
        self.enable[env_idx, mesh_idx] = 1
        self.names[env_idx][mesh_idx] = name

    def set_enabled(self, name: str, enabled: bool, env_idx: int = 0) -> None:
        """Enable or disable a mesh for collision checking."""
        idx = self.get_idx(name, env_idx)
        self.enable[env_idx, idx] = int(enabled)

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_idx(self, name: str, env_idx: int = 0) -> int:
        """Get the index of a mesh by name."""
        try:
            return self.names[env_idx].index(name)
        except ValueError:
            log_and_raise(f"Mesh with name '{name}' not found in environment {env_idx}")

    def get_active_count(self, env_idx: int = 0) -> int:
        """Get the number of active meshes in an environment."""
        return int(self.count[env_idx].item())

    def get_names(self, env_idx: int = 0) -> List[str]:
        """Get the names of all meshes in an environment."""
        count = int(self.count[env_idx].item())
        return self.names[env_idx][:count]

    def get_cached_mesh_names(self) -> List[str]:
        """Get the names of all meshes loaded in the Warp cache."""
        return list(self.wp_cache.keys())

    def clear(self, env_idx: Optional[int] = None, clear_warp_cache: bool = False) -> None:
        """Clear meshes from one or all environments.

        Args:
            env_idx: Environment to clear. If None, clears all environments.
            clear_warp_cache: If True, also clears the shared Warp mesh cache.
        """
        if env_idx is not None:
            self.enable[env_idx, :] = 0
            self.count[env_idx] = 0
            self.names[env_idx] = [None] * self.max_n
        else:
            self.enable[:] = 0
            self.count[:] = 0
            self.names = [[None] * self.max_n for _ in range(self.num_envs)]

        if clear_warp_cache:
            self.wp_cache.clear()

    # -------------------------------------------------------------------------
    # Warp Conversion
    # -------------------------------------------------------------------------

    def to_warp(self, max_dist: Optional[float] = None) -> MeshDataWarp:
        """Convert to Warp struct for kernel launches.

        Args:
            max_dist: Override max query distance. If None, uses self.max_dist.

        Returns:
            MeshDataWarp struct for use in Warp kernels.
        """
        if max_dist is None:
            max_dist = self.max_dist

        s = MeshDataWarp()
        s.mesh_ids = wp.from_torch(self.mesh_ids.view(-1), dtype=wp.uint64)
        s.dims = wp.from_torch(self.dims.view(-1, 4), dtype=wp.float32)
        s.inv_pose = wp.from_torch(self.inv_pose.view(-1, 8), dtype=wp.float32)
        s.enable = wp.from_torch(self.enable.view(-1), dtype=wp.uint8)
        s.n_per_env = wp.from_torch(self.count.view(-1), dtype=wp.int32)
        s.max_n = self.max_n
        s.num_envs = self.num_envs
        s.max_dist = max_dist
        return s


# =============================================================================
# Warp Struct
# =============================================================================


@wp.struct
class MeshDataWarp:
    """Warp struct for mesh obstacle data.

    Attributes:
        mesh_ids: Warp mesh handles for BVH queries. Shape: (num_envs * max_n,).
        dims: Bounding box dimensions [x, y, z, pad]. Shape: (num_envs * max_n, 4).
        inv_pose: Inverse pose [x, y, z, qw, qx, qy, qz, pad]. Shape: (num_envs * max_n, 8).
        enable: Enable flag per mesh. 1 = active, 0 = disabled.
        n_per_env: Number of active meshes per environment.
        max_n: Maximum meshes per environment.
        num_envs: Number of environments.
        max_dist: Maximum query distance for mesh SDF.
    """

    mesh_ids: wp.array(dtype=wp.uint64)
    dims: wp.array2d(dtype=wp.float32)
    inv_pose: wp.array2d(dtype=wp.float32)
    enable: wp.array(dtype=wp.uint8)
    n_per_env: wp.array(dtype=wp.int32)
    max_n: wp.int32
    num_envs: wp.int32
    max_dist: wp.float32


# =============================================================================
# SDF Implementation (Generic Kernel Overloads)
# =============================================================================

_SDF_EPS = 1e-6


def is_obs_enabled(
    obs_set: MeshDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
) -> wp.bool:
    """Check if a mesh obstacle is enabled.

    Args:
        obs_set: Warp struct for mesh data.
        env_idx: Environment index.
        local_idx: Local index of the mesh within the environment.

    Returns:
        True if the mesh is enabled, False otherwise.
    """
    if local_idx >= obs_set.n_per_env[env_idx]:
        return False
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return obs_set.enable[flat_idx] == wp.uint8(1)


def load_obstacle_transform(
    obs_set: MeshDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
) -> wp.transform:
    """Load world-to-local transform for a mesh obstacle.

    The returned transform converts world-frame points to local-frame.
    Use wp.transform_point(t, world_pt) to transform points.
    Use wp.transform_vector(wp.transform_inverse(t), grad_local) for gradients.

    Args:
        obs_set: Warp struct for mesh data.
        env_idx: Environment index.
        local_idx: Local index of the mesh within the environment.

    Returns:
        wp.transform for world-to-local transformation.
    """
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return load_transform_from_inv_pose(obs_set.inv_pose, flat_idx)


def compute_local_sdf(
    obs_set: MeshDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.float32:
    """Compute SDF value for a mesh obstacle (no gradient).

    The query point must already be in the obstacle's local frame.
    Use load_obstacle_transform() and wp.transform_point() to transform
    world-frame points before calling this function.

    Args:
        obs_set: Warp struct for mesh data.
        env_idx: Environment index.
        local_idx: Local index of the mesh within the environment.
        local_pt: Query point in obstacle local frame.

    Returns:
        Signed distance: negative inside, positive outside.
    """
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    mesh_id = obs_set.mesh_ids[flat_idx]

    # calculate max_distance based on bounding box of the mesh so that we
    # can make sure even very interior points can be accounted for.
    bounding_box_size = wp.vec3(obs_set.dims[flat_idx,0],
    obs_set.dims[flat_idx,1],
    obs_set.dims[flat_idx,2])
    max_distance = wp.length(bounding_box_size) * 0.5

    # Query mesh for closest point
    result = wp.mesh_query_point(mesh_id, local_pt, max_distance)

    if not result.result:
        return max_distance

    # Get closest point on mesh surface
    cl_pt = wp.mesh_eval_position(mesh_id, result.face, result.u, result.v)
    dis_length = wp.length(cl_pt - local_pt)

    # Signed distance: negative inside, positive outside
    return dis_length * result.sign


def compute_local_sdf_with_grad(
    obs_set: MeshDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.vec4:
    """Compute SDF and gradient for a mesh obstacle in local frame.

    The query point must already be in the obstacle's local frame.
    The returned gradient is also in the local frame. Use
    wp.transform_vector(wp.transform_inverse(inv_t), grad_local) to
    convert to world frame.

    Args:
        obs_set: Warp struct for mesh data.
        env_idx: Environment index.
        local_idx: Local index of the mesh within the environment.
        local_pt: Query point in obstacle local frame.

    Returns:
        vec4(signed_dist, grad_local_x, grad_local_y, grad_local_z).
    """
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    mesh_id = obs_set.mesh_ids[flat_idx]

    # calculate max_distance based on bounding box of the mesh so that we
    # can make sure even very interior points can be accounted for.
    # obs_set.dims[flat_idx] contains the bounding box dimensions of the mesh[x, y, z].
    bounding_box_size = wp.vec3(obs_set.dims[flat_idx,0],
    obs_set.dims[flat_idx,1],
    obs_set.dims[flat_idx,2])
    max_distance = wp.length(bounding_box_size) * 0.5

    # Query mesh for closest point
    result = wp.mesh_query_point(mesh_id, local_pt, max_distance)

    if not result.result:
        return wp.vec4(max_distance, 0.0, 0.0, 0.0)

    # Get closest point on mesh surface
    cl_pt = wp.mesh_eval_position(mesh_id, result.face, result.u, result.v)
    delta = cl_pt - local_pt
    dis_length = wp.length(delta)

    # Signed distance: negative inside, positive outside
    signed_dist = dis_length * result.sign

    # Gradient in local frame pointing toward obstacle
    grad_local = wp.vec3(0.0, 0.0, 0.0)
    if dis_length > _SDF_EPS:
        grad_local = -delta / dis_length

    return wp.vec4(signed_dist, grad_local[0], grad_local[1], grad_local[2])
