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
from dataclasses import dataclass
from typing import List, Optional

# Third Party
import numpy as np
import torch
import warp as wp

# CuRobo
from curobo.geom.sdf.warp_primitives import SdfMeshWarpPy, SweptSdfMeshWarpPy
from curobo.geom.sdf.world import (
    CollisionQueryBuffer,
    WorldCollisionConfig,
    WorldPrimitiveCollision,
)
from curobo.geom.types import Mesh, WorldConfig
from curobo.types.math import Pose
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.warp import init_warp


@dataclass(frozen=True)
class WarpMeshData:
    name: str
    m_id: int
    vertices: wp.array
    faces: wp.array
    mesh: wp.Mesh


class WorldMeshCollision(WorldPrimitiveCollision):
    """World Mesh Collision using Nvidia's warp library

    This currently requires passing int64 array from torch to warp which is only
    available when compiled from source.
    """

    def __init__(self, config: WorldCollisionConfig):
        # WorldCollision.(self)
        init_warp()

        self.tensor_args = config.tensor_args

        self._env_n_mesh = None
        self._mesh_tensor_list = None
        self._env_mesh_names = None
        self._wp_device = wp.torch.device_from_torch(self.tensor_args.device)
        self._wp_mesh_cache = {}  # stores warp meshes across environments

        super().__init__(config)

    def _init_cache(self):
        if (
            self.cache is not None
            and "mesh" in self.cache
            and (not self.cache["mesh"] in [None, 0])
        ):
            self._create_mesh_cache(self.cache["mesh"])

        return super()._init_cache()

    def load_collision_model(
        self,
        world_model: WorldConfig,
        env_idx: int = 0,
        load_obb_obs: bool = True,
        fix_cache_reference: bool = False,
    ):
        max_nmesh = len(world_model.mesh)
        if max_nmesh > 0:
            if self._mesh_tensor_list is None or self._mesh_tensor_list[0].shape[1] < max_nmesh:
                log_warn("Creating new Mesh cache: " + str(max_nmesh))
                self._create_mesh_cache(max_nmesh)

            # load all meshes as a batch:
            name_list, w_mid, w_inv_pose = self._load_batch_mesh_to_warp(world_model.mesh)
            self._mesh_tensor_list[0][env_idx, :max_nmesh] = w_mid
            self._mesh_tensor_list[1][env_idx, :max_nmesh, :7] = w_inv_pose
            self._mesh_tensor_list[2][env_idx, :max_nmesh] = 1
            self._mesh_tensor_list[2][env_idx, max_nmesh:] = 0

            self._env_mesh_names[env_idx][:max_nmesh] = name_list
            self._env_n_mesh[env_idx] = max_nmesh

            self.collision_types["mesh"] = True
        if load_obb_obs:
            super().load_collision_model(
                world_model, env_idx, fix_cache_reference=fix_cache_reference
            )
        else:
            self.world_model = world_model

    def load_batch_collision_model(self, world_config_list: List[WorldConfig]):
        max_nmesh = max([len(x.mesh) for x in world_config_list])
        if self._mesh_tensor_list is None or self._mesh_tensor_list[0].shape[1] < max_nmesh:
            log_warn("Creating new Mesh cache: " + str(max_nmesh))
            self._create_mesh_cache(max_nmesh)

        for env_idx, world_model in enumerate(world_config_list):
            self.load_collision_model(world_model, env_idx=env_idx, load_obb_obs=False)
        super().load_batch_collision_model(world_config_list)

    def _load_mesh_to_warp(self, mesh: Mesh):
        verts, faces = mesh.get_mesh_data()
        v = wp.array(verts, dtype=wp.vec3, device=self._wp_device)
        f = wp.array(np.ravel(faces), dtype=int, device=self._wp_device)
        new_mesh = wp.Mesh(points=v, indices=f)
        return WarpMeshData(mesh.name, new_mesh.id, v, f, new_mesh)

    def _load_mesh_into_cache(self, mesh: Mesh) -> WarpMeshData:
        #
        if mesh.name not in self._wp_mesh_cache:
            # load mesh into cache:
            self._wp_mesh_cache[mesh.name] = self._load_mesh_to_warp(mesh)
            # return self._wp_mesh_cache[mesh.name]
        else:
            log_warn("Object already in warp cache, using existing instance for: " + mesh.name)
        return self._wp_mesh_cache[mesh.name]

    def _load_batch_mesh_to_warp(self, mesh_list: List[Mesh]):
        # First load all verts and faces:
        name_list = []
        pose_list = []
        id_list = torch.zeros((len(mesh_list)), device=self.tensor_args.device, dtype=torch.int64)
        for i, m_idx in enumerate(mesh_list):
            m_data = self._load_mesh_into_cache(m_idx)
            pose_list.append(m_idx.pose)

            id_list[i] = m_data.m_id
            name_list.append(m_data.name)
        pose_buffer = Pose.from_batch_list(pose_list, self.tensor_args)
        inv_pose_buffer = pose_buffer.inverse()
        return name_list, id_list, inv_pose_buffer.get_pose_vector()

    def add_mesh(self, new_mesh: Mesh, env_idx: int = 0):
        if self._env_n_mesh[env_idx] >= self._mesh_tensor_list[0].shape[1]:
            log_error(
                "Cannot add new mesh as we are at mesh cache limit, increase cache limit in WorldMeshCollision"
            )
            return

        wp_mesh_data = self._load_mesh_into_cache(new_mesh)

        # get mesh pose:
        w_obj_pose = Pose.from_list(new_mesh.pose, self.tensor_args)
        # add loaded mesh into scene:

        curr_idx = self._env_n_mesh[env_idx]
        self._mesh_tensor_list[0][env_idx, curr_idx] = wp_mesh_data.m_id
        self._mesh_tensor_list[1][env_idx, curr_idx, :7] = w_obj_pose.inverse().get_pose_vector()
        self._mesh_tensor_list[2][env_idx, curr_idx] = 1
        self._env_mesh_names[env_idx][curr_idx] = wp_mesh_data.name
        self._env_n_mesh[env_idx] = curr_idx + 1

    def get_mesh_idx(
        self,
        name: str,
        env_idx: int = 0,
    ) -> int:
        if name not in self._env_mesh_names[env_idx]:
            log_error("Obstacle with name: " + name + " not found in current world", exc_info=True)
        return self._env_mesh_names[env_idx].index(name)

    def create_collision_cache(self, mesh_cache=None, obb_cache=None, n_envs=None):
        if n_envs is not None:
            self.n_envs = n_envs
        if mesh_cache is not None:
            self._create_mesh_cache(mesh_cache)
        if obb_cache is not None:
            self._create_obb_cache(obb_cache)

    def _create_mesh_cache(self, mesh_cache):
        # create cache to store meshes, mesh poses and inverse poses

        self._env_n_mesh = torch.zeros(
            (self.n_envs), device=self.tensor_args.device, dtype=torch.int32
        )

        obs_enable = torch.zeros(
            (self.n_envs, mesh_cache), dtype=torch.uint8, device=self.tensor_args.device
        )
        obs_inverse_pose = torch.zeros(
            (self.n_envs, mesh_cache, 8),
            dtype=self.tensor_args.dtype,
            device=self.tensor_args.device,
        )
        obs_ids = torch.zeros(
            (self.n_envs, mesh_cache), device=self.tensor_args.device, dtype=torch.int64
        )
        # v_empty = [[None for _ in range(mesh_cache)] for _ in range(self.n_envs)]
        # @f_empty = [[None for _ in range(mesh_cache)] for _ in range(self.n_envs)]
        # wp_m_empty = [[None for _ in range(mesh_cache)] for _ in range(self.n_envs)]
        # warp requires uint64 for mesh indices, supports conversion from int64 to uint64
        self._mesh_tensor_list = [
            obs_ids,
            obs_inverse_pose,
            obs_enable,
        ]  # 0=mesh idx, 1=pose, 2=mesh enable
        self.collision_types["mesh"] = True  # TODO: enable this after loading first mesh
        self._env_mesh_names = [[None for _ in range(mesh_cache)] for _ in range(self.n_envs)]

        self._wp_mesh_cache = {}

    def update_mesh_pose(
        self,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        name: Optional[str] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        w_inv_pose = self._get_obstacle_poses(w_obj_pose, obj_w_pose)

        if name is not None:
            obs_idx = self.get_mesh_idx(name, env_idx)
            self._mesh_tensor_list[1][env_idx, obs_idx, :7] = w_inv_pose.get_pose_vector()
        elif env_obj_idx is not None:
            self._mesh_tensor_list[1][env_idx, env_obj_idx, :7] = w_inv_pose.get_pose_vector()
        else:
            raise ValueError("name or env_obj_idx needs to be given to update mesh pose")

    def update_all_mesh_pose(
        self,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        name: Optional[List[str]] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Update poses for a list of meshes in the same environment

        Args:
            w_obj_pose (Optional[Pose], optional): _description_. Defaults to None.
            obj_w_pose (Optional[Pose], optional): _description_. Defaults to None.
            name (Optional[List[str]], optional): _description_. Defaults to None.
            env_obj_idx (Optional[torch.Tensor], optional): _description_. Defaults to None.
            env_idx (int, optional): _description_. Defaults to 0.
        """
        w_inv_pose = self._get_obstacle_poses(w_obj_pose, obj_w_pose)
        raise NotImplementedError

    def update_mesh_pose_env(
        self,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        name: Optional[str] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: List[int] = [0],
    ):
        """Update pose of a single object in a list of environments

        Args:
            w_obj_pose (Optional[Pose], optional): _description_. Defaults to None.
            obj_w_pose (Optional[Pose], optional): _description_. Defaults to None.
            name (Optional[List[str]], optional): _description_. Defaults to None.
            env_obj_idx (Optional[torch.Tensor], optional): _description_. Defaults to None.
            env_idx (List[int], optional): _description_. Defaults to [0].
        """
        w_inv_pose = self._get_obstacle_poses(w_obj_pose, obj_w_pose)
        # collect index of mesh across environments:
        # index_tensor = torch.zeros((1, len(env_idx)), dtype=torch.long, device=self.tensor_args.device)

        # for i, e in enumerate[env_idx]:
        #    index_tensor[0,i] = self.get_mesh_idx(name, e)
        raise NotImplementedError
        # self._mesh_tensor_list[1][env_idx, obj_idx]

    def update_mesh_from_warp(
        self,
        warp_mesh_idx: int,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        obj_idx: int = 0,
        env_idx: int = 0,
        name: Optional[str] = None,
    ):
        if name is not None:
            obj_idx = self.get_mesh_idx(name, env_idx)

        if obj_idx >= self._mesh_tensor_list[0][env_idx].shape[0]:
            log_error("Out of cache memory")
        w_inv_pose = self._get_obstacle_poses(w_obj_pose, obj_w_pose)

        self._mesh_tensor_list[0][env_idx, obj_idx] = warp_mesh_idx
        self._mesh_tensor_list[1][env_idx, obj_idx] = w_inv_pose
        self._mesh_tensor_list[2][env_idx, obj_idx] = 1
        self._env_mesh_names[env_idx][obj_idx] = name
        if self._env_n_mesh[env_idx] <= obj_idx:
            self._env_n_mesh[env_idx] = obj_idx + 1

    def update_obstacle_pose(
        self,
        name: str,
        w_obj_pose: Pose,
        env_idx: int = 0,
    ):
        if self._env_mesh_names is not None and name in self._env_mesh_names[env_idx]:
            self.update_mesh_pose(name=name, w_obj_pose=w_obj_pose, env_idx=env_idx)
        elif self._env_obbs_names is not None and name in self._env_obbs_names[env_idx]:
            self.update_obb_pose(name=name, w_obj_pose=w_obj_pose, env_idx=env_idx)
        else:
            log_error("obstacle not found in OBB world model: " + name)

    def enable_obstacle(
        self,
        name: str,
        enable: bool = True,
        env_idx: int = 0,
    ):
        if self._env_mesh_names is not None and name in self._env_mesh_names[env_idx]:
            self.enable_mesh(enable, name, None, env_idx)
        elif self._env_obbs_names is not None and name in self._env_obbs_names[env_idx]:
            self.enable_obb(enable, name, None, env_idx)
        else:
            log_error("Obstacle not found in world model: " + name)
        self.world_model.objects

    def enable_mesh(
        self,
        enable: bool = True,
        name: Optional[str] = None,
        env_mesh_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Update obstacle dimensions

        Args:
            obj_dims (torch.Tensor): [dim.x,dim.y, dim.z], give as [b,3]
            obj_idx (torch.Tensor or int):

        """
        if env_mesh_idx is not None:
            self._mesh_tensor_list[2][env_mesh_idx] = int(enable)  # enable == 1
        else:
            # find index of given name:
            obs_idx = self.get_mesh_idx(name, env_idx)
            self._mesh_tensor_list[2][env_idx, obs_idx] = int(enable)

    def _get_sdf(
        self,
        query_spheres,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx=None,
        return_loss=False,
        compute_esdf=False,
    ):
        d = SdfMeshWarpPy.apply(
            query_spheres,
            collision_query_buffer.mesh_collision_buffer.distance_buffer,
            collision_query_buffer.mesh_collision_buffer.grad_distance_buffer,
            collision_query_buffer.mesh_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            self._mesh_tensor_list[0],
            self._mesh_tensor_list[1],
            self._mesh_tensor_list[2],
            self._env_n_mesh,
            self.max_distance,
            env_query_idx,
            return_loss,
            compute_esdf,
        )
        return d

    def _get_swept_sdf(
        self,
        query_spheres,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        sweep_steps: int,
        enable_speed_metric=False,
        env_query_idx=None,
        return_loss: bool = False,
    ):
        d = SweptSdfMeshWarpPy.apply(
            query_spheres,
            collision_query_buffer.mesh_collision_buffer.distance_buffer,
            collision_query_buffer.mesh_collision_buffer.grad_distance_buffer,
            collision_query_buffer.mesh_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            speed_dt,
            self._mesh_tensor_list[0],
            self._mesh_tensor_list[1],
            self._mesh_tensor_list[2],
            self._env_n_mesh,
            self.max_distance,
            sweep_steps,
            enable_speed_metric,
            env_query_idx,
            return_loss,
        )
        return d

    def get_sphere_distance(
        self,
        query_sphere: torch.Tensor,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        sum_collisions: bool = True,
        compute_esdf: bool = False,
    ):
        # TODO: if no mesh object exist, call primitive
        if "mesh" not in self.collision_types or not self.collision_types["mesh"]:
            return super().get_sphere_distance(
                query_sphere,
                collision_query_buffer,
                weight=weight,
                activation_distance=activation_distance,
                env_query_idx=env_query_idx,
                return_loss=return_loss,
                sum_collisions=sum_collisions,
                compute_esdf=compute_esdf,
            )

        d = self._get_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
            compute_esdf=compute_esdf,
        )

        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            return d
        d_prim = super().get_sphere_distance(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
            sum_collisions=sum_collisions,
            compute_esdf=compute_esdf,
        )
        if compute_esdf:
            d_val = torch.maximum(d.view(d_prim.shape), d_prim)
        else:
            d_val = d.view(d_prim.shape) + d_prim
        return d_val

    def get_sphere_collision(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx=None,
        return_loss=False,
        **kwargs,
    ):
        if "mesh" not in self.collision_types or not self.collision_types["mesh"]:
            return super().get_sphere_collision(
                query_sphere,
                collision_query_buffer,
                weight=weight,
                activation_distance=activation_distance,
                env_query_idx=env_query_idx,
                return_loss=return_loss,
            )

        d = self._get_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
        )

        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            return d

        d_prim = super().get_sphere_collision(
            query_sphere,
            collision_query_buffer,
            weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
        )
        d_val = d.view(d_prim.shape) + d_prim

        return d_val

    def get_swept_sphere_distance(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        sweep_steps: int,
        enable_speed_metric=False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        sum_collisions: bool = True,
    ):
        # log_warn("Swept: Mesh + Primitive Collision Checking is experimental")
        if "mesh" not in self.collision_types or not self.collision_types["mesh"]:
            return super().get_swept_sphere_distance(
                query_sphere,
                collision_query_buffer,
                weight=weight,
                env_query_idx=env_query_idx,
                sweep_steps=sweep_steps,
                activation_distance=activation_distance,
                speed_dt=speed_dt,
                enable_speed_metric=enable_speed_metric,
                return_loss=return_loss,
                sum_collisions=sum_collisions,
            )

        d = self._get_swept_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            env_query_idx=env_query_idx,
            sweep_steps=sweep_steps,
            activation_distance=activation_distance,
            speed_dt=speed_dt,
            enable_speed_metric=enable_speed_metric,
            return_loss=return_loss,
        )
        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            return d

        d_prim = super().get_swept_sphere_distance(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            env_query_idx=env_query_idx,
            sweep_steps=sweep_steps,
            activation_distance=activation_distance,
            speed_dt=speed_dt,
            enable_speed_metric=enable_speed_metric,
            return_loss=return_loss,
            sum_collisions=sum_collisions,
        )
        d_val = d.view(d_prim.shape) + d_prim

        return d_val

    def get_swept_sphere_collision(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        sweep_steps,
        activation_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        enable_speed_metric=False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ):
        if "mesh" not in self.collision_types or not self.collision_types["mesh"]:
            return super().get_swept_sphere_collision(
                query_sphere,
                collision_query_buffer,
                weight=weight,
                env_query_idx=env_query_idx,
                sweep_steps=sweep_steps,
                activation_distance=activation_distance,
                speed_dt=speed_dt,
                enable_speed_metric=enable_speed_metric,
                return_loss=return_loss,
            )
        d = self._get_swept_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            speed_dt=speed_dt,
            env_query_idx=env_query_idx,
            sweep_steps=sweep_steps,
            enable_speed_metric=enable_speed_metric,
            return_loss=return_loss,
        )
        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            return d

        d_prim = super().get_swept_sphere_collision(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            env_query_idx=env_query_idx,
            sweep_steps=sweep_steps,
            activation_distance=activation_distance,
            speed_dt=speed_dt,
            enable_speed_metric=enable_speed_metric,
            return_loss=return_loss,
        )
        d_val = d.view(d_prim.shape) + d_prim

        return d_val

    def clear_cache(self):
        self._wp_mesh_cache = {}
        if self._mesh_tensor_list is not None:
            self._mesh_tensor_list[2][:] = 0
        if self._env_n_mesh is not None:
            self._env_n_mesh[:] = 0
        if self._env_mesh_names is not None:
            self._env_mesh_names = [
                [None for _ in range(self.cache["mesh"])] for _ in range(self.n_envs)
            ]

        super().clear_cache()
