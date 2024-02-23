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
from enum import Enum
from typing import Dict, List, Optional, Union

# Third Party
import torch

# CuRobo
from curobo.curobolib.geom import SdfSphereOBB, SdfSweptSphereOBB
from curobo.geom.types import Cuboid, Mesh, Obstacle, WorldConfig, batch_tensor_cube
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import log_error, log_info, log_warn


@dataclass
class CollisionBuffer:
    distance_buffer: torch.Tensor
    grad_distance_buffer: torch.Tensor
    sparsity_index_buffer: torch.Tensor
    shape: Optional[torch.Size] = None

    def __post_init__(self):
        self.shape = self.distance_buffer.shape

    @classmethod
    def initialize_from_shape(cls, shape: torch.Size, tensor_args: TensorDeviceType):
        batch, horizon, n_spheres, _ = shape
        distance_buffer = torch.zeros(
            (batch, horizon, n_spheres), device=tensor_args.device, dtype=tensor_args.dtype
        )
        grad_distance_buffer = torch.zeros(
            (batch, horizon, n_spheres, 4), device=tensor_args.device, dtype=tensor_args.dtype
        )
        sparsity_idx = torch.zeros(
            (batch, horizon, n_spheres),
            device=tensor_args.device,
            dtype=torch.uint8,
        )
        return CollisionBuffer(distance_buffer, grad_distance_buffer, sparsity_idx)

    def _update_from_shape(self, shape: torch.Size, tensor_args: TensorDeviceType):
        batch, horizon, n_spheres, _ = shape
        self.distance_buffer = torch.zeros(
            (batch, horizon, n_spheres), device=tensor_args.device, dtype=tensor_args.dtype
        )
        self.grad_distance_buffer = torch.zeros(
            (batch, horizon, n_spheres, 4), device=tensor_args.device, dtype=tensor_args.dtype
        )
        self.sparsity_index_buffer = torch.zeros(
            (batch, horizon, n_spheres),
            device=tensor_args.device,
            dtype=torch.uint8,
        )
        self.shape = shape[:3]

    def update_buffer_shape(self, shape: torch.Size, tensor_args: TensorDeviceType):
        if self.shape != shape[:3]:
            # print("Recreating PRIM: ",self.shape, shape)

            # self = CollisionBuffer.initialize_from_shape(
            #    shape,
            #    tensor_args,
            # )
            self._update_from_shape(shape, tensor_args)
            # print("New shape:",self.shape)

    def clone(self):
        dist_buffer = self.distance_buffer.clone()
        grad_buffer = self.grad_distance_buffer.clone()
        sparse_buffer = self.sparsity_index_buffer.clone()
        return CollisionBuffer(dist_buffer, grad_buffer, sparse_buffer)

    def __mul__(self, scalar: float):
        self.distance_buffer *= scalar
        self.grad_distance_buffer *= scalar
        self.sparsity_index_buffer *= int(scalar)
        return self


@dataclass
class CollisionQueryBuffer:
    """Class stores all buffers required to query collisions
    This class currently has three main buffers. We initialize the required
    buffers based on ?
    """

    primitive_collision_buffer: Optional[CollisionBuffer] = None
    mesh_collision_buffer: Optional[CollisionBuffer] = None
    blox_collision_buffer: Optional[CollisionBuffer] = None
    shape: Optional[torch.Size] = None

    def __post_init__(self):
        if self.shape is None:
            if self.primitive_collision_buffer is not None:
                self.shape = self.primitive_collision_buffer.shape
            elif self.mesh_collision_buffer is not None:
                self.shape = self.mesh_collision_buffer.shape
            elif self.blox_collision_buffer is not None:
                self.shape = self.blox_collision_buffer.shape

    def __mul__(self, scalar: float):
        if self.primitive_collision_buffer is not None:
            self.primitive_collision_buffer = self.primitive_collision_buffer * scalar
        if self.mesh_collision_buffer is not None:
            self.mesh_collision_buffer = self.mesh_collision_buffer * scalar
        if self.blox_collision_buffer is not None:
            self.blox_collision_buffer = self.blox_collision_buffer * scalar
        return self

    def clone(self):
        prim_buffer = mesh_buffer = blox_buffer = None
        if self.primitive_collision_buffer is not None:
            prim_buffer = self.primitive_collision_buffer.clone()
        if self.mesh_collision_buffer is not None:
            mesh_buffer = self.mesh_collision_buffer.clone()
        if self.blox_collision_buffer is not None:
            blox_buffer = self.blox_collision_buffer.clone()
        return CollisionQueryBuffer(prim_buffer, mesh_buffer, blox_buffer, self.shape)

    @classmethod
    def initialize_from_shape(
        cls,
        shape: torch.Size,
        tensor_args: TensorDeviceType,
        collision_types: Dict[str, bool],
    ):
        primitive_buffer = mesh_buffer = blox_buffer = None
        if "primitive" in collision_types and collision_types["primitive"]:
            primitive_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        if "mesh" in collision_types and collision_types["mesh"]:
            mesh_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        if "blox" in collision_types and collision_types["blox"]:
            blox_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        return CollisionQueryBuffer(primitive_buffer, mesh_buffer, blox_buffer)

    def create_from_shape(
        self,
        shape: torch.Size,
        tensor_args: TensorDeviceType,
        collision_types: Dict[str, bool],
    ):
        if "primitive" in collision_types and collision_types["primitive"]:
            self.primitive_collision_buffer = CollisionBuffer.initialize_from_shape(
                shape, tensor_args
            )
        if "mesh" in collision_types and collision_types["mesh"]:
            self.mesh_collision_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        if "blox" in collision_types and collision_types["blox"]:
            self.blox_collision_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        self.shape = shape
        # return self

    def update_buffer_shape(
        self,
        shape: torch.Size,
        tensor_args: TensorDeviceType,
        collision_types: Optional[Dict[str, bool]],
    ):
        # print(shape, self.shape)
        # update buffers:
        assert len(shape) == 4  # shape is: batch, horizon, n_spheres, 4
        if self.shape is None:  # buffers not initialized:
            self.create_from_shape(shape, tensor_args, collision_types)
            # print("Creating new memory", self.shape)
        else:
            # update buffers if shape doesn't match:
            # TODO: allow for dynamic change of collision_types
            if self.primitive_collision_buffer is not None:
                # print(self.primitive_collision_buffer.shape, shape)
                self.primitive_collision_buffer.update_buffer_shape(shape, tensor_args)
            if self.mesh_collision_buffer is not None:
                self.mesh_collision_buffer.update_buffer_shape(shape, tensor_args)
            if self.blox_collision_buffer is not None:
                self.blox_collision_buffer.update_buffer_shape(shape, tensor_args)
            self.shape = shape

    def get_gradient_buffer(
        self,
    ):
        prim_buffer = mesh_buffer = blox_buffer = None
        current_buffer = None
        if self.primitive_collision_buffer is not None:
            prim_buffer = self.primitive_collision_buffer.grad_distance_buffer
            current_buffer = prim_buffer.clone()

        if self.mesh_collision_buffer is not None:
            mesh_buffer = self.mesh_collision_buffer.grad_distance_buffer
            if current_buffer is None:
                current_buffer = mesh_buffer.clone()
            else:
                current_buffer += mesh_buffer
        if self.blox_collision_buffer is not None:
            blox_buffer = self.blox_collision_buffer.grad_distance_buffer
            if current_buffer is None:
                current_buffer = blox_buffer.clone()
            else:
                current_buffer += blox_buffer

        return current_buffer


class CollisionCheckerType(Enum):
    """Type of collision checker to use.
    Args:
        Enum (_type_): _description_
    """

    PRIMITIVE = "PRIMITIVE"
    BLOX = "BLOX"
    MESH = "MESH"


@dataclass
class WorldCollisionConfig:
    tensor_args: TensorDeviceType
    world_model: Optional[Union[List[WorldConfig], WorldConfig]] = None
    cache: Optional[Dict[Obstacle, int]] = None
    n_envs: int = 1
    checker_type: CollisionCheckerType = CollisionCheckerType.PRIMITIVE
    max_distance: float = 0.01

    def __post_init__(self):
        if self.world_model is not None and isinstance(self.world_model, list):
            self.n_envs = len(self.world_model)

    @staticmethod
    def load_from_dict(
        world_coll_checker_dict: Dict,
        world_model_dict: Union[WorldConfig, Dict, List[WorldConfig]] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        world_cfg = world_model_dict
        if world_model_dict is not None:
            if isinstance(world_model_dict, list) and isinstance(world_model_dict[0], dict):
                world_cfg = [WorldConfig.from_dict(x) for x in world_model_dict]
            elif isinstance(world_model_dict, dict):
                world_cfg = WorldConfig.from_dict(world_model_dict)
        world_coll_checker_dict["checker_type"] = CollisionCheckerType(
            world_coll_checker_dict["checker_type"]
        )
        return WorldCollisionConfig(
            tensor_args=tensor_args, world_model=world_cfg, **world_coll_checker_dict
        )


class WorldCollision(WorldCollisionConfig):
    def __init__(self, config: Optional[WorldCollisionConfig] = None):
        if config is not None:
            WorldCollisionConfig.__init__(self, **vars(config))
        self.collision_types = {}  # Use this dictionary to store collision types

    def load_collision_model(self, world_model: WorldConfig):
        raise NotImplementedError

    def get_sphere_distance(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ):
        """
        Computes the signed distance via analytic function
        Args:
        tensor_sphere: b, n, 4
        """
        raise NotImplementedError

    def get_sphere_collision(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ):
        """
        Computes the signed distance via analytic function
        Args:
        tensor_sphere: b, n, 4
        we assume we don't need gradient for this function. If you need gradient, use get_sphere_distance
        """

        raise NotImplementedError

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
    ):
        raise NotImplementedError

    def get_swept_sphere_collision(
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
    ):
        raise NotImplementedError

    def get_sphere_trace(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        sweep_steps: int,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ):
        raise NotImplementedError


class WorldPrimitiveCollision(WorldCollision):
    """World Oriented Bounding Box representation object

    We represent the world with oriented bounding boxes. For speed, we assume there is a
    maximum number of obbs that can be instantiated. This number is read from the WorldCollisionConfig.
    If no cache is setup, we use the number from the first call of load_collision_model.
    """

    def __init__(self, config: WorldCollisionConfig):
        super().__init__(config)
        self._world_cubes = None
        self._cube_tensor_list = None
        self._env_n_obbs = None
        self._env_obbs_names = None
        self._init_cache()
        if self.world_model is not None:
            if isinstance(self.world_model, list):
                self.load_batch_collision_model(self.world_model)
            else:
                self.load_collision_model(self.world_model)

    def _init_cache(self):
        if self.cache is not None and "obb" in self.cache and self.cache["obb"] not in [None, 0]:
            self._create_obb_cache(self.cache["obb"])

    def load_collision_model(
        self, world_config: WorldConfig, env_idx=0, fix_cache_reference: bool = False
    ):
        self._load_collision_model_in_cache(
            world_config, env_idx, fix_cache_reference=fix_cache_reference
        )

    def load_batch_collision_model(self, world_config_list: List[WorldConfig]):
        """Load a batch of collision environments from a list of world configs.

        Args:
            world_config_list: list of world configs to load from.
        """
        # First find largest number of cuboid:
        c_len = []
        pose_batch = []
        dims_batch = []
        names_batch = []
        for i in world_config_list:
            c = i.cuboid
            if c is not None:
                c_len.append(len(c))
                pose_batch.extend([i.pose for i in c])
                dims_batch.extend([i.dims for i in c])
                names_batch.extend([i.name for i in c])
            else:
                c_len.append(0)

        max_obb = max(c_len)
        if max_obb < 1:
            log_warn("No obbs found")
            return
        # check if number of environments is same as config:
        reset_buffers = False
        if self._env_n_obbs is not None and len(world_config_list) != len(self._env_n_obbs):
            log_warn(
                "env_n_obbs is not same as world_config_list, reloading collision buffers (breaks CG)"
            )
            reset_buffers = True
            self.n_envs = len(world_config_list)
            self._env_n_obbs = torch.zeros(
                (self.n_envs), device=self.tensor_args.device, dtype=torch.int32
            )

        if self._cube_tensor_list is not None and self._cube_tensor_list[0].shape[1] < max_obb:
            log_warn(
                "number of obbs is greater than buffer, reloading collision buffers (breaks CG)"
            )
            reset_buffers = True
        # create cache if does not exist:
        if self._cube_tensor_list is None or reset_buffers:
            log_info("Creating Obb cache" + str(max_obb))
            self._create_obb_cache(max_obb)

        # load obstacles:
        ## load data into gpu:
        cube_batch = batch_tensor_cube(pose_batch, dims_batch, self.tensor_args)
        c_start = 0
        for i in range(len(self._env_n_obbs)):
            if c_len[i] > 0:
                # load obb:
                self._cube_tensor_list[0][i, : c_len[i], :3] = cube_batch[0][
                    c_start : c_start + c_len[i]
                ]
                self._cube_tensor_list[1][i, : c_len[i], :7] = cube_batch[1][
                    c_start : c_start + c_len[i]
                ]
                self._cube_tensor_list[2][i, : c_len[i]] = 1
                self._env_obbs_names[i][: c_len[i]] = names_batch[c_start : c_start + c_len[i]]
                self._cube_tensor_list[2][i, c_len[i] :] = 0
                c_start += c_len[i]
        self._env_n_obbs[:] = torch.as_tensor(
            c_len, dtype=torch.int32, device=self.tensor_args.device
        )
        self.collision_types["primitive"] = True

    def _load_collision_model_in_cache(
        self, world_config: WorldConfig, env_idx: int = 0, fix_cache_reference: bool = False
    ):
        cube_objs = world_config.cuboid
        max_obb = len(cube_objs)
        self.world_model = world_config
        if max_obb < 1:
            log_info("No OBB objs")
            return
        if self._cube_tensor_list is None or self._cube_tensor_list[0].shape[1] < max_obb:
            if not fix_cache_reference:
                log_info("Creating Obb cache" + str(max_obb))
                self._create_obb_cache(max_obb)
            else:
                log_error("number of OBB is larger than collision cache, create larger cache.")

        # load as a batch:
        pose_batch = [c.pose for c in cube_objs]
        dims_batch = [c.dims for c in cube_objs]
        names_batch = [c.name for c in cube_objs]
        cube_batch = batch_tensor_cube(pose_batch, dims_batch, self.tensor_args)

        self._cube_tensor_list[0][env_idx, :max_obb, :3] = cube_batch[0]
        self._cube_tensor_list[1][env_idx, :max_obb, :7] = cube_batch[1]

        self._cube_tensor_list[2][env_idx, :max_obb] = 1  # enabling obstacle

        self._cube_tensor_list[2][env_idx, max_obb:] = 0  # disabling obstacle
        # self._cube_tensor_list[1][env_idx, max_obb:, 0] = 1000.0  # Not needed. TODO: remove

        self._env_n_obbs[env_idx] = max_obb
        self._env_obbs_names[env_idx][:max_obb] = names_batch
        self.collision_types["primitive"] = True

    def _create_obb_cache(self, obb_cache):
        box_dims = (
            torch.zeros(
                (self.n_envs, obb_cache, 4),
                dtype=self.tensor_args.dtype,
                device=self.tensor_args.device,
            )
            + 0.01
        )
        box_pose = torch.zeros(
            (self.n_envs, obb_cache, 8),
            dtype=self.tensor_args.dtype,
            device=self.tensor_args.device,
        )
        box_pose[..., 3] = 1.0
        obs_enable = torch.zeros(
            (self.n_envs, obb_cache), dtype=torch.uint8, device=self.tensor_args.device
        )
        self._env_n_obbs = torch.zeros(
            (self.n_envs), device=self.tensor_args.device, dtype=torch.int32
        )
        self._cube_tensor_list = [box_dims, box_pose, obs_enable]
        self.collision_types["primitive"] = True
        self._env_obbs_names = [[None for _ in range(obb_cache)] for _ in range(self.n_envs)]

    def add_obb_from_raw(
        self,
        name: str,
        dims: torch.Tensor,
        env_idx: int,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
    ):
        """
        Args:
        dims: lenght, width, height
        position: x,y,z
        rotation: matrix (3x3)
        """
        assert w_obj_pose is not None or obj_w_pose is not None
        if name in self._env_obbs_names[env_idx]:
            log_error("Obstacle already exists with name: " + name, exc_info=True)
        if w_obj_pose is not None:
            obj_w_pose = w_obj_pose.inverse()
        # cube = tensor_cube(w_obj_pose, dims, tensor_args=self.tensor_args)

        self._cube_tensor_list[0][env_idx, self._env_n_obbs[env_idx], :3] = dims
        self._cube_tensor_list[1][
            env_idx, self._env_n_obbs[env_idx], :7
        ] = obj_w_pose.get_pose_vector()
        self._cube_tensor_list[2][env_idx, self._env_n_obbs[env_idx]] = 1
        self._env_obbs_names[env_idx][self._env_n_obbs[env_idx]] = name
        self._env_n_obbs[env_idx] += 1
        return self._env_n_obbs[env_idx] - 1

    def add_obb(
        self,
        cuboid: Cuboid,
        env_idx: int = 0,
    ):
        return self.add_obb_from_raw(
            cuboid.name,
            self.tensor_args.to_device(cuboid.dims),
            env_idx,
            Pose.from_list(cuboid.pose, self.tensor_args),
        )

    def update_obb_dims(
        self,
        obj_dims: torch.Tensor,
        name: Optional[str] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Update obstacle dimensions

        Args:
            obj_dims (torch.Tensor): [dim.x,dim.y, dim.z], give as [b,3]
            obj_idx (torch.Tensor or int):

        """
        if env_obj_idx is not None:
            self._cube_tensor_list[0][env_obj_idx, :3] = obj_dims
        else:
            # find index of given name:
            obs_idx = self.get_obb_idx(name, env_idx)

            self._cube_tensor_list[0][env_idx, obs_idx, :3] = obj_dims

    def enable_obstacle(
        self,
        name: str,
        enable: bool = True,
        env_idx: int = 0,
    ):
        return self.enable_obb(enable, name, None, env_idx)

    def enable_obb(
        self,
        enable: bool = True,
        name: Optional[str] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Update obstacle dimensions

        Args:
            obj_dims (torch.Tensor): [dim.x,dim.y, dim.z], give as [b,3]
            obj_idx (torch.Tensor or int):

        """
        if env_obj_idx is not None:
            self._cube_tensor_list[2][env_obj_idx] = int(enable)  # enable == 1
        else:
            # find index of given name:
            obs_idx = self.get_obb_idx(name, env_idx)

            self._cube_tensor_list[2][env_idx, obs_idx] = int(enable)

    def update_obstacle_pose(
        self,
        name: str,
        w_obj_pose: Pose,
        env_idx: int = 0,
    ):
        if self._env_obbs_names is not None and name in self._env_obbs_names[env_idx]:
            self.update_obb_pose(name=name, w_obj_pose=w_obj_pose, env_idx=env_idx)
        else:
            log_error("obstacle not found in OBB world model: " + name)

    def update_obb_pose(
        self,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        name: Optional[str] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Update pose of a specific objects.
        This also updates the signed distance grid to account for the updated object pose.
        Args:
        obj_w_pose: Pose
        obj_idx:
        """
        obj_w_pose = self._get_obstacle_poses(w_obj_pose, obj_w_pose)
        if env_obj_idx is not None:
            self._cube_tensor_list[1][env_obj_idx, :7] = obj_w_pose.get_pose_vector()
        else:
            obs_idx = self.get_obb_idx(name, env_idx)
            self._cube_tensor_list[1][env_idx, obs_idx, :7] = obj_w_pose.get_pose_vector()

    @classmethod
    def _get_obstacle_poses(
        cls,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
    ):
        if w_obj_pose is not None:
            w_inv_pose = w_obj_pose.inverse()
        elif obj_w_pose is not None:
            w_inv_pose = obj_w_pose
        else:
            raise ValueError("Object pose is not given")
        return w_inv_pose

    def get_obb_idx(
        self,
        name: str,
        env_idx: int = 0,
    ) -> int:
        if name not in self._env_obbs_names[env_idx]:
            log_error("Obstacle with name: " + name + " not found in current world", exc_info=True)
        return self._env_obbs_names[env_idx].index(name)

    def get_sphere_distance(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss=False,
    ):
        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            raise ValueError("Primitive Collision has no obstacles")

        b, h, n, _ = query_sphere.shape  # This can be read from collision query buffer
        use_batch_env = True
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx = self._env_n_obbs

        dist = SdfSphereOBB.apply(
            query_sphere,
            collision_query_buffer.primitive_collision_buffer.distance_buffer,
            collision_query_buffer.primitive_collision_buffer.grad_distance_buffer,
            collision_query_buffer.primitive_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            self._cube_tensor_list[0],
            self._cube_tensor_list[0],
            self._cube_tensor_list[1],
            self._cube_tensor_list[2],
            self._env_n_obbs,
            env_query_idx,
            self._cube_tensor_list[0].shape[1],
            b,
            h,
            n,
            query_sphere.requires_grad,
            True,
            use_batch_env,
            return_loss,
        )

        return dist

    def get_sphere_collision(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss=False,
    ):
        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            raise ValueError("Primitive Collision has no obstacles")
        if return_loss:
            raise ValueError("cannot return loss for classification, use get_sphere_distance")
        b, h, n, _ = query_sphere.shape
        use_batch_env = True
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx = self._env_n_obbs

        dist = SdfSphereOBB.apply(
            query_sphere,
            collision_query_buffer.primitive_collision_buffer.distance_buffer,
            collision_query_buffer.primitive_collision_buffer.grad_distance_buffer,
            collision_query_buffer.primitive_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            self._cube_tensor_list[0],
            self._cube_tensor_list[0],
            self._cube_tensor_list[1],
            self._cube_tensor_list[2],
            self._env_n_obbs,
            env_query_idx,
            self._cube_tensor_list[0].shape[1],
            b,
            h,
            n,
            query_sphere.requires_grad,
            False,
            use_batch_env,
            return_loss,
        )
        return dist

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
        return_loss=False,
    ):
        """
        Computes the signed distance via analytic function
        Args:
        tensor_sphere: b, n, 4
        """
        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            raise ValueError("Primitive Collision has no obstacles")

        b, h, n, _ = query_sphere.shape
        use_batch_env = True
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx = self._env_n_obbs

        dist = SdfSweptSphereOBB.apply(
            query_sphere,
            collision_query_buffer.primitive_collision_buffer.distance_buffer,
            collision_query_buffer.primitive_collision_buffer.grad_distance_buffer,
            collision_query_buffer.primitive_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            speed_dt,
            self._cube_tensor_list[0],
            self._cube_tensor_list[0],
            self._cube_tensor_list[1],
            self._cube_tensor_list[2],
            self._env_n_obbs,
            env_query_idx,
            self._cube_tensor_list[0].shape[1],
            b,
            h,
            n,
            sweep_steps,
            enable_speed_metric,
            query_sphere.requires_grad,
            True,
            use_batch_env,
            return_loss,
        )

        return dist

    def get_swept_sphere_collision(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        sweep_steps: int,
        enable_speed_metric=False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss=False,
    ):
        """
        Computes the signed distance via analytic function
        Args:
        tensor_sphere: b, n, 4
        """
        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            raise ValueError("Primitive Collision has no obstacles")
        if return_loss:
            raise ValueError("cannot return loss for classify, use get_swept_sphere_distance")
        b, h, n, _ = query_sphere.shape

        use_batch_env = True
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx = self._env_n_obbs
        dist = SdfSweptSphereOBB.apply(
            query_sphere,
            collision_query_buffer.primitive_collision_buffer.distance_buffer,
            collision_query_buffer.primitive_collision_buffer.grad_distance_buffer,
            collision_query_buffer.primitive_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            speed_dt,
            self._cube_tensor_list[0],
            self._cube_tensor_list[0],
            self._cube_tensor_list[1],
            self._cube_tensor_list[2],
            self._env_n_obbs,
            env_query_idx,
            self._cube_tensor_list[0].shape[1],
            b,
            h,
            n,
            sweep_steps,
            enable_speed_metric,
            query_sphere.requires_grad,
            False,
            use_batch_env,
            return_loss,
        )

        return dist

    def clear_cache(self):
        if self._cube_tensor_list is not None:
            self._cube_tensor_list[2][:] = 1
            self._env_n_obbs[:] = 0

    def get_voxels_in_bounding_box(
        self,
        cuboid: Cuboid,
        voxel_size: float = 0.02,
    ) -> Union[List[Cuboid], torch.Tensor]:
        bounds = cuboid.dims
        low = [-bounds[0], -bounds[1], -bounds[2]]
        high = [bounds[0], bounds[1], bounds[2]]
        trange = [h - l for l, h in zip(low, high)]

        x = torch.linspace(
            -bounds[0], bounds[0], int(trange[0] // voxel_size) + 1, device=self.tensor_args.device
        )
        y = torch.linspace(
            -bounds[1], bounds[1], int(trange[1] // voxel_size) + 1, device=self.tensor_args.device
        )
        z = torch.linspace(
            -bounds[2], bounds[2], int(trange[2] // voxel_size) + 1, device=self.tensor_args.device
        )
        w, l, h = x.shape[0], y.shape[0], z.shape[0]
        xyz = (
            torch.stack(torch.meshgrid(x, y, z, indexing="ij")).permute((1, 2, 3, 0)).reshape(-1, 3)
        )

        pose = Pose.from_list(cuboid.pose, tensor_args=self.tensor_args)
        xyz = pose.transform_points(xyz.contiguous())
        r = torch.zeros_like(xyz[:, 0:1]) + voxel_size
        xyzr = torch.cat([xyz, r], dim=1)
        xyzr = xyzr.reshape(-1, 1, 1, 4)
        collision_buffer = CollisionQueryBuffer()
        collision_buffer.update_buffer_shape(
            xyzr.shape,
            self.tensor_args,
            self.collision_types,
        )
        weight = self.tensor_args.to_device([1.0])
        act_distance = self.tensor_args.to_device([0.0])

        d_sph = self.get_sphere_collision(
            xyzr,
            collision_buffer,
            weight,
            act_distance,
        )
        d_sph = d_sph.reshape(-1)
        xyzr = xyzr.reshape(-1, 4)
        # get occupied voxels:
        occupied = xyzr[d_sph > 0.0]
        return occupied

    def get_mesh_in_bounding_box(
        self,
        cuboid: Cuboid,
        voxel_size: float = 0.02,
    ) -> Mesh:
        voxels = self.get_voxels_in_bounding_box(cuboid, voxel_size)
        # voxels = voxels.cpu().numpy()
        # cuboids = [Cuboid(name="c_"+str(x), pose=[voxels[x,0],voxels[x,1],voxels[x,2], 1,0,0,0], dims=[voxel_size, voxel_size, voxel_size]) for x in range(voxels.shape[0])]
        # mesh = WorldConfig(cuboid=cuboids).get_mesh_world(True).mesh[0]
        mesh = Mesh.from_pointcloud(
            voxels[:, :3].detach().cpu().numpy(),
            pitch=voxel_size * 1.1,
        )
        return mesh
