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
"""World representations for computing signed distance are implemented in this module."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

# Third Party
import torch

# CuRobo
from curobo.curobolib.geom import SdfSphereOBB, SdfSweptSphereOBB
from curobo.geom.types import Cuboid, Mesh, Obstacle, VoxelGrid, WorldConfig, batch_tensor_cube
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import log_error, log_info, log_warn


@dataclass
class CollisionBuffer:
    """Helper class stores all buffers required to compute collision cost and gradients."""

    #: Buffer to store signed distance cost value for each query sphere.
    distance_buffer: torch.Tensor

    #: Buffer to store gradient of signed distance cost value for each query sphere.
    grad_distance_buffer: torch.Tensor

    #: Buffer to store sparsity index for each query sphere. If sphere's value is not in collsiion,
    #: sparsity index is set to 0, else 1. Used to prevent rewriting 0 values in distance_buffer
    #: and grad_distance_buffer.
    sparsity_index_buffer: torch.Tensor

    #: Shape of the distance buffer. This is used to check if the buffer needs to be recreated.
    shape: Optional[torch.Size] = None

    def __post_init__(self):
        """Initialize the buffer shape if not provided."""
        self.shape = self.distance_buffer.shape

    @classmethod
    def initialize_from_shape(
        cls, shape: torch.Size, tensor_args: TensorDeviceType
    ) -> CollisionBuffer:
        """Initialize the CollisionBuffer from the given shape of query spheres.

        Args:
            shape: Input shape of the query spheres. The shape is (batch, horizon, n_spheres, 4).
            tensor_args: Device and precision of the tensors.

        Returns:
            CollisionBuffer: Initialized CollisionBuffer object.
        """
        batch, horizon, n_spheres, _ = shape
        distance_buffer = torch.zeros(
            (batch, horizon, n_spheres),
            device=tensor_args.device,
            dtype=tensor_args.collision_distance_dtype,
        )
        grad_distance_buffer = torch.zeros(
            (batch, horizon, n_spheres, 4),
            device=tensor_args.device,
            dtype=tensor_args.collision_gradient_dtype,
        )
        sparsity_idx = torch.zeros(
            (batch, horizon, n_spheres),
            device=tensor_args.device,
            dtype=torch.uint8,
        )
        return CollisionBuffer(distance_buffer, grad_distance_buffer, sparsity_idx)

    def _update_from_shape(self, shape: torch.Size, tensor_args: TensorDeviceType):
        """Update shape of buffers.

        Args:
            shape: New shape of the query spheres.
            tensor_args: device and precision of the tensors.
        """
        batch, horizon, n_spheres, _ = shape
        self.distance_buffer = torch.zeros(
            (batch, horizon, n_spheres),
            device=tensor_args.device,
            dtype=tensor_args.collision_distance_dtype,
        )
        self.grad_distance_buffer = torch.zeros(
            (batch, horizon, n_spheres, 4),
            device=tensor_args.device,
            dtype=tensor_args.collision_gradient_dtype,
        )
        self.sparsity_index_buffer = torch.zeros(
            (batch, horizon, n_spheres),
            device=tensor_args.device,
            dtype=torch.uint8,
        )
        self.shape = shape[:3]

    def update_buffer_shape(self, shape: torch.Size, tensor_args: TensorDeviceType):
        """Update the buffer shape if the shape of the query spheres changes.

        Args:
            shape: New shape of the query spheres.
            tensor_args: device and precision of the tensors.
        """
        if self.shape != shape[:3]:
            self._update_from_shape(shape, tensor_args)

    def clone(self) -> CollisionBuffer:
        """Clone the CollisionBuffer object."""
        dist_buffer = self.distance_buffer.clone()
        grad_buffer = self.grad_distance_buffer.clone()
        sparse_buffer = self.sparsity_index_buffer.clone()
        return CollisionBuffer(dist_buffer, grad_buffer, sparse_buffer)

    def __mul__(self, scalar: float) -> CollisionBuffer:
        """Multiply the CollisionBuffer by a scalar value."""
        self.distance_buffer *= scalar
        self.grad_distance_buffer *= scalar
        self.sparsity_index_buffer *= int(scalar)
        return self


@dataclass
class CollisionQueryBuffer:
    """Class stores all buffers required to query collisions across world representations."""

    #: Buffer to store signed distance cost value for Cuboid world obstacles.
    primitive_collision_buffer: Optional[CollisionBuffer] = None

    #: Buffer to store signed distance cost value for Mesh world obstacles.
    mesh_collision_buffer: Optional[CollisionBuffer] = None

    #: Buffer to store signed distance cost value for Blox world obstacles.
    blox_collision_buffer: Optional[CollisionBuffer] = None

    #: Buffer to store signed distance cost value for Voxel world obstacles.
    voxel_collision_buffer: Optional[CollisionBuffer] = None

    #: Shape of the query spheres. This is used to check if the buffer needs to be recreated.
    shape: Optional[torch.Size] = None

    def __post_init__(self):
        """Initialize the shape of the query spheres if not provided."""
        if self.shape is None:
            if self.primitive_collision_buffer is not None:
                self.shape = self.primitive_collision_buffer.shape
            elif self.mesh_collision_buffer is not None:
                self.shape = self.mesh_collision_buffer.shape
            elif self.blox_collision_buffer is not None:
                self.shape = self.blox_collision_buffer.shape
            elif self.voxel_collision_buffer is not None:
                self.shape = self.voxel_collision_buffer.shape

    def __mul__(self, scalar: float) -> CollisionQueryBuffer:
        """Multiply tensors by a scalar value."""
        if self.primitive_collision_buffer is not None:
            self.primitive_collision_buffer = self.primitive_collision_buffer * scalar
        if self.mesh_collision_buffer is not None:
            self.mesh_collision_buffer = self.mesh_collision_buffer * scalar
        if self.blox_collision_buffer is not None:
            self.blox_collision_buffer = self.blox_collision_buffer * scalar
        if self.voxel_collision_buffer is not None:
            self.voxel_collision_buffer = self.voxel_collision_buffer * scalar
        return self

    def clone(self) -> CollisionQueryBuffer:
        """Clone the CollisionQueryBuffer object."""
        prim_buffer = mesh_buffer = blox_buffer = voxel_buffer = None
        if self.primitive_collision_buffer is not None:
            prim_buffer = self.primitive_collision_buffer.clone()
        if self.mesh_collision_buffer is not None:
            mesh_buffer = self.mesh_collision_buffer.clone()
        if self.blox_collision_buffer is not None:
            blox_buffer = self.blox_collision_buffer.clone()
        if self.voxel_collision_buffer is not None:
            voxel_buffer = self.voxel_collision_buffer.clone()
        return CollisionQueryBuffer(
            prim_buffer,
            mesh_buffer,
            blox_buffer,
            voxel_collision_buffer=voxel_buffer,
            shape=self.shape,
        )

    @classmethod
    def initialize_from_shape(
        cls,
        shape: torch.Size,
        tensor_args: TensorDeviceType,
        collision_types: Dict[str, bool],
    ) -> CollisionQueryBuffer:
        """Initialize the CollisionQueryBuffer from the given shape of query spheres.

        Args:
            shape: Input shape of the query spheres. The shape is (batch, horizon, n_spheres, 4).
            tensor_args: Device and precision of the tensors.
            collision_types: Dictionary of collision types to initialize buffers for.

        Returns:
            CollisionQueryBuffer: Initialized CollisionQueryBuffer object.
        """
        primitive_buffer = mesh_buffer = blox_buffer = voxel_buffer = None
        if "primitive" in collision_types and collision_types["primitive"]:
            primitive_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        if "mesh" in collision_types and collision_types["mesh"]:
            mesh_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        if "blox" in collision_types and collision_types["blox"]:
            blox_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        if "voxel" in collision_types and collision_types["voxel"]:
            voxel_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        return CollisionQueryBuffer(
            primitive_buffer, mesh_buffer, blox_buffer, voxel_collision_buffer=voxel_buffer
        )

    def create_from_shape(
        self,
        shape: torch.Size,
        tensor_args: TensorDeviceType,
        collision_types: Dict[str, bool],
    ) -> CollisionQueryBuffer:
        """Create the CollisionQueryBuffer from the given shape of query spheres.

        Args:
            shape: Input shape of the query spheres. The shape is (batch, horizon, n_spheres, 4).
            tensor_args: Device and precision of the tensors.
            collision_types: Dictionary of collision types to initialize buffers for.

        Returns:
            CollisionQueryBuffer: Initialized CollisionQueryBuffer object.
        """
        if "primitive" in collision_types and collision_types["primitive"]:
            self.primitive_collision_buffer = CollisionBuffer.initialize_from_shape(
                shape, tensor_args
            )
        if "mesh" in collision_types and collision_types["mesh"]:
            self.mesh_collision_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        if "blox" in collision_types and collision_types["blox"]:
            self.blox_collision_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        if "voxel" in collision_types and collision_types["voxel"]:
            self.voxel_collision_buffer = CollisionBuffer.initialize_from_shape(shape, tensor_args)
        self.shape = shape

    def update_buffer_shape(
        self,
        shape: torch.Size,
        tensor_args: TensorDeviceType,
        collision_types: Optional[Dict[str, bool]],
    ):
        """Update buffer shape if it doesn't match existing shape.

        Args:
            shape: New shape of the query spheres.
            tensor_args: Device and precision of the tensors.
            collision_types: Dictionary of collision types to update buffers for.
        """
        # update buffers:
        assert len(shape) == 4  # shape is: batch, horizon, n_spheres, 4
        if self.shape is None:  # buffers not initialized:
            self.create_from_shape(shape, tensor_args, collision_types)
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
            if self.voxel_collision_buffer is not None:
                self.voxel_collision_buffer.update_buffer_shape(shape, tensor_args)
            self.shape = shape

    def get_gradient_buffer(
        self,
    ) -> Optional[torch.Tensor]:
        """Compute the gradient buffer by summing the gradient buffers of all collision types.

        Returns:
            torch.Tensor: Gradient buffer for all collision types
        """
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
        if self.voxel_collision_buffer is not None:
            voxel_buffer = self.voxel_collision_buffer.grad_distance_buffer
            if current_buffer is None:
                current_buffer = voxel_buffer.clone()
            else:
                current_buffer += voxel_buffer

        return current_buffer


class CollisionCheckerType(Enum):
    """Type of collision checker to use."""

    PRIMITIVE = "PRIMITIVE"
    BLOX = "BLOX"
    MESH = "MESH"
    VOXEL = "VOXEL"


@dataclass
class WorldCollisionConfig:
    """Configuration parameters for the WorldCollision object."""

    #: Device and precision of the tensors.
    tensor_args: TensorDeviceType

    #: World obstacles to load for collision checking.
    world_model: Optional[Union[List[WorldConfig], WorldConfig]] = None

    #: Number of obstacles to cache for collision checking across representations.
    #: Use this to create a fixed size buffer for collision checking, e.g, {'obb': 1000} will
    #: create a buffer of 1000 cuboids for each environment.
    cache: Optional[Dict[Obstacle, int]] = None

    #: Number of environments to use for collision checking.
    n_envs: int = 1

    #: Type of collision checker to use.
    checker_type: CollisionCheckerType = CollisionCheckerType.PRIMITIVE

    #: Maximum distance to compute collision checking cost outside the object. This value is
    #: added in addition to a query sphere radius and collision activation distance. A smaller
    #: value will speedup collision checking but can result in slower convergence with swept
    #: sphere collision checking.
    max_distance: Union[torch.Tensor, float] = 0.1

    #: Maximum distance outside an obstacle to use when computing euclidean signed distance field
    #: (ESDF) from different world representations.
    max_esdf_distance: Union[torch.Tensor, float] = 100.0

    def __post_init__(self):
        """Post initialization method to set default values."""
        if self.world_model is not None and isinstance(self.world_model, list):
            self.n_envs = len(self.world_model)
        if isinstance(self.max_distance, float):
            self.max_distance = self.tensor_args.to_device([self.max_distance])
        if isinstance(self.max_esdf_distance, float):
            self.max_esdf_distance = self.tensor_args.to_device([self.max_esdf_distance])

    @staticmethod
    def load_from_dict(
        world_coll_checker_dict: Dict,
        world_model_dict: Union[WorldConfig, Dict, List[WorldConfig]] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> WorldCollisionConfig:
        """Load the WorldCollisionConfig from a dictionary.

        Args:
            world_coll_checker_dict: Dictionary containing the configuration parameters.
            world_model_dict: Dictionary containing obstacles.
            tensor_args: Device and precision of the tensors.

        Returns:
            WorldCollisionConfig: Initialized WorldCollisionConfig object.
        """
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
    """Base class for computing signed distance between query spheres and world obstacles."""

    def __init__(self, config: Optional[WorldCollisionConfig] = None):
        """Initialize the WorldCollision object.

        Args:
            config: Configuration parameters for the WorldCollision object.
        """
        if config is not None:
            WorldCollisionConfig.__init__(self, **vars(config))
        self.collision_types = {}  # Use this dictionary to store collision types
        self._cache_voxelization = None
        self._cache_voxelization_collision_buffer = None

    def load_collision_model(self, world_model: WorldConfig):
        """Load the world obstacles for collision checking."""
        raise NotImplementedError

    def update_obstacle_pose_in_world_model(self, name: str, pose: Pose, env_idx: int = 0):
        """Update the pose of an obstacle in the world model.

        Args:
            name: Name of the obstacle to update.
            pose: Pose to update the obstacle.
            env_idx: Environment index to update the obstacle.
        """
        if self.world_model is None:
            return

        if isinstance(self.world_model, list):
            world = self.world_model[env_idx]
        else:
            world = self.world_model
        obstacle = world.get_obstacle(name)
        if obstacle is not None:
            obstacle.pose = pose.to_list()

    def get_sphere_distance(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        sum_collisions: bool = True,
        compute_esdf: bool = False,
    ):
        """Compute the signed distance between query spheres and world obstacles."""
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
        """Compute binary collision between query spheres and world obstacles."""

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
        sum_collisions: bool = True,
    ):
        """Compute the signed distance between trajectory of spheres and world obstacles."""
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
        """Compute binary collision between trajectory of spheres and world obstacles."""
        raise NotImplementedError

    def get_voxels_in_bounding_box(
        self,
        cuboid: Cuboid = Cuboid(name="test", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1]),
        voxel_size: float = 0.02,
    ) -> Union[List[Cuboid], torch.Tensor]:
        """Get occupied voxels in a grid bounded by the given cuboid.

        Args:
            cuboid: Bounding box to get the occupied voxels.
            voxel_size: Size of the voxel grid.

        Returns:
            Tensor with the occupied voxels in the bounding box.
        """
        new_grid = self.get_occupancy_in_bounding_box(cuboid, voxel_size)
        occupied = new_grid.get_occupied_voxels(0.0)
        return occupied

    def clear_voxelization_cache(self):
        """Clear cache that contains voxelization locations."""
        self._cache_voxelization = None

    def update_cache_voxelization(self, new_grid: VoxelGrid):
        """Update locaiton of voxels based on new grid parameters. Only for debugging.

        Args:
            new_grid: New grid to use for getting voxelized occupancy of current world obstacles.
        """
        if (
            self._cache_voxelization is None
            or self._cache_voxelization.voxel_size != new_grid.voxel_size
            or self._cache_voxelization.dims != new_grid.dims
            or self._cache_voxelization.xyzr_tensor is None
        ):
            self._cache_voxelization = new_grid
            self._cache_voxelization.xyzr_tensor = self._cache_voxelization.create_xyzr_tensor(
                transform_to_origin=True, tensor_args=self.tensor_args
            )
            self._cache_voxelization_collision_buffer = CollisionQueryBuffer()
            xyzr = self._cache_voxelization.xyzr_tensor.view(-1, 1, 1, 4)

            self._cache_voxelization_collision_buffer.update_buffer_shape(
                xyzr.shape,
                self.tensor_args,
                self.collision_types,
            )

    def get_occupancy_in_bounding_box(
        self,
        cuboid: Cuboid = Cuboid(name="test", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1]),
        voxel_size: float = 0.02,
    ) -> VoxelGrid:
        """Get the occupancy of voxels in a grid bounded by the given cuboid.

        Args:
            cuboid: Cuboid to get the occupancy of voxels. Provide pose and dimenstions to
                create occupancy information.
            voxel_size: Size in meters to use as the voxel size.

        Returns:
            Grid with the occupancy of voxels in the bounding box.
        """
        new_grid = VoxelGrid(
            name=cuboid.name, dims=cuboid.dims, pose=cuboid.pose, voxel_size=voxel_size
        )

        self.update_cache_voxelization(new_grid)

        xyzr = self._cache_voxelization.xyzr_tensor

        xyzr = xyzr.view(-1, 1, 1, 4)

        weight = self.tensor_args.to_device([1.0])
        act_distance = self.tensor_args.to_device([0.0])

        d_sph = self.get_sphere_collision(
            xyzr,
            self._cache_voxelization_collision_buffer,
            weight,
            act_distance,
        )
        d_sph = d_sph.reshape(-1)
        new_grid.xyzr_tensor = self._cache_voxelization.xyzr_tensor
        new_grid.feature_tensor = d_sph
        return new_grid

    def get_esdf_in_bounding_box(
        self,
        cuboid: Cuboid = Cuboid(name="test", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1]),
        voxel_size: float = 0.02,
        dtype=torch.float32,
    ) -> VoxelGrid:
        """Get the Euclidean signed distance in a grid bounded by the given cuboid.

        Distance is positive inside obstacles and negative outside obstacles.

        Args:
            cuboid: Bounding cuboid to query signed distance.
            voxel_size: Size of the voxels in meters.
            dtype: Data type of the feature tensor. Use :var:`torch.bfloat16` or
                :var:`torch.float8_e4m3fn` for reduced memory usage.

        Returns:
            Voxels with the Euclidean signed distance in the bounding box.
        """

        new_grid = VoxelGrid(
            name=cuboid.name,
            dims=cuboid.dims,
            pose=cuboid.pose,
            voxel_size=voxel_size,
            feature_dtype=dtype,
        )

        self.update_cache_voxelization(new_grid)

        xyzr = self._cache_voxelization.xyzr_tensor
        xyzr = xyzr.view(-1, 1, 1, 4)

        weight = self.tensor_args.to_device([1.0])

        d_sph = self.get_sphere_distance(
            xyzr,
            self._cache_voxelization_collision_buffer,
            weight,
            self.max_distance,
            sum_collisions=False,
            compute_esdf=True,
        )

        d_sph = d_sph.reshape(-1)
        voxel_grid = self._cache_voxelization
        voxel_grid.feature_tensor = d_sph

        return voxel_grid

    def get_mesh_in_bounding_box(
        self,
        cuboid: Cuboid = Cuboid(name="test", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1]),
        voxel_size: float = 0.02,
        run_marching_cubes: bool = True,
    ) -> Mesh:
        """Get a mesh representation of the world obstacles based on occupancy in a bounding box.

        This uses marching cubes to create a mesh representation of the world obstacles. Use this
        to debug world representations.

        Args:
            cuboid: Bounding box to get the mesh representation.
            voxel_size: Size of the voxels in meters.
            run_marching_cubes: Runs marching cubes over occupied voxels to generate a mesh. If
                set to False, then all occupied voxels are merged into a mesh and returned.

        Returns:
            Mesh representation of the world obstacles in the bounding box.
        """
        voxels = self.get_voxels_in_bounding_box(cuboid, voxel_size)
        voxels = voxels.cpu().numpy()

        if run_marching_cubes:
            mesh = Mesh.from_pointcloud(
                voxels[:, :3].detach().cpu().numpy(),
                pitch=voxel_size * 1.1,
            )
        else:
            cuboids = [
                Cuboid(
                    name="c_" + str(x),
                    pose=[voxels[x, 0], voxels[x, 1], voxels[x, 2], 1, 0, 0, 0],
                    dims=[voxel_size, voxel_size, voxel_size],
                )
                for x in range(voxels.shape[0])
            ]
            mesh = WorldConfig(cuboid=cuboids).get_mesh_world(True).mesh[0]

        return mesh

    def get_obstacle_names(self, env_idx: int = 0) -> List[str]:
        """Get the names of the obstacles in the world.

        Args:
            env_idx: Environment index to get the obstacle names.

        Returns:
            Obstacle names in the world.
        """
        return []

    def check_obstacle_exists(self, name: str, env_idx: int = 0) -> bool:
        """Check if an obstacle exists in the world by name.

        Args:
            name: Name of the obstacle to check.
            env_idx: Environment index to check the obstacle.

        Returns:
            True if the obstacle exists in the world, else False.
        """
        obstacle_names = self.get_obstacle_names(env_idx)

        if name in obstacle_names:
            return True

        return False


class WorldPrimitiveCollision(WorldCollision):
    """World collision checking with oriented bounding boxes (cuboids) for obstacles."""

    def __init__(self, config: WorldCollisionConfig):
        """Initialize the WorldPrimitiveCollision object.

        Args:
            config: Configuration parameters for the WorldPrimitiveCollision object.
        """
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
        """Initialize obstacles cache to allow for dynamic addition of obstacles."""
        if self.cache is not None and "obb" in self.cache and self.cache["obb"] not in [None, 0]:
            self._create_obb_cache(self.cache["obb"])

    def load_collision_model(
        self, world_config: WorldConfig, env_idx=0, fix_cache_reference: bool = False
    ):
        """Load world obstacles into collision checker.

        Args:
            world_config: Obstacles to load into the collision checker.
            env_idx: Environment index to load the obstacles.
            fix_cache_reference: If True, throws error if number of obstacles is greater than
                cache. If False, creates a larger cache. Note that when using collision checker
                inside a recorded cuda graph, recreating the cache will break the graph as the
                reference pointer to the cache will change.
        """
        self._load_collision_model_in_cache(
            world_config, env_idx, fix_cache_reference=fix_cache_reference
        )

    def get_obstacle_names(self, env_idx: int = 0) -> List[str]:
        """Get the names of the obstacles in the world.

        Args:
            env_idx: Environment index to get the obstacle names.

        Returns:
            Obstacle names in the world.
        """
        base_obstacles = super().get_obstacle_names(env_idx)
        return self._env_obbs_names[env_idx] + base_obstacles

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
        """Load world obstacles into collision checker cache. This only loads cuboids.

        Args:
            world_config: World obstacles to load into the collision checker.
            env_idx: Environment index to load the obstacles.
            fix_cache_reference: If True, does not allow to load more obstacles than cache size.
        """
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

    def _create_obb_cache(self, obb_cache: int):
        """Create cache for cuboid (oriented bounding box) obstacles.

        Args:
            obb_cache: Number of cuboids to cache for collision checking.
        """
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
    ) -> int:
        """Add cuboid obstacle to world.

        Args:
            name: Name of the obstacle. Must be unique.
            dims: Dimensions of the cuboid obstacle [length, width, height].
            env_idx: Environment index to add the obstacle to.
            w_obj_pose: Pose of the obstacle in world frame.
            obj_w_pose: Inverse pose of the obstacle in world frame.

        Returns:
            Index of the obstacle in the world.
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
    ) -> int:
        """Add cuboid obstacle to world.

        Args:
            cuboid: Cuboid to add.
            env_idx: Environment index to add the obstacle to.

        Returns:
            Index of the obstacle in the world.
        """
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
        """Update dimensinots of an existing cuboid obstacle.

        Args:
            obj_dims: [dim.x,dim.y, dim.z].
            name: Name of the obstacle to update.
            env_obj_idx: Index of the obstacle to update. Not required if name is provided.
            env_idx: Environment index to update the obstacle.
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
        """Enable/Disable object in collision checking functions.

        Args:
            name: Name of the obstacle to enable.
            enable: True to enable, False to disable.
            env_idx: Index of the environment to enable the obstacle in.
        """
        return self.enable_obb(enable, name, None, env_idx)

    def enable_obb(
        self,
        enable: bool = True,
        name: Optional[str] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Enable/Disable cuboid in collision checking functions.

        Args:
            enable: True to enable, False to disable.
            name: Name of the obstacle to enable.
            env_obj_idx: Index of the obstacle to enable. Not required if name is provided.
            env_idx: Index of the environment to enable the obstacle in.
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
        update_cpu_reference: bool = False,
    ):
        """Update pose of an existing obstacle.

        Args:
            name: Name of obstacle.
            w_obj_pose: Pose of the obstacle in world frame.
            env_idx: Index of the environment to update the obstacle in.
            update_cpu_reference: If True, updates the CPU reference with the new pose. This is
                useful for debugging and visualization. Only supported for env_idx=0.
        """
        if self._env_obbs_names is not None and name in self._env_obbs_names[env_idx]:
            self.update_obb_pose(
                name=name,
                w_obj_pose=w_obj_pose,
                env_idx=env_idx,
            )
        else:
            log_warn("obstacle not found in OBB world model: " + name)

        if update_cpu_reference:
            self.update_obstacle_pose_in_world_model(name, w_obj_pose, env_idx)

    def update_obb_pose(
        self,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        name: Optional[str] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Update pose of an existing cuboid obstacle.

        Args:
            w_obj_pose: Pose of the obstacle in world frame.
            obj_w_pose: Inverse pose of the obstacle in world frame. Not required if w_obj_pose is
                provided.
            name: Name of the obstacle to update.
            env_obj_idx: Index of the obstacle to update. Not required if name is provided.
            env_idx: Index of the environment to update the obstacle.
            update_cpu_reference: If True, updates the CPU reference with the new pose. This is
                useful for debugging and visualization. Only supported for env_idx=0.
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
        """Get pose of world from obstacle frame of reference.

        Args:
            w_obj_pose: Pose of the obstacle in world frame.
            obj_w_pose: Pose of world in obstacle frame of reference.

        Returns:
            Pose of world in obstacle frame of reference.
        """
        if w_obj_pose is not None:
            w_inv_pose = w_obj_pose.inverse()
        elif obj_w_pose is not None:
            w_inv_pose = obj_w_pose
        else:
            log_error("Object pose is not given")
        return w_inv_pose

    def get_obb_idx(
        self,
        name: str,
        env_idx: int = 0,
    ) -> int:
        """Get index of the cuboid obstacle in the world.

        Args:
            name: Name of the obstacle to get the index.
            env_idx: Environment index to get the obstacle index.

        Returns:
            Index of the obstacle in the world.
        """
        if name not in self._env_obbs_names[env_idx]:
            log_error("Obstacle with name: " + name + " not found in current world", exc_info=True)
        return self._env_obbs_names[env_idx].index(name)

    def get_sphere_distance(
        self,
        query_sphere: torch.Tensor,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss=False,
        sum_collisions: bool = True,
        compute_esdf: bool = False,
    ) -> torch.Tensor:
        """Compute the signed distance between query spheres and world obstacles.

        This distance can be used as a collision cost for optimization.

        Args:
            query_sphere: Input tensor with query spheres [batch, horizon, number of spheres, 4].
                With [x, y, z, radius] as the last column for each sphere.
            collision_query_buffer: Buffer to store collision query results.
            weight: Weight of the collision cost.
            activation_distance: Distance outside the object to start computing the cost.
            env_query_idx: Environment index for each batch of query spheres.
            return_loss: If the returned tensor will be scaled or changed before calling backward,
                set this to True. If the returned tensor will be used directly through addition,
                set this to False.
            sum_collisions: Sum the collision cost across all obstacles. This variable is currently
                not passed to the underlying CUDA kernel as setting this to False caused poor
                performance.
            compute_esdf: Compute Euclidean signed distance instead of collision cost. When True,
                the returned tensor will be the signed distance with positive values inside an
                obstacle and negative values outside obstacles.

        Returns:
            Signed distance between query spheres and world obstacles.
        """
        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            log_error("Primitive Collision has no obstacles")

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
            self.max_distance,
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
            sum_collisions,
            compute_esdf,
        )

        return dist

    def get_sphere_collision(
        self,
        query_sphere: torch.Tensor,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss=False,
        **kwargs,
    ) -> torch.Tensor:
        """Compute binary collision between query spheres and world obstacles.

        Args:
            query_sphere: Input tensor with query spheres [batch, horizon, number of spheres, 4].
                With [x, y, z, radius] as the last column for each sphere.
            collision_query_buffer: Collision query buffer to store the results.
            weight: Weight to scale the collision cost.
            activation_distance: Distance outside the object to start computing the cost.
            env_query_idx: Environment index for each batch of query spheres.
            return_loss: True is not supported for binary classification. Set to False.

        Returns:
            Tensor with binary collision results.
        """
        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            log_error("Primitive Collision has no obstacles")
        if return_loss:
            log_error("cannot return loss for classification, use get_sphere_distance")
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
            self.max_distance,
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

    def get_swept_sphere_distance(
        self,
        query_sphere: torch.Tensor,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        sweep_steps: int,
        enable_speed_metric=False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss=False,
        sum_collisions: bool = True,
    ) -> torch.Tensor:
        """Compute the signed distance between trajectory of spheres and world obstacles.

        Args:
            query_sphere: Input tensor with query spheres [batch, horizon, number of spheres, 4].
                With [x, y, z, radius] as the last column for each sphere.
            collision_query_buffer: Collision query buffer to store the results.
            weight: Collision cost weight.
            activation_distance: Distance outside the object to start computing the cost. A smooth
                scaling is applied to the cost starting from this distance. See
                :ref:`research_page` for more details.
            speed_dt: Length of time (seconds) to use when calculating the speed of the sphere
                using finite difference.
            sweep_steps: Number of steps to sweep the sphere along the trajectory. More steps will
                allow for catching small obstacles, taking more time to compute.
            enable_speed_metric: True will scale the collision cost by the speed of the sphere.
                This has the effect of slowing down the robot when near obstacles. This also has
                shown to improve convergence from poor initialization.
            env_query_idx: Environment index for each batch of query spheres.
            return_loss: If the returned tensor will be scaled or changed before calling backward,
                set this to True. If the returned tensor will be used directly through addition,
                set this to False.
            sum_collisions: Sum the collision cost across all obstacles. This variable is currently
                not passed to the underlying CUDA kernel as setting this to False caused poor
                performance.

        Returns:
            Collision cost between trajectory of spheres and world obstacles.
        """

        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            log_error("Primitive Collision has no obstacles")

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
            sum_collisions,
        )

        return dist

    def get_swept_sphere_collision(
        self,
        query_sphere: torch.Tensor,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        sweep_steps: int,
        enable_speed_metric=False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss=False,
    ) -> torch.Tensor:
        """Get binary collision between trajectory of spheres and world obstacles.

        Args:
            query_sphere: Input tensor with query spheres [batch, horizon, number of spheres, 4].
                With [x, y, z, radius] as the last column for each sphere.
            collision_query_buffer: Collision query buffer to store the results.
            weight: Collision cost weight.
            activation_distance: Distance outside the object to start computing the cost. A smooth
                scaling is applied to the cost starting from this distance. See
                :ref:`research_page` for more details.
            speed_dt: Length of time (seconds) to use when calculating the speed of the sphere
                using finite difference. This is not used.
            sweep_steps: Number of steps to sweep the sphere along the trajectory. More steps will
                allow for catching small obstacles, taking more time to compute.
            enable_speed_metric: True will scale the collision cost by the speed of the sphere.
                This has the effect of slowing down the robot when near obstacles. This also has
                shown to improve convergence from poor initialization. This is not used.
            env_query_idx: Environment index for each batch of query spheres.
            return_loss: This is not supported for binary classification. Set to False.

        Returns:
            Collision value between trajectory of spheres and world obstacles.
        """

        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            log_error("Primitive Collision has no obstacles")
        if return_loss:
            log_error("cannot return loss for classify, use get_swept_sphere_distance")
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

    def clear_cache(self):
        """Delete all cuboid obstacles from the world."""
        if self._cube_tensor_list is not None:
            self._cube_tensor_list[2][:] = 0
            self._env_n_obbs[:] = 0
