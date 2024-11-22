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
"""World represented by euclidean signed distance grids."""

# Standard Library
import math
from typing import Any, Dict, List, Optional

# Third Party
import numpy as np
import torch

# CuRobo
from curobo.curobolib.geom import SdfSphereVoxel, SdfSweptSphereVoxel
from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollisionConfig
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import VoxelGrid, WorldConfig
from curobo.types.math import Pose
from curobo.util.logger import log_error, log_info, log_warn


class WorldVoxelCollision(WorldMeshCollision):
    """Voxel grid representation of World, with each voxel containing Euclidean Signed Distance."""

    def __init__(self, config: WorldCollisionConfig):
        """Initialize with a world collision configuration."""
        self._env_n_voxels = None
        self._voxel_tensor_list = None
        self._env_voxel_names = None

        super().__init__(config)

    def _init_cache(self):
        """Initialize the cache for the world."""
        if (
            self.cache is not None
            and "voxel" in self.cache
            and self.cache["voxel"] not in [None, 0]
        ):
            self._create_voxel_cache(self.cache["voxel"])
        return super()._init_cache()

    def _create_voxel_cache(self, voxel_cache: Dict[str, Any]):
        """Create a cache for voxel grid representation of the world.

        Args:
            voxel_cache: Parameters for the voxel grid representation. The dictionary should
                contain the following keys: layers, dims, voxel_size, feature_dtype. Current
                implementation assumes that all voxel grids have the same number of voxels. Though
                different layers can have different resolutions, this is not yet thoroughly tested.
        """
        n_layers = voxel_cache["layers"]
        dims = voxel_cache["dims"]
        voxel_size = voxel_cache["voxel_size"]
        feature_dtype = voxel_cache["feature_dtype"]
        grid_shape = VoxelGrid(
            "test", pose=[0, 0, 0, 1, 0, 0, 0], dims=dims, voxel_size=voxel_size
        ).get_grid_shape()[0]
        n_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]

        voxel_params = torch.zeros(
            (self.n_envs, n_layers, 4),
            dtype=self.tensor_args.dtype,
            device=self.tensor_args.device,
        )
        voxel_pose = torch.zeros(
            (self.n_envs, n_layers, 8),
            dtype=self.tensor_args.dtype,
            device=self.tensor_args.device,
        )
        voxel_pose[..., 3] = 1.0
        voxel_enable = torch.zeros(
            (self.n_envs, n_layers), dtype=torch.uint8, device=self.tensor_args.device
        )
        self._env_n_voxels = torch.zeros(
            (self.n_envs), device=self.tensor_args.device, dtype=torch.int32
        )
        voxel_features = torch.zeros(
            (self.n_envs, n_layers, n_voxels, 1),
            device=self.tensor_args.device,
            dtype=feature_dtype,
        )

        if feature_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            voxel_features[:] = -1.0 * self.max_esdf_distance
        else:
            if self.max_esdf_distance > 100.0:
                log_warn("Using fp8 for WorldVoxelCollision will reduce max_esdf_distance to 100")
                self.max_esdf_distance = 100.0
            voxel_features = (voxel_features.to(dtype=torch.float16) - self.max_esdf_distance).to(
                dtype=feature_dtype
            )
        self._voxel_tensor_list = [voxel_params, voxel_pose, voxel_enable, voxel_features]
        self.collision_types["voxel"] = True
        self._env_voxel_names = [[None for _ in range(n_layers)] for _ in range(self.n_envs)]

    def load_collision_model(
        self, world_model: WorldConfig, env_idx=0, fix_cache_reference: bool = False
    ):
        """Load collision representation from world obstacles.

        Args:
            world_model: Obstacles in world to load.
            env_idx: Environment index to load obstacles into.
            fix_cache_reference: If True, throws error if number of obstacles is greater than
                cache. If False, creates a larger cache. Note that when using collision checker
                inside a recorded cuda graph, recreating the cache will break the graph as the
                reference pointer to the cache will change.
        """
        self._load_voxel_collision_model_in_cache(
            world_model, env_idx, fix_cache_reference=fix_cache_reference
        )
        super().load_collision_model(
            world_model, env_idx=env_idx, fix_cache_reference=fix_cache_reference
        )

    def _load_voxel_collision_model_in_cache(
        self, world_config: WorldConfig, env_idx: int = 0, fix_cache_reference: bool = False
    ):
        """Load voxel grid representation of the world into the cache.

        Args:
            world_config: Obstacles in world to load.
            env_idx: Environment index to load obstacles into.
            fix_cache_reference: If True, throws error if number of obstacles is greater than
                cache. If False, creates a larger cache. Note that when using collision checker
                inside a recorded cuda graph, recreating the cache will break the graph as the
                reference pointer to the cache will change.
        """
        voxel_objs = world_config.voxel
        max_obs = len(voxel_objs)
        self.world_model = world_config
        if max_obs < 1:
            log_info("No Voxel objs")
            return
        if self._voxel_tensor_list is None or self._voxel_tensor_list[0].shape[1] < max_obs:
            if not fix_cache_reference:
                log_info("Creating Voxel cache" + str(max_obs))
                self._create_voxel_cache(
                    {
                        "layers": max_obs,
                        "dims": voxel_objs[0].dims,
                        "voxel_size": voxel_objs[0].voxel_size,
                        "feature_dtype": voxel_objs[0].feature_dtype,
                    }
                )
            else:
                log_error("number of OBB is larger than collision cache, create larger cache.")

        # load as a batch:
        pose_batch = [c.pose for c in voxel_objs]
        dims_batch = [c.dims for c in voxel_objs]
        names_batch = [c.name for c in voxel_objs]
        size_batch = [c.voxel_size for c in voxel_objs]
        voxel_batch = self._batch_tensor_voxel(pose_batch, dims_batch, size_batch)
        self._voxel_tensor_list[0][env_idx, :max_obs, :] = voxel_batch[0]
        self._voxel_tensor_list[1][env_idx, :max_obs, :7] = voxel_batch[1]

        self._voxel_tensor_list[2][env_idx, :max_obs] = 1  # enabling obstacle

        self._voxel_tensor_list[2][env_idx, max_obs:] = 0  # disabling obstacle

        # copy voxel grid features:

        self._env_n_voxels[env_idx] = max_obs
        self._env_voxel_names[env_idx][:max_obs] = names_batch
        self.collision_types["voxel"] = True

    def _batch_tensor_voxel(
        self, pose: List[List[float]], dims: List[float], voxel_size: List[float]
    ) -> List[torch.Tensor]:
        """Create a list of tensors that represent the voxel parameters.

        Args:
            pose: Pose of voxel grids.
            dims: Dimensions of voxel grids.
            voxel_size: Resolution of voxel grids.

        Returns:
            List of tensors representing the voxel parameters.
        """
        w_T_b = Pose.from_batch_list(pose, tensor_args=self.tensor_args)
        b_T_w = w_T_b.inverse()
        dims_t = torch.as_tensor(
            np.array(dims), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        size_t = torch.as_tensor(
            np.array(voxel_size), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        ).unsqueeze(-1)
        params_t = torch.cat([dims_t, size_t], dim=-1)

        voxel_list = [params_t, b_T_w.get_pose_vector()]
        return voxel_list

    def load_batch_collision_model(self, world_config_list: List[WorldConfig]):
        """Load voxel grid for batched environments

        Args:
            world_config_list: List of world obstacles for each environment.
        """
        log_error("Not Implemented")
        # First find largest number of cuboid:
        c_len = []
        pose_batch = []
        dims_batch = []
        names_batch = []
        vsize_batch = []
        for i in world_config_list:
            c = i.cuboid
            if c is not None:
                c_len.append(len(c))
                pose_batch.extend([i.pose for i in c])
                dims_batch.extend([i.dims for i in c])
                names_batch.extend([i.name for i in c])
                vsize_batch.extend([i.voxel_size for i in c])
            else:
                c_len.append(0)

        max_obs = max(c_len)
        if max_obs < 1:
            log_warn("No obbs found")
            return
        # check if number of environments is same as config:
        reset_buffers = False
        if self._env_n_voxels is not None and len(world_config_list) != len(self._env_n_voxels):
            log_warn(
                "env_n_voxels is not same as world_config_list, reloading collision buffers (breaks CG)"
            )
            reset_buffers = True
            self.n_envs = len(world_config_list)
            self._env_n_voxels = torch.zeros(
                (self.n_envs), device=self.tensor_args.device, dtype=torch.int32
            )

        if self._voxel_tensor_list is not None and self._voxel_tensor_list[0].shape[1] < max_obs:
            log_warn(
                "number of obbs is greater than buffer, reloading collision buffers (breaks CG)"
            )
            reset_buffers = True
        # create cache if does not exist:
        if self._voxel_tensor_list is None or reset_buffers:
            log_info("Creating Obb cache" + str(max_obs))
            self._create_obb_cache(max_obs)

        # load obstacles:
        ## load data into gpu:
        voxel_batch = self._batch_tensor_voxel(pose_batch, dims_batch, vsize_batch)
        c_start = 0
        for i in range(len(self._env_n_voxels)):
            if c_len[i] > 0:
                # load obb:
                self._voxel_tensor_list[0][i, : c_len[i], :] = voxel_batch[0][
                    c_start : c_start + c_len[i]
                ]
                self._voxel_tensor_list[1][i, : c_len[i], :7] = voxel_batch[1][
                    c_start : c_start + c_len[i]
                ]
                self._voxel_tensor_list[2][i, : c_len[i]] = 1
                self._env_voxel_names[i][: c_len[i]] = names_batch[c_start : c_start + c_len[i]]
                self._voxel_tensor_list[2][i, c_len[i] :] = 0
                c_start += c_len[i]
        self._env_n_voxels[:] = torch.as_tensor(
            c_len, dtype=torch.int32, device=self.tensor_args.device
        )
        self.collision_types["voxel"] = True

        super().load_batch_collision_model(world_config_list)

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
        if self._env_voxel_names is not None and name in self._env_voxel_names[env_idx]:
            self.enable_voxel(enable, name, None, env_idx)
        else:
            return super().enable_obstacle(name, enable, env_idx)

    def get_obstacle_names(self, env_idx: int = 0) -> List[str]:
        """Get names of all obstacles in the environment.

        Args:
            env_idx: Environment index to get obstacles from.

        Returns:
            List of obstacle names.
        """
        base_obstacles = super().get_obstacle_names(env_idx)
        return self._env_voxel_names[env_idx] + base_obstacles

    def enable_voxel(
        self,
        enable: bool = True,
        name: Optional[str] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Enable/Disable voxel grid in collision checking functions.

        Args:
            enable: True to enable, False to disable.
            name: Name of voxel grid to enable.
            env_obj_idx: Index of voxel grid. If name is provided, this is ignored.
            env_idx: Environment index to enable the voxel grid in.
        """
        if env_obj_idx is not None:
            self._voxel_tensor_list[2][env_obj_idx] = int(enable)  # enable == 1
        else:
            # find index of given name:
            obs_idx = self.get_voxel_idx(name, env_idx)

            self._voxel_tensor_list[2][env_idx, obs_idx] = int(enable)

    def update_obstacle_pose(
        self,
        name: str,
        w_obj_pose: Pose,
        env_idx: int = 0,
        update_cpu_reference: bool = False,
    ):
        """Update pose of obstacle.

        Args:
            name: Name of the obstacle.
            w_obj_pose: Pose of obstacle in world frame.
            env_idx: Environment index to update obstacle in.
            update_cpu_reference: If True, updates the CPU reference with the new pose. This is
                useful for debugging and visualization. Only supported for env_idx=0.
        """
        if self._env_voxel_names is not None and name in self._env_voxel_names[env_idx]:
            self.update_voxel_pose(name=name, w_obj_pose=w_obj_pose, env_idx=env_idx)
            if update_cpu_reference:
                self.update_obstacle_pose_in_world_model(name, w_obj_pose, env_idx)
        else:
            super().update_obstacle_pose(name, w_obj_pose, env_idx, update_cpu_reference)

    def update_voxel_data(self, new_voxel: VoxelGrid, env_idx: int = 0):
        """Update parameters of a voxel grid. This can also updates signed distance values.

        Args:
            new_voxel: New parameters.
            env_idx: Environment index to update voxel grid in.
        """
        obs_idx = self.get_voxel_idx(new_voxel.name, env_idx)

        feature_tensor = new_voxel.feature_tensor.view(new_voxel.feature_tensor.shape[0], -1)
        if (
            feature_tensor.shape[0] != self._voxel_tensor_list[3][env_idx, obs_idx, :, :].shape[0]
            or feature_tensor.shape[1]
            != self._voxel_tensor_list[3][env_idx, obs_idx, :, :].shape[1]
        ):
            log_error(
                "Feature tensor shape mismatch, existing shape: "
                + str(self._voxel_tensor_list[3][env_idx, obs_idx, :, :].shape)
                + " New shape: "
                + str(feature_tensor.shape)
            )
        self._voxel_tensor_list[3][env_idx, obs_idx, :, :].copy_(feature_tensor)
        self._voxel_tensor_list[0][env_idx, obs_idx, :3].copy_(torch.as_tensor(new_voxel.dims))
        self._voxel_tensor_list[0][env_idx, obs_idx, 3] = new_voxel.voxel_size
        self._voxel_tensor_list[1][env_idx, obs_idx, :7] = (
            Pose.from_list(new_voxel.pose, self.tensor_args).inverse().get_pose_vector()
        )
        self._voxel_tensor_list[2][env_idx, obs_idx] = int(True)

    def update_voxel_features(
        self,
        features: torch.Tensor,
        name: Optional[str] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Update signed distance values in a voxel grid.

        Args:
            features: New signed distance values.
            name: Name of voxel grid obstacle.
            env_obj_idx: Index of voxel grid. If name is provided, this is ignored.
            env_idx: Environment index to update voxel grid in.
        """

        if env_obj_idx is not None:
            self._voxel_tensor_list[3][env_obj_idx, :] = features.to(
                dtype=self._voxel_tensor_list[3].dtype
            )
        else:
            obs_idx = self.get_voxel_idx(name, env_idx)
            self._voxel_tensor_list[3][env_idx, obs_idx, :] = features.to(
                dtype=self._voxel_tensor_list[3].dtype
            )

    def update_voxel_pose(
        self,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        name: Optional[str] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Update pose of voxel grid.

        Args:
            w_obj_pose: Pose of voxel grid in world frame.
            obj_w_pose: Inverse pose of voxel grid. If provided, w_obj_pose is ignored.
            name: Name of the voxel grid.
            env_obj_idx: Index of voxel grid. If name is provided, this is ignored.
            env_idx: Environment index to update voxel grid in.
        """
        obj_w_pose = self._get_obstacle_poses(w_obj_pose, obj_w_pose)
        if env_obj_idx is not None:
            self._voxel_tensor_list[1][env_obj_idx, :7] = obj_w_pose.get_pose_vector()
        else:
            obs_idx = self.get_voxel_idx(name, env_idx)
            self._voxel_tensor_list[1][env_idx, obs_idx, :7] = obj_w_pose.get_pose_vector()

    def get_voxel_idx(
        self,
        name: str,
        env_idx: int = 0,
    ) -> int:
        """Get index of voxel grid in the environment.

        Args:
            name: Name of the voxel grid.
            env_idx: Environment index to get voxel grid from.

        Returns:
            Index of voxel grid.
        """
        if name not in self._env_voxel_names[env_idx]:
            log_error("Obstacle with name: " + name + " not found in current world", exc_info=True)
        return self._env_voxel_names[env_idx].index(name)

    def get_voxel_grid(
        self,
        name: str,
        env_idx: int = 0,
    ) -> VoxelGrid:
        """Get voxel grid from world obstacles.

        Args:
            name: Name of voxel grid.
            env_idx: Environment index to get voxel grid from.

        Returns:
            Voxel grid object.
        """
        obs_idx = self.get_voxel_idx(name, env_idx)
        voxel_params = np.round(
            self._voxel_tensor_list[0][env_idx, obs_idx, :].cpu().numpy().astype(np.float64), 6
        ).tolist()
        voxel_pose = Pose(
            position=self._voxel_tensor_list[1][env_idx, obs_idx, :3],
            quaternion=self._voxel_tensor_list[1][env_idx, obs_idx, 3:7],
        )
        voxel_features = self._voxel_tensor_list[3][env_idx, obs_idx, :]
        voxel_grid = VoxelGrid(
            name=name,
            dims=voxel_params[:3],
            pose=voxel_pose.to_list(),
            voxel_size=voxel_params[3],
            feature_tensor=voxel_features,
        )
        return voxel_grid

    def get_sphere_distance(
        self,
        query_sphere,
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
        if "voxel" not in self.collision_types or not self.collision_types["voxel"]:
            return super().get_sphere_distance(
                query_sphere,
                collision_query_buffer,
                weight,
                activation_distance,
                env_query_idx=env_query_idx,
                return_loss=return_loss,
                sum_collisions=sum_collisions,
                compute_esdf=compute_esdf,
            )

        b, h, n, _ = query_sphere.shape  # This can be read from collision query buffer
        use_batch_env = True
        env_query_idx_voxel = env_query_idx
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx_voxel = self._env_n_voxels
        dist = SdfSphereVoxel.apply(
            query_sphere,
            collision_query_buffer.voxel_collision_buffer.distance_buffer,
            collision_query_buffer.voxel_collision_buffer.grad_distance_buffer,
            collision_query_buffer.voxel_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            self.max_esdf_distance,
            self._voxel_tensor_list[3],
            self._voxel_tensor_list[0],
            self._voxel_tensor_list[1],
            self._voxel_tensor_list[2],
            self._env_n_voxels,
            env_query_idx_voxel,
            self._voxel_tensor_list[0].shape[1],
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
        if ("primitive" not in self.collision_types or not self.collision_types["primitive"]) and (
            "mesh" not in self.collision_types or not self.collision_types["mesh"]
        ):
            return dist
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

            d_val = torch.maximum(dist.view(d_prim.shape), d_prim)
        else:
            d_val = dist.view(d_prim.shape) + d_prim

        return d_val

    def get_sphere_collision(
        self,
        query_sphere,
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
        if "voxel" not in self.collision_types or not self.collision_types["voxel"]:
            return super().get_sphere_collision(
                query_sphere,
                collision_query_buffer,
                weight,
                activation_distance,
                env_query_idx=env_query_idx,
                return_loss=return_loss,
            )

        if return_loss:
            log_error("cannot return loss for classification, use get_sphere_distance")
        b, h, n, _ = query_sphere.shape
        use_batch_env = True
        env_query_idx_voxel = env_query_idx
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx_voxel = self._env_n_voxels
        dist = SdfSphereVoxel.apply(
            query_sphere,
            collision_query_buffer.voxel_collision_buffer.distance_buffer,
            collision_query_buffer.voxel_collision_buffer.grad_distance_buffer,
            collision_query_buffer.voxel_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            self.max_esdf_distance,
            self._voxel_tensor_list[3],
            self._voxel_tensor_list[0],
            self._voxel_tensor_list[1],
            self._voxel_tensor_list[2],
            self._env_n_voxels,
            env_query_idx_voxel,
            self._voxel_tensor_list[0].shape[1],
            b,
            h,
            n,
            query_sphere.requires_grad,
            False,
            use_batch_env,
            False,
            False,
            False,
        )

        if ("primitive" not in self.collision_types or not self.collision_types["primitive"]) and (
            "mesh" not in self.collision_types or not self.collision_types["mesh"]
        ):
            return dist
        d_prim = super().get_sphere_collision(
            query_sphere,
            collision_query_buffer,
            weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
        )
        d_val = dist.view(d_prim.shape) + d_prim
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
        if "voxel" not in self.collision_types or not self.collision_types["voxel"]:
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
        b, h, n, _ = query_sphere.shape
        use_batch_env = True
        env_query_idx_voxel = env_query_idx
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx_voxel = self._env_n_voxels

        dist = SdfSweptSphereVoxel.apply(
            query_sphere,
            collision_query_buffer.voxel_collision_buffer.distance_buffer,
            collision_query_buffer.voxel_collision_buffer.grad_distance_buffer,
            collision_query_buffer.voxel_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            self.max_esdf_distance,
            speed_dt,
            self._voxel_tensor_list[3],
            self._voxel_tensor_list[0],
            self._voxel_tensor_list[1],
            self._voxel_tensor_list[2],
            self._env_n_voxels,
            env_query_idx_voxel,
            self._voxel_tensor_list[0].shape[1],
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
        if ("primitive" not in self.collision_types or not self.collision_types["primitive"]) and (
            "mesh" not in self.collision_types or not self.collision_types["mesh"]
        ):
            return dist
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

        d_val = dist.view(d_prim.shape) + d_prim
        return d_val

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
        if "voxel" not in self.collision_types or not self.collision_types["voxel"]:
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
        if return_loss:
            log_error("cannot return loss for classify, use get_swept_sphere_distance")
        b, h, n, _ = query_sphere.shape

        use_batch_env = True
        env_query_idx_voxel = env_query_idx
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx_voxel = self._env_n_voxels
        dist = SdfSweptSphereVoxel.apply(
            query_sphere,
            collision_query_buffer.voxel_collision_buffer.distance_buffer,
            collision_query_buffer.voxel_collision_buffer.grad_distance_buffer,
            collision_query_buffer.voxel_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            self.max_esdf_distance,
            speed_dt,
            self._voxel_tensor_list[3],
            self._voxel_tensor_list[0],
            self._voxel_tensor_list[1],
            self._voxel_tensor_list[2],
            self._env_n_voxels,
            env_query_idx_voxel,
            self._voxel_tensor_list[0].shape[1],
            b,
            h,
            n,
            sweep_steps,
            enable_speed_metric,
            query_sphere.requires_grad,
            False,
            use_batch_env,
            return_loss,
            True,
        )
        if ("primitive" not in self.collision_types or not self.collision_types["primitive"]) and (
            "mesh" not in self.collision_types or not self.collision_types["mesh"]
        ):
            return dist
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
        d_val = dist.view(d_prim.shape) + d_prim
        return d_val

    def clear_cache(self):
        """Clear obstacles in world cache."""
        if self._voxel_tensor_list is not None:
            self._voxel_tensor_list[2][:] = 0
            if self._voxel_tensor_list[3].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                self._voxel_tensor_list[3][:] = -1.0 * self.max_esdf_distance
            else:
                self._voxel_tensor_list[3][:] = (
                    self._voxel_tensor_list[3].to(dtype=torch.float16) * 0.0
                    - self.max_esdf_distance
                ).to(dtype=self._voxel_tensor_list[3].dtype)
            self._env_n_voxels[:] = 0
        super().clear_cache()

    def get_voxel_grid_shape(
        self, env_idx: int = 0, obs_idx: int = 0, name: Optional[str] = None
    ) -> torch.Size:
        """Get dimensions of the voxel grid.

        Args:
            env_idx: Environment index.
            obs_idx: Obstacle index.
            name: Name of obstacle. When provided, obs_idx is ignored.

        Returns:
            Shape of the voxel grid.
        """
        if name is not None:
            obs_idx = self.get_voxel_idx(name, env_idx)
        return self._voxel_tensor_list[3][env_idx, obs_idx].shape
