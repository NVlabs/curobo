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
        self._env_n_voxels = None
        self._voxel_tensor_list = None
        self._env_voxel_names = None

        super().__init__(config)

    def _init_cache(self):
        if (
            self.cache is not None
            and "voxel" in self.cache
            and self.cache["voxel"] not in [None, 0]
        ):
            self._create_voxel_cache(self.cache["voxel"])
        return super()._init_cache()

    def _create_voxel_cache(self, voxel_cache: Dict[str, Any]):
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
        self._load_voxel_collision_model_in_cache(
            world_model, env_idx, fix_cache_reference=fix_cache_reference
        )
        return super().load_collision_model(
            world_model, env_idx=env_idx, fix_cache_reference=fix_cache_reference
        )

    def _load_voxel_collision_model_in_cache(
        self, world_config: WorldConfig, env_idx: int = 0, fix_cache_reference: bool = False
    ):
        """TODO:

        _extended_summary_

        Args:
            world_config: _description_
            env_idx: _description_
            fix_cache_reference: _description_
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
    ):
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

        _extended_summary_

        Args:
            world_config_list: _description_

        Returns:
            _description_
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

        return super().load_batch_collision_model(world_config_list)

    def enable_obstacle(
        self,
        name: str,
        enable: bool = True,
        env_idx: int = 0,
    ):
        if self._env_voxel_names is not None and name in self._env_voxel_names[env_idx]:
            self.enable_voxel(enable, name, None, env_idx)
        else:
            return super().enable_obstacle(name, enable, env_idx)

    def enable_voxel(
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
    ):
        if self._env_voxel_names is not None and name in self._env_voxel_names[env_idx]:
            self.update_voxel_pose(name=name, w_obj_pose=w_obj_pose, env_idx=env_idx)
        else:
            log_error("obstacle not found in OBB world model: " + name)

    def update_voxel_data(self, new_voxel: VoxelGrid, env_idx: int = 0):
        obs_idx = self.get_voxel_idx(new_voxel.name, env_idx)
        self._voxel_tensor_list[3][env_idx, obs_idx, :, :] = new_voxel.feature_tensor.view(
            new_voxel.feature_tensor.shape[0], -1
        ).to(dtype=self._voxel_tensor_list[3].dtype)
        self._voxel_tensor_list[0][env_idx, obs_idx, :3] = self.tensor_args.to_device(
            new_voxel.dims
        )
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
        """Update pose of a specific objects.
        This also updates the signed distance grid to account for the updated object pose.
        Args:
        obj_w_pose: Pose
        obj_idx:
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
        """Update pose of a specific objects.
        This also updates the signed distance grid to account for the updated object pose.
        Args:
        obj_w_pose: Pose
        obj_idx:
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
        if name not in self._env_voxel_names[env_idx]:
            log_error("Obstacle with name: " + name + " not found in current world", exc_info=True)
        return self._env_voxel_names[env_idx].index(name)

    def get_voxel_grid(
        self,
        name: str,
        env_idx: int = 0,
    ):
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
    ):
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
    ):
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
            raise ValueError("cannot return loss for classification, use get_sphere_distance")
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
    ):
        """
        Computes the signed distance via analytic function
        Args:
        tensor_sphere: b, n, 4
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
    ):
        """
        Computes the signed distance via analytic function
        Args:
        tensor_sphere: b, n, 4
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
            raise ValueError("cannot return loss for classify, use get_swept_sphere_distance")
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

    def get_voxel_grid_shape(self, env_idx: int = 0, obs_idx: int = 0):
        return self._voxel_tensor_list[3][env_idx, obs_idx].shape
