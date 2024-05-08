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
from typing import List, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollisionConfig
from curobo.geom.sdf.world_voxel import WorldVoxelCollision
from curobo.geom.types import Cuboid, Mesh, Sphere, SphereFitType, WorldConfig
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.util.logger import log_error, log_info, log_warn

try:
    # Third Party
    from nvblox_torch.mapper import Mapper
except ImportError:
    log_warn("nvblox torch wrapper is not installed, loading abstract class")
    # Standard Library
    from abc import ABC as Mapper


class WorldBloxCollision(WorldVoxelCollision):
    """World Collision Representaiton using Nvidia's nvblox library.

    This class depends on pytorch wrapper for nvblox.
    Additionally, this representation does not support batched environments as we only store
    one world via nvblox.

    There are two ways to use nvblox, one is by loading maps from disk and the other is by
    creating maps online. In both these instances, we might load more than one map and need to check
    collisions against all maps.

    To facilitate online map creation and updation, we build apis in this
    class that provide.
    """

    def __init__(self, config: WorldCollisionConfig):
        self._pose_offset = None
        self._blox_mapper = None
        self._blox_tensor_list = None
        self._blox_voxel_sizes = [0.02]
        super().__init__(config)

    def load_collision_model(self, world_model: WorldConfig, fix_cache_reference: bool = False):
        # load nvblox mesh
        if len(world_model.blox) > 0:
            # check if there is a mapper instance:
            for k in world_model.blox:
                if k.mapper_instance is not None:
                    if self._blox_mapper is None:
                        self._blox_mapper = k.mapper_instance
                    else:
                        log_error("Only one blox mapper instance is supported")
            if self._blox_mapper is None:
                voxel_sizes = []
                integrator_types = []
                # get voxel sizes and integration types:
                for k in world_model.blox:
                    voxel_sizes.append(k.voxel_size)
                    integrator_types.append(k.integrator_type)
                # create a mapper instance:
                self._blox_mapper = Mapper(
                    voxel_sizes=voxel_sizes,
                    integrator_types=integrator_types,
                    free_on_destruction=False,
                    cuda_device_id=self.tensor_args.device.index,
                )
                self._blox_voxel_sizes = voxel_sizes
            # load map from file if it exists:

            names = []
            pose = []

            for i, k in enumerate(world_model.blox):
                names.append(k.name)
                pose.append(k.pose)
                if k.map_path is not None:
                    log_info("loading nvblox map for layer: " + str(i) + " from " + k.map_path)
                    self._blox_mapper.load_from_file(k.map_path, i)

            # load buffers:
            pose_tensor = torch.zeros(
                (len(names), 8), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            enable_tensor = torch.ones(
                (len(names)), device=self.tensor_args.device, dtype=torch.uint8
            )

            pose_tensor[:, 3] = 1.0

            pose_tensor[:, :7] = (
                Pose.from_batch_list(pose, tensor_args=self.tensor_args).inverse().get_pose_vector()
            )
            self._blox_tensor_list = [pose_tensor, enable_tensor]
            self._blox_names = names
            self.collision_types["blox"] = True

        return super().load_collision_model(world_model, fix_cache_reference=fix_cache_reference)

    def clear_cache(self):
        self._blox_mapper.clear()
        self._blox_mapper.update_hashmaps()
        super().clear_cache()

    def clear_blox_layer(self, layer_name: str):
        index = self._blox_names.index(layer_name)
        self._blox_mapper.clear(index)
        self._blox_mapper.update_hashmaps()

    def _get_blox_sdf(
        self,
        query_spheres,
        collision_query_buffer: CollisionQueryBuffer,
        weight,
        activation_distance,
        return_loss: bool = False,
        compute_esdf: bool = False,
    ):
        d = self._blox_mapper.query_sphere_sdf_cost(
            query_spheres,
            collision_query_buffer.blox_collision_buffer.distance_buffer,
            collision_query_buffer.blox_collision_buffer.grad_distance_buffer,
            collision_query_buffer.blox_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            self.max_esdf_distance,
            self._blox_tensor_list[0],
            self._blox_tensor_list[1],
            return_loss,
            compute_esdf,
        )
        return d

    def _get_blox_swept_sdf(
        self,
        query_spheres,
        collision_query_buffer,
        weight,
        activation_distance,
        speed_dt,
        sweep_steps,
        enable_speed_metric,
        return_loss=False,
    ):
        d = self._blox_mapper.query_sphere_trajectory_sdf_cost(
            query_spheres,
            collision_query_buffer.blox_collision_buffer.distance_buffer,
            collision_query_buffer.blox_collision_buffer.grad_distance_buffer,
            collision_query_buffer.blox_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            speed_dt,
            self._blox_tensor_list[0],
            self._blox_tensor_list[1],
            sweep_steps,
            enable_speed_metric,
            return_loss,
            use_experimental=False,
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
        if "blox" not in self.collision_types or not self.collision_types["blox"]:
            return super().get_sphere_distance(
                query_sphere,
                collision_query_buffer,
                weight,
                activation_distance,
                env_query_idx,
                return_loss,
                sum_collisions=sum_collisions,
                compute_esdf=compute_esdf,
            )
        d = self._get_blox_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            return_loss=return_loss,
            compute_esdf=compute_esdf,
        )

        if ("primitive" not in self.collision_types or not self.collision_types["primitive"]) and (
            "mesh" not in self.collision_types or not self.collision_types["mesh"]
        ):
            return d
        d_base = super().get_sphere_distance(
            query_sphere,
            collision_query_buffer,
            weight,
            activation_distance,
            env_query_idx,
            return_loss,
            sum_collisions=sum_collisions,
            compute_esdf=compute_esdf,
        )
        if compute_esdf:
            d = torch.maximum(d, d_base)
        else:
            d = d + d_base

        return d

    def get_sphere_collision(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx=None,
        return_loss: bool = False,
        **kwargs,
    ):
        if "blox" not in self.collision_types or not self.collision_types["blox"]:
            return super().get_sphere_collision(
                query_sphere,
                collision_query_buffer,
                weight,
                activation_distance,
                env_query_idx,
                return_loss,
            )

        d = self._get_blox_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            return_loss=return_loss,
        )
        if ("primitive" not in self.collision_types or not self.collision_types["primitive"]) and (
            "mesh" not in self.collision_types or not self.collision_types["mesh"]
        ):
            return d
        d_base = super().get_sphere_collision(
            query_sphere,
            collision_query_buffer,
            weight,
            activation_distance,
            env_query_idx,
            return_loss,
        )
        d = d + d_base

        return d

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
        if "blox" not in self.collision_types or not self.collision_types["blox"]:
            return super().get_swept_sphere_distance(
                query_sphere,
                collision_query_buffer,
                weight,
                activation_distance,
                speed_dt,
                sweep_steps,
                enable_speed_metric,
                env_query_idx,
                return_loss=return_loss,
                sum_collisions=sum_collisions,
            )

        d = self._get_blox_swept_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            speed_dt=speed_dt,
            sweep_steps=sweep_steps,
            enable_speed_metric=enable_speed_metric,
            return_loss=return_loss,
        )

        if ("primitive" not in self.collision_types or not self.collision_types["primitive"]) and (
            "mesh" not in self.collision_types or not self.collision_types["mesh"]
        ):
            return d
        d_base = super().get_swept_sphere_distance(
            query_sphere,
            collision_query_buffer,
            weight,
            activation_distance,
            speed_dt,
            sweep_steps,
            enable_speed_metric,
            env_query_idx,
            return_loss=return_loss,
            sum_collisions=sum_collisions,
        )
        d = d + d_base

        return d

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
        if "blox" not in self.collision_types or not self.collision_types["blox"]:
            return super().get_swept_sphere_collision(
                query_sphere,
                collision_query_buffer,
                weight,
                sweep_steps,
                activation_distance,
                speed_dt,
                enable_speed_metric,
                env_query_idx,
                return_loss=return_loss,
            )
        d = self._get_blox_swept_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            speed_dt=speed_dt,
            sweep_steps=sweep_steps,
            enable_speed_metric=enable_speed_metric,
            return_loss=return_loss,
        )

        if ("primitive" not in self.collision_types or not self.collision_types["primitive"]) and (
            "mesh" not in self.collision_types or not self.collision_types["mesh"]
        ):
            return d
        d_base = super().get_swept_sphere_collision(
            query_sphere,
            collision_query_buffer,
            weight,
            activation_distance,
            speed_dt,
            sweep_steps,
            enable_speed_metric,
            env_query_idx,
            return_loss=return_loss,
        )
        d = d + d_base
        return d

    def enable_obstacle(
        self,
        name: str,
        enable: bool = True,
        env_idx: int = 0,
    ):
        if self._blox_names is not None and name in self._blox_names:
            self.enable_blox(enable, name)
        else:
            super().enable_obstacle(name, enable, env_idx)

    def enable_blox(self, enable: bool = True, name: Optional[str] = None):
        index = self._blox_names.index(name)
        self._blox_tensor_list[1][index] = int(enable)

    def update_blox_pose(
        self,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        name: Optional[str] = None,
    ):
        obj_w_pose = self._get_obstacle_poses(w_obj_pose, obj_w_pose)
        index = self._blox_names.index(name)
        self._blox_tensor_list[0][index][:7] = obj_w_pose.get_pose_vector()

    def clear_bounding_box(
        self,
        cuboid: Cuboid,
        layer_name: Optional[str] = None,
    ):
        log_error("Not implemented")

    def get_bounding_spheres(
        self,
        bounding_box: Cuboid,
        obstacle_name: Optional[str] = None,
        n_spheres: int = 1,
        surface_sphere_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        pre_transform_pose: Optional[Pose] = None,
        clear_region: bool = False,
    ) -> List[Sphere]:
        mesh = self.get_mesh_in_bounding_box(bounding_box, obstacle_name, clear_region=clear_region)
        if clear_region:
            self.clear_bounding_box(bounding_box, obstacle_name)
        spheres = mesh.get_bounding_spheres(
            n_spheres=n_spheres,
            surface_sphere_radius=surface_sphere_radius,
            fit_type=fit_type,
            voxelize_method=voxelize_method,
            pre_transform_pose=pre_transform_pose,
            tensor_args=self.tensor_args,
        )
        return spheres

    @profiler.record_function("world_blox/add_camera_frame")
    def add_camera_frame(self, camera_observation: CameraObservation, layer_name: str):
        index = self._blox_names.index(layer_name)
        pose_mat = camera_observation.pose.get_matrix().view(4, 4)
        if camera_observation.rgb_image is not None:
            if camera_observation.rgb_image.shape[-1] != 4:
                log_error("nvblox color should be of shape HxWx4")
            with profiler.record_function("world_blox/add_color_frame"):
                self._blox_mapper.add_color_frame(
                    camera_observation.rgb_image,
                    pose_mat,
                    camera_observation.intrinsics,
                    mapper_id=index,
                )

        if camera_observation.depth_image is not None:
            with profiler.record_function("world_blox/add_depth_frame"):
                self._blox_mapper.add_depth_frame(
                    camera_observation.depth_image,
                    pose_mat,
                    camera_observation.intrinsics,
                    mapper_id=index,
                )

    def process_camera_frames(self, layer_name: Optional[str] = None, process_aux: bool = False):
        self.update_blox_esdf(layer_name)
        if process_aux:
            self.update_blox_mesh(layer_name)

    @profiler.record_function("world_blox/update_hashes")
    def update_blox_hashes(self):
        self._blox_mapper.update_hashmaps()

    @profiler.record_function("world_blox/update_esdf")
    def update_blox_esdf(self, layer_name: Optional[str] = None):
        index = -1
        if layer_name is not None:
            index = self._blox_names.index(layer_name)
        self._blox_mapper.update_esdf(index)

    @profiler.record_function("world_blox/update_mesh")
    def update_blox_mesh(self, layer_name: Optional[str] = None):
        index = -1
        if layer_name is not None:
            index = self._blox_names.index(layer_name)
        self._blox_mapper.update_mesh(index)

    @profiler.record_function("world_blox/get_mesh")
    def get_mesh_from_blox_layer(self, layer_name: str, mode: str = "nvblox") -> Mesh:
        index = self._blox_names.index(layer_name)
        if mode == "nvblox":
            mesh_data = self._blox_mapper.get_mesh(index)
            mesh = Mesh(
                name=self._blox_names[index],
                pose=self._blox_tensor_list[0][index, :7].squeeze().cpu().tolist(),
                vertices=mesh_data["vertices"].tolist(),
                faces=mesh_data["triangles"].tolist(),
                vertex_colors=mesh_data["colors"],
                vertex_normals=mesh_data["normals"],
            )
        elif mode == "voxel":
            # query points using collision checker and use trimesh to create a mesh:
            mesh = self.get_mesh_in_bounding_box(
                Cuboid("test", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1.5, 1.5, 1]),
                voxel_size=0.03,
            )
        return mesh

    def save_layer(self, layer_name: str, file_name: str) -> bool:
        index = self._blox_names.index(layer_name)
        status = self._blox_mapper.save_map(file_name, index)
        return status

    def decay_layer(self, layer_name: str):
        index = self._blox_names.index(layer_name)
        self._blox_mapper.decay_occupancy(mapper_id=index)
