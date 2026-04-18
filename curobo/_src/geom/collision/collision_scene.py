# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Scene collision combining data and operations.

This module provides SceneCollision, a container that bundles:
- SceneData: GPU tensor storage for obstacles (cuboids, meshes, voxels)
- CollisionChecker: Stateless collision operations using generic Warp kernels

The design separates:
- Data (SceneData): Handles obstacle storage and updates
- Operations (CollisionChecker): Handles collision queries
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Union

# Third Party
import torch

# CuRobo
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.checker_collision import CollisionChecker
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose

if TYPE_CHECKING:
    from curobo._src.robot.kinematics.kinematics_state import KinematicsState


@dataclass
class SceneCollisionCfg:
    """Configuration for SceneCollision.

    Attributes:
        device_cfg: Device and dtype configuration.
        scene_model: Scene configuration or list of scene configurations.
        num_envs: Number of parallel environments.
        max_distance: Maximum distance for mesh collision queries.
        cache: Cache sizes {"cuboid": N, "mesh": M, "voxel": L}.
    """

    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
    scene_model: Optional[Union[SceneCfg, List[SceneCfg]]] = None
    num_envs: int = 1
    max_distance: float = 0.1
    cache: Optional[Dict[str, int]] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.scene_model is not None and isinstance(self.scene_model, list):
            self.num_envs = len(self.scene_model)


@dataclass
class SceneCollision:
    """Container bundling scene collision data and operations.

    This provides a unified interface for collision checking by combining:
    - SceneData: GPU tensor storage for obstacles
    - CollisionChecker: Stateless collision operations

    Example:
        >>> cfg = SceneCollisionCfg(device_cfg=device_cfg, scene_model=scene_cfg)
        >>> scene = SceneCollision.from_config(cfg)
        >>> dist = scene.get_sphere_distance(state, buffer, weight, act_dist)
    """

    #: GPU tensor storage for obstacles.
    data: SceneData

    #: Collision checking operations.
    checker: CollisionChecker

    #: Device configuration.
    device_cfg: DeviceCfg

    #: CPU reference to scene configuration.
    scene_model: Optional[Union[SceneCfg, List[SceneCfg]]] = None

    @classmethod
    def from_config(cls, config: SceneCollisionCfg) -> "SceneCollision":
        """Create SceneCollision from configuration.

        Args:
            config: Configuration for scene collision.

        Returns:
            Configured SceneCollision.
        """
        # Determine cache sizes
        cache = config.cache or {}
        cuboid_cache = cache.get("cuboid", cache.get("primitive", cache.get("obb", None)))
        mesh_cache = cache.get("mesh", None)
        voxel_cache = cache.get("voxel", None)

        # Create scene data
        if config.scene_model is not None:
            if isinstance(config.scene_model, list):
                data = SceneData.from_batch_scene_cfg(
                    config.scene_model,
                    config.device_cfg,
                    cuboid_cache=cuboid_cache,
                    mesh_cache=mesh_cache,
                    voxel_cache=voxel_cache,
                )
            else:
                data = SceneData.from_scene_cfg(
                    config.scene_model,
                    config.device_cfg,
                    num_envs=config.num_envs,
                    cuboid_cache=cuboid_cache,
                    mesh_cache=mesh_cache,
                    voxel_cache=voxel_cache,
                )
        else:
            # Create empty cache - only allocate storage for requested types
            data = SceneData.create_cache(
                num_envs=config.num_envs,
                device_cfg=config.device_cfg,
                cuboid_cache=cuboid_cache,
                mesh_cache=mesh_cache,
                voxel_cache=voxel_cache,
            )

        # Create checker
        checker = CollisionChecker(
            device_cfg=config.device_cfg,
            max_distance=config.max_distance,
        )

        return cls(
            data=data,
            checker=checker,
            device_cfg=config.device_cfg,
            scene_model=config.scene_model,
        )

    @property
    def collision_types(self) -> Dict[str, bool]:
        """Get active collision types."""
        return {
            "cuboid": self.data.has_cuboids(),
            "mesh": self.data.has_meshes(),
            "voxel": self.data.has_voxels(),
        }

    @property
    def num_envs(self) -> int:
        """Get number of environments."""
        return self.data.num_envs

    # -------------------------------------------------------------------------
    # Collision Query Methods
    # -------------------------------------------------------------------------

    def get_sphere_distance(
        self,
        state: "KinematicsState",
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        """Compute signed distance between query spheres and scene obstacles.

        Args:
            state: Kinematics result containing robot_spheres.
            collision_buffer: Pre-allocated buffer for collision queries.
            weight: Collision cost weight.
            activation_distance: Distance outside obstacles to start computing cost.
            env_query_idx: Environment index for each batch.
            return_loss: True if result will be scaled before backward pass.

        Returns:
            Collision distance tensor [batch, horizon, num_spheres].
        """
        return self.checker.get_sphere_distance(
            scene=self.data,
            query_sphere=state.robot_spheres,
            collision_buffer=collision_buffer,
            weight=weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
        )

    def get_sphere_collision(
        self,
        state: "KinematicsState",
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        """Compute binary collision between query spheres and scene obstacles.

        Args:
            state: Kinematics result containing robot_spheres.
            collision_buffer: Pre-allocated buffer for collision queries.
            weight: Collision cost weight.
            activation_distance: Distance outside obstacles to start computing cost.
            env_query_idx: Environment index for each batch.
            return_loss: True if result will be scaled before backward pass.

        Returns:
            Collision cost tensor [batch, horizon, num_spheres].
        """
        return self.checker.get_sphere_collision(
            scene=self.data,
            query_sphere=state.robot_spheres,
            collision_buffer=collision_buffer,
            weight=weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
        )

    def get_swept_sphere_distance(
        self,
        state: "KinematicsState",
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        trajectory_dt: torch.Tensor,
        enable_speed_metric: bool = False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        """Compute signed distance between trajectory of spheres and obstacles.

        Args:
            state: Kinematics result containing robot_spheres.
            collision_buffer: Pre-allocated buffer for collision queries.
            weight: Collision cost weight.
            activation_distance: Distance outside obstacles to start computing cost.
            trajectory_dt: Time delta between trajectory steps.
            enable_speed_metric: Scale collision cost by sphere speed.
            env_query_idx: Environment index for each batch.
            return_loss: True if result will be scaled before backward pass.

        Returns:
            Collision distance tensor [batch, horizon, num_spheres].
        """
        return self.checker.get_swept_sphere_distance(
            scene=self.data,
            query_sphere=state.robot_spheres,
            collision_buffer=collision_buffer,
            weight=weight,
            activation_distance=activation_distance,
            trajectory_dt=trajectory_dt,
            enable_speed_metric=enable_speed_metric,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
        )

    def get_swept_sphere_collision(
        self,
        state: "KinematicsState",
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        trajectory_dt: torch.Tensor,
        enable_speed_metric: bool = False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        """Compute binary collision between trajectory of spheres and obstacles.

        Args:
            state: Kinematics result containing robot_spheres.
            collision_buffer: Pre-allocated buffer for collision queries.
            weight: Collision cost weight.
            activation_distance: Distance outside obstacles to start computing cost.
            trajectory_dt: Time delta between trajectory steps.
            enable_speed_metric: Scale collision cost by sphere speed.
            env_query_idx: Environment index for each batch.
            return_loss: True if result will be scaled before backward pass.

        Returns:
            Collision cost tensor [batch, horizon, num_spheres].
        """
        return self.checker.get_swept_sphere_collision(
            scene=self.data,
            query_sphere=state.robot_spheres,
            collision_buffer=collision_buffer,
            weight=weight,
            activation_distance=activation_distance,
            trajectory_dt=trajectory_dt,
            enable_speed_metric=enable_speed_metric,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
        )

    # -------------------------------------------------------------------------
    # Raw Query Methods (without KinematicsState)
    # -------------------------------------------------------------------------

    def get_sphere_distance_raw(
        self,
        query_spheres: torch.Tensor,
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        """Compute collision distance using raw sphere tensor.

        Args:
            query_spheres: Query spheres [batch, horizon, num_spheres, 4].
            collision_buffer: Pre-allocated buffer for collision queries.
            weight: Collision cost weight.
            activation_distance: Distance outside obstacles to start computing cost.
            env_query_idx: Environment index for each batch.
            return_loss: True if result will be scaled before backward pass.

        Returns:
            Collision distance tensor [batch, horizon, num_spheres].
        """
        return self.checker.get_sphere_distance(
            scene=self.data,
            query_sphere=query_spheres,
            collision_buffer=collision_buffer,
            weight=weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
        )

    def get_swept_sphere_distance_raw(
        self,
        query_spheres: torch.Tensor,
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        trajectory_dt: torch.Tensor,
        enable_speed_metric: bool = False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        """Compute swept collision distance using raw sphere tensor.

        Args:
            query_spheres: Query spheres [batch, horizon, num_spheres, 4].
            collision_buffer: Pre-allocated buffer for collision queries.
            weight: Collision cost weight.
            activation_distance: Distance outside obstacles to start computing cost.
            trajectory_dt: Time delta between trajectory steps.
            enable_speed_metric: Scale collision cost by sphere speed.
            env_query_idx: Environment index for each batch.
            return_loss: True if result will be scaled before backward pass.

        Returns:
            Collision distance tensor [batch, horizon, num_spheres].
        """
        return self.checker.get_swept_sphere_distance(
            scene=self.data,
            query_sphere=query_spheres,
            collision_buffer=collision_buffer,
            weight=weight,
            activation_distance=activation_distance,
            trajectory_dt=trajectory_dt,
            enable_speed_metric=enable_speed_metric,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
        )

    # -------------------------------------------------------------------------
    # Obstacle Management
    # -------------------------------------------------------------------------

    def load_collision_model(self, scene_model: SceneCfg, env_idx: int = 0):
        """Load scene obstacles for collision checking.

        Args:
            scene_model: Scene configuration with obstacles.
            env_idx: Environment index to load into.
        """
        self.data.load_from_scene_cfg(scene_model, env_idx)
        self.scene_model = scene_model

    def update_obstacle_pose(
        self,
        name: str,
        w_obj_pose: Pose,
        env_idx: int = 0,
    ):
        """Update pose of an obstacle.

        Args:
            name: Name of the obstacle.
            w_obj_pose: New pose in world frame.
            env_idx: Environment index.
        """
        self.data.update_obstacle_pose(name, w_obj_pose, env_idx)

    def enable_obstacle(self, name: str, enable: bool = True, env_idx: int = 0):
        """Enable or disable an obstacle.

        Args:
            name: Name of the obstacle.
            enable: Whether to enable or disable.
            env_idx: Environment index.
        """
        self.data.enable_obstacle(name, enable, env_idx)

    def get_obstacle_names(self, env_idx: int = 0) -> List[str]:
        """Get names of all obstacles in an environment.

        Args:
            env_idx: Environment index.

        Returns:
            List of obstacle names.
        """
        return self.data.get_obstacle_names(env_idx)

    def check_obstacle_exists(self, name: str, env_idx: int = 0) -> bool:
        """Check if an obstacle exists.

        Args:
            name: Name of the obstacle.
            env_idx: Environment index.

        Returns:
            True if obstacle exists.
        """
        return self.data.check_obstacle_exists(name, env_idx)

    def clear_cache(self, env_idx: Optional[int] = None):
        """Clear all obstacles.

        Args:
            env_idx: If provided, clear only this environment.
                    If None, clear all environments.
        """
        self.data.clear(env_idx)

    def get_num_scene_collision_checkers(self) -> int:
        """Get the number of active collision types.

        Returns:
            Count of active collision types.
        """
        return sum(1 for v in self.collision_types.values() if v)

    # -------------------------------------------------------------------------
    # Voxel Access
    # -------------------------------------------------------------------------

    def update_voxel_data(
        self,
        voxel_coords: torch.Tensor,
        features: torch.Tensor,
        env_idx: int = 0,
    ):
        """Update voxel grid data.

        Args:
            voxel_coords: Voxel coordinates [N, 3].
            features: Voxel features [N] (float16).
            env_idx: Environment index.
        """
        if self.data.has_voxels():
            self.data.voxels.update_data(voxel_coords, features, env_idx)

    def get_voxel_grid(self, env_idx: int = 0) -> Optional[Cuboid]:
        """Get voxel grid bounds as a cuboid.

        Args:
            env_idx: Environment index.

        Returns:
            Cuboid representing voxel grid bounds, or None if no voxels.
        """
        if not self.data.has_voxels():
            return None
        voxels = self.data.voxels
        # Extract grid bounds from voxel data
        origin = voxels.grid_origin[env_idx].cpu().numpy()
        dims = voxels.grid_dims.cpu().numpy()
        voxel_size = float(voxels.voxel_size[env_idx].item())
        size = dims * voxel_size
        center = origin + size / 2
        return Cuboid(
            name=f"voxel_grid_{env_idx}",
            pose=[float(center[0]), float(center[1]), float(center[2]), 1, 0, 0, 0],
            dims=[float(size[0]), float(size[1]), float(size[2])],
        )


def create_scene_collision(config: SceneCollisionCfg) -> SceneCollision:
    """Factory function to create SceneCollision.

    Args:
        config: Scene collision configuration.

    Returns:
        Configured SceneCollision instance.
    """
    return SceneCollision.from_config(config)
