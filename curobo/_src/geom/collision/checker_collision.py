# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Collision checker using generic Warp kernels.

This module provides a collision checker that uses the generic 2D parallel
Warp kernels with atomic accumulation. It supports collision queries against
capsules, cuboids, cylinders, meshes, and voxel grids in a scene.

Design features:
- 2D parallelization (num_spheres × n_obstacles threads)
- Atomic accumulation into unified buffer
- Type-generic kernels via Warp function overloading
- Optional speed metric for motion-aware collision
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

# Third Party
import torch

# CuRobo
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.wp_autograd import (
    SphereObstacleCollision,
    SweptSphereObstacleCollision,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.geom.data.data_scene import SceneData


@dataclass
class CollisionChecker:
    """Collision checker using generic 2D parallel Warp kernels.

    This class provides collision checking between query spheres and scene
    obstacles (analytic primitives, meshes, voxels). It uses generic Warp kernels that
    dispatch to the correct SDF implementation via function overloading.

    Key features:
    - Uses 2D parallelization (num_spheres × n_obstacles threads)
    - Accumulates results via atomic operations
    - Pre-zeros output buffers before each query
    - Supports optional speed metric for swept collision

    Attributes:
        device_cfg: Device configuration for tensor operations.
        max_distance: Maximum query distance for mesh SDF.
    """

    #: Device configuration for tensor operations.
    device_cfg: DeviceCfg

    #: Maximum distance for mesh queries.
    max_distance: Union[float, torch.Tensor] = 1.0

    #: Internal tensor for max_distance.
    _max_distance_t: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        """Convert float distances to tensors."""
        if isinstance(self.max_distance, float):
            self._max_distance_t = self.device_cfg.to_device([self.max_distance])
        else:
            self._max_distance_t = self.max_distance

    @staticmethod
    def _query_metadata(
        scene: "SceneData",
        spheres: torch.Tensor,
        buffer: CollisionBuffer,
        mask: Optional[torch.Tensor],
        env_idx: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Validate layouts without reading GPU values or allocating query tensors."""
        if spheres.ndim != 4 or spheres.shape[-1] != 4:
            log_and_raise("query spheres must have shape [batch, horizon, spheres, 4]")
        if buffer.distance.shape != spheres.shape[:-1]:
            log_and_raise("collision buffer must match the query sphere shape")
        if scene.device_cfg.device != spheres.device:
            log_and_raise("scene and query spheres must use the same device")
        if mask is None:
            mask = buffer.default_sphere_collision_mask
        if (
            mask.ndim != 2
            or mask.shape[1] != spheres.shape[2]
            or mask.shape[0] < 1
            or mask.dtype != torch.uint32
            or mask.device != spheres.device
            or not mask.is_contiguous()
        ):
            log_and_raise(
                "sphere_collision_mask must be contiguous uint32 [kinematics num_envs, "
                "num_spheres] on the query device"
            )
        use_multi_env = env_idx is not None
        if env_idx is None:
            env_idx = buffer.default_env_query_idx
        if (
            env_idx.shape != (spheres.shape[0],)
            or env_idx.dtype != torch.int32
            or env_idx.device != spheres.device
            or not env_idx.is_contiguous()
        ):
            log_and_raise("env_query_idx must be contiguous int32 [batch] on the query device")
        return mask, env_idx, use_multi_env

    # -------------------------------------------------------------------------
    # Sphere Distance Methods
    # -------------------------------------------------------------------------

    def get_sphere_distance(
        self,
        scene: "SceneData",
        query_sphere: torch.Tensor,
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        sum_distance: bool = False,
        sphere_collision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute collision distance between query spheres and scene obstacles.

        Uses unified autograd function that queries all obstacle types
        and accumulates into a single buffer.

        Args:
            scene: SceneData containing all obstacles.
            query_sphere: Query spheres [batch, horizon, num_spheres, 4].
            collision_buffer: Pre-allocated buffer for collision queries.
            weight: Collision cost weight.
            activation_distance: Distance outside obstacles to start computing cost.
            env_query_idx: Environment index for each batch. If None, uses single env.
            return_loss: True if result will be scaled before backward pass.
            sum_distance: Sum collision costs across robot spheres.
            sphere_collision_mask: Shared uint32 [kinematics num_envs, num_spheres];
                None uses preallocated common-layer masks for raw queries.

        Returns:
            Collision cost tensor with shape ``[batch, horizon, 1]`` when
            ``sum_distance`` is True, otherwise
            ``[batch, horizon, num_spheres]``.
        """
        sphere_collision_mask, env_query_idx, use_multi_env = self._query_metadata(
            scene, query_sphere, collision_buffer, sphere_collision_mask, env_query_idx
        )

        return SphereObstacleCollision.apply(
            query_sphere,
            collision_buffer,
            scene,
            weight,
            activation_distance,
            self._max_distance_t,
            env_query_idx,
            use_multi_env,
            return_loss,
            sum_distance,
            sphere_collision_mask,
        )

    # -------------------------------------------------------------------------
    # Swept Sphere Methods
    # -------------------------------------------------------------------------

    def get_swept_sphere_distance(
        self,
        scene: "SceneData",
        query_sphere: torch.Tensor,
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        trajectory_dt: torch.Tensor,
        enable_speed_metric: bool = False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        sum_distance: bool = False,
        sphere_collision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute swept collision distance using unified autograd function.

        Uses unified autograd function that queries all obstacle types,
        applies sweep interpolation, and optionally fuses the linear speed
        metric into each obstacle contribution.

        Args:
            scene: SceneData containing all obstacles.
            query_sphere: Query spheres [batch, horizon, num_spheres, 4].
            collision_buffer: Pre-allocated buffer for collision queries.
            weight: Collision cost weight.
            activation_distance: Distance outside obstacles to start computing cost.
            trajectory_dt: Time delta between trajectory steps.
            enable_speed_metric: Scale collision cost by sphere speed.
            env_query_idx: Environment index for each batch.
            return_loss: True if result will be scaled before backward pass.
            sum_distance: Sum collision costs across robot spheres.
            sphere_collision_mask: Shared uint32 [kinematics num_envs, num_spheres];
                None uses preallocated common-layer masks for raw queries.

        Returns:
            Collision cost tensor with shape ``[batch, horizon, 1]`` when
            ``sum_distance`` is True, otherwise
            ``[batch, horizon, num_spheres]``.
        """
        sphere_collision_mask, env_query_idx, use_multi_env = self._query_metadata(
            scene, query_sphere, collision_buffer, sphere_collision_mask, env_query_idx
        )

        return SweptSphereObstacleCollision.apply(
            query_sphere,
            collision_buffer,
            scene,
            weight,
            activation_distance,
            self._max_distance_t,
            trajectory_dt,
            enable_speed_metric,
            env_query_idx,
            use_multi_env,
            return_loss,
            sum_distance,
            sphere_collision_mask,
        )

    # -------------------------------------------------------------------------
    # Collision Methods (aliases for compatibility)
    # -------------------------------------------------------------------------

    def get_sphere_collision(
        self,
        scene: "SceneData",
        query_sphere: torch.Tensor,
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        sum_distance: bool = False,
        sphere_collision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute collision between query spheres and scene obstacles.

        Alias for get_sphere_distance for API compatibility.
        """
        return self.get_sphere_distance(
            scene=scene,
            query_sphere=query_sphere,
            collision_buffer=collision_buffer,
            weight=weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
            sum_distance=sum_distance,
            sphere_collision_mask=sphere_collision_mask,
        )

    def get_swept_sphere_collision(
        self,
        scene: "SceneData",
        query_sphere: torch.Tensor,
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        trajectory_dt: torch.Tensor,
        enable_speed_metric: bool = False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        sum_distance: bool = False,
        sphere_collision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute swept collision between query spheres and scene obstacles.

        Alias for get_swept_sphere_distance for API compatibility.
        """
        return self.get_swept_sphere_distance(
            scene=scene,
            query_sphere=query_sphere,
            collision_buffer=collision_buffer,
            weight=weight,
            activation_distance=activation_distance,
            trajectory_dt=trajectory_dt,
            enable_speed_metric=enable_speed_metric,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
            sum_distance=sum_distance,
            sphere_collision_mask=sphere_collision_mask,
        )
