# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""GPU data storage and SDF functions for cuboid (OBB) obstacles.

This module provides:
- CuboidData: Python dataclass owning GPU tensors
- CuboidDataWarp: Warp struct for kernel access
- is_obs_enabled, load_obstacle_transform, compute_local_sdf,
  compute_local_sdf_with_grad: Generic kernel overloads
"""

#from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.geom.data.helper_pose import (
    get_obs_idx,
    load_transform_from_inv_pose,
)
from curobo._src.geom.types import Cuboid, SceneCfg, batch_tensor_cube
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise

# =============================================================================
# Python Dataclass
# =============================================================================


# =============================================================================
# Warp Struct
# =============================================================================


@wp.struct
class CuboidDataWarp:
    """Warp struct for cuboid obstacle data.

    Attributes:
        dims: Full extents [x_length, y_length, z_length, pad]. Shape: (num_envs * max_n, 4).
            Note: SDF functions divide by 2 at runtime to get half_extents.
        inv_pose: Inverse pose [x, y, z, qw, qx, qy, qz, pad]. Shape: (num_envs * max_n, 8).
        enable: Enable flag per cuboid. 1 = active, 0 = disabled.
        n_per_env: Number of active cuboids per environment.
        max_n: Maximum cuboids per environment.
        num_envs: Number of environments.
    """

    dims: wp.array2d(dtype=wp.float32)
    inv_pose: wp.array2d(dtype=wp.float32)
    enable: wp.array(dtype=wp.uint8)
    n_per_env: wp.array(dtype=wp.int32)
    max_n: wp.int32
    num_envs: wp.int32



@dataclass
class CuboidData:
    """GPU tensor storage for cuboid (oriented bounding box) obstacles.

    This dataclass owns the GPU tensor buffers for cuboid obstacles and provides
    methods for adding, updating, and querying cuboids. It can be shared between
    multiple consumers (collision checker, TSDF integrator, visualization) to
    avoid duplicate GPU allocations and ensure synchronized pose updates.

    Tensor layouts:
        - dims: (num_envs, max_n, 4) - full extents [x, y, z, pad]
        - inv_pose: (num_envs, max_n, 8) - inverse pose [x, y, z, qw, qx, qy, qz, pad]
        - enable: (num_envs, max_n) - uint8 flags
        - count: (num_envs,) - int32 active counts
    """

    #: Full dimensions [x_length, y_length, z_length] with padding.
    dims: torch.Tensor

    #: Inverse pose [x, y, z, qw, qx, qy, qz] with padding.
    inv_pose: torch.Tensor

    #: Enable flag per cuboid. 1 = active, 0 = disabled.
    enable: torch.Tensor

    #: Number of active cuboids per environment.
    count: torch.Tensor

    #: Name lookup per environment. names[env_idx][cuboid_idx] = name.
    names: List[List[Optional[str]]]

    #: Maximum number of cuboids per environment.
    max_n: int

    #: Number of environments.
    num_envs: int

    #: Device configuration.
    device_cfg: DeviceCfg

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def create_cache(
        cls,
        max_n: int,
        num_envs: int,
        device_cfg: DeviceCfg,
    ) -> "CuboidData":
        """Create an empty cache for cuboid obstacles.

        Args:
            max_n: Maximum number of cuboids per environment.
            num_envs: Number of environments.
            device_cfg: Device and dtype configuration.

        Returns:
            Empty CuboidData with allocated GPU buffers.
        """
        dims = (
            torch.zeros(
                (num_envs, max_n, 4),
                dtype=device_cfg.dtype,
                device=device_cfg.device,
            )
            + 0.01  # Small default to avoid zero-size cuboids
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

        return cls(
            dims=dims,
            inv_pose=inv_pose,
            enable=enable,
            count=count,
            names=names,
            max_n=max_n,
            num_envs=num_envs,
            device_cfg=device_cfg,
        )

    @classmethod
    def from_scene_cfg(
        cls,
        scene_cfg: SceneCfg,
        device_cfg: DeviceCfg,
        env_idx: int = 0,
        num_envs: int = 1,
        max_n: Optional[int] = None,
    ) -> "CuboidData":
        """Create CuboidData from a scene configuration.

        Args:
            scene_cfg: Scene configuration containing cuboid obstacles.
            device_cfg: Device and dtype configuration.
            env_idx: Environment index to load the cuboids into.
            num_envs: Number of environments.
            max_n: Maximum cuboids to allocate. If None, uses len(scene_cfg.cuboid).

        Returns:
            CuboidData populated with cuboids from the scene config.
        """
        cuboids = scene_cfg.cuboid
        num_cuboids = len(cuboids) if cuboids else 0

        if max_n is None:
            max_n = max(num_cuboids, 1)

        instance = cls.create_cache(max_n, num_envs, device_cfg)

        if num_cuboids > 0:
            instance.load_batch(cuboids, env_idx)

        return instance

    @classmethod
    def from_batch_scene_cfg(
        cls,
        scene_cfg_list: List[SceneCfg],
        device_cfg: DeviceCfg,
        max_n: Optional[int] = None,
    ) -> "CuboidData":
        """Create CuboidData from a list of scene configurations.

        Each scene config is loaded into a separate environment.

        Args:
            scene_cfg_list: List of scene configurations, one per environment.
            device_cfg: Device and dtype configuration.
            max_n: Maximum cuboids per environment. If None, uses max across scenes.

        Returns:
            CuboidData populated with cuboids from all scene configs.
        """
        num_envs = len(scene_cfg_list)
        cuboid_counts = [len(cfg.cuboid) if cfg.cuboid else 0 for cfg in scene_cfg_list]

        if max_n is None:
            max_n = max(cuboid_counts) if cuboid_counts else 1

        instance = cls.create_cache(max_n, num_envs, device_cfg)

        for env_idx, scene_cfg in enumerate(scene_cfg_list):
            if scene_cfg.cuboid:
                instance.load_batch(scene_cfg.cuboid, env_idx)

        return instance

    # -------------------------------------------------------------------------
    # Load / Add Methods
    # -------------------------------------------------------------------------

    def load_batch(self, cuboids: List[Cuboid], env_idx: int) -> None:
        """Load a batch of cuboids into an environment, replacing existing ones.

        Args:
            cuboids: List of Cuboid objects to load.
            env_idx: Environment index to load into.
        """
        num_cuboids = len(cuboids)
        if num_cuboids > self.max_n:
            log_and_raise(
                f"Cannot load {num_cuboids} cuboids, max cache size is {self.max_n}"
            )

        if num_cuboids == 0:
            self.enable[env_idx, :] = 0
            self.count[env_idx] = 0
            return

        pose_batch = [c.pose for c in cuboids]
        dims_batch = [c.dims for c in cuboids]
        names_batch = [c.name for c in cuboids]

        cube_tensors = batch_tensor_cube(pose_batch, dims_batch, self.device_cfg)

        self.dims[env_idx, :num_cuboids, :3] = cube_tensors[0]
        self.inv_pose[env_idx, :num_cuboids, :7] = cube_tensors[1]
        self.enable[env_idx, :num_cuboids] = 1
        self.enable[env_idx, num_cuboids:] = 0
        self.names[env_idx][:num_cuboids] = names_batch
        self.count[env_idx] = num_cuboids

    def add(self, cuboid: Cuboid, env_idx: int = 0) -> int:
        """Add a single cuboid to an environment.

        Args:
            cuboid: Cuboid object to add.
            env_idx: Environment index to add to.

        Returns:
            Index of the added cuboid in the environment.

        Raises:
            RuntimeError: If cache is full or name already exists.
        """
        return self.add_from_raw(
            name=cuboid.name,
            dims=self.device_cfg.to_device(cuboid.dims),
            w_obj_pose=Pose.from_list(cuboid.pose, self.device_cfg),
            env_idx=env_idx,
        )

    def add_from_raw(
        self,
        name: str,
        dims: torch.Tensor,
        env_idx: int,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
    ) -> int:
        """Add a cuboid using raw tensor inputs.

        Args:
            name: Unique name for the cuboid.
            dims: Dimensions tensor [length, width, height].
            env_idx: Environment index to add to.
            w_obj_pose: Pose of cuboid in world frame.
            obj_w_pose: Inverse pose. Used if w_obj_pose is None.

        Returns:
            Index of the added cuboid.

        Raises:
            RuntimeError: If cache is full or name already exists.
        """
        if w_obj_pose is None and obj_w_pose is None:
            log_and_raise("Either w_obj_pose or obj_w_pose must be provided")

        current_count = int(self.count[env_idx].item())
        if current_count >= self.max_n:
            log_and_raise(f"Cannot add cuboid, cache is full ({self.max_n} cuboids)")

        if name in self.names[env_idx]:
            log_and_raise(f"Cuboid already exists with name: {name}")

        if w_obj_pose is not None:
            obj_w_pose = w_obj_pose.inverse()

        self.dims[env_idx, current_count, :3] = dims
        self.inv_pose[env_idx, current_count, :7] = obj_w_pose.get_pose_vector()
        self.enable[env_idx, current_count] = 1
        self.names[env_idx][current_count] = name
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
        """Update the pose of an existing cuboid.

        Args:
            name: Name of the cuboid to update.
            w_obj_pose: New pose in world frame.
            obj_w_pose: New inverse pose. Used if w_obj_pose is None.
            env_idx: Environment index.

        Raises:
            RuntimeError: If cuboid with given name not found.
        """
        if w_obj_pose is None and obj_w_pose is None:
            log_and_raise("Either w_obj_pose or obj_w_pose must be provided")

        idx = self.get_idx(name, env_idx)

        if w_obj_pose is not None:
            obj_w_pose = w_obj_pose.inverse()

        self.inv_pose[env_idx, idx, :7] = obj_w_pose.get_pose_vector()

    def update_dims(self, name: str, dims: torch.Tensor, env_idx: int = 0) -> None:
        """Update the dimensions of an existing cuboid.

        Args:
            name: Name of the cuboid to update.
            dims: New dimensions [length, width, height].
            env_idx: Environment index.

        Raises:
            RuntimeError: If cuboid with given name not found.
        """
        idx = self.get_idx(name, env_idx)
        self.dims[env_idx, idx, :3] = dims

    def set_enabled(self, name: str, enabled: bool, env_idx: int = 0) -> None:
        """Enable or disable a cuboid for collision checking.

        Args:
            name: Name of the cuboid.
            enabled: True to enable, False to disable.
            env_idx: Environment index.

        Raises:
            RuntimeError: If cuboid with given name not found.
        """
        idx = self.get_idx(name, env_idx)
        self.enable[env_idx, idx] = int(enabled)

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_idx(self, name: str, env_idx: int = 0) -> int:
        """Get the index of a cuboid by name.

        Args:
            name: Name of the cuboid.
            env_idx: Environment index.

        Returns:
            Index of the cuboid in the environment.

        Raises:
            ValueError: If cuboid with given name not found.
        """
        return self.names[env_idx].index(name)

    def get_active_count(self, env_idx: int = 0) -> int:
        """Get the number of active cuboids in an environment."""
        return int(self.count[env_idx].item())

    def get_names(self, env_idx: int = 0) -> List[str]:
        """Get the names of all cuboids in an environment."""
        count = int(self.count[env_idx].item())
        return self.names[env_idx][:count]

    def clear(self, env_idx: Optional[int] = None) -> None:
        """Clear all cuboids from one or all environments.

        Args:
            env_idx: Environment to clear. If None, clears all environments.
        """
        if env_idx is not None:
            self.enable[env_idx, :] = 0
            self.count[env_idx] = 0
            self.names[env_idx] = [None] * self.max_n
        else:
            self.enable[:] = 0
            self.count[:] = 0
            self.names = [[None] * self.max_n for _ in range(self.num_envs)]

    # -------------------------------------------------------------------------
    # Warp Conversion
    # -------------------------------------------------------------------------

    def to_warp(self) -> CuboidDataWarp:
        """Convert to Warp struct for kernel launches.

        Creates a Warp struct with wp.from_torch() wrappers around the tensor data.
        The result can be passed directly to Warp kernels.

        Note:
            For CUDA graph compatibility, cache the result and reuse it across frames.
            The Warp arrays share memory with PyTorch tensors, so pose updates are
            automatically reflected without recreating the struct.

        Returns:
            CuboidDataWarp struct for use in Warp kernels.
        """
        s = CuboidDataWarp()
        s.dims = wp.from_torch(self.dims.view(-1, 4), dtype=wp.float32)
        s.inv_pose = wp.from_torch(self.inv_pose.view(-1, 8), dtype=wp.float32)
        s.enable = wp.from_torch(self.enable.view(-1), dtype=wp.uint8)
        s.n_per_env = wp.from_torch(self.count.view(-1), dtype=wp.int32)
        s.max_n = self.max_n
        s.num_envs = self.num_envs
        return s


# =============================================================================
# SDF Implementation (Generic Kernel Overloads)
# =============================================================================

_SDF_EPS = 1e-6


def is_obs_enabled(
    obs_set: CuboidDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
) -> wp.bool:
    """Check if a cuboid obstacle is enabled.

    Args:
        obs_set: Warp struct for cuboid data.
        env_idx: Environment index.
        local_idx: Local index of the cuboid within the environment.

    Returns:
        True if the cuboid is enabled, False otherwise.
    """
    if local_idx >= obs_set.n_per_env[env_idx]:
        return False
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return obs_set.enable[flat_idx] == wp.uint8(1)


def load_obstacle_transform(
    obs_set: CuboidDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
) -> wp.transform:
    """Load world-to-local transform for a cuboid obstacle.

    The returned transform converts world-frame points to local-frame.
    Use wp.transform_point(t, world_pt) to transform points.
    Use wp.transform_vector(wp.transform_inverse(t), grad_local) for gradients.

    Args:
        obs_set: Warp struct for cuboid data.
        env_idx: Environment index.
        local_idx: Local index of the cuboid within the environment.

    Returns:
        wp.transform for world-to-local transformation.
    """
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return load_transform_from_inv_pose(obs_set.inv_pose, flat_idx)


def compute_local_sdf(
    obs_set: CuboidDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.float32:
    """Compute SDF value for a cuboid obstacle (no gradient).

    The query point must already be in the obstacle's local frame.
    Use load_obstacle_transform() and wp.transform_point() to transform
    world-frame points before calling this function.

    Args:
        obs_set: Warp struct for cuboid data.
        env_idx: Environment index.
        local_idx: Local index of the cuboid within the environment.
        local_pt: Query point in obstacle local frame.

    Returns:
        Signed distance: negative inside, positive outside.
    """
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)

    # Load half-extents as scalars (avoid vec3 construction)
    hx = obs_set.dims[flat_idx, 0] * 0.5
    hy = obs_set.dims[flat_idx, 1] * 0.5
    hz = obs_set.dims[flat_idx, 2] * 0.5

    # Signed distance components along each axis
    qx = wp.abs(local_pt[0]) - hx
    qy = wp.abs(local_pt[1]) - hy
    qz = wp.abs(local_pt[2]) - hz

    # Clamped components for outside distance
    cx = wp.max(qx, 0.0)
    cy = wp.max(qy, 0.0)
    cz = wp.max(qz, 0.0)

    outside_dist = wp.sqrt(cx * cx + cy * cy + cz * cz)
    return outside_dist + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


def compute_local_sdf_with_grad(
    obs_set: CuboidDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.vec4:
    """Compute SDF and gradient for a cuboid obstacle in local frame.

    The query point must already be in the obstacle's local frame.
    The returned gradient is also in the local frame. Use
    wp.transform_vector(wp.transform_inverse(inv_t), grad_local) to
    convert to world frame.

    Uses scalar operations throughout to minimize register pressure
    (avoids intermediate vec3 constructions).

    Args:
        obs_set: Warp struct for cuboid data.
        env_idx: Environment index.
        local_idx: Local index of the cuboid within the environment.
        local_pt: Query point in obstacle local frame.

    Returns:
        vec4(signed_dist, grad_local_x, grad_local_y, grad_local_z).
    """
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)

    # Load half-extents as scalars (avoid vec3 construction)
    hx = obs_set.dims[flat_idx, 0] * 0.5
    hy = obs_set.dims[flat_idx, 1] * 0.5
    hz = obs_set.dims[flat_idx, 2] * 0.5

    # Signed distance components along each axis
    qx = wp.abs(local_pt[0]) - hx
    qy = wp.abs(local_pt[1]) - hy
    qz = wp.abs(local_pt[2]) - hz

    # Clamped components (reused for both distance and gradient)
    cx = wp.max(qx, 0.0)
    cy = wp.max(qy, 0.0)
    cz = wp.max(qz, 0.0)

    outside_dist = wp.sqrt(cx * cx + cy * cy + cz * cz)
    sdf = outside_dist + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)

    # Gradient computation using scalars (single vec4 construction at end)
    gx = wp.float32(0.0)
    gy = wp.float32(0.0)
    gz = wp.float32(0.0)

    if outside_dist > _SDF_EPS:
        # Outside: gradient = -sign(local_pt) * clamped_q / outside_dist
        inv_od = -1.0 / outside_dist
        gx = cx * inv_od
        gy = cy * inv_od
        gz = cz * inv_od
        # Correct for abs(): flip sign for negative coordinates
        if local_pt[0] < 0.0:
            gx = -gx
        if local_pt[1] < 0.0:
            gy = -gy
        if local_pt[2] < 0.0:
            gz = -gz
    else:
        # Inside: gradient toward nearest face (axis with largest q)
        max_q = wp.max(qx, wp.max(qy, qz))
        if wp.abs(qx - max_q) < _SDF_EPS:
            gx = -1.0
            if local_pt[0] < 0.0:
                gx = 1.0
        elif wp.abs(qy - max_q) < _SDF_EPS:
            gy = -1.0
            if local_pt[1] < 0.0:
                gy = 1.0
        else:
            gz = -1.0
            if local_pt[2] < 0.0:
                gz = 1.0

    return wp.vec4(sdf, gx, gy, gz)
