# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU storage and analytic SDF functions for finite cylinder obstacles."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import List, Optional, Union

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.geom.data.data_obstacle_bounds import update_world_aabb
from curobo._src.geom.data.helper_pose import get_obs_idx, load_transform_from_inv_pose
from curobo._src.geom.types import Cylinder, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise


@wp.struct
class CylinderDataWarp:
    """Warp tensor views for cylinder obstacle queries."""

    params: wp.array2d(dtype=wp.float32)
    world_bounds: wp.array2d(dtype=wp.float32)
    inv_pose: wp.array2d(dtype=wp.float32)
    enable: wp.array(dtype=wp.uint8)
    n_per_env: wp.array(dtype=wp.int32)
    max_n: wp.int32
    num_envs: wp.int32


@dataclass
class CylinderData:
    """GPU tensor storage for finite local-Z cylinder obstacles."""

    #: Geometry ``[radius, height, pad, pad]``.
    params: torch.Tensor

    #: World AABB ``[center_x, center_y, center_z, pad, extent_x, extent_y, extent_z, pad]``.
    world_bounds: torch.Tensor

    #: World-to-local poses ``[x, y, z, qw, qx, qy, qz, pad]``.
    inv_pose: torch.Tensor

    #: Per-cylinder enable flags.
    enable: torch.Tensor

    #: Unsigned collision-layer membership [num_envs, max_n].
    obstacle_layers: torch.Tensor = field(init=False)

    #: Number of loaded cylinders per environment.
    count: torch.Tensor

    #: Name lookup per environment.
    names: List[List[Optional[str]]]

    #: Maximum cylinders per environment.
    max_n: int

    #: Number of environments.
    num_envs: int

    #: Tensor device and dtype configuration.
    device_cfg: DeviceCfg

    def __post_init__(self) -> None:
        """Allocate common-layer metadata once alongside obstacle storage."""
        self.obstacle_layers = torch.ones_like(self.enable, dtype=torch.uint32)

    @classmethod
    def create_cache(cls, max_n: int, num_envs: int, device_cfg: DeviceCfg) -> CylinderData:
        """Create empty fixed-capacity cylinder storage."""
        params = torch.zeros(
            (num_envs, max_n, 4), dtype=device_cfg.dtype, device=device_cfg.device
        )
        inv_pose = torch.zeros(
            (num_envs, max_n, 8), dtype=device_cfg.dtype, device=device_cfg.device
        )
        inv_pose[..., 3] = 1.0
        world_bounds = torch.zeros(
            (num_envs, max_n, 8), dtype=device_cfg.dtype, device=device_cfg.device
        )
        enable = torch.zeros((num_envs, max_n), dtype=torch.uint8, device=device_cfg.device)
        count = torch.zeros((num_envs,), dtype=torch.int32, device=device_cfg.device)
        names = [[None for _ in range(max_n)] for _ in range(num_envs)]
        return cls(
            params=params,
            world_bounds=world_bounds,
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
    ) -> CylinderData:
        """Create cylinder storage from one scene configuration."""
        cylinders = scene_cfg.cylinder
        num_cylinders = len(cylinders) if cylinders else 0
        if max_n is None:
            max_n = max(num_cylinders, 1)
        instance = cls.create_cache(max_n, num_envs, device_cfg)
        if cylinders:
            instance.load_batch(cylinders, env_idx)
        return instance

    @classmethod
    def from_batch_scene_cfg(
        cls,
        scene_cfg_list: List[SceneCfg],
        device_cfg: DeviceCfg,
        max_n: Optional[int] = None,
    ) -> CylinderData:
        """Create cylinder storage for multiple environments."""
        num_envs = len(scene_cfg_list)
        counts = [len(scene.cylinder) if scene.cylinder else 0 for scene in scene_cfg_list]
        if max_n is None:
            max_n = max(counts) if counts else 1
        instance = cls.create_cache(max_n, num_envs, device_cfg)
        for env_idx, scene_cfg in enumerate(scene_cfg_list):
            if scene_cfg.cylinder:
                instance.load_batch(scene_cfg.cylinder, env_idx)
        return instance

    @staticmethod
    def _get_local_bounds(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local AABB lower and upper corners for cylinder parameters."""
        radius = params[..., 0]
        half_height = 0.5 * params[..., 1]
        extent = torch.stack((radius, radius, half_height), dim=-1)
        return -extent, extent

    def load_batch(self, cylinders: List[Cylinder], env_idx: int) -> None:
        """Replace one environment with a batch of cylinders."""
        self.obstacle_layers[env_idx].fill_(1)
        num_cylinders = len(cylinders)
        if num_cylinders > self.max_n:
            log_and_raise(f"Cannot load {num_cylinders} cylinders, max cache size is {self.max_n}")
        if num_cylinders == 0:
            self.clear(env_idx)
            return

        pose_batch = [cylinder.pose for cylinder in cylinders]
        geometry = torch.tensor(
            [[cylinder.radius, cylinder.height, 0.0, 0.0] for cylinder in cylinders],
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )
        world_pose = Pose.from_batch_list(pose_batch, self.device_cfg)
        self.params[env_idx, :num_cylinders].copy_(geometry)
        self.inv_pose[env_idx, :num_cylinders, :7].copy_(world_pose.inverse().get_pose_vector())
        lower, upper = self._get_local_bounds(geometry)
        update_world_aabb(self.world_bounds[env_idx, :num_cylinders], lower, upper, world_pose)
        self.enable[env_idx, :num_cylinders] = 1
        self.enable[env_idx, num_cylinders:] = 0
        self.names[env_idx] = [cylinder.name for cylinder in cylinders] + [None] * (
            self.max_n - num_cylinders
        )
        self.count[env_idx] = num_cylinders

    def add(self, cylinder: Cylinder, env_idx: int = 0) -> int:
        """Add one cylinder to an environment and return its local index."""
        return self.add_from_raw(
            name=cylinder.name,
            radius=cylinder.radius,
            height=cylinder.height,
            env_idx=env_idx,
            w_obj_pose=Pose.from_list(cylinder.pose, self.device_cfg),
        )

    def add_from_raw(
        self,
        name: str,
        radius: Union[float, torch.Tensor],
        height: Union[float, torch.Tensor],
        env_idx: int,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
    ) -> int:
        """Add one cylinder from scalar geometry and a forward or inverse pose."""
        if w_obj_pose is None and obj_w_pose is None:
            log_and_raise("Either w_obj_pose or obj_w_pose must be provided")
        current_count = int(self.count[env_idx].item())
        if current_count >= self.max_n:
            log_and_raise(f"Cannot add cylinder, cache is full ({self.max_n} cylinders)")
        if name in self.names[env_idx]:
            log_and_raise(f"Cylinder already exists with name: {name}")
        if w_obj_pose is not None:
            obj_w_pose = w_obj_pose.inverse()
        else:
            w_obj_pose = obj_w_pose.inverse()

        params = self.params[env_idx, current_count]
        params[0] = radius
        params[1] = height
        self.inv_pose[env_idx, current_count, :7] = obj_w_pose.get_pose_vector()
        lower, upper = self._get_local_bounds(params)
        update_world_aabb(self.world_bounds[env_idx, current_count], lower, upper, w_obj_pose)
        self.enable[env_idx, current_count] = 1
        self.names[env_idx][current_count] = name
        self.count[env_idx] += 1
        return current_count

    def update_pose(
        self,
        name: str,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        env_idx: int = 0,
    ) -> None:
        """Update an existing cylinder pose."""
        if w_obj_pose is None and obj_w_pose is None:
            log_and_raise("Either w_obj_pose or obj_w_pose must be provided")
        idx = self.get_idx(name, env_idx)
        if w_obj_pose is not None:
            obj_w_pose = w_obj_pose.inverse()
        else:
            w_obj_pose = obj_w_pose.inverse()
        self.inv_pose[env_idx, idx, :7] = obj_w_pose.get_pose_vector()
        lower, upper = self._get_local_bounds(self.params[env_idx, idx])
        update_world_aabb(self.world_bounds[env_idx, idx], lower, upper, w_obj_pose)

    def set_enabled(self, name: str, enabled: bool, env_idx: int = 0) -> None:
        """Enable or disable a cylinder."""
        self.enable[env_idx, self.get_idx(name, env_idx)] = int(enabled)

    def has_name(self, name: str, env_idx: int = 0) -> bool:
        """Return whether an environment contains a named cylinder."""
        return name in self.names[env_idx]

    def get_idx(self, name: str, env_idx: int = 0) -> int:
        """Return a cylinder's local environment index."""
        return self.names[env_idx].index(name)

    def get_active_count(self, env_idx: int = 0) -> int:
        """Return the loaded cylinder count for an environment."""
        return int(self.count[env_idx].item())

    def get_names(self, env_idx: int = 0) -> List[str]:
        """Return loaded cylinder names for an environment."""
        return self.names[env_idx][: self.get_active_count(env_idx)]

    def clear(self, env_idx: Optional[int] = None) -> None:
        """Clear cylinders from one or all environments."""
        layers = self.obstacle_layers if env_idx is None else self.obstacle_layers[env_idx]
        layers.fill_(1)
        if env_idx is not None:
            self.enable[env_idx, :] = 0
            self.count[env_idx] = 0
            self.names[env_idx] = [None] * self.max_n
            return
        self.enable[:] = 0
        self.count[:] = 0
        self.names = [[None] * self.max_n for _ in range(self.num_envs)]

    def to_warp(self) -> CylinderDataWarp:
        """Create Warp views over the cylinder tensors."""
        data = CylinderDataWarp()
        data.params = wp.from_torch(self.params.view(-1, 4), dtype=wp.float32)
        data.world_bounds = wp.from_torch(self.world_bounds.view(-1, 8), dtype=wp.float32)
        data.inv_pose = wp.from_torch(self.inv_pose.view(-1, 8), dtype=wp.float32)
        data.enable = wp.from_torch(self.enable.view(-1), dtype=wp.uint8)
        data.n_per_env = wp.from_torch(self.count.view(-1), dtype=wp.int32)
        data.max_n = self.max_n
        data.num_envs = self.num_envs
        return data


_SDF_EPS = 1.0e-6


def is_obs_enabled(obs_set: CylinderDataWarp, env_idx: wp.int32, local_idx: wp.int32) -> wp.bool:
    """Return whether a cylinder slot is loaded and enabled."""
    if local_idx >= obs_set.n_per_env[env_idx]:
        return False
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return obs_set.enable[flat_idx] == wp.uint8(1)


def load_obstacle_transform(
    obs_set: CylinderDataWarp, env_idx: wp.int32, local_idx: wp.int32
) -> wp.transform:
    """Load the world-to-cylinder transform."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return load_transform_from_inv_pose(obs_set.inv_pose, flat_idx)


def compute_local_sdf(
    obs_set: CylinderDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.float32:
    """Compute exact signed distance to a finite local-Z cylinder."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    radial_distance = wp.sqrt(local_pt[0] * local_pt[0] + local_pt[1] * local_pt[1])
    radial_delta = radial_distance - obs_set.params[flat_idx, 0]
    axial_delta = wp.abs(local_pt[2]) - 0.5 * obs_set.params[flat_idx, 1]
    outside_radial = wp.max(radial_delta, 0.0)
    outside_axial = wp.max(axial_delta, 0.0)
    outside_distance = wp.sqrt(outside_radial * outside_radial + outside_axial * outside_axial)
    return outside_distance + wp.min(wp.max(radial_delta, axial_delta), 0.0)


def compute_swept_sphere_clearance_lower_bound(
    obs_set: CylinderDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    current_world: wp.vec3,
    sweep_min_world: wp.vec3,
    sweep_max_world: wp.vec3,
    radius_adjusted: wp.float32,
    sweep_reach: wp.float32,
) -> wp.float32:
    """Return a conservative swept-sphere clearance bound from the cylinder AABB."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    center_x = obs_set.world_bounds[flat_idx, 0]
    center_y = obs_set.world_bounds[flat_idx, 1]
    center_z = obs_set.world_bounds[flat_idx, 2]
    extent_x = obs_set.world_bounds[flat_idx, 4]
    extent_y = obs_set.world_bounds[flat_idx, 5]
    extent_z = obs_set.world_bounds[flat_idx, 6]
    if sweep_max_world[0] < center_x - extent_x:
        return center_x - extent_x - sweep_max_world[0]
    if sweep_min_world[0] > center_x + extent_x:
        return sweep_min_world[0] - center_x - extent_x
    if sweep_max_world[1] < center_y - extent_y:
        return center_y - extent_y - sweep_max_world[1]
    if sweep_min_world[1] > center_y + extent_y:
        return sweep_min_world[1] - center_y - extent_y
    if sweep_max_world[2] < center_z - extent_z:
        return center_z - extent_z - sweep_max_world[2]
    if sweep_min_world[2] > center_z + extent_z:
        return sweep_min_world[2] - center_z - extent_z
    return wp.float32(0.0)


def compute_local_sdf_with_grad(
    obs_set: CylinderDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
    query_distance: wp.float32,
) -> wp.vec4:
    """Compute cylinder SDF and collision-cost gradient in the local frame."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    radial_distance = wp.sqrt(local_pt[0] * local_pt[0] + local_pt[1] * local_pt[1])
    radial_delta = radial_distance - obs_set.params[flat_idx, 0]
    axial_delta = wp.abs(local_pt[2]) - 0.5 * obs_set.params[flat_idx, 1]
    outside_radial = wp.max(radial_delta, 0.0)
    outside_axial = wp.max(axial_delta, 0.0)
    outside_distance = wp.sqrt(outside_radial * outside_radial + outside_axial * outside_axial)
    sdf = outside_distance + wp.min(wp.max(radial_delta, axial_delta), 0.0)

    gx = wp.float32(0.0)
    gy = wp.float32(0.0)
    gz = wp.float32(0.0)
    if outside_distance > _SDF_EPS:
        if outside_radial > 0.0 and radial_distance > _SDF_EPS:
            radial_scale = -outside_radial / (outside_distance * radial_distance)
            gx = radial_scale * local_pt[0]
            gy = radial_scale * local_pt[1]
        if outside_axial > 0.0:
            gz = -outside_axial / outside_distance
            if local_pt[2] < 0.0:
                gz = -gz
    elif radial_delta >= axial_delta:
        if radial_distance > _SDF_EPS:
            gx = -local_pt[0] / radial_distance
            gy = -local_pt[1] / radial_distance
        else:
            gx = -1.0
    else:
        gz = -1.0
        if local_pt[2] < 0.0:
            gz = 1.0

    return wp.vec4(sdf, gx, gy, gz)
