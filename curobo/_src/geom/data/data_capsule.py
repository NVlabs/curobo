# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU storage and analytic SDF functions for capsule obstacles."""

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
from curobo._src.geom.types import Capsule, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise


@wp.struct
class CapsuleDataWarp:
    """Warp tensor views for capsule obstacle queries."""

    params: wp.array2d(dtype=wp.float32)
    world_bounds: wp.array2d(dtype=wp.float32)
    inv_pose: wp.array2d(dtype=wp.float32)
    enable: wp.array(dtype=wp.uint8)
    n_per_env: wp.array(dtype=wp.int32)
    max_n: wp.int32
    num_envs: wp.int32


@dataclass
class CapsuleData:
    """GPU tensor storage for capsule obstacles.

    A capsule is the Minkowski sum of its local-frame ``base``-to-``tip``
    segment and a sphere. Coincident endpoints therefore represent a sphere.
    """

    #: Geometry ``[base_xyz, unit_axis_xyz, segment_length, radius]``.
    params: torch.Tensor

    #: World AABB ``[center_x, center_y, center_z, pad, extent_x, extent_y, extent_z, pad]``.
    world_bounds: torch.Tensor

    #: World-to-local poses ``[x, y, z, qw, qx, qy, qz, pad]``.
    inv_pose: torch.Tensor

    #: Per-capsule enable flags.
    enable: torch.Tensor

    #: Unsigned collision-layer membership [num_envs, max_n].
    obstacle_layers: torch.Tensor = field(init=False)

    #: Number of loaded capsules per environment.
    count: torch.Tensor

    #: Name lookup per environment.
    names: List[List[Optional[str]]]

    #: Maximum capsules per environment.
    max_n: int

    #: Number of environments.
    num_envs: int

    #: Tensor device and dtype configuration.
    device_cfg: DeviceCfg

    def __post_init__(self) -> None:
        """Allocate common-layer metadata once alongside obstacle storage."""
        self.obstacle_layers = torch.ones_like(self.enable, dtype=torch.uint32)

    @classmethod
    def create_cache(cls, max_n: int, num_envs: int, device_cfg: DeviceCfg) -> CapsuleData:
        """Create empty fixed-capacity capsule storage."""
        params = torch.zeros(
            (num_envs, max_n, 8), dtype=device_cfg.dtype, device=device_cfg.device
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
    ) -> CapsuleData:
        """Create capsule storage from one scene configuration."""
        capsules = scene_cfg.capsule
        num_capsules = len(capsules) if capsules else 0
        if max_n is None:
            max_n = max(num_capsules, 1)
        instance = cls.create_cache(max_n, num_envs, device_cfg)
        if capsules:
            instance.load_batch(capsules, env_idx)
        return instance

    @classmethod
    def from_batch_scene_cfg(
        cls,
        scene_cfg_list: List[SceneCfg],
        device_cfg: DeviceCfg,
        max_n: Optional[int] = None,
    ) -> CapsuleData:
        """Create capsule storage for multiple environments."""
        num_envs = len(scene_cfg_list)
        counts = [len(scene.capsule) if scene.capsule else 0 for scene in scene_cfg_list]
        if max_n is None:
            max_n = max(counts) if counts else 1
        instance = cls.create_cache(max_n, num_envs, device_cfg)
        for env_idx, scene_cfg in enumerate(scene_cfg_list):
            if scene_cfg.capsule:
                instance.load_batch(scene_cfg.capsule, env_idx)
        return instance

    @staticmethod
    def _pack_geometry(
        base: torch.Tensor,
        tip: torch.Tensor,
        radius: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Pack endpoints as a base, unit axis, segment length, and radius."""
        segment = tip - base
        segment_length = torch.linalg.vector_norm(segment, dim=-1, keepdim=True)
        is_segment = segment_length > _SDF_EPS
        unit_axis = torch.where(
            is_segment,
            segment / torch.clamp_min(segment_length, _SDF_EPS),
            torch.zeros_like(segment),
        )
        segment_length = torch.where(
            is_segment,
            segment_length,
            torch.zeros_like(segment_length),
        )
        radius_tensor = torch.as_tensor(radius, dtype=base.dtype, device=base.device)
        if radius_tensor.ndim == segment_length.ndim - 1:
            radius_tensor = radius_tensor.unsqueeze(-1)
        radius_tensor = torch.broadcast_to(radius_tensor, segment_length.shape)
        return torch.cat((base, unit_axis, segment_length, radius_tensor), dim=-1)

    @staticmethod
    def _get_local_bounds(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local AABB lower and upper corners for capsule parameters."""
        base = params[..., :3]
        tip = base + params[..., 3:6] * params[..., 6:7]
        radius = params[..., 7:8]
        lower = torch.minimum(base, tip) - radius
        upper = torch.maximum(base, tip) + radius
        return lower, upper

    def load_batch(self, capsules: List[Capsule], env_idx: int) -> None:
        """Replace one environment with a batch of capsules."""
        self.obstacle_layers[env_idx].fill_(1)
        num_capsules = len(capsules)
        if num_capsules > self.max_n:
            log_and_raise(f"Cannot load {num_capsules} capsules, max cache size is {self.max_n}")
        if num_capsules == 0:
            self.clear(env_idx)
            return

        pose_batch = [capsule.pose for capsule in capsules]
        base = torch.tensor(
            [capsule.base for capsule in capsules],
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )
        tip = torch.tensor(
            [capsule.tip for capsule in capsules],
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )
        radius = torch.tensor(
            [capsule.radius for capsule in capsules],
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )
        geometry = self._pack_geometry(base, tip, radius)
        world_pose = Pose.from_batch_list(pose_batch, self.device_cfg)
        self.params[env_idx, :num_capsules].copy_(geometry)
        self.inv_pose[env_idx, :num_capsules, :7].copy_(world_pose.inverse().get_pose_vector())
        lower, upper = self._get_local_bounds(geometry)
        update_world_aabb(self.world_bounds[env_idx, :num_capsules], lower, upper, world_pose)
        self.enable[env_idx, :num_capsules] = 1
        self.enable[env_idx, num_capsules:] = 0
        self.names[env_idx] = [capsule.name for capsule in capsules] + [None] * (
            self.max_n - num_capsules
        )
        self.count[env_idx] = num_capsules

    def add(self, capsule: Capsule, env_idx: int = 0) -> int:
        """Add one capsule to an environment and return its local index."""
        return self.add_from_raw(
            name=capsule.name,
            base=self.device_cfg.to_device(capsule.base),
            tip=self.device_cfg.to_device(capsule.tip),
            radius=capsule.radius,
            env_idx=env_idx,
            w_obj_pose=Pose.from_list(capsule.pose, self.device_cfg),
        )

    def add_from_raw(
        self,
        name: str,
        base: torch.Tensor,
        tip: torch.Tensor,
        radius: Union[float, torch.Tensor],
        env_idx: int,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
    ) -> int:
        """Add one capsule from tensors and a forward or inverse pose."""
        if w_obj_pose is None and obj_w_pose is None:
            log_and_raise("Either w_obj_pose or obj_w_pose must be provided")
        current_count = int(self.count[env_idx].item())
        if current_count >= self.max_n:
            log_and_raise(f"Cannot add capsule, cache is full ({self.max_n} capsules)")
        if name in self.names[env_idx]:
            log_and_raise(f"Capsule already exists with name: {name}")
        if w_obj_pose is not None:
            obj_w_pose = w_obj_pose.inverse()
        else:
            w_obj_pose = obj_w_pose.inverse()

        params = self.params[env_idx, current_count]
        params.copy_(self._pack_geometry(base, tip, radius))
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
        """Update an existing capsule pose."""
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
        """Enable or disable a capsule."""
        self.enable[env_idx, self.get_idx(name, env_idx)] = int(enabled)

    def has_name(self, name: str, env_idx: int = 0) -> bool:
        """Return whether an environment contains a named capsule."""
        return name in self.names[env_idx]

    def get_idx(self, name: str, env_idx: int = 0) -> int:
        """Return a capsule's local environment index."""
        return self.names[env_idx].index(name)

    def get_active_count(self, env_idx: int = 0) -> int:
        """Return the loaded capsule count for an environment."""
        return int(self.count[env_idx].item())

    def get_names(self, env_idx: int = 0) -> List[str]:
        """Return loaded capsule names for an environment."""
        return self.names[env_idx][: self.get_active_count(env_idx)]

    def clear(self, env_idx: Optional[int] = None) -> None:
        """Clear capsules from one or all environments."""
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

    def to_warp(self) -> CapsuleDataWarp:
        """Create Warp views over the capsule tensors."""
        data = CapsuleDataWarp()
        data.params = wp.from_torch(self.params.view(-1, 8), dtype=wp.float32)
        data.world_bounds = wp.from_torch(self.world_bounds.view(-1, 8), dtype=wp.float32)
        data.inv_pose = wp.from_torch(self.inv_pose.view(-1, 8), dtype=wp.float32)
        data.enable = wp.from_torch(self.enable.view(-1), dtype=wp.uint8)
        data.n_per_env = wp.from_torch(self.count.view(-1), dtype=wp.int32)
        data.max_n = self.max_n
        data.num_envs = self.num_envs
        return data


_SDF_EPS = 1.0e-6


def is_obs_enabled(obs_set: CapsuleDataWarp, env_idx: wp.int32, local_idx: wp.int32) -> wp.bool:
    """Return whether a capsule slot is loaded and enabled."""
    if local_idx >= obs_set.n_per_env[env_idx]:
        return False
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return obs_set.enable[flat_idx] == wp.uint8(1)


def load_obstacle_transform(
    obs_set: CapsuleDataWarp, env_idx: wp.int32, local_idx: wp.int32
) -> wp.transform:
    """Load the world-to-capsule transform."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return load_transform_from_inv_pose(obs_set.inv_pose, flat_idx)


@wp.func
def _capsule_delta(
    obs_set: CapsuleDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.vec4:
    """Return point-to-segment delta xyz and capsule radius."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    ax = obs_set.params[flat_idx, 0]
    ay = obs_set.params[flat_idx, 1]
    az = obs_set.params[flat_idx, 2]
    axis_x = obs_set.params[flat_idx, 3]
    axis_y = obs_set.params[flat_idx, 4]
    axis_z = obs_set.params[flat_idx, 5]
    segment_length = obs_set.params[flat_idx, 6]
    pax = local_pt[0] - ax
    pay = local_pt[1] - ay
    paz = local_pt[2] - az
    segment_t = wp.float32(0.0)
    if segment_length > _SDF_EPS:
        segment_t = wp.clamp(
            pax * axis_x + pay * axis_y + paz * axis_z,
            0.0,
            segment_length,
        )
    return wp.vec4(
        pax - segment_t * axis_x,
        pay - segment_t * axis_y,
        paz - segment_t * axis_z,
        obs_set.params[flat_idx, 7],
    )


def compute_local_sdf(
    obs_set: CapsuleDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.float32:
    """Compute exact signed distance to a local-frame capsule."""
    delta = _capsule_delta(obs_set, env_idx, local_idx, local_pt)
    return wp.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]) - delta[3]


def compute_swept_sphere_clearance_lower_bound(
    obs_set: CapsuleDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    current_world: wp.vec3,
    sweep_min_world: wp.vec3,
    sweep_max_world: wp.vec3,
    radius_adjusted: wp.float32,
    sweep_reach: wp.float32,
) -> wp.float32:
    """Return a conservative swept-sphere clearance bound from the capsule AABB."""
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
    obs_set: CapsuleDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
    query_distance: wp.float32,
) -> wp.vec4:
    """Compute capsule SDF and collision-cost gradient in the local frame."""
    delta = _capsule_delta(obs_set, env_idx, local_idx, local_pt)
    delta_length = wp.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
    sdf = delta_length - delta[3]
    if delta_length > _SDF_EPS:
        inv_length = -1.0 / delta_length
        return wp.vec4(
            sdf,
            delta[0] * inv_length,
            delta[1] * inv_length,
            delta[2] * inv_length,
        )

    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    segment_length = obs_set.params[flat_idx, 6]
    if segment_length <= _SDF_EPS:
        return wp.vec4(sdf, -1.0, 0.0, 0.0)

    unit_axis = wp.vec3(
        obs_set.params[flat_idx, 3],
        obs_set.params[flat_idx, 4],
        obs_set.params[flat_idx, 5],
    )
    reference = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(unit_axis[0]) > wp.abs(unit_axis[1]):
        reference = wp.vec3(0.0, 1.0, 0.0)
    normal = wp.cross(unit_axis, reference)
    normal_length = wp.length(normal)
    if normal_length <= _SDF_EPS:
        return wp.vec4(sdf, -1.0, 0.0, 0.0)
    normal = -normal / normal_length
    return wp.vec4(sdf, normal[0], normal[1], normal[2])
