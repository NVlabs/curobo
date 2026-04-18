# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""GPU data storage and SDF functions for voxel grid obstacles (ESDF).

This module provides:
- VoxelData: Python dataclass owning GPU tensors for ESDF voxel grids
- VoxelDataWarp: Warp struct for kernel access
- is_obs_enabled, load_obstacle_transform, compute_local_sdf,
  compute_local_sdf_with_grad: Generic kernel overloads

Note: ESDF features are stored as float16 for memory efficiency. Warp kernels
convert to float32 at runtime for computation.
"""

from __future__ import annotations

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
from curobo._src.geom.types import SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise

# =============================================================================
# Python Dataclass
# =============================================================================


@dataclass
class VoxelData:
    """GPU tensor storage for voxel grid (ESDF) obstacles.

    This dataclass owns the GPU tensor buffers for voxel grid obstacles and provides
    methods for adding, updating, and querying voxel grids. Voxel grids store
    Euclidean Signed Distance Field (ESDF) values for each voxel.

    ESDF features are stored as float16 to reduce memory usage by 50%.

    Tensor layouts:
        - params: (num_envs, max_n, 4) - [dims_x, dims_y, dims_z, voxel_size] (voxel counts)
        - dims: (num_envs, max_n, 4) - world bounding box [x, y, z, pad] in meters
        - inv_pose: (num_envs, max_n, 8) - inverse pose [x, y, z, qw, qx, qy, qz, pad]
        - features: (num_envs, max_n, n_voxels, 1) - ESDF values per voxel (float16)
        - enable: (num_envs, max_n) - uint8 flags
        - count: (num_envs,) - int32 active counts
    """

    #: Voxel grid parameters [dims_x, dims_y, dims_z, voxel_size] (voxel counts).
    params: torch.Tensor

    #: Voxel grid world dimensions [x, y, z, pad] in meters.
    dims: torch.Tensor

    #: Inverse pose [x, y, z, qw, qx, qy, qz] with padding.
    inv_pose: torch.Tensor

    #: Enable flag per voxel grid. 1 = active, 0 = disabled.
    enable: torch.Tensor

    #: ESDF feature values per voxel (float16).
    features: torch.Tensor

    #: Number of active voxel grids per environment.
    count: torch.Tensor

    #: Name lookup per environment. names[env_idx][voxel_idx] = name.
    names: List[List[Optional[str]]]

    #: Maximum number of voxel grids per environment.
    max_n: int

    #: Number of environments.
    num_envs: int

    #: Device configuration.
    device_cfg: DeviceCfg

    #: Maximum ESDF distance for initialization.
    max_esdf_distance: float

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def create_cache(
        cls,
        max_n: int,
        num_envs: int,
        device_cfg: DeviceCfg,
        grid_dims: List[float],
        voxel_size: float,
        max_esdf_distance: float = 10000.0,
    ) -> VoxelData:
        """Create an empty cache for voxel grid obstacles.

        Args:
            max_n: Maximum number of voxel grids (layers) per environment.
            num_envs: Number of environments.
            device_cfg: Device and dtype configuration.
            grid_dims: Dimensions of the voxel grid [x, y, z] in meters.
            voxel_size: Size of each voxel in meters.
            max_esdf_distance: Maximum ESDF distance for initialization.

        Returns:
            Empty VoxelData with allocated GPU buffers.
        """
        # Calculate number of voxels from grid shape
        grid_shape = VoxelGrid(
            "temp", pose=[0, 0, 0, 1, 0, 0, 0], dims=grid_dims, voxel_size=voxel_size
        ).get_grid_shape()[0]
        n_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]

        params = torch.zeros(
            (num_envs, max_n, 4), dtype=device_cfg.dtype, device=device_cfg.device
        )
        dims = torch.zeros(
            (num_envs, max_n, 4), dtype=device_cfg.dtype, device=device_cfg.device
        )
        inv_pose = torch.zeros(
            (num_envs, max_n, 8), dtype=device_cfg.dtype, device=device_cfg.device
        )
        inv_pose[..., 3] = 1.0  # Identity quaternion (qw=1)

        enable = torch.zeros((num_envs, max_n), dtype=torch.uint8, device=device_cfg.device)
        count = torch.zeros((num_envs,), device=device_cfg.device, dtype=torch.int32)

        # Features stored as float16 for memory efficiency
        features = torch.full(
            (num_envs, max_n, n_voxels, 1),
            max_esdf_distance,
            device=device_cfg.device,
            dtype=torch.float16,
        )

        names = [[None for _ in range(max_n)] for _ in range(num_envs)]

        return cls(
            params=params,
            dims=dims,
            inv_pose=inv_pose,
            enable=enable,
            features=features,
            count=count,
            names=names,
            max_n=max_n,
            num_envs=num_envs,
            device_cfg=device_cfg,
            max_esdf_distance=max_esdf_distance,
        )

    @classmethod
    def create_from_voxel_grids(
        cls,
        voxel_grids: List[VoxelGrid],
        device_cfg: DeviceCfg,
        env_idx: int = 0,
        num_envs: int = 1,
        max_n: Optional[int] = None,
        max_esdf_distance: float = 1000.0,
    ) -> VoxelData:
        """Create VoxelData by directly referencing feature tensors from VoxelGrids.

        This method avoids copying feature data by directly using the feature tensor
        from the provided VoxelGrid. Only supports exactly 1 voxel grid and 1 environment.

        Args:
            voxel_grids: List containing exactly 1 VoxelGrid with feature_tensor.
            device_cfg: Device and dtype configuration.
            env_idx: Environment index to load into (must be 0).
            num_envs: Number of environments (must be 1).
            max_n: Maximum number of voxel grids (must be 1 or None).
            max_esdf_distance: Maximum ESDF distance for initialization.

        Returns:
            VoxelData with feature tensor referenced directly from the VoxelGrid.

        Raises:
            ValueError: If len(voxel_grids) != 1, num_envs != 1, or max_n > 1.
        """
        import numpy as np

        num_voxels = len(voxel_grids)
        if num_voxels != 1:
            log_and_raise(
                f"create_from_voxel_grids only supports exactly 1 voxel grid, "
                f"got {num_voxels}. Use create_cache + load_batch for multiple grids."
            )

        if num_envs != 1:
            log_and_raise(
                f"create_from_voxel_grids only supports num_envs=1, got {num_envs}. "
                f"Use create_cache + load_batch for multiple environments."
            )

        if max_n is None:
            max_n = 1
        elif max_n != 1:
            log_and_raise(
                f"create_from_voxel_grids only supports max_n=1, got {max_n}. "
                f"Use create_cache + load_batch for multiple grids."
            )

        voxel_grid = voxel_grids[0]
        if voxel_grid.feature_tensor is None:
            log_and_raise(
                f"VoxelGrid '{voxel_grid.name}' must have a feature_tensor to use "
                f"create_from_voxel_grids."
            )

        # Use feature tensor directly by reference (reshape to expected layout)
        # Expected shape: (num_envs=1, max_n=1, n_voxels, 1)
        feature_tensor = voxel_grid.feature_tensor
        n_voxels_flat = feature_tensor.numel()

        # Reshape to (1, 1, n_voxels, 1) - creates a view, no data copy
        features = feature_tensor.view(1, 1, n_voxels_flat, 1)

        # Allocate metadata tensors
        params = torch.zeros(
            (num_envs, max_n, 4), dtype=device_cfg.dtype, device=device_cfg.device
        )
        dims = torch.zeros(
            (num_envs, max_n, 4), dtype=device_cfg.dtype, device=device_cfg.device
        )
        inv_pose = torch.zeros(
            (num_envs, max_n, 8), dtype=device_cfg.dtype, device=device_cfg.device
        )
        inv_pose[..., 3] = 1.0  # Identity quaternion (qw=1)

        enable = torch.zeros((num_envs, max_n), dtype=torch.uint8, device=device_cfg.device)
        count = torch.zeros((num_envs,), device=device_cfg.device, dtype=torch.int32)

        names = [[None for _ in range(max_n)] for _ in range(num_envs)]

        # Populate metadata for env_idx
        w_T_b = Pose.from_batch_list([voxel_grid.pose], device_cfg=device_cfg)
        b_T_w = w_T_b.inverse()

        dims_t = torch.as_tensor(
            np.array([voxel_grid.dims]), device=device_cfg.device, dtype=device_cfg.dtype
        )
        size_t = torch.as_tensor(
            np.array([voxel_grid.voxel_size]), device=device_cfg.device, dtype=device_cfg.dtype
        ).unsqueeze(-1)
        grid_t = dims_t / size_t
        params_t = torch.cat([grid_t, size_t], dim=-1)

        params[env_idx, 0, :] = params_t[0]
        dims[env_idx, 0, :3] = dims_t[0]
        inv_pose[env_idx, 0, :7] = b_T_w.get_pose_vector()[0]
        enable[env_idx, 0] = 1
        names[env_idx][0] = voxel_grid.name
        count[env_idx] = 1

        return cls(
            params=params,
            dims=dims,
            inv_pose=inv_pose,
            enable=enable,
            features=features,
            count=count,
            names=names,
            max_n=max_n,
            num_envs=num_envs,
            device_cfg=device_cfg,
            max_esdf_distance=max_esdf_distance,
        )

    @classmethod
    def from_scene_cfg(
        cls,
        scene_cfg: SceneCfg,
        device_cfg: DeviceCfg,
        env_idx: int = 0,
        num_envs: int = 1,
        max_n: Optional[int] = None,
        max_esdf_distance: float = 100.0,
    ) -> VoxelData:
        """Create VoxelData from a scene configuration.

        Args:
            scene_cfg: Scene configuration containing voxel grid obstacles.
            device_cfg: Device and dtype configuration.
            env_idx: Environment index to load the voxel grids into.
            num_envs: Number of environments.
            max_n: Maximum voxel grids to allocate. If None, uses len(scene_cfg.voxel).
            max_esdf_distance: Maximum ESDF distance for initialization.

        Returns:
            VoxelData populated with voxel grids from the scene config.
        """
        voxels = scene_cfg.voxel
        num_voxels = len(voxels) if voxels else 0

        if num_voxels == 0:
            return cls.create_cache(
                max_n=max_n or 1,
                num_envs=num_envs,
                device_cfg=device_cfg,
                grid_dims=[1.0, 1.0, 1.0],
                voxel_size=0.02,
                max_esdf_distance=max_esdf_distance,
            )

        if max_n is None:
            max_n = num_voxels

        # Use direct reference when exactly 1 voxel grid and 1 env, otherwise use cache + load_batch
        if num_voxels == 1 and max_n == 1 and num_envs == 1:
            # Direct reference to feature tensor (no copy)
            instance = cls.create_from_voxel_grids(
                voxels, device_cfg, env_idx, num_envs, max_n, max_esdf_distance
            )
        else:
            # Multiple grids require cache allocation and data copy
            first_voxel = voxels[0]
            instance = cls.create_cache(
                max_n=max_n,
                num_envs=num_envs,
                device_cfg=device_cfg,
                grid_dims=first_voxel.dims,
                voxel_size=first_voxel.voxel_size,
                max_esdf_distance=max_esdf_distance,
            )
            instance.load_batch(voxels, env_idx)

        return instance

    @classmethod
    def from_batch_scene_cfg(
        cls,
        scene_cfg_list: List[SceneCfg],
        device_cfg: DeviceCfg,
        max_n: Optional[int] = None,
        max_esdf_distance: float = 100.0,
    ) -> VoxelData:
        """Create VoxelData from a list of scene configurations.

        Args:
            scene_cfg_list: List of scene configurations, one per environment.
            device_cfg: Device and dtype configuration.
            max_n: Maximum voxel grids per environment. If None, uses max across scenes.
            max_esdf_distance: Maximum ESDF distance for initialization.

        Returns:
            VoxelData populated with voxel grids from all scene configs.
        """
        num_envs = len(scene_cfg_list)
        voxel_counts = [len(cfg.voxel) if cfg.voxel else 0 for cfg in scene_cfg_list]

        if max_n is None:
            max_n = max(voxel_counts) if voxel_counts else 1

        first_voxel = None
        for cfg in scene_cfg_list:
            if cfg.voxel and len(cfg.voxel) > 0:
                first_voxel = cfg.voxel[0]
                break

        if first_voxel is None:
            return cls.create_cache(
                max_n=max_n,
                num_envs=num_envs,
                device_cfg=device_cfg,
                grid_dims=[1.0, 1.0, 1.0],
                voxel_size=0.02,
                max_esdf_distance=max_esdf_distance,
            )

        instance = cls.create_cache(
            max_n=max_n,
            num_envs=num_envs,
            device_cfg=device_cfg,
            grid_dims=first_voxel.dims,
            voxel_size=first_voxel.voxel_size,
            max_esdf_distance=max_esdf_distance,
        )

        for env_idx, scene_cfg in enumerate(scene_cfg_list):
            if scene_cfg.voxel:
                instance.load_batch(scene_cfg.voxel, env_idx)

        return instance

    # -------------------------------------------------------------------------
    # Load / Add Methods
    # -------------------------------------------------------------------------

    def load_batch(self, voxel_grids: List[VoxelGrid], env_idx: int) -> None:
        """Load a batch of voxel grids into an environment, replacing existing ones.

        Args:
            voxel_grids: List of VoxelGrid objects to load.
            env_idx: Environment index to load into.
        """
        import numpy as np

        num_voxels = len(voxel_grids)
        if num_voxels > self.max_n:
            log_and_raise(
                f"Cannot load {num_voxels} voxel grids, max cache size is {self.max_n}"
            )

        if num_voxels == 0:
            self.enable[env_idx, :] = 0
            self.count[env_idx] = 0
            return

        pose_batch = [v.pose for v in voxel_grids]
        dims_batch = [v.dims for v in voxel_grids]
        size_batch = [v.voxel_size for v in voxel_grids]
        names_batch = [v.name for v in voxel_grids]

        w_T_b = Pose.from_batch_list(pose_batch, device_cfg=self.device_cfg)
        b_T_w = w_T_b.inverse()

        dims_t = torch.as_tensor(
            np.array(dims_batch), device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )


        size_t = torch.as_tensor(
            np.array(size_batch), device=self.device_cfg.device, dtype=self.device_cfg.dtype
        ).unsqueeze(-1)
        grid_t = dims_t / size_t
        params_t = torch.cat([grid_t, size_t], dim=-1)

        self.params[env_idx, :num_voxels, :] = params_t

        # Compute world dimensions: dims_in_meters = dims_in_voxels * voxel_size
        self.dims[env_idx, :num_voxels, :3] = dims_t
        self.inv_pose[env_idx, :num_voxels, :7] = b_T_w.get_pose_vector()
        self.enable[env_idx, :num_voxels] = 1
        self.enable[env_idx, num_voxels:] = 0
        self.names[env_idx][:num_voxels] = names_batch
        self.count[env_idx] = num_voxels


        for i in range(num_voxels):
            flat_buffer = voxel_grids[i].feature_tensor.view(-1)
            buffer_capacity = self.features.shape[2]
            n_new = flat_buffer.shape[0]
            if n_new > buffer_capacity:
                log_and_raise(
                    f"Feature tensor too large for buffer: capacity={buffer_capacity}"
                    f" new={n_new}. Increase max_voxels_per_layer."
                )
            grid_shape = voxel_grids[i].get_grid_shape()[0]
            dims_count = grid_shape[0] * grid_shape[1] * grid_shape[2]
            if n_new != dims_count:
                log_and_raise(
                    f"Feature tensor size {n_new} doesn't match grid dims"
                    f" {grid_shape} = {dims_count}"
                )
            self.features[env_idx, i, :n_new, :] = flat_buffer.view(-1, 1)
            if n_new < buffer_capacity:
                self.features[env_idx, i, n_new:, :] = 0


    # -------------------------------------------------------------------------
    # Update Methods
    # -------------------------------------------------------------------------

    def update_data(
        self,
        voxel_grid: VoxelGrid,
        env_idx: int = 0,
        name: Optional[str] = None,
    ) -> None:
        """Update parameters and ESDF values of a voxel grid.

        Args:
            voxel_grid: New voxel grid data.
            env_idx: Environment index.
            name: Name of voxel grid to update. If provided, uses this name.
        """
        voxel_name = name if name is not None else voxel_grid.name
        idx = self.get_idx(voxel_name, env_idx)

        feature_tensor = voxel_grid.feature_tensor.view(voxel_grid.feature_tensor.shape[0], -1)
        buffer_capacity = self.features.shape[2]
        n_new = feature_tensor.shape[0]
        if n_new > buffer_capacity:
            log_and_raise(
                f"Feature tensor too large for buffer: capacity={buffer_capacity}"
                f" new={n_new}. Increase max_voxels_per_layer."
            )

        self.features[env_idx, idx, :n_new, :].copy_(feature_tensor.to(torch.float16))
        if n_new < buffer_capacity:
            self.features[env_idx, idx, n_new:, :] = 0
        # params[:3] stores voxel counts (matches load_batch and collision kernel)
        dims_t = torch.as_tensor(
            voxel_grid.dims, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        grid_t = dims_t / voxel_grid.voxel_size
        self.params[env_idx, idx, :3].copy_(grid_t)
        self.params[env_idx, idx, 3] = voxel_grid.voxel_size
        # dims[:3] stores world dimensions in meters
        self.dims[env_idx, idx, :3] = dims_t
        self.inv_pose[env_idx, idx, :7] = (
            Pose.from_list(voxel_grid.pose, self.device_cfg).inverse().get_pose_vector()
        )
        self.enable[env_idx, idx] = 1

    def update_features(
        self,
        features: torch.Tensor,
        name: str,
        env_idx: int = 0,
    ) -> None:
        """Update ESDF feature values in a voxel grid.

        Args:
            features: New ESDF feature values.
            name: Name of voxel grid to update.
            env_idx: Environment index.
        """
        idx = self.get_idx(name, env_idx)
        self.features[env_idx, idx, :] = features.to(dtype=torch.float16)

    def update_pose(
        self,
        name: str,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        env_idx: int = 0,
    ) -> None:
        """Update the pose of an existing voxel grid.

        Args:
            name: Name of the voxel grid to update.
            w_obj_pose: New pose in world frame.
            obj_w_pose: New inverse pose. Used if w_obj_pose is None.
            env_idx: Environment index.
        """
        if w_obj_pose is None and obj_w_pose is None:
            log_and_raise("Either w_obj_pose or obj_w_pose must be provided")

        idx = self.get_idx(name, env_idx)

        if w_obj_pose is not None:
            obj_w_pose = w_obj_pose.inverse()

        self.inv_pose[env_idx, idx, :7] = obj_w_pose.get_pose_vector()

    def set_enabled(self, name: str, enabled: bool, env_idx: int = 0) -> None:
        """Enable or disable a voxel grid for collision checking."""
        idx = self.get_idx(name, env_idx)
        self.enable[env_idx, idx] = int(enabled)

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_idx(self, name: str, env_idx: int = 0) -> int:
        """Get the index of a voxel grid by name."""
        try:
            return self.names[env_idx].index(name)
        except ValueError:
            log_and_raise(f"Voxel grid with name '{name}' not found in environment {env_idx}")

    def get_active_count(self, env_idx: int = 0) -> int:
        """Get the number of active voxel grids in an environment."""
        return int(self.count[env_idx].item())

    def get_names(self, env_idx: int = 0) -> List[str]:
        """Get the names of all voxel grids in an environment."""
        count = int(self.count[env_idx].item())
        return self.names[env_idx][:count]

    def get_voxel_grid(self, name: str, env_idx: int = 0) -> VoxelGrid:
        """Get a voxel grid from the cache.

        Args:
            name: Name of the voxel grid.
            env_idx: Environment index.

        Returns:
            VoxelGrid object with current data.
        """
        import numpy as np

        idx = self.get_idx(name, env_idx)

        voxel_params = np.round(
            self.params[env_idx, idx, :].cpu().numpy().astype(np.float64), 6
        ).tolist()
        voxel_pose = Pose(
            position=self.inv_pose[env_idx, idx, :3],
            quaternion=self.inv_pose[env_idx, idx, 3:7],
        )
        voxel_features = self.features[env_idx, idx, :]

        return VoxelGrid(
            name=name,
            dims=voxel_params[:3],
            pose=voxel_pose.to_list(),
            voxel_size=voxel_params[3],
            feature_tensor=voxel_features,
        )

    def get_grid_shape(
        self, env_idx: int = 0, name: Optional[str] = None, idx: int = 0
    ) -> torch.Size:
        """Get the shape of a voxel grid's feature tensor."""
        if name is not None:
            idx = self.get_idx(name, env_idx)
        return self.features[env_idx, idx].shape

    def clear(self, env_idx: Optional[int] = None) -> None:
        """Clear all voxel grids from one or all environments.

        Args:
            env_idx: Environment to clear. If None, clears all environments.
        """
        if env_idx is not None:
            self.enable[env_idx, :] = 0
            self.count[env_idx] = 0
            self.features[env_idx, :] = -self.max_esdf_distance
            self.names[env_idx] = [None] * self.max_n
        else:
            self.enable[:] = 0
            self.count[:] = 0
            self.features[:] = -self.max_esdf_distance
            self.names = [[None] * self.max_n for _ in range(self.num_envs)]

    # -------------------------------------------------------------------------
    # Warp Conversion
    # -------------------------------------------------------------------------

    def to_warp(self) -> VoxelDataWarp:
        """Convert to Warp struct for kernel launches.

        Features are stored as float16 in the Warp struct. Kernels convert
        to float32 at read time for computation.

        Returns:
            VoxelDataWarp struct for use in Warp kernels.
        """
        n_voxels_per_layer = self.features.shape[2]

        s = VoxelDataWarp()
        s.params = wp.from_torch(self.params.view(-1, 4), dtype=wp.float32)
        s.dims = wp.from_torch(self.dims.view(-1, 4), dtype=wp.float32)
        s.inv_pose = wp.from_torch(self.inv_pose.view(-1, 8), dtype=wp.float32)
        s.enable = wp.from_torch(self.enable.view(-1), dtype=wp.uint8)
        s.features = wp.from_torch(self.features.view(-1), dtype=wp.float16)
        s.n_per_env = wp.from_torch(self.count.view(-1), dtype=wp.int32)
        s.n_voxels_per_layer = n_voxels_per_layer
        s.max_n = self.max_n
        s.num_envs = self.num_envs
        s.max_dist = self.max_esdf_distance
        return s


# =============================================================================
# Warp Struct
# =============================================================================


@wp.struct
class VoxelDataWarp:
    """Warp struct for voxel grid (ESDF) obstacle data.

    Attributes:
        params: Grid parameters [dims_x, dims_y, dims_z, voxel_size]. Shape: (num_envs * max_n, 4).
        dims: World bounding box [x, y, z, pad] in meters. Shape: (num_envs * max_n, 4).
        inv_pose: Inverse pose [x, y, z, qw, qx, qy, qz, pad]. Shape: (num_envs * max_n, 8).
        enable: Enable flag per grid. 1 = active, 0 = disabled.
        features: Flattened ESDF values for all grids (float16 for memory efficiency).
        n_per_env: Number of active grids per environment.
        n_voxels_per_layer: Number of voxels per grid layer.
        max_n: Maximum grids per environment.
        num_envs: Number of environments.
        max_dist: Maximum ESDF distance (unobserved value).
    """

    params: wp.array2d(dtype=wp.float32)
    dims: wp.array2d(dtype=wp.float32)
    inv_pose: wp.array2d(dtype=wp.float32)
    enable: wp.array(dtype=wp.uint8)
    features: wp.array(dtype=wp.float16)
    n_per_env: wp.array(dtype=wp.int32)
    n_voxels_per_layer: wp.int32
    max_n: wp.int32
    num_envs: wp.int32
    max_dist: wp.float32


# =============================================================================
# Voxel Grid Helper Functions
# =============================================================================


@wp.func
def world_to_voxel_idx(
    local_pt: wp.vec3,
    grid_dims: wp.vec3i,
    voxel_size: wp.float32,
) -> wp.vec3i:
    """Convert local point to voxel grid indices."""
    offset_x = wp.float32(grid_dims[0]) * voxel_size * 0.5
    offset_y = wp.float32(grid_dims[1]) * voxel_size * 0.5
    offset_z = wp.float32(grid_dims[2]) * voxel_size * 0.5

    ix = wp.int32((local_pt[0] + offset_x) / voxel_size)
    iy = wp.int32((local_pt[1] + offset_y) / voxel_size)
    iz = wp.int32((local_pt[2] + offset_z) / voxel_size)

    return wp.vec3i(ix, iy, iz)


@wp.func
def voxel_idx_to_flat(idx: wp.vec3i, grid_dims: wp.vec3i) -> wp.int32:
    """Convert 3D voxel index to flat array index.

    Uses C-order (row-major) indexing to match PyTorch tensor layout.
    For tensor shape (nx, ny, nz), z varies fastest in memory.

    Args:
        idx: Voxel index (x, y, z).
        grid_dims: Grid dimensions (nx, ny, nz).

    Returns:
        Flat index into the 1D feature array.
    """
    # C-order: flat = x * ny * nz + y * nz + z
    return idx[0] * grid_dims[1] * grid_dims[2] + idx[1] * grid_dims[2] + idx[2]


@wp.func
def is_voxel_valid(idx: wp.vec3i, grid_dims: wp.vec3i) -> wp.bool:
    """Check if voxel index is within grid bounds."""
    return (
        idx[0] >= 0
        and idx[0] < grid_dims[0]
        and idx[1] >= 0
        and idx[1] < grid_dims[1]
        and idx[2] >= 0
        and idx[2] < grid_dims[2]
    )


@wp.func
def sample_voxel_sdf(
    features: wp.array(dtype=wp.float16),
    layer_start_idx: wp.int32,
    idx: wp.vec3i,
    grid_dims: wp.vec3i,
    default_val: wp.float32,
) -> wp.vec2:
    """Sample SDF value from voxel grid at given index (nearest neighbor).

    Reads float16 from storage and converts to float32 for computation.

    Returns:
        vec2(sdf_value, valid_flag) where valid_flag is 1.0 if in-bounds, 0.0 otherwise.
    """
    if not is_voxel_valid(idx, grid_dims):
        return wp.vec2(default_val, 0.0)

    flat_idx = voxel_idx_to_flat(idx, grid_dims)
    access_idx = layer_start_idx + flat_idx

    # Bounds check on features array
    if access_idx < 0:# or access_idx >= features.shape[0]:
        assert False, "Access index is negative"
    if access_idx >= features.shape[0]:
        assert False, "Access index is out of bounds"
    #    return wp.vec2(default_val, 0.0)

    return wp.vec2(wp.float32(features[access_idx]), 1.0)



@wp.func
def sample_voxel_sdf_with_grad(
    features: wp.array(dtype=wp.float16),
    layer_start_idx: wp.int32,
    local_pt: wp.vec3,
    grid_dims: wp.vec3i,
    voxel_size: wp.float32,
    default_val: wp.float32,
) -> wp.vec4:
    """Sample SDF and compute analytical gradient using trilinear interpolation.

    Uses trilinear interpolation for smooth SDF values and computes the exact
    analytical gradient from the same 8 corner samples. Only valid voxels
    contribute to interpolation and gradient.

    The gradient is computed as bilinear interpolation of differences:
      grad_x = (1/voxel_size) * bilinear_yz(s1xx - s0xx)

    Args:
        features: Flat array of SDF values (float16).
        layer_start_idx: Start index in features for this voxel grid layer.
        local_pt: Query point in obstacle local frame.
        grid_dims: Voxel grid dimensions (nx, ny, nz).
        voxel_size: Size of each voxel in meters.
        default_val: Value to return for out-of-bounds queries. (Usually 100.0)

    Returns:
        vec4(sdf, grad_x, grad_y, grad_z). Gradient is zero if invalid.
    """
    # For grids too small for trilinear, return SDF with zero gradient
    if grid_dims[0] < 2 or grid_dims[1] < 2 or grid_dims[2] < 2:
        voxel_idx = world_to_voxel_idx(local_pt, grid_dims, voxel_size)
        result = sample_voxel_sdf(features, layer_start_idx, voxel_idx, grid_dims, default_val)
        return wp.vec4(result[0], 0.0, 0.0, 0.0)

    inv_voxel = 1.0 / voxel_size
    half_x = wp.float32(grid_dims[0]) * 0.5
    half_y = wp.float32(grid_dims[1]) * 0.5
    half_z = wp.float32(grid_dims[2]) * 0.5

    # Continuous voxel coordinates (can be fractional)
    # Subtract 0.5 to use align_corners=True convention (voxel centers at integers)
    # This matches PyTorch grid_sample with align_corners=True
    vx = local_pt[0] * inv_voxel + half_x - 0.5
    vy = local_pt[1] * inv_voxel + half_y - 0.5
    vz = local_pt[2] * inv_voxel + half_z - 0.5

    # Get integer base indices (floor)
    x0 = wp.int32(wp.floor(vx))
    y0 = wp.int32(wp.floor(vy))
    z0 = wp.int32(wp.floor(vz))
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Fractional parts for interpolation weights
    fx = vx - wp.float32(x0)
    fy = vy - wp.float32(y0)
    fz = vz - wp.float32(z0)
    fx1 = 1.0 - fx
    fy1 = 1.0 - fy
    fz1 = 1.0 - fz

    # Precompute strides for flat index calculation
    stride_x = grid_dims[1] * grid_dims[2]
    stride_y = grid_dims[2]

    # Factored bounds checks (6 checks instead of 48)
    x0_ok = x0 >= 0 and x0 < grid_dims[0]
    x1_ok = x1 >= 0 and x1 < grid_dims[0]
    y0_ok = y0 >= 0 and y0 < grid_dims[1]
    y1_ok = y1 >= 0 and y1 < grid_dims[1]
    z0_ok = z0 >= 0 and z0 < grid_dims[2]
    z1_ok = z1 >= 0 and z1 < grid_dims[2]

    # Check if all 8 corners are valid (fast path for interior points)
    all_valid = x0_ok and x1_ok and y0_ok and y1_ok and z0_ok and z1_ok

    # Compute base flat index and offsets
    base_idx = layer_start_idx + x0 * stride_x + y0 * stride_y + z0

    if all_valid:
        # Fast path: all corners valid, no validity weighting needed
        s000 = wp.float32(features[base_idx])
        s001 = wp.float32(features[base_idx + 1])
        s010 = wp.float32(features[base_idx + stride_y])
        s011 = wp.float32(features[base_idx + stride_y + 1])
        s100 = wp.float32(features[base_idx + stride_x])
        s101 = wp.float32(features[base_idx + stride_x + 1])
        s110 = wp.float32(features[base_idx + stride_x + stride_y])
        s111 = wp.float32(features[base_idx + stride_x + stride_y + 1])

        # Trilinear interpolation (weights sum to 1.0, no normalization needed)
        sdf = (
            s000 * fx1 * fy1 * fz1
            + s001 * fx1 * fy1 * fz
            + s010 * fx1 * fy * fz1
            + s011 * fx1 * fy * fz
            + s100 * fx * fy1 * fz1
            + s101 * fx * fy1 * fz
            + s110 * fx * fy * fz1
            + s111 * fx * fy * fz
        )

        # Analytical gradient (bilinear interpolation of differences)
        # No validity checks or weight normalization needed
        grad_x = (
            (s100 - s000) * fy1 * fz1
            + (s101 - s001) * fy1 * fz
            + (s110 - s010) * fy * fz1
            + (s111 - s011) * fy * fz
        ) * inv_voxel

        grad_y = (
            (s010 - s000) * fx1 * fz1
            + (s011 - s001) * fx1 * fz
            + (s110 - s100) * fx * fz1
            + (s111 - s101) * fx * fz
        ) * inv_voxel

        grad_z = (
            (s001 - s000) * fx1 * fy1
            + (s011 - s010) * fx1 * fy
            + (s101 - s100) * fx * fy1
            + (s111 - s110) * fx * fy
        ) * inv_voxel

        return wp.vec4(sdf, grad_x, grad_y, grad_z)

    # Slow path: boundary case with validity weighting
    # Sample all 8 corners with validity flags
    s000 = wp.float32(default_val)
    s001 = wp.float32(default_val)
    s010 = wp.float32(default_val)
    s011 = wp.float32(default_val)
    s100 = wp.float32(default_val)
    s101 = wp.float32(default_val)
    s110 = wp.float32(default_val)
    s111 = wp.float32(default_val)
    v000 = wp.float32(0.0)
    v001 = wp.float32(0.0)
    v010 = wp.float32(0.0)
    v011 = wp.float32(0.0)
    v100 = wp.float32(0.0)
    v101 = wp.float32(0.0)
    v110 = wp.float32(0.0)
    v111 = wp.float32(0.0)

    # Sample each corner using precomputed validity and offsets
    if x0_ok and y0_ok and z0_ok:
        s000 = wp.float32(features[base_idx])
        v000 = 1.0

    if x0_ok and y0_ok and z1_ok:
        s001 = wp.float32(features[base_idx + 1])
        v001 = 1.0

    if x0_ok and y1_ok and z0_ok:
        s010 = wp.float32(features[base_idx + stride_y])
        v010 = 1.0

    if x0_ok and y1_ok and z1_ok:
        s011 = wp.float32(features[base_idx + stride_y + 1])
        v011 = 1.0

    if x1_ok and y0_ok and z0_ok:
        s100 = wp.float32(features[base_idx + stride_x])
        v100 = 1.0

    if x1_ok and y0_ok and z1_ok:
        s101 = wp.float32(features[base_idx + stride_x + 1])
        v101 = 1.0

    if x1_ok and y1_ok and z0_ok:
        s110 = wp.float32(features[base_idx + stride_x + stride_y])
        v110 = 1.0

    if x1_ok and y1_ok and z1_ok:
        s111 = wp.float32(features[base_idx + stride_x + stride_y + 1])
        v111 = 1.0

    # Compute weighted SDF (trilinear interpolation)
    w000 = fx1 * fy1 * fz1
    w001 = fx1 * fy1 * fz
    w010 = fx1 * fy * fz1
    w011 = fx1 * fy * fz
    w100 = fx * fy1 * fz1
    w101 = fx * fy1 * fz
    w110 = fx * fy * fz1
    w111 = fx * fy * fz

    weighted_sum = (
        s000 * w000 * v000
        + s001 * w001 * v001
        + s010 * w010 * v010
        + s011 * w011 * v011
        + s100 * w100 * v100
        + s101 * w101 * v101
        + s110 * w110 * v110
        + s111 * w111 * v111
    )
    weight_sum = (
        w000 * v000 + w001 * v001 + w010 * v010 + w011 * v011 + w100 * v100 + w101 * v101 + w110 * v110 + w111 * v111
    )

    if weight_sum <= 0.0:
        return wp.vec4(default_val, 0.0, 0.0, 0.0)

    sdf = weighted_sum / weight_sum

    # Compute analytical gradient using bilinear interpolation of differences
    # grad_x = (1/v) * Σ (s1yz - s0yz) * bilinear_weight_yz * valid_pair
    # A pair is valid only if both corners are valid

    # X gradient: bilinear in yz of (s1xx - s0xx)
    gx_sum = wp.float32(0.0)
    gx_weight = wp.float32(0.0)
    if v000 > 0.0 and v100 > 0.0:
        w = fy1 * fz1
        gx_sum += (s100 - s000) * w
        gx_weight += w
    if v001 > 0.0 and v101 > 0.0:
        w = fy1 * fz
        gx_sum += (s101 - s001) * w
        gx_weight += w
    if v010 > 0.0 and v110 > 0.0:
        w = fy * fz1
        gx_sum += (s110 - s010) * w
        gx_weight += w
    if v011 > 0.0 and v111 > 0.0:
        w = fy * fz
        gx_sum += (s111 - s011) * w
        gx_weight += w

    # Y gradient: bilinear in xz of (sx1z - sx0z)
    gy_sum = wp.float32(0.0)
    gy_weight = wp.float32(0.0)
    if v000 > 0.0 and v010 > 0.0:
        w = fx1 * fz1
        gy_sum += (s010 - s000) * w
        gy_weight += w
    if v001 > 0.0 and v011 > 0.0:
        w = fx1 * fz
        gy_sum += (s011 - s001) * w
        gy_weight += w
    if v100 > 0.0 and v110 > 0.0:
        w = fx * fz1
        gy_sum += (s110 - s100) * w
        gy_weight += w
    if v101 > 0.0 and v111 > 0.0:
        w = fx * fz
        gy_sum += (s111 - s101) * w
        gy_weight += w

    # Z gradient: bilinear in xy of (sxy1 - sxy0)
    gz_sum = wp.float32(0.0)
    gz_weight = wp.float32(0.0)
    if v000 > 0.0 and v001 > 0.0:
        w = fx1 * fy1
        gz_sum += (s001 - s000) * w
        gz_weight += w
    if v010 > 0.0 and v011 > 0.0:
        w = fx1 * fy
        gz_sum += (s011 - s010) * w
        gz_weight += w
    if v100 > 0.0 and v101 > 0.0:
        w = fx * fy1
        gz_sum += (s101 - s100) * w
        gz_weight += w
    if v110 > 0.0 and v111 > 0.0:
        w = fx * fy
        gz_sum += (s111 - s110) * w
        gz_weight += w

    # Normalize gradients (divide by voxel_size to get world-space gradient)
    grad_x = gx_sum / gx_weight * inv_voxel if gx_weight > 0.0 else 0.0
    grad_y = gy_sum / gy_weight * inv_voxel if gy_weight > 0.0 else 0.0
    grad_z = gz_sum / gz_weight * inv_voxel if gz_weight > 0.0 else 0.0

    return wp.vec4(sdf, grad_x, grad_y, grad_z)


# =============================================================================
# SDF Implementation (Generic Kernel Overloads)
# =============================================================================

_SDF_EPS = 1e-6


def is_obs_enabled(
    obs_set: VoxelDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
) -> wp.bool:
    """Check if a voxel grid obstacle is enabled.

    Args:
        obs_set: Warp struct for voxel data.
        env_idx: Environment index.
        local_idx: Local index of the grid within the environment.

    Returns:
        True if the voxel grid is enabled, False otherwise.
    """
    if local_idx >= obs_set.n_per_env[env_idx]:
        return False
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return obs_set.enable[flat_idx] == wp.uint8(1)


def load_obstacle_transform(
    obs_set: VoxelDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
) -> wp.transform:
    """Load world-to-local transform for a voxel grid obstacle.

    The returned transform converts world-frame points to local-frame.
    Use wp.transform_point(t, world_pt) to transform points.
    Use wp.transform_vector(wp.transform_inverse(t), grad_local) for gradients.

    Args:
        obs_set: Warp struct for voxel data.
        env_idx: Environment index.
        local_idx: Local index of the grid within the environment.

    Returns:
        wp.transform for world-to-local transformation.
    """
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return load_transform_from_inv_pose(obs_set.inv_pose, flat_idx)


def compute_local_sdf(
    obs_set: VoxelDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.float32:
    """Compute SDF value for a voxel grid obstacle using trilinear interpolation.

    The query point must already be in the obstacle's local frame.
    Use load_obstacle_transform() and wp.transform_point() to transform
    world-frame points before calling this function.

    Uses trilinear interpolation for smooth SDF values between voxel centers.

    Args:
        obs_set: Warp struct for voxel data.
        env_idx: Environment index.
        local_idx: Local index of the grid within the environment.
        local_pt: Query point in obstacle local frame.

    Returns:
        ESDF value (negative inside, positive outside).
    """
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)

    dims_x = wp.int32(obs_set.params[flat_idx, 0])
    dims_y = wp.int32(obs_set.params[flat_idx, 1])
    dims_z = wp.int32(obs_set.params[flat_idx, 2])
    voxel_size = obs_set.params[flat_idx, 3]
    grid_dims = wp.vec3i(dims_x, dims_y, dims_z)

    layer_start_idx = flat_idx * obs_set.n_voxels_per_layer

    result = sample_voxel_sdf_with_grad(
        obs_set.features, layer_start_idx, local_pt, grid_dims, voxel_size, obs_set.max_dist
    )

    return result[0]


def compute_local_sdf_with_grad(
    obs_set: VoxelDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.vec4:
    """Compute SDF and gradient for a voxel grid obstacle in local frame.

    The query point must already be in the obstacle's local frame.
    The returned gradient is also in the local frame. Use
    wp.transform_vector(wp.transform_inverse(inv_t), grad_local) to
    convert to world frame.

    Uses trilinear interpolation for both SDF value and analytical gradient
    computation from the same 8 corner samples.

    Args:
        obs_set: Warp struct for voxel data.
        env_idx: Environment index.
        local_idx: Local index of the grid within the environment.
        local_pt: Query point in obstacle local frame.

    Returns:
        vec4(sdf, grad_local_x, grad_local_y, grad_local_z).
    """
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)

    dims_x = wp.int32(obs_set.params[flat_idx, 0])
    dims_y = wp.int32(obs_set.params[flat_idx, 1])
    dims_z = wp.int32(obs_set.params[flat_idx, 2])
    voxel_size = obs_set.params[flat_idx, 3]
    grid_dims = wp.vec3i(dims_x, dims_y, dims_z)

    layer_start_idx = flat_idx * obs_set.n_voxels_per_layer

    result = sample_voxel_sdf_with_grad(
        obs_set.features, layer_start_idx, local_pt, grid_dims, voxel_size, obs_set.max_dist
    )
    sdf = result[0]

    if sdf >= obs_set.max_dist:
        return wp.vec4(obs_set.max_dist, 0.0, 0.0, 0.0)

    grad_local = wp.vec3(-result[1], -result[2], -result[3])
    grad_len = wp.length(grad_local)
    if grad_len > _SDF_EPS:
        grad_local = grad_local / grad_len
    else:
        grad_local = wp.vec3(0.0, 0.0, 0.0)

    return wp.vec4(sdf, grad_local[0], grad_local[1], grad_local[2])
