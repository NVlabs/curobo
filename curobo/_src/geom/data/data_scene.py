# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Aggregate GPU data storage for all scene obstacle types.

This module provides:
- SceneData: Container holding CuboidData, MeshData, and VoxelData
- SceneDataWarp: Warp struct for kernel access to all obstacle types
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import List, Optional, Union

# Third Party
import warp as wp

# CuRobo
from curobo._src.geom.data.data_cuboid import CuboidData, CuboidDataWarp
from curobo._src.geom.data.data_mesh import MeshData, MeshDataWarp
from curobo._src.geom.data.data_voxel import VoxelData, VoxelDataWarp
from curobo._src.geom.types import Cuboid, Mesh, Obstacle, SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise

# =============================================================================
# Python Dataclass
# =============================================================================


@dataclass
class SceneData:
    """Aggregate GPU tensor storage for all scene obstacle types.

    This dataclass holds references to individual data storage for each obstacle
    type (cuboids, meshes, voxels). It provides a unified interface for managing
    scene obstacles while delegating to the appropriate type-specific storage.

    Use this as the primary container for scene data. It can be passed to
    collision checkers, TSDF integrators, or visualizers.

    Example:
        >>> scene = SceneData.from_scene_cfg(scene_cfg, device_cfg)
        >>> scene.add_obstacle(Cuboid(name="box", pose=[0,0,0,1,0,0,0], dims=[0.1,0.1,0.1]))
        >>> scene.update_obstacle_pose("box", new_pose)
    """

    #: Cuboid (OBB) tensor storage. None if no cuboids configured.
    cuboids: Optional[CuboidData] = None

    #: Mesh tensor storage with Warp BVH. None if no meshes configured.
    meshes: Optional[MeshData] = None

    #: Voxel grid (ESDF) tensor storage. None if no voxels configured.
    voxels: Optional[VoxelData] = None

    #: Number of parallel environments.
    num_envs: int = 1

    #: Device configuration for tensor creation.
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)

    #: CPU reference to scene configuration (for reference updates).
    scene_model: Optional[Union[SceneCfg, List[SceneCfg]]] = None

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------
    def get_valid_data(self) -> List[Union[CuboidData, MeshData, VoxelData]]:
        valid_data = []
        if self.cuboids is not None:
            valid_data.append(self.cuboids)
        if self.meshes is not None:
            valid_data.append(self.meshes)
        if self.voxels is not None:
            valid_data.append(self.voxels)
        return valid_data

    @classmethod
    def create_cache(
        cls,
        num_envs: int,
        device_cfg: DeviceCfg,
        cuboid_cache: Optional[int] = None,
        mesh_cache: Optional[int] = None,
        mesh_max_dist: float = 0.1,
        voxel_cache: Optional[dict] = None,
    ) -> SceneData:
        """Create an empty cache for scene obstacles.

        Args:
            num_envs: Number of parallel environments.
            device_cfg: Device and dtype configuration.
            cuboid_cache: Maximum number of cuboids per environment. None to skip.
            mesh_cache: Maximum number of meshes per environment. None to skip.
            mesh_max_dist: Maximum query distance for mesh SDF.
            voxel_cache: Voxel cache configuration dict with keys:
                - layers: Maximum number of voxel grids
                - dims: Grid dimensions [x, y, z] in meters
                - voxel_size: Size of each voxel in meters

        Returns:
            Empty SceneData with allocated GPU buffers.
        """
        cuboids = None
        meshes = None
        voxels = None

        if cuboid_cache is not None and cuboid_cache > 0:
            cuboids = CuboidData.create_cache(cuboid_cache, num_envs, device_cfg)

        if mesh_cache is not None and mesh_cache > 0:
            meshes = MeshData.create_cache(mesh_cache, num_envs, device_cfg, mesh_max_dist)

        if voxel_cache is not None:
            voxels = VoxelData.create_cache(
                max_n=voxel_cache.get("layers", 1),
                num_envs=num_envs,
                device_cfg=device_cfg,
                grid_dims=voxel_cache.get("dims", [1.0, 1.0, 1.0]),
                voxel_size=voxel_cache.get("voxel_size", 0.02),
            )

        return cls(
            cuboids=cuboids,
            meshes=meshes,
            voxels=voxels,
            num_envs=num_envs,
            device_cfg=device_cfg,
        )

    @classmethod
    def from_scene_cfg(
        cls,
        scene_cfg: SceneCfg,
        device_cfg: DeviceCfg,
        num_envs: int = 1,
        env_idx: int = 0,
        cuboid_cache: Optional[int] = None,
        mesh_cache: Optional[int] = None,
        mesh_max_dist: float = 0.1,
        voxel_cache: Optional[dict] = None,
    ) -> SceneData:
        """Create SceneData from a scene configuration.

        Args:
            scene_cfg: Scene configuration containing obstacles.
            device_cfg: Device and dtype configuration.
            num_envs: Number of parallel environments.
            env_idx: Environment index to load the obstacles into.
            cuboid_cache: Max cuboids. If None, uses len(scene_cfg.cuboid).
            mesh_cache: Max meshes. If None, uses len(scene_cfg.mesh).
            mesh_max_dist: Maximum query distance for mesh SDF.
            voxel_cache: Voxel cache config. If None, derives from scene_cfg.voxel.

        Returns:
            SceneData populated with obstacles from the scene config.
        """
        cuboids = None
        meshes = None
        voxels = None

        # Handle cuboids
        if scene_cfg.cuboid and len(scene_cfg.cuboid) > 0:
            max_cuboids = cuboid_cache or len(scene_cfg.cuboid)
            cuboids = CuboidData.from_scene_cfg(
                scene_cfg, device_cfg, env_idx, num_envs, max_cuboids
            )
        elif cuboid_cache is not None and cuboid_cache > 0:
            cuboids = CuboidData.create_cache(cuboid_cache, num_envs, device_cfg)

        # Handle meshes
        if scene_cfg.mesh and len(scene_cfg.mesh) > 0:
            max_meshes = mesh_cache or len(scene_cfg.mesh)
            meshes = MeshData.from_scene_cfg(
                scene_cfg, device_cfg, env_idx, num_envs, max_meshes, mesh_max_dist
            )
        elif mesh_cache is not None and mesh_cache > 0:
            meshes = MeshData.create_cache(mesh_cache, num_envs, device_cfg, mesh_max_dist)

        # Handle voxels
        if scene_cfg.voxel and len(scene_cfg.voxel) > 0:
            voxels = VoxelData.from_scene_cfg(scene_cfg, device_cfg, env_idx, num_envs)
        elif voxel_cache is not None:
            voxels = VoxelData.create_cache(
                max_n=voxel_cache.get("layers", 1),
                num_envs=num_envs,
                device_cfg=device_cfg,
                grid_dims=voxel_cache.get("dims", [1.0, 1.0, 1.0]),
                voxel_size=voxel_cache.get("voxel_size", 0.02),
            )

        return cls(
            cuboids=cuboids,
            meshes=meshes,
            voxels=voxels,
            num_envs=num_envs,
            device_cfg=device_cfg,
            scene_model=scene_cfg,
        )

    @classmethod
    def from_batch_scene_cfg(
        cls,
        scene_cfg_list: List[SceneCfg],
        device_cfg: DeviceCfg,
        cuboid_cache: Optional[int] = None,
        mesh_cache: Optional[int] = None,
        mesh_max_dist: float = 0.1,
        voxel_cache: Optional[dict] = None,
    ) -> SceneData:
        """Create SceneData from a list of scene configurations.

        Each scene config is loaded into a separate environment.

        Args:
            scene_cfg_list: List of scene configurations, one per environment.
            device_cfg: Device and dtype configuration.
            cuboid_cache: Max cuboids per env. If None, uses max across scenes.
            mesh_cache: Max meshes per env. If None, uses max across scenes.
            mesh_max_dist: Maximum query distance for mesh SDF.
            voxel_cache: Voxel cache config.

        Returns:
            SceneData populated with obstacles from all scene configs.
        """
        num_envs = len(scene_cfg_list)
        cuboids = None
        meshes = None
        voxels = None

        # Check if any scene has cuboids
        cuboid_counts = [len(cfg.cuboid) if cfg.cuboid else 0 for cfg in scene_cfg_list]
        if max(cuboid_counts) > 0 or (cuboid_cache is not None and cuboid_cache > 0):
            cuboids = CuboidData.from_batch_scene_cfg(scene_cfg_list, device_cfg, cuboid_cache)

        # Check if any scene has meshes
        mesh_counts = [len(cfg.mesh) if cfg.mesh else 0 for cfg in scene_cfg_list]
        if max(mesh_counts) > 0 or (mesh_cache is not None and mesh_cache > 0):
            meshes = MeshData.from_batch_scene_cfg(
                scene_cfg_list, device_cfg, mesh_cache, mesh_max_dist
            )

        # Check if any scene has voxels
        voxel_counts = [len(cfg.voxel) if cfg.voxel else 0 for cfg in scene_cfg_list]
        if max(voxel_counts) > 0:
            voxels = VoxelData.from_batch_scene_cfg(scene_cfg_list, device_cfg)
        elif voxel_cache is not None:
            voxels = VoxelData.create_cache(
                max_n=voxel_cache.get("layers", 1),
                num_envs=num_envs,
                device_cfg=device_cfg,
                grid_dims=voxel_cache.get("dims", [1.0, 1.0, 1.0]),
                voxel_size=voxel_cache.get("voxel_size", 0.02),
            )

        return cls(
            cuboids=cuboids,
            meshes=meshes,
            voxels=voxels,
            num_envs=num_envs,
            device_cfg=device_cfg,
            scene_model=scene_cfg_list,
        )

    # -------------------------------------------------------------------------
    # Obstacle Management Methods
    # -------------------------------------------------------------------------

    def add_obstacle(self, obstacle: Obstacle, env_idx: int = 0) -> int:
        """Add an obstacle to the scene.

        Automatically routes to the appropriate storage based on obstacle type.

        Args:
            obstacle: Obstacle to add (Cuboid, Mesh, or VoxelGrid).
            env_idx: Environment index to add to.

        Returns:
            Index of the added obstacle within its type.

        Raises:
            ValueError: If obstacle type is not supported or cache not initialized.
        """
        if isinstance(obstacle, Cuboid):
            if self.cuboids is None:
                log_and_raise("Cuboid cache not initialized")
            return self.cuboids.add(obstacle, env_idx)
        elif isinstance(obstacle, Mesh):
            if self.meshes is None:
                log_and_raise("Mesh cache not initialized")
            return self.meshes.add(obstacle, env_idx)
        elif isinstance(obstacle, VoxelGrid):
            if self.voxels is None:
                log_and_raise("Voxel cache not initialized")
            self.voxels.load_batch([obstacle], env_idx)
            return 0
        else:
            log_and_raise(f"Unsupported obstacle type: {type(obstacle)}")

    def update_obstacle_pose(
        self,
        name: str,
        pose: Pose,
        env_idx: int = 0,
    ) -> None:
        """Update the pose of an existing obstacle by name.

        Searches across all obstacle types to find the obstacle.

        Args:
            name: Name of the obstacle to update.
            pose: New pose in world frame.
            env_idx: Environment index.

        Raises:
            ValueError: If obstacle not found in any type.
        """
        if self.cuboids is not None:
            try:
                self.cuboids.update_pose(name, w_obj_pose=pose, env_idx=env_idx)
                return
            except RuntimeError:
                pass

        if self.meshes is not None:
            try:
                self.meshes.update_pose(name, w_obj_pose=pose, env_idx=env_idx)
                return
            except RuntimeError:
                pass

        if self.voxels is not None:
            try:
                self.voxels.update_pose(name, w_obj_pose=pose, env_idx=env_idx)
                return
            except RuntimeError:
                pass

        log_and_raise(f"Obstacle '{name}' not found in environment {env_idx}")

    def enable_obstacle(
        self,
        name: str,
        enabled: bool = True,
        env_idx: int = 0,
    ) -> None:
        """Enable or disable an obstacle for collision checking.

        Args:
            name: Name of the obstacle.
            enabled: True to enable, False to disable.
            env_idx: Environment index.

        Raises:
            ValueError: If obstacle not found in any type.
        """
        if self.cuboids is not None:
            try:
                self.cuboids.set_enabled(name, enabled, env_idx)
                return
            except RuntimeError:
                pass

        if self.meshes is not None:
            try:
                self.meshes.set_enabled(name, enabled, env_idx)
                return
            except RuntimeError:
                pass

        if self.voxels is not None:
            try:
                self.voxels.set_enabled(name, enabled, env_idx)
                return
            except RuntimeError:
                pass

        log_and_raise(f"Obstacle '{name}' not found in environment {env_idx}")

    def get_obstacle_names(self, env_idx: int = 0) -> List[str]:
        """Get names of all obstacles in an environment.

        Args:
            env_idx: Environment index.

        Returns:
            List of obstacle names across all types.
        """
        names = []
        if self.cuboids is not None:
            names.extend(self.cuboids.get_names(env_idx))
        if self.meshes is not None:
            names.extend(self.meshes.get_names(env_idx))
        if self.voxels is not None:
            names.extend(self.voxels.get_names(env_idx))
        return names

    def check_obstacle_exists(self, name: str, env_idx: int = 0) -> bool:
        """Check if an obstacle exists in the scene."""
        return name in self.get_obstacle_names(env_idx)

    def clear(self, env_idx: Optional[int] = None) -> None:
        """Clear all obstacles from one or all environments.

        Args:
            env_idx: Environment to clear. If None, clears all environments.
        """
        if self.cuboids is not None:
            self.cuboids.clear(env_idx)
        if self.meshes is not None:
            self.meshes.clear(env_idx)
        if self.voxels is not None:
            self.voxels.clear(env_idx)

    def load_from_scene_cfg(
        self,
        scene_cfg: SceneCfg,
        env_idx: int = 0,
        store_reference: bool = True,
    ) -> None:
        """Load obstacles from a scene configuration into existing storage.

        Args:
            scene_cfg: Scene configuration containing obstacles to load.
            env_idx: Environment index to load obstacles into.
            store_reference: If True, stores scene_cfg in scene_model.
        """
        if store_reference:
            self.scene_model = scene_cfg
        self.clear(env_idx)
        if scene_cfg.cuboid:
            for cuboid in scene_cfg.cuboid:
                self.add_obstacle(cuboid, env_idx)
        if scene_cfg.mesh:
            for mesh in scene_cfg.mesh:
                self.add_obstacle(mesh, env_idx)
        if scene_cfg.voxel:
            for voxel in scene_cfg.voxel:
                self.add_obstacle(voxel, env_idx)

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def has_cuboids(self) -> bool:
        """Check if cuboid data is available."""
        return self.cuboids is not None

    def has_meshes(self) -> bool:
        """Check if mesh data is available."""
        return self.meshes is not None

    def has_voxels(self) -> bool:
        """Check if voxel data is available."""
        return self.voxels is not None

    def get_active_types(self) -> dict:
        """Get a dictionary of active obstacle types.

        Returns:
            Dict with keys 'cuboid', 'mesh', 'voxel' and boolean values.
        """
        return {
            "cuboid": self.has_cuboids(),
            "mesh": self.has_meshes(),
            "voxel": self.has_voxels(),
        }

    # -------------------------------------------------------------------------
    # Warp Conversion
    # -------------------------------------------------------------------------

    def to_warp(self, mesh_max_dist: Optional[float] = None) -> SceneDataWarp:
        """Convert to Warp struct for kernel launches.

        Args:
            mesh_max_dist: Override max query distance for meshes.

        Returns:
            SceneDataWarp struct for use in Warp kernels.
        """
        s = SceneDataWarp()

        if self.cuboids is not None:
            s.cuboids = self.cuboids.to_warp()
            s.has_cuboids = True
        else:
            s.has_cuboids = False

        if self.meshes is not None:
            s.meshes = self.meshes.to_warp(mesh_max_dist)
            s.has_meshes = True
        else:
            s.has_meshes = False

        if self.voxels is not None:
            s.voxels = self.voxels.to_warp()
            s.has_voxels = True
        else:
            s.has_voxels = False

        s.num_envs = self.num_envs
        return s


# =============================================================================
# Warp Struct
# =============================================================================


@wp.struct
class SceneDataWarp:
    """Warp struct for aggregate scene obstacle data.

    Attributes:
        cuboids: Cuboid data (valid if has_cuboids is True).
        meshes: Mesh data (valid if has_meshes is True).
        voxels: Voxel data (valid if has_voxels is True).
        has_cuboids: Whether cuboid data is present.
        has_meshes: Whether mesh data is present.
        has_voxels: Whether voxel data is present.
        num_envs: Number of environments.
    """

    cuboids: CuboidDataWarp
    meshes: MeshDataWarp
    voxels: VoxelDataWarp
    has_cuboids: wp.bool
    has_meshes: wp.bool
    has_voxels: wp.bool
    num_envs: wp.int32
