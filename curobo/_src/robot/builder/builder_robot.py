# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""High-level API for building and editing robot model configurations from URDF files.

This module provides :class:`RobotBuilder` for creating cuRobo robot configurations
with collision spheres and optimized self-collision matrices.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import tqdm
import trimesh

from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo._src.geom.sphere_fit.types import SphereFitMetrics
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg
from curobo._src.robot.parser.parser_urdf import UrdfRobotParser
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise, log_info, log_warn
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.viser_visualizer import ViserVisualizer
from curobo._src.util.xrdf_util import convert_curobo_to_xrdf
from curobo._src.util_file import join_path, load_yaml, write_yaml
from curobo.content import get_assets_path, get_robot_configs_path


class RobotBuilder:
    """Build and edit robot model configurations from URDF files.

    This class provides a high-level interface for creating cuRobo robot configurations.
    It handles:

    - Loading URDF files and parsing kinematic structure
    - Fitting collision spheres to robot link meshes
    - Computing optimized self-collision ignore matrices
    - Generating and saving cuRobo-compatible configuration files
    - Interactive visualization with Viser

    The builder supports both creating new configurations from URDF and editing existing
    configurations (e.g., refitting spheres for specific links).

    Typical workflow for creating new robot:
        1. Initialize with URDF path
        2. Fit collision spheres
        3. Compute collision matrix
        4. Build configuration
        5. Save to file

    Typical workflow for editing existing config:
        1. Load with :meth:`from_config`
        2. Modify (e.g., :meth:`refit_link_spheres`)
        3. Build configuration
        4. Save to file

    Example:
        Creating new robot model::

            >>> from curobo._src.robot.builder.builder_robot import RobotBuilder
            >>>
            >>> # Create builder
            >>> builder = RobotBuilder("robot.urdf", "assets/")
            >>>
            >>> # Fit collision spheres to all links
            >>> builder.fit_collision_spheres(sphere_density=2.0)
            >>>
            >>> # Compute and optimize collision matrix
            >>> builder.compute_collision_matrix(num_samples=2000)
            >>>
            >>> # Build final configuration
            >>> config = builder.build()
            >>>
            >>> # Save to file
            >>> builder.save(config, "my_robot.yml")

        Editing existing robot model::

            >>> # Load existing config
            >>> builder = RobotBuilder.from_config("franka.yml")
            >>>
            >>> # Refit spheres for gripper with higher density
            >>> builder.refit_link_spheres("panda_hand", sphere_density=3.0)
            >>>
            >>> # Add manual collision ignore
            >>> builder.add_collision_ignore("panda_link1", ["panda_link5"])
            >>>
            >>> # Rebuild and save
            >>> config = builder.build()
            >>> builder.save(config, "franka_updated.yml")
    """

    def __init__(
        self,
        urdf_path: str,
        asset_path: str = "",
        tool_frames: Optional[List[str]] = None,
        device_cfg: Optional[DeviceCfg] = None,
    ):
        """Initialize robot model builder from URDF file.

        Args:
            urdf_path: Path to URDF file.
            asset_path: Path to mesh assets directory (for resolving relative mesh paths in URDF).
            base_link: Robot base link name. If None, auto-detected from URDF root.
            device_cfg: Device configuration for CUDA operations. Defaults to CUDA:0.
        """
        self.device_cfg = device_cfg or DeviceCfg()

        # Resolve to absolute paths for URDF parsing
        abs_urdf = str(Path(urdf_path).resolve())
        abs_asset = str(Path(asset_path).resolve()) if asset_path else ""

        # Store paths relative to assets dir so KinematicsLoaderCfg doesn't double-prefix
        assets_root = str(get_assets_path())
        if abs_urdf.startswith(assets_root + "/"):
            self.urdf_path = abs_urdf[len(assets_root) + 1 :]
        else:
            self.urdf_path = abs_urdf
        if abs_asset and abs_asset.startswith(assets_root + "/"):
            self.asset_path = abs_asset[len(assets_root) + 1 :]
        else:
            self.asset_path = abs_asset

        # Parse URDF
        self._parser = UrdfRobotParser(
            abs_urdf,
            load_meshes=True,
            mesh_root=abs_asset,
            build_scene_graph=True,
        )

        # Extract robot structure
        self._link_names = self._parser.get_link_names_from_urdf()

        # store links that have meshes seperately:
        self._mesh_link_names = [
            link_name
            for link_name in self._link_names
            if len(self._parser.get_link_geometry(link_name)) > 0
        ]
        self._joint_names = [
            j
            for j in self._parser.get_joint_names_from_urdf()
            if j not in self._link_names
        ]
        self._base_link = self._parser.root_link
        self._tool_frames = tool_frames or self._link_names[-1:]

        # Build link parent relationships
        self._parser.build_link_parent()

        # Storage for computed data
        self._collision_spheres: Optional[Dict[str, List[Dict]]] = None
        self._self_collision_ignore: Optional[Dict[str, List[str]]] = None
        self._self_collision_buffer: Dict[str, float] = {}
        self._cspace_config: Optional[Dict] = None
        self._link_metrics: Dict[str, SphereFitMetrics] = {}

    @classmethod
    def from_config(
        cls,
        config_path: str,
        device_cfg: Optional[DeviceCfg] = None,
    ) -> "RobotBuilder":
        """Load existing robot configuration for editing.

        This classmethod allows loading an existing .yml configuration file and
        creating a builder instance that can be used to modify the configuration
        (e.g., refit spheres, update collision ignore matrix).

        Args:
            config_path: Path to existing robot configuration .yml file.
            device_cfg: Device configuration. Defaults to CUDA:0.

        Returns:
            RobotBuilder instance initialized with existing configuration.

        Raises:
            FileNotFoundError: If config_path doesn't exist.
            KeyError: If config file is missing required fields.

        Example:
            >>> builder = RobotBuilder.from_config("franka.yml")
            >>> builder.refit_link_spheres("panda_hand", sphere_density=3.0)
            >>> config = builder.build()
            >>> builder.save(config, "franka_updated.yml")
        """
        # Load YAML config
        config_data = load_yaml(config_path)
        if "robot_cfg" in config_data:
            config_data = config_data["robot_cfg"]

        kinematics_data = config_data["kinematics"]

        # Create instance from existing config
        urdf_path = join_path(get_assets_path(), kinematics_data["urdf_path"])
        asset_path = join_path(get_assets_path(), kinematics_data.get("asset_root_path", ""))

        tool_frames = kinematics_data.get("tool_frames", None)
        instance = cls(urdf_path=urdf_path, asset_path=asset_path, tool_frames=tool_frames, device_cfg=device_cfg)

        # Load existing collision data
        if "collision_spheres" in kinematics_data:
            if isinstance(kinematics_data["collision_spheres"], str):
                kinematics_data["collision_spheres"] = load_yaml(join_path(get_robot_configs_path(), kinematics_data["collision_spheres"]))
            instance._collision_spheres = kinematics_data["collision_spheres"].copy()

        if "self_collision_ignore" in kinematics_data:
            instance._self_collision_ignore = kinematics_data[
                "self_collision_ignore"
            ].copy()

        if "self_collision_buffer" in kinematics_data:
            instance._self_collision_buffer = kinematics_data[
                "self_collision_buffer"
            ].copy()

        if "cspace" in kinematics_data:
            instance._cspace_config = kinematics_data["cspace"].copy()

        log_info(f"Loaded robot configuration from {config_path}")

        return instance

    # ==========================================================================
    # COLLISION SPHERE FITTING
    # ==========================================================================

    @staticmethod
    def _resolve_clip_plane(axis: str, offset: float) -> tuple:
        """Convert an axis letter and offset to a ``((nx, ny, nz), offset)`` tuple."""
        axis_map = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
        key = axis.lower().lstrip("-")
        if key not in axis_map:
            raise ValueError(f"Invalid clip axis '{axis}', expected one of x, y, z")
        normal = axis_map[key]
        if axis.startswith("-"):
            normal = tuple(-v for v in normal)
        return (normal, offset)

    def fit_collision_spheres(
        self,
        sphere_density: float = 1.0,
        surface_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.MORPHIT,
        use_collision_mesh: bool = False,
        iterations: int = 200,
        coverage_weight: Optional[float] = None,
        protrusion_weight: Optional[float] = None,
        compute_metrics: bool = False,
        clip_links: Optional[Dict[str, Tuple[str, float]]] = None,
    ) -> Dict[str, List[Dict]]:
        """Fit collision spheres to all robot links with collision meshes.

        Loads each link's mesh, preprocesses it, and fits spheres to approximate
        its volume.  The number of spheres per link scales automatically with
        the link's bounding-box volume and *sphere_density*.

        Args:
            sphere_density: Dimensionless density multiplier.  ``1.0`` (default)
                gives a balanced count; ``2.0`` doubles it; ``0.5`` halves it.
                Practical range: ``0.1`` – ``10.0``.
            surface_radius: Radius added to surface-sampled spheres.  Only
                affects the ``SURFACE`` fit type and the surface-sampling
                fallback.
            fit_type: Sphere fitting algorithm.  ``MORPHIT`` (default) usually
                gives the best results.
            use_collision_mesh: When True, use collision geometry instead of
                visual geometry.
            iterations: Optimization iterations for the ``MORPHIT`` fit type.
            coverage_weight: MorphIt coverage loss weight.  Higher values force
                spheres to fill the mesh volume more completely (default 1000.0).
                Only used when *fit_type* is ``MORPHIT``.
            protrusion_weight: MorphIt protrusion loss weight.  Higher values
                penalise sphere surface area outside the mesh (default 10.0).
                Only used when *fit_type* is ``MORPHIT``.
            compute_metrics: When True, compute and log per-link quality metrics
                (coverage, protrusion, surface gap) after fitting.
            clip_links: Per-link half-plane clipping constraints.  Keys are link
                names; values are ``(axis, offset)`` tuples where *axis* is
                ``"x"``, ``"y"``, or ``"z"`` (prefix with ``"-"`` for negative
                direction) and *offset* is the plane position along that axis.
                Spheres on the specified link will not extend past the plane.

        Returns:
            Dictionary mapping link names to lists of sphere dicts, each with
            keys ``"center"`` (3-element list) and ``"radius"`` (float).

        Example:
            >>> spheres = builder.fit_collision_spheres()
            >>> spheres = builder.fit_collision_spheres(sphere_density=2.0)
            >>> spheres = builder.fit_collision_spheres(
            ...     coverage_weight=2000.0, protrusion_weight=5.0, compute_metrics=True
            ... )
            >>> spheres = builder.fit_collision_spheres(
            ...     clip_links={"base_link": ("z", 0.0)}
            ... )
        """
        collision_spheres = {}
        total_spheres = 0

        log_info("Fitting collision spheres to robot links...")

        for link_name in tqdm.tqdm(self._mesh_link_names, desc="Fitting spheres"):
            clip_plane = None
            if clip_links and link_name in clip_links:
                axis, offset = clip_links[link_name]
                clip_plane = self._resolve_clip_plane(axis, offset)

            spheres = self._fit_single_link(
                link_name,
                sphere_density=sphere_density,
                surface_radius=surface_radius,
                fit_type=fit_type,
                use_collision_mesh=use_collision_mesh,
                iterations=iterations,
                coverage_weight=coverage_weight,
                protrusion_weight=protrusion_weight,
                compute_metrics=compute_metrics,
                clip_plane=clip_plane,
            )

            if spheres is not None:
                collision_spheres[link_name] = spheres
                total_spheres += len(spheres)

        log_info(
            f"Fitted {total_spheres} collision spheres across {len(collision_spheres)} links"
        )

        self._collision_spheres = collision_spheres
        return collision_spheres

    def refit_link_spheres(
        self,
        link_name: str,
        num_spheres: Optional[int] = None,
        sphere_density: float = 1.0,
        surface_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.MORPHIT,
        use_collision_mesh: bool = False,
        iterations: int = 200,
        coverage_weight: Optional[float] = None,
        protrusion_weight: Optional[float] = None,
        compute_metrics: bool = False,
        clip_plane: Optional[tuple] = None,
    ) -> List[Dict]:
        """Refit collision spheres for a single link.

        Useful for tuning individual links without refitting the whole robot.

        Args:
            link_name: Name of link to refit.
            num_spheres: Explicit sphere count for this link.  When ``None``,
                estimated automatically using *sphere_density*.
            sphere_density: Density multiplier (see :meth:`fit_collision_spheres`).
            surface_radius: Surface-sampling sphere radius.
            fit_type: Sphere fitting algorithm.
            use_collision_mesh: When True, use collision geometry.
            iterations: Optimization iterations for the ``MORPHIT`` fit type.
            coverage_weight: MorphIt coverage loss weight (default 1000.0).
            protrusion_weight: MorphIt protrusion loss weight (default 10.0).
            compute_metrics: When True, log quality metrics for this link.
            clip_plane: Half-plane constraint ``((nx, ny, nz), offset)`` in
                link-local coordinates.  When provided, spheres that extend past
                the plane are penalised during optimization and hard-clamped.

        Returns:
            List of sphere dicts for this link.

        Raises:
            ValueError: If link has no mesh geometry.

        Example:
            >>> builder.refit_link_spheres("panda_hand", sphere_density=3.0)
            >>> builder.refit_link_spheres("panda_link0", num_spheres=5)
        """
        if self._collision_spheres is None:
            self._collision_spheres = {}

        spheres = self._fit_single_link(
            link_name,
            num_spheres=num_spheres,
            sphere_density=sphere_density,
            surface_radius=surface_radius,
            fit_type=fit_type,
            use_collision_mesh=use_collision_mesh,
            iterations=iterations,
            coverage_weight=coverage_weight,
            protrusion_weight=protrusion_weight,
            compute_metrics=compute_metrics,
            clip_plane=clip_plane,
        )

        if spheres is None:
            log_and_raise(f"Link '{link_name}' has no mesh geometry")

        self._collision_spheres[link_name] = spheres
        log_info(f"Refitted {len(spheres)} spheres for link {link_name}")

        return spheres

    # ==========================================================================
    # COLLISION MATRIX COMPUTATION
    # ==========================================================================

    def compute_collision_matrix(
        self,
        prune_collisions: bool = True,
        num_samples: int = 1000,
        batch_size: int = 10000,
        seed: int = 345,
        custom_ignore: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        """Compute optimized self-collision ignore matrix.

        This method creates a dictionary of link pairs that should be ignored for
        collision checking. The matrix includes:

        1. Neighboring links (connected by joints) - always ignored
        2. Links that collide at default joint configuration - ignored (false positives)
        3. Links that never collide (optional, via sampling) - ignored for efficiency

        Args:
            prune_collisions: Whether to sample random configurations to find never-
                colliding link pairs. This significantly reduces collision checking
                overhead but takes longer to compute.
            num_samples: Number of random configurations to sample for pruning.
                More samples = better pruning but slower computation.
            batch_size: Batch size for parallel collision checking during sampling.
            seed: Random seed for reproducible sampling.
            custom_ignore: Additional collision ignore pairs to add. Dict mapping
                link names to lists of links to ignore. Example::

                    {"link_1": ["link_5", "link_6"]}

        Returns:
            Dictionary mapping link names to lists of links to ignore for collision.
            This is the self_collision_ignore matrix used in the robot configuration.

        Raises:
            ValueError: If :meth:`fit_collision_spheres` hasn't been called yet.

        Example:
            >>> # Compute with default pruning
            >>> matrix = builder.compute_collision_matrix()
            >>>
            >>> # More aggressive pruning
            >>> matrix = builder.compute_collision_matrix(num_samples=5000)
            >>>
            >>> # Skip pruning for faster computation
            >>> matrix = builder.compute_collision_matrix(prune_collisions=False)
            >>>
            >>> # With custom ignores
            >>> matrix = builder.compute_collision_matrix(
            ...     custom_ignore={"link_1": ["link_5"]}
            ... )
        """
        if self._collision_spheres is None:
            log_and_raise(
                "Must call fit_collision_spheres() before compute_collision_matrix()"
            )

        # Step 1: Create ignore matrix for neighboring links
        log_info("Creating collision ignore matrix for neighboring links...")
        self._self_collision_ignore = self._create_neighbor_ignore_matrix()

        # Step 2: Add custom ignore pairs
        if custom_ignore:
            log_info("Adding custom collision ignore pairs...")
            self._merge_collision_ignore(custom_ignore)

        # Step 3: Build initial config to check collisions
        temp_config = self._build_temp_config()

        # Step 4: Check for collisions at default joint configuration and add to ignore
        log_info("Checking for collisions at default joint configuration...")
        num_added = self._check_default_joint_configuration_collisions(temp_config)
        if num_added > 0:
            log_info(f"Added {num_added} pairs that collide at default joint configuration to ignore matrix")
            # Rebuild config with updated ignore matrix
            temp_config = self._build_temp_config()

        # Step 5: Prune never-colliding pairs (optional)
        if prune_collisions:
            log_info("Pruning never-colliding link pairs via sampling...")
            num_pruned = self._prune_collision_pairs(
                temp_config, num_samples, batch_size, seed
            )
            log_info(f"Pruned {num_pruned} never-colliding link pairs")

        return self._self_collision_ignore

    def add_collision_ignore(
        self,
        link_name: str,
        ignore_links: List[str],
    ) -> None:
        """Add links to collision ignore list.

        This manually adds link pairs to the collision ignore matrix. Useful when
        editing existing configurations or when you know certain links should never
        be checked for collision.

        Args:
            link_name: Link to add ignores for.
            ignore_links: List of link names to ignore collision with.

        Example:
            >>> builder.add_collision_ignore("link_1", ["link_5", "link_6"])
            >>> # Now link_1 will not be checked against link_5 or link_6
        """
        if self._self_collision_ignore is None:
            self._self_collision_ignore = {}

        if link_name not in self._self_collision_ignore:
            self._self_collision_ignore[link_name] = []

        for ignore_link in ignore_links:
            if ignore_link not in self._self_collision_ignore[link_name]:
                self._self_collision_ignore[link_name].append(ignore_link)

        log_info(f"Added {len(ignore_links)} collision ignores for link {link_name}")

    def remove_collision_ignore(
        self,
        link_name: str,
        ignore_links: List[str],
    ) -> None:
        """Remove links from collision ignore list.

        Args:
            link_name: Link to remove ignores from.
            ignore_links: List of link names to stop ignoring.

        Example:
            >>> builder.remove_collision_ignore("link_1", ["link_5"])
            >>> # Now link_1 WILL be checked against link_5
        """
        if (
            self._self_collision_ignore is None
            or link_name not in self._self_collision_ignore
        ):
            return

        for ignore_link in ignore_links:
            if ignore_link in self._self_collision_ignore[link_name]:
                self._self_collision_ignore[link_name].remove(ignore_link)

        log_info(f"Removed {len(ignore_links)} collision ignores for link {link_name}")

    # ==========================================================================
    # BUILD & SAVE
    # ==========================================================================

    def build(self) -> KinematicsLoaderCfg:
        """Build final robot configuration.

        This creates a :class:`KinematicsLoaderCfg` from all the computed data
        (spheres, collision matrix, etc.). This config can then be saved to file or
        used directly to create a :class:`Kinematics`.

        Returns:
            KinematicsLoaderCfg ready for use with cuRobo.

        Raises:
            ValueError: If :meth:`fit_collision_spheres` hasn't been called yet.

        Example:
            >>> builder.fit_collision_spheres()
            >>> builder.compute_collision_matrix()
            >>> config = builder.build()
            >>> # Use config with Kinematics
            >>> from curobo._src.robot.kinematics.kinematics import Kinematics
            >>> robot_model = Kinematics(config)
        """
        if self._collision_spheres is None:
            log_warn("Building robot configuration without fitting spheres.")
            self._collision_spheres = {}

        if self._self_collision_ignore is None:
            log_warn("Building robot configuration without computing collision matrix.")
            self._self_collision_ignore = self._create_neighbor_ignore_matrix()


        collision_link_names = list(self._collision_spheres.keys())

        config = KinematicsLoaderCfg(
            base_link=self._base_link,
            tool_frames=self._tool_frames,
            urdf_path=self.urdf_path,
            asset_root_path=self.asset_path,
            collision_link_names=collision_link_names,
            collision_spheres=self._collision_spheres,
            mesh_link_names=collision_link_names,
            self_collision_buffer=self._self_collision_buffer,
            self_collision_ignore=self._self_collision_ignore,
            device_cfg=self.device_cfg,
        )

        log_info("Built robot configuration")

        return config

    def save(
        self,
        config: KinematicsLoaderCfg,
        output_path: str,
        include_cspace: bool = True,
    ) -> None:
        """Save robot configuration to YAML file.

        Args:
            config: Robot configuration to save (from :meth:`build`).
            output_path: Path to output .yml file.
            include_cspace: Whether to include cspace (joint limits) configuration.

        Example:
            >>> config = builder.build()
            >>> builder.save(config, "my_robot.yml")
        """
        data_dict = {"kinematics": vars(config).copy()}
        data_dict["kinematics"]["format_version"] = 2.0

        # Add cspace if available and requested
        if include_cspace:
            if self._cspace_config is not None:
                data_dict["kinematics"]["cspace"] = self._cspace_config
            else:
                # Generate cspace from config
                try:
                    robot_model_config = KinematicsCfg.from_data_dict(
                        data_dict["kinematics"]
                    )
                    cspace_config = robot_model_config.kinematics_config.cspace
                    cspace_dict = vars(cspace_config).copy()

                    # Convert tensors to lists
                    for k in list(cspace_dict.keys()):
                        if isinstance(cspace_dict[k], torch.Tensor):
                            cspace_dict[k] = cspace_dict[k].cpu().tolist()

                    if "device_cfg" in cspace_dict:
                        del cspace_dict["device_cfg"]

                    data_dict["kinematics"]["cspace"] = cspace_dict
                except Exception as e:
                    log_warn(f"Could not generate cspace configuration: {e}")

        # Clean up non-serializable fields
        if "device_cfg" in data_dict["kinematics"]:
            del data_dict["kinematics"]["device_cfg"]

        # Create output directory if needed
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write YAML
        write_yaml(data_dict, str(output_path))
        log_info(f"Saved robot configuration to: {output_path}")

    def save_xrdf(
        self,
        config: KinematicsLoaderCfg,
        output_path: str,
        geometry_name: str = "collision_model",
    ) -> None:
        """Save robot configuration as XRDF file.

        XRDF (eXtended Robot Description Format) is NVIDIA's format for robot
        descriptions with collision spheres and self-collision matrices. It's useful
        for sharing robot configurations with Isaac Sim and other NVIDIA tools.

        The XRDF format contains the same information as the YAML format but
        structured according to the XRDF specification.

        Args:
            config: Robot configuration to save (from :meth:`build`).
            output_path: Path to output .xrdf file (YAML format).
            geometry_name: Name for the geometry section in XRDF (default: "collision_model").

        Example:
            >>> config = builder.build()
            >>> builder.save_xrdf(config, "my_robot.xrdf")
            >>> # Can also save as YAML
            >>> builder.save(config, "my_robot.yml")

        Note:
            XRDF files use YAML format but with a specific structure defined by
            the XRDF specification. See NVIDIA's XRDF documentation for details.
        """
        # Convert config to dict
        config_dict = {"kinematics": vars(config).copy()}

        # Add cspace if available
        if self._cspace_config is not None:
            config_dict["kinematics"]["cspace"] = self._cspace_config
        else:
            # Generate cspace from config
            try:
                robot_model_config = KinematicsCfg.from_data_dict(
                    config_dict["kinematics"]
                )
                cspace_config = robot_model_config.kinematics_config.cspace
                cspace_dict = vars(cspace_config).copy()

                # Convert tensors to lists
                for k in list(cspace_dict.keys()):
                    if isinstance(cspace_dict[k], torch.Tensor):
                        cspace_dict[k] = cspace_dict[k].cpu().tolist()

                if "device_cfg" in cspace_dict:
                    del cspace_dict["device_cfg"]

                config_dict["kinematics"]["cspace"] = cspace_dict
            except Exception as e:
                log_warn(f"Could not generate cspace for XRDF: {e}")

        # Clean up non-serializable fields
        if "device_cfg" in config_dict["kinematics"]:
            del config_dict["kinematics"]["device_cfg"]

        # Convert to XRDF format using utility function
        xrdf_dict = convert_curobo_to_xrdf(
            config_dict,
            geometry_name=geometry_name,
        )

        # Create output directory if needed
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write YAML (XRDF uses YAML format)
        write_yaml(xrdf_dict, str(output_path))
        log_info(f"Saved robot configuration as XRDF: {output_path}")

    def visualize(
        self,
        config: Optional[KinematicsLoaderCfg] = None,
        port: int = 8080,
        show_meshes: bool = False,
        show_spheres: bool = True,
        timeout_sec: int = -1,
    ) -> ViserVisualizer:
        """Start interactive visualization of robot configuration with Viser.

        This creates a web-based visualization server that shows the robot with its
        collision spheres. Useful for inspecting sphere fitting quality before saving.

        Args:
            config: Config to visualize. If None, calls :meth:`build` first.
            port: Viser server port (opens in browser at http://localhost:port).
            show_meshes: Whether to show original collision meshes.
            show_spheres: Whether to show fitted collision spheres.

        Returns:
            ViserVisualizer instance. Keep this reference alive to maintain visualization.

        Example:
            >>> builder.fit_collision_spheres()
            >>> viser = builder.visualize()
            >>> # Opens browser at http://localhost:8080
            >>> # Adjust sphere parameters if needed
            >>> builder.fit_collision_spheres(sphere_density=2.0)
            >>> viser = builder.visualize()  # Visualize again
            >>> # When satisfied, save
            >>> config = builder.build()
            >>> builder.save(config, "robot.yml")
        """
        if config is None:
            config = self.build()

        # Create robot content
        robot_content = ContentPath(robot_config_file={"kinematics": vars(config)})

        # Create Viser visualization
        viser_visualizer = ViserVisualizer(
            content_path=robot_content,
            connect_ip="0.0.0.0",
            connect_port=port,
            add_control_frames=False,
            visualize_robot_spheres=show_spheres,
        )

        # Add meshes if requested
        if show_meshes and self._collision_spheres:

            time.sleep(1.0)  # Give Viser time to initialize

            for link_name in self._collision_spheres.keys():
                mesh_list = self._parser.get_link_geometry(link_name, use_collision_mesh=True)
                for mesh in mesh_list:
                    mesh_instance = mesh.get_trimesh_mesh(transform_with_pose=True)
                    viser_visualizer.add_mesh(mesh_instance, name=f"mesh/{link_name}")

        log_info(f"Started Viser visualization server at http://localhost:{port}")
        start_time = time.time()
        while True:
            time.sleep(0.01)
            if timeout_sec > 0 and time.time() - start_time > timeout_sec:
                break

        return viser_visualizer

    # ==========================================================================
    # PROPERTIES (for inspection)
    # ==========================================================================

    @property
    def tool_frames(self) -> List[str]:
        """Get list of all robot link names."""
        return self._tool_frames

    @property
    def collision_link_names(self) -> List[str]:
        """Get list of links with fitted collision spheres."""
        if self._collision_spheres is None:
            return []
        return list(self._collision_spheres.keys())

    @property
    def collision_spheres(self) -> Optional[Dict[str, List[Dict]]]:
        """Get fitted collision spheres (None if not yet fitted)."""
        return self._collision_spheres

    @property
    def collision_matrix(self) -> Optional[Dict[str, List[str]]]:
        """Get collision ignore matrix (None if not yet computed)."""
        return self._self_collision_ignore

    @property
    def num_spheres(self) -> int:
        """Get total number of collision spheres across all links."""
        if self._collision_spheres is None:
            return 0
        return sum(len(spheres) for spheres in self._collision_spheres.values())

    @property
    def link_metrics(self) -> Dict[str, SphereFitMetrics]:
        """Per-link sphere fit quality metrics.

        Populated when :meth:`fit_collision_spheres` or :meth:`refit_link_spheres`
        is called with ``compute_metrics=True``.

        Returns:
            Dictionary mapping link names to :class:`SphereFitMetrics`.  Empty
            if metrics have not been computed.
        """
        return self._link_metrics

    # ==========================================================================
    # PRIVATE HELPER METHODS
    # ==========================================================================

    def _fit_single_link(
        self,
        link_name: str,
        num_spheres: Optional[int] = None,
        sphere_density: float = 1.0,
        surface_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.MORPHIT,
        use_collision_mesh: bool = False,
        iterations: int = 200,
        coverage_weight: Optional[float] = None,
        protrusion_weight: Optional[float] = None,
        compute_metrics: bool = False,
        clip_plane: Optional[tuple] = None,
    ) -> Optional[List[Dict]]:
        """Fit spheres to a single link (internal helper).

        Concatenates all geometries for the link into a single mesh before
        fitting, so spheres are distributed coherently across the whole link.
        """
        geometry_list = self._parser.get_link_geometry(
            link_name, use_collision_mesh=use_collision_mesh,
        )
        if not geometry_list:
            return None

        meshes = []
        for geometry in geometry_list:
            m = geometry.get_trimesh_mesh(transform_with_pose=True)
            if m is not None:
                m.fill_holes()
                trimesh.repair.fix_normals(m)
                trimesh.repair.fix_inversion(m)
                trimesh.repair.fix_winding(m)
                meshes.append(m)

        if not meshes:
            return None

        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

        fit_result = fit_spheres_to_mesh(
            mesh,
            num_spheres=num_spheres,
            sphere_density=sphere_density,
            surface_radius=surface_radius,
            fit_type=fit_type,
            iterations=iterations,
            compute_metrics=compute_metrics,
            coverage_weight=coverage_weight,
            protrusion_weight=protrusion_weight,
            clip_plane=clip_plane,
        )
        log_info(f"  {link_name}: {fit_result.num_spheres} spheres")

        if compute_metrics and fit_result.metrics is not None:
            m = fit_result.metrics
            self._link_metrics[link_name] = m
            log_info(
                f"  {link_name} metrics: "
                f"coverage={m.coverage:.3f}  "
                f"protrusion={m.protrusion:.3f}  "
                f"protrusion_dist_mean={m.protrusion_dist_mean:.4f}m  "
                f"protrusion_dist_p95={m.protrusion_dist_p95:.4f}m  "
                f"surface_gap_mean={m.surface_gap_mean:.4f}m  "
                f"surface_gap_p95={m.surface_gap_p95:.4f}m  "
                f"volume_ratio={m.volume_ratio:.2f}"
            )

        return [
            {"center": c, "radius": r}
            for c, r in zip(fit_result.centers.tolist(), fit_result.radii.tolist())
        ]

    def _create_neighbor_ignore_matrix(self) -> Dict[str, List[str]]:
        """Create collision ignore matrix for neighboring (parent-child) links."""
        ignore_matrix = {}

        for link_name in self._link_names:
            if link_name == self._base_link:
                continue

            link_params = self._parser.get_link_parameters(link_name)
            parent_link_name = link_params.parent_link_name

            if parent_link_name is not None:
                # Add bidirectional ignore
                if link_name not in ignore_matrix:
                    ignore_matrix[link_name] = []
                if parent_link_name not in ignore_matrix[link_name]:
                    ignore_matrix[link_name].append(parent_link_name)

                if parent_link_name not in ignore_matrix:
                    ignore_matrix[parent_link_name] = []
                if link_name not in ignore_matrix[parent_link_name]:
                    ignore_matrix[parent_link_name].append(link_name)

        return ignore_matrix

    def _merge_collision_ignore(self, custom_ignore: Dict[str, List[str]]) -> None:
        """Merge custom ignore pairs into existing matrix."""
        for link_name, ignore_list in custom_ignore.items():
            if link_name not in self._self_collision_ignore:
                self._self_collision_ignore[link_name] = []
            for ignore_link in ignore_list:
                if ignore_link not in self._self_collision_ignore[link_name]:
                    self._self_collision_ignore[link_name].append(ignore_link)

    def _build_temp_config(self) -> KinematicsCfg:
        """Build temporary config for collision checking (internal)."""
        temp_gen_config = KinematicsLoaderCfg(
            base_link=self._base_link,
            tool_frames=self._tool_frames,
            urdf_path=self.urdf_path,
            asset_root_path=self.asset_path,
            collision_link_names=list(self._collision_spheres.keys()),
            collision_spheres=self._collision_spheres,
            mesh_link_names=list(self._collision_spheres.keys()),
            self_collision_buffer=self._self_collision_buffer,
            self_collision_ignore=self._self_collision_ignore,
            device_cfg=self.device_cfg,
            load_meshes=False,
        )

        data_dict = {"kinematics": vars(temp_gen_config)}
        return KinematicsCfg.from_data_dict(data_dict["kinematics"])

    def _check_default_joint_configuration_collisions(self, robot_config: KinematicsCfg) -> int:
        """Check for collisions at default joint configuration and add to ignore matrix. Returns num added."""
        # Create robot model
        kinematics = Kinematics(robot_config)
        kinematics_tensor_config = robot_config.kinematics_config

        # Setup collision checker
        self_collision_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                self_collision_kin_config=robot_config.self_collision_config,
                store_pair_distance=True,
            )
        )

        # Check at default joint configuration
        joint_state = kinematics.default_joint_state.clone()
        joint_state.position = joint_state.position.view(1, -1)
        kin_state = kinematics.compute_kinematics(joint_state)
        robot_spheres = kin_state.robot_spheres

        self_collision_cost.setup_batch_tensors(1, 1)
        out_distance = self_collision_cost.forward(robot_spheres)
        pair_distance = self_collision_cost._pair_distance.view(-1)

        # Find colliding pairs (positive distance = collision)
        colliding_sphere_indicators = pair_distance > 0

        if torch.count_nonzero(colliding_sphere_indicators) == 0:
            return 0

        # Map colliding sphere pairs to link pairs
        collision_pairs = robot_config.self_collision_config.collision_pairs
        colliding_sphere_indices = collision_pairs[colliding_sphere_indicators, :]
        colliding_link_indices = kinematics_tensor_config.link_sphere_idx_map[
            colliding_sphere_indices.to(dtype=torch.int32)
        ]
        unique_colliding_link_indices = torch.unique(colliding_link_indices, dim=0)

        # Get link index to name mapping
        link_idx_to_name_map = {
            v: k for k, v in kinematics_tensor_config.link_name_to_idx_map.items()
        }

        # Add to ignore matrix
        unique_colliding_link_indices = unique_colliding_link_indices.cpu().tolist()
        num_added = 0

        for link_pair in unique_colliding_link_indices:
            link_idx1, link_idx2 = link_pair
            link_name1 = link_idx_to_name_map[link_idx1]
            link_name2 = link_idx_to_name_map[link_idx2]

            if link_name1 not in self._self_collision_ignore:
                self._self_collision_ignore[link_name1] = []
            if link_name2 not in self._self_collision_ignore[link_name1]:
                self._self_collision_ignore[link_name1].append(link_name2)
                num_added += 1

        return num_added

    def _prune_collision_pairs(
        self,
        robot_config: KinematicsCfg,
        num_samples: int,
        batch_size: int,
        seed: int,
    ) -> int:
        """Prune never-colliding pairs via sampling. Returns number of pairs pruned."""
        # Create robot model
        kinematics = Kinematics(robot_config)
        kinematics_tensor_config = robot_config.kinematics_config

        # Setup collision checker
        self_collision_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                self_collision_kin_config=robot_config.self_collision_config,
                store_pair_distance=True,
            )
        )

        # Get joint limits
        joint_limits = kinematics.config.get_joint_limits()

        # Create sample generator
        sample_generator = SampleBuffer.create_halton_sample_buffer(
            robot_config.dof,
            device_cfg=robot_config.device_cfg,
            low_bounds=joint_limits.position_lower_limits - 0.1,
            up_bounds=joint_limits.position_upper_limits + 0.1,
            seed=seed,
            store_buffer=None,
        )

        # Group collision pairs by link pairs
        collision_pairs = robot_config.self_collision_config.collision_pairs
        link_sphere_idx_map = kinematics_tensor_config.link_sphere_idx_map
        collision_link_pairs = link_sphere_idx_map[collision_pairs.to(dtype=torch.int32)]

        # Get unique link pairs and create mapping
        unique_link_pairs, inverse_indices = torch.unique(
            collision_link_pairs, dim=0, return_inverse=True
        )

        # Create tensor mapping: for each unique link pair, which sphere pair indices belong to it
        max_spheres_per_link_pair = torch.bincount(inverse_indices).max().item()
        link_pair_sphere_indices = torch.full(
            (len(unique_link_pairs), max_spheres_per_link_pair),
            -1,
            dtype=torch.long,
            device=collision_pairs.device,
        )
        link_pair_sphere_counts = torch.zeros(len(unique_link_pairs), dtype=torch.long)

        # Fill the mapping tensor
        for i, link_pair_idx in enumerate(inverse_indices):
            count = link_pair_sphere_counts[link_pair_idx]
            link_pair_sphere_indices[link_pair_idx, count] = i
            link_pair_sphere_counts[link_pair_idx] += 1

        # Track which link pairs never collide
        link_pairs_never_collide = torch.ones(
            len(unique_link_pairs), dtype=torch.bool, device=collision_pairs.device
        )

        # Sample and check collisions
        pbar = tqdm.tqdm(range(num_samples), desc="Pruning collisions")
        for i in pbar:
            samples = sample_generator.get_samples(batch_size, bounded=True)

            # Compute collisions
            kin_state = kinematics.compute_kinematics(
                JointState.from_position(samples.contiguous(), joint_names=kinematics.joint_names)
            )
            robot_spheres = kin_state.robot_spheres
            self_collision_cost.setup_batch_tensors(batch_size, 1)
            out_distance = self_collision_cost.forward(robot_spheres)
            pair_distance = self_collision_cost._pair_distance.view(batch_size, -1)

            # Check each link pair
            active_mask = link_pairs_never_collide
            active_indices = torch.where(active_mask)[0]

            if len(active_indices) > 0:
                for link_pair_idx in active_indices:
                    sphere_indices = link_pair_sphere_indices[link_pair_idx]
                    valid_indices = sphere_indices[sphere_indices >= 0]

                    if len(valid_indices) > 0:
                        link_pair_distances = pair_distance[:, valid_indices]
                        if torch.any(link_pair_distances > 0.0):
                            link_pairs_never_collide[link_pair_idx] = False

            non_colliding_count = torch.sum(link_pairs_never_collide).item()
            total_link_pairs = len(unique_link_pairs)
            pbar.set_description(
                f"Non-colliding: {(non_colliding_count / total_link_pairs) * 100:.1f}%"
            )

        # Get never-colliding link pairs
        never_colliding_link_pairs = unique_link_pairs[link_pairs_never_collide]
        never_colliding_link_pairs_cpu = never_colliding_link_pairs.cpu().tolist()

        # Add to ignore matrix
        link_idx_to_name_map = {
            v: k for k, v in kinematics_tensor_config.link_name_to_idx_map.items()
        }

        for link_pair in never_colliding_link_pairs_cpu:
            link_idx1, link_idx2 = link_pair
            link_name1 = link_idx_to_name_map[link_idx1]
            link_name2 = link_idx_to_name_map[link_idx2]

            if link_name1 not in self._self_collision_ignore:
                self._self_collision_ignore[link_name1] = []
            if link_name2 not in self._self_collision_ignore[link_name1]:
                self._self_collision_ignore[link_name1].append(link_name2)

        return len(never_colliding_link_pairs_cpu)

