# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Generates a Tensor representation of kinematics for use in
:class:`~curobo.cuda_robot_model.Kinematics`. This module reads the robot from a
:class:`~curobo._src.robot.parser.parser_base.RobotParser` and
generates the necessary tensors for kinematics computation.

"""

from __future__ import annotations

# Standard Library
import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.robot.types.link_params import LinkParams
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml


@dataclass
class KinematicsLoaderCfg:
    """Robot representation generator configuration, loads from a dictionary."""

    #: Name of base link for kinematic tree.
    base_link: str

    #: Device to load cuda robot model.
    device_cfg: DeviceCfg = DeviceCfg()

    #: Name of tool frames (end-effector links) to compute pose.
    tool_frames: Optional[List[str]] = None

    #: Name of links to compute sphere positions for use in collision checking.
    collision_link_names: Optional[List[str]] = None

    #: Collision spheres that fill the volume occupied by the links of the robot.
    #: Collision spheres can be generated for robot using `Isaac Sim Robot Description Editor <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/advanced_tutorials/tutorial_motion_generation_robot_description_editor.html#collision-spheres>`_.
    collision_spheres: Union[None, str, Dict[str, Any]] = None

    #: Radius buffer to add to collision spheres as padding.
    collision_sphere_buffer: Union[float, Dict[str, float]] = 0.0

    #: Padding to add for self collision between links. Some robots use a large padding
    #: for self collision avoidance (e.g., `MoveIt Panda Issue <https://github.com/ros-planning/panda_moveit_config/pull/35#issuecomment-671333863>`_).
    self_collision_buffer: Optional[Dict[str, float]] = None

    #: Dictionary with each key as a link name and value as a list of link names to ignore self
    #: collision. E.g., {"link1": ["link2", "link3"], "link2": ["link3", "link4"]} will
    #: ignore self collision between link1 and link2, link1 and link3, link2 and link3, link2 and
    #: link4. The mapping is bidirectional so it's sufficient to mention the mapping in one
    #: direction (i.e., not necessary to mention "link1" in ignore list for "link2").
    self_collision_ignore: Optional[Dict[str, List[str]]] = None

    #: Debugging information to pass to kinematics module.
    debug: Optional[Dict[str, Any]] = None

    #: Path of meshes of robot links. Currently not used as we represent robot link geometry with
    #: collision spheres.
    asset_root_path: str = ""

    #: Names of links to load meshes for visualization. This is only used for exporting
    #: visualizations.
    mesh_link_names: Optional[List[str]] = None

    #: Names of links that are in contact with the gripper. This is used to disable collision for these links during grasp.
    #: This is used to disable collision for these links during grasp.
    grasp_contact_link_names: Optional[List[str]] = None

    #: Set this to true to add mesh_link_names to tool_frames when computing kinematics.
    load_tool_frames_with_mesh: bool = False

    #: Path to load robot urdf.
    urdf_path: Optional[str] = None


    #: Lock active joints in the kinematic tree. This will convert the joint to a fixed joint with
    #: joint angle given from this dictionary.
    lock_joints: Optional[Dict[str, float]] = None

    #: Additional links to add to parsed kinematics tree. This is useful for adding fixed links
    #: that are not present in the URDF or USD.
    extra_links: Optional[Dict[str, LinkParams]] = None

    #: Deprecated way to add a fixed link.
    add_object_link: bool = False

    #: Deprecated flag to load assets from external module. Now, pass absolute path to
    #: asset_root_path or use :class:`~curobo.util.file_path.ContentPath`.
    use_external_assets: bool = False

    #: Deprecated path to load assets from external module. Use
    #: :class:`~curobo.util.file_path.ContentPath` instead.
    external_asset_path: Optional[str] = None

    #: Deprecated path to load robot configs from external module. Use
    #: :class:`~curobo.util.file_path.ContentPath` instead.
    external_robot_configs_path: Optional[str] = None

    #: Create n collision spheres for links with name
    extra_collision_spheres: Optional[Dict[str, int]] = None

    #: When False, skip loading collision spheres even if they are specified
    #: in the robot config. This saves memory and FK compute when neither
    #: self-collision nor environment collision checking is needed.
    load_collision_spheres: bool = True

    #: Number of environment configurations for link_spheres. When > 1, link_spheres
    #: is allocated as [num_envs, N, 4] to support per-env attached object states.
    num_envs: int = 1

    #: Configuration space parameters for robot (e.g, acceleration, jerk limits).
    cspace: Union[None, CSpaceParams, Dict[str, List[Any]]] = None

    #: Enable loading meshes from kinematics parser.
    load_meshes: bool = False

    #: Deprecated flag to enable global cumulative transformation matrix.
    use_global_cumul: bool = True

    format_version: float = 2.0

    def __post_init__(self):
        """Post initialization adds absolute paths, converts dictionaries to objects."""
        # add root path:
        # Check if an external asset path is provided:
        asset_path = get_assets_path()
        robot_path = get_robot_configs_path()
        if self.external_asset_path is not None:
            log_warn("Deprecated: external_asset_path is deprecated, use ContentPath")
            asset_path = self.external_asset_path
        if self.external_robot_configs_path is not None:
            log_warn("Deprecated: external_robot_configs_path is deprecated, use ContentPath")
            robot_path = self.external_robot_configs_path

        if self.urdf_path is not None:
            self.urdf_path = join_path(asset_path, self.urdf_path)
        if self.asset_root_path != "":
            self.asset_root_path = join_path(asset_path, self.asset_root_path)
        elif self.urdf_path is not None:
            self.asset_root_path = os.path.dirname(self.urdf_path)

        if not self.load_collision_spheres:
            self.collision_spheres = None
            self.collision_link_names = []
            self.extra_collision_spheres = None

        if self.collision_spheres is None and (
            self.collision_link_names is not None and len(self.collision_link_names) > 0
        ):
            log_and_raise("collision link names are provided without robot collision spheres")
        if self.load_tool_frames_with_mesh:
            if self.tool_frames is None:
                self.tool_frames = copy.deepcopy(self.mesh_link_names)
            else:
                self.tool_frames = copy.deepcopy(self.tool_frames)

                for i in self.mesh_link_names:
                    if i not in self.tool_frames:
                        self.tool_frames.append(i)
        if self.tool_frames is None:
            log_and_raise("tool_frames must be specified")
        if self.collision_link_names is None:
            self.collision_link_names = []
        if self.collision_spheres is not None:
            if isinstance(self.collision_spheres, str):
                log_warn("Collision spheres cannot be loaded from a file")
                coll_yml = join_path(robot_path, self.collision_spheres)
                coll_params = load_yaml(coll_yml)

                self.collision_spheres = coll_params["collision_spheres"]
            if self.extra_collision_spheres is not None:
                for k in self.extra_collision_spheres.keys():
                    new_spheres = [
                        {"center": [0.0, 0.0, 0.0], "radius": -100.0}
                        for n in range(self.extra_collision_spheres[k])
                    ]
                    self.collision_spheres[k] = new_spheres

        if self.extra_links is None:
            self.extra_links = {}
        else:
            for k in self.extra_links.keys():
                if isinstance(self.extra_links[k], dict):
                    self.extra_links[k] = LinkParams.create(self.extra_links[k])
        if isinstance(self.cspace, Dict):
            if "device_cfg" not in self.cspace:
                self.cspace["device_cfg"] = self.device_cfg
            self.cspace = CSpaceParams(**self.cspace)
        if self.mesh_link_names is None:
            self.mesh_link_names = []

        if self.grasp_contact_link_names is not None:
            self.grasp_contact_link_names = copy.deepcopy(self.grasp_contact_link_names)

