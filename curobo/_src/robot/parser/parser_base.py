# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Base module for parsing kinematics from different representations.

cuRobo provides kinematics parsing from an URDF and a partial implementation for parsing from
a USD. To parse from other representations, an user can extend the :class:`~RobotParser`
class and implement only the abstract methods. Optionally, user can also provide functions for
reading meshes, useful for debugging and visualization.

"""

from __future__ import annotations

# Standard Library
from abc import abstractmethod
from typing import Dict, List, Optional

# Third Party
from curobo._src.geom.types import Mesh, Obstacle

# CuRobo
from curobo._src.robot.types.link_params import LinkParams
from curobo._src.robot.types.joint_types import JointType
from curobo._src.util.logging import log_and_raise


class RobotParser:
    """Base class for parsing kinematics.

    Implement abstractmethods to parse kinematics from any representation. Optionally, implement
    methods for reading meshes for visualization and debugging.
    """

    def __init__(self, extra_links: Optional[Dict[str, LinkParams]] = None) -> None:
        """Initialize the RobotParser.

        Args:
            extra_links: Additional links to be added to the kinematic tree.
        """
        #: Parent link for all link in the kinematic tree.
        self._parent_map = {}
        self.extra_links = extra_links
        self.build_link_parent()
        # add extra links to parent:
        if self.extra_links is not None and len(list(self.extra_links.keys())) > 0:
            for i in self.extra_links:
                self._parent_map[i] = {"parent": self.extra_links[i].parent_link_name}
                if self.extra_links[i].child_link_name is not None:
                    self._parent_map[self.extra_links[i].child_link_name]["parent"] = i

    @abstractmethod
    def build_link_parent(self):
        """Build a map of parent links to each link in the kinematic tree.

        Use this function to fill ``_parent_map``. Check
        :meth:`curobo._src.robot.parser.parser_urdf.UrdfRobotParser.build_link_parent`
        for an example implementation.
        """
        pass

    @abstractmethod
    def get_link_parameters(self, link_name: str, base: bool = False) -> LinkParams:
        """Get parameters of a link in the kinematic tree.

        Args:
            link_name: Name of the link.
            base: Is this the base link of the robot?

        Returns:
            LinkParams: Parameters of the link.
        """
        pass

    def add_absolute_path_to_link_meshes(self, mesh_dir: str = ""):
        """Add absolute path to link meshes.

        Args:
            mesh_dir: Absolute path to the directory containing link meshes.
        """
        pass

    def get_link_mesh(self, link_name: str) -> Mesh:
        """Get mesh of a link.

        Args:
            link_name: Name of the link.

        Returns:
            Mesh: Mesh of the link.
        """
        pass
    def get_link_geometry(self, link_name: str) -> List[Obstacle]:
        """Get geometry of a link.

        Args:
            link_name: Name of the link.

        Returns:
            List[Geometry]: List of geometry of the link.
        """
        pass
    def get_chain(self, base_link: str, ee_link: str) -> List[str]:
        """Get list of links attaching ee_link to base_link.

        Args:
            base_link (str): Name of base link.
            ee_link (str): Name of end-effector link.

        Returns:
            List[str]: List of link names starting from base_link to ee_link.
        """
        chain_links = [ee_link]
        link = ee_link
        while link != base_link:
            if link not in self._parent_map:
                log_and_raise(f"Link {link} not found in parent map {self._parent_map}")
            link = self._parent_map[link]["parent"]
            # add link to chain:
            chain_links.append(link)
        chain_links.reverse()
        return chain_links

    def get_actuated_joint_names(self) -> List[str]:
        """Get names of independently actuated joints in the robot.

        Returns:
            Names of all non-fixed, non-mimic joints in the robot.
        """
        joint_names = []
        for parent_data in self._parent_map.values():
            if "joint_name" not in parent_data:
                continue
            joint_name = parent_data["joint_name"]
            joint = self._robot.joint_map[joint_name]
            if joint.type != "fixed" and joint.mimic is None:
                joint_names.append(joint_name)
        if self.extra_links is not None:
            for link_params in self.extra_links.values():
                if (
                    link_params.joint_type != JointType.FIXED
                    and link_params.mimic_joint_name is None
                    and link_params.joint_name not in joint_names
                ):
                    joint_names.append(link_params.joint_name)
        return joint_names

    def get_mimic_joint_map(self) -> Dict[str, str]:
        """Get raw mimic joint names mapped to their active joint names.

        Returns:
            Mapping from mimic joint name to the independently actuated joint name it mimics.
        """
        mimic_joint_map = {}
        for parent_data in self._parent_map.values():
            if "joint_name" not in parent_data:
                continue
            joint_name = parent_data["joint_name"]
            joint = self._robot.joint_map[joint_name]
            if joint.mimic is not None:
                mimic_joint_map[joint_name] = joint.mimic.joint
        if self.extra_links is not None:
            for link_params in self.extra_links.values():
                if link_params.mimic_joint_name is not None:
                    mimic_joint_map[link_params.mimic_joint_name] = link_params.joint_name
        return mimic_joint_map

    def _get_from_extra_links(self, link_name: str) -> LinkParams:
        """Get link parameters for extra links.

        Args:
            link_name: Name of the link.

        Returns:
            LinkParams: Link parameters if found, else None.
        """
        if self.extra_links is None:
            return None
        if link_name in self.extra_links.keys():
            return self.extra_links[link_name]
        return None

    def get_link_names(self) -> List[str]:
        """Get names of all links in the kinematic tree.

        Returns:
            List[str]: List of link names.
        """
        return list(self._parent_map.keys())
