#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""
Base module for parsing kinematics from different representations.

cuRobo provides kinematics parsing from an URDF and a partial implementation for parsing from
a USD. To parse from other representations, an user can extend the :class:`~KinematicsParser`
class and implement only the abstract methods. Optionally, user can also provide functions for
reading meshes, useful for debugging and visualization.

"""
from __future__ import annotations

# Standard Library
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Third Party
import numpy as np

# CuRobo
from curobo.cuda_robot_model.types import JointType
from curobo.geom.types import Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose


@dataclass
class LinkParams:
    """Parameters of a link in the kinematic tree."""

    link_name: str
    joint_name: str
    joint_type: JointType
    fixed_transform: np.ndarray
    parent_link_name: Optional[str] = None
    joint_limits: Optional[List[float]] = None
    joint_axis: Optional[np.ndarray] = None
    joint_id: Optional[int] = None
    joint_velocity_limits: List[float] = field(default_factory=lambda: [-2.0, 2.0])
    joint_offset: List[float] = field(default_factory=lambda: [1.0, 0.0])
    mimic_joint_name: Optional[str] = None

    @staticmethod
    def from_dict(dict_data: Dict[str, Any]) -> LinkParams:
        """Create a LinkParams object from a dictionary.

        Args:
            dict_data: Dictionary containing link parameters.

        Returns:
            LinkParams: Link parameters object.
        """
        dict_data["joint_type"] = JointType[dict_data["joint_type"]]
        dict_data["fixed_transform"] = (
            Pose.from_list(dict_data["fixed_transform"], tensor_args=TensorDeviceType())
            .get_numpy_matrix()
            .reshape(4, 4)
        )

        return LinkParams(**dict_data)


class KinematicsParser:
    """
    Base class for parsing kinematics.

    Implement abstractmethods to parse kinematics from any representation. Optionally, implement
    methods for reading meshes for visualization and debugging.
    """

    def __init__(self, extra_links: Optional[Dict[str, LinkParams]] = None) -> None:
        """Initialize the KinematicsParser.

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

    @abstractmethod
    def build_link_parent(self):
        """Build a map of parent links to each link in the kinematic tree.

        Use this function to fill ``_parent_map``. Check
        :meth:`curobo.cuda_robot_model.urdf_kinematics_parser.UrdfKinematicsParser.build_link_parent`
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
            link = self._parent_map[link]["parent"]
            # add link to chain:
            chain_links.append(link)
        chain_links.reverse()
        return chain_links

    def get_controlled_joint_names(self) -> List[str]:
        """Get names of all controlled joints in the robot.

        Returns:
            Names of all controlled joints in the robot.
        """
        j_list = []
        for k in self._parent_map.keys():
            joint_name = self._parent_map[k]["joint_name"]
            joint = self._robot.joint_map[joint_name]
            if joint.type != "fixed" and joint.mimic is None:
                j_list.append(joint_name)
        return j_list

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
