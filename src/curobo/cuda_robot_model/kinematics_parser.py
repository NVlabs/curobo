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

# Standard Library
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Third Party
import numpy as np

# CuRobo
from curobo.cuda_robot_model.types import JointType
from curobo.geom.types import Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose


@dataclass
class LinkParams:
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
    def from_dict(dict_data):
        dict_data["joint_type"] = JointType[dict_data["joint_type"]]
        dict_data["fixed_transform"] = (
            Pose.from_list(dict_data["fixed_transform"], tensor_args=TensorDeviceType())
            .get_numpy_matrix()
            .reshape(4, 4)
        )

        return LinkParams(**dict_data)


class KinematicsParser:
    def __init__(self, extra_links: Optional[Dict[str, LinkParams]] = None) -> None:
        self._parent_map = {}
        self.extra_links = extra_links
        self.build_link_parent()
        # add extra links to parent:
        if self.extra_links is not None and len(list(self.extra_links.keys())) > 0:
            for i in self.extra_links:
                self._parent_map[i] = {"parent": self.extra_links[i].parent_link_name}

    def build_link_parent(self):
        """Build a map of parent links to each link in the kinematic tree.

        NOTE: Use this function to fill self._parent_map.
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

    def _get_from_extra_links(self, link_name: str) -> LinkParams:
        if self.extra_links is None:
            return None
        if link_name in self.extra_links.keys():
            return self.extra_links[link_name]
        return None

    def get_link_parameters(self, link_name: str, base: bool = False) -> LinkParams:
        pass

    def add_absolute_path_to_link_meshes(self, mesh_dir: str = ""):
        pass

    def get_link_mesh(self, link_name: str) -> Mesh:
        pass
