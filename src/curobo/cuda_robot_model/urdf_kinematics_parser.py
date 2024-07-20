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
Parses Kinematics from an `URDF <https://wiki.ros.org/urdf>`__ file that describes the
kinematic tree of a robot.
"""

# Standard Library
from typing import Dict, List, Optional, Tuple

# Third Party
import numpy as np
import yourdfpy
from lxml import etree

# CuRobo
from curobo.cuda_robot_model.kinematics_parser import KinematicsParser, LinkParams
from curobo.cuda_robot_model.types import JointType
from curobo.geom.types import Mesh as CuroboMesh
from curobo.types.math import Pose
from curobo.util.logger import log_error, log_warn
from curobo.util_file import join_path


class UrdfKinematicsParser(KinematicsParser):
    """Parses Kinematics from an URDF file and provides access to the kinematic tree."""

    def __init__(
        self,
        urdf_path,
        load_meshes: bool = False,
        mesh_root: str = "",
        extra_links: Optional[Dict[str, LinkParams]] = None,
        build_scene_graph: bool = False,
    ) -> None:
        """Initialize instance with URDF file path.

        Args:
            urdf_path: Path to the URDF file.
            load_meshes: Load meshes for links from the URDF file.
            mesh_root: Absolute path to the directory where link meshes are stored.
            extra_links: Extra links to add to the kinematic tree.
            build_scene_graph: Build scene graph for the robot. Set this to True if you want to
                determine the root link of the robot.
        """
        # load robot from urdf:
        self._robot = yourdfpy.URDF.load(
            urdf_path,
            load_meshes=load_meshes,
            build_scene_graph=build_scene_graph,
            mesh_dir=mesh_root,
            filename_handler=yourdfpy.filename_handler_null,
        )
        super().__init__(extra_links)

    def build_link_parent(self):
        """Build parent map for the robot."""
        self._parent_map = {}
        for jid, j in enumerate(self._robot.joint_map):
            self._parent_map[self._robot.joint_map[j].child] = {
                "parent": self._robot.joint_map[j].parent,
                "jid": jid,
                "joint_name": j,
            }

    def _get_joint_name(self, idx) -> str:
        """Get the name of the joint at the given index.

        Args:
            idx: Index of the joint.

        Returns:
            str: Name of the joint.
        """
        joint = self._robot.joint_names[idx]
        return joint

    def _get_joint_limits(self, joint: yourdfpy.Joint) -> Tuple[Dict[str, float], str]:
        """Get the limits of a joint.

        This function converts continuous joints to revolute joints with limits [-6.28, 6.28].

        Args:
            joint: Instance of the joint.

        Returns:
            Tuple[Dict[str, float], str]: Limits of the joint and the type of the joint
                (revolute or prismatic).
        """

        joint_type = joint.type
        if joint_type != "continuous":
            joint_limits = {
                "effort": joint.limit.effort,
                "lower": joint.limit.lower,
                "upper": joint.limit.upper,
                "velocity": joint.limit.velocity,
            }
        else:
            log_warn("Converting continuous joint to revolute with limits[-6.28,6.28]")
            joint_type = "revolute"
            joint_limits = {
                "effort": joint.limit.effort,
                "lower": -3.14 * 2,
                "upper": 3.14 * 2,
                "velocity": joint.limit.velocity,
            }
        return joint_limits, joint_type

    def get_link_parameters(self, link_name: str, base=False) -> LinkParams:
        """Get parameters of a link in the kinematic tree.

        Args:
            link_name: Name of the link.
            base: Is this the base link of the robot?

        Returns:
            LinkParams: Parameters of the link.
        """

        link_params = self._get_from_extra_links(link_name)
        if link_params is not None:
            return link_params
        body_params = {}
        body_params["link_name"] = link_name
        mimic_joint_name = None
        if base:
            body_params["parent_link_name"] = None
            joint_transform = np.eye(4)
            joint_name = "base_joint"
            active_joint_name = joint_name
            joint_type = "fixed"
            joint_limits = None
            joint_axis = None
            body_params["joint_id"] = 0
            body_params["joint_type"] = JointType.FIXED

        else:
            parent_data = self._parent_map[link_name]
            body_params["parent_link_name"] = parent_data["parent"]

            jid, joint_name = parent_data["jid"], parent_data["joint_name"]
            body_params["joint_id"] = jid
            joint = self._robot.joint_map[joint_name]
            active_joint_name = joint_name
            joint_transform = joint.origin
            if joint_transform is None:
                joint_transform = np.eye(4)
            joint_type = joint.type
            joint_limits = None
            joint_axis = None
            body_params["joint_type"] = JointType.FIXED

            if joint_type != "fixed":
                joint_offset = [1.0, 0.0]
                joint_limits, joint_type = self._get_joint_limits(joint)

                if joint.mimic is not None:
                    joint_offset = [joint.mimic.multiplier, joint.mimic.offset]
                    # read joint limits of active joint:
                    mimic_joint_name = joint_name
                    active_joint_name = joint.mimic.joint
                    active_joint = self._robot.joint_map[active_joint_name]
                    active_joint_limits, _ = self._get_joint_limits(active_joint)
                    # check to make sure mimic joint limits are not larger than active joint:
                    if (
                        active_joint_limits["lower"] * joint_offset[0] + joint_offset[1]
                        < joint_limits["lower"]
                    ):
                        log_error(
                            "mimic joint can go out of it's lower limit as active joint has larger range "
                            + "FIX: make mimic joint's lower limit even lower "
                            + active_joint_name
                            + " "
                            + mimic_joint_name
                        )
                    if (
                        active_joint_limits["upper"] * joint_offset[0] + joint_offset[1]
                        > joint_limits["upper"]
                    ):
                        log_error(
                            "mimic joint can go out of it's upper limit as active joint has larger range "
                            + "FIX: make mimic joint's upper limit higher"
                            + active_joint_name
                            + " "
                            + mimic_joint_name
                        )
                    if active_joint_limits["velocity"] * joint_offset[0] > joint_limits["velocity"]:
                        log_error(
                            "mimic joint can move at higher velocity than active joint,"
                            + "increase velocity limit for mimic joint"
                            + active_joint_name
                            + " "
                            + mimic_joint_name
                        )
                    joint_limits = active_joint_limits

                joint_axis = joint.axis

                body_params["joint_limits"] = [joint_limits["lower"], joint_limits["upper"]]
                body_params["joint_velocity_limits"] = [
                    -1.0 * joint_limits["velocity"],
                    joint_limits["velocity"],
                ]
                if joint_type == "prismatic":
                    if abs(joint_axis[0]) == 1:
                        joint_type = JointType.X_PRISM
                    if abs(joint_axis[1]) == 1:
                        joint_type = JointType.Y_PRISM
                    if abs(joint_axis[2]) == 1:
                        joint_type = JointType.Z_PRISM
                elif joint_type == "revolute":
                    if abs(joint_axis[0]) == 1:
                        joint_type = JointType.X_ROT
                    if abs(joint_axis[1]) == 1:
                        joint_type = JointType.Y_ROT
                    if abs(joint_axis[2]) == 1:
                        joint_type = JointType.Z_ROT
                else:
                    log_error("Joint type not supported")
                if joint_axis[0] == -1 or joint_axis[1] == -1 or joint_axis[2] == -1:
                    joint_offset[0] = -1.0 * joint_offset[0]
                    joint_axis = [abs(x) for x in joint_axis]
                body_params["joint_type"] = joint_type
                body_params["joint_offset"] = joint_offset

        body_params["fixed_transform"] = joint_transform
        body_params["joint_name"] = active_joint_name

        body_params["joint_axis"] = joint_axis
        body_params["mimic_joint_name"] = mimic_joint_name

        link_params = LinkParams(**body_params)

        return link_params

    def add_absolute_path_to_link_meshes(self, mesh_dir: str = ""):
        """Add absolute path to link meshes.

        Args:
            mesh_dir: Absolute path to the directory containing link meshes.
        """
        # read all link meshes and update their mesh paths by prepending mesh_dir
        links = self._robot.link_map
        for k in links.keys():
            # read visual and collision
            vis = links[k].visuals
            for i in range(len(vis)):
                m = vis[i].geometry.mesh
                if m is not None:
                    m.filename = join_path(mesh_dir, m.filename)
            col = links[k].collisions
            for i in range(len(col)):
                m = col[i].geometry.mesh
                if m is not None:
                    m.filename = join_path(mesh_dir, m.filename)

    def get_urdf_string(self) -> str:
        """Get the contents of URDF as a string."""
        txt = etree.tostring(self._robot.write_xml(), method="xml", encoding="unicode")
        return txt

    def get_link_mesh(self, link_name: str) -> CuroboMesh:
        """Get mesh of a link.

        Args:
            link_name: Name of the link.

        Returns:
            Mesh: Mesh of the link.
        """

        link_data = self._robot.link_map[link_name]

        if len(link_data.visuals) == 0:
            log_error(link_name + " not found in urdf, remove from mesh_link_names")
        m = link_data.visuals[0].geometry.mesh
        mesh_pose = self._robot.link_map[link_name].visuals[0].origin
        # read visual material:
        if mesh_pose is None:
            mesh_pose = [0, 0, 0, 1, 0, 0, 0]
        else:
            # convert to list:
            mesh_pose = Pose.from_matrix(mesh_pose).to_list()

        return CuroboMesh(
            name=link_name,
            pose=mesh_pose,
            scale=m.scale,
            file_path=m.filename,
        )

    @property
    def root_link(self) -> str:
        """Returns the name of the base link of the robot.

        Only available when the URDF is loaded with build_scene_graph=True.

        Returns:
            str: Name of the base link.
        """
        return self._robot.base_link
