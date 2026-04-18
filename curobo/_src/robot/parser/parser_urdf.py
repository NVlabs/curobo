# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Parses Kinematics from an `URDF <https://wiki.ros.org/urdf>`__ file that describes the
kinematic tree of a robot.
"""

# Standard Library
from typing import Dict, List, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
import yourdfpy
from lxml import etree

from curobo._src.geom.types import Cuboid, Cylinder, Obstacle, Sphere
from curobo._src.geom.types import Mesh as CuroboMesh

# CuRobo
from curobo._src.robot.parser.parser_base import RobotParser
from curobo._src.robot.types.joint_types import JointType
from curobo._src.robot.types.link_params import LinkParams
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util_file import join_path


class UrdfRobotParser(RobotParser):
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
        self._mesh_root = mesh_root

        self._robot = yourdfpy.URDF.load(
            urdf_path,
            load_meshes=load_meshes,
            build_scene_graph=build_scene_graph,
            filename_handler=self._file_name_handler,
        )

        super().__init__(extra_links)

    def _file_name_handler(self, fname: str) -> str:
        # This will also remove package:// from the file name:
        fname = fname.replace("package://", "")
        return join_path(self._mesh_root, fname)

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
            log_warn(
                "Converting continuous joint: "
                + joint.name
                + " to revolute with limits [-6.28,6.28]"
            )
            joint_type = "revolute"
            joint_limits = {
                "effort": joint.limit.effort,
                "lower": -3.14 * 2,
                "upper": 3.14 * 2,
                "velocity": joint.limit.velocity,
            }

        if joint_type != "fixed":
            if joint_limits["velocity"] is None:
                log_warn(f"Joint {joint.name} has no velocity limit, setting to 100.0")
                joint_limits["velocity"] = 100.0
            if joint_limits["effort"] is None:
                log_warn(f"Joint {joint.name} has no effort limit, setting to 100.0")
                joint_limits["effort"] = 100.0
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
        mass = 0.01  # small mass
        com = np.array([0.0, 0.0, 0.0])
        inertia = np.array([1e-2, 1e-2, 1e-2, 0.0, 0.0, 0.0]) * mass
        # read link mass, com, and inertia:
        link_data = self._robot.link_map[link_name]
        if link_data.inertial is not None:
            mass_value = link_data.inertial.mass
            mass = mass_value if mass_value > 0.0 else 0.01
            com_pose_matrix = link_data.inertial.origin
            if com_pose_matrix is None:
                com_pose_matrix = np.eye(4)
            # com pose is rigidly attached to link, so any rotation that com has can be
            # applied to the position to get an absolute position (with identity rotation).
            com_rotation_as_pose = Pose.from_matrix(
                torch.as_tensor(com_pose_matrix, device="cuda", dtype=torch.float32)
            )

            com_position_as_pose = Pose.from_numpy(
                com_pose_matrix[:3, 3],
                np.ravel([1, 0, 0, 0]),
                device_cfg=DeviceCfg(device="cuda", dtype=torch.float32),
            )
            com = com_rotation_as_pose.multiply(com_position_as_pose).position.view(3).cpu().numpy()

            inertia_matrix = link_data.inertial.inertia
            inertia_array = np.array(
                [
                    inertia_matrix[0, 0],
                    inertia_matrix[1, 1],
                    inertia_matrix[2, 2],
                    inertia_matrix[0, 1],
                    inertia_matrix[0, 2],
                    inertia_matrix[1, 2],
                ]
            )
            # if all are zero:
            if np.all(inertia_array == 0.0):
                inertia_array = np.array([1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0]) * mass
            if com.shape != (3,):
                log_and_raise(f"com read from urdf shape does not match: {com.shape} != (3,)")
            if inertia.shape != (6,):
                log_and_raise(f"inertia read from urdf shape does not match: {inertia.shape} != (6,)")

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
                        new_value = active_joint_limits["lower"] * joint_offset[0] + joint_offset[1]
                        log_warn(
                            "mimic joint can go out of it's lower limit as active joint has larger range "
                            + "FIX: make mimic joint ["
                            + mimic_joint_name
                            + "] lower limit "
                            + "less than "
                            + str(new_value)
                        )
                        joint_limits["lower"] = new_value - 1e-3

                    if (
                        active_joint_limits["upper"] * joint_offset[0] + joint_offset[1]
                        > joint_limits["upper"]
                    ):
                        new_value = active_joint_limits["upper"] * joint_offset[0] + joint_offset[1]
                        log_warn(
                            "mimic joint can go out of it's upper limit as active joint has larger range "
                            + "FIX: make mimic joint [ "
                            + mimic_joint_name
                            + "] lower limit "
                            + "greater than "
                            + str(new_value)
                        )
                        joint_limits["upper"] = new_value + 1e-3

                    if active_joint_limits["velocity"] * joint_offset[0] > joint_limits["velocity"]:
                        new_value = active_joint_limits["velocity"] * joint_offset[0]
                        log_warn(
                            "mimic joint can move at higher velocity than active joint,"
                            + "increase velocity limit for mimic joint ["
                            + mimic_joint_name
                            + "] to "
                            + str(new_value)
                        )
                        joint_limits["velocity"] = new_value

                    joint_limits = active_joint_limits

                joint_axis = joint.axis

                body_params["joint_limits"] = [joint_limits["lower"], joint_limits["upper"]]
                body_params["joint_velocity_limits"] = [
                    -1.0 * joint_limits["velocity"],
                    joint_limits["velocity"],
                ]
                body_params["joint_effort_limit"] = [joint_limits["effort"]]
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
                    log_and_raise("Joint type not supported")
                if joint_axis[0] == -1 or joint_axis[1] == -1 or joint_axis[2] == -1:
                    joint_offset[0] = -1.0 * joint_offset[0]
                    joint_axis = [abs(x) for x in joint_axis]
                body_params["joint_type"] = joint_type
                body_params["joint_offset"] = joint_offset

        body_params["fixed_transform"] = joint_transform[:3, :4]
        body_params["joint_name"] = active_joint_name

        body_params["joint_axis"] = joint_axis
        body_params["mimic_joint_name"] = mimic_joint_name
        body_params["link_mass"] = mass
        body_params["link_com"] = com
        body_params["link_inertia"] = inertia

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
    def get_link_geometry(self, link_name: str,
                          use_collision_mesh: bool = False) -> List[Obstacle]:
        """Get geometry of a link.

        Args:
            link_name: Name of the link.

        Returns:
            List[Obstacle]: List of geometry of the link.
        """
        geometry_list = []
        link_data = self._robot.link_map[link_name]
        if use_collision_mesh:
            link_geometry = link_data.collisions
        else:
            link_geometry = link_data.visuals

        if len(link_geometry) == 0:
            return []

        for l_element in link_geometry:
            geometry = l_element.geometry
            origin = l_element.origin
            if origin is None:
                origin = [0, 0, 0, 1, 0, 0, 0]
            else:
                origin = Pose.from_matrix(origin).to_list()
            name = l_element.name

            # check what type of geometry it is
            if geometry.box is not None:
                geometry_list.append(Cuboid(name=name, pose=origin, dims=geometry.box.size))
            elif geometry.sphere is not None:
                geometry_list.append(Sphere(name=name, pose=origin, radius=geometry.sphere.radius))
            elif geometry.cylinder is not None:
                geometry_list.append(Cylinder(name=name, pose=origin, radius=geometry.cylinder.radius, height=geometry.cylinder.length))
            elif geometry.mesh is not None:
                local_path = geometry.mesh.filename
                absolute_path = self._file_name_handler(local_path)
                geometry_list.append(CuroboMesh(name=name, pose=origin, file_path=absolute_path))
            else:
                continue
        return geometry_list
    def get_link_mesh(
        self, link_name: str, use_collision_mesh: bool = False
    ) -> Union[CuroboMesh, None]:
        """Get mesh of a link.

        Args:
            link_name: Name of the link.

        Returns:
            Mesh: Mesh of the link.
        """
        link_data = self._robot.link_map[link_name]

        if use_collision_mesh:
            link_geometry = link_data.collisions
        else:
            link_geometry = link_data.visuals

        if len(link_geometry) == 0:
            return None

        # merge meshes from all geometries:
        meshes = []
        poses = []
        for geometry in link_geometry:
            m = geometry.geometry.mesh
            if m is not None:
                meshes.append(m)
                if geometry.origin is None:
                    poses.append([0, 0, 0, 1, 0, 0, 0])
                else:
                    poses.append(geometry.origin)
        if len(meshes) == 0:
            return None

        # Use the first mesh and its pose
        m = meshes[0]
        mesh_pose = poses[0]

        # Convert pose to list format if it's a matrix
        if isinstance(mesh_pose, list):
            pass  # Already a list
        else:
            # convert from matrix to list:
            mesh_pose = Pose.from_matrix(mesh_pose).to_list()

        local_path = m.filename
        absolute_path = self._file_name_handler(local_path)
        return CuroboMesh(
            name=link_name,
            pose=mesh_pose,
            scale=m.scale,
            file_path=absolute_path,
        )

    @property
    def root_link(self) -> str:
        """Returns the name of the base link of the robot.

        Only available when the URDF is loaded with build_scene_graph=True.

        Returns:
            str: Name of the base link.
        """
        return self._robot.base_link

    def get_link_names_from_urdf(self) -> List[str]:
        """Get names of all links in the kinematic tree from the URDF.

        Returns:
            List[str]: List of link names.
        """
        return list(self._robot.link_map.keys())

    def get_joint_names_from_urdf(self) -> List[str]:
        """Get names of all joints in the kinematic tree from the URDF.

        Returns:
            List[str]: List of joint names.
        """
        # exclude
        return list(self._robot.joint_map.keys())
