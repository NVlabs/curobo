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
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.autograd.profiler as profiler

from curobo._src.curobolib.cuda_ops.kinematics import KinematicsFusedFunction
from curobo._src.geom.types import tensor_sphere
from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg
from curobo._src.robot.parser.parser_urdf import UrdfRobotParser
from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.robot.types.joint_types import JointType
from curobo._src.robot.types.kinematics_params import KinematicsParams

# CuRobo
from curobo._src.robot.types.link_params import LinkParams
from curobo._src.robot.types.self_collision_params import (
    SelfCollisionKinematicsCfg,
)
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import append_joints_to_state
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise


class KinematicsLoader(KinematicsLoaderCfg):
    """Robot Kinematics Representation Generator.

    The word "Chain" is used interchangeably with "Tree" in this class.

    """

    def __init__(self, config: KinematicsLoaderCfg) -> None:
        """Initialize the robot generator.

        Args:
            config: Parameters to initialize the robot generator.
        """
        super().__init__(**vars(config))
        # if self.device_cfg.device != "cuda":
        #    log_and_raise(f"device_cfg.device is not cuda, but is {self.device_cfg.device}")

        self.cpu_tensor_args = DeviceCfg(device="cpu", dtype=torch.float32)

        self._self_collision_data = None
        self.non_fixed_joint_names = []
        self._num_dof = 1
        self._kinematics_config = None

        self.initialize_tensors()

    @property
    def kinematics_config(self) -> KinematicsParams:
        """Kinematics representation as Tensors."""
        return self._kinematics_config

    @property
    def self_collision_config(self) -> SelfCollisionKinematicsCfg:
        """Self collision configuration for robot."""
        return self._self_collision_data

    @property
    def kinematics_parser(self):
        """Kinematics parser used to generate robot parameters."""
        return self._kinematics_parser

    @profiler.record_function("robot_generator/initialize_tensors")
    def initialize_tensors(self):
        """Initialize tensors for kinematics representatiobn."""
        self._joint_limits = None
        self._self_collision_data = None
        self.lock_jointstate = None
        self.lin_jac, self.ang_jac = None, None

        self._link_spheres_tensor = torch.zeros(
            (1, 1, 4), device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        self._link_sphere_idx_map = torch.zeros(
            (1), dtype=torch.int16, device=self.device_cfg.device
        )
        self.total_spheres = 0

        # create a mega list of all links that we need:
        other_links = copy.deepcopy(self.tool_frames)

        for i in self.collision_link_names:
            if i not in self.tool_frames:
                other_links.append(i)
        for i in self.extra_links:
            p_name = self.extra_links[i].parent_link_name
            if p_name not in self.tool_frames and p_name not in other_links:
                other_links.append(p_name)

        # other_links = list(set(self.tool_frames + self.collision_link_names))

        self._kinematics_parser = UrdfRobotParser(
            self.urdf_path,
            mesh_root=self.asset_root_path,
            extra_links=self.extra_links,
            load_meshes=self.load_meshes,
        )

        if self.lock_joints is None:
            self._build_kinematics(self.base_link, other_links, self.tool_frames)
        else:
            self._build_kinematics_with_lock_joints(
                self.base_link, other_links, self.tool_frames, self.lock_joints
            )
        if self.cspace is None:
            jpv = self._get_joint_position_velocity_limits()
            self.cspace = CSpaceParams.load_from_joint_limits(
                jpv["position"][1, :], jpv["position"][0, :], self.joint_names, self.device_cfg
            )
        # we reduce cspace to only contain joints from current kinematic structure:

        # instead we lock joints that are not in the kinematic structure:
        extra_lock_joints = {
            k: self.cspace.default_joint_position[self.cspace.joint_names.index(k)].item()
            for k in self.cspace.joint_names
            if k not in self.joint_names
        }

        # add extra lock joints to lock_jointstate:
        # convert dictionary to list:
        if len(extra_lock_joints) > 0:
            extra_lock_joint_position = torch.tensor(
                [extra_lock_joints[k] for k in extra_lock_joints.keys()],
                device=self.device_cfg.device,
            )
            extra_lock_jointstate = JointState.from_position(
                extra_lock_joint_position, joint_names=list(extra_lock_joints.keys())
            )
            if self.lock_jointstate is not None:
                # only append if not already in lock_jointstate
                for ki, kv in enumerate(extra_lock_joints.keys()):
                    if kv not in self.lock_jointstate.joint_names:
                        self.lock_jointstate = append_joints_to_state(
                            self.lock_jointstate,
                            JointState.from_position(
                                extra_lock_joint_position[ki], joint_names=[kv]
                            ),
                        )

            else:
                self.non_fixed_joint_names += list(extra_lock_joints.keys())
                self.non_fixed_joint_names = list(set(self.non_fixed_joint_names))
                self.lock_jointstate = extra_lock_jointstate

        self.cspace.inplace_reindex(self.joint_names)
        self._update_joint_limits()

        # create kinematics tensor:
        self._kinematics_config = KinematicsParams(
            fixed_transforms=self._fixed_transform,
            link_map=self._link_map,
            joint_map=self._joint_map,
            joint_map_type=self._joint_map_type,
            joint_offset_map=self._joint_offset_map,
            tool_frame_map=self._tool_frame_map,
            link_chain_data=self._link_chain_data,
            link_chain_offsets=self._link_chain_offsets,
            joint_links_data=self._joint_links_data,
            joint_links_offsets=self._joint_links_offsets,
            joint_affects_endeffector=self._joint_affects_endeffector,
            tool_frames=self.tool_frames,
            link_spheres=self._link_spheres_tensor,
            link_sphere_idx_map=self._link_sphere_idx_map,
            num_dof=self._num_dof,
            joint_limits=self._joint_limits,
            non_fixed_joint_names=self.non_fixed_joint_names,
            total_spheres=self.total_spheres,
            link_name_to_idx_map=self._name_to_idx_map,
            joint_names=self.joint_names,
            debug=self.debug,
            mesh_link_names=self.mesh_link_names,
            cspace=self.cspace,
            base_link=self.base_link,
            lock_jointstate=self.lock_jointstate,
            mimic_joints=self._mimic_joint_data,
            link_masses_com=self._link_masses_com,
            link_inertias=self._link_inertias,
            device_cfg=self.device_cfg,
            grasp_contact_link_names=self.grasp_contact_link_names,
        )
        self._kinematics_config.make_contiguous()
        if self.asset_root_path is not None and self.asset_root_path != "":
            self._kinematics_parser.add_absolute_path_to_link_meshes(self.asset_root_path)

    def add_link(self, link_params: LinkParams):
        """Add an extra link to the robot kinematics tree.

        Args:
            link_params: Parameters of the link to add.
        """
        self.extra_links[link_params.link_name] = link_params

    def add_fixed_link(
        self,
        link_name: str,
        parent_link_name: str,
        joint_name: Optional[str] = None,
        transform: Optional[Pose] = None,
    ):
        """Add a fixed link to the robot kinematics tree.

        Args:
            link_name: Name of the link to add.
            parent_link_name: Parent link to add the fixed link to.
            joint_name: Name of fixed to joint to create.
            transform: Offset transform of the fixed link from the joint.
        """
        if transform is None:
            transform = Pose.from_list(
                [0, 0, 0, 1, 0, 0, 0], self.device_cfg
            ).get_numpy_affine_matrix()
        if joint_name is None:
            joint_name = link_name + "_j_" + parent_link_name
        link_params = LinkParams(
            link_name=link_name,
            parent_link_name=parent_link_name,
            joint_name=joint_name,
            fixed_transform=transform,
            joint_type=JointType.FIXED,
        )
        self.add_link(link_params)

    @profiler.record_function("robot_generator/build_chain")
    def _build_chain(
        self,
        base_link: str,
        other_links: List[str],
    ) -> List[str]:
        """Build kinematic tree of the robot.

        Args:
            base_link: Name of base link for the chain.
            other_links: List of other links to add to the chain.

        Returns:
            List[str]: List of link names in the chain.
        """
        self._num_dof = 0
        self._controlled_links = []
        self._bodies = []
        self._name_to_idx_map = dict()
        self.base_link = base_link
        self.joint_names = []
        self._fixed_transform = []
        self._link_masses_com = []
        self._link_inertias = []
        # Build chain to first link in tool_frames (primary end-effector)
        chain_link_names = self._kinematics_parser.get_chain(base_link, self.tool_frames[0])
        self._add_body_to_tree(chain_link_names[0], base=True)
        for i, l_name in enumerate(chain_link_names[1:]):
            self._add_body_to_tree(l_name)
        # check if all links are in the built tree:

        for i in other_links:
            if i in self._name_to_idx_map:
                continue
            if i not in self.extra_links.keys():
                chain_l_names = self._kinematics_parser.get_chain(base_link, i)

                for k in chain_l_names:
                    if k in chain_link_names:
                        continue
                    # if link name is not in chain, add to chain
                    chain_link_names.append(k)
                    # add to tree:
                    self._add_body_to_tree(k, base=False)
        for i in self.extra_links.keys():
            if i not in chain_link_names:
                self._add_body_to_tree(i, base=False)
                chain_link_names.append(i)

        self.non_fixed_joint_names = self.joint_names.copy()
        return chain_link_names

    def _get_mimic_joint_data(self) -> Dict[str, List[int]]:
        """Get joints that are mimicked from actuated joints joints.

        Returns:
            Dict[str, List[int]]: Dictionary containing name of actuated joint and list of mimic
                joint indices.
        """
        # get joint types:
        mimic_joint_data = {}
        for i in range(1, len(self._bodies)):
            body = self._bodies[i]
            if i in self._controlled_links:
                if body.mimic_joint_name is not None:
                    if body.joint_name not in mimic_joint_data:
                        mimic_joint_data[body.joint_name] = []
                    mimic_joint_data[body.joint_name].append(i)
        return mimic_joint_data

    def _precompute_joint_mappings(
        self, joint_map: List[int], ordered_link_names: List[str], base_link: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute joint-link mappings and joint-endeffector interaction matrix.

        This function precomputes which links are connected to each joint and creates
        a joint-endeffector interaction matrix for efficient jacobian computation.

        Args:
            joint_map: Mapping from link indices to joint indices
            ordered_link_names: List of ordered link names for endeffector computation
            base_link: Base link name for getting kinematic chains

        Returns:
            Tuple containing:
                - joint_links_data_tensor: Flat tensor of link indices for each joint
                - joint_links_offsets_tensor: Offset tensor for each joint's connected links
                - joint_affects_endeffector_tensor: Boolean tensor for joint-endeffector interactions
        """
        # Precompute which links are connected to each joint for efficient jacobian
        # computation. This eliminates the need to search through all links in the
        # kinematic chain for each joint
        joint_links_data = []  # Flat list of link indices for each joint
        joint_links_offsets = [0]  # Offset for each joint's connected links

        for joint_idx in range(len(self.joint_names)):
            # Find all links connected to this joint
            connected_links = []
            for link_idx in range(len(self._bodies)):
                if joint_map[link_idx] == joint_idx:
                    connected_links.append(link_idx)

            # Store connected links in flat array
            joint_links_data.extend(connected_links)
            joint_links_offsets.append(len(joint_links_data))

        # Convert joint-link mapping to tensors
        joint_links_data_tensor = torch.tensor(
            joint_links_data, dtype=torch.int16, device=self.cpu_tensor_args.device
        )
        joint_links_offsets_tensor = torch.tensor(
            joint_links_offsets, dtype=torch.int16, device=self.cpu_tensor_args.device
        )

        # Precompute joint-endeffector interaction matrix for kernel optimization
        # This eliminates nested loop searches in both forward and backward kernels
        joint_affects_endeffector_data = []  # Flat boolean array
        for joint_idx in range(len(self.joint_names)):
            for ee_idx in range(len(ordered_link_names)):
                ee_link_name = ordered_link_names[ee_idx]
                ee_chain = self._kinematics_parser.get_chain(base_link, ee_link_name)

                # Check if this joint affects this end-effector
                joint_affects_ee = False

                # Get connected links for this joint
                joint_start = joint_links_offsets[joint_idx]
                joint_end = joint_links_offsets[joint_idx + 1]

                for jl_idx in range(joint_start, joint_end):
                    joint_link_idx = joint_links_data[jl_idx]
                    joint_link_name = None

                    # Find the link name from link index
                    for link_name, link_idx in self._name_to_idx_map.items():
                        if link_idx == joint_link_idx:
                            joint_link_name = link_name
                            break

                    if joint_link_name and joint_link_name in ee_chain:
                        joint_affects_ee = True
                        break

                joint_affects_endeffector_data.append(joint_affects_ee)

        # Convert to tensor [njoints * n_tool_frames] - flattened row-major
        joint_affects_endeffector_tensor = torch.tensor(
            joint_affects_endeffector_data, dtype=torch.bool, device=self.cpu_tensor_args.device
        )

        return joint_links_data_tensor, joint_links_offsets_tensor, joint_affects_endeffector_tensor

    @profiler.record_function("robot_generator/build_kinematics_tensors")
    def _build_kinematics_tensors(self, base_link, tool_frames, chain_link_names):
        """Create kinematic tensors for robot given kinematic tree.

        Args:
            base_link: Name of base link for the tree.
            tool_frames: Name of tool frames to compute kinematics for. This is used to determine
                link indices to store pose during forward kinematics.
            chain_link_names: List of link names in the kinematic tree. Used to traverse the
                kinematic tree.
        """
        self._active_joints = []
        self._mimic_joint_data = {}
        link_map = [0 for i in range(len(self._bodies))]
        tool_frame_map = []  # [-1 for i in range(len(self._bodies))]

        joint_map = [
            -1 if i not in self._controlled_links else i for i in range(len(self._bodies))
        ]  #
        joint_map_type = [
            -1 if i not in self._controlled_links else i for i in range(len(self._bodies))
        ]
        all_joint_names = []
        ordered_link_names = []
        joint_offset_map = [[1.0, 0.0]]
        # add body 0 details:
        if self._bodies[0].link_name in tool_frames:
            tool_frame_map.append(chain_link_names.index(self._bodies[0].link_name))
            ordered_link_names.append(self._bodies[0].link_name)
        # get joint types:
        for i in range(1, len(self._bodies)):
            body = self._bodies[i]
            parent_name = body.parent_link_name
            link_map[i] = self._name_to_idx_map[parent_name]
            joint_offset_map.append(body.joint_offset)
            joint_map_type[i] = body.joint_type.value
            if body.link_name in tool_frames:
                tool_frame_map.append(chain_link_names.index(body.link_name))
                ordered_link_names.append(body.link_name)
            if body.joint_name not in all_joint_names:
                all_joint_names.append(body.joint_name)
            if i in self._controlled_links:
                joint_map[i] = self.joint_names.index(body.joint_name)
                if body.mimic_joint_name is not None:
                    if body.joint_name not in self._mimic_joint_data:
                        self._mimic_joint_data[body.joint_name] = []
                    self._mimic_joint_data[body.joint_name].append(
                        {"joint_offset": body.joint_offset, "joint_name": body.mimic_joint_name}
                    )
                else:
                    self._active_joints.append(i)
        self.tool_frames = ordered_link_names
        # NEW: Precompute actual link indices that affect each link's gradient
        all_chain_indices = []  # Flat list of all actual link indices
        chain_offsets = [0]  # Offset for each link's chain

        # iterate and set true:
        for i in range(len(chain_link_names)):
            chain_l_names = self._kinematics_parser.get_chain(base_link, chain_link_names[i])

            # CORRECTED: Convert chain link names to ACTUAL link indices using _name_to_idx_map
            actual_link_indices = [self._name_to_idx_map[link_name] for link_name in chain_l_names]

            # Store actual link indices in flat array
            all_chain_indices.extend(actual_link_indices)
            chain_offsets.append(len(all_chain_indices))

        # Convert chain data to tensors
        link_chain_data = torch.tensor(
            all_chain_indices, dtype=torch.int16, device=self.cpu_tensor_args.device
        )
        link_chain_offsets = torch.tensor(
            chain_offsets, dtype=torch.int16, device=self.cpu_tensor_args.device
        )

        # NEW: Precompute which links are connected to each joint for efficient jacobian
        joint_links_data_tensor, joint_links_offsets_tensor, joint_affects_endeffector_tensor = (
            self._precompute_joint_mappings(joint_map, ordered_link_names, base_link)
        )

        # Store precomputed joint-endeffector interaction matrix
        self._joint_affects_endeffector = joint_affects_endeffector_tensor.to(
            device=self.device_cfg.device
        )

        self._link_map = torch.as_tensor(
            link_map, device=self.device_cfg.device, dtype=torch.int16
        )
        self._joint_map = torch.as_tensor(
            joint_map, device=self.device_cfg.device, dtype=torch.int16
        )
        self._joint_map_type = torch.as_tensor(
            joint_map_type, device=self.device_cfg.device, dtype=torch.int8
        )
        self._tool_frame_map = torch.as_tensor(
            tool_frame_map, device=self.device_cfg.device, dtype=torch.int16
        )
        self._joint_offset_map = torch.as_tensor(
            joint_offset_map, device=self.device_cfg.device, dtype=torch.float32
        )
        self._joint_offset_map = self._joint_offset_map.view(-1).contiguous()

        # Store precomputed chain data with actual link indices
        self._link_chain_data = link_chain_data.to(device=self.device_cfg.device)
        self._link_chain_offsets = link_chain_offsets.to(device=self.device_cfg.device)

        # Store precomputed joint-link mapping data
        self._joint_links_data = joint_links_data_tensor.to(device=self.device_cfg.device)
        self._joint_links_offsets = joint_links_offsets_tensor.to(device=self.device_cfg.device)

        self._fixed_transform = torch.cat((self._fixed_transform), dim=0).to(
            device=self.device_cfg.device
        )
        self._link_inertias = torch.cat((self._link_inertias), dim=0).to(
            device=self.device_cfg.device
        )
        self._link_masses_com = torch.cat((self._link_masses_com), dim=0).to(
            device=self.device_cfg.device
        )
        self._all_joint_names = all_joint_names

    @profiler.record_function("robot_generator/build_kinematics")
    def _build_kinematics(
        self, base_link: str, other_links: List[str], tool_frames: List[str]
    ):
        """Build kinematics tensors given base link and other links.

        Args:
            base_link: Name of base link for the kinematic tree.
            other_links: List of other links to add to the kinematic tree.
            tool_frames: List of tool frames to store poses after kinematics computation.
        """
        chain_link_names = self._build_chain(base_link, other_links)
        self._build_kinematics_tensors(base_link, tool_frames, chain_link_names)
        if self.collision_spheres is not None and len(self.collision_link_names) > 0:
            self._build_collision_model(
                self.collision_spheres,
                self.collision_link_names,
                self.collision_sphere_buffer,
            )

    @profiler.record_function("robot_generator/build_kinematics_with_lock_joints")
    def _build_kinematics_with_lock_joints(
        self,
        base_link: str,
        other_links: List[str],
        tool_frames: List[str],
        lock_joints: Dict[str, float],
    ):
        """Build kinematics with locked joints.

        This function will first build the chain with no locked joints, find the transforms
        when the locked joints are set to the given values, and then use these transforms as
        fixed transforms for the locked joints.

        Args:
            base_link: Base link of the kinematic tree.
            other_links: Other links to add to the kinematic tree.
            tool_frames: List of tool frames to store poses after kinematics computation.
            lock_joints: Joints to lock in the kinematic tree with value to lock at.
        """
        chain_link_names = self._build_chain(base_link, other_links)
        # find links attached to lock joints:
        lock_joint_names = list(lock_joints.keys())

        joint_data = self._get_joint_links(lock_joint_names)

        lock_links = list(
            [joint_data[j]["parent"] for j in joint_data.keys()]
            + [joint_data[j]["child"] for j in joint_data.keys()]
        )

        for k in lock_joint_names:
            if "mimic" in joint_data[k]:
                mimic_link_names = [[x["parent"], x["child"]] for x in joint_data[k]["mimic"]]
                mimic_link_names = [x for xs in mimic_link_names for x in xs]
                lock_links += mimic_link_names
        lock_links = list(set(lock_links))

        new_tool_frames = list(set(tool_frames + lock_links))

        # rebuild kinematic tree with tool frames added to link pose computation:
        self._build_kinematics_tensors(base_link, new_tool_frames, chain_link_names)
        if self.collision_spheres is not None and len(self.collision_link_names) > 0:
            self._build_collision_model(
                self.collision_spheres,
                self.collision_link_names,
                self.collision_sphere_buffer,
            )
        # do forward kinematics and get transform for locked joints:
        q = torch.zeros(
            (1, self._num_dof), device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        # set lock joints in the joint angles:
        l_idx = torch.as_tensor(
            [self.joint_names.index(l) for l in lock_joints.keys()],
            dtype=torch.long,
            device=self.device_cfg.device,
        )
        l_val = self.device_cfg.to_device([lock_joints[l] for l in lock_joints.keys()])

        q[0, l_idx] = l_val
        kinematics_config = KinematicsParams(
            device_cfg=self.device_cfg,
            fixed_transforms=self._fixed_transform,
            link_masses_com=self._link_masses_com,
            link_inertias=self._link_inertias,
            link_map=self._link_map,
            joint_map=self._joint_map,
            joint_map_type=self._joint_map_type,
            joint_offset_map=self._joint_offset_map,
            tool_frame_map=self._tool_frame_map,
            link_chain_data=self._link_chain_data,
            link_chain_offsets=self._link_chain_offsets,
            joint_links_data=self._joint_links_data,
            joint_links_offsets=self._joint_links_offsets,
            joint_affects_endeffector=self._joint_affects_endeffector,
            tool_frames=self.tool_frames,
            link_spheres=self._link_spheres_tensor,
            link_sphere_idx_map=self._link_sphere_idx_map,
            num_dof=self._num_dof,
            joint_limits=self._joint_limits,
            non_fixed_joint_names=self.non_fixed_joint_names,
            total_spheres=self.total_spheres,
            grasp_contact_link_names=self.grasp_contact_link_names,
        )

        link_poses = self._get_link_poses(q, lock_links, kinematics_config)
        # remove lock links from store map:
        tool_frame_map = [chain_link_names.index(l) for l in tool_frames]
        self._tool_frame_map = torch.as_tensor(
            tool_frame_map, device=self.device_cfg.device, dtype=torch.int16
        )
        self.tool_frames = tool_frames
        # compute a fixed transform for fixing joints:
        with profiler.record_function("cuda_robot_generator/fix_locked_joints"):
            # convert tensors to cpu:
            self._joint_map_type = self._joint_map_type.to(device=self.cpu_tensor_args.device)
            self._joint_map = self._joint_map.to(device=self.cpu_tensor_args.device)

            for j in lock_joint_names:
                w_parent = lock_links.index(joint_data[j]["parent"])
                w_child = lock_links.index(joint_data[j]["child"])
                parent_t_child = (
                    link_poses.get_index(0, w_parent)
                    .inverse()
                    .multiply(link_poses.get_index(0, w_child))
                )
                # Make this joint as fixed
                i = joint_data[j]["link_index"]
                self._fixed_transform[i] = parent_t_child.get_affine_matrix()

                if "mimic" in joint_data[j]:
                    for mimic_joint in joint_data[j]["mimic"]:
                        w_parent = lock_links.index(mimic_joint["parent"])
                        w_child = lock_links.index(mimic_joint["child"])
                        parent_t_child = (
                            link_poses.get_index(0, w_parent)
                            .inverse()
                            .multiply(link_poses.get_index(0, w_child))
                        )
                        i_q = mimic_joint["link_index"]
                        self._fixed_transform[i_q] = parent_t_child.get_affine_matrix()
                        self._controlled_links.remove(i_q)
                        self._joint_map_type[i_q] = -1
                        self._joint_map[i_q] = -1

                i = joint_data[j]["link_index"]
                self._joint_map_type[i] = -1
                self._joint_map[i:] -= 1
                self._joint_map[i] = -1
                self._controlled_links.remove(i)
                self.joint_names.remove(j)
                self._num_dof -= 1
                self._active_joints.remove(i)
            self._joint_map[self._joint_map < -1] = -1
            self._joint_map = self._joint_map.to(device=self.device_cfg.device)
            self._joint_map_type = self._joint_map_type.to(device=self.device_cfg.device)
        # recompute joint mappings as some joints are now fixed:
        joint_links_data_tensor, joint_links_offsets_tensor, joint_affects_endeffector_tensor = (
            self._precompute_joint_mappings(self._joint_map, self.tool_frames, base_link)
        )
        self._joint_links_data = torch.as_tensor(
            joint_links_data_tensor, device=self.device_cfg.device
        )
        self._joint_links_offsets = torch.as_tensor(
            joint_links_offsets_tensor, device=self.device_cfg.device
        )
        self._joint_affects_endeffector = torch.as_tensor(
            joint_affects_endeffector_tensor, device=self.device_cfg.device
        )

        if len(self.lock_joints.keys()) > 0:
            self.lock_jointstate = JointState(
                position=l_val, joint_names=list(self.lock_joints.keys())
            )

    @profiler.record_function("robot_generator/build_collision_model")
    def _build_collision_model(
        self,
        collision_spheres: Dict,
        collision_link_names: List[str],
        collision_sphere_buffer: Union[float, Dict[str, float]] = 0.0,
    ):
        """Build collision model for robot.

        Args:
            collision_spheres: Spheres for each link of the robot.
            collision_link_names: Name of links to load spheres for.
            collision_sphere_buffer: Additional padding to add to collision spheres.
        """
        # We create all tensors on cpu and then finally move them to gpu
        coll_link_spheres = []
        # we store as [n_link, 7]
        link_sphere_idx_map = []
        cpu_tensor_args = DeviceCfg(device="cpu", dtype=torch.float32)
        self_collision_buffer = self.self_collision_buffer.copy()

        with profiler.record_function("robot_generator/build_collision_spheres"):
            for j_idx, j in enumerate(collision_link_names):
                num_spheres = len(collision_spheres[j])
                link_spheres = torch.zeros(
                    (num_spheres, 4), dtype=cpu_tensor_args.dtype, device=cpu_tensor_args.device
                )
                # find link index in global map:
                l_idx = self._name_to_idx_map[j]
                offset_radius = 0.0
                if isinstance(collision_sphere_buffer, float):
                    offset_radius = collision_sphere_buffer
                elif j in collision_sphere_buffer:
                    offset_radius = collision_sphere_buffer[j]
                if j in self_collision_buffer:
                    self_collision_buffer[j] -= offset_radius
                else:
                    self_collision_buffer[j] = -offset_radius
                for i in range(num_spheres):
                    padded_radius = collision_spheres[j][i]["radius"] + offset_radius
                    # if padded_radius <= 0.0 and padded_radius > -1.0:
                    #    padded_radius = 0.001
                    link_spheres[i, :] = tensor_sphere(
                        collision_spheres[j][i]["center"],
                        padded_radius,
                        device_cfg=cpu_tensor_args,
                        tensor=link_spheres[i, :],
                    )
                    link_sphere_idx_map.append(l_idx)
                coll_link_spheres.append(link_spheres)
                self.total_spheres += num_spheres

        self._link_spheres_tensor = torch.cat(coll_link_spheres, dim=0).unsqueeze(0)
        self._link_spheres_tensor = self._link_spheres_tensor.repeat(self.num_envs, 1, 1)
        self._link_sphere_idx_map = torch.as_tensor(
            link_sphere_idx_map, dtype=torch.int16, device=cpu_tensor_args.device
        )
        self._link_sphere_idx_map = self._link_sphere_idx_map.to(device=self.device_cfg.device)
        self._link_spheres_tensor = self._link_spheres_tensor.to(device=self.device_cfg.device)

        self._self_collision_data = SelfCollisionKinematicsCfg.create_from_link_pairs(
            collision_link_names=collision_link_names,
            link_name_to_sphere_index=self._name_to_idx_map,
            self_collision_link_pair_ignores=self.self_collision_ignore,
            self_collision_link_padding=self.self_collision_buffer,
            all_link_spheres=self._link_spheres_tensor[0],
            link_index_to_sphere_index=self._link_sphere_idx_map,
            device_cfg=self.device_cfg,
        )

    @profiler.record_function("robot_generator/add_body_to_tree")
    def _add_body_to_tree(self, link_name: str, base=False):
        """Add link to kinematic tree.

        Args:
            link_name: Name of the link to add.
            base: Is this the base link of the kinematic tree?
        """
        body_idx = len(self._bodies)

        rigid_body_params = self._kinematics_parser.get_link_parameters(link_name, base=base)
        self._bodies.append(rigid_body_params)
        if rigid_body_params.joint_type != JointType.FIXED:
            self._controlled_links.append(body_idx)
            if rigid_body_params.joint_name not in self.joint_names:
                self.joint_names.append(rigid_body_params.joint_name)
                self._num_dof = self._num_dof + 1
        self._fixed_transform.append(
            torch.as_tensor(
                rigid_body_params.fixed_transform,
                device=self.cpu_tensor_args.device,
                dtype=self.cpu_tensor_args.dtype,
            ).unsqueeze(0)
        )
        self._link_masses_com.append(
            torch.as_tensor(
                rigid_body_params.get_link_com_and_mass(),
                device=self.cpu_tensor_args.device,
                dtype=self.cpu_tensor_args.dtype,
            ).unsqueeze(0)
        )
        # Pad inertia from 6 to 8 floats for float4 alignment in CUDA kernels
        inertia_6 = torch.as_tensor(
            rigid_body_params.link_inertia,
            device=self.cpu_tensor_args.device,
            dtype=self.cpu_tensor_args.dtype,
        )
        # Layout: [ixx, iyy, izz, ixy, ixz, iyz, 0, 0]
        inertia_8 = torch.zeros(8, device=self.cpu_tensor_args.device, dtype=self.cpu_tensor_args.dtype)
        inertia_8[:6] = inertia_6
        self._link_inertias.append(inertia_8.unsqueeze(0))
        self._name_to_idx_map[rigid_body_params.link_name] = body_idx

    def _get_joint_links(self, joint_names: List[str]) -> Dict[str, Dict[str, Union[str, int]]]:
        """Get data (parent link, child link, mimic, link_index) for joints given in the list.

        Args:
            joint_names: Names of joints to get data for.

        Returns:
            Dict[str, Dict[str, Union[str, int]]]: Dictionary containing joint name as key and
                dictionary containing parent link, child link, and link index as
                values. Also includes mimic joint data if present.
        """
        j_data = {}

        for j in joint_names:
            for bi, b in enumerate(self._bodies):
                if b.joint_name == j:
                    if j not in j_data:
                        j_data[j] = {}
                    if b.mimic_joint_name is None:
                        j_data[j]["parent"] = b.parent_link_name
                        j_data[j]["child"] = b.link_name
                        j_data[j]["link_index"] = bi
                    else:
                        if "mimic" not in j_data[j]:
                            j_data[j]["mimic"] = []
                        j_data[j]["mimic"].append(
                            {
                                "parent": b.parent_link_name,
                                "child": b.link_name,
                                "link_index": bi,
                                "joint_offset": b.joint_offset,
                            }
                        )

        return j_data

    @profiler.record_function("robot_generator/get_link_poses")
    def _get_link_poses(
        self, q: torch.Tensor, query_link_names: List[str], kinematics_config: KinematicsParams
    ) -> Pose:
        """Get Pose of links at given joint angles using forward kinematics.

        This is implemented here to avoid circular dependencies with
        :class:`~curobo.cuda_robot_model.cuda_robot_model.Kinematics` module. This is used
        to calculate fixed transforms for locked joints in this class. This implementation
        does not compute position of robot spheres.

        Args:
            q: Joint angles to compute forward kinematics for.
            query_link_names: Name of links to return pose.
            kinematics_config: Tensor Configuration for kinematics computation.

        Returns:
            Pose: Pose of links at given joint angles.
        """
        q = q.to(device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        q = q.view(1, 1, -1)
        if not q.is_cuda:
            log_and_raise(f"q is not on cuda: {self.device_cfg.device}")
        buffers = KinematicsFusedFunction.create_buffers(
            1,
            1,
            kinematics_config,
            self.device_cfg,
        )
        env_query_idx = torch.zeros(
            (1,), dtype=torch.int32, device=self.device_cfg.device,
        )
        link_pos_seq, link_quat_seq, _, _, _ = KinematicsFusedFunction.apply(
            q,
            buffers["batch_link_position"],
            buffers["batch_link_quaternion"],
            buffers["batch_robot_spheres"],
            buffers["batch_com"],
            buffers["batch_jacobian"],
            buffers["batch_cumul_mat"],
            kinematics_config,
            buffers["grad_out_q"],
            buffers["grad_out_q_jacobian"],
            buffers["grad_in_link_pos"],
            buffers["grad_in_link_quat"],
            buffers["grad_in_robot_spheres"],
            buffers["grad_in_com"],
            False,
            False,
            False,
            env_query_idx,
            1,
        )
        link_pos_seq = link_pos_seq[:, 0]
        link_quat_seq = link_quat_seq[:, 0]
        position = torch.zeros(
            (1, len(query_link_names), 3),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        quaternion = torch.zeros(
            (1, len(query_link_names), 4),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )

        for li, l in enumerate(query_link_names):
            i = self.tool_frames.index(l)
            position[:, li, :] = link_pos_seq[:, i, :]
            quaternion[:, li, :] = link_quat_seq[:, i, :]
        return Pose(position=position.clone(), quaternion=quaternion.clone())

    def get_joint_limits(self) -> JointLimits:
        """Get joint limits for the robot."""
        return self._joint_limits

    @profiler.record_function("robot_generator/get_joint_limits")
    def _get_joint_position_velocity_limits(self) -> Dict[str, torch.Tensor]:
        """Compute joint position and velocity limits for the robot.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing position and velocity limits for the
                robot. Each value is a tensor of shape (2, n_joints) with first row containing
                minimum limits and second row containing maximum limits.
        """
        joint_limits = {
            "position": [[], []],
            "velocity": [[], []],
            "effort": [[], []],
        }

        for idx in self._active_joints:
            joint_limits["position"][0].append(self._bodies[idx].joint_limits[0])
            joint_limits["position"][1].append(self._bodies[idx].joint_limits[1])
            joint_limits["velocity"][0].append(self._bodies[idx].joint_velocity_limits[0])
            joint_limits["velocity"][1].append(self._bodies[idx].joint_velocity_limits[1])
            joint_limits["effort"][0].append(-1.0 * self._bodies[idx].joint_effort_limit[0])
            joint_limits["effort"][1].append(self._bodies[idx].joint_effort_limit[0])

        for k in joint_limits:
            joint_limits[k] = torch.as_tensor(
                joint_limits[k], device=self.device_cfg.device, dtype=self.device_cfg.dtype
            )
        return joint_limits

    @profiler.record_function("robot_generator/update_joint_limits")
    def _update_joint_limits(self):
        """Update limits from CSpaceParams (acceleration, jerk limits and position clips)."""
        joint_limits = self._get_joint_position_velocity_limits()
        joint_limits["jerk"] = torch.cat(
            [-1.0 * self.cspace.max_jerk.unsqueeze(0), self.cspace.max_jerk.unsqueeze(0)]
        )
        joint_limits["acceleration"] = torch.cat(
            [
                -1.0 * self.cspace.max_acceleration.unsqueeze(0),
                self.cspace.max_acceleration.unsqueeze(0),
            ]
        )
        if len(joint_limits["velocity"][0]) != len(self.cspace.velocity_scale):
            log_and_raise(
                f"velocity scale length does not match number of joints: {len(joint_limits['velocity'][0])} != {len(self.cspace.velocity_scale)}"
            )
        # clip joint position:
        joint_limits["position"][0] += self.cspace.position_limit_clip
        joint_limits["position"][1] -= self.cspace.position_limit_clip
        joint_limits["velocity"][0] *= self.cspace.velocity_scale
        joint_limits["velocity"][1] *= self.cspace.velocity_scale

        self._joint_limits = JointLimits(joint_names=self.joint_names, **joint_limits)
