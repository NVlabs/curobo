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
Generates a Tensor representation of kinematics for use in
:class:`~curobo.cuda_robot_model.CudaRobotModel`. This module reads the robot from a
:class:`~curobo.cuda_robot_model.kinematics_parser.KinematicsParser` and
generates the necessary tensors for kinematics computation.

"""

from __future__ import annotations

# Standard Library
import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.cuda_robot_model.kinematics_parser import LinkParams
from curobo.cuda_robot_model.types import (
    CSpaceConfig,
    JointLimits,
    JointType,
    KinematicsTensorConfig,
    SelfCollisionKinematicsConfig,
)
from curobo.cuda_robot_model.urdf_kinematics_parser import UrdfKinematicsParser
from curobo.curobolib.kinematics import get_cuda_kinematics
from curobo.geom.types import tensor_sphere
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml

try:
    # CuRobo
    from curobo.cuda_robot_model.usd_kinematics_parser import UsdKinematicsParser
except ImportError:
    log_info(
        "USDParser failed to import, install curobo with pip install .[usd] "
        + "or pip install usd-core, NOTE: Do not install this if using with Isaac Sim."
    )


@dataclass
class CudaRobotGeneratorConfig:
    """Robot representation generator configuration, loads from a dictionary."""

    #: Name of base link for kinematic tree.
    base_link: str

    #: Name of end-effector link to compute pose.
    ee_link: str

    #: Device to load cuda robot model.
    tensor_args: TensorDeviceType = TensorDeviceType()

    #: Name of link names to compute pose in addition to ee_link.
    link_names: Optional[List[str]] = None

    #: Name of links to compute sphere positions for use in collision checking.
    collision_link_names: Optional[List[str]] = None

    #: Collision spheres that fill the volume occupied by the links of the robot.
    #: Collision spheres can be generated for robot using `Isaac Sim Robot Description Editor <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/advanced_tutorials/tutorial_motion_generation_robot_description_editor.html#collision-spheres>`_.
    collision_spheres: Union[None, str, Dict[str, Any]] = None

    #: Radius buffer to add to collision spheres as padding.
    collision_sphere_buffer: Union[float, Dict[str, float]] = 0.0

    #: Compute jacobian of link poses. Currently not supported.
    compute_jacobian: bool = False

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

    #: Enabling this flag writes out the cumulative transformation matrix to global memory. This
    #: allows for reusing the cumulative matrix during backward of kinematics (15% speedup over
    #: recomputing cumul in backward).
    use_global_cumul: bool = True

    #: Path of meshes of robot links. Currently not used as we represent robot link geometry with
    #: collision spheres.
    asset_root_path: str = ""

    #: Names of links to load meshes for visualization. This is only used for exporting
    #: visualizations.
    mesh_link_names: Optional[List[str]] = None

    #: Set this to true to add mesh_link_names to link_names when computing kinematics.
    load_link_names_with_mesh: bool = False

    #: Path to load robot urdf.
    urdf_path: Optional[str] = None

    #: Path to load robot usd.
    usd_path: Optional[str] = None

    #: Root prim of robot in usd.
    usd_robot_root: Optional[str] = None

    #: Path of robot in Isaac server.
    isaac_usd_path: Optional[str] = None

    #: Load Kinematics chain from usd.
    use_usd_kinematics: bool = False

    #: Joints to flip axis when loading from USD
    usd_flip_joints: Optional[List[str]] = None

    #: Flip joint limits in USD.
    usd_flip_joint_limits: Optional[List[str]] = None

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

    #: Configuration space parameters for robot (e.g, acceleration, jerk limits).
    cspace: Union[None, CSpaceConfig, Dict[str, List[Any]]] = None

    #: Enable loading meshes from kinematics parser.
    load_meshes: bool = False

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
        if self.usd_path is not None:
            self.usd_path = join_path(asset_path, self.usd_path)
        if self.asset_root_path != "":
            self.asset_root_path = join_path(asset_path, self.asset_root_path)
        elif self.urdf_path is not None:
            self.asset_root_path = os.path.dirname(self.urdf_path)

        if self.collision_spheres is None and (
            self.collision_link_names is not None and len(self.collision_link_names) > 0
        ):
            log_error("collision link names are provided without robot collision spheres")
        if self.load_link_names_with_mesh:
            if self.link_names is None:
                self.link_names = copy.deepcopy(self.mesh_link_names)
            else:
                for i in self.mesh_link_names:
                    if i not in self.link_names:
                        self.link_names.append(i)
        if self.link_names is None:
            self.link_names = [self.ee_link]
        if self.collision_link_names is None:
            self.collision_link_names = []
        if self.ee_link not in self.link_names:
            self.link_names.append(self.ee_link)
        if self.collision_spheres is not None:
            if isinstance(self.collision_spheres, str):
                coll_yml = join_path(robot_path, self.collision_spheres)
                coll_params = load_yaml(coll_yml)

                self.collision_spheres = coll_params["collision_spheres"]
            if self.extra_collision_spheres is not None:
                for k in self.extra_collision_spheres.keys():
                    new_spheres = [
                        {"center": [0.0, 0.0, 0.0], "radius": -10.0}
                        for n in range(self.extra_collision_spheres[k])
                    ]
                    self.collision_spheres[k] = new_spheres
        if self.use_usd_kinematics and self.usd_path is None:
            log_error("usd_path is required to load kinematics from usd")
        if self.usd_flip_joints is None:
            self.usd_flip_joints = {}
        if self.usd_flip_joint_limits is None:
            self.usd_flip_joint_limits = []
        if self.extra_links is None:
            self.extra_links = {}
        else:
            for k in self.extra_links.keys():
                if isinstance(self.extra_links[k], dict):
                    self.extra_links[k] = LinkParams.from_dict(self.extra_links[k])
        if isinstance(self.cspace, Dict):
            self.cspace = CSpaceConfig(**self.cspace, tensor_args=self.tensor_args)


class CudaRobotGenerator(CudaRobotGeneratorConfig):
    """Robot Kinematics Representation Generator.

    The word "Chain" is used interchangeably with "Tree" in this class.

    """

    def __init__(self, config: CudaRobotGeneratorConfig) -> None:
        """Initialize the robot generator.

        Args:
            config: Parameters to initialize the robot generator.
        """
        super().__init__(**vars(config))
        self.cpu_tensor_args = self.tensor_args.cpu()

        self._self_collision_data = None
        self.non_fixed_joint_names = []
        self._n_dofs = 1
        self._kinematics_config = None
        self.initialize_tensors()

    @property
    def kinematics_config(self) -> KinematicsTensorConfig:
        """Kinematics representation as Tensors."""
        return self._kinematics_config

    @property
    def self_collision_config(self) -> SelfCollisionKinematicsConfig:
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

        self._link_spheres_tensor = torch.empty(
            (0, 4), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self._link_sphere_idx_map = torch.empty(
            (0), dtype=torch.int16, device=self.tensor_args.device
        )
        self.total_spheres = 0
        self.self_collision_distance = (
            torch.zeros(
                (self.total_spheres, self.total_spheres),
                dtype=self.tensor_args.dtype,
                device=self.tensor_args.device,
            )
            - torch.inf
        )
        self.self_collision_offset = torch.zeros(
            (self.total_spheres), dtype=self.tensor_args.dtype, device=self.tensor_args.device
        )
        # create a mega list of all links that we need:
        other_links = copy.deepcopy(self.link_names)

        for i in self.collision_link_names:
            if i not in self.link_names:
                other_links.append(i)
        for i in self.extra_links:
            p_name = self.extra_links[i].parent_link_name
            if p_name not in self.link_names and p_name not in other_links:
                other_links.append(p_name)

        # other_links = list(set(self.link_names + self.collision_link_names))

        # load kinematics parser based on file type:
        # NOTE: Also add option to load from data buffers.
        if self.use_usd_kinematics:
            self._kinematics_parser = UsdKinematicsParser(
                self.usd_path,
                flip_joints=self.usd_flip_joints,
                flip_joint_limits=self.usd_flip_joint_limits,
                extra_links=self.extra_links,
                usd_robot_root=self.usd_robot_root,
            )
        else:
            self._kinematics_parser = UrdfKinematicsParser(
                self.urdf_path,
                mesh_root=self.asset_root_path,
                extra_links=self.extra_links,
                load_meshes=self.load_meshes,
            )

        if self.lock_joints is None:
            self._build_kinematics(self.base_link, self.ee_link, other_links, self.link_names)
        else:
            self._build_kinematics_with_lock_joints(
                self.base_link, self.ee_link, other_links, self.link_names, self.lock_joints
            )
        if self.cspace is None:
            jpv = self._get_joint_position_velocity_limits()
            self.cspace = CSpaceConfig.load_from_joint_limits(
                jpv["position"][1, :], jpv["position"][0, :], self.joint_names, self.tensor_args
            )

        self.cspace.inplace_reindex(self.joint_names)
        self._update_joint_limits()
        self._ee_idx = self.link_names.index(self.ee_link)

        # create kinematics tensor:
        self._kinematics_config = KinematicsTensorConfig(
            fixed_transforms=self._fixed_transform,
            link_map=self._link_map,
            joint_map=self._joint_map,
            joint_map_type=self._joint_map_type,
            joint_offset_map=self._joint_offset_map,
            store_link_map=self._store_link_map,
            link_chain_map=self._link_chain_map,
            link_names=self.link_names,
            link_spheres=self._link_spheres_tensor,
            link_sphere_idx_map=self._link_sphere_idx_map,
            n_dof=self._n_dofs,
            joint_limits=self._joint_limits,
            non_fixed_joint_names=self.non_fixed_joint_names,
            total_spheres=self.total_spheres,
            link_name_to_idx_map=self._name_to_idx_map,
            joint_names=self.joint_names,
            debug=self.debug,
            ee_idx=self._ee_idx,
            mesh_link_names=self.mesh_link_names,
            cspace=self.cspace,
            base_link=self.base_link,
            ee_link=self.ee_link,
            lock_jointstate=self.lock_jointstate,
            mimic_joints=self._mimic_joint_data,
        )
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
            transform = (
                Pose.from_list([0, 0, 0, 1, 0, 0, 0], self.tensor_args)
                .get_matrix()
                .view(4, 4)
                .cpu()
                .numpy()
            )
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
        ee_link: str,
        other_links: List[str],
    ) -> List[str]:
        """Build kinematic tree of the robot.

        Args:
            base_link: Name of base link for the chain.
            ee_link: Name of end-effector link for the chain.
            other_links: List of other links to add to the chain.

        Returns:
            List[str]: List of link names in the chain.
        """
        self._n_dofs = 0
        self._controlled_links = []
        self._bodies = []
        self._name_to_idx_map = dict()
        self.base_link = base_link
        self.ee_link = ee_link
        self.joint_names = []
        self._fixed_transform = []
        chain_link_names = self._kinematics_parser.get_chain(base_link, ee_link)
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

    @profiler.record_function("robot_generator/build_kinematics_tensors")
    def _build_kinematics_tensors(self, base_link, link_names, chain_link_names):
        """Create kinematic tensors for robot given kinematic tree.

        Args:
            base_link: Name of base link for the tree.
            link_names: Namer of links to compute kinematics for. This is used to determine link
                indices to store pose during forward kinematics.
            chain_link_names: List of link names in the kinematic tree. Used to traverse the
                kinematic tree.
        """
        self._active_joints = []
        self._mimic_joint_data = {}
        link_map = [0 for i in range(len(self._bodies))]
        store_link_map = []  # [-1 for i in range(len(self._bodies))]

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
        if self._bodies[0].link_name in link_names:
            store_link_map.append(chain_link_names.index(self._bodies[0].link_name))
            ordered_link_names.append(self._bodies[0].link_name)
        # get joint types:
        for i in range(1, len(self._bodies)):
            body = self._bodies[i]
            parent_name = body.parent_link_name
            link_map[i] = self._name_to_idx_map[parent_name]
            joint_offset_map.append(body.joint_offset)
            joint_map_type[i] = body.joint_type.value
            if body.link_name in link_names:
                store_link_map.append(chain_link_names.index(body.link_name))
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
        self.link_names = ordered_link_names
        # do a for loop to get link matrix:
        link_chain_map = torch.eye(
            len(chain_link_names), dtype=torch.int16, device=self.cpu_tensor_args.device
        )

        # iterate and set true:
        for i in range(len(chain_link_names)):
            chain_l_names = self._kinematics_parser.get_chain(base_link, chain_link_names[i])
            for k in chain_l_names:
                link_chain_map[i, chain_link_names.index(k)] = 1.0

        self._link_map = torch.as_tensor(
            link_map, device=self.tensor_args.device, dtype=torch.int16
        )
        self._joint_map = torch.as_tensor(
            joint_map, device=self.tensor_args.device, dtype=torch.int16
        )
        self._joint_map_type = torch.as_tensor(
            joint_map_type, device=self.tensor_args.device, dtype=torch.int8
        )
        self._store_link_map = torch.as_tensor(
            store_link_map, device=self.tensor_args.device, dtype=torch.int16
        )
        self._joint_offset_map = torch.as_tensor(
            joint_offset_map, device=self.tensor_args.device, dtype=torch.float32
        )
        self._joint_offset_map = self._joint_offset_map.view(-1).contiguous()
        self._link_chain_map = link_chain_map.to(device=self.tensor_args.device)
        self._fixed_transform = torch.cat((self._fixed_transform), dim=0).to(
            device=self.tensor_args.device
        )
        self._all_joint_names = all_joint_names

    @profiler.record_function("robot_generator/build_kinematics")
    def _build_kinematics(
        self, base_link: str, ee_link: str, other_links: List[str], link_names: List[str]
    ):
        """Build kinematics tensors given base link, end-effector link and other links.

        Args:
            base_link: Name of base link for the kinematic tree.
            ee_link: Name of end-effector link for the kinematic tree.
            other_links: List of other links to add to the kinematic tree.
            link_names: List of link names to store poses after kinematics computation.
        """
        chain_link_names = self._build_chain(base_link, ee_link, other_links)
        self._build_kinematics_tensors(base_link, link_names, chain_link_names)
        if self.collision_spheres is not None and len(self.collision_link_names) > 0:
            self._build_collision_model(
                self.collision_spheres, self.collision_link_names, self.collision_sphere_buffer
            )

    @profiler.record_function("robot_generator/build_kinematics_with_lock_joints")
    def _build_kinematics_with_lock_joints(
        self,
        base_link: str,
        ee_link: str,
        other_links: List[str],
        link_names: List[str],
        lock_joints: Dict[str, float],
    ):
        """Build kinematics with locked joints.

        This function will first build the chain with no locked joints, find the transforms
        when the locked joints are set to the given values, and then use these transforms as
        fixed transforms for the locked joints.

        Args:
            base_link: Base link of the kinematic tree.
            ee_link: End-effector link of the kinematic tree.
            other_links: Other links to add to the kinematic tree.
            link_names: List of link names to store poses after kinematics computation.
            lock_joints: Joints to lock in the kinematic tree with value to lock at.
        """
        chain_link_names = self._build_chain(base_link, ee_link, other_links)
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

        new_link_names = list(set(link_names + lock_links))

        # rebuild kinematic tree with link names added to link pose computation:
        self._build_kinematics_tensors(base_link, new_link_names, chain_link_names)
        if self.collision_spheres is not None and len(self.collision_link_names) > 0:
            self._build_collision_model(
                self.collision_spheres, self.collision_link_names, self.collision_sphere_buffer
            )
        # do forward kinematics and get transform for locked joints:
        q = torch.zeros(
            (1, self._n_dofs), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        # set lock joints in the joint angles:
        l_idx = torch.as_tensor(
            [self.joint_names.index(l) for l in lock_joints.keys()],
            dtype=torch.long,
            device=self.tensor_args.device,
        )
        l_val = self.tensor_args.to_device([lock_joints[l] for l in lock_joints.keys()])

        q[0, l_idx] = l_val
        kinematics_config = KinematicsTensorConfig(
            fixed_transforms=self._fixed_transform,
            link_map=self._link_map,
            joint_map=self._joint_map,
            joint_map_type=self._joint_map_type,
            joint_offset_map=self._joint_offset_map,
            store_link_map=self._store_link_map,
            link_chain_map=self._link_chain_map,
            link_names=self.link_names,
            link_spheres=self._link_spheres_tensor,
            link_sphere_idx_map=self._link_sphere_idx_map,
            n_dof=self._n_dofs,
            joint_limits=self._joint_limits,
            non_fixed_joint_names=self.non_fixed_joint_names,
            total_spheres=self.total_spheres,
        )
        link_poses = self._get_link_poses(q, lock_links, kinematics_config)
        # remove lock links from store map:
        store_link_map = [chain_link_names.index(l) for l in link_names]
        self._store_link_map = torch.as_tensor(
            store_link_map, device=self.tensor_args.device, dtype=torch.int16
        )
        self.link_names = link_names
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
                self._fixed_transform[i] = parent_t_child.get_matrix()

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
                        self._fixed_transform[i_q] = parent_t_child.get_matrix()
                        self._controlled_links.remove(i_q)
                        self._joint_map_type[i_q] = -1
                        self._joint_map[i_q] = -1

                i = joint_data[j]["link_index"]
                self._joint_map_type[i] = -1
                self._joint_map[i:] -= 1
                self._joint_map[i] = -1
                self._controlled_links.remove(i)
                self.joint_names.remove(j)
                self._n_dofs -= 1
                self._active_joints.remove(i)
            self._joint_map[self._joint_map < -1] = -1
            self._joint_map = self._joint_map.to(device=self.tensor_args.device)
            self._joint_map_type = self._joint_map_type.to(device=self.tensor_args.device)
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
        cpu_tensor_args = self.tensor_args.cpu()
        self_collision_buffer = self.self_collision_buffer.copy()
        with profiler.record_function("robot_generator/build_collision_spheres"):
            for j_idx, j in enumerate(collision_link_names):
                # print(j_idx)
                n_spheres = len(collision_spheres[j])
                link_spheres = torch.zeros(
                    (n_spheres, 4), dtype=cpu_tensor_args.dtype, device=cpu_tensor_args.device
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
                for i in range(n_spheres):
                    padded_radius = collision_spheres[j][i]["radius"] + offset_radius
                    if padded_radius <= 0.0 and padded_radius > -1.0:
                        padded_radius = 0.001
                    link_spheres[i, :] = tensor_sphere(
                        collision_spheres[j][i]["center"],
                        padded_radius,
                        tensor_args=cpu_tensor_args,
                        tensor=link_spheres[i, :],
                    )
                    link_sphere_idx_map.append(l_idx)
                coll_link_spheres.append(link_spheres)
                self.total_spheres += n_spheres

        self._link_spheres_tensor = torch.cat(coll_link_spheres, dim=0)
        self._link_sphere_idx_map = torch.as_tensor(
            link_sphere_idx_map, dtype=torch.int16, device=cpu_tensor_args.device
        )

        # build self collision distance tensor:
        self_collision_distance = (
            torch.zeros(
                (self.total_spheres, self.total_spheres),
                dtype=cpu_tensor_args.dtype,
                device=cpu_tensor_args.device,
            )
            - torch.inf
        )
        self.self_collision_offset = torch.zeros(
            (self.total_spheres), dtype=cpu_tensor_args.dtype, device=cpu_tensor_args.device
        )
        with profiler.record_function("robot_generator/self_collision_distance"):
            # iterate through each link:
            for j_idx, j in enumerate(collision_link_names):
                ignore_links = []
                if j in self.self_collision_ignore.keys():
                    ignore_links = self.self_collision_ignore[j]
                link1_idx = self._name_to_idx_map[j]
                link1_spheres_idx = torch.nonzero(self._link_sphere_idx_map == link1_idx)

                rad1 = self._link_spheres_tensor[link1_spheres_idx, 3]
                if j not in self_collision_buffer.keys():
                    self_collision_buffer[j] = 0.0
                c1 = self_collision_buffer[j]
                self.self_collision_offset[link1_spheres_idx] = c1
                for _, i_name in enumerate(collision_link_names):
                    if i_name == j or i_name in ignore_links:
                        continue
                    if i_name not in collision_link_names:
                        log_error("Self Collision Link name not found in collision_link_names")
                    # find index of this link name:
                    if i_name not in self_collision_buffer.keys():
                        self_collision_buffer[i_name] = 0.0
                    c2 = self_collision_buffer[i_name]
                    link2_idx = self._name_to_idx_map[i_name]
                    # update collision distance between spheres from these two links:
                    link2_spheres_idx = torch.nonzero(self._link_sphere_idx_map == link2_idx)
                    rad2 = self._link_spheres_tensor[link2_spheres_idx, 3]

                    for k1 in range(len(rad1)):
                        sp1 = link1_spheres_idx[k1]
                        for k2 in range(len(rad2)):
                            sp2 = link2_spheres_idx[k2]
                            self_collision_distance[sp1, sp2] = rad1[k1] + rad2[k2] + c1 + c2

        self_collision_distance = self_collision_distance.to(device=self.tensor_args.device)
        with profiler.record_function("robot_generator/self_collision_min"):
            d_mat = self_collision_distance
            self_collision_distance = torch.minimum(d_mat, d_mat.transpose(0, 1))

        (
            self._self_coll_thread_locations,
            self._self_coll_idx,
            valid_data,
            checks_per_thread,
        ) = self._create_self_collision_thread_data(self_collision_distance)
        use_experimental_kernel = True

        if (
            self.debug is not None
            and "self_collision_experimental" in self.debug
            and self.debug["self_collision_experimental"] is not None
        ):
            use_experimental_kernel = self.debug["self_collision_experimental"]

        if not valid_data:
            use_experimental_kernel = False
            log_warn(
                "Self Collision checks are greater than 32 * 512, using slower kernel."
                + " Number of spheres: "
                + str(self_collision_distance.shape[0])
            )
        if use_experimental_kernel:
            self_coll_matrix = torch.zeros((2), device=self.tensor_args.device, dtype=torch.uint8)
        else:
            self_coll_matrix = (self_collision_distance != -(torch.inf)).to(dtype=torch.uint8)
        # self_coll_matrix = (self_collision_distance != -(torch.inf)).to(dtype=torch.uint8)

        # convert all tensors to gpu:
        self._link_sphere_idx_map = self._link_sphere_idx_map.to(device=self.tensor_args.device)
        self._link_spheres_tensor = self._link_spheres_tensor.to(device=self.tensor_args.device)
        self.self_collision_offset = self.self_collision_offset.to(device=self.tensor_args.device)
        self._self_collision_data = SelfCollisionKinematicsConfig(
            offset=self.self_collision_offset,
            thread_location=self._self_coll_thread_locations,
            thread_max=self._self_coll_idx,
            collision_matrix=self_coll_matrix,
            experimental_kernel=use_experimental_kernel,
            checks_per_thread=checks_per_thread,
        )

    @profiler.record_function("robot_generator/create_self_collision_thread_data")
    def _create_self_collision_thread_data(
        self, collision_threshold: torch.Tensor
    ) -> Tuple[torch.Tensor, int, bool, int]:
        """Create thread data for self collision checks.

        Args:
            collision_threshold: Collision distance between spheres of the robot. Used to
                skip self collision checks when distance is -inf.

        Returns:
            Tuple[torch.Tensor, int, bool, int]: Thread location for self collision checks,
                number of self collision checks, if thread calculation was successful,
                and number of checks per thread.

        """
        coll_cpu = collision_threshold.cpu()
        max_checks_per_thread = 512
        thread_loc = torch.zeros((2 * 32 * max_checks_per_thread), dtype=torch.int16) - 1
        n_spheres = coll_cpu.shape[0]
        sl_idx = 0
        skip_count = 0
        all_val = 0
        valid_data = True
        for i in range(n_spheres):
            if not valid_data:
                break
            if torch.max(coll_cpu[i]) == -torch.inf:
                log_info("skip" + str(i))
            for j in range(i + 1, n_spheres):
                if sl_idx > thread_loc.shape[0] - 1:
                    valid_data = False
                    log_warn(
                        "Self Collision checks are greater than "
                        + str(32 * max_checks_per_thread)
                        + ", using slower kernel"
                    )
                    break
                if coll_cpu[i, j] != -torch.inf:
                    thread_loc[sl_idx] = i
                    sl_idx += 1
                    thread_loc[sl_idx] = j
                    sl_idx += 1
                else:
                    skip_count += 1
                all_val += 1
        log_info("Self Collision threads, skipped %: " + str(100 * float(skip_count) / all_val))
        log_info("Self Collision count: " + str(sl_idx / (2)))
        log_info("Self Collision per thread: " + str(sl_idx / (2 * 1024)))

        max_checks_per_thread = 512
        val = sl_idx / (2 * 1024)
        if val < 1:
            max_checks_per_thread = 1
        elif val < 2:
            max_checks_per_thread = 2
        elif val < 4:
            max_checks_per_thread = 4
        elif val < 8:
            max_checks_per_thread = 8
        elif val < 32:
            max_checks_per_thread = 32
        elif val < 64:
            max_checks_per_thread = 64
        elif val < 128:
            max_checks_per_thread = 128
        elif val < 512:
            max_checks_per_thread = 512
        else:
            log_error(
                "Self Collision not supported as checks are greater than 32 * 512, \
                      reduce number of spheres used to approximate the robot."
            )

        if max_checks_per_thread < 2:
            max_checks_per_thread = 2
        log_info("Self Collision using: " + str(max_checks_per_thread))

        return (
            thread_loc.to(device=collision_threshold.device),
            sl_idx,
            valid_data,
            max_checks_per_thread,
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
                self._n_dofs = self._n_dofs + 1
        self._fixed_transform.append(
            torch.as_tensor(
                rigid_body_params.fixed_transform,
                device=self.cpu_tensor_args.device,
                dtype=self.cpu_tensor_args.dtype,
            ).unsqueeze(0)
        )
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
        self, q: torch.Tensor, link_names: List[str], kinematics_config: KinematicsTensorConfig
    ) -> Pose:
        """Get Pose of links at given joint angles using forward kinematics.

        This is implemented here to avoid circular dependencies with
        :class:`~curobo.cuda_robot_model.cuda_robot_model.CudaRobotModel` module. This is used
        to calculate fixed transforms for locked joints in this class. This implementation
        does not compute position of robot spheres.

        Args:
            q: Joint angles to compute forward kinematics for.
            link_names: Name of links to return pose.
            kinematics_config: Tensor Configuration for kinematics computation.

        Returns:
            Pose: Pose of links at given joint angles.
        """
        q = q.view(1, -1)
        link_pos_seq = torch.zeros(
            (1, len(self.link_names), 3),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        link_quat_seq = torch.zeros(
            (1, len(self.link_names), 4),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        batch_robot_spheres = torch.zeros(
            (1, 0, 4),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        grad_out_q = torch.zeros(
            (1 * q.shape[-1]),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        global_cumul_mat = torch.zeros(
            (1, self._link_map.shape[0], 4, 4),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )

        link_pos_seq, link_quat_seq, _ = get_cuda_kinematics(
            link_pos_seq,
            link_quat_seq,
            batch_robot_spheres.contiguous(),
            global_cumul_mat,
            q,
            kinematics_config.fixed_transforms.contiguous(),
            kinematics_config.link_spheres.contiguous(),
            kinematics_config.link_map,  # tells which link is attached to which link i
            kinematics_config.joint_map,  # tells which joint is attached to a link i
            kinematics_config.joint_map_type,  # joint type
            kinematics_config.store_link_map,
            kinematics_config.link_sphere_idx_map.contiguous(),  # sphere idx map
            kinematics_config.link_chain_map,
            kinematics_config.joint_offset_map,
            grad_out_q,
            False,
        )
        position = torch.zeros(
            (q.shape[0], len(link_names), 3),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        quaternion = torch.zeros(
            (q.shape[0], len(link_names), 4),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )

        for li, l in enumerate(link_names):
            i = self.link_names.index(l)
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
        joint_limits = {"position": [[], []], "velocity": [[], []]}

        for idx in self._active_joints:
            joint_limits["position"][0].append(self._bodies[idx].joint_limits[0])
            joint_limits["position"][1].append(self._bodies[idx].joint_limits[1])
            joint_limits["velocity"][0].append(self._bodies[idx].joint_velocity_limits[0])
            joint_limits["velocity"][1].append(self._bodies[idx].joint_velocity_limits[1])
        for k in joint_limits:
            joint_limits[k] = torch.as_tensor(
                joint_limits[k], device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
        return joint_limits

    @profiler.record_function("robot_generator/update_joint_limits")
    def _update_joint_limits(self):
        """Update limits from CSpaceConfig (acceleration, jerk limits and position clips)."""
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
        # clip joint position:
        joint_limits["position"][0] += self.cspace.position_limit_clip
        joint_limits["position"][1] -= self.cspace.position_limit_clip
        joint_limits["velocity"][0] *= self.cspace.velocity_scale
        joint_limits["velocity"][1] *= self.cspace.velocity_scale

        self._joint_limits = JointLimits(joint_names=self.joint_names, **joint_limits)
