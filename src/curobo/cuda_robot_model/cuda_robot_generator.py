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
from __future__ import annotations

# Standard Library
import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Third Party
import importlib_resources
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
        + "or pip install usd-core, NOTE: Do not install this if using with ISAAC SIM."
    )


@dataclass
class CudaRobotGeneratorConfig:
    """Create Cuda Robot Model Configuration."""

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
    #: Collision spheres can be generated for robot using https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/advanced_tutorials/tutorial_motion_generation_robot_description_editor.html#collision-spheres
    collision_spheres: Union[None, str, Dict[str, Any]] = None

    #: Radius buffer to add to collision spheres as padding.
    collision_sphere_buffer: float = 0.0

    #: Compute jacobian of link poses. Currently not supported.
    compute_jacobian: bool = False

    #: Padding to add for self collision between links. Some robots use a large padding
    #: for self collision avoidance (https://github.com/ros-planning/panda_moveit_config/pull/35#issuecomment-671333863)
    self_collision_buffer: Optional[Dict[str, float]] = None

    #: Self-collision ignore
    self_collision_ignore: Optional[Dict[str, List[str]]] = None

    #: debug config
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

    extra_links: Optional[Dict[str, LinkParams]] = None

    #: this is deprecated
    add_object_link: bool = False

    use_external_assets: bool = False

    external_asset_path: Optional[str] = None
    external_robot_configs_path: Optional[str] = None

    #: Create n collision spheres for links with name
    extra_collision_spheres: Optional[Dict[str, int]] = None

    #: cspace config
    cspace: Union[None, CSpaceConfig, Dict[str, List[Any]]] = None

    #: Enable loading meshes from kinematics parser.
    load_meshes: bool = False

    def __post_init__(self):
        # add root path:
        # Check if an external asset path is provided:
        asset_path = get_assets_path()
        robot_path = get_robot_configs_path()
        if self.external_asset_path is not None:
            asset_path = self.external_asset_path
        if self.external_robot_configs_path is not None:
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
    def __init__(self, config: CudaRobotGeneratorConfig) -> None:
        super().__init__(**vars(config))
        self.cpu_tensor_args = self.tensor_args.cpu()

        self._self_collision_data = None
        self.non_fixed_joint_names = []
        self._n_dofs = 1

        self.initialize_tensors()

    @property
    def kinematics_config(self):
        return self._kinematics_config

    @property
    def self_collision_config(self):
        return self._self_collision_data

    @property
    def kinematics_parser(self):
        return self._kinematics_parser

    @profiler.record_function("robot_generator/initialize_tensors")
    def initialize_tensors(self):
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
        )
        if self.asset_root_path != "":
            self._kinematics_parser.add_absolute_path_to_link_meshes(self.asset_root_path)

    def add_link(self, link_params: LinkParams):
        self.extra_links[link_params.link_name] = link_params

    def add_fixed_link(
        self,
        link_name: str,
        parent_link_name: str,
        joint_name: Optional[str] = None,
        transform: Optional[Pose] = None,
    ):
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
    def _build_chain(self, base_link, ee_link, other_links, link_names):
        self._n_dofs = 0
        self._controlled_joints = []
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

    @profiler.record_function("robot_generator/build_kinematics_tensors")
    def _build_kinematics_tensors(self, base_link, ee_link, link_names, chain_link_names):
        link_map = [0 for i in range(len(self._bodies))]
        store_link_map = []  # [-1 for i in range(len(self._bodies))]

        joint_map = [
            -1 if i not in self._controlled_joints else i for i in range(len(self._bodies))
        ]  #
        joint_map_type = [
            -1 if i not in self._controlled_joints else i for i in range(len(self._bodies))
        ]
        all_joint_names = []
        j_count = 0
        ordered_link_names = []
        # add body 0 details:
        if self._bodies[0].link_name in link_names:
            store_link_map.append(chain_link_names.index(self._bodies[0].link_name))
            ordered_link_names.append(self._bodies[0].link_name)
        # get joint types:
        for i in range(1, len(self._bodies)):
            body = self._bodies[i]
            parent_name = body.parent_link_name
            link_map[i] = self._name_to_idx_map[parent_name]
            all_joint_names.append(body.joint_name)
            if body.link_name in link_names:
                store_link_map.append(chain_link_names.index(body.link_name))
                ordered_link_names.append(body.link_name)
            if i in self._controlled_joints:
                joint_map[i] = j_count
                joint_map_type[i] = body.joint_type.value
                j_count += 1
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
        self._link_chain_map = link_chain_map.to(device=self.tensor_args.device)
        self._fixed_transform = torch.cat((self._fixed_transform), dim=0).to(
            device=self.tensor_args.device
        )
        self._all_joint_names = all_joint_names

    @profiler.record_function("robot_generator/build_kinematics")
    def _build_kinematics(self, base_link, ee_link, other_links, link_names):
        chain_link_names = self._build_chain(base_link, ee_link, other_links, link_names)
        self._build_kinematics_tensors(base_link, ee_link, link_names, chain_link_names)
        if self.collision_spheres is not None and len(self.collision_link_names) > 0:
            self._build_collision_model(
                self.collision_spheres, self.collision_link_names, self.collision_sphere_buffer
            )

    @profiler.record_function("robot_generator/build_kinematics_with_lock_joints")
    def _build_kinematics_with_lock_joints(
        self,
        base_link,
        ee_link,
        other_links,
        link_names,
        lock_joints: Dict[str, float],
    ):
        chain_link_names = self._build_chain(base_link, ee_link, other_links, link_names)
        # find links attached to lock joints:
        lock_joint_names = list(lock_joints.keys())

        joint_data = self._get_joint_links(lock_joint_names)

        lock_links = list(
            set(
                [joint_data[j]["parent"] for j in joint_data.keys()]
                + [joint_data[j]["child"] for j in joint_data.keys()]
            )
        )
        new_link_names = link_names + lock_links

        # rebuild kinematic tree with link names added to link pose computation:
        self._build_kinematics_tensors(base_link, ee_link, new_link_names, chain_link_names)
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
                i = self._all_joint_names.index(j) + 1
                self._joint_map_type[i] = -1
                self._joint_map[i:] -= 1
                self._joint_map[i] = -1
                self._n_dofs -= 1
                self._fixed_transform[i] = parent_t_child.get_matrix()
                self._controlled_joints.remove(i)
                self.joint_names.remove(j)
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
        collision_sphere_buffer: float = 0.0,
    ):
        """

        Args:
            collision_spheres (_type_): _description_
            collision_link_names (_type_): _description_
            collision_sphere_buffer (float, optional): _description_. Defaults to 0.0.
        """

        # We create all tensors on cpu and then finally move them to gpu
        coll_link_spheres = []
        # we store as [n_link, 7]
        link_sphere_idx_map = []
        cpu_tensor_args = self.tensor_args.cpu()
        with profiler.record_function("robot_generator/build_collision_spheres"):
            for j_idx, j in enumerate(collision_link_names):
                # print(j_idx)
                n_spheres = len(collision_spheres[j])
                link_spheres = torch.zeros(
                    (n_spheres, 4), dtype=cpu_tensor_args.dtype, device=cpu_tensor_args.device
                )
                # find link index in global map:
                l_idx = self._name_to_idx_map[j]

                for i in range(n_spheres):
                    link_spheres[i, :] = tensor_sphere(
                        collision_spheres[j][i]["center"],
                        collision_spheres[j][i]["radius"],
                        tensor_args=cpu_tensor_args,
                        tensor=link_spheres[i, :],
                    )
                    link_sphere_idx_map.append(l_idx)
                coll_link_spheres.append(link_spheres)
                self.total_spheres += n_spheres

        self._link_spheres_tensor = torch.cat(coll_link_spheres, dim=0)
        new_radius = self._link_spheres_tensor[..., 3] + collision_sphere_buffer
        flag = torch.logical_and(new_radius > -1.0, new_radius <= 0.0)
        new_radius[flag] = 0.001
        self._link_spheres_tensor[:, 3] = new_radius
        self._link_sphere_idx_map = torch.as_tensor(
            link_sphere_idx_map, dtype=torch.int16, device=cpu_tensor_args.device
        )

        # build self collision distance tensor:
        self.self_collision_distance = (
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
                if j not in self.self_collision_buffer.keys():
                    self.self_collision_buffer[j] = 0.0
                c1 = self.self_collision_buffer[j]
                self.self_collision_offset[link1_spheres_idx] = c1
                for _, i_name in enumerate(collision_link_names):
                    if i_name == j or i_name in ignore_links:
                        continue
                    if i_name not in collision_link_names:
                        log_error("Self Collision Link name not found in collision_link_names")
                    # find index of this link name:
                    if i_name not in self.self_collision_buffer.keys():
                        self.self_collision_buffer[i_name] = 0.0
                    c2 = self.self_collision_buffer[i_name]
                    link2_idx = self._name_to_idx_map[i_name]
                    # update collision distance between spheres from these two links:
                    link2_spheres_idx = torch.nonzero(self._link_sphere_idx_map == link2_idx)
                    rad2 = self._link_spheres_tensor[link2_spheres_idx, 3]

                    for k1 in range(len(rad1)):
                        sp1 = link1_spheres_idx[k1]
                        for k2 in range(len(rad2)):
                            sp2 = link2_spheres_idx[k2]
                            self.self_collision_distance[sp1, sp2] = rad1[k1] + rad2[k2] + c1 + c2

        with profiler.record_function("robot_generator/self_collision_min"):
            d_mat = self.self_collision_distance
            self.self_collision_distance = torch.minimum(d_mat, d_mat.transpose(0, 1))

        if self.debug is not None and "self_collision_experimental" in self.debug:
            use_experimental_kernel = self.debug["self_collision_experimental"]
        self.self_collision_distance = self.self_collision_distance.to(
            device=self.tensor_args.device
        )
        (
            self._self_coll_thread_locations,
            self._self_coll_idx,
            valid_data,
            checks_per_thread,
        ) = self._create_self_collision_thread_data(self.self_collision_distance)
        self._self_coll_matrix = (self.self_collision_distance != -(torch.inf)).to(
            dtype=torch.uint8
        )

        use_experimental_kernel = True
        # convert all tensors to gpu:
        self._link_sphere_idx_map = self._link_sphere_idx_map.to(device=self.tensor_args.device)
        self._link_spheres_tensor = self._link_spheres_tensor.to(device=self.tensor_args.device)
        self.self_collision_offset = self.self_collision_offset.to(device=self.tensor_args.device)
        self._self_collision_data = SelfCollisionKinematicsConfig(
            offset=self.self_collision_offset,
            distance_threshold=self.self_collision_distance,
            thread_location=self._self_coll_thread_locations,
            thread_max=self._self_coll_idx,
            collision_matrix=self._self_coll_matrix,
            experimental_kernel=valid_data and use_experimental_kernel,
            checks_per_thread=checks_per_thread,
        )

    @profiler.record_function("robot_generator/create_self_collision_thread_data")
    def _create_self_collision_thread_data(self, collision_threshold):
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
                print("skip", i)
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
            raise ValueError("Self Collision not supported")

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
    def _add_body_to_tree(self, link_name, base=False):
        body_idx = len(self._bodies)

        rigid_body_params = self._kinematics_parser.get_link_parameters(link_name, base=base)
        self._bodies.append(rigid_body_params)
        if rigid_body_params.joint_type != JointType.FIXED:
            self._controlled_joints.append(body_idx)
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

    def _get_joint_links(self, joint_names: List[str]):
        j_data = {}
        for j in joint_names:
            for b in self._bodies:
                if b.joint_name == j:
                    j_data[j] = {"parent": b.parent_link_name, "child": b.link_name}
        return j_data

    @profiler.record_function("robot_generator/get_link_poses")
    def _get_link_poses(
        self, q: torch.Tensor, link_names: List[str], kinematics_config: KinematicsTensorConfig
    ) -> Pose:
        if q.is_contiguous():
            q_in = q.view(-1)
        else:
            q_in = q.reshape(-1)  # .reshape(-1)
        # q_in = q.reshape(-1)
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
            (1, self.total_spheres, 4),
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
            # self._link_mat_seq,  # data will be stored here
            link_pos_seq,
            link_quat_seq,
            batch_robot_spheres,
            global_cumul_mat,
            q_in,
            kinematics_config.fixed_transforms,
            kinematics_config.link_spheres,
            kinematics_config.link_map,  # tells which link is attached to which link i
            kinematics_config.joint_map,  # tells which joint is attached to a link i
            kinematics_config.joint_map_type,  # joint type
            kinematics_config.store_link_map,
            kinematics_config.link_sphere_idx_map,  # sphere idx map
            kinematics_config.link_chain_map,
            grad_out_q,
            self.use_global_cumul,
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
        return Pose(position=position, quaternion=quaternion)

    @property
    def get_joint_limits(self):
        return self._joint_limits

    @profiler.record_function("robot_generator/get_joint_limits")
    def _get_joint_position_velocity_limits(self):
        joint_limits = {"position": [[], []], "velocity": [[], []]}

        for idx in self._controlled_joints:
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
        # TODO: change this to be per joint
        joint_limits["position"][0] += self.cspace.position_limit_clip
        joint_limits["position"][1] -= self.cspace.position_limit_clip
        joint_limits["velocity"][0] *= self.cspace.velocity_scale
        joint_limits["velocity"][1] *= self.cspace.velocity_scale

        self._joint_limits = JointLimits(joint_names=self.joint_names, **joint_limits)
