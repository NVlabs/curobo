# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Kinematics tensor configuration for robot kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from curobo._src.geom.data.data_collision_layers import layer_bits_to_int32
from curobo._src.robot.types.collision_geometry import RobotCollisionGeometry
from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.robot.types.joint_types import JointType
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.robot.dynamics.dynamics import _DynamicsModelTensors


@dataclass
class KinematicsParams:
    """Stores robot's kinematics parameters as Tensors to use in Kinematics computations.

    Use :meth:`curobo.cuda_robot_model.cuda_robot_generator.KinematicsLoader` to generate this
    configuration from a urdf or usd.

    """

    #: Static Homogenous Transform from parent link to child link for all links [num_links,3,4].
    fixed_transforms: torch.Tensor

    #: Index of fixed_transform given link index [num_links].
    link_map: torch.Tensor

    #: Joint index given link index [num_links].
    joint_map: torch.Tensor

    #: Type of joint given link index [num_links].
    #: value is from JointType enum.
    joint_map_type: torch.Tensor

    #: Joint offset to store scalars for mimic joints and negative axis joints.
    joint_offset_map: torch.Tensor

    #: Index of link to write out pose [n_tool_frames].
    tool_frame_map: torch.Tensor

    #: Packed array of actual link indices for all kinematic chains. Each link's chain is
    #: stored consecutively. Contains the actual link indices (as used in kernel arrays)
    #: that affect each link's gradient.
    link_chain_data: torch.Tensor

    #: Offset array indicating where each link's chain starts in link_chain_data [num_links + 1].
    #: link_chain_offsets[i] = start index in link_chain_data for link i's chain
    #: link_chain_offsets[i+1] - link_chain_offsets[i] = length of link i's chain
    link_chain_offsets: torch.Tensor

    #: Packed array of link indices connected to each joint. Each joint's connected links
    #: stored consecutively. Contains the actual link indices that are connected to each
    #: joint for efficient jacobian computation.
    joint_links_data: torch.Tensor

    #: Offset array indicating where each joint's connected links start in joint_links_data
    #: [n_joints + 1]. joint_links_offsets[j] = start index in joint_links_data for joint
    #: j's connected links. joint_links_offsets[j+1] - joint_links_offsets[j] = number of
    #: links connected to joint j
    joint_links_offsets: torch.Tensor

    #: Name of tool frames (end-effector links) to compute pose during kinematics [n_tool_frames].
    tool_frames: List[str]

    #: Joint limits
    joint_limits: JointLimits

    #: Name of joints including ones that are locked. This does not include fixed joints.
    non_fixed_joint_names: List[str]

    #: Number of joints that are active. Each joint is only actuated along 1 dimension.
    num_dof: int

    #: Bit-packed descendant-link mask for every physical joint link. Shape is
    #: [num_links, ceil(num_links / 32)], stored as int32 and read as uint32 in CUDA.
    #: Computed from link_chain_data in __post_init__.
    joint_link_influence_words: Optional[torch.Tensor] = None

    #: Packed physical joint-link contributors for every (tool frame, output joint) pair.
    #: Computed from the chain and mimic-joint maps in __post_init__.
    tool_joint_links_data: Optional[torch.Tensor] = None

    #: CSR offsets into tool_joint_links_data. Shape is
    #: [num_pose_links * num_dof + 1]. Computed in __post_init__.
    tool_joint_links_offsets: Optional[torch.Tensor] = None

    #: Name of links which have a mesh. Currently only used for debugging and rendering.
    mesh_link_names: Optional[List[str]] = None

    #: Name of all actuated joints.
    joint_names: Optional[List[str]] = None

    #: Name of joints to lock to a fixed value along with the locked value
    lock_jointstate: Optional[JointState] = None

    #: Joints that mimic other joints. This will be populated by :class:~`KinematicsLoader`
    #  when parsing the kinematics of the robot.
    mimic_joints: Optional[dict] = None

    #: Sphere representation of the robot's geometry. Shape is [n_configs, num_spheres, 4].
    #: n_configs=1 by default (single robot configuration). When n_configs > 1, each config
    #: represents a different attached-object state, indexed by idxs_env at runtime.
    link_spheres: Optional[torch.Tensor] = None

    #: Unsigned world-collision masks [num_envs, num_spheres], selected exactly like link_spheres.
    sphere_collision_mask: Optional[torch.Tensor] = None

    #: Mapping of link index to sphere index. Shape is [num_spheres]. Gives the link index for each
    #: sphere.
    link_sphere_idx_map: Optional[torch.Tensor] = None

    #: Mapping of link name to link index. This is used to get link index from link name.
    link_name_to_idx_map: Optional[Dict[str, int]] = None

    #: Total number of spheres that represent the robot's geometry.
    total_spheres: int = 0

    #: Additional debug parameters.
    debug: Optional[Any] = None

    #: Cspace parameters for the robot.
    cspace: Optional[CSpaceParams] = None

    #: Name of base link. This is the root link from which all kinematic parameters were computed.
    base_link: str = "base_link"

    #: A copy of link spheres used as reference. Shape matches link_spheres [n_configs, num_spheres, 4].
    reference_link_spheres: Optional[torch.Tensor] = None

    #: Link masses and center of mass. [num_links, 4] - xyz=local CoM, w=mass
    link_masses_com: Optional[torch.Tensor] = None

    #: Link inertias. [num_links, 8] - ixx, iyy, izz, ixy, ixz, iyz, pad0, pad1
    #: Padded to 8 floats for float4 alignment in CUDA kernels.
    link_inertias: Optional[torch.Tensor] = None

    #: Tensor device configuration
    device_cfg: DeviceCfg = DeviceCfg()

    #: Names of links that are in contact with the gripper. This is used to disable collision for
    #: these links during grasp.
    grasp_contact_link_names: Optional[List[str]] = None

    #: Link indices sorted by tree depth level (BFS level-order). Links at the same depth
    #: have no parent-child dependencies, enabling parallel processing.
    #: Shape: [num_links], dtype: int16. Computed from link_map in __post_init__.
    link_level_data: Optional[torch.Tensor] = None

    #: CSR offsets into link_level_data per depth level. link_level_offsets[i] is the start
    #: index in link_level_data for level i. Number of levels = len(link_level_offsets) - 1.
    #: Shape: [n_levels + 1], dtype: int16. Computed from link_map in __post_init__.
    link_level_offsets: Optional[torch.Tensor] = None

    #: Maximum number of links at any single tree depth level. Used to determine
    #: threads-per-batch for tree-parallel CUDA kernels.
    max_level_width: int = 0

    # Derived model tensors are shared; each Dynamics instance owns its workspace.
    _dynamics_model: Optional[_DynamicsModelTensors] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        """Post initialization checks and data transfer to device tensors."""
        if self.cspace is None and self.joint_limits is not None and self.joint_names is not None:
            self.load_cspace_cfg_from_kinematics()
        if self.joint_limits is not None and self.cspace is not None:
            self.joint_limits = self.cspace.scale_joint_limits(self.joint_limits)
        if self.link_spheres is not None and self.reference_link_spheres is None:
            self.reference_link_spheres = self.link_spheres.clone()
        if self.link_spheres is not None:
            if self.sphere_collision_mask is None:
                self.sphere_collision_mask = torch.ones(
                    self.link_spheres.shape[:2], dtype=torch.uint32,
                    device=self.link_spheres.device,
                )
            self._validate_sphere_collision_mask()
        if self.fixed_transforms is not None:
            if self.fixed_transforms.shape[0] < 1:
                log_and_raise(
                    f"fixed_transforms has 0 links: {self.fixed_transforms.shape}"
                )
            if self.fixed_transforms.shape[1] != 3 or self.fixed_transforms.shape[2] != 4:
                log_and_raise(
                    f"fixed_transforms shape does not match: {self.fixed_transforms.shape} != (num_links, 3, 4)"
                )
        if self.link_masses_com is not None:
            if self.link_masses_com.shape[1] != 4:
                log_and_raise(
                    f"link_masses_com shape does not match: {self.link_masses_com.shape} != (num_links, 4)"
                )

        if (
            self.joint_link_influence_words is None
            or self.tool_joint_links_data is None
            or self.tool_joint_links_offsets is None
        ):
            self._compute_topology_caches()
        if (
            self.tool_joint_links_data is not None
            and self.tool_joint_links_offsets is not None
            and int(self.tool_joint_links_offsets[-1].item())
            != self.tool_joint_links_data.numel()
        ):
            log_and_raise(
                "tool_joint_links_offsets[-1] must equal "
                "tool_joint_links_data.numel()"
            )

        # check if tensors are on the same device as device_cfg.
        device = self.device_cfg.device
        if self.fixed_transforms is not None and not self.device_cfg.is_same_torch_device(self.fixed_transforms.device):
            log_and_raise(f"fixed_transforms is on device: {self.fixed_transforms.device} but device_cfg is on device: {device}")
        if self.link_map is not None and not self.device_cfg.is_same_torch_device(self.link_map.device):
            log_and_raise(f"link_map is on device: {self.link_map.device} but device_cfg is on device: {device}")
        if self.joint_map is not None and not self.device_cfg.is_same_torch_device(self.joint_map.device):
            log_and_raise(f"joint_map is on device: {self.joint_map.device} but device_cfg is on device: {device}")
        if self.joint_map_type is not None and not self.device_cfg.is_same_torch_device(self.joint_map_type.device):
            log_and_raise(f"joint_map_type is on device: {self.joint_map_type.device} but device_cfg is on device: {device}")
        if self.joint_offset_map is not None and not self.device_cfg.is_same_torch_device(self.joint_offset_map.device):
            log_and_raise(f"joint_offset_map is on device: {self.joint_offset_map.device} but device_cfg is on device: {device}")
        if self.tool_frame_map is not None and not self.device_cfg.is_same_torch_device(self.tool_frame_map.device):
            log_and_raise(f"tool_frame_map is on device: {self.tool_frame_map.device} but device_cfg is on device: {device}")
        if self.link_chain_data is not None and not self.device_cfg.is_same_torch_device(self.link_chain_data.device):
            log_and_raise(f"link_chain_data is on device: {self.link_chain_data.device} but device_cfg is on device: {device}")
        if self.link_chain_offsets is not None and not self.device_cfg.is_same_torch_device(self.link_chain_offsets.device):
            log_and_raise(f"link_chain_offsets is on device: {self.link_chain_offsets.device} but device_cfg is on device: {device}")
        if self.joint_links_data is not None and not self.device_cfg.is_same_torch_device(self.joint_links_data.device):
            log_and_raise(f"joint_links_data is on device: {self.joint_links_data.device} but device_cfg is on device: {device}")
        if self.joint_links_offsets is not None and not self.device_cfg.is_same_torch_device(self.joint_links_offsets.device):
            log_and_raise(f"joint_links_offsets is on device: {self.joint_links_offsets.device} but device_cfg is on device: {device}")
        if self.joint_link_influence_words is not None and not self.device_cfg.is_same_torch_device(self.joint_link_influence_words.device):
            log_and_raise(f"joint_link_influence_words is on device: {self.joint_link_influence_words.device} but device_cfg is on device: {device}")
        if self.tool_joint_links_data is not None and not self.device_cfg.is_same_torch_device(self.tool_joint_links_data.device):
            log_and_raise(f"tool_joint_links_data is on device: {self.tool_joint_links_data.device} but device_cfg is on device: {device}")
        if self.tool_joint_links_offsets is not None and not self.device_cfg.is_same_torch_device(self.tool_joint_links_offsets.device):
            log_and_raise(f"tool_joint_links_offsets is on device: {self.tool_joint_links_offsets.device} but device_cfg is on device: {device}")

        self.validate_shapes()

        # Compute tree-level order from link_map if not already set
        if self.link_level_data is None and self.link_map is not None:
            self._compute_link_level_order()

    def validate_shapes(self):
        """Validate cross-field shape consistency of CSR and topology tensors."""
        if self.fixed_transforms is None:
            return
        num_links = self.fixed_transforms.shape[0]
        n_active_joints = self.num_dof
        n_tool_frames = len(self.tool_frames)

        if (
            self.link_chain_offsets is not None
            and self.link_chain_offsets.size(0) != num_links + 1
        ):
            log_and_raise(
                f"link_chain_offsets.size(0) = {self.link_chain_offsets.size(0)}, "
                f"expected num_links + 1 = {num_links + 1}"
            )
        if (
            self.joint_links_offsets is not None
            and self.joint_links_offsets.size(0) != n_active_joints + 1
        ):
            log_and_raise(
                f"joint_links_offsets.size(0) = {self.joint_links_offsets.size(0)}, "
                f"expected num_dof + 1 = {n_active_joints + 1}"
            )
        expected_influence_words = (num_links + 31) // 32
        if (
            self.joint_link_influence_words is not None
            and tuple(self.joint_link_influence_words.shape)
            != (num_links, expected_influence_words)
        ):
            log_and_raise(
                f"joint_link_influence_words.shape = "
                f"{tuple(self.joint_link_influence_words.shape)}, expected "
                f"({num_links}, {expected_influence_words})"
            )
        expected_tool_offsets = n_active_joints * n_tool_frames + 1
        if (
            self.tool_joint_links_offsets is not None
            and self.tool_joint_links_offsets.numel() != expected_tool_offsets
        ):
            log_and_raise(
                f"tool_joint_links_offsets.numel() = "
                f"{self.tool_joint_links_offsets.numel()}, expected "
                f"num_dof * n_tool_frames + 1 = {expected_tool_offsets}"
            )

    def _compute_topology_caches(self):
        """Build immutable bitset and tool-joint CSR maps from robot topology."""
        required = (
            self.fixed_transforms,
            self.link_chain_data,
            self.link_chain_offsets,
            self.joint_links_data,
            self.joint_links_offsets,
            self.tool_frame_map,
        )
        if any(value is None for value in required):
            return

        device = self.fixed_transforms.device
        num_links = self.fixed_transforms.shape[0]
        chain_lengths = (
            self.link_chain_offsets[1:] - self.link_chain_offsets[:-1]
        ).to(dtype=torch.long)
        destination_links = torch.repeat_interleave(
            torch.arange(num_links, device=device, dtype=torch.long),
            chain_lengths,
        )
        source_joint_links = self.link_chain_data.to(dtype=torch.long)
        joint_link_affects_link = torch.zeros(
            (num_links, num_links), device=device, dtype=torch.bool
        )
        joint_link_affects_link[source_joint_links, destination_links] = True

        influence_word_count = (num_links + 31) // 32
        influence_words = torch.zeros(
            (num_links, influence_word_count), device=device, dtype=torch.int32
        )
        for word_index in range(influence_word_count):
            link_start = word_index * 32
            link_end = min(link_start + 32, num_links)
            bit_values = torch.bitwise_left_shift(
                torch.ones(link_end - link_start, device=device, dtype=torch.int64),
                torch.arange(link_end - link_start, device=device, dtype=torch.int64),
            )
            influence_words[:, word_index] = torch.sum(
                joint_link_affects_link[:, link_start:link_end].to(dtype=torch.int64)
                * bit_values,
                dim=1,
            ).to(dtype=torch.int32)
        self.joint_link_influence_words = influence_words

        joint_link_counts = (
            self.joint_links_offsets[1:] - self.joint_links_offsets[:-1]
        ).to(dtype=torch.long)
        output_joint_indices = torch.repeat_interleave(
            torch.arange(self.num_dof, device=device, dtype=torch.long),
            joint_link_counts,
        )
        joint_link_mask = torch.zeros(
            (self.num_dof, num_links), device=device, dtype=torch.bool
        )
        joint_link_mask[
            output_joint_indices, self.joint_links_data.to(dtype=torch.long)
        ] = True
        tool_link_mask = joint_link_affects_link[
            :, self.tool_frame_map.to(dtype=torch.long)
        ].T
        tool_joint_link_mask = tool_link_mask[:, None, :] & joint_link_mask[None, :, :]
        tool_joint_link_indices = torch.nonzero(tool_joint_link_mask, as_tuple=False)
        self.tool_joint_links_data = tool_joint_link_indices[:, 2].to(dtype=torch.int16)
        tool_joint_link_counts = torch.sum(
            tool_joint_link_mask, dim=-1, dtype=torch.int16
        ).reshape(-1)
        self.tool_joint_links_offsets = torch.zeros(
            (len(self.tool_frames) * self.num_dof + 1,),
            device=device,
            dtype=torch.int16,
        )
        self.tool_joint_links_offsets[1:] = torch.cumsum(
            tool_joint_link_counts, dim=0, dtype=torch.int16
        )

    def _compute_link_level_order(self):
        """Compute BFS level-order grouping of links from link_map.

        Groups links by depth in the kinematic tree. Links at the same depth have no
        parent-child dependencies within the same level, enabling parallel processing
        in CUDA kernels.

        Sets link_level_data, link_level_offsets, and max_level_width.
        """
        num_links = len(self.link_map)
        link_map_cpu = self.link_map.cpu().numpy()

        # Compute depth of each link
        depth = [0] * num_links
        for k in range(num_links):
            parent = int(link_map_cpu[k])
            if parent >= 0 and parent != k:
                depth[k] = depth[parent] + 1

        max_depth = max(depth) if depth else 0
        n_levels = max_depth + 1

        # Group links by depth
        levels = [[] for _ in range(n_levels)]
        for k in range(num_links):
            levels[depth[k]].append(k)

        # Build CSR arrays
        offsets = [0]
        data = []
        max_width = 0
        for level in levels:
            data.extend(level)
            offsets.append(len(data))
            max_width = max(max_width, len(level))

        device = self.link_map.device
        self.link_level_data = torch.tensor(data, dtype=torch.int16, device=device)
        self.link_level_offsets = torch.tensor(offsets, dtype=torch.int16, device=device)
        self.max_level_width = max_width

    def clone(self) -> KinematicsParams:
        """Clone the kinematics tensor config."""
        return KinematicsParams(
            fixed_transforms=self.fixed_transforms.clone(),
            link_map=self.link_map.clone(),
            joint_map=self.joint_map.clone(),
            joint_map_type=self.joint_map_type.clone(),
            joint_offset_map=self.joint_offset_map.clone(),
            tool_frame_map=self.tool_frame_map.clone(),
            link_chain_data=self.link_chain_data.clone(),
            link_chain_offsets=self.link_chain_offsets.clone(),
            joint_links_data=self.joint_links_data.clone(),
            joint_links_offsets=self.joint_links_offsets.clone(),
            tool_frames=self.tool_frames.copy(),
            joint_limits=self.joint_limits.clone(),
            non_fixed_joint_names=self.non_fixed_joint_names.copy(),
            num_dof=self.num_dof,
            joint_link_influence_words=self.joint_link_influence_words.clone(),
            tool_joint_links_data=self.tool_joint_links_data.clone(),
            tool_joint_links_offsets=self.tool_joint_links_offsets.clone(),
            mesh_link_names=self.mesh_link_names.copy(),
            joint_names=self.joint_names.copy(),
            lock_jointstate=(
                self.lock_jointstate.clone() if self.lock_jointstate is not None else None
            ),
            mimic_joints=self.mimic_joints.copy() if self.mimic_joints is not None else None,
            link_spheres=self.link_spheres.clone() if self.link_spheres is not None else None,
            sphere_collision_mask=(
                self.sphere_collision_mask.clone()
                if self.sphere_collision_mask is not None else None
            ),
            link_sphere_idx_map=(
                self.link_sphere_idx_map.clone() if self.link_sphere_idx_map is not None else None
            ),
            link_name_to_idx_map=(
                self.link_name_to_idx_map.copy() if self.link_name_to_idx_map is not None else None
            ),
            total_spheres=self.total_spheres,
            debug=self.debug.copy() if self.debug is not None else None,
            base_link=self.base_link,
            cspace=self.cspace.clone() if self.cspace is not None else None,
            reference_link_spheres=(
                self.reference_link_spheres.clone()
                if self.reference_link_spheres is not None
                else None
            ),
            link_masses_com=(
                self.link_masses_com.clone() if self.link_masses_com is not None else None
            ),
            link_inertias=self.link_inertias.clone() if self.link_inertias is not None else None,
            grasp_contact_link_names=self.grasp_contact_link_names.copy() if self.grasp_contact_link_names is not None else None,
            link_level_data=(
                self.link_level_data.clone() if self.link_level_data is not None else None
            ),
            link_level_offsets=(
                self.link_level_offsets.clone() if self.link_level_offsets is not None else None
            ),
            max_level_width=self.max_level_width,
        )

    @property
    def all_link_names(self) -> List[str]:
        """Get all link names in the kinematic tree."""
        return list(self.link_name_to_idx_map.keys())

    def make_contiguous(self):
        """Return a contiguous copy of the kinematics tensor config."""
        if self.fixed_transforms is not None and not self.fixed_transforms.is_contiguous():
            self.fixed_transforms = self.fixed_transforms.contiguous()
        if self.link_map is not None and not self.link_map.is_contiguous():
            self.link_map = self.link_map.contiguous()
        if self.joint_map is not None and not self.joint_map.is_contiguous():
            self.joint_map = self.joint_map.contiguous()
        if self.joint_map_type is not None and not self.joint_map_type.is_contiguous():
            self.joint_map_type = self.joint_map_type.contiguous()
        if self.tool_frame_map is not None and not self.tool_frame_map.is_contiguous():
            self.tool_frame_map = self.tool_frame_map.contiguous()
        if self.link_chain_data is not None and not self.link_chain_data.is_contiguous():
            self.link_chain_data = self.link_chain_data.contiguous()
        if self.link_chain_offsets is not None and not self.link_chain_offsets.is_contiguous():
            self.link_chain_offsets = self.link_chain_offsets.contiguous()
        if self.joint_links_data is not None and not self.joint_links_data.is_contiguous():
            self.joint_links_data = self.joint_links_data.contiguous()
        if self.joint_links_offsets is not None and not self.joint_links_offsets.is_contiguous():
            self.joint_links_offsets = self.joint_links_offsets.contiguous()
        if (
            self.joint_link_influence_words is not None
            and not self.joint_link_influence_words.is_contiguous()
        ):
            self.joint_link_influence_words = self.joint_link_influence_words.contiguous()
        if self.tool_joint_links_data is not None and not self.tool_joint_links_data.is_contiguous():
            self.tool_joint_links_data = self.tool_joint_links_data.contiguous()
        if self.tool_joint_links_offsets is not None and not self.tool_joint_links_offsets.is_contiguous():
            self.tool_joint_links_offsets = self.tool_joint_links_offsets.contiguous()
        if self.joint_offset_map is not None and not self.joint_offset_map.is_contiguous():
            self.joint_offset_map = self.joint_offset_map.contiguous()
        if self.link_sphere_idx_map is not None and not self.link_sphere_idx_map.is_contiguous():
            self.link_sphere_idx_map = self.link_sphere_idx_map.contiguous()
        if self.link_spheres is not None and not self.link_spheres.is_contiguous():
            self.link_spheres = self.link_spheres.contiguous()
        if (
            self.reference_link_spheres is not None
            and not self.reference_link_spheres.is_contiguous()
        ):
            self.reference_link_spheres = self.reference_link_spheres.contiguous()
        if self.link_masses_com is not None and not self.link_masses_com.is_contiguous():
            self.link_masses_com = self.link_masses_com.contiguous()
        if self.link_inertias is not None and not self.link_inertias.is_contiguous():
            self.link_inertias = self.link_inertias.contiguous()
        if self.link_level_data is not None and not self.link_level_data.is_contiguous():
            self.link_level_data = self.link_level_data.contiguous()
        if self.link_level_offsets is not None and not self.link_level_offsets.is_contiguous():
            self.link_level_offsets = self.link_level_offsets.contiguous()

    def copy_(self, new_config: KinematicsParams) -> KinematicsParams:
        """Copy parameters from another instance into current instance.

        This maintains reference and copies the data. Assumes that the new instance has the same
        number of joints as the current instance and also same shape of tensors.

        Args:
            new_config: New parameters to copy into current instance.

        Returns:
            KinematicsParams: Same instance of kinematics configuration which has updated
                parameters.
        """
        self.fixed_transforms.copy_(new_config.fixed_transforms)
        self.link_map.copy_(new_config.link_map)
        self.joint_map.copy_(new_config.joint_map)
        self.joint_map_type.copy_(new_config.joint_map_type)
        self.tool_frame_map.copy_(new_config.tool_frame_map)
        self.link_chain_data.copy_(new_config.link_chain_data)
        self.link_chain_offsets.copy_(new_config.link_chain_offsets)
        self.joint_links_data.copy_(new_config.joint_links_data)
        self.joint_links_offsets.copy_(new_config.joint_links_offsets)
        self.joint_link_influence_words.copy_(new_config.joint_link_influence_words)
        self.tool_joint_links_data.copy_(new_config.tool_joint_links_data)
        self.tool_joint_links_offsets.copy_(new_config.tool_joint_links_offsets)

        self.joint_limits.copy_(new_config.joint_limits)
        self.joint_offset_map.copy_(new_config.joint_offset_map)
        if new_config.link_spheres is not None and self.link_spheres is not None:
            self._validate_sphere_collision_mask()
            new_config._validate_sphere_collision_mask()
            if self.sphere_collision_mask.shape != new_config.sphere_collision_mask.shape:
                log_and_raise("cannot change collision mask layout during copy_")
            self.link_spheres.copy_(new_config.link_spheres)
            self.sphere_collision_mask.copy_(new_config.sphere_collision_mask)
        if new_config.link_sphere_idx_map is not None and self.link_sphere_idx_map is not None:
            self.link_sphere_idx_map.copy_(new_config.link_sphere_idx_map)
        if new_config.link_name_to_idx_map is not None and self.link_name_to_idx_map is not None:
            self.link_name_to_idx_map = new_config.link_name_to_idx_map.copy()
        if (
            new_config.reference_link_spheres is not None
            and self.reference_link_spheres is not None
        ):
            self.reference_link_spheres.copy_(new_config.reference_link_spheres)
        self.base_link = new_config.base_link
        self.debug = new_config.debug
        self.cspace.copy_(new_config.cspace)
        self.num_dof = new_config.num_dof
        self.non_fixed_joint_names = new_config.non_fixed_joint_names
        self.joint_names = new_config.joint_names
        self.tool_frames = new_config.tool_frames
        self.mesh_link_names = new_config.mesh_link_names
        self.total_spheres = new_config.total_spheres
        if self.lock_jointstate is not None and new_config.lock_jointstate is not None:
            self.lock_jointstate.copy_(new_config.lock_jointstate)
        self.mimic_joints = new_config.mimic_joints
        if new_config.link_masses_com is not None and self.link_masses_com is not None:
            self.link_masses_com.copy_(new_config.link_masses_com)
        if new_config.link_inertias is not None and self.link_inertias is not None:
            self.link_inertias.copy_(new_config.link_inertias)
        if new_config.grasp_contact_link_names is not None and self.grasp_contact_link_names is not None:
            self.grasp_contact_link_names = new_config.grasp_contact_link_names.copy()
        if new_config.link_level_data is not None and self.link_level_data is not None:
            self.link_level_data.copy_(new_config.link_level_data)
        if new_config.link_level_offsets is not None and self.link_level_offsets is not None:
            self.link_level_offsets.copy_(new_config.link_level_offsets)
        self.max_level_width = new_config.max_level_width
        if self._dynamics_model is not None:
            self._dynamics_model.refresh(self)
        self.validate_shapes()
        return self

    def load_cspace_cfg_from_kinematics(self):
        """Load CSpace configuration from joint limits.

        This sets the default joint configuration to the middle of the joint limits
        and all weights to 1.
        """
        default_joint_position = (
            (self.joint_limits.position[1] + self.joint_limits.position[0]) / 2
        ).flatten()
        null_space_weight = torch.ones(self.num_dof, **(self.device_cfg.as_torch_dict()))
        cspace_distance_weight = torch.ones(self.num_dof, **(self.device_cfg.as_torch_dict()))
        joint_names = self.joint_names
        self.cspace = CSpaceParams(
            joint_names,
            default_joint_position,
            cspace_distance_weight,
            null_space_weight,
            device_cfg=self.device_cfg,
            max_acceleration=self.joint_limits.acceleration[1],
            max_jerk=self.joint_limits.jerk[1],
        )

    def get_sphere_index_from_link_name(self, link_name: str) -> torch.Tensor:
        """Get indices of spheres for a link.

        Args:
            link_name: Name of the link.

        Returns:
            torch.Tensor: Indices of spheres for the link.
        """
        link_idx = self.link_name_to_idx_map[link_name]
        link_spheres_idx = torch.nonzero(self.link_sphere_idx_map == link_idx).view(-1)
        return link_spheres_idx

    def update_link_spheres(
        self,
        link_name: str,
        sphere_position_radius: torch.Tensor,
        start_sph_idx: int = 0,
        config_idx: Optional[int] = None,
    ):
        """Update sphere parameters of a specific link given by name.

        Args:
            link_name: Name of the link.
            sphere_position_radius: Tensor of shape [num_spheres, 4] with columns [x, y, z, r].
            start_sph_idx: If providing a subset of spheres, this is the starting index.
            config_idx: Sphere config to update. None updates all configs.
        """
        link_sphere_index = self.get_sphere_index_from_link_name(link_name)[
            start_sph_idx : start_sph_idx + sphere_position_radius.shape[0]
        ]
        if config_idx is None:
            self.link_spheres[:, link_sphere_index, :] = sphere_position_radius
        else:
            self.link_spheres[config_idx, link_sphere_index, :] = sphere_position_radius

    def get_link_spheres(
        self,
        link_name: str,
        config_idx: int = 0,
    ) -> torch.Tensor:
        """Get spheres of a link for a specific config.

        Args:
            link_name: Name of link.
            config_idx: Which sphere config to read from.

        Returns:
            torch.Tensor: Spheres of the link with shape [num_spheres, 4] with columns [x, y, z, r].
        """
        link_sphere_index = self.get_sphere_index_from_link_name(link_name)
        return self.link_spheres[config_idx, link_sphere_index, :]

    def get_reference_link_spheres(
        self,
        link_name: str,
        config_idx: int = 0,
    ) -> torch.Tensor:
        """Get link spheres from the original robot configuration data before any modifications.

        Args:
            link_name: Name of link.
            config_idx: Which sphere config to read from.

        Returns:
            torch.Tensor: Spheres of the link with shape [num_spheres, 4] with columns [x, y, z, r].
        """
        link_sphere_index = self.get_sphere_index_from_link_name(link_name)
        return self.reference_link_spheres[config_idx, link_sphere_index, :]

    def get_number_of_spheres(self, link_name: str) -> int:
        """Get number of spheres for a link

        Args:
            link_name: name of link
        """
        return self.get_link_spheres(link_name).shape[0]

    def disable_link_spheres(self, link_name: str):
        """Disable spheres of a link by setting radius to negative across all configs.

        Args:
            link_name: Name of the link to disable spheres.
        """
        if link_name not in self.link_name_to_idx_map.keys():
            log_and_raise(link_name + " not found in spheres")
        link_sphere_index = self.get_sphere_index_from_link_name(link_name)
        self.link_spheres[:, link_sphere_index, 3] = -100.0

    def enable_link_spheres(self, link_name: str):
        """Enable spheres of a link by restoring radius from reference across all configs.

        Only restores radius, not positions. Use reset_link_spheres for full restore.

        Args:
            link_name: Name of the link to enable spheres.
        """
        if link_name not in self.link_name_to_idx_map.keys():
            log_and_raise(link_name + " not found in spheres")
        link_sphere_index = self.get_sphere_index_from_link_name(link_name)
        self.link_spheres[:, link_sphere_index, 3] = self.reference_link_spheres[
            :, link_sphere_index, 3
        ]

    def reset_link_spheres(self, link_name: str):
        """Reset spheres of a link to reference values (positions + radius) across all configs.

        Args:
            link_name: Name of the link to reset.
        """
        if link_name not in self.link_name_to_idx_map.keys():
            log_and_raise(link_name + " not found in spheres")
        link_sphere_index = self.get_sphere_index_from_link_name(link_name)
        self.link_spheres[:, link_sphere_index, :] = self.reference_link_spheres[
            :, link_sphere_index, :
        ]


    def get_link_masses_com(self, link_name: str) -> torch.Tensor:
        """Get link masses and center of mass for a link

        Args:
            link_name: name of link

        Returns:
            link_masses_com: link masses and center of mass for a link [4]
        """
        if self.link_masses_com is None:
            log_and_raise("link_masses_com is not set")
        if link_name not in self.link_name_to_idx_map.keys():
            log_and_raise(link_name + " not found in link_masses_com")
        return self.link_masses_com[self.link_name_to_idx_map[link_name]]

    def update_link_mass(self, link_name: str, mass: float):
        """Update link mass for a link

        Args:
            link_name: name of link
            mass: mass of link
        """
        if self.link_masses_com is None:
            log_and_raise("link_masses_com is not set")
        if link_name not in self.link_name_to_idx_map.keys():
            log_and_raise(link_name + " not found in link_masses_com")
        self.link_masses_com[self.link_name_to_idx_map[link_name], 3] = mass

    def update_link_com(self, link_name: str, com: torch.Tensor):
        """Update link center of mass for a link

        Args:
            link_name: name of link
            com: center of mass of link [3]
        """
        if self.link_masses_com is None:
            log_and_raise("link_masses_com is not set")
        if link_name not in self.link_name_to_idx_map.keys():
            log_and_raise(link_name + " not found in link_masses_com")
        if com.shape != (3,):
            log_and_raise(f"com shape does not match: {com.shape} != (3,)")
        self.link_masses_com[self.link_name_to_idx_map[link_name], :3] = com

    def update_link_inertia(self, link_name: str, inertia: torch.Tensor):
        """Update link inertia for a link

        Args:
            link_name: name of link
            inertia: inertia of link [6] - [ixx, iyy, izz, ixy, ixz, iyz]
        """
        if self.link_inertias is None:
            log_and_raise("link_inertias is not set")
        if link_name not in self.link_name_to_idx_map.keys():
            log_and_raise(link_name + " not found in link_inertias")
        if inertia.shape != (6,):
            log_and_raise(f"inertia shape does not match: {inertia.shape} != (6,)")
        # Internal storage is padded to 8 floats for CUDA float4 alignment
        self.link_inertias[self.link_name_to_idx_map[link_name], :6] = inertia

    def get_link_inertia(self, link_name: str) -> torch.Tensor:
        """Get link inertia for a link

        Args:
            link_name: name of link

        Returns:
            Inertia tensor [6] - [ixx, iyy, izz, ixy, ixz, iyz]
        """
        if self.link_inertias is None:
            log_and_raise("link_inertias is not set")
        if link_name not in self.link_name_to_idx_map.keys():
            log_and_raise(link_name + " not found in link_inertias")
        # Internal storage is padded to 8 floats; return only the 6 meaningful elements
        return self.link_inertias[self.link_name_to_idx_map[link_name], :6]

    def _validate_sphere_collision_mask(self) -> None:
        """Validate metadata without reading device values or replacing shared storage."""
        masks = self.sphere_collision_mask
        if (
            masks is None
            or masks.shape != self.link_spheres.shape[:2]
            or masks.dtype != torch.uint32
            or masks.device != self.link_spheres.device
            or not masks.is_contiguous()
        ):
            log_and_raise("sphere_collision_mask must be contiguous uint32 [num_envs,num_spheres]")

    def set_link_collision_mask(
        self, link_name: str, layers: int, config_idx: Optional[int] = None
    ) -> None:
        """Set one link's layer masks; None updates every environment in place."""
        self._validate_sphere_collision_mask()
        bits = layer_bits_to_int32(layers)
        if config_idx is not None and not 0 <= config_idx < self.num_envs:
            log_and_raise("sphere environment index is out of range")
        link_idx = self.link_name_to_idx_map[link_name]
        selected = self.link_sphere_idx_map == link_idx
        # Torch does not implement uint32 indexed assignment. This view aliases
        # the same allocation and preserves all unsigned bits, including bit 31.
        masks = self.sphere_collision_mask.view(torch.int32)
        if config_idx is None:
            masks.masked_fill_(selected.unsqueeze(0), bits)
        else:
            masks[config_idx].masked_fill_(selected, bits)

    def get_robot_collision_geometry(self) -> RobotCollisionGeometry:
        """Get robot collision geometry"""
        if self.link_spheres is not None:
            self._validate_sphere_collision_mask()
        robot_collision_geometry = RobotCollisionGeometry(
            link_sphere_idx_map=self.link_sphere_idx_map,
            num_links=self.num_links,
            sphere_collision_mask=self.sphere_collision_mask,
        )
        return robot_collision_geometry

    @property
    def num_pose_links(self) -> int:
        """Get number of links to store in the buffers"""
        return len(self.tool_frames)

    @property
    def num_links(self) -> int:
        """Get number of joints to store in the buffers"""
        return self.link_map.shape[0]

    @property
    def num_spheres(self) -> int:
        """Get number of spheres per config."""
        if self.link_spheres is None:
            return 0
        return self.link_spheres.shape[-2]

    @property
    def num_envs(self) -> int:
        """Get number of environment configurations (K dimension of link_spheres)."""
        if self.link_spheres is None:
            return 0
        return self.link_spheres.shape[0]

    @property
    def n_tree_levels(self) -> int:
        """Get number of depth levels in the kinematic tree."""
        if self.link_level_offsets is None:
            return 0
        return len(self.link_level_offsets) - 1

    def export_to_urdf(
        self,
        robot_name: str = "robot",
        output_path: Optional[str] = None,
        include_spheres: bool = False,
        kinematics_parser=None,
    ):
        """Export KinematicsParams to URDF format.

        This method converts the kinematics configuration to URDF format, including:
        - Link hierarchy and names
        - Joint types, axes, limits, and transforms
        - Inertial properties (mass, center of mass, inertia tensors)
        - Optional sphere collision geometries
        - Optional visual meshes (when kinematics_parser is provided)

        Args:
            robot_name: Name for the robot in the URDF.
            output_path: Path to write the URDF file. If None, returns URDF object without writing.
            include_spheres: Whether to include sphere collision geometries in the URDF.
            kinematics_parser: Optional robot parser (e.g. UrdfRobotParser) used to
                query visual mesh geometry per link. When provided, links that have
                mesh visuals in the original URDF will include ``<visual>`` elements
                in the exported URDF.

        Returns:
            yourdfpy.URDF object if successful, None if required data is missing.

        Raises:
            ValueError: If required data (masses, inertias) is missing.

        Example:
            >>> urdf = kinematics_config.export_to_urdf(
            ...     robot_name="my_robot",
            ...     output_path="/tmp/robot.urdf",
            ...     include_spheres=True
            ... )
        """
        import numpy as np
        import yourdfpy
        from yourdfpy.urdf import Collision, Geometry, Inertial, Joint, Limit, Link, Robot, Sphere

        # Validate required data for dynamics
        if self.link_masses_com is None:
            log_and_raise("link_masses_com is required for URDF export with inertial properties")
            return None
        if self.link_inertias is None:
            log_and_raise("link_inertias is required for URDF export with inertial properties")
            return None

        # Helper functions
        def _joint_type_to_urdf_type(joint_type_value: int):
            """Convert CuRobo JointType to URDF joint type and axis."""
            joint_type = JointType(joint_type_value)
            if joint_type == JointType.FIXED:
                return "fixed", None  # np.array([0.0, 0.0, 1.0])
            elif joint_type == JointType.X_PRISM or joint_type == JointType.X_PRISM_NEG:
                return "prismatic", np.array([1.0, 0.0, 0.0])
            elif joint_type == JointType.Y_PRISM or joint_type == JointType.Y_PRISM_NEG:
                return "prismatic", np.array([0.0, 1.0, 0.0])
            elif joint_type == JointType.Z_PRISM or joint_type == JointType.Z_PRISM_NEG:
                return "prismatic", np.array([0.0, 0.0, 1.0])
            elif joint_type == JointType.X_ROT or joint_type == JointType.X_ROT_NEG:
                return "revolute", np.array([1.0, 0.0, 0.0])
            elif joint_type == JointType.Y_ROT or joint_type == JointType.Y_ROT_NEG:
                return "revolute", np.array([0.0, 1.0, 0.0])
            elif joint_type == JointType.Z_ROT or joint_type == JointType.Z_ROT_NEG:
                return "revolute", np.array([0.0, 0.0, 1.0])
            else:
                return "fixed", None  # np.array([0.0, 0.0, 1.0])

        def _create_inertial(link_idx: int) -> Inertial:
            """Create URDF Inertial object from link mass and inertia data."""
            # Get mass and COM
            mass_com = self.link_masses_com[link_idx].cpu().numpy()
            mass = float(mass_com[3])
            com_xyz = mass_com[:3]

            # Create COM origin transform
            com_origin = np.eye(4)
            com_origin[0:3, 3] = com_xyz

            # Get inertia tensor
            inertia_data = self.link_inertias[link_idx].cpu().numpy()

            return Inertial(
                mass=mass,
                origin=com_origin,
                inertia=np.array(
                    [
                        [inertia_data[0], inertia_data[3], inertia_data[4]],
                        [inertia_data[3], inertia_data[1], inertia_data[5]],
                        [inertia_data[4], inertia_data[5], inertia_data[2]],
                    ]
                ),
            )

        def _get_link_spheres(link_idx: int):
            """Get spheres for a specific link (uses config 0)."""
            if self.link_spheres is None or not include_spheres:
                return []

            sphere_mask = self.link_sphere_idx_map == link_idx
            link_spheres = self.link_spheres[0, sphere_mask]

            # Create collision objects
            collisions = []
            for i, sphere_data in enumerate(link_spheres):
                if sphere_data[3] > 0:  # Only add spheres with positive radius
                    sphere_origin = np.eye(4)
                    sphere_origin[0:3, 3] = sphere_data[0:3].cpu().numpy()

                    collision = Collision(
                        name=f"collision_{i}",
                        origin=sphere_origin,
                        geometry=Geometry(
                            sphere=Sphere(radius=float(sphere_data[3].cpu().numpy()))
                        ),
                    )
                    collisions.append(collision)

            return collisions

        def _get_link_visuals(link_name: str):
            """Get visual elements (with meshes and materials) from the parser."""
            if kinematics_parser is None:
                return []
            try:
                robot_model = kinematics_parser._robot
                link_data = robot_model.link_map[link_name]
            except (KeyError, AttributeError):
                return []
            return list(link_data.visuals)

        # Create links
        links = []
        for link_name in self.all_link_names:
            link_idx = self.link_name_to_idx_map[link_name]

            # Get collision geometries (spheres)
            collisions = _get_link_spheres(link_idx)

            # Create inertial properties
            inertial = _create_inertial(link_idx)

            # Get visual meshes from the parser if available
            visuals = _get_link_visuals(link_name)

            link = Link(
                name=link_name,
                collisions=collisions,
                inertial=inertial,
                visuals=visuals,
            )
            links.append(link)

        # Create joints
        joints = []
        used_joint_names = set()
        for link_name in self.all_link_names:
            link_idx = self.link_name_to_idx_map[link_name]

            # Skip the base link (it has no parent)
            if link_name == self.base_link:
                continue

            # Get parent link index
            parent_link_idx = int(self.link_map[link_idx].cpu().numpy())
            parent_link_name = self.all_link_names[parent_link_idx]

            # Get joint information
            joint_idx = int(self.joint_map[link_idx].cpu().numpy())
            joint_type_value = int(self.joint_map_type[link_idx].cpu().numpy())

            # Convert to URDF joint type and axis
            urdf_joint_type, axis = _joint_type_to_urdf_type(joint_type_value)

            # A joint authored with a negative URDF axis (e.g. <axis xyz="-1 0 0"/>) is
            # stored by UrdfRobotParser as the POSITIVE joint type plus a -1 joint-offset
            # scale (parser_urdf.py), so _joint_type_to_urdf_type returns the positive axis.
            # Re-apply that stored sign here, otherwise the exported axis loses it and FK on
            # the exported URDF rotates the opposite way for the same joint value.
            if axis is not None and joint_idx >= 0:
                offset_scale = float(self.joint_offset_map[2 * link_idx].cpu().numpy())
                if offset_scale < 0:
                    axis = -axis

            # Get fixed transform (origin)
            fixed_transform = self.fixed_transforms[link_idx].cpu().numpy()
            origin = np.eye(4)
            origin[0:3, :] = fixed_transform

            # Create joint name, ensuring uniqueness (mimic joints share the
            # same joint_map index as their source, which would produce duplicates)
            if joint_idx >= 0 and joint_idx < len(self.joint_names):
                joint_name = self.joint_names[joint_idx]
                if joint_name in used_joint_names:
                    joint_name = f"{parent_link_name}_to_{link_name}_joint"
            else:
                joint_name = f"{parent_link_name}_to_{link_name}_joint"
            used_joint_names.add(joint_name)

            # Get joint limits if available
            limit = None
            if urdf_joint_type in ["revolute", "prismatic"] and joint_idx >= 0:
                if joint_idx < self.joint_limits.position.shape[1]:
                    lower = float(self.joint_limits.position[0, joint_idx].cpu().numpy())
                    upper = float(self.joint_limits.position[1, joint_idx].cpu().numpy())
                    velocity = float(self.joint_limits.velocity[1, joint_idx].cpu().numpy())

                    effort = 100.0  # Default effort
                    if (
                        self.joint_limits.effort is not None
                        and joint_idx < self.joint_limits.effort.shape[1]
                    ):
                        effort = float(self.joint_limits.effort[1, joint_idx].cpu().numpy())

                    limit = Limit(lower=lower, upper=upper, velocity=velocity, effort=effort)

            joint = Joint(
                name=joint_name,
                type=urdf_joint_type,
                parent=parent_link_name,
                child=link_name,
                origin=origin,
                axis=axis,
                limit=limit,
            )
            joints.append(joint)

        # Create robot and URDF
        robot = Robot(name=robot_name, links=links, joints=joints)

        urdf = yourdfpy.URDF(robot=robot, build_scene_graph=False, load_meshes=False)

        # Write to file if path provided
        if output_path is not None:
            urdf.write_xml_file(output_path)

        return urdf
