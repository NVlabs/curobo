# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Differentiable robot dynamics using native CUDA RNEA and CRBA.

No external physics library required; uses KinematicsParams directly.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, replace
from typing import Optional, Union

import torch

import curobo._src.runtime as curobo_runtime
from curobo._src.curobolib.cuda_ops.dynamics import CRBAForwardFunction, RNEAForwardFunction
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float16_tensors,
    check_float32_tensors,
)
from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.robot.types.joint_types import JointType
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise, log_info

_RNEA_POLICY_REFERENCE_SM_COUNT = 170


def _compute_threads_per_batch(
    max_level_width: int,
    num_links: int,
    num_levels: int,
) -> int:
    """Compute threads-per-batch for tree-parallel CUDA kernels.

    Args:
        max_level_width: Maximum number of links at any single tree depth level.
        num_links: Number of links in the tree.
        num_levels: Number of tree depth levels.

    Returns:
        Recommended TPB. Sustained branches use at least 16 workers, growing to
        the next power of two when required and capped at 32. Returns 1 when
        parallel branches are too sparse to repay synchronization.
    """
    has_sustained_parallelism = num_links * 2 >= num_levels * 3
    if max_level_width >= 2 and has_sustained_parallelism:
        tpb = 1
        while tpb < max_level_width:
            tpb *= 2
        return min(max(tpb, 16), 32)
    return 1


def _compute_runtime_threads_per_batch(
    max_level_width: int,
    num_links: int,
    num_levels: int,
    batch_size: int,
    policy_sm_count: int,
) -> int:
    """Select RNEA cooperative width from tree shape and states per SM.

    Args:
        max_level_width: Maximum number of independent chains at one tree level.
        num_links: Number of retained links in the reduced dynamics tree.
        num_levels: Number of maximal-chain dependency levels.
        batch_size: Total number of robot states in the kernel launch.
        policy_sm_count: Effective SM count selected by the runtime policy.

    Returns:
        Threads assigned to each robot state.
    """
    topology_threads = _compute_threads_per_batch(
        max_level_width,
        num_links,
        num_levels,
    )

    # Preserve the measured RTX 5090 thresholds while scaling them by the
    # effective SM count. Cross multiplication avoids rounding states-per-SM.
    def within_scaled_batch(reference_batch_size: int) -> bool:
        return (
            batch_size * _RNEA_POLICY_REFERENCE_SM_COUNT
            <= reference_batch_size * policy_sm_count
        )

    if topology_threads > 1:
        if num_levels > 1:
            return 32 if within_scaled_batch(512) else 16
        return 32 if within_scaled_batch(1024) else 4

    if within_scaled_batch(1024):
        return 32
    if within_scaled_batch(4096):
        return 4
    if within_scaled_batch(8192):
        return 2
    return 1


def _compute_mass_matrix_threads_per_batch(num_dof: int) -> int:
    """Return a warp-bounded power-of-two worker count for CRBA rows."""
    threads_per_batch = 1
    while threads_per_batch < num_dof:
        threads_per_batch *= 2
    return min(threads_per_batch, 32)


@dataclass(frozen=True)
class _DynamicsModelTensors:
    """Minimal tensor model consumed by the RNEA kernels."""

    #: Static parent-to-child transforms with shape ``[num_links, 3, 4]``.
    fixed_transforms: torch.Tensor
    #: Link masses and centers of mass with shape ``[num_links, 4]``.
    link_masses_com: torch.Tensor
    #: Packed center-of-mass inertias with shape ``[num_links, 8]``.
    link_inertias: torch.Tensor
    #: Joint type per link with shape ``[num_links]``.
    joint_map_type: torch.Tensor
    #: Joint index per link with shape ``[num_links]``.
    joint_map: torch.Tensor
    #: Parent index per link with shape ``[num_links]``.
    link_map: torch.Tensor
    #: Next link in the same maximal chain, or -1, with shape ``[num_links]``.
    chain_map: torch.Tensor
    #: Mimic multiplier and offset per link with shape ``[num_links * 2]``.
    joint_offset_map: torch.Tensor
    #: Root and tail link pairs grouped by chain level, shape ``[num_segments * 2]``.
    level_links: torch.Tensor
    #: CSR segment offsets for :attr:`level_links`, shape ``[num_levels + 1]``.
    level_starts: torch.Tensor
    #: Number of retained rigid bodies.
    num_links: int
    #: Number of maximal-chain dependency levels.
    num_levels: int
    #: Maximum number of independent chains at one dependency level.
    max_level_width: int
    #: Cooperative threads assigned to each batch element.
    threads_per_batch: int
    #: Whether every retained link maps to a movable joint.
    all_movable: bool
    #: Whether link indices directly index unit-scale, zero-offset joints.
    direct_joints: bool
    #: Original link index for each retained body, shape ``[num_links]``.
    external_force_link_map: torch.Tensor
    #: CSR offsets for fixed links projected into each retained body.
    external_force_starts: torch.Tensor
    #: Original fixed-link indices grouped by retained owner.
    external_force_links: torch.Tensor
    #: Static owner-to-original transforms, packed as R[9], p[3].
    external_force_transforms: torch.Tensor
    #: Retained owner for each original link, or -1 for world reactions.
    external_force_owners: torch.Tensor
    #: Number of links in the public external-force tensor.
    num_original_links: int
    #: Whether external wrenches require projection into collapsed bodies.
    project_external_forces: bool

    def refresh(self, kinematics: KinematicsParams) -> None:
        """Copy a same-topology model update into the tensors borrowed by all workspaces."""
        updated = _collapse_fixed_bodies(kinematics)
        for name, value in vars(updated).items():
            current = getattr(self, name)
            if isinstance(value, torch.Tensor):
                if current.shape != value.shape:
                    log_and_raise("Dynamics topology changed; recreate the robot model")
            elif current != value:
                log_and_raise("Dynamics topology changed; recreate the robot model")
        for name, value in vars(updated).items():
            if isinstance(value, torch.Tensor):
                getattr(self, name).copy_(value)


def _has_direct_joints(
    joint_map: torch.Tensor,
    joint_offset_map: torch.Tensor,
    num_links: int,
) -> bool:
    """Return whether links and joints have an identity one-to-one mapping."""
    expected = torch.arange(num_links, dtype=joint_map.dtype, device=joint_map.device)
    offsets = joint_offset_map.reshape(-1, 2)
    return bool(
        torch.equal(joint_map, expected)
        and torch.all(offsets[:, 0] == 1.0).item()
        and torch.all(offsets[:, 1] == 0.0).item()
    )


def _build_chain_schedule(
    parents: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build a level schedule over maximal dependency chains."""
    num_links = len(parents)
    children = [[] for _ in range(num_links)]
    for link_index, parent in enumerate(parents):
        if parent >= 0 and parent != link_index:
            children[parent].append(link_index)

    segment_roots = [
        link_index
        for link_index, parent in enumerate(parents)
        if parent < 0 or parent == link_index or len(children[parent]) != 1
    ]
    chain_map = [-1] * num_links
    link_to_segment = [-1] * num_links
    segments = []
    for segment_index, root in enumerate(segment_roots):
        current = root
        while True:
            link_to_segment[current] = segment_index
            if len(children[current]) != 1:
                break
            child = children[current][0]
            chain_map[current] = child
            current = child
        segments.append((root, current))

    segment_depths = []
    for root, _ in segments:
        parent = parents[root]
        if parent < 0 or parent == root:
            segment_depths.append(0)
        else:
            segment_depths.append(segment_depths[link_to_segment[parent]] + 1)

    num_levels = max(segment_depths) + 1
    levels = [
        [index for index, depth in enumerate(segment_depths) if depth == level]
        for level in range(num_levels)
    ]
    level_starts = [0]
    segment_ends = []
    for level in levels:
        level_starts.append(level_starts[-1] + len(level))
        for segment_index in level:
            segment_ends.extend(segments[segment_index])

    return (
        torch.tensor(chain_map, dtype=torch.int16, device=device),
        torch.tensor(level_starts, dtype=torch.int16, device=device),
        torch.tensor(segment_ends, dtype=torch.int16, device=device),
        num_levels,
        max(len(level) for level in levels),
    )


def _skew(vector: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric matrix for a three-vector."""
    x, y, z = vector.unbind()
    zero = torch.zeros((), dtype=vector.dtype)
    return torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )


def _transform_matrix(transform: torch.Tensor) -> torch.Tensor:
    """Convert a ``[3, 4]`` rigid transform to a homogeneous matrix."""
    result = torch.eye(4, dtype=transform.dtype)
    result[:3, :4] = transform
    return result


def _spatial_inertia_matrix(mass_com: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
    """Expand packed mass properties into a ``[6, 6]`` spatial inertia."""
    center = mass_com[:3]
    mass = mass_com[3]
    center_cross = _skew(center)
    center_inertia = torch.stack(
        (
            torch.stack((inertia[0], inertia[3], inertia[4])),
            torch.stack((inertia[3], inertia[1], inertia[5])),
            torch.stack((inertia[4], inertia[5], inertia[2])),
        )
    )
    result = torch.zeros((6, 6), dtype=mass_com.dtype)
    result[:3, :3] = center_inertia + mass * center_cross.T @ center_cross
    result[:3, 3:] = mass * center_cross
    result[3:, :3] = mass * center_cross.T
    result[3:, 3:] = mass * torch.eye(3, dtype=mass_com.dtype)
    return result


def _pack_spatial_inertia(spatial_inertia: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a spatial inertia into center-of-mass parameters used by CUDA."""
    mass = torch.trace(spatial_inertia[3:, 3:]) / 3.0
    if abs(float(mass)) > 1e-12:
        center_cross = spatial_inertia[:3, 3:] / mass
        center_cross = 0.5 * (center_cross - center_cross.T)
        center = torch.stack((center_cross[2, 1], center_cross[0, 2], center_cross[1, 0]))
    else:
        center = torch.zeros(3, dtype=spatial_inertia.dtype)
        center_cross = _skew(center)
    center_inertia = spatial_inertia[:3, :3] - mass * center_cross.T @ center_cross
    center_inertia = 0.5 * (center_inertia + center_inertia.T)
    mass_com = torch.cat((center, mass.reshape(1)))
    packed_inertia = torch.stack(
        (
            center_inertia[0, 0],
            center_inertia[1, 1],
            center_inertia[2, 2],
            center_inertia[0, 1],
            center_inertia[0, 2],
            center_inertia[1, 2],
            torch.zeros((), dtype=spatial_inertia.dtype),
            torch.zeros((), dtype=spatial_inertia.dtype),
        )
    )
    return mass_com, packed_inertia


def _collapse_fixed_bodies(config: KinematicsParams) -> _DynamicsModelTensors:
    """Collapse fixed bodies into movable ancestors while preserving link wrenches."""
    device = config.fixed_transforms.device
    output_dtype = config.fixed_transforms.dtype
    fixed = config.fixed_transforms.detach().cpu().to(torch.float64)
    mass_com = config.link_masses_com.detach().cpu().to(torch.float64)
    inertias = config.link_inertias.detach().cpu().to(torch.float64)
    parents = [int(value) for value in config.link_map.detach().cpu().tolist()]
    joint_types = [int(value) for value in config.joint_map_type.detach().cpu().tolist()]
    num_original_links = len(parents)
    fixed_type = JointType.FIXED.value

    roots = {index for index, parent in enumerate(parents) if parent < 0 or parent == index}
    retained = [index for index, joint_type in enumerate(joint_types) if joint_type != fixed_type]
    if not retained:
        log_and_raise("Inverse dynamics requires at least one movable link")
    retained_set = set(retained)
    old_to_new = {old_index: new_index for new_index, old_index in enumerate(retained)}

    if len(retained) == num_original_links:
        (
            chain_map,
            level_starts,
            level_links,
            num_levels,
            max_level_width,
        ) = _build_chain_schedule(
            parents,
            device,
        )
        identity_matrix = torch.eye(4, dtype=output_dtype, device=device)
        identity_transform = torch.cat(
            (identity_matrix[:3, :3].reshape(9), identity_matrix[:3, 3])
        )
        return _DynamicsModelTensors(
            fixed_transforms=config.fixed_transforms,
            link_masses_com=config.link_masses_com,
            link_inertias=config.link_inertias,
            joint_map_type=config.joint_map_type,
            joint_map=config.joint_map,
            link_map=config.link_map,
            chain_map=chain_map,
            joint_offset_map=config.joint_offset_map,
            level_links=level_links,
            level_starts=level_starts,
            num_links=config.num_links,
            num_levels=num_levels,
            max_level_width=max_level_width,
            threads_per_batch=_compute_threads_per_batch(
                max_level_width,
                config.num_links,
                num_levels,
            ),
            all_movable=bool(torch.all(config.joint_map >= 0).item()),
            direct_joints=_has_direct_joints(
                config.joint_map,
                config.joint_offset_map,
                config.num_links,
            ),
            external_force_link_map=torch.arange(
                num_original_links, dtype=torch.int16, device=device
            ),
            external_force_starts=torch.zeros(
                num_original_links + 1, dtype=torch.int16, device=device
            ),
            external_force_links=torch.empty(0, dtype=torch.int16, device=device),
            external_force_transforms=identity_transform.repeat(num_original_links, 1),
            external_force_owners=torch.arange(
                num_original_links, dtype=torch.int16, device=device
            ),
            num_original_links=num_original_links,
            project_external_forces=False,
        )

    def retained_ancestor(link_index: int) -> Optional[int]:
        current = link_index
        while current not in retained_set:
            parent = parents[current]
            if parent < 0 or parent == current:
                return None
            current = parent
        return current

    def compose_from_ancestor(ancestor: Optional[int], descendant: int) -> torch.Tensor:
        path = []
        current = descendant
        while current != ancestor:
            path.append(current)
            parent = parents[current]
            if parent < 0 or parent == current:
                break
            current = parent
        result = torch.eye(4, dtype=torch.float64)
        for link_index in reversed(path):
            result = result @ _transform_matrix(fixed[link_index])
        return result

    aggregate_inertias = [torch.zeros((6, 6), dtype=torch.float64) for _ in range(len(retained))]
    for link_index in range(num_original_links):
        owner = retained_ancestor(link_index)
        if owner is None:
            continue
        transform = compose_from_ancestor(owner, link_index)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        motion_transform = torch.zeros((6, 6), dtype=torch.float64)
        rotation_transpose = rotation.T
        motion_transform[:3, :3] = rotation_transpose
        motion_transform[3:, :3] = -rotation_transpose @ _skew(translation)
        motion_transform[3:, 3:] = rotation_transpose
        link_inertia = _spatial_inertia_matrix(mass_com[link_index], inertias[link_index])
        aggregate_inertias[old_to_new[owner]] += (
            motion_transform.T @ link_inertia @ motion_transform
        )

    external_force_owners = [-1] * num_original_links
    identity_matrix = torch.eye(4, dtype=torch.float64)
    identity_transform = torch.cat((identity_matrix[:3, :3].reshape(9), identity_matrix[:3, 3]))
    external_force_transforms = [identity_transform] * num_original_links
    fixed_links_by_owner = [[] for _ in retained]
    for link_index in range(num_original_links):
        owner = retained_ancestor(link_index)
        if owner is None:
            continue
        owner_index = old_to_new[owner]
        external_force_owners[link_index] = owner_index
        transform = compose_from_ancestor(owner, link_index)
        external_force_transforms[link_index] = torch.cat(
            (transform[:3, :3].reshape(9), transform[:3, 3])
        )
        if link_index != owner:
            fixed_links_by_owner[owner_index].append(link_index)

    external_force_starts = [0]
    external_force_links = []
    for owned_links in fixed_links_by_owner:
        external_force_links.extend(owned_links)
        external_force_starts.append(len(external_force_links))

    reduced_fixed = []
    reduced_parents = []
    for old_index in retained:
        if old_index in roots:
            reduced_parents.append(old_to_new[old_index])
            reduced_fixed.append(fixed[old_index])
            continue
        old_parent = retained_ancestor(parents[old_index])
        reduced_parents.append(
            old_to_new[old_parent] if old_parent is not None else old_to_new[old_index]
        )
        reduced_fixed.append(compose_from_ancestor(old_parent, old_index)[:3, :4])

    reduced_mass_com = []
    reduced_inertias = []
    for spatial_inertia in aggregate_inertias:
        packed_mass_com, packed_inertia = _pack_spatial_inertia(spatial_inertia)
        reduced_mass_com.append(packed_mass_com)
        reduced_inertias.append(packed_inertia)

    chain_map, level_starts, level_links, num_levels, max_level_width = _build_chain_schedule(
        reduced_parents, device
    )
    threads_per_batch = _compute_threads_per_batch(
        max_level_width,
        level_links.numel() // 2,
        num_levels,
    )

    retained_indices = torch.tensor(retained, dtype=torch.long, device=device)
    reduced_joint_map = config.joint_map.index_select(0, retained_indices).contiguous()
    reduced_joint_offsets = (
        config.joint_offset_map.reshape(-1, 2)
        .index_select(0, retained_indices)
        .reshape(-1)
        .contiguous()
    )
    return _DynamicsModelTensors(
        fixed_transforms=torch.stack(reduced_fixed).to(device=device, dtype=output_dtype),
        link_masses_com=torch.stack(reduced_mass_com).to(device=device, dtype=output_dtype),
        link_inertias=torch.stack(reduced_inertias).to(device=device, dtype=output_dtype),
        joint_map_type=config.joint_map_type.index_select(0, retained_indices).contiguous(),
        joint_map=reduced_joint_map,
        link_map=torch.tensor(reduced_parents, dtype=torch.int16, device=device),
        chain_map=chain_map,
        joint_offset_map=reduced_joint_offsets,
        level_links=level_links,
        level_starts=level_starts,
        num_links=len(retained),
        num_levels=num_levels,
        max_level_width=max_level_width,
        threads_per_batch=threads_per_batch,
        all_movable=bool(torch.all(reduced_joint_map >= 0).item()),
        direct_joints=_has_direct_joints(
            reduced_joint_map,
            reduced_joint_offsets,
            len(retained),
        ),
        external_force_link_map=torch.tensor(retained, dtype=torch.int16, device=device),
        external_force_starts=torch.tensor(
            external_force_starts, dtype=torch.int16, device=device
        ),
        external_force_links=torch.tensor(external_force_links, dtype=torch.int16, device=device),
        external_force_transforms=torch.stack(external_force_transforms).to(
            device=device, dtype=output_dtype
        ),
        external_force_owners=torch.tensor(
            external_force_owners, dtype=torch.int16, device=device
        ),
        num_original_links=num_original_links,
        project_external_forces=True,
    )


class Dynamics:
    """Differentiable robot dynamics using native CUDA RNEA and CRBA kernels.

    Uses cuRobo's KinematicsParams directly, with no URDF export and no external library.
    RNEA forward/backward and CRBA mass-matrix evaluation run as CUDA kernels
    with topology-aware parallelism for branched robots.

    Usage::

        cfg = DynamicsCfg(kinematics_config=kp, device_cfg=device_cfg)
        dynamics = Dynamics(cfg)
        dynamics.setup_batch_size(batch_size=1000)
        tau = dynamics.compute_inverse_dynamics(joint_state)
        mass_matrix = dynamics.compute_mass_matrix(joint_state.position)
    """

    def __init__(self, config: DynamicsCfg) -> None:
        """Initialize dynamics model.

        Args:
            config: Configuration with KinematicsParams and gravity.
        """
        self.config = config
        self.kinematics_config = config.kinematics_config
        self.device_cfg = config.device_cfg
        self.dof = config.kinematics_config.num_dof

        device_str = str(config.device_cfg.device)
        self.device = device_str.split(":")[0] if ":" in device_str else device_str

        self._batch_size = None
        self._horizon = None
        self._total_batch = None

        self._backend_joint_names = None
        self._joint_reorder_indices = None

        self._gravity_spatial = config.get_gravity_spatial()

        kp = config.kinematics_config
        self._fixed_transforms = kp.fixed_transforms
        self._link_masses_com = kp.link_masses_com
        self._link_inertias = kp.link_inertias
        self._joint_map_type = kp.joint_map_type
        self._joint_map = kp.joint_map
        self._link_map = kp.link_map
        self._joint_offset_map = kp.joint_offset_map
        self._n_links = kp.num_links
        self._n_dof = kp.num_dof
        self._mass_matrix_threads_per_batch = _compute_mass_matrix_threads_per_batch(self._n_dof)

        self._level_starts = kp.link_level_offsets
        self._level_links = kp.link_level_data
        self._n_levels = kp.n_tree_levels
        if kp._dynamics_model is None:
            kp._dynamics_model = _collapse_fixed_bodies(kp)
        self._reduced_model = kp._dynamics_model
        self._threads_per_batch = self._reduced_model.threads_per_batch
        self._forward_cache_floats = (
            32 if self._reduced_model.threads_per_batch > 1 else 20
        ) * self._reduced_model.num_links

        self._tau_buffer = None
        self._grad_q_buffer = None
        self._grad_qd_buffer = None
        self._grad_qdd_buffer = None
        self._grad_f_ext_buffer = None
        self._forward_cache = None
        self._mass_matrix_buffer = None
        self._mass_matrix_grad_q_buffer = None

    def _update_runtime_rnea_policy(self) -> None:
        """Apply the tree-, batch-, and SM-aware RNEA cooperative width."""
        threads_per_batch = self._reduced_model.threads_per_batch
        if self._total_batch is not None:
            device_properties = torch.cuda.get_device_properties(self._gravity_spatial.device)
            policy_sm_count = curobo_runtime.resolve_kernel_launch_sm_count(
                device_properties.multi_processor_count
            )
            threads_per_batch = _compute_runtime_threads_per_batch(
                self._reduced_model.max_level_width,
                self._reduced_model.num_links,
                self._reduced_model.num_levels,
                self._total_batch,
                policy_sm_count,
            )
            self._reduced_model = replace(
                self._reduced_model,
                threads_per_batch=threads_per_batch,
            )
        self._threads_per_batch = threads_per_batch
        self._forward_cache_floats = (
            32 if threads_per_batch > 1 else 20
        ) * self._reduced_model.num_links

    def setup_batch_size(self, batch_size: int, horizon: int = 1) -> None:
        """Allocate forward and backward output buffers for the given batch size.

        The backward kernel zeros gradient buffers internally, so these can be
        reused across calls without Python-side zeroing.

        Args:
            batch_size: Number of parallel robot instances.
            horizon: Number of timesteps per batch.
        """
        self._batch_size = batch_size
        self._horizon = horizon
        self._total_batch = batch_size * horizon
        self._update_runtime_rnea_policy()

        shape = (self._total_batch, self._n_dof)
        device = self._gravity_spatial.device

        self._tau_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
        self._grad_q_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
        self._grad_qd_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
        self._grad_qdd_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
        # Cooperative trees also persist R+p(12) after v+a(12) and f(8).
        self._forward_cache = torch.zeros(
            (self._total_batch, self._forward_cache_floats),
            device=device,
            dtype=torch.float32,
        )

        log_info(
            f"Dynamics: setup batch_size={batch_size}, horizon={horizon}, "
            f"num_links={self._n_links}, num_dof={self._n_dof}, "
            f"n_levels={self._n_levels}, tpb={self._threads_per_batch}"
        )

    def _ensure_mass_matrix_buffers(self, batch_size: int, device: torch.device) -> None:
        """Allocate or grow the reusable CRBA forward and backward workspace."""
        if (
            self._mass_matrix_buffer is not None
            and self._mass_matrix_buffer.shape[0] >= batch_size
        ):
            return

        self._mass_matrix_buffer = torch.zeros(
            (batch_size, self._n_dof, self._n_dof),
            device=device,
            dtype=torch.float32,
        )
        self._mass_matrix_grad_q_buffer = torch.zeros(
            (batch_size, self._n_dof),
            device=device,
            dtype=torch.float32,
        )

    def compute_mass_matrix(self, joint_position: torch.Tensor) -> torch.Tensor:
        """Compute the dense joint-space mass matrix ``M(q)`` with CRBA.

        Args:
            joint_position: Joint positions with shape ``[dof]``, ``[batch, dof]``,
                or ``[batch, horizon, dof]``.

        Returns:
            Symmetric mass matrices with shape ``[..., dof, dof]``.

        Raises:
            RuntimeError: If :meth:`setup_batch_size` has not been called.
            ValueError: If the input is not contiguous float32 CUDA data after reshape.
        """
        if self._tau_buffer is None:
            log_and_raise("Call setup_batch_size() before compute_mass_matrix()")
        if joint_position.shape[-1] != self._n_dof:
            log_and_raise(
                f"joint_position: expected last dimension {self._n_dof}, "
                f"got shape={tuple(joint_position.shape)}"
            )

        original_shape = joint_position.shape
        q = joint_position.reshape(-1, self._n_dof)
        check_float32_tensors(q.device, q=q)
        actual_batch = q.shape[0]
        self._ensure_mass_matrix_buffers(actual_batch, q.device)
        model = self._reduced_model
        mass_matrix = CRBAForwardFunction.apply(
            q,
            self._mass_matrix_buffer[:actual_batch],
            self._mass_matrix_grad_q_buffer[:actual_batch],
            model.fixed_transforms,
            model.link_masses_com,
            model.link_inertias,
            model.joint_map_type,
            model.joint_map,
            model.link_map,
            model.chain_map,
            model.joint_offset_map,
            model.level_starts,
            model.level_links,
            model.num_links,
            self._n_dof,
            model.num_levels,
            self._mass_matrix_threads_per_batch,
            model.all_movable,
            model.direct_joints,
        )
        output_shape = tuple(original_shape[:-1]) + (self._n_dof, self._n_dof)
        return mass_matrix.reshape(output_shape)

    def compute_inverse_dynamics(
        self,
        joint_state: JointState,
        f_ext: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute inverse dynamics (joint torques) for given joint state.

        Args:
            joint_state: Joint state with position, velocity, acceleration.
                Shape: [dof], [batch, dof], or [batch, horizon, dof].
            f_ext: External spatial wrenches on each link (optional).
                Shape: [batch, num_links, 6] or [num_links, 6] (broadcast).
                Each f_ext[b,k] is [torque(3), force(3)] in link k's frame.
                Forces are subtracted: f[k] = I*a + v x*(I*v) - f_ext[k].
                When f_ext.requires_grad=True, gradients are computed.

        Returns:
            tau: Joint torques with same shape as input position.

        Raises:
            RuntimeError: If setup_batch_size hasn't been called.
            ValueError: If acceleration is not provided.
        """
        if self._tau_buffer is None:
            log_and_raise("Call setup_batch_size() before compute_inverse_dynamics()")

        if joint_state.acceleration is None:
            log_and_raise("JointState.acceleration is required for inverse dynamics")

        original_shape = joint_state.position.shape
        q = joint_state.position.reshape(-1, self._n_dof)
        qd = joint_state.velocity.reshape(-1, self._n_dof)
        qdd = joint_state.acceleration.reshape(-1, self._n_dof)
        _check_dyn = check_float16_tensors if q.dtype == torch.float16 else check_float32_tensors
        _check_dyn(q.device, q=q, qd=qd, qdd=qdd)

        actual_batch = q.shape[0]

        f_ext_flat = None
        if f_ext is not None:
            if f_ext.dim() == 2:
                f_ext_flat = f_ext.unsqueeze(0).expand(actual_batch, -1, -1).contiguous()
            else:
                f_ext_flat = f_ext.reshape(actual_batch, self._n_links, 6)
            _check_f_ext = (
                check_float16_tensors
                if f_ext_flat.dtype == torch.float16
                else check_float32_tensors
            )
            _check_f_ext(f_ext_flat.device, f_ext_flat=f_ext_flat)

        if self._tau_buffer.shape[0] < actual_batch:
            shape = (actual_batch, self._n_dof)
            device = q.device
            self._tau_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
            self._grad_q_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
            self._grad_qd_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
            self._grad_qdd_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
            self._forward_cache = torch.zeros(
                (actual_batch, self._forward_cache_floats),
                device=device,
                dtype=torch.float32,
            )

        tau = self._tau_buffer[:actual_batch]

        grad_f_ext_buf = None
        if f_ext_flat is not None and f_ext_flat.requires_grad:
            if self._grad_f_ext_buffer is None or self._grad_f_ext_buffer.shape[0] < actual_batch:
                self._grad_f_ext_buffer = torch.zeros(
                    (actual_batch, self._n_links, 6),
                    device=q.device,
                    dtype=torch.float32,
                )
            grad_f_ext_buf = self._grad_f_ext_buffer[:actual_batch]

        model = self._reduced_model

        tau = RNEAForwardFunction.apply(
            q,
            qd,
            qdd,
            tau,
            self._grad_q_buffer[:actual_batch],
            self._grad_qd_buffer[:actual_batch],
            self._grad_qdd_buffer[:actual_batch],
            self._forward_cache[:actual_batch],
            model.fixed_transforms,
            model.link_masses_com,
            model.link_inertias,
            model.joint_map_type,
            model.joint_map,
            model.link_map,
            model.chain_map,
            model.joint_offset_map,
            self._gravity_spatial,
            model.level_starts,
            model.level_links,
            model.num_links,
            self._n_dof,
            model.num_levels,
            model.threads_per_batch,
            model.all_movable,
            model.direct_joints,
            model.external_force_link_map,
            model.external_force_starts,
            model.external_force_links,
            model.external_force_transforms,
            model.external_force_owners,
            model.num_original_links,
            model.project_external_forces,
            f_ext_flat,
            grad_f_ext_buf,
        )

        tau = tau.reshape(original_shape)
        return tau

    def _refresh_reduced_model(self) -> None:
        """Refresh transformed inertias without replacing buffers captured by CUDA graphs."""
        updated = _collapse_fixed_bodies(self.kinematics_config)
        # Inertial updates preserve the tree and launch policy. Captured forward
        # and backward kernels must keep the same addresses for these tensors.
        self._reduced_model.link_masses_com.copy_(updated.link_masses_com)
        self._reduced_model.link_inertias.copy_(updated.link_inertias)

    def _check_and_reorder_joints(self, joint_state: JointState) -> JointState:
        """Check if joint ordering matches backend and reorder if necessary.

        This ensures gradients flow correctly even if joint order differs between
        input and backend expectations.

        Args:
            joint_state: Input joint state with joint_names

        Returns:
            Joint state with joints reordered to match backend model

        Raises:
            ValueError: If joint names don't match backend model.
        """
        if joint_state.joint_names is None:
            return joint_state

        if joint_state.joint_names == self._backend_joint_names:
            return joint_state

        if self._joint_reorder_indices is None:
            self._joint_reorder_indices = []
            for backend_joint in self._backend_joint_names:
                try:
                    idx = joint_state.joint_names.index(backend_joint)
                    self._joint_reorder_indices.append(idx)
                except ValueError:
                    log_and_raise(
                        f"Joint name mismatch: '{backend_joint}' from backend not found "
                        f"in input. Input joints: {joint_state.joint_names}"
                    )

            self._joint_reorder_indices = torch.tensor(
                self._joint_reorder_indices, device=self.device, dtype=torch.long
            )
            log_info(
                f"Joint reordering enabled: {joint_state.joint_names} -> "
                f"{self._backend_joint_names}"
            )

        original_shape = joint_state.position.shape
        if len(original_shape) == 1:
            reordered_position = joint_state.position[self._joint_reorder_indices]
            reordered_velocity = joint_state.velocity[self._joint_reorder_indices]
        elif len(original_shape) == 2:
            reordered_position = joint_state.position[:, self._joint_reorder_indices]
            reordered_velocity = joint_state.velocity[:, self._joint_reorder_indices]
        elif len(original_shape) == 3:
            reordered_position = joint_state.position[:, :, self._joint_reorder_indices]
            reordered_velocity = joint_state.velocity[:, :, self._joint_reorder_indices]
        else:
            log_and_raise(f"Unsupported shape: {original_shape}")

        return JointState(
            position=reordered_position,
            velocity=reordered_velocity,
            acceleration=joint_state.acceleration,
            jerk=joint_state.jerk,
            joint_names=self._backend_joint_names,
        )

    def _get_link_index(self, link_name: str) -> int:
        """Get link index from link name.

        Args:
            link_name: Name of the link.

        Returns:
            int: Link index.

        Raises:
            ValueError: If link name not found.
        """
        link_name_to_idx = self.kinematics_config.link_name_to_idx_map
        if link_name_to_idx is None:
            log_and_raise("link_name_to_idx_map is not set in kinematics_config")
        if link_name not in link_name_to_idx:
            log_and_raise(
                f"Link '{link_name}' not found. Available links: {list(link_name_to_idx.keys())}"
            )
        return link_name_to_idx[link_name]

    def update_link_mass(self, link_name: str, mass: float) -> None:
        """Update mass of a specific link.

        Args:
            link_name: Name of the link to update.
            mass: New mass value in kg.
        """
        if self._link_masses_com is None:
            log_and_raise("link_masses_com is not set in kinematics_config")

        link_idx = self._get_link_index(link_name)

        with torch.no_grad():
            self._link_masses_com[link_idx, 3] = mass
        self._refresh_reduced_model()

        log_info(f"Updated mass of link '{link_name}' to {mass} kg")

    def update_link_com(self, link_name: str, com: torch.Tensor) -> None:
        """Update center of mass of a specific link.

        Args:
            link_name: Name of the link to update.
            com: Center of mass in link frame. Shape: [3].
        """
        if com.shape != (3,):
            log_and_raise(f"COM must have shape [3], got {com.shape}")
        if self._link_masses_com is None:
            log_and_raise("link_masses_com is not set in kinematics_config")

        link_idx = self._get_link_index(link_name)

        with torch.no_grad():
            com_device = com.to(
                device=self._link_masses_com.device,
                dtype=self._link_masses_com.dtype,
            )
            self._link_masses_com[link_idx, :3] = com_device
        self._refresh_reduced_model()

        log_info(f"Updated COM of link '{link_name}' to {com.cpu().numpy()}")

    def update_link_inertia(self, link_name: str, inertia: torch.Tensor) -> None:
        """Update inertia tensor of a specific link.

        Args:
            link_name: Name of the link to update.
            inertia: Inertia tensor. Shape: [6] - [ixx, iyy, izz, ixy, ixz, iyz].
        """
        if inertia.shape != (6,):
            log_and_raise(f"Inertia must have shape [6], got {inertia.shape}")
        if self._link_inertias is None:
            log_and_raise("link_inertias is not set in kinematics_config")

        link_idx = self._get_link_index(link_name)

        with torch.no_grad():
            inertia_device = inertia.to(
                device=self._link_inertias.device,
                dtype=self._link_inertias.dtype,
            )
            self._link_inertias[link_idx, :6] = inertia_device
        self._refresh_reduced_model()

        log_info(f"Updated inertia of link '{link_name}'")

    def update_link_inertial(
        self,
        link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ) -> None:
        """Update inertial properties of a single link.

        Args:
            link_name: Name of the link to update.
            mass: New mass value in kg (optional).
            com: Center of mass in link frame, shape [3] (optional).
            inertia: Inertia tensor, shape [6] - [ixx, iyy, izz, ixy, ixz, iyz] (optional).

        Raises:
            ValueError: If no properties are provided.
        """
        if mass is None and com is None and inertia is None:
            log_and_raise("At least one property (mass, com, or inertia) must be provided")

        if mass is not None:
            self.update_link_mass(link_name, mass)
        if com is not None:
            self.update_link_com(link_name, com)
        if inertia is not None:
            self.update_link_inertia(link_name, inertia)

    def update_links_inertial(
        self,
        link_properties: dict[str, dict[str, Union[float, torch.Tensor]]],
    ) -> None:
        """Update inertial properties of multiple links.

        Args:
            link_properties: Dictionary mapping link names to their properties:
                {
                    "link1": {"mass": 1.0, "com": torch.tensor([0,0,0])},
                    "link2": {"mass": 2.0, "inertia": torch.tensor([...])},
                }
                Each link can specify any subset of: mass, com, inertia.

        Raises:
            ValueError: If link_properties is empty or contains invalid keys.
        """
        if not link_properties:
            log_and_raise("link_properties dictionary cannot be empty")

        updated_links = []

        for link_name, properties in link_properties.items():
            if not properties:
                log_and_raise(f"No properties specified for link '{link_name}'")

            self.update_link_inertial(
                link_name,
                mass=properties.get("mass"),
                com=properties.get("com"),
                inertia=properties.get("inertia"),
            )
            updated_links.append(link_name)

        log_info(
            f"Updated inertial properties for {len(updated_links)} link(s): "
            f"{', '.join(updated_links)}"
        )
