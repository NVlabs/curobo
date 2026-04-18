# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Differentiable robot dynamics using native CUDA RNEA.

No external physics library required; uses KinematicsParams directly.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from curobo._src.curobolib.cuda_ops.dynamics import RNEAForwardFunction
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float16_tensors,
    check_float32_tensors,
)
from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise, log_info


def _compute_threads_per_batch(max_level_width: int) -> int:
    """Compute threads-per-batch for tree-parallel CUDA kernels.

    Args:
        max_level_width: Maximum number of links at any single tree depth level.

    Returns:
        Recommended TPB (next power of 2 >= max_level_width, capped at 32).
        Returns 1 for narrow trees (max_level_width < 4) where sync overhead
        exceeds the parallelism benefit.
    """
    if max_level_width >= 4:
        tpb = 1
        while tpb < max_level_width:
            tpb *= 2
        return min(tpb, 32)
    return 1


class Dynamics:
    """Differentiable robot dynamics using native CUDA RNEA kernels.

    Uses cuRobo's KinematicsParams directly, with no URDF export and no external library.
    The RNEA forward and backward passes run as CUDA kernels with tree-level
    parallelism for branched robots.

    Usage::

        cfg = DynamicsCfg(kinematics_config=kp, device_cfg=device_cfg)
        dynamics = Dynamics(cfg)
        dynamics.setup_batch_size(batch_size=1000)
        tau = dynamics.compute_inverse_dynamics(joint_state)
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

        self._level_starts = kp.link_level_offsets
        self._level_links = kp.link_level_data
        self._n_levels = kp.n_tree_levels
        self._threads_per_batch = _compute_threads_per_batch(kp.max_level_width)

        self._tau_buffer = None
        self._grad_q_buffer = None
        self._grad_qd_buffer = None
        self._grad_qdd_buffer = None
        self._grad_f_ext_buffer = None
        self._forward_cache = None

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

        shape = (self._total_batch, self._n_dof)
        device = self._gravity_spatial.device

        self._tau_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
        self._grad_q_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
        self._grad_qd_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
        self._grad_qdd_buffer = torch.zeros(shape, device=device, dtype=torch.float32)
        # Forward cache for backward kernel reuse: 20 floats per link
        # (v+a:12 packed + f:8 padded). R+p recomputed in backward.
        self._forward_cache = torch.zeros(
            (self._total_batch, self._n_links * 20),
            device=device,
            dtype=torch.float32,
        )

        log_info(
            f"Dynamics: setup batch_size={batch_size}, horizon={horizon}, "
            f"num_links={self._n_links}, num_dof={self._n_dof}, "
            f"n_levels={self._n_levels}, tpb={self._threads_per_batch}"
        )

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
        _check_dyn = (
            check_float16_tensors if q.dtype == torch.float16 else check_float32_tensors
        )
        _check_dyn(q.device, q=q, qd=qd, qdd=qdd)

        actual_batch = q.shape[0]

        f_ext_flat = None
        if f_ext is not None:
            if f_ext.dim() == 2:
                f_ext_flat = f_ext.unsqueeze(0).expand(actual_batch, -1, -1)
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
            self._grad_qdd_buffer = torch.zeros(
                shape, device=device, dtype=torch.float32
            )
            self._forward_cache = torch.zeros(
                (actual_batch, self._n_links * 20),
                device=device,
                dtype=torch.float32,
            )

        tau = self._tau_buffer[:actual_batch]

        grad_f_ext_buf = None
        if f_ext_flat is not None and f_ext_flat.requires_grad:
            if (
                self._grad_f_ext_buffer is None
                or self._grad_f_ext_buffer.shape[0] < actual_batch
            ):
                self._grad_f_ext_buffer = torch.zeros(
                    (actual_batch, self._n_links, 6),
                    device=q.device,
                    dtype=torch.float32,
                )
            grad_f_ext_buf = self._grad_f_ext_buffer[:actual_batch]

        tau = RNEAForwardFunction.apply(
            q,
            qd,
            qdd,
            tau,
            self._grad_q_buffer[:actual_batch],
            self._grad_qd_buffer[:actual_batch],
            self._grad_qdd_buffer[:actual_batch],
            self._forward_cache[:actual_batch],
            self._fixed_transforms,
            self._link_masses_com,
            self._link_inertias,
            self._joint_map_type,
            self._joint_map,
            self._link_map,
            self._joint_offset_map,
            self._gravity_spatial,
            self._level_starts,
            self._level_links,
            self._n_links,
            self._n_dof,
            self._n_levels,
            self._threads_per_batch,
            f_ext_flat,
            grad_f_ext_buf,
        )

        tau = tau.reshape(original_shape)
        return tau

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
                f"Link '{link_name}' not found. Available links: "
                f"{list(link_name_to_idx.keys())}"
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
            log_and_raise(
                "At least one property (mass, com, or inertia) must be provided"
            )

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
