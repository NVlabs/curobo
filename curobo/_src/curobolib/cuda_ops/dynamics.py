# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""PyTorch autograd functions for native rigid-body dynamics.

Wraps the CUDA RNEA and CRBA kernels in :class:`torch.autograd.Function`.

The forward kernel saves intermediates (v, a, f) to a forward_cache tensor.
Cooperative tree kernels also persist R and p; serial kernels recompute them.

External forces (f_ext) are supported:
  - Forward: f[k] = I·a + v×*(I·v) - f_ext[k]
  - Backward: grad_f_ext[k] = -f_bar[k]

When f_ext is an optimization variable (requires_grad=True), gradients
are computed and propagated through the RNEA backward kernel.
"""

# Standard Library
from typing import Optional

# Third Party
import torch
from torch.autograd import Function

# CuRobo
from curobo._src.curobolib.backends import dynamics as dynamics_cu
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float32_tensors,
    check_int8_tensors,
    check_int16_tensors,
)
from curobo._src.types.device_cfg import DeviceCfg

# Must match the cache strides in dynamics_constants.h.
_CACHE_FLOATS_PER_LINK = 20
_CACHE_FLOATS_PER_LINK_WITH_TRANSFORMS = 32


class CRBAForwardFunction(Function):
    """Autograd function for the joint-space mass matrix computed by CRBA.

    The forward pass launches CRBA. The backward pass launches a direct reverse
    CRBA and returns the first-order VJP with respect to joint positions.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        mass_matrix: torch.Tensor,
        grad_q_buf: torch.Tensor,
        fixed_transforms: torch.Tensor,
        link_masses_com: torch.Tensor,
        link_inertias: torch.Tensor,
        joint_map_type: torch.Tensor,
        joint_map: torch.Tensor,
        link_map: torch.Tensor,
        chain_map: torch.Tensor,
        joint_offset_map: torch.Tensor,
        level_starts: torch.Tensor,
        level_links: torch.Tensor,
        num_links: int,
        num_dof: int,
        n_levels: int,
        mass_matrix_threads_per_batch: int,
        all_movable: bool,
        direct_joints: bool,
    ) -> torch.Tensor:
        """Compute ``M(q)`` with shape ``[batch_size, num_dof, num_dof]``."""
        batch_size = q.shape[0]
        device = q.device
        check_float32_tensors(
            device,
            q=q,
            mass_matrix=mass_matrix,
            grad_q_buf=grad_q_buf,
            fixed_transforms=fixed_transforms,
            link_masses_com=link_masses_com,
            link_inertias=link_inertias,
            joint_offset_map=joint_offset_map,
        )
        check_int8_tensors(device, joint_map_type=joint_map_type)
        check_int16_tensors(
            device,
            joint_map=joint_map,
            link_map=link_map,
            chain_map=chain_map,
            level_starts=level_starts,
            level_links=level_links,
        )
        dynamics_cu.launch_crba_forward(
            mass_matrix=mass_matrix,
            q=q,
            fixed_transforms=fixed_transforms,
            link_masses_com=link_masses_com,
            link_inertias=link_inertias,
            joint_map_type=joint_map_type,
            joint_map=joint_map,
            link_map=link_map,
            joint_offset_map=joint_offset_map,
            level_starts=level_starts,
            level_links=level_links,
            batch_size=batch_size,
            num_links=num_links,
            num_dof=num_dof,
            n_levels=n_levels,
            threads_per_batch=mass_matrix_threads_per_batch,
            all_movable=all_movable,
            direct_joints=direct_joints,
        )

        ctx.save_for_backward(q)
        ctx.grad_q_buf = grad_q_buf
        ctx.fixed_transforms = fixed_transforms
        ctx.link_masses_com = link_masses_com
        ctx.link_inertias = link_inertias
        ctx.joint_map_type = joint_map_type
        ctx.joint_map = joint_map
        ctx.link_map = link_map
        ctx.chain_map = chain_map
        ctx.joint_offset_map = joint_offset_map
        ctx.level_starts = level_starts
        ctx.level_links = level_links
        ctx.num_links = num_links
        ctx.num_dof = num_dof
        ctx.n_levels = n_levels
        ctx.mass_matrix_threads_per_batch = mass_matrix_threads_per_batch
        ctx.all_movable = all_movable
        ctx.direct_joints = direct_joints
        return mass_matrix

    @staticmethod
    def backward(ctx, grad_mass_matrix: torch.Tensor) -> tuple:
        """Compute the first-order position VJP with reverse CRBA."""
        (q,) = ctx.saved_tensors
        batch_size = q.shape[0]
        grad_q = ctx.grad_q_buf[:batch_size]
        dynamics_cu.launch_crba_backward(
            grad_q=grad_q,
            grad_mass_matrix=grad_mass_matrix.contiguous(),
            q=q,
            fixed_transforms=ctx.fixed_transforms,
            link_masses_com=ctx.link_masses_com,
            link_inertias=ctx.link_inertias,
            joint_map_type=ctx.joint_map_type,
            joint_map=ctx.joint_map,
            link_map=ctx.link_map,
            chain_map=ctx.chain_map,
            joint_offset_map=ctx.joint_offset_map,
            level_starts=ctx.level_starts,
            level_links=ctx.level_links,
            batch_size=batch_size,
            num_links=ctx.num_links,
            num_dof=ctx.num_dof,
            n_levels=ctx.n_levels,
            threads_per_batch=ctx.mass_matrix_threads_per_batch,
            all_movable=ctx.all_movable,
            direct_joints=ctx.direct_joints,
        )
        return (grad_q,) + (None,) * 18


class RNEAForwardFunction(Function):
    """Autograd function for RNEA forward pass (inverse dynamics).

    Forward: launches CUDA kernel to compute τ = RNEA(q, q̇, q̈) and saves
             the intermediates needed by the selected backward specialization.
    Backward: launches CUDA RNEA backward (VJP) kernel using cached data.
    """

    @staticmethod
    def create_buffers(
        batch_size: int,
        num_dof: int,
        num_links: int,
        device_cfg: DeviceCfg = DeviceCfg(),
        with_external_forces: bool = False,
        threads_per_batch: int = 1,
    ) -> dict:
        """Create output buffers for RNEA forward and backward passes.

        The backward kernel zeros all gradient elements internally, so these
        buffers can be reused across calls without Python-side zeroing.

        Args:
            batch_size: Number of batch elements.
            num_dof: Number of degrees of freedom.
            num_links: Number of links in the kinematic tree.
            device_cfg: Device configuration.
            with_external_forces: If True, also create grad_f_ext buffer.
            threads_per_batch: Threads cooperating on one batch element. Values
                above one allocate transform persistence in the forward cache.

        Returns:
            Dictionary with 'tau', 'grad_q', 'grad_qd', 'grad_qdd',
            'forward_cache', and optionally 'grad_f_ext' buffers.
        """
        shape = (batch_size, num_dof)
        device = device_cfg.device
        dtype = device_cfg.dtype

        cache_stride = (
            _CACHE_FLOATS_PER_LINK_WITH_TRANSFORMS
            if threads_per_batch > 1
            else _CACHE_FLOATS_PER_LINK
        )
        buffers = {
            "tau": torch.zeros(shape, device=device, dtype=dtype),
            "grad_q": torch.zeros(shape, device=device, dtype=dtype),
            "grad_qd": torch.zeros(shape, device=device, dtype=dtype),
            "grad_qdd": torch.zeros(shape, device=device, dtype=dtype),
            "forward_cache": torch.zeros(
                (batch_size, num_links * cache_stride),
                device=device,
                dtype=dtype,
            ),
        }

        if with_external_forces:
            buffers["grad_f_ext"] = torch.zeros(
                (batch_size, num_links, 6),
                device=device,
                dtype=dtype,
            )

        return buffers

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        tau: torch.Tensor,
        grad_q_buf: torch.Tensor,
        grad_qd_buf: torch.Tensor,
        grad_qdd_buf: torch.Tensor,
        forward_cache: torch.Tensor,
        fixed_transforms: torch.Tensor,
        link_masses_com: torch.Tensor,
        link_inertias: torch.Tensor,
        joint_map_type: torch.Tensor,
        joint_map: torch.Tensor,
        link_map: torch.Tensor,
        chain_map: torch.Tensor,
        joint_offset_map: torch.Tensor,
        gravity: torch.Tensor,
        level_starts: torch.Tensor,
        level_links: torch.Tensor,
        num_links: int,
        num_dof: int,
        n_levels: int,
        threads_per_batch: int,
        all_movable: bool,
        direct_joints: bool,
        external_force_link_map: torch.Tensor,
        external_force_starts: torch.Tensor,
        external_force_links: torch.Tensor,
        external_force_transforms: torch.Tensor,
        external_force_owners: torch.Tensor,
        num_original_links: int,
        project_external_forces: bool,
        f_ext: Optional[torch.Tensor] = None,
        grad_f_ext_buf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute inverse dynamics using RNEA CUDA kernel.

        Also saves forward intermediates to forward_cache for the backward kernel.
        Cooperative trees persist R and p; serial trees recompute them.

        Args:
            ctx: Autograd context for saving tensors.
            q: [batch_size, num_dof] Joint positions.
            qd: [batch_size, num_dof] Joint velocities.
            qdd: [batch_size, num_dof] Joint accelerations.
            tau: [batch_size, num_dof] Pre-allocated output buffer.
            grad_q_buf: [batch_size, num_dof] Pre-allocated backward buffer for dL/dq.
            grad_qd_buf: [batch_size, num_dof] Pre-allocated backward buffer for dL/dqd.
            grad_qdd_buf: [batch_size, num_dof] Pre-allocated backward buffer for dL/dqdd.
            forward_cache: [batch_size, num_links * (20 or 32)] cache for backward reuse.
            fixed_transforms: [num_links, 3, 4] Static transforms.
            link_masses_com: [num_links, 4] Mass and CoM per link.
            link_inertias: [num_links, 8] Inertia tensors per link (padded).
            joint_map_type: [num_links] Joint types (int8).
            joint_map: [num_links] Link→joint index (int16).
            link_map: [num_links] Parent link index (int16).
            chain_map: [num_links] Next link in the same maximal chain, or -1.
            joint_offset_map: [num_links * 2] Mimic joint parameters.
            gravity: [6] Spatial gravity vector.
            level_starts: [n_levels + 1] Tree-level CSR offsets (int16).
            level_links: [num_links] Link indices sorted by level (int16).
            num_links: Number of links.
            num_dof: Number of DOF.
            n_levels: Number of tree depth levels.
            threads_per_batch: Threads per batch element for tree-parallel execution.
            all_movable: Whether every link maps to a movable joint.
            direct_joints: Whether link indices directly index identity-mapped joints.
            external_force_link_map: Original link index for each retained body.
            external_force_starts: CSR offsets for projected fixed-link wrenches.
            external_force_links: Original fixed-link indices grouped by retained owner.
            external_force_transforms: Packed owner-to-original transforms.
            external_force_owners: Retained owner index for each original link.
            num_original_links: Link dimension of the public external-force tensor.
            project_external_forces: Whether the dynamics topology was collapsed.
            f_ext: [batch_size, num_original_links, 6] External spatial wrenches.
            grad_f_ext_buf: Matching pre-allocated buffer for dL/df_ext.

        Returns:
            tau: [batch_size, num_dof] Joint torques.
        """
        batch_size = q.shape[0]

        device = q.device
        check_float32_tensors(
            device,
            q=q,
            qd=qd,
            qdd=qdd,
            tau=tau,
            grad_q_buf=grad_q_buf,
            grad_qd_buf=grad_qd_buf,
            grad_qdd_buf=grad_qdd_buf,
            forward_cache=forward_cache,
            fixed_transforms=fixed_transforms,
            link_masses_com=link_masses_com,
            link_inertias=link_inertias,
            gravity=gravity,
            joint_offset_map=joint_offset_map,
            external_force_transforms=external_force_transforms,
        )
        check_int8_tensors(device, joint_map_type=joint_map_type)
        check_int16_tensors(
            device,
            joint_map=joint_map,
            link_map=link_map,
            chain_map=chain_map,
            level_starts=level_starts,
            level_links=level_links,
            external_force_link_map=external_force_link_map,
            external_force_starts=external_force_starts,
            external_force_links=external_force_links,
            external_force_owners=external_force_owners,
        )
        if f_ext is not None:
            check_float32_tensors(device, f_ext=f_ext)
        if grad_f_ext_buf is not None:
            check_float32_tensors(device, grad_f_ext_buf=grad_f_ext_buf)

        dynamics_cu.launch_rnea_forward(
            tau=tau,
            q=q,
            qd=qd,
            qdd=qdd,
            fixed_transforms=fixed_transforms,
            link_masses_com=link_masses_com,
            link_inertias=link_inertias,
            joint_map_type=joint_map_type,
            joint_map=joint_map,
            link_map=link_map,
            chain_map=chain_map,
            joint_offset_map=joint_offset_map,
            gravity=gravity,
            level_starts=level_starts,
            level_links=level_links,
            forward_cache=forward_cache,
            batch_size=batch_size,
            num_links=num_links,
            num_dof=num_dof,
            n_levels=n_levels,
            threads_per_batch=threads_per_batch,
            all_movable=all_movable,
            direct_joints=direct_joints,
            external_force_link_map=external_force_link_map,
            external_force_starts=external_force_starts,
            external_force_links=external_force_links,
            external_force_transforms=external_force_transforms,
            num_original_links=num_original_links,
            project_external_forces=project_external_forces,
            f_ext=f_ext,
        )

        # Save for backward pass: q needed for R,p recomputation, qd for adjoint
        # f_ext saved only if it requires grad
        f_ext_needs_grad = f_ext is not None and f_ext.requires_grad
        ctx.save_for_backward(q, qd, forward_cache, f_ext if f_ext_needs_grad else None)
        ctx.grad_q_buf = grad_q_buf
        ctx.grad_qd_buf = grad_qd_buf
        ctx.grad_qdd_buf = grad_qdd_buf
        ctx.grad_f_ext_buf = grad_f_ext_buf
        ctx.f_ext_needs_grad = f_ext_needs_grad
        ctx.fixed_transforms = fixed_transforms
        ctx.link_masses_com = link_masses_com
        ctx.link_inertias = link_inertias
        ctx.joint_map_type = joint_map_type
        ctx.joint_map = joint_map
        ctx.link_map = link_map
        ctx.chain_map = chain_map
        ctx.joint_offset_map = joint_offset_map
        ctx.gravity = gravity
        ctx.level_starts = level_starts
        ctx.level_links = level_links
        ctx.num_links = num_links
        ctx.num_dof = num_dof
        ctx.n_levels = n_levels
        ctx.threads_per_batch = threads_per_batch
        ctx.all_movable = all_movable
        ctx.direct_joints = direct_joints
        ctx.external_force_link_map = external_force_link_map
        ctx.external_force_owners = external_force_owners
        ctx.external_force_transforms = external_force_transforms
        ctx.num_original_links = num_original_links
        ctx.project_external_forces = project_external_forces

        return tau

    @staticmethod
    def backward(ctx, grad_tau: torch.Tensor):
        """Backward pass: compute dL/dq, dL/dqd, dL/dqdd (and optionally dL/df_ext).

        Launches the CUDA RNEA backward (VJP) kernel which loads forward
        intermediates from forward_cache and performs the 2-pass adjoint sweep.

        Uses pre-allocated gradient buffers stored in ctx. The backward kernel
        clears these buffers before accumulating into them.

        Args:
            ctx: Autograd context populated by the forward pass.
            grad_tau: [batch_size, num_dof] Upstream gradient dL/dtau.

        Returns:
            Tuple of gradients matching forward() signature.
            q, qd, qdd, and optionally f_ext get gradients; others return None.
        """
        saved = ctx.saved_tensors
        q, qd, forward_cache = saved[0], saved[1], saved[2]
        batch_size = qd.shape[0]

        grad_q = ctx.grad_q_buf[:batch_size]
        grad_qd = ctx.grad_qd_buf[:batch_size]
        grad_qdd = ctx.grad_qdd_buf[:batch_size]

        # Prepare grad_f_ext if f_ext requires grad
        grad_f_ext = None
        if ctx.f_ext_needs_grad and ctx.grad_f_ext_buf is not None:
            grad_f_ext = ctx.grad_f_ext_buf[:batch_size]

        if not grad_tau.is_contiguous():
            grad_tau = grad_tau.contiguous()

        device = grad_tau.device
        check_float32_tensors(
            device,
            grad_tau=grad_tau,
            grad_q=grad_q,
            grad_qd=grad_qd,
            grad_qdd=grad_qdd,
            q=q,
            qd=qd,
            forward_cache=forward_cache,
        )

        dynamics_cu.launch_rnea_backward(
            grad_q=grad_q,
            grad_qd=grad_qd,
            grad_qdd=grad_qdd,
            grad_tau=grad_tau,
            q=q,
            qd=qd,
            fixed_transforms=ctx.fixed_transforms,
            link_masses_com=ctx.link_masses_com,
            link_inertias=ctx.link_inertias,
            joint_map_type=ctx.joint_map_type,
            joint_map=ctx.joint_map,
            link_map=ctx.link_map,
            chain_map=ctx.chain_map,
            joint_offset_map=ctx.joint_offset_map,
            gravity=ctx.gravity,
            level_starts=ctx.level_starts,
            level_links=ctx.level_links,
            forward_cache=forward_cache,
            batch_size=batch_size,
            num_links=ctx.num_links,
            num_dof=ctx.num_dof,
            n_levels=ctx.n_levels,
            threads_per_batch=ctx.threads_per_batch,
            all_movable=ctx.all_movable,
            direct_joints=ctx.direct_joints,
            external_force_link_map=ctx.external_force_link_map,
            external_force_owners=ctx.external_force_owners,
            external_force_transforms=ctx.external_force_transforms,
            num_original_links=ctx.num_original_links,
            project_external_forces=ctx.project_external_forces,
            grad_f_ext=grad_f_ext,
        )

        # Return gradients for all forward() args:
        # q, qd, qdd, tau, grad_q_buf, grad_qd_buf, grad_qdd_buf,
        # forward_cache, fixed_transforms, link_masses_com, link_inertias,
        # joint_map_type, joint_map, link_map, chain_map, joint_offset_map, gravity,
        # level_starts, level_links,
        # num_links, num_dof, n_levels, threads_per_batch, all_movable, direct_joints,
        # external-force projection tensors and flags, f_ext, grad_f_ext_buf
        return (
            grad_q,
            grad_qd,
            grad_qdd,
            None,  # tau (buffer)
            None,  # grad_q_buf
            None,  # grad_qd_buf
            None,  # grad_qdd_buf
            None,  # forward_cache (buffer)
            None,  # fixed_transforms
            None,  # link_masses_com
            None,  # link_inertias
            None,  # joint_map_type
            None,  # joint_map
            None,  # link_map
            None,  # chain_map
            None,  # joint_offset_map
            None,  # gravity
            None,  # level_starts
            None,  # level_links
            None,  # num_links
            None,  # num_dof
            None,  # n_levels
            None,  # threads_per_batch
            None,  # all_movable
            None,  # direct_joints
            None,  # external_force_link_map
            None,  # external_force_starts
            None,  # external_force_links
            None,  # external_force_transforms
            None,  # external_force_owners
            None,  # num_original_links
            None,  # project_external_forces
            grad_f_ext,  # f_ext gradient
            None,  # grad_f_ext_buf (buffer)
        )
