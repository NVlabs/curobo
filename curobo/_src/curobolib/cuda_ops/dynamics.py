# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""PyTorch autograd function for RNEA inverse dynamics.

Wraps the CUDA RNEA forward kernel in a torch.autograd.Function for
seamless integration with PyTorch's automatic differentiation.

The forward kernel saves intermediates (v, a, f) to a forward_cache
tensor, which the backward kernel loads instead of recomputing.
R and p are recomputed in the backward kernel from fixed_transforms + q.

External forces (f_ext) are supported:
  - Forward: f[k] = I·a + v×*(I·v) - f_ext[k]
  - Backward: grad_f_ext[k] = -f_bar[k]

When f_ext is an optimization variable (requires_grad=True), gradients
are computed and propagated through the RNEA backward kernel.
"""

from typing import Optional

import torch
from torch.autograd import Function

from curobo._src.curobolib.backends import dynamics as dynamics_cu
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float32_tensors,
    check_int16_tensors,
    check_int8_tensors,
)
from curobo._src.types.device_cfg import DeviceCfg

# Must match CACHE_FLOATS_PER_LINK in dynamics_constants.h
_CACHE_FLOATS_PER_LINK = 20


class RNEAForwardFunction(Function):
    """Autograd function for RNEA forward pass (inverse dynamics).

    Forward: launches CUDA kernel to compute τ = RNEA(q, q̇, q̈) and saves
             intermediates (v, a, f, R, p) to forward_cache.
    Backward: launches CUDA RNEA backward (VJP) kernel using cached data.
    """

    @staticmethod
    def create_buffers(
        batch_size: int,
        num_dof: int,
        num_links: int,
        device_cfg: DeviceCfg = DeviceCfg(),
        with_external_forces: bool = False,
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

        Returns:
            Dictionary with 'tau', 'grad_q', 'grad_qd', 'grad_qdd',
            'forward_cache', and optionally 'grad_f_ext' buffers.
        """
        shape = (batch_size, num_dof)
        device = device_cfg.device
        dtype = device_cfg.dtype

        buffers = {
            "tau": torch.zeros(shape, device=device, dtype=dtype),
            "grad_q": torch.zeros(shape, device=device, dtype=dtype),
            "grad_qd": torch.zeros(shape, device=device, dtype=dtype),
            "grad_qdd": torch.zeros(shape, device=device, dtype=dtype),
            "forward_cache": torch.zeros(
                (batch_size, num_links * _CACHE_FLOATS_PER_LINK),
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
        joint_offset_map: torch.Tensor,
        gravity: torch.Tensor,
        level_starts: torch.Tensor,
        level_links: torch.Tensor,
        num_links: int,
        num_dof: int,
        n_levels: int,
        threads_per_batch: int,
        f_ext: Optional[torch.Tensor] = None,
        grad_f_ext_buf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute inverse dynamics using RNEA CUDA kernel.

        Also saves forward intermediates (v, a, f) to forward_cache
        for use by the backward kernel. R and p are recomputed in backward.

        Args:
            ctx: Autograd context for saving tensors.
            q: [batch_size, num_dof] Joint positions.
            qd: [batch_size, num_dof] Joint velocities.
            qdd: [batch_size, num_dof] Joint accelerations.
            tau: [batch_size, num_dof] Pre-allocated output buffer.
            grad_q_buf: [batch_size, num_dof] Pre-allocated backward buffer for dL/dq.
            grad_qd_buf: [batch_size, num_dof] Pre-allocated backward buffer for dL/dqd.
            grad_qdd_buf: [batch_size, num_dof] Pre-allocated backward buffer for dL/dqdd.
            forward_cache: [batch_size, num_links * 20] Cache for backward reuse.
            fixed_transforms: [num_links, 3, 4] Static transforms.
            link_masses_com: [num_links, 4] Mass and CoM per link.
            link_inertias: [num_links, 8] Inertia tensors per link (padded).
            joint_map_type: [num_links] Joint types (int8).
            joint_map: [num_links] Link→joint index (int16).
            link_map: [num_links] Parent link index (int16).
            joint_offset_map: [num_links * 2] Mimic joint parameters.
            gravity: [6] Spatial gravity vector.
            level_starts: [n_levels + 1] Tree-level CSR offsets (int16).
            level_links: [num_links] Link indices sorted by level (int16).
            num_links: Number of links.
            num_dof: Number of DOF.
            n_levels: Number of tree depth levels.
            threads_per_batch: Threads per batch element for tree-parallel execution.
            f_ext: [batch_size, num_links, 6] External spatial wrenches (optional).
            grad_f_ext_buf: [batch_size, num_links, 6] Pre-allocated buffer for dL/df_ext.

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
        )
        check_int8_tensors(device, joint_map_type=joint_map_type)
        check_int16_tensors(
            device,
            joint_map=joint_map,
            link_map=link_map,
            level_starts=level_starts,
            level_links=level_links,
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
        ctx.joint_offset_map = joint_offset_map
        ctx.gravity = gravity
        ctx.level_starts = level_starts
        ctx.level_links = level_links
        ctx.num_links = num_links
        ctx.num_dof = num_dof
        ctx.n_levels = n_levels
        ctx.threads_per_batch = threads_per_batch

        return tau

    @staticmethod
    def backward(ctx, grad_tau: torch.Tensor):
        """Backward pass: compute dL/dq, dL/dqd, dL/dqdd (and optionally dL/df_ext).

        Launches the CUDA RNEA backward (VJP) kernel which loads forward
        intermediates from forward_cache and performs the 2-pass adjoint sweep.

        Uses pre-allocated gradient buffers stored in ctx. Buffers are zeroed
        here via cudaMemsetAsync (tensor.zero_()) before kernel launch, faster
        than per-thread scalar stores inside the kernel.

        Args:
            grad_tau: [batch_size, num_dof] Upstream gradient dL/dtau.

        Returns:
            Tuple of gradients matching forward() signature.
            q, qd, qdd, and optionally f_ext get gradients; others return None.
        """
        saved = ctx.saved_tensors
        q, qd, forward_cache = saved[0], saved[1], saved[2]
        f_ext = saved[3] if len(saved) > 3 else None
        batch_size = qd.shape[0]

        grad_q = ctx.grad_q_buf[:batch_size]
        grad_qd = ctx.grad_qd_buf[:batch_size]
        grad_qdd = ctx.grad_qdd_buf[:batch_size]

        # Zero via cudaMemsetAsync: uses copy engine, not compute threads
        grad_q.zero_()
        grad_qd.zero_()
        grad_qdd.zero_()

        # Prepare grad_f_ext if f_ext requires grad
        grad_f_ext = None
        if ctx.f_ext_needs_grad and ctx.grad_f_ext_buf is not None:
            grad_f_ext = ctx.grad_f_ext_buf[:batch_size]
            grad_f_ext.zero_()

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
            grad_f_ext=grad_f_ext,
        )

        # Return gradients for all forward() args:
        # q, qd, qdd, tau, grad_q_buf, grad_qd_buf, grad_qdd_buf,
        # forward_cache, fixed_transforms, link_masses_com, link_inertias,
        # joint_map_type, joint_map, link_map, joint_offset_map, gravity,
        # level_starts, level_links,
        # num_links, num_dof, n_levels, threads_per_batch, f_ext, grad_f_ext_buf
        return (
            grad_q, grad_qd, grad_qdd,
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
            None,  # joint_offset_map
            None,  # gravity
            None,  # level_starts
            None,  # level_links
            None,  # num_links
            None,  # num_dof
            None,  # n_levels
            None,  # threads_per_batch
            grad_f_ext,  # f_ext gradient
            None,  # grad_f_ext_buf (buffer)
        )
