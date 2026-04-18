# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""cuda.core backend for RNEA dynamics kernels.

Provides runtime compilation and launch of RNEA forward and backward kernels.
"""

from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.dynamics_config import (
    DynamicsKernelCfg,
    DynamicsLaunchCfg,
)
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel


def launch_rnea_forward(
    tau: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
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
    forward_cache: torch.Tensor,
    batch_size: int,
    num_links: int,
    num_dof: int,
    n_levels: int,
    threads_per_batch: int = 1,
    f_ext: Optional[torch.Tensor] = None,
):
    """Launch RNEA forward kernel using cuda.core runtime compilation.

    Computes inverse dynamics: τ = RNEA(q, q̇, q̈, f_ext) and saves forward
    intermediates (v, a, f) to forward_cache for backward reuse.

    Args:
        tau: [batch_size, num_dof] Output torques (modified in-place).
        q: [batch_size, num_dof] Joint positions.
        qd: [batch_size, num_dof] Joint velocities.
        qdd: [batch_size, num_dof] Joint accelerations.
        fixed_transforms: [num_links, 3, 4] Static transforms.
        link_masses_com: [num_links, 4] Mass and CoM data.
        link_inertias: [num_links, 8] Inertia tensors at CoM (padded).
        joint_map_type: [num_links] Joint type per link (int8).
        joint_map: [num_links] Link→joint index (int16).
        link_map: [num_links] Parent link index (int16).
        joint_offset_map: [num_links * 2] Mimic joint [multiplier, offset] pairs.
        gravity: [6] Spatial gravity vector.
        level_starts: [n_levels + 1] CSR offsets for tree-level grouping (int16).
        level_links: [num_links] Link indices sorted by tree level (int16).
        forward_cache: [batch_size, num_links, 20] Cache for backward reuse.
        batch_size: Number of batch elements.
        num_links: Number of links.
        num_dof: Number of degrees of freedom.
        n_levels: Number of tree depth levels.
        threads_per_batch: Threads per batch element (1=serial, >1=tree-parallel).
        f_ext: [batch_size, num_links, 6] External spatial wrenches (optional).
               When provided, forces are subtracted: f[k] = I·a + v×*(I·v) - f_ext[k].
               Each f_ext[b,k] is a 6D wrench [torque(3), force(3)] in link k's frame.
    """
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()

    kernel_config = DynamicsKernelCfg()
    config = DynamicsLaunchCfg.calculate_forward_config(
        batch_size, num_links, threads_per_batch
    )

    # Template on N_LINKS, N_DOF, TPB, and HAS_EXTERNAL_FORCES
    has_f_ext = f_ext is not None
    has_f_ext_str = "true" if has_f_ext else "false"
    kernel_name = (
        f"curobo::dynamics::rnea_forward_kernel"
        f"<{num_links}, {num_dof}, {threads_per_batch}, {has_f_ext_str}>"
    )

    kernel_files = [
        kernel_config.kernel_dir / f
        for f in kernel_config.get_kernel_files("forward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    pt_stream = torch.cuda.current_stream(q.device)
    stream = cache.get_stream_wrapper(pt_stream)

    # f_ext pointer: use actual pointer if provided, else 0 (nullptr)
    f_ext_ptr = f_ext.data_ptr() if has_f_ext else 0

    kernel_args = (
        tau.data_ptr(),
        q.data_ptr(),
        qd.data_ptr(),
        qdd.data_ptr(),
        fixed_transforms.data_ptr(),
        link_masses_com.data_ptr(),
        link_inertias.data_ptr(),
        joint_map_type.data_ptr(),
        joint_map.data_ptr(),
        link_map.data_ptr(),
        joint_offset_map.data_ptr(),
        gravity.data_ptr(),
        level_starts.data_ptr(),
        level_links.data_ptr(),
        forward_cache.data_ptr(),
        f_ext_ptr,
        batch_size,
        n_levels,
    )

    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def launch_rnea_backward(
    grad_q: torch.Tensor,
    grad_qd: torch.Tensor,
    grad_qdd: torch.Tensor,
    grad_tau: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
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
    forward_cache: torch.Tensor,
    batch_size: int,
    num_links: int,
    num_dof: int,
    n_levels: int,
    threads_per_batch: int = 1,
    grad_f_ext: Optional[torch.Tensor] = None,
):
    """Launch RNEA backward (VJP) kernel using cuda.core runtime compilation.

    Computes dL/dq, dL/dqd, dL/dqdd from dL/dtau. Uses forward_cache
    (populated by the forward kernel) for v, a, f and recomputes R, p
    from fixed_transforms + q.

    When grad_f_ext is provided, also computes dL/df_ext = -f_bar.

    Args:
        grad_q: [batch_size, num_dof] Output gradient w.r.t. q (modified in-place).
        grad_qd: [batch_size, num_dof] Output gradient w.r.t. qd (modified in-place).
        grad_qdd: [batch_size, num_dof] Output gradient w.r.t. qdd (modified in-place).
        grad_tau: [batch_size, num_dof] Upstream gradient dL/dtau.
        q: [batch_size, num_dof] Joint positions (for R, p recomputation).
        qd: [batch_size, num_dof] Joint velocities.
        fixed_transforms: [num_links, 3, 4] Static transforms (for R, p recomputation).
        link_masses_com: [num_links, 4] Mass and CoM data.
        link_inertias: [num_links, 8] Inertia tensors at CoM (padded).
        joint_map_type: [num_links] Joint type per link (int8).
        joint_map: [num_links] Link→joint index (int16).
        link_map: [num_links] Parent link index (int16).
        joint_offset_map: [num_links * 2] Mimic joint [multiplier, offset] pairs.
        gravity: [6] Spatial gravity vector.
        level_starts: [n_levels + 1] CSR offsets for tree-level grouping (int16).
        level_links: [num_links] Link indices sorted by tree level (int16).
        forward_cache: [batch_size, num_links, 20] Cached forward intermediates.
        batch_size: Number of batch elements.
        num_links: Number of links.
        num_dof: Number of degrees of freedom.
        n_levels: Number of tree depth levels.
        threads_per_batch: Threads per batch element (1=serial, >1=tree-parallel).
        grad_f_ext: [batch_size, num_links, 6] Output gradient w.r.t. external forces.
                    When provided, computes grad_f_ext = -f_bar. Optional.
    """
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()

    kernel_config = DynamicsKernelCfg()
    config = DynamicsLaunchCfg.calculate_backward_config(
        batch_size, num_links, threads_per_batch
    )

    # Template on N_LINKS, N_DOF, TPB, and HAS_EXTERNAL_FORCES
    has_f_ext = grad_f_ext is not None
    has_f_ext_str = "true" if has_f_ext else "false"
    kernel_name = (
        f"curobo::dynamics::rnea_backward_kernel"
        f"<{num_links}, {num_dof}, {threads_per_batch}, {has_f_ext_str}>"
    )

    kernel_files = [
        kernel_config.kernel_dir / f
        for f in kernel_config.get_kernel_files("backward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    pt_stream = torch.cuda.current_stream(qd.device)
    stream = cache.get_stream_wrapper(pt_stream)

    # grad_f_ext pointer: use actual pointer if provided, else 0 (nullptr)
    grad_f_ext_ptr = grad_f_ext.data_ptr() if has_f_ext else 0

    kernel_args = (
        grad_q.data_ptr(),
        grad_qd.data_ptr(),
        grad_qdd.data_ptr(),
        grad_f_ext_ptr,
        grad_tau.data_ptr(),
        q.data_ptr(),
        qd.data_ptr(),
        fixed_transforms.data_ptr(),
        link_masses_com.data_ptr(),
        link_inertias.data_ptr(),
        joint_map_type.data_ptr(),
        joint_map.data_ptr(),
        link_map.data_ptr(),
        joint_offset_map.data_ptr(),
        gravity.data_ptr(),
        level_starts.data_ptr(),
        level_links.data_ptr(),
        forward_cache.data_ptr(),
        batch_size,
        n_levels,
    )

    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)
