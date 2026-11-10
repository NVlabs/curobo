# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""cuda.core backend for geometry/collision kernels.

Only self-collision uses CUDA kernels. World collision (OBB, voxel) uses Warp.
"""

# Standard Library
from typing import Any

# Third Party
import torch
from cuda.core import LaunchConfig

# CuRobo
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.geometry_config import GeometryKernelCfg
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.util import ceil_div
from curobo._src.util.logging import log_and_raise

COLLISION_PAIR_SIZE = 8  # sizeof(CollisionPair): float(4) + int16(2) + int16(2)
STATIC_SMEM_OVERHEAD = COLLISION_PAIR_SIZE * 33  # block_reduce_shared_data[32] + reduced_max_d
SPHERE_SIZE = 16  # sizeof(float4)


def get_self_collision_shared_memory_size(
    num_spheres: int,
    num_collision_links: int = 0,
    sum_collisions: bool = False,
    per_sphere_max_collisions: bool = False,
) -> int:
    """Return total shared-memory bytes for a self-collision kernel.

    The exact kernel stores one ``float4`` per sphere. The link-AABB kernel also
    stores minimum and maximum ``float4`` bounds for every collision link. Summed
    collision costs additionally store one ``float4`` gradient accumulator per sphere.
    Per-sphere-max costs also store one packed 64-bit max-contact entry per sphere.
    """
    return (
        SPHERE_SIZE
        * (num_spheres + 2 * num_collision_links + (num_spheres if sum_collisions else 0))
        + (8 * num_spheres if per_sphere_max_collisions else 0)
        + STATIC_SMEM_OVERHEAD
    )


def get_device_max_shared_memory(device: torch.device) -> int:
    """Return the opt-in shared-memory limit per block for a CUDA device."""
    properties = torch.cuda.get_device_properties(device)
    optin_limit = getattr(properties, "shared_memory_per_block_optin", 0)
    return int(optin_limit or properties.shared_memory_per_block)


def _validate_and_configure_shared_memory(
    kernel: Any,
    dynamic_smemsize: int,
    nspheres: int,
    kernel_name: str,
    device: torch.device,
) -> None:
    import cuda.bindings.runtime as cudart

    total_smem = dynamic_smemsize + STATIC_SMEM_OVERHEAD
    device_max_smem = get_device_max_shared_memory(device)

    if total_smem > device_max_smem:
        log_and_raise(
            f"Self-collision kernel {kernel_name} requires {total_smem} bytes of shared "
            f"memory ({dynamic_smemsize} dynamic + {STATIC_SMEM_OVERHEAD} static) for "
            f"{nspheres} spheres, but device {device} supports at most {device_max_smem} bytes"
        )

    if dynamic_smemsize > 48000:
        kernel_ptr = int(kernel._handle)
        err = cudart.cudaFuncSetAttribute(
            kernel_ptr,
            cudart.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_smemsize,
        )
        if err[0] != cudart.cudaError_t.cudaSuccess:
            log_and_raise(
                f"cudaFuncSetAttribute failed for self-collision kernel {kernel_name}: {err}"
            )


def self_collision_distance(
    out_distance: torch.Tensor,
    out_vec: torch.Tensor,
    pair_distance: torch.Tensor,
    sparse_index: torch.Tensor,
    robot_spheres: torch.Tensor,
    sphere_padding: torch.Tensor,
    weight: torch.Tensor,
    pair_locations: torch.Tensor,
    max_threads_per_block: int,
    batch_size: int,
    horizon: int,
    nspheres: int,
    num_collision_pairs: int,
    store_pair_distance: bool,
    compute_grad: bool,
    sum_collisions: bool,
    per_sphere_max_collisions: bool,
) -> None:
    """Launch one templated exact self-collision kernel."""
    if sum_collisions and per_sphere_max_collisions:
        log_and_raise("Select only one self-collision reduction mode")
    if store_pair_distance and (sum_collisions or per_sphere_max_collisions):
        log_and_raise("Pair-distance storage is supported only for global-max self collision")

    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = GeometryKernelCfg()
    reduction_mode = 2 if per_sphere_max_collisions else 1 if sum_collisions else 0
    num_threads_per_block = min(max_threads_per_block, num_collision_pairs)
    num_threads_per_block = min(
        ceil_div(num_threads_per_block, 32) * 32, max_threads_per_block
    )
    kernel_name = (
        "curobo::geometry::self_collision::self_collision_exact_distance_kernel"
        f"<{reduction_mode}, {str(store_pair_distance).lower()}, "
        f"{str(compute_grad).lower()}, {nspheres}, {num_collision_pairs}, "
        f"{num_threads_per_block}>"
    )
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("self_collision")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )
    smemsize = SPHERE_SIZE * nspheres
    if reduction_mode != 0:
        smemsize += SPHERE_SIZE * nspheres
    if reduction_mode == 2:
        smemsize += 8 * nspheres
    _validate_and_configure_shared_memory(
        kernel, smemsize, nspheres, kernel_name, out_distance.device
    )
    stream = cache.get_stream_wrapper(torch.cuda.current_stream(out_distance.device))
    config = LaunchConfig(
        grid=batch_size * horizon,
        block=num_threads_per_block,
        shmem_size=smemsize,
    )
    kernel_args = (
        out_distance.data_ptr(),
        out_vec.data_ptr(),
        pair_distance.data_ptr(),
        sparse_index.data_ptr(),
        robot_spheres.data_ptr(),
        sphere_padding.data_ptr(),
        weight.data_ptr(),
        pair_locations.data_ptr(),
        batch_size,
        horizon,
        nspheres,
        num_collision_pairs,
    )
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def self_collision_distance_link_aabb(
    out_distance: torch.Tensor,
    out_vec: torch.Tensor,
    sparse_index: torch.Tensor,
    robot_spheres: torch.Tensor,
    sphere_padding: torch.Tensor,
    weight: torch.Tensor,
    pair_locations: torch.Tensor,
    link_sphere_indices: torch.Tensor,
    link_sphere_offsets: torch.Tensor,
    link_pairs: torch.Tensor,
    link_pair_offsets: torch.Tensor,
    max_threads_per_block: int,
    batch_size: int,
    horizon: int,
    nspheres: int,
    num_collision_links: int,
    num_link_pairs: int,
    compute_grad: bool,
    sum_collisions: bool,
    per_sphere_max_collisions: bool,
) -> None:
    """Launch one templated link-AABB self-collision kernel."""
    if sum_collisions and per_sphere_max_collisions:
        log_and_raise("Select only one self-collision reduction mode")
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = GeometryKernelCfg()
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("self_collision")
    ]
    include_dirs = kernel_config.get_include_dirs()
    compile_flags = kernel_config.get_compile_flags()
    pt_stream = torch.cuda.current_stream(out_distance.device)
    stream = cache.get_stream_wrapper(pt_stream)

    reduction_mode = 2 if per_sphere_max_collisions else 1 if sum_collisions else 0
    num_pair_threads = min(max_threads_per_block, max(32, num_link_pairs * 32))
    num_pair_threads = min(max_threads_per_block, ceil_div(num_pair_threads, 32) * 32)
    kernel_name = (
        "curobo::geometry::self_collision::self_collision_link_aabb_distance_kernel"
        f"<{reduction_mode}, {str(compute_grad).lower()}, {nspheres}, "
        f"{num_link_pairs}, {num_collision_links}, {num_pair_threads}>"
    )
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=include_dirs,
        compile_flags=compile_flags,
    )
    smemsize = SPHERE_SIZE * (nspheres + 2 * num_collision_links)
    if reduction_mode != 0:
        smemsize += SPHERE_SIZE * nspheres
    if reduction_mode == 2:
        smemsize += 8 * nspheres
    _validate_and_configure_shared_memory(
        kernel, smemsize, nspheres, kernel_name, out_distance.device
    )
    config = LaunchConfig(
        grid=batch_size * horizon,
        block=num_pair_threads,
        shmem_size=smemsize,
    )
    kernel_args = (
        out_distance.data_ptr(),
        out_vec.data_ptr(),
        sparse_index.data_ptr(),
        robot_spheres.data_ptr(),
        sphere_padding.data_ptr(),
        weight.data_ptr(),
        pair_locations.data_ptr(),
        link_sphere_indices.data_ptr(),
        link_sphere_offsets.data_ptr(),
        link_pairs.data_ptr(),
        link_pair_offsets.data_ptr(),
        batch_size,
        horizon,
        nspheres,
        num_link_pairs,
        num_collision_links,
    )
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)
