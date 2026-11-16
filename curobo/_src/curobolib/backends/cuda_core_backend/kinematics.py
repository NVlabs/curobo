# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Runtime compilation and launch helpers for kinematics CUDA kernels."""

# Third Party
import torch
from cuda.core import LaunchConfig

import curobo._src.runtime as curobo_runtime

# CuRobo
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.kinematics_config import (
    KinematicsKernelCfg,
    KinematicsLaunchCfg,
)
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.util.logging import log_and_raise


def _select_kinematics_backward_threads(actual_sm_count: int) -> int:
    """Select the reverse-kinematics block width for the active GPU policy."""
    if curobo_runtime.use_small_gpu_launch_policy(actual_sm_count):
        return 64
    return 128


def launch_kinematics_forward(
    link_pos: torch.Tensor,
    link_quat: torch.Tensor,
    batch_center_of_mass: torch.Tensor,
    global_cumul_mat: torch.Tensor,
    joint_vec: torch.Tensor,
    fixed_transform: torch.Tensor,
    link_masses_com: torch.Tensor,
    joint_map_type: torch.Tensor,
    joint_map: torch.Tensor,
    link_map: torch.Tensor,
    tool_frame_map: torch.Tensor,
    joint_offset_map: torch.Tensor,
    batch_size: int,
    horizon: int,
    n_joints: int,
    compute_com: bool = False,
):
    """Launch forward kinematics without sphere output."""
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = KinematicsKernelCfg()

    num_links = link_map.shape[0]
    n_tool_frames = tool_frame_map.shape[0]
    config = KinematicsLaunchCfg.calculate_forward_config(
        batch_size, num_links, max_threads=256
    )

    pt_stream = torch.cuda.current_stream(joint_vec.device)
    stream = cache.get_stream_wrapper(pt_stream)

    template_args = (
        f"{num_links}, {str(compute_com).lower()}, {n_joints}, {n_tool_frames}"
    )
    kernel_name = (
        f"curobo::kinematics::kinematics_forward_kernel<{template_args}>"
    )
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("forward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    kernel_args = (
        link_pos.data_ptr(),
        link_quat.data_ptr(),
        batch_center_of_mass.data_ptr(),
        global_cumul_mat.data_ptr(),
        joint_vec.data_ptr(),
        fixed_transform.data_ptr(),
        link_masses_com.data_ptr(),
        joint_map_type.data_ptr(),
        joint_map.data_ptr(),
        link_map.data_ptr(),
        tool_frame_map.data_ptr(),
        joint_offset_map.data_ptr(),
        batch_size,
        horizon,
        num_links,
        n_joints,
        n_tool_frames,
    )
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def launch_kinematics_forward_spheres(
    link_pos: torch.Tensor,
    link_quat: torch.Tensor,
    batch_robot_spheres: torch.Tensor,
    batch_center_of_mass: torch.Tensor,
    global_cumul_mat: torch.Tensor,
    joint_vec: torch.Tensor,
    fixed_transform: torch.Tensor,
    robot_spheres: torch.Tensor,
    link_masses_com: torch.Tensor,
    joint_map_type: torch.Tensor,
    joint_map: torch.Tensor,
    link_map: torch.Tensor,
    tool_frame_map: torch.Tensor,
    link_sphere_map: torch.Tensor,
    joint_offset_map: torch.Tensor,
    env_query_idx: torch.Tensor,
    num_envs: int,
    batch_size: int,
    horizon: int,
    n_joints: int,
    num_spheres: int,
    write_global_cumul: bool = True,
    compute_com: bool = False,
):
    """Launch forward kinematics with sphere output."""
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = KinematicsKernelCfg()

    output_threads_per_batch = 32
    num_links = link_map.shape[0]
    n_tool_frames = tool_frame_map.shape[0]
    config = KinematicsLaunchCfg.calculate_forward_config(
        batch_size, num_links, max_threads=256
    )

    pt_stream = torch.cuda.current_stream(joint_vec.device)
    stream = cache.get_stream_wrapper(pt_stream)

    template_args = (
        f"{num_links}, {output_threads_per_batch}, "
        f"{str(write_global_cumul).lower()}, {str(compute_com).lower()}, "
        f"{n_joints}, {num_spheres}, {n_tool_frames}, {num_envs}"
    )
    kernel_name = (
        f"curobo::kinematics::kinematics_forward_spheres_kernel<{template_args}>"
    )
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("forward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    kernel_args = (
        link_pos.data_ptr(),
        link_quat.data_ptr(),
        batch_robot_spheres.data_ptr(),
        batch_center_of_mass.data_ptr(),
        global_cumul_mat.data_ptr(),
        joint_vec.data_ptr(),
        fixed_transform.data_ptr(),
        robot_spheres.data_ptr(),
        link_masses_com.data_ptr(),
        joint_map_type.data_ptr(),
        joint_map.data_ptr(),
        link_map.data_ptr(),
        tool_frame_map.data_ptr(),
        link_sphere_map.data_ptr(),
        joint_offset_map.data_ptr(),
        env_query_idx.data_ptr(),
        batch_size,
        horizon,
        num_spheres,
        num_envs,
        num_links,
        n_joints,
        n_tool_frames,
    )
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def launch_kinematics_forward_spheres_jacobian(
    link_pos: torch.Tensor,
    link_quat: torch.Tensor,
    batch_robot_spheres: torch.Tensor,
    batch_center_of_mass: torch.Tensor,
    batch_jacobian: torch.Tensor,
    global_cumul_mat: torch.Tensor,
    joint_vec: torch.Tensor,
    fixed_transform: torch.Tensor,
    robot_spheres: torch.Tensor,
    link_masses_com: torch.Tensor,
    joint_map_type: torch.Tensor,
    joint_map: torch.Tensor,
    link_map: torch.Tensor,
    tool_frame_map: torch.Tensor,
    link_sphere_map: torch.Tensor,
    tool_joint_links_data: torch.Tensor,
    tool_joint_links_offsets: torch.Tensor,
    joint_offset_map: torch.Tensor,
    env_query_idx: torch.Tensor,
    num_envs: int,
    batch_size: int,
    horizon: int,
    n_joints: int,
    num_spheres: int,
    write_global_cumul: bool = True,
    compute_com: bool = False,
):
    """Launch forward kinematics with sphere and Jacobian output."""
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = KinematicsKernelCfg()

    output_threads_per_batch = 32
    num_links = link_map.shape[0]
    n_tool_frames = tool_frame_map.shape[0]
    config = KinematicsLaunchCfg.calculate_forward_config(
        batch_size, num_links, max_threads=256
    )

    pt_stream = torch.cuda.current_stream(joint_vec.device)
    stream = cache.get_stream_wrapper(pt_stream)

    template_args = (
        f"{num_links}, {output_threads_per_batch}, "
        f"{str(write_global_cumul).lower()}, {str(compute_com).lower()}, "
        f"{n_joints}, {num_spheres}, {n_tool_frames}, {num_envs}"
    )
    kernel_name = (
        "curobo::kinematics::kinematics_forward_spheres_jacobian_kernel"
        f"<{template_args}>"
    )
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("forward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    kernel_args = (
        link_pos.data_ptr(),
        link_quat.data_ptr(),
        batch_robot_spheres.data_ptr(),
        batch_center_of_mass.data_ptr(),
        batch_jacobian.data_ptr(),
        global_cumul_mat.data_ptr(),
        joint_vec.data_ptr(),
        fixed_transform.data_ptr(),
        robot_spheres.data_ptr(),
        link_masses_com.data_ptr(),
        joint_map_type.data_ptr(),
        joint_map.data_ptr(),
        link_map.data_ptr(),
        tool_frame_map.data_ptr(),
        link_sphere_map.data_ptr(),
        tool_joint_links_data.data_ptr(),
        tool_joint_links_offsets.data_ptr(),
        joint_offset_map.data_ptr(),
        env_query_idx.data_ptr(),
        batch_size,
        horizon,
        num_spheres,
        num_envs,
        num_links,
        n_joints,
        n_tool_frames,
    )
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def launch_kinematics_link_wrench_backward(
    grad_out: torch.Tensor,
    grad_nlinks_pos: torch.Tensor,
    grad_nlinks_quat: torch.Tensor,
    grad_spheres: torch.Tensor,
    grad_center_of_mass: torch.Tensor,
    batch_center_of_mass: torch.Tensor,
    global_cumul_mat: torch.Tensor,
    robot_spheres: torch.Tensor,
    link_masses_com: torch.Tensor,
    joint_map_type: torch.Tensor,
    joint_link_influence_words: torch.Tensor,
    joint_links_data: torch.Tensor,
    joint_links_offsets: torch.Tensor,
    joint_offset_map: torch.Tensor,
    tool_frame_map: torch.Tensor,
    link_sphere_map: torch.Tensor,
    env_query_idx: torch.Tensor,
    batch_size: int,
    horizon: int,
    num_envs: int,
    n_joints: int,
    num_spheres: int,
    compute_com: bool,
    backward_threads: int | None = None,
    grad_spheres_addend: torch.Tensor | None = None,
):
    """Launch the sole pose, sphere, and center-of-mass VJP kernel."""
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = KinematicsKernelCfg()
    num_links = global_cumul_mat.shape[-3]
    n_tool_frames = tool_frame_map.shape[0]
    if backward_threads is None:
        device_properties = torch.cuda.get_device_properties(grad_out.device)
        backward_threads = _select_kinematics_backward_threads(
            device_properties.multi_processor_count
        )
    if backward_threads not in (32, 64, 96, 128):
        log_and_raise("backward_threads must be one of 32, 64, 96, or 128")
    config = LaunchConfig(
        grid=batch_size,
        block=backward_threads,
        shmem_size=num_links * (8 + 12) * 4,
    )
    kernel_name = (
        "curobo::kinematics::kinematics_link_wrench_backward_kernel"
        f"<float, {num_links}, {n_joints}, {num_spheres}, {n_tool_frames}, "
        f"{str(compute_com).lower()}, {backward_threads}, "
        f"{str(joint_links_data.shape[0] == n_joints).lower()}>"
    )
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("backward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    pt_stream = torch.cuda.current_stream(grad_out.device)
    stream = cache.get_stream_wrapper(pt_stream)
    launch_kernel(
        kernel_name,
        stream,
        config,
        kernel,
        grad_out.data_ptr(),
        grad_nlinks_pos.data_ptr(),
        grad_nlinks_quat.data_ptr(),
        grad_spheres.data_ptr(),
        0 if grad_spheres_addend is None else grad_spheres_addend.data_ptr(),
        grad_center_of_mass.data_ptr(),
        batch_center_of_mass.data_ptr(),
        global_cumul_mat.data_ptr(),
        robot_spheres.data_ptr(),
        link_masses_com.data_ptr(),
        joint_map_type.data_ptr(),
        joint_link_influence_words.data_ptr(),
        joint_links_data.data_ptr(),
        joint_links_offsets.data_ptr(),
        joint_offset_map.data_ptr(),
        tool_frame_map.data_ptr(),
        link_sphere_map.data_ptr(),
        env_query_idx.data_ptr(),
        batch_size,
        horizon,
        num_envs,
    )


def launch_kinematics_jacobian_backward(
    grad_out: torch.Tensor,
    grad_jacobian: torch.Tensor,
    global_cumul_mat: torch.Tensor,
    joint_map_type: torch.Tensor,
    tool_joint_links_data: torch.Tensor,
    tool_joint_links_offsets: torch.Tensor,
    joint_link_influence_words: torch.Tensor,
    joint_offset_map: torch.Tensor,
    tool_frame_map: torch.Tensor,
    batch_size: int,
    n_joints: int,
    accumulate: bool,
):
    """Launch the direct Jacobian-cotangent contraction."""
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = KinematicsKernelCfg()
    num_links = global_cumul_mat.shape[-3]
    n_tool_frames = tool_frame_map.shape[0]
    config = KinematicsLaunchCfg.calculate_jacobian_vjp_config(batch_size, num_links)
    kernel_name = (
        "curobo::kinematics::kinematics_jacobian_backward_kernel"
        f"<float, {num_links}, {n_joints}, {n_tool_frames}, "
        f"{str(accumulate).lower()}>"
    )
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("backward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )
    pt_stream = torch.cuda.current_stream(grad_out.device)
    stream = cache.get_stream_wrapper(pt_stream)
    launch_kernel(
        kernel_name,
        stream,
        config,
        kernel,
        grad_out.data_ptr(),
        grad_jacobian.data_ptr(),
        global_cumul_mat.data_ptr(),
        joint_map_type.data_ptr(),
        tool_joint_links_data.data_ptr(),
        tool_joint_links_offsets.data_ptr(),
        joint_link_influence_words.data_ptr(),
        joint_offset_map.data_ptr(),
        tool_frame_map.data_ptr(),
        batch_size,
    )
