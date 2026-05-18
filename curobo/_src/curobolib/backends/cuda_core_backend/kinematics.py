# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""cuda.core backend for kinematics kernels.
Provides runtime compilation of CUDA kernels with the same interface as PyBind11 extensions.
"""

# Third Party
import torch

# CuRobo
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.kinematics_config import (
    KinematicsKernelCfg,
    KinematicsLaunchCfg,
)
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel


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
        batch_size, num_links, threads_per_batch=32, max_threads=256
    )

    pt_stream = torch.cuda.current_stream(joint_vec.device)
    stream = cache.get_stream_wrapper(pt_stream)

    kernel_name = (
        f"curobo::kinematics::"
        f"kinematics_forward_kernel"
        f"<{num_links}, {str(compute_com).lower()}>"
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
    output_threads_per_batch: int,
    write_global_cumul: bool = True,
    compute_com: bool = False,
):
    """Launch forward kinematics with sphere output."""
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = KinematicsKernelCfg()

    if output_threads_per_batch not in (32, 64, 128):
        raise ValueError("output_threads_per_batch must be one of 32, 64, or 128")

    num_links = link_map.shape[0]
    n_tool_frames = tool_frame_map.shape[0]
    config = KinematicsLaunchCfg.calculate_forward_config(
        batch_size, num_links, threads_per_batch=output_threads_per_batch, max_threads=256
    )

    pt_stream = torch.cuda.current_stream(joint_vec.device)
    stream = cache.get_stream_wrapper(pt_stream)

    write_cumul_template_arg = ""
    if not write_global_cumul or compute_com:
        write_cumul_template_arg = f", {str(write_global_cumul).lower()}"
    compute_com_template_arg = ", true" if compute_com else ""
    kernel_name = (
        f"curobo::kinematics::"
        f"kinematics_forward_spheres_kernel<{num_links}, {output_threads_per_batch}"
        f"{write_cumul_template_arg}{compute_com_template_arg}>"
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
    link_chain_data: torch.Tensor,
    link_chain_offsets: torch.Tensor,
    joint_links_data: torch.Tensor,
    joint_links_offsets: torch.Tensor,
    joint_affects_endeffector: torch.Tensor,
    joint_offset_map: torch.Tensor,
    env_query_idx: torch.Tensor,
    num_envs: int,
    batch_size: int,
    horizon: int,
    n_joints: int,
    num_spheres: int,
    output_threads_per_batch: int,
    write_global_cumul: bool = True,
    compute_com: bool = False,
):
    """Launch forward kinematics with sphere and Jacobian output."""
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = KinematicsKernelCfg()

    if output_threads_per_batch not in (32, 64, 128):
        raise ValueError("output_threads_per_batch must be one of 32, 64, or 128")

    num_links = link_map.shape[0]
    n_tool_frames = tool_frame_map.shape[0]
    config = KinematicsLaunchCfg.calculate_forward_config(
        batch_size, num_links, threads_per_batch=output_threads_per_batch, max_threads=256
    )

    pt_stream = torch.cuda.current_stream(joint_vec.device)
    stream = cache.get_stream_wrapper(pt_stream)

    write_cumul_template_arg = ""
    if not write_global_cumul or compute_com:
        write_cumul_template_arg = f", {str(write_global_cumul).lower()}"
    compute_com_template_arg = ", true" if compute_com else ""
    kernel_name = (
        f"curobo::kinematics::"
        f"kinematics_forward_spheres_jacobian_kernel<{num_links}, {output_threads_per_batch}"
        f"{write_cumul_template_arg}{compute_com_template_arg}>"
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
        link_chain_data.data_ptr(),
        link_chain_offsets.data_ptr(),
        joint_links_data.data_ptr(),
        joint_links_offsets.data_ptr(),
        joint_affects_endeffector.data_ptr(),
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


def launch_kinematics_backward(
    grad_out: torch.Tensor,
    grad_nlinks_pos: torch.Tensor,
    grad_nlinks_quat: torch.Tensor,
    grad_spheres: torch.Tensor,
    grad_center_of_mass: torch.Tensor,
    batch_center_of_mass: torch.Tensor,
    grad_jacobian: torch.Tensor,
    global_cumul_mat: torch.Tensor,
    robot_spheres: torch.Tensor,
    link_masses_com: torch.Tensor,
    link_map: torch.Tensor,
    joint_map: torch.Tensor,
    joint_map_type: torch.Tensor,
    tool_frame_map: torch.Tensor,
    link_sphere_map: torch.Tensor,
    link_chain_data: torch.Tensor,
    link_chain_offsets: torch.Tensor,
    joint_links_data: torch.Tensor,
    joint_links_offsets: torch.Tensor,
    joint_affects_endeffector: torch.Tensor,
    joint_offset_map: torch.Tensor,
    env_query_idx: torch.Tensor,
    num_envs: int,
    batch_size: int,
    horizon: int,
    n_joints: int,
    num_spheres: int,
    compute_com: bool,
    compute_jacobian_grad: bool,
):
    """Launch saved-cumul backward with optional fused Jacobian-output gradient."""
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = KinematicsKernelCfg()

    num_links = link_map.shape[0]
    n_tool_frames = tool_frame_map.shape[0]
    config, threads_per_batch, use_warp_reduce, max_joints_template = (
        KinematicsLaunchCfg.calculate_backward_config(
            batch_size, num_links, num_spheres, n_tool_frames, n_joints
        )
    )

    kernel_name = (
        f"curobo::kinematics::"
        f"kinematics_backward_kernel"
        f"<float, float, {max_joints_template}, {str(use_warp_reduce).lower()}, "
        f"{str(compute_com).lower()}, {str(compute_jacobian_grad).lower()}>"
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

    kernel_args = (
        grad_out.data_ptr(),
        grad_nlinks_pos.data_ptr(),
        grad_nlinks_quat.data_ptr(),
        grad_spheres.data_ptr(),
        grad_center_of_mass.data_ptr(),
        batch_center_of_mass.data_ptr(),
        grad_jacobian.data_ptr(),
        global_cumul_mat.data_ptr(),
        robot_spheres.data_ptr(),
        link_masses_com.data_ptr(),
        joint_map_type.data_ptr(),
        joint_map.data_ptr(),
        link_map.data_ptr(),
        tool_frame_map.data_ptr(),
        link_sphere_map.data_ptr(),
        env_query_idx.data_ptr(),
        link_chain_data.data_ptr(),
        link_chain_offsets.data_ptr(),
        joint_links_data.data_ptr(),
        joint_links_offsets.data_ptr(),
        joint_affects_endeffector.data_ptr(),
        joint_offset_map.data_ptr(),
        batch_size,
        horizon,
        num_spheres,
        num_links,
        n_joints,
        n_tool_frames,
        num_envs,
        threads_per_batch,
    )

    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)
