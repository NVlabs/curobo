# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""cuda.core backend for kinematics kernels.
Provides runtime compilation of CUDA kernels with the same interface as PyBind11 extensions.
"""

# Standard Library
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
    batch_robot_spheres: torch.Tensor,
    batch_center_of_mass: torch.Tensor,
    batch_jacobian: torch.Tensor,
    global_cumul_mat: torch.Tensor,
    joint_vec: torch.Tensor,
    fixed_transform: torch.Tensor,
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
    compute_jacobian: bool,
    compute_com: bool,
):
    """Launch kinematics forward kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Args:
        All tensor arguments and parameters as expected by kinematics forward kernel

    Note:
        Modifies output tensors in-place (link_pos, link_quat, batch_robot_spheres,
        batch_center_of_mass, batch_jacobian, global_cumul_mat)
    """
    runtime = get_runtime()

    # Get cuda.core kernel cache
    cache = runtime.get_cuda_core_cache()

    # Get kernel configuration
    kernel_config = KinematicsKernelCfg()

    # Calculate launch configuration
    num_links = link_map.shape[0]
    n_tool_frames = tool_frame_map.shape[0]

    compile_n_links = num_links
    config_main, config_separate = KinematicsLaunchCfg.calculate_forward_config(
        batch_size, num_links, num_spheres, n_tool_frames, compute_jacobian
    )

    # Get stream wrapper from the tensor's device to ensure proper context
    pt_stream = torch.cuda.current_stream(joint_vec.device)
    stream = cache.get_stream_wrapper(pt_stream)

    # Branch based on whether we use single fused kernel or separate kernels
    if config_separate is None:
        # Single fused kernel path (num_spheres < 100)
        # Select kernel based on whether jacobian is needed
        # Template parameter COMPUTE_COM is always false (default)
        if compute_jacobian:
            kernel_name = f"curobo::kinematics::kinematics_fused_jacobian_kernel<{str(compute_com).lower()}, {compile_n_links}>"
        else:
            kernel_name = f"curobo::kinematics::kinematics_fused_kernel<{str(compute_com).lower()}, {compile_n_links}>"

        # Get or compile kernel
        kernel_files = [
            kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("forward")
        ]
        kernel = cache.get_or_compile_kernel(
            source_files=kernel_files,
            kernel_name=kernel_name,
            include_dirs=kernel_config.get_include_dirs(),
            compile_flags=kernel_config.get_compile_flags(),
        )

        # Prepare kernel arguments as data pointers
        if compute_jacobian:
            # Jacobian kernel has additional arguments
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
                num_links,
                n_joints,
                n_tool_frames,
                num_envs,
            )
        else:
            # Standard forward kernel
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
                joint_links_data.data_ptr(),
                joint_links_offsets.data_ptr(),
                joint_offset_map.data_ptr(),
                env_query_idx.data_ptr(),
                batch_size,
                horizon,
                num_spheres,
                num_links,
                n_joints,
                n_tool_frames,
                num_envs,
            )

        # Launch kernel
        launch_kernel(kernel_name, stream, config_main, kernel, *kernel_args)

    else:
        # Separate kernel path for large num_spheres (>= 100)
        # Need to launch two kernels: cumul kernel, then spheres/links kernel

        # First kernel: kinematics_cumul_kernel
        kernel_name_cumul = f"curobo::kinematics::kinematics_cumul_kernel<{compile_n_links}>"

        kernel_files = [
            kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("forward")
        ]
        kernel_cumul = cache.get_or_compile_kernel(
            source_files=kernel_files,
            kernel_name=kernel_name_cumul,
            include_dirs=kernel_config.get_include_dirs(),
            compile_flags=kernel_config.get_compile_flags(),
        )

        # Prepare cumul kernel arguments matching C++ signature (lines 248-259)
        kernel_args_cumul = (
            global_cumul_mat.data_ptr(),
            joint_vec.data_ptr(),
            fixed_transform.data_ptr(),
            joint_map_type.data_ptr(),
            joint_map.data_ptr(),
            link_map.data_ptr(),
            joint_offset_map.data_ptr(),
            batch_size,
            num_links,
            n_joints,
        )

        # Launch cumul kernel
        launch_kernel(kernel_name_cumul, stream, config_main, kernel_cumul, *kernel_args_cumul)

        # Calculate separate_batches_per_block (same logic as in calculate_forward_config)
        max_shared_mem = KinematicsLaunchCfg.DEFAULT_MAX_SHARED_MEM
        estimated_threads_per_batch = min(128, max(num_spheres, n_tool_frames))
        separate_shared_mem_per_batch = num_links * 12 * 4 + estimated_threads_per_batch * 16
        separate_batches_per_block = max_shared_mem // separate_shared_mem_per_batch
        separate_batches_per_block = min(separate_batches_per_block, 4)
        separate_batches_per_block = max(1, separate_batches_per_block)

        # Second kernel: spheres/links kernel (branch based on compute_jacobian)
        if compute_jacobian:
            # Launch kinematics_spheres_links_jacobian_kernel
            kernel_name_spheres = f"curobo::kinematics::kinematics_spheres_links_jacobian_kernel<{str(compute_com).lower()}>"

            kernel_spheres = cache.get_or_compile_kernel(
                source_files=kernel_files,
                kernel_name=kernel_name_spheres,
                include_dirs=kernel_config.get_include_dirs(),
                compile_flags=kernel_config.get_compile_flags(),
            )

            # Prepare kernel arguments matching C++ signature (lines 266-292)
            kernel_args_spheres = (
                link_pos.data_ptr(),
                link_quat.data_ptr(),
                batch_robot_spheres.data_ptr(),
                batch_center_of_mass.data_ptr(),
                batch_jacobian.data_ptr(),
                global_cumul_mat.data_ptr(),
                robot_spheres.data_ptr(),
                link_masses_com.data_ptr(),
                tool_frame_map.data_ptr(),
                link_sphere_map.data_ptr(),
                env_query_idx.data_ptr(),
                joint_map_type.data_ptr(),
                joint_map.data_ptr(),
                link_map.data_ptr(),
                link_chain_data.data_ptr(),
                link_chain_offsets.data_ptr(),
                joint_links_data.data_ptr(),
                joint_links_offsets.data_ptr(),
                joint_affects_endeffector.data_ptr(),
                joint_offset_map.data_ptr(),
                batch_size,
                horizon,
                num_spheres,
                num_envs,
                num_links,
                n_joints,
                n_tool_frames,
                separate_batches_per_block,
            )
        else:
            # Launch kinematics_spheres_links_kernel
            kernel_name_spheres = (
                f"curobo::kinematics::kinematics_spheres_links_kernel<{str(compute_com).lower()}>"
            )

            kernel_spheres = cache.get_or_compile_kernel(
                source_files=kernel_files,
                kernel_name=kernel_name_spheres,
                include_dirs=kernel_config.get_include_dirs(),
                compile_flags=kernel_config.get_compile_flags(),
            )

            # Prepare kernel arguments matching C++ signature (lines 299-314)
            kernel_args_spheres = (
                link_pos.data_ptr(),
                link_quat.data_ptr(),
                batch_robot_spheres.data_ptr(),
                batch_center_of_mass.data_ptr(),
                global_cumul_mat.data_ptr(),
                robot_spheres.data_ptr(),
                link_masses_com.data_ptr(),
                tool_frame_map.data_ptr(),
                link_sphere_map.data_ptr(),
                env_query_idx.data_ptr(),
                batch_size,
                horizon,
                num_spheres,
                num_envs,
                num_links,
                n_tool_frames,
                separate_batches_per_block,
            )

        # Launch spheres/links kernel
        launch_kernel(
            kernel_name_spheres, stream, config_separate, kernel_spheres, *kernel_args_spheres
        )


def launch_kinematics_backward(
    grad_out: torch.Tensor,
    grad_nlinks_pos: torch.Tensor,
    grad_nlinks_quat: torch.Tensor,
    grad_spheres: torch.Tensor,
    grad_center_of_mass: torch.Tensor,
    batch_center_of_mass: torch.Tensor,
    global_cumul_mat: torch.Tensor,
    joint_vec: torch.Tensor,
    fixed_transform: torch.Tensor,
    robot_spheres: torch.Tensor,
    link_masses_com: torch.Tensor,
    link_map: torch.Tensor,
    joint_map: torch.Tensor,
    joint_map_type: torch.Tensor,
    tool_frame_map: torch.Tensor,
    link_sphere_map: torch.Tensor,
    link_chain_data: torch.Tensor,
    link_chain_offsets: torch.Tensor,
    joint_offset_map: torch.Tensor,
    env_query_idx: torch.Tensor,
    num_envs: int,
    batch_size: int,
    horizon: int,
    n_joints: int,
    num_spheres: int,
    compute_com: bool,
):
    """Launch kinematics backward kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Args:
        All tensor arguments and parameters as expected by kinematics backward kernel

    Note:
        Modifies grad_out tensor in-place with joint gradients
    """
    runtime = get_runtime()

    # Get cuda.core kernel cache
    cache = runtime.get_cuda_core_cache()

    # Get kernel configuration
    kernel_config = KinematicsKernelCfg()

    # Calculate launch configuration and get kernel template parameters
    num_links = link_map.shape[0]
    n_tool_frames = tool_frame_map.shape[0]
    config, threads_per_batch, use_warp_reduce, max_joints_template = (
        KinematicsLaunchCfg.calculate_backward_config(
            batch_size, num_links, num_spheres, n_tool_frames, n_joints
        )
    )

    # Build kernel name with template parameters
    # kernel signature: kinematics_fused_backward_unified_kernel<float, float, MAX_JOINTS, use_warp_reduce>
    kernel_name = (
        f"curobo::kinematics::kinematics_fused_backward_unified_kernel<float, float, "
        f"{max_joints_template}, {str(use_warp_reduce).lower()}, {str(compute_com).lower()}>"
    )

    # Get or compile kernel
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("backward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Get stream wrapper from the tensor's device to ensure proper context
    pt_stream = torch.cuda.current_stream(grad_out.device)
    stream = cache.get_stream_wrapper(pt_stream)

    # Prepare kernel arguments as data pointers
    # Match the C++ kernel signature from lines 77-82 and 166-187 in kinematics_backward_kernel_launch.cu
    kernel_args = (
        grad_out.data_ptr(),
        grad_nlinks_pos.data_ptr(),
        grad_nlinks_quat.data_ptr(),
        grad_spheres.data_ptr(),
        grad_center_of_mass.data_ptr(),
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
        env_query_idx.data_ptr(),
        link_chain_data.data_ptr(),
        link_chain_offsets.data_ptr(),
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

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def launch_kinematics_jacobian_backward(
    grad_joint: torch.Tensor,
    grad_jacobian: torch.Tensor,
    global_cumul_mat: torch.Tensor,
    joint_map_type: torch.Tensor,
    joint_map: torch.Tensor,
    link_map: torch.Tensor,
    link_chain_data: torch.Tensor,
    link_chain_offsets: torch.Tensor,
    joint_links_data: torch.Tensor,
    joint_links_offsets: torch.Tensor,
    joint_affects_endeffector: torch.Tensor,
    tool_frame_map: torch.Tensor,
    joint_offset_map: torch.Tensor,
    batch_size: int,
    n_joints: int,
    n_tool_frames: int,
):
    """Launch kinematics jacobian backward kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Args:
        All tensor arguments and parameters as expected by kinematics jacobian backward kernel

    Note:
        Modifies grad_joint tensor in-place with joint gradients
    """
    runtime = get_runtime()

    # Get cuda.core kernel cache
    cache = runtime.get_cuda_core_cache()

    # Get kernel configuration
    kernel_config = KinematicsKernelCfg()

    # Calculate launch configuration and get kernel template parameters
    num_links = link_map.shape[0]
    config, threads_per_batch, use_warp_reduce, max_joints_template = (
        KinematicsLaunchCfg.calculate_jacobian_backward_config(
            batch_size, num_links, n_tool_frames, n_joints
        )
    )

    # Build kernel name with template parameters
    # kernel signature: kinematics_jacobian_gradient_backward_kernel<float, MAX_JOINTS, use_warp_reduce>
    kernel_name = (
        f"curobo::kinematics::kinematics_jacobian_gradient_backward_kernel<float, "
        f"{max_joints_template}, {'true' if use_warp_reduce else 'false'}>"
    )

    # Get or compile kernel
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("jacobian_backward")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Get stream wrapper from the tensor's device to ensure proper context
    pt_stream = torch.cuda.current_stream(grad_joint.device)
    stream = cache.get_stream_wrapper(pt_stream)

    # Prepare kernel arguments as data pointers
    # Match the C++ kernel signature from lines 89-91 and 161-176 in kinematics_backward_jacobian_kernel_launch.cu
    kernel_args = (
        grad_joint.data_ptr(),
        grad_jacobian.data_ptr(),
        global_cumul_mat.data_ptr(),
        joint_map_type.data_ptr(),
        joint_map.data_ptr(),
        link_map.data_ptr(),
        link_chain_data.data_ptr(),
        link_chain_offsets.data_ptr(),
        joint_links_data.data_ptr(),
        joint_links_offsets.data_ptr(),
        joint_affects_endeffector.data_ptr(),
        tool_frame_map.data_ptr(),
        joint_offset_map.data_ptr(),
        batch_size,
        n_joints,
        num_links,
        n_tool_frames,
        threads_per_batch,
    )

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)
