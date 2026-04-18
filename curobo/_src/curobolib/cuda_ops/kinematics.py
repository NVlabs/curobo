# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Third Party
from typing import Optional, Tuple

import torch
from torch.autograd import Function

from curobo._src.curobolib.backends import kinematics as kinematics_cu
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_bool_tensors,
    check_float32_tensors,
    check_int8_tensors,
    check_int16_tensors,
    check_int32_tensors,
)
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.types.device_cfg import DeviceCfg

# CuRobo
from curobo._src.util.logging import log_and_raise


class KinematicsFusedFunction(Function):
    @staticmethod
    def create_buffers(
        batch: int,
        horizon: int,
        kinematics_config: KinematicsParams,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        batch_link_position = torch.zeros(
            (batch, horizon, kinematics_config.num_pose_links, 3),
            device=device_cfg.device,
            dtype=device_cfg.dtype,
        )
        batch_link_quaternion = torch.zeros(
            (batch, horizon, kinematics_config.num_pose_links, 4),
            device=device_cfg.device,
            dtype=device_cfg.dtype,
        )
        batch_robot_spheres = torch.zeros(
            (batch, horizon, kinematics_config.num_spheres, 4),
            device=device_cfg.device,
            dtype=device_cfg.dtype,
        )
        batch_jacobian = torch.zeros(
            (batch, horizon, kinematics_config.num_pose_links, 6, kinematics_config.num_dof),
            device=device_cfg.device,
            dtype=device_cfg.dtype,
        )
        batch_com = torch.zeros(
            (batch, horizon, 4),
            device=device_cfg.device,
            dtype=device_cfg.dtype,
        )
        batch_cumul_mat = torch.zeros(
            (batch, horizon, kinematics_config.num_links, 3, 4),
            device=device_cfg.device,
            dtype=device_cfg.dtype,
        )

        # backward buffers:
        grad_link_pos = torch.zeros_like(batch_link_position)
        grad_link_quat = torch.zeros_like(batch_link_quaternion)
        grad_robot_spheres = torch.zeros_like(batch_robot_spheres)
        grad_com = torch.zeros_like(batch_com)
        grad_out_q = torch.zeros(
            (batch, horizon, kinematics_config.num_dof),
            device=device_cfg.device,
            dtype=device_cfg.dtype,
        )
        grad_out_q_jacobian = torch.zeros_like(grad_out_q)

        buffers = {
            "batch_link_position": batch_link_position,
            "batch_link_quaternion": batch_link_quaternion,
            "batch_robot_spheres": batch_robot_spheres,
            "batch_com": batch_com,
            "batch_jacobian": batch_jacobian,
            "batch_cumul_mat": batch_cumul_mat,
            "grad_out_q": grad_out_q,
            "grad_out_q_jacobian": grad_out_q_jacobian,
            "grad_in_link_pos": grad_link_pos,
            "grad_in_link_quat": grad_link_quat,
            "grad_in_robot_spheres": grad_robot_spheres,
            "grad_in_com": grad_com,
        }
        return buffers

    @staticmethod
    def forward(
        ctx,
        joint_seq: torch.Tensor,
        batch_link_position: torch.Tensor,
        batch_link_quaternion: torch.Tensor,
        batch_robot_spheres: torch.tensor,
        batch_com: torch.Tensor,
        batch_jacobian: torch.Tensor,
        batch_cumul_mat: torch.Tensor,
        kinematics_config: KinematicsParams,
        grad_out: torch.Tensor,
        grad_out_q_jacobian: torch.Tensor,
        grad_in_link_pos: torch.Tensor,
        grad_in_link_quat: torch.Tensor,
        grad_in_robot_spheres: torch.Tensor,
        grad_in_com: torch.Tensor,
        compute_jacobian: bool,
        compute_spheres: bool,
        compute_com: bool,
        env_query_idx: torch.Tensor,
        horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b_size = batch_link_position.shape[0] * batch_link_position.shape[1]
        num_spheres = batch_robot_spheres.shape[2]
        n_joints = joint_seq.shape[-1]
        num_links = kinematics_config.link_map.shape[0]
        num_envs = kinematics_config.num_envs
        if not compute_spheres:
            num_spheres = 0
        ctx.set_materialize_grads(False)
        device = joint_seq.device
        check_float32_tensors(
            device,
            joint_seq=joint_seq,
            batch_link_position=batch_link_position,
            batch_link_quaternion=batch_link_quaternion,
            batch_robot_spheres=batch_robot_spheres,
            batch_com=batch_com,
            batch_jacobian=batch_jacobian,
            batch_cumul_mat=batch_cumul_mat,
            fixed_transforms=kinematics_config.fixed_transforms,
            link_spheres=kinematics_config.link_spheres,
            link_masses_com=kinematics_config.link_masses_com,
            joint_offset_map=kinematics_config.joint_offset_map,
        )
        check_int16_tensors(
            device,
            link_map=kinematics_config.link_map,
            joint_map=kinematics_config.joint_map,
            tool_frame_map=kinematics_config.tool_frame_map,
            link_sphere_idx_map=kinematics_config.link_sphere_idx_map,
            link_chain_data=kinematics_config.link_chain_data,
            link_chain_offsets=kinematics_config.link_chain_offsets,
            joint_links_data=kinematics_config.joint_links_data,
            joint_links_offsets=kinematics_config.joint_links_offsets,
        )
        check_int8_tensors(device, joint_map_type=kinematics_config.joint_map_type)
        check_bool_tensors(
            device,
            joint_affects_endeffector=kinematics_config.joint_affects_endeffector,
        )
        check_int32_tensors(device, env_query_idx=env_query_idx)
        kinematics_config.validate_shapes()
        kinematics_cu.launch_kinematics_forward(
            batch_link_position,
            batch_link_quaternion,
            batch_robot_spheres,
            batch_com,
            batch_jacobian,
            batch_cumul_mat,
            joint_seq,
            kinematics_config.fixed_transforms,
            kinematics_config.link_spheres,
            kinematics_config.link_masses_com,
            kinematics_config.link_map,
            kinematics_config.joint_map,
            kinematics_config.joint_map_type,
            kinematics_config.tool_frame_map,
            kinematics_config.link_sphere_idx_map,
            kinematics_config.link_chain_data,
            kinematics_config.link_chain_offsets,
            kinematics_config.joint_links_data,
            kinematics_config.joint_links_offsets,
            kinematics_config.joint_affects_endeffector,
            kinematics_config.joint_offset_map,
            env_query_idx,
            num_envs,
            b_size,
            horizon,
            n_joints,
            num_spheres,
            compute_jacobian,
            compute_com,
        )

        ctx.mark_non_differentiable(
            batch_cumul_mat,
            grad_out,
        )
        ctx.kinematics_config = kinematics_config
        ctx.compute_jacobian = compute_jacobian
        ctx.compute_spheres = compute_spheres
        ctx.compute_com = compute_com
        ctx.env_query_idx = env_query_idx
        ctx.horizon = horizon
        ctx.grad_in_link_pos = grad_in_link_pos
        ctx.grad_in_link_quat = grad_in_link_quat
        ctx.grad_in_robot_spheres = grad_in_robot_spheres
        ctx.grad_in_com = grad_in_com

        ctx.save_for_backward(
            joint_seq,
            grad_out,
            batch_cumul_mat,
            grad_out_q_jacobian,
            batch_com,
        )
        return (
            batch_link_position,
            batch_link_quaternion,
            batch_robot_spheres,
            batch_com,
            batch_jacobian,
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_in_link_pos: Optional[torch.Tensor],
        grad_in_link_quat: Optional[torch.Tensor],
        grad_in_spheres: Optional[torch.Tensor],
        grad_in_com: Optional[torch.Tensor],
        grad_in_link_jacobian: Optional[torch.Tensor],
    ):
        grad_joint = None
        if ctx.needs_input_grad[0]:
            (
                joint_seq,
                grad_out,
                batch_cumul_mat,
                grad_out_q_jacobian,
                batch_com,
            ) = ctx.saved_tensors
            kinematics_config = ctx.kinematics_config
            if grad_in_link_pos is None:
                grad_in_link_pos = ctx.grad_in_link_pos
            if grad_in_link_quat is None:
                grad_in_link_quat = ctx.grad_in_link_quat
            if grad_in_spheres is None:
                grad_in_spheres = ctx.grad_in_robot_spheres
            if grad_in_com is None:
                grad_in_com = ctx.grad_in_com

            num_spheres = grad_in_spheres.shape[2]
            if not ctx.compute_spheres:
                num_spheres = 0

            device = joint_seq.device
            check_float32_tensors(
                device,
                grad_in_spheres=grad_in_spheres,
                grad_out=grad_out,
                grad_in_link_pos=grad_in_link_pos,
                grad_in_link_quat=grad_in_link_quat,
                batch_cumul_mat=batch_cumul_mat,
                batch_com=batch_com,
                grad_in_com=grad_in_com,
                joint_seq=joint_seq,
            )

            b_size = joint_seq.shape[0] * joint_seq.shape[1]
            n_joints = joint_seq.shape[-1]
            n_tool_frames = kinematics_config.tool_frame_map.shape[0]
            if grad_in_link_quat.data_ptr() % 16 != 0:
                log_and_raise("grad_in_link_quat is not aligned to 16 bytes")
            kinematics_cu.launch_kinematics_backward(
                grad_out,
                grad_in_link_pos,
                grad_in_link_quat,
                grad_in_spheres,
                grad_in_com,
                batch_com,
                batch_cumul_mat,
                joint_seq,
                kinematics_config.fixed_transforms,
                kinematics_config.link_spheres,
                kinematics_config.link_masses_com,
                kinematics_config.link_map,
                kinematics_config.joint_map,
                kinematics_config.joint_map_type,
                kinematics_config.tool_frame_map,
                kinematics_config.link_sphere_idx_map,
                kinematics_config.link_chain_data,
                kinematics_config.link_chain_offsets,
                kinematics_config.joint_offset_map,
                ctx.env_query_idx,
                kinematics_config.num_envs,
                b_size,
                ctx.horizon,
                n_joints,
                num_spheres,
                ctx.compute_com,
            )
            grad_joint = grad_out

            if ctx.compute_jacobian and grad_in_link_jacobian is not None:
                check_float32_tensors(device, grad_in_link_jacobian=grad_in_link_jacobian)

                kinematics_cu.launch_kinematics_jacobian_backward(
                    grad_out_q_jacobian,
                    grad_in_link_jacobian,
                    batch_cumul_mat,
                    kinematics_config.joint_map_type,
                    kinematics_config.joint_map,
                    kinematics_config.link_map,
                    kinematics_config.link_chain_data,
                    kinematics_config.link_chain_offsets,
                    kinematics_config.joint_links_data,
                    kinematics_config.joint_links_offsets,
                    kinematics_config.joint_affects_endeffector,
                    kinematics_config.tool_frame_map,
                    kinematics_config.joint_offset_map,
                    b_size,
                    n_joints,
                    n_tool_frames,
                )
                grad_joint = grad_joint + grad_out_q_jacobian

        return (
            grad_joint,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
