#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Third Party
import torch
from torch.autograd import Function

# CuRobo
from curobo.util.logger import log_warn

try:
    # CuRobo
    from curobo.curobolib import kinematics_fused_cu
except ImportError:
    log_warn("kinematics_fused_cu not found, JIT compiling...")
    # Third Party
    from torch.utils.cpp_extension import load

    # CuRobo
    from curobo.curobolib.util_file import add_cpp_path

    kinematics_fused_cu = load(
        name="kinematics_fused_cu",
        sources=add_cpp_path(
            [
                "kinematics_fused_cuda.cpp",
                "kinematics_fused_kernel.cu",
            ]
        ),
    )


def rotation_matrix_to_quaternion(in_mat, out_quat):
    r = kinematics_fused_cu.matrix_to_quaternion(out_quat, in_mat.reshape(-1, 9))
    return r[0]


class KinematicsFusedFunction(Function):
    @staticmethod
    def _call_cuda(
        link_pos_out,
        link_quat_out,
        robot_sphere_out,
        global_cumul_mat,
        angle,
        fixed_transform,
        robot_spheres,
        link_map,
        joint_map,
        joint_map_type,
        store_link_map,
        link_sphere_map,
    ):
        b_shape = link_pos_out.shape
        b_size = b_shape[0]
        n_spheres = robot_sphere_out.shape[1]
        r = kinematics_fused_cu.forward(
            link_pos_out,
            link_quat_out,
            robot_sphere_out,
            global_cumul_mat,
            angle,
            fixed_transform,
            robot_spheres,
            link_map,
            joint_map,
            joint_map_type,
            store_link_map,
            link_sphere_map,
            b_size,
            n_spheres,
            False,
        )
        out_link_pos = r[0]
        out_link_quat = r[1]
        out_spheres = r[2]
        global_cumul_mat = r[3]
        return out_link_pos, out_link_quat, out_spheres, global_cumul_mat

    @staticmethod
    def _call_backward_cuda(
        grad_out,
        link_pos_out,
        link_quat_out,
        robot_sphere_out,
        global_cumul_mat,
        angle,
        fixed_transform,
        robot_spheres,
        link_map,
        joint_map,
        joint_map_type,
        store_link_map,
        link_sphere_map,
        link_chain_map,
        sparsity_opt=True,
    ):
        b_shape = link_pos_out.shape
        b_size = b_shape[0]
        n_spheres = robot_sphere_out.shape[1]
        if grad_out.is_contiguous():
            grad_out = grad_out.view(-1)
        else:
            grad_out = grad_out.reshape(-1)
        r = kinematics_fused_cu.backward(
            grad_out,
            link_pos_out,
            link_quat_out,
            robot_sphere_out,
            global_cumul_mat,
            angle,
            fixed_transform,
            robot_spheres,
            link_map,
            joint_map,
            joint_map_type,
            store_link_map,
            link_sphere_map,
            link_chain_map,
            b_size,
            n_spheres,
            sparsity_opt,
            False,
        )
        out_q = r[0].view(b_size, -1)

        return out_q

    @staticmethod
    def forward(
        ctx,
        # link_mats: torch.Tensor,
        link_pos: torch.Tensor,
        link_quat: torch.Tensor,
        b_robot_spheres: torch.tensor,
        global_cumul_mat: torch.Tensor,
        joint_seq: torch.Tensor,
        fixed_transform: torch.tensor,
        robot_spheres: torch.tensor,
        link_map: torch.tensor,
        joint_map: torch.Tensor,
        joint_map_type: torch.Tensor,
        store_link_map: torch.Tensor,
        link_sphere_map: torch.Tensor,
        link_chain_map: torch.Tensor,
        grad_out: torch.Tensor,
    ):
        link_pos, link_quat, b_robot_spheres, global_cumul_mat = KinematicsFusedFunction._call_cuda(
            link_pos,
            link_quat,
            b_robot_spheres,
            global_cumul_mat,
            joint_seq,
            fixed_transform,
            robot_spheres,
            link_map,
            joint_map,
            joint_map_type,
            store_link_map,
            link_sphere_map,
        )
        ctx.save_for_backward(
            joint_seq,
            fixed_transform,
            robot_spheres,
            link_map,
            joint_map,
            joint_map_type,
            store_link_map,
            link_sphere_map,
            link_chain_map,
            grad_out,
            global_cumul_mat,
        )

        return link_pos, link_quat, b_robot_spheres

    @staticmethod
    def backward(ctx, grad_out_link_pos, grad_out_link_quat, grad_out_spheres):
        grad_joint = None

        if ctx.needs_input_grad[4]:
            (
                joint_seq,
                fixed_transform,
                robot_spheres,
                link_map,
                joint_map,
                joint_map_type,
                store_link_map,
                link_sphere_map,
                link_chain_map,
                grad_out,
                global_cumul_mat,
            ) = ctx.saved_tensors
            grad_joint = KinematicsFusedFunction._call_backward_cuda(
                grad_out,
                grad_out_link_pos,
                grad_out_link_quat,
                grad_out_spheres,
                global_cumul_mat,
                joint_seq,
                fixed_transform,
                robot_spheres,
                link_map,
                joint_map,
                joint_map_type,
                store_link_map,
                link_sphere_map,
                link_chain_map,
                True,
            )

        return (
            None,
            None,
            None,
            None,
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
        )


class KinematicsFusedGlobalCumulFunction(Function):
    @staticmethod
    def _call_cuda(
        link_pos_out,
        link_quat_out,
        robot_sphere_out,
        global_cumul_mat,
        angle,
        fixed_transform,
        robot_spheres,
        link_map,
        joint_map,
        joint_map_type,
        store_link_map,
        link_sphere_map,
    ):
        b_shape = link_pos_out.shape
        b_size = b_shape[0]
        n_spheres = robot_sphere_out.shape[1]
        # print(n_spheres)
        # print(angle)
        r = kinematics_fused_cu.forward(
            link_pos_out,
            link_quat_out,
            robot_sphere_out,
            global_cumul_mat,
            angle,
            fixed_transform,
            robot_spheres,
            link_map,
            joint_map,
            joint_map_type,
            store_link_map,
            link_sphere_map,
            b_size,
            n_spheres,
            True,
        )
        out_link_pos = r[0]
        out_link_quat = r[1]
        out_spheres = r[2]
        global_cumul_mat = r[3]
        return out_link_pos, out_link_quat, out_spheres, global_cumul_mat

    @staticmethod
    def _call_backward_cuda(
        grad_out,
        link_pos_out,
        link_quat_out,
        robot_sphere_out,
        global_cumul_mat,
        angle,
        fixed_transform,
        robot_spheres,
        link_map,
        joint_map,
        joint_map_type,
        store_link_map,
        link_sphere_map,
        link_chain_map,
        sparsity_opt=True,
    ):
        b_shape = link_pos_out.shape
        b_size = b_shape[0]
        # b_size = link_mat_out.shape[0]
        n_spheres = robot_sphere_out.shape[1]
        if grad_out.is_contiguous():
            grad_out = grad_out.view(-1)
        else:
            grad_out = grad_out.reshape(-1)
        # create backward tensors?

        r = kinematics_fused_cu.backward(
            grad_out,
            link_pos_out,
            link_quat_out,
            robot_sphere_out,
            global_cumul_mat,
            angle,
            fixed_transform,
            robot_spheres,
            link_map,
            joint_map,
            joint_map_type,
            store_link_map,
            link_sphere_map,
            link_chain_map,
            b_size,
            n_spheres,
            sparsity_opt,
            True,
        )
        out_q = r[0]  # .view(angle.shape)
        out_q = out_q.view(b_size, -1)
        return out_q

    @staticmethod
    def forward(
        ctx,
        # link_mats: torch.Tensor,
        link_pos: torch.Tensor,
        link_quat: torch.Tensor,
        b_robot_spheres: torch.tensor,
        global_cumul_mat: torch.Tensor,
        joint_seq: torch.Tensor,
        fixed_transform: torch.tensor,
        robot_spheres: torch.tensor,
        link_map: torch.tensor,
        joint_map: torch.Tensor,
        joint_map_type: torch.Tensor,
        store_link_map: torch.Tensor,
        link_sphere_map: torch.Tensor,
        link_chain_map: torch.Tensor,
        grad_out: torch.Tensor,
    ):
        if joint_seq.is_contiguous():
            joint_seq = joint_seq.view(-1)
        else:
            joint_seq = joint_seq.reshape(-1)
        (
            link_pos,
            link_quat,
            b_robot_spheres,
            global_cumul_mat,
        ) = KinematicsFusedGlobalCumulFunction._call_cuda(
            link_pos,
            link_quat,
            b_robot_spheres,
            global_cumul_mat,
            joint_seq,
            fixed_transform,
            robot_spheres,
            link_map,
            joint_map,
            joint_map_type,
            store_link_map,
            link_sphere_map,
        )
        ctx.save_for_backward(
            joint_seq,
            fixed_transform,
            robot_spheres,
            link_map,
            joint_map,
            joint_map_type,
            store_link_map,
            link_sphere_map,
            link_chain_map,
            grad_out,
            global_cumul_mat,
        )

        return link_pos, link_quat, b_robot_spheres

    @staticmethod
    def backward(ctx, grad_out_link_pos, grad_out_link_quat, grad_out_spheres):
        grad_joint = None

        if ctx.needs_input_grad[4]:
            (
                joint_seq,
                fixed_transform,
                robot_spheres,
                link_map,
                joint_map,
                joint_map_type,
                store_link_map,
                link_sphere_map,
                link_chain_map,
                grad_out,
                global_cumul_mat,
            ) = ctx.saved_tensors
            grad_joint = KinematicsFusedGlobalCumulFunction._call_backward_cuda(
                grad_out,
                grad_out_link_pos,
                grad_out_link_quat,
                grad_out_spheres,
                global_cumul_mat,
                joint_seq,
                fixed_transform,
                robot_spheres,
                link_map,
                joint_map,
                joint_map_type,
                store_link_map,
                link_sphere_map,
                link_chain_map,
                True,
            )

        return (
            None,
            None,
            None,
            None,
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
        )


def get_cuda_kinematics(
    link_pos_seq,
    link_quat_seq,
    batch_robot_spheres,
    global_cumul_mat,
    q_in,
    fixed_transform,
    link_spheres_tensor,
    link_map,  # tells which link is attached to which link i
    joint_map,  # tells which joint is attached to a link i
    joint_map_type,  # joint type
    store_link_map,
    link_sphere_idx_map,  # sphere idx map
    link_chain_map,
    grad_out_q,
    use_global_cumul: bool = True,
):
    if use_global_cumul:
        link_pos, link_quat, robot_spheres = KinematicsFusedGlobalCumulFunction.apply(
            link_pos_seq,
            link_quat_seq,
            batch_robot_spheres,
            global_cumul_mat,
            q_in,
            fixed_transform,
            link_spheres_tensor,
            link_map,  # tells which link is attached to which link i
            joint_map,  # tells which joint is attached to a link i
            joint_map_type,  # joint type
            store_link_map,
            link_sphere_idx_map,  # sphere idx map
            link_chain_map,
            grad_out_q,
        )
    else:
        link_pos, link_quat, robot_spheres = KinematicsFusedFunction.apply(
            link_pos_seq,
            link_quat_seq,
            batch_robot_spheres,
            global_cumul_mat,
            q_in,
            fixed_transform,
            link_spheres_tensor,
            link_map,  # tells which link is attached to which link i
            joint_map,  # tells which joint is attached to a link i
            joint_map_type,  # joint type
            store_link_map,
            link_sphere_idx_map,  # sphere idx map
            link_chain_map,
            grad_out_q,
        )
    return link_pos, link_quat, robot_spheres
