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
    from curobo.util_file import add_cpp_path

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
    def forward(
        ctx,
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
        joint_offset_map: torch.Tensor,
        grad_out: torch.Tensor,
        use_global_cumul: bool = True,
    ):
        ctx.use_global_cumul = use_global_cumul
        b_shape = link_pos.shape
        b_size = b_shape[0]
        n_spheres = b_robot_spheres.shape[1]
        n_joints = joint_seq.shape[-1]

        r = kinematics_fused_cu.forward(
            link_pos,
            link_quat,
            b_robot_spheres,
            global_cumul_mat.view(-1),
            joint_seq,
            fixed_transform.view(-1),
            robot_spheres.view(-1),
            link_map,
            joint_map,
            joint_map_type.view(-1),
            store_link_map,
            link_sphere_map,
            joint_offset_map,
            b_size,
            n_joints,
            n_spheres,
            use_global_cumul,
        )
        out_link_pos = r[0]
        out_link_quat = r[1]
        out_spheres = r[2]
        global_cumul_mat = r[3]

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
            joint_offset_map,
            grad_out,
            global_cumul_mat,
        )
        return out_link_pos, out_link_quat, out_spheres

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
                joint_offset_map,
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
                joint_offset_map,
                True,
                use_global_cumul=ctx.use_global_cumul,
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
            None,
            None,
        )

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
        joint_offset_map,
        sparsity_opt=True,
        use_global_cumul=False,
    ):
        b_shape = grad_out.shape
        b_size = b_shape[0]
        n_spheres = robot_sphere_out.shape[1]
        n_joints = angle.shape[-1]
        grad_out = grad_out.contiguous()
        link_pos_out = link_pos_out.contiguous()
        link_quat_out = link_quat_out.contiguous()
        # if grad_out.is_contiguous():
        #    grad_out = grad_out.view(-1)
        # else:
        #    grad_out = grad_out.reshape(-1)

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
            joint_offset_map,  # offset_joint_map
            b_size,
            n_joints,
            n_spheres,
            sparsity_opt,
            use_global_cumul,
        )
        out_q = r[0].view(b_size, -1)

        return out_q


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
    joint_offset_map,
    grad_out_q,
    use_global_cumul: bool = True,
):
    # if not q_in.is_contiguous():
    #    q_in = q_in.contiguous()
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
        joint_offset_map,
        grad_out_q,
        use_global_cumul,
    )
    return link_pos, link_quat, robot_spheres
