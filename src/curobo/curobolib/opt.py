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
    from curobo.curobolib import lbfgs_step_cu
except ImportError:
    log_warn("lbfgs_step_cu not found, JIT compiling...")
    # Third Party
    from torch.utils.cpp_extension import load

    # CuRobo
    from curobo.util_file import add_cpp_path

    lbfgs_step_cu = load(
        name="lbfgs_step_cu",
        sources=add_cpp_path(
            [
                "lbfgs_step_cuda.cpp",
                "lbfgs_step_kernel.cu",
            ]
        ),
    )


class LBFGScu(Function):
    @staticmethod
    def forward(
        ctx,
        step_vec,
        rho_buffer,
        y_buffer,
        s_buffer,
        q,
        grad_q,
        x_0,
        grad_0,
        epsilon=0.1,
        stable_mode=False,
        use_shared_buffers=True,
    ):
        m, b, v_dim, _ = y_buffer.shape

        R = lbfgs_step_cu.forward(
            step_vec,  # .view(-1),
            rho_buffer,  # .view(-1),
            y_buffer,  # .view(-1),
            s_buffer,  # .view(-1),
            q,
            grad_q,  # .view(-1),
            x_0,
            grad_0,
            epsilon,
            b,
            m,
            v_dim,
            stable_mode,
            use_shared_buffers,
        )
        step_v = R[0].view(step_vec.shape)

        # ctx.save_for_backward(batch_spheres, robot_spheres, link_mats, link_sphere_map)
        return step_v

    @staticmethod
    def backward(ctx, grad_output):
        return (
            None,
            None,
            None,
            None,
            None,
            None,
        )
