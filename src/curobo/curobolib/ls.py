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

# CuRobo
from curobo.util.logger import log_warn

try:
    # CuRobo
    from curobo.curobolib import line_search_cu
except ImportError:
    log_warn("line_search_cu not found, JIT compiling...")

    # Third Party
    from torch.utils.cpp_extension import load

    # CuRobo
    from curobo.util_file import add_cpp_path

    line_search_cu = load(
        name="line_search_cu",
        sources=add_cpp_path(
            [
                "line_search_cuda.cpp",
                "line_search_kernel.cu",
                "update_best_kernel.cu",
            ]
        ),
    )


def wolfe_line_search(
    # m_idx,
    best_x,
    best_c,
    best_grad,
    g_x,
    x_set,
    sv,
    c,
    c_idx,
    c_1: float,
    c_2: float,
    al,
    sw: bool,
    aw: bool,
):
    batchsize = g_x.shape[0]
    l1 = g_x.shape[1]
    l2 = g_x.shape[2]
    r = line_search_cu.line_search(
        best_x,
        best_c,
        best_grad,
        g_x,
        x_set,
        sv,
        c,
        al,
        c_idx,
        c_1,
        c_2,
        sw,
        aw,
        l1,
        l2,
        batchsize,
    )
    return (r[0], r[1], r[2])


def update_best(
    best_cost,
    best_q,
    best_iteration,
    current_iteration,
    cost,
    q,
    d_opt: int,
    iteration: int,
    delta_threshold: float = 1e-5,
    relative_threshold: float = 0.999,
):
    cost_s1 = cost.shape[0]
    cost_s2 = cost.shape[1]
    r = line_search_cu.update_best(
        best_cost,
        best_q,
        best_iteration,
        current_iteration,
        cost,
        q,
        d_opt,
        cost_s1,
        cost_s2,
        iteration,
        delta_threshold,
        relative_threshold,
    )
    # print("batchsize:" + str(batchsize))
    return (r[0], r[1], r[2])  # output: best_cost, best_q, best_iteration
