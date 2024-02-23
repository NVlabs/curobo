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
from packaging import version

# CuRobo
from curobo.util.logger import log_warn


def find_first_idx(array, value, EQUAL=False):
    if EQUAL:
        f_idx = torch.nonzero(array >= value, as_tuple=False)[0].item()
    else:
        f_idx = torch.nonzero(array > value, as_tuple=False)[0].item()
    return f_idx


def find_last_idx(array, value):
    f_idx = torch.nonzero(array <= value, as_tuple=False)[-1].item()
    return f_idx


def is_cuda_graph_available():
    if version.parse(torch.__version__) < version.parse("1.10"):
        log_warn("WARNING: Disabling CUDA Graph as pytorch < 1.10")
        return False
    return True


def is_torch_compile_available():
    if version.parse(torch.__version__) < version.parse("2.0"):
        log_warn("WARNING: Disabling compile as pytorch < 2.0")
        return False
    return True
