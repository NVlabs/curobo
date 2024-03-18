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
# Standard Library
from typing import List

# Third Party
import torch

# CuRobo
from curobo.util.torch_utils import get_torch_jit_decorator


def check_tensor_shapes(new_tensor: torch.Tensor, mem_tensor: torch.Tensor):
    if not isinstance(mem_tensor, torch.Tensor):
        return False
    if len(mem_tensor.shape) != len(new_tensor.shape):
        return False
    if mem_tensor.shape == new_tensor.shape:
        return True


def copy_tensor(new_tensor: torch.Tensor, mem_tensor: torch.Tensor):
    if check_tensor_shapes(new_tensor, mem_tensor):
        mem_tensor.copy_(new_tensor)
        return True
    return False


def copy_if_not_none(new_tensor, ref_tensor):
    """Clones x if it's not None.
    TODO: Rename this to clone_if_not_none


    Args:
        x (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    if ref_tensor is not None and new_tensor is not None:
        ref_tensor.copy_(new_tensor)
    elif ref_tensor is None and new_tensor is not None:
        ref_tensor = new_tensor

    return ref_tensor


def clone_if_not_none(x):
    """Clones x if it's not None.


    Args:
        x (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    if x is not None:
        return x.clone()
    return None


@get_torch_jit_decorator()
def cat_sum(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.sum(torch.stack(tensor_list, dim=0), dim=0)
    return cat_tensor


@get_torch_jit_decorator()
def cat_sum_horizon(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.sum(torch.stack(tensor_list, dim=0), dim=(0, -1))
    return cat_tensor


@get_torch_jit_decorator()
def cat_max(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.max(torch.stack(tensor_list, dim=0), dim=0)[0]
    return cat_tensor


def tensor_repeat_seeds(tensor, num_seeds):
    return (
        tensor.view(tensor.shape[0], 1, tensor.shape[-1])
        .repeat(1, num_seeds, 1)
        .reshape(tensor.shape[0] * num_seeds, tensor.shape[-1])
    )


@get_torch_jit_decorator()
def fd_tensor(p: torch.Tensor, dt: torch.Tensor):
    out = ((torch.roll(p, -1, -2) - p) * (1 / dt).unsqueeze(-1))[..., :-1, :]
    return out
