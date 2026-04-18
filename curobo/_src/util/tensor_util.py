# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standard Library
from typing import List, Optional, Protocol, Tuple, Union

# Third Party
import torch

# CuRobo
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


class TensorLike(Protocol):
    """Protocol for tensor-like objects that support copying and have a shape attribute."""

    def copy_(self, other) -> None:
        """Copy data from another object."""
        ...

    @property
    def shape(self):
        """Shape of the object."""
        ...

    def clone(self):
        """Clone the object."""
        ...


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


def copy_or_clone(
    new_tensor: Optional[TensorLike], ref_tensor: Optional[TensorLike], allow_clone: bool = True
) -> Optional[TensorLike]:
    """Copy tensor if they are the same shape. If not and clone is allowed, clone the new tensor.

    This function can be applied to any data structure that has methods: "copy_", "clone", and
    "shape" attribute.

    Args:
        new_tensor: The tensor/object to copy from.
        ref_tensor: The tensor/object to copy to.
        allow_clone: If True, clone the new tensor if shapes don't match.

    Returns:
        The tensor/object that was copied to.

    Raises:
        ValueError: If the tensor shapes are not the same and clone is not allowed
    """
    if ref_tensor is None and new_tensor is None:
        return None
    if allow_clone:
        if ref_tensor is None:
            ref_tensor = new_tensor.clone()
            return ref_tensor
        elif new_tensor is None:
            return ref_tensor
        else:
            if ref_tensor.shape == new_tensor.shape:
                ref_tensor.copy_(new_tensor)
    if new_tensor is None:
        return ref_tensor
    if ref_tensor is None:
        log_and_raise("ref_tensor is None")
    if ref_tensor.shape == new_tensor.shape:
        ref_tensor.copy_(new_tensor)
    else:
        log_and_raise(f"ref_tensor.shape {ref_tensor.shape} != new_tensor.shape {new_tensor.shape}")
    return ref_tensor


def clone_if_not_none(x: Union[torch.Tensor, None]) -> Union[torch.Tensor, None]:
    """Clones x if it's not None.

    Args:
        x (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    if x is not None:
        return x.clone()
    return None


@get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
def cat_sum(
    tensor_list: List[torch.Tensor], sum_dim: Union[Tuple[int, int], Tuple[int], int] = (0,)
):
    cat_tensor = torch.cat(tensor_list, dim=-1)  # [batch_size, horizon, costs]
    sum_tensor = torch.sum(cat_tensor, dim=sum_dim)
    return sum_tensor


@get_torch_jit_decorator(slow_to_compile=True)
def cat_max(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.max(torch.stack(tensor_list, dim=0), dim=0)[0]
    return cat_tensor


@get_torch_jit_decorator(slow_to_compile=True)
def tensor_repeat_seeds(tensor: torch.Tensor, num_seeds: int):
    return (
        tensor.view(tensor.shape[0], 1, tensor.shape[-1])
        .repeat(1, num_seeds, 1)
        .reshape(tensor.shape[0] * num_seeds, tensor.shape[-1])
    )


@get_torch_jit_decorator(slow_to_compile=True)
def fd_tensor(p: torch.Tensor, dt: torch.Tensor):
    out = ((torch.roll(p, -1, -2) - p) * (1 / dt).unsqueeze(-1))[..., :-1, :]
    return out


@get_torch_jit_decorator(slow_to_compile=True)
def check_nan_last_dimension(position_trajectory):
    label = torch.any(torch.isnan(position_trajectory), dim=-1)
    return label


@get_torch_jit_decorator(slow_to_compile=True)
def shift_buffer(buffer, shift_d: int, action_dim: int, shift_steps: int = 1):
    buffer = buffer.clone().roll(-shift_d, -2)
    end_value = -(shift_steps - 1) * action_dim
    if end_value == 0:
        end_value = buffer.shape[-2]
    buffer[..., -shift_d:end_value, :] = buffer[..., -shift_d - action_dim : -shift_d, :].clone()

    return buffer


def jit_copy_buffer(ref_buffer, buffer):
    if buffer is not None:
        if ref_buffer is not None:
            ref_buffer.copy_(buffer)
        else:
            ref_buffer = buffer.clone()
    return ref_buffer


def find_first_idx(array, value, EQUAL=False):
    if EQUAL:
        f_idx = torch.nonzero(array >= value, as_tuple=False)[0].item()
    else:
        f_idx = torch.nonzero(array > value, as_tuple=False)[0].item()
    return f_idx


def find_last_idx(array, value):
    f_idx = torch.nonzero(array <= value, as_tuple=False)[-1].item()
    return f_idx


@get_torch_jit_decorator(slow_to_compile=True)
def round_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    y = torch.trunc(x + 0.5 * torch.sign(x))
    return y


def stable_topk(input_tensor: torch.Tensor, k: int, dim: int = -1, largest: bool = True):
    return torch.topk(input_tensor, k, dim=dim, largest=largest)

    # sort the input tensor
    sorted_input, sorted_indices = torch.sort(input_tensor, dim=dim, descending=largest)

    # get the top k indices
    topk_indices = sorted_indices[..., :k]

    # get the top k values
    topk_values = sorted_input[..., :k]

    return topk_values, topk_indices
