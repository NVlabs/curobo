# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.util.tensor_util import (
    cat_max,
    cat_sum,
    check_nan_last_dimension,
    check_tensor_shapes,
    clone_if_not_none,
    copy_or_clone,
    copy_tensor,
    fd_tensor,
    find_first_idx,
    find_last_idx,
    jit_copy_buffer,
    round_away_from_zero,
    shift_buffer,
    stable_topk,
    tensor_repeat_seeds,
)


class TestCheckTensorShapes:
    def test_same_shapes(self):
        a = torch.zeros(3, 4)
        b = torch.ones(3, 4)
        assert check_tensor_shapes(a, b) is True

    def test_different_shapes(self):
        a = torch.zeros(3, 4)
        b = torch.ones(3, 5)
        result = check_tensor_shapes(a, b)
        # Function returns None for different shapes, not False
        assert result is None or result is False

    def test_different_ndim(self):
        a = torch.zeros(3, 4)
        b = torch.ones(3, 4, 5)
        assert check_tensor_shapes(a, b) is False

    def test_non_tensor(self):
        a = torch.zeros(3, 4)
        b = [1, 2, 3]
        assert check_tensor_shapes(a, b) is False


class TestCopyTensor:
    def test_copy_same_shape(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.zeros(3)
        result = copy_tensor(a, b)
        assert result is True
        assert torch.equal(a, b)

    def test_copy_different_shape(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.zeros(4)
        result = copy_tensor(a, b)
        assert result is False


class TestCopyOrClone:
    def test_both_none(self):
        result = copy_or_clone(None, None)
        assert result is None

    def test_clone_new_tensor(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        result = copy_or_clone(a, None, allow_clone=True)
        assert torch.equal(result, a)
        assert result is not a

    def test_copy_same_shape(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.zeros(3)
        result = copy_or_clone(a, b, allow_clone=True)
        assert torch.equal(result, a)
        assert result is b

    def test_error_different_shape_no_clone(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.zeros(4)
        with pytest.raises(ValueError):
            copy_or_clone(a, b, allow_clone=False)

    def test_new_tensor_none(self):
        b = torch.zeros(3)
        result = copy_or_clone(None, b)
        assert result is b


class TestCloneIfNotNone:
    def test_clone_tensor(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        result = clone_if_not_none(a)
        assert torch.equal(result, a)
        assert result is not a

    def test_none_input(self):
        result = clone_if_not_none(None)
        assert result is None


class TestCatSum:
    def test_cat_sum_basic(self):
        a = torch.ones(2, 3)
        b = torch.ones(2, 3)
        result = cat_sum([a, b], sum_dim=(0,))
        assert result.shape == (6,)


class TestCatMax:
    def test_cat_max_basic(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        result = cat_max([a, b])
        assert result.shape == (2, 2)


class TestTensorRepeatSeeds:
    def test_repeat_seeds(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0]])
        result = tensor_repeat_seeds(tensor, 3)
        assert result.shape == (3, 3)


class TestFdTensor:
    def test_fd_tensor(self):
        p = torch.ones(1, 5, 3)
        dt = torch.ones(1, 5)
        result = fd_tensor(p, dt)
        assert result.shape == (1, 4, 3)


class TestCheckNanLastDimension:
    def test_no_nan(self):
        tensor = torch.ones(2, 3, 4)
        result = check_nan_last_dimension(tensor)
        assert not result.any()

    def test_with_nan(self):
        tensor = torch.ones(2, 3, 4)
        tensor[0, 1, 2] = float("nan")
        result = check_nan_last_dimension(tensor)
        # Compare boolean value, not identity
        assert result[0, 1].item() == True


class TestShiftBuffer:
    def test_shift_buffer(self):
        buffer = torch.arange(12).float().reshape(1, 6, 2)
        result = shift_buffer(buffer, shift_d=2, action_dim=2)
        assert result.shape == buffer.shape


class TestJitCopyBuffer:
    def test_copy_buffer(self):
        ref = torch.zeros(3, 4)
        buf = torch.ones(3, 4)
        result = jit_copy_buffer(ref, buf)
        assert torch.equal(result, buf)

    def test_none_buffer(self):
        ref = torch.zeros(3, 4)
        result = jit_copy_buffer(ref, None)
        assert result is ref


class TestFindIdx:
    def test_find_first_idx(self):
        array = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        idx = find_first_idx(array, 2.5)
        assert idx == 2

    def test_find_first_idx_equal(self):
        array = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        idx = find_first_idx(array, 3.0, EQUAL=True)
        assert idx == 2

    def test_find_last_idx(self):
        array = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        idx = find_last_idx(array, 3.0)
        assert idx == 2


class TestRoundAwayFromZero:
    def test_positive_numbers(self):
        x = torch.tensor([1.4, 2.5, 3.6])
        result = round_away_from_zero(x)
        expected = torch.tensor([1.0, 3.0, 4.0])
        assert torch.equal(result, expected)

    def test_negative_numbers(self):
        x = torch.tensor([-1.4, -2.5, -3.6])
        result = round_away_from_zero(x)
        expected = torch.tensor([-1.0, -3.0, -4.0])
        assert torch.equal(result, expected)


class TestStableTopk:
    def test_stable_topk(self):
        tensor = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        values, indices = stable_topk(tensor, 3)
        assert len(values) == 3
        assert len(indices) == 3

