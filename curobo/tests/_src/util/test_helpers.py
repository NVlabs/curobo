# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.util.helpers import default_to_regular, list_idx_if_not_none, robust_floor


class TestDefaultToRegular:
    def test_regular_dict(self):
        d = {"a": 1, "b": 2}
        result = default_to_regular(d)
        assert result == {"a": 1, "b": 2}

    def test_defaultdict(self):
        from collections import defaultdict

        d = defaultdict(int)
        d["a"] = 1
        d["b"] = 2
        result = default_to_regular(d)
        assert result == {"a": 1, "b": 2}
        assert not isinstance(result, defaultdict)

    def test_nested_defaultdict(self):
        from collections import defaultdict

        d = defaultdict(lambda: defaultdict(int))
        d["a"]["x"] = 1
        d["b"]["y"] = 2
        result = default_to_regular(d)
        assert result == {"a": {"x": 1}, "b": {"y": 2}}
        assert not isinstance(result["a"], defaultdict)


class TestListIdxIfNotNone:
    def test_with_int_index(self):
        d_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        result = list_idx_if_not_none(d_list, 1)
        assert len(result) == 2
        assert result[0] == 2
        assert result[1] == 5

    def test_with_none_elements(self):
        d_list = [torch.tensor([1, 2, 3]), None, torch.tensor([4, 5, 6])]
        result = list_idx_if_not_none(d_list, 1)
        assert len(result) == 3
        assert result[0] == 2
        assert result[1] is None
        assert result[2] == 5

    def test_with_tensor_index(self):
        d_list = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8])]
        idx = torch.tensor([0, 2])
        result = list_idx_if_not_none(d_list, idx)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([1, 3]))
        assert torch.equal(result[1], torch.tensor([5, 7]))

    def test_out_of_range_error(self):
        d_list = [torch.tensor([1, 2, 3])]
        with pytest.raises(ValueError):
            list_idx_if_not_none(d_list, 10)


class TestRobustFloor:
    def test_exact_integer(self):
        assert robust_floor(5.0) == 5
        assert robust_floor(10.0) == 10

    def test_near_integer(self):
        assert robust_floor(5.00001) == 5
        assert robust_floor(4.99999) == 5

    def test_far_from_integer(self):
        assert robust_floor(5.5) == 5
        assert robust_floor(4.3) == 4

    def test_negative_numbers(self):
        assert robust_floor(-5.0) == -5
        assert robust_floor(-5.5) == -6

    def test_custom_threshold(self):
        assert robust_floor(5.001, threshold=0.01) == 5
        assert robust_floor(5.02, threshold=0.01) == 5

