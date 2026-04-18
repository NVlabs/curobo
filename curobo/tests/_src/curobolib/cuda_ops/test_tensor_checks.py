# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tensor validation utilities."""

from __future__ import annotations

import pytest
import torch

from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_bool_tensors,
    check_float16_tensors,
    check_float32_tensors,
    check_int8_tensors,
    check_int16_tensors,
    check_int32_tensors,
    check_uint8_tensors,
)

CPU = torch.device("cpu")


class TestCheckFloat32Tensors:
    def test_valid_tensors(self):
        a = torch.zeros(3, 4, dtype=torch.float32)
        b = torch.ones(2, dtype=torch.float32)
        check_float32_tensors(CPU, a=a, b=b)

    def test_wrong_device_raises(self):
        t = torch.zeros(3, dtype=torch.float32)
        fake_device = torch.device("cpu", 1)
        with pytest.raises(ValueError, match="my_tensor.*expected device"):
            check_float32_tensors(fake_device, my_tensor=t)

    def test_non_contiguous_raises(self):
        t = torch.zeros(4, 4, dtype=torch.float32).t()
        assert not t.is_contiguous()
        with pytest.raises(ValueError, match="bad_tensor.*expected contiguous"):
            check_float32_tensors(CPU, bad_tensor=t)

    def test_wrong_dtype_raises(self):
        t = torch.zeros(3, dtype=torch.float16)
        with pytest.raises(ValueError, match="half_tensor.*expected dtype.*float32"):
            check_float32_tensors(CPU, half_tensor=t)

    def test_empty_kwargs_passes(self):
        check_float32_tensors(CPU)

    def test_single_element(self):
        t = torch.tensor([1.0], dtype=torch.float32)
        check_float32_tensors(CPU, scalar=t)


class TestCheckFloat16Tensors:
    def test_valid(self):
        t = torch.zeros(2, 3, dtype=torch.float16)
        check_float16_tensors(CPU, t=t)

    def test_float32_rejected(self):
        t = torch.zeros(2, dtype=torch.float32)
        with pytest.raises(ValueError, match="expected dtype.*float16"):
            check_float16_tensors(CPU, t=t)


class TestCheckInt8Tensors:
    def test_valid(self):
        t = torch.zeros(5, dtype=torch.int8)
        check_int8_tensors(CPU, idx=t)

    def test_int16_rejected(self):
        t = torch.zeros(5, dtype=torch.int16)
        with pytest.raises(ValueError, match="expected dtype.*int8"):
            check_int8_tensors(CPU, idx=t)


class TestCheckUint8Tensors:
    def test_valid(self):
        t = torch.zeros(3, dtype=torch.uint8)
        check_uint8_tensors(CPU, flags=t)

    def test_bool_rejected(self):
        t = torch.zeros(3, dtype=torch.bool)
        with pytest.raises(ValueError, match="expected dtype.*uint8"):
            check_uint8_tensors(CPU, flags=t)


class TestCheckInt16Tensors:
    def test_valid(self):
        t = torch.zeros(4, dtype=torch.int16)
        check_int16_tensors(CPU, map_tensor=t)

    def test_int32_rejected(self):
        t = torch.zeros(4, dtype=torch.int32)
        with pytest.raises(ValueError, match="expected dtype.*int16"):
            check_int16_tensors(CPU, map_tensor=t)


class TestCheckInt32Tensors:
    def test_valid(self):
        t = torch.zeros(2, dtype=torch.int32)
        check_int32_tensors(CPU, idx=t)

    def test_int16_rejected(self):
        t = torch.zeros(2, dtype=torch.int16)
        with pytest.raises(ValueError, match="expected dtype.*int32"):
            check_int32_tensors(CPU, idx=t)


class TestCheckBoolTensors:
    def test_valid(self):
        t = torch.zeros(3, dtype=torch.bool)
        check_bool_tensors(CPU, mask=t)

    def test_uint8_rejected(self):
        t = torch.zeros(3, dtype=torch.uint8)
        with pytest.raises(ValueError, match="expected dtype.*bool"):
            check_bool_tensors(CPU, mask=t)


class TestMultipleTensors:
    def test_error_includes_tensor_name(self):
        """Error message should identify which tensor failed."""
        good = torch.zeros(3, dtype=torch.float32)
        bad = torch.zeros(3, dtype=torch.float16)
        with pytest.raises(ValueError, match="bad_one.*expected dtype.*float32"):
            check_float32_tensors(CPU, good_one=good, bad_one=bad)

    def test_first_failure_reported(self):
        """With multiple bad tensors, the first one (in iteration order) is reported."""
        wrong_dtype = torch.zeros(3, dtype=torch.float16)
        also_wrong = torch.zeros(3, dtype=torch.int32)
        with pytest.raises(ValueError, match="wrong_one"):
            check_float32_tensors(CPU, wrong_one=wrong_dtype, also_wrong=also_wrong)
