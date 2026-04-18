# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library

# CuRobo
from curobo._src.util.python_util import ceildiv


class TestCeilDiv:
    def test_ceildiv_basic(self):
        assert ceildiv(10, 3) == 4
        assert ceildiv(9, 3) == 3
        assert ceildiv(1, 1) == 1

    def test_ceildiv_exact_division(self):
        assert ceildiv(10, 5) == 2
        assert ceildiv(100, 10) == 10

    def test_ceildiv_one_remainder(self):
        assert ceildiv(11, 5) == 3
        assert ceildiv(21, 10) == 3

    def test_ceildiv_edge_cases(self):
        assert ceildiv(1, 10) == 1
        assert ceildiv(5, 10) == 1

