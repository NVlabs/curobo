# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for `curobo` package version."""

# CuRobo
import curobo


def test_curobo_version():
    """Test `curobo` package version is set."""
    assert curobo.__version__ is not None
    assert curobo.__version__ != ""
