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

"""Unit tests for `curobo` package version."""

# CuRobo
import curobo


def test_curobo_version():
    """Test `curobo` package version is set."""
    assert curobo.__version__ is not None
    assert curobo.__version__ != ""
