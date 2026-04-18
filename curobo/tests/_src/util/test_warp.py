# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
from unittest.mock import MagicMock

# Third Party
import pytest

# CuRobo


class TestWarp:
    def test_init_warp(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import init_warp

            result = init_warp(quiet=True)
            assert result is True
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_sdf_struct(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import warp_support_sdf_struct

            result = warp_support_sdf_struct()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_kernel_key(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import warp_support_kernel_key

            result = warp_support_kernel_key()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_bvh_constructor_type(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import warp_support_bvh_constructor_type

            result = warp_support_bvh_constructor_type()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_sdf_struct_with_old_version(self):
        try:
            # CuRobo
            from curobo._src.util.warp import warp_support_sdf_struct

            mock_wp = MagicMock()
            mock_wp.config.version = "0.9.0"
            result = warp_support_sdf_struct(mock_wp)
            assert result is False
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_kernel_key_with_old_version(self):
        try:
            # CuRobo
            from curobo._src.util.warp import warp_support_kernel_key

            mock_wp = MagicMock()
            mock_wp.config.version = "1.0.0"
            result = warp_support_kernel_key(mock_wp)
            assert result is False
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_bvh_constructor_type_with_old_version(self):
        try:
            # CuRobo
            from curobo._src.util.warp import warp_support_bvh_constructor_type

            mock_wp = MagicMock()
            mock_wp.config.version = "1.5.0"
            result = warp_support_bvh_constructor_type(mock_wp)
            assert result is False
        except ImportError:
            pytest.skip("Warp not installed")

