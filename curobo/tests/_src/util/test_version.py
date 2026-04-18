# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
from unittest.mock import patch

# Third Party
# CuRobo
from curobo._src.util.version import get_version


class TestGetVersion:
    def test_get_version_returns_string(self):
        version = get_version()
        assert isinstance(version, str)
        assert len(version) > 0

    @patch("pathlib.Path.exists")
    @patch("setuptools_scm.get_version")
    def test_get_version_from_git(self, mock_get_version, mock_exists):
        mock_exists.side_effect = lambda: True
        mock_get_version.return_value = "0.8.0"
        version = get_version()
        assert isinstance(version, str)

    def test_get_version_fallback(self):
        with patch("pathlib.Path.exists", return_value=False):
            version = get_version()
            assert isinstance(version, str)

    @patch("pathlib.Path.exists")
    def test_get_version_with_importlib_metadata_error(self, mock_exists):
        """Test fallback when importlib.metadata fails."""
        mock_exists.return_value = False
        with patch("importlib.metadata.version", side_effect=Exception("Package not found")):
            version = get_version()
            assert version == "v0.8.0-no-tag"

    @patch("pathlib.Path.exists")
    def test_get_version_with_shallow_git(self, mock_exists):
        """Test when .git/shallow exists (should use importlib)."""

        def side_effect_exists(path=None):
            if path is None:
                return False
            path_str = str(path)
            if path_str.endswith(".git"):
                return True
            if path_str.endswith(".git/shallow"):
                return True
            return False

        mock_exists.side_effect = side_effect_exists
        version = get_version()
        assert isinstance(version, str)

