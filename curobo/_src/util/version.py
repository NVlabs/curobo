# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Version detection utilities for cuRobo package."""


def get_version() -> str:
    """Return the version string for the cuRobo package.

    Returns version from git tags in development mode, or from package
    metadata in installed mode.

    Returns:
        Version string.
    """
    # Standard Library
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    if (root / ".git").exists() and not (root / ".git/shallow").exists():
        # Third Party
        import setuptools_scm

        # See the `setuptools_scm` documentation for the description of the schemes used below.
        # https://pypi.org/project/setuptools-scm/
        # NOTE: If these values are updated, they need to be also updated in `pyproject.toml`.
        return setuptools_scm.get_version(
            root=root,
            version_scheme="no-guess-dev",
            local_scheme="dirty-tag",
        )
    else:  # Get the version from the _version.py setuptools_scm file.
        try:
            # Standard Library
            from importlib.metadata import version
        except ModuleNotFoundError:
            # NOTE: `importlib.resources` is part of the standard library in Python 3.9.
            # `importlib_metadata` is the back ported library for older versions of python.
            # Third Party
            from importlib_metadata import version
        try:
            return version("nvidia_curobo")
        except:
            return "v0.8.0-no-tag"
