# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Version information for cuRobo package."""

from curobo._src.util.version import get_version

__version__ = get_version()
del get_version
