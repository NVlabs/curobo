# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# CuRobo
from curobo._src.util.logging import log_warn
from curobo._src.util.torch_util import *

# Issue a deprecation warning using cuRobo's logger
log_warn(
    "The module 'curobo.util.torch_utils' is deprecated and will be removed in a future version. "
    "Please use 'curobo.util.torch_util' instead."
)
