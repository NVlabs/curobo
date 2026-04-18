# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CuRobo
from curobo._src.types.pose import *
from curobo._src.util.logging import log_warn

log_warn(
    "The module 'curobo.types.math' is deprecated and will be removed in a future version. "
    "Please use 'curobo.types.pose' instead."
)
