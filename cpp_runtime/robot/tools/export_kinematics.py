#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for exporting kinematics-only robot artifacts."""

import sys

from export_robot_aot import main

if __name__ == "__main__":
    if "--kinematics-only" not in sys.argv:
        sys.argv.append("--kinematics-only")
    main()
