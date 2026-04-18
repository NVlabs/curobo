# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Joint type definitions for robot kinematics."""

from __future__ import annotations

from enum import Enum


class JointType(Enum):
    """Type of Joint. Arbitrary axis of change is not supported."""

    #: Fixed joint.
    FIXED = -1

    #: Prismatic joint along x-axis.
    X_PRISM = 0

    #: Prismatic joint along y-axis.
    Y_PRISM = 1

    #: Prismatic joint along z-axis.
    Z_PRISM = 2

    #: Revolute joint along x-axis.
    X_ROT = 3

    #: Revolute joint along y-axis.
    Y_ROT = 4

    #: Revolute joint along z-axis.
    Z_ROT = 5

    #: Prismatic joint along negative x-axis.
    X_PRISM_NEG = 6

    #: Prismatic joint along negative y-axis.
    Y_PRISM_NEG = 7

    #: Prismatic joint along negative z-axis.
    Z_PRISM_NEG = 8

    #: Revolute joint along negative x-axis.
    X_ROT_NEG = 9

    #: Revolute joint along negative y-axis.
    Y_ROT_NEG = 10

    #: Revolute joint along negative z-axis.
    Z_ROT_NEG = 11

