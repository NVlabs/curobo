# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library
from enum import Enum


class PoseErrorType(Enum):
    SINGLE_GOAL = 0  #: Distance will be computed to a single goal pose
    BATCH_GOAL = 1  #: Distance will be computed pairwise between query batch and goal batch
    GOALSET = 2  #: Shortest Distance will be computed to a goal set
    BATCH_GOALSET = 3  #: Shortest Distance to a batch goal set
