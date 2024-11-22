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
# Standard Library
import math
from collections import defaultdict
from typing import List


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def list_idx_if_not_none(d_list: List, idx: int):
    idx_list = []
    for x in d_list:
        if x is not None:
            idx_list.append(x[idx])
        else:
            idx_list.append(None)
    return idx_list


def robust_floor(x: float, threshold: float = 1e-04) -> int:
    nearest_int = round(x)
    if abs(x - nearest_int) < threshold:
        return nearest_int
    else:
        return int(math.floor(x))
