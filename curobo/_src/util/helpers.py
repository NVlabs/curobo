# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standard Library
import math
from collections import defaultdict
from typing import List, Union

# Third Party
import torch

# CuRobo
from curobo._src.util.logging import log_and_raise


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def list_idx_if_not_none(d_list: List, idx: Union[int, torch.Tensor]):
    idx_list = []
    if isinstance(idx, int):
        for x in d_list:
            if x is not None:
                # print index and shape of x:
                if idx > len(x):
                    log_and_raise(f"Index {idx} out of range for {x.shape}")
                idx_list.append(x[idx])
            else:
                idx_list.append(None)
    else:
        for x in d_list:
            if x is not None:
                # if torch.max(idx) > len(x):
                #    log_and_raise(f"Index {torch.max(idx)} out of range for {x.shape}")
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
