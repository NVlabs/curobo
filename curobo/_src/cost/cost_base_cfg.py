# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Type, Union

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


@dataclass
class BaseCostCfg:
    #: Weight vector for the cost term. This should be of shape (num_costs). The same weight
    #: will be applied to batch size and horizon.
    weight: Union[torch.Tensor, float, List[float]]

    class_type: Type[BaseCost] = BaseCost

    #: Device and dtype for the tensor.
    device_cfg: DeviceCfg = DeviceCfg()

    #: Whether to convert the cost to a classification problem. Output will be 1 if cost is
    #: greater than 0, otherwise 0.
    convert_to_binary: bool = False

    #: When True, the backward for the cost function will consider any operations done after the
    #: cost function. If the cost output is only summed, followed by a backward pass, this
    #: should be set to False
    use_grad_input: bool = False


    def __post_init__(self):
        if isinstance(self.weight, int):
            log_and_raise("BaseCostCfg: weight must be a tensor, float, or list of floats, got int")
        if isinstance(self.weight, float):
            self.weight = self.device_cfg.to_device([self.weight])
        elif isinstance(self.weight, list):
            self.weight = self.device_cfg.to_device(self.weight)
        elif isinstance(self.weight, torch.Tensor):
            self.weight = self.weight.to(self.device_cfg.device, dtype=self.device_cfg.dtype)


    def clone(self):
        return BaseCostCfg(
            weight=self.weight.clone(),
            device_cfg=self.device_cfg,
            convert_to_binary=self.convert_to_binary,
        )

