# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Any, Optional, Union

# Third Party
import torch

# CuRobo
from curobo._src.util.logging import log_warn

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_base_cfg import BaseCostCfg


class BaseCost:
    def __init__(self, config: BaseCostCfg):
        """Initialize class

        Args:
            config (Optional[CostConfig], optional): To initialize this class directly, pass a config.
            If this is a base class, it's assumed that you will initialize the child class with `CostConfig`.
            Defaults to None.
        """
        self.config = config
        self.device_cfg = config.device_cfg
        self._init_post_config()
        self._batch_size = -1
        self._horizon = -1
        self._dt = 1

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        if batch_size != self._batch_size or horizon != self._horizon:
            self._batch_size = batch_size
            self._horizon = horizon

    def _init_post_config(self):
        self._weight = self.config.weight.clone()
        self.cost_fn = None
        self._cost_enabled = True
        if torch.sum(self._weight) == 0.0:
            self.disable_cost()

    def forward(self, **kwargs) -> Union[Any, torch.Tensor]:
        """Compute the cost for a given input."""
        log_warn("BaseCost forward is not implemented")
        cost = torch.zeros(
            (self._batch_size, self._horizon, 1),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        return cost

    def disable_cost(self):
        if not self._cost_enabled:
            return
        self._weight *= 0.0
        self._cost_enabled = False

    def enable_cost(self):
        if self._cost_enabled:
            return
        self._weight.copy_(self.config.weight)
        if torch.sum(self._weight) == 0.0:
            self._cost_enabled = False
        else:
            self._cost_enabled = True

    @property
    def enabled(self):
        return self._cost_enabled

    def update_dt(self, dt: Union[float, torch.Tensor]):
        self._dt = dt

    def reset(
        self,
        reset_problem_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        pass

