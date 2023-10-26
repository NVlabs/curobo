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
from dataclasses import dataclass
from typing import List, Optional, Union

# Third Party
import torch

# CuRobo
from curobo.types.base import TensorDeviceType


@dataclass
class CostConfig:
    weight: Union[torch.Tensor, float, List[float]]
    tensor_args: TensorDeviceType = None
    distance_threshold: float = 0.0
    classify: bool = False
    terminal: bool = False
    run_weight: Optional[float] = None
    dof: int = 7
    vec_weight: Optional[Union[torch.Tensor, List[float], float]] = None
    max_value: Optional[float] = None
    hinge_value: Optional[float] = None
    vec_convergence: Optional[List[float]] = None
    threshold_value: Optional[float] = None
    return_loss: bool = False

    def __post_init__(self):
        self.weight = self.tensor_args.to_device(self.weight)
        if len(self.weight.shape) == 0:
            self.weight = torch.tensor(
                [self.weight], device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
        if self.vec_weight is not None:
            self.vec_weight = self.tensor_args.to_device(self.vec_weight)
        if self.max_value is not None:
            self.max_value = self.tensor_args.to_device(self.max_value)
        if self.hinge_value is not None:
            self.hinge_value = self.tensor_args.to_device(self.hinge_value)
        if self.threshold_value is not None:
            self.threshold_value = self.tensor_args.to_device(self.threshold_value)

    def update_vec_weight(self, vec_weight):
        self.vec_weight = self.tensor_args.to_device(vec_weight)


class CostBase(torch.nn.Module, CostConfig):
    def __init__(self, config: Optional[CostConfig] = None):
        """Initialize class

        Args:
            config (Optional[CostConfig], optional): To initialize this class directly, pass a config.
            If this is a base class, it's assumed that you will initialize the child class with `CostConfig`.
            Defaults to None.
        """
        self._run_weight_vec = None
        super(CostBase, self).__init__()
        if config is not None:
            CostConfig.__init__(self, **vars(config))
        CostBase._init_post_config(self)
        self._batch_size = -1
        self._horizon = -1
        self._dof = -1
        self._dt = 1

    def _init_post_config(self):
        self._weight = self.weight.clone()
        self.cost_fn = None
        self._cost_enabled = True
        self._z_scalar = self.tensor_args.to_device(0.0)
        if torch.sum(self.weight) == 0.0:
            self.disable_cost()

    def forward(self, q):
        batch_size = q.shape[0]
        horizon = q.shape[1]
        q = q.view(batch_size * horizon, q.shape[2])

        res = self.cost_fn(q)

        res = res.view(batch_size, horizon)
        res += self.distance_threshold
        res = torch.nn.functional.relu(res, inplace=True)
        if self.classify:
            res = torch.where(res > 0, res + 1.0, res)
        cost = self.weight * res
        return cost

    def disable_cost(self):
        self.weight.copy_(self._weight * 0.0)
        self._cost_enabled = False

    def enable_cost(self):
        self.weight.copy_(self._weight.clone())
        if torch.sum(self.weight) == 0.0:
            self._cost_enabled = False
        else:
            self._cost_enabled = True

    def update_weight(self, weight: float):
        if weight == 0.0:
            self.disable_cost()
        else:
            self.weight.copy_(self._weight * 0.0 + weight)

    @property
    def enabled(self):
        return self._cost_enabled

    def update_dt(self, dt: Union[float, torch.Tensor]):
        self._dt = dt
