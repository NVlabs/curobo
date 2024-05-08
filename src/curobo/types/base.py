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

# Third Party
import numpy as np
import torch


@dataclass(frozen=True)
class TensorDeviceType:
    device: torch.device = torch.device("cuda", 0)
    dtype: torch.dtype = torch.float32
    collision_geometry_dtype: torch.dtype = torch.float32
    collision_gradient_dtype: torch.dtype = torch.float32
    collision_distance_dtype: torch.dtype = torch.float32

    @staticmethod
    def from_basic(device: str, dev_id: int):
        return TensorDeviceType(torch.device(device, dev_id))

    def to_device(self, data_tensor):
        if isinstance(data_tensor, torch.Tensor):
            return data_tensor.to(device=self.device, dtype=self.dtype)
        else:
            return torch.as_tensor(np.array(data_tensor), device=self.device, dtype=self.dtype)

    def to_int8_device(self, data_tensor):
        return data_tensor.to(device=self.device, dtype=torch.int8)

    def cpu(self):
        return TensorDeviceType(device=torch.device("cpu"), dtype=self.dtype)

    def as_torch_dict(self):
        return {"device": self.device, "dtype": self.dtype}
