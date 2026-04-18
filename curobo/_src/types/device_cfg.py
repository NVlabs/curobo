# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Device configuration for tensor operations."""

# Standard Library
from dataclasses import dataclass

# Third Party
import numpy as np
import torch


@dataclass(frozen=True)
class DeviceCfg:
    """Configuration for device and data types used in tensor operations."""

    device: torch.device = torch.device("cuda", 0)
    dtype: torch.dtype = torch.float32
    collision_geometry_dtype: torch.dtype = torch.float32
    collision_gradient_dtype: torch.dtype = torch.float32
    collision_distance_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if isinstance(self.device, str):
            object.__setattr__(self, "device", torch.device(self.device))
            # log_and_raise(f"device is a string: {self.device}")

    @staticmethod
    def from_basic(device: str, dev_id: int):
        return DeviceCfg(torch.device(device, dev_id))

    def to_device(self, data_tensor):
        if isinstance(data_tensor, torch.Tensor):
            return data_tensor.to(device=self.device, dtype=self.dtype)
        else:
            return torch.as_tensor(np.array(data_tensor), device=self.device, dtype=self.dtype)

    def to_int8_device(self, data_tensor):
        return data_tensor.to(device=self.device, dtype=torch.int8)

    def cpu(self):
        return DeviceCfg(device=torch.device("cpu"), dtype=self.dtype)

    def as_torch_dict(self):
        return {"device": self.device, "dtype": self.dtype}

    def is_same_torch_device(self, other: torch.device) -> bool:
        """Check if a torch.device refers to the same physical device as this config.

        Handles the case where "cuda" and "cuda:0" are equivalent devices.

        Args:
            other: torch.device to compare against.

        Returns:
            True if devices refer to the same physical device.
        """
        if self.device == other:
            return True
        # Normalize devices: "cuda" without index defaults to "cuda:0"
        self_type = self.device.type
        self_index = self.device.index if self.device.index is not None else 0
        other_type = other.type
        other_index = other.index if other.index is not None else 0
        return self_type == other_type and self_index == other_index

