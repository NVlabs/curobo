# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for DeviceCfg."""

# Third Party
import numpy as np
import pytest
import torch

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg


class TestDeviceCfg:
    """Test DeviceCfg class."""

    def test_default_initialization(self):
        """Test default tensor device configuration."""
        cfg = DeviceCfg()
        assert cfg.device == torch.device("cuda", 0)
        assert cfg.dtype == torch.float32
        assert cfg.collision_geometry_dtype == torch.float32
        assert cfg.collision_gradient_dtype == torch.float32
        assert cfg.collision_distance_dtype == torch.float32

    def test_initialization_with_cpu(self):
        """Test tensor device configuration with CPU."""
        cfg = DeviceCfg(device=torch.device("cpu"))
        assert cfg.device == torch.device("cpu")
        assert cfg.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_initialization_with_cuda(self):
        """Test tensor device configuration with CUDA."""
        cfg = DeviceCfg(device=torch.device("cuda", 0))
        assert cfg.device == torch.device("cuda", 0)

    def test_initialization_with_string_device(self):
        """Test tensor device configuration with string device."""
        cfg = DeviceCfg(device="cpu")
        assert cfg.device == torch.device("cpu")
        assert isinstance(cfg.device, torch.device)

    def test_initialization_with_float64(self):
        """Test tensor device configuration with float64."""
        cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float64)
        assert cfg.dtype == torch.float64

    def test_from_basic_cpu(self):
        """Test creating DeviceCfg from basic parameters."""
        cfg = DeviceCfg.from_basic("cpu", 0)
        assert cfg.device == torch.device("cpu", 0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_from_basic_cuda(self):
        """Test creating DeviceCfg from basic parameters with CUDA."""
        cfg = DeviceCfg.from_basic("cuda", 0)
        assert cfg.device == torch.device("cuda", 0)

    def test_to_device_with_torch_tensor(self):
        """Test converting torch tensor to device."""
        cfg = DeviceCfg(device=torch.device("cpu"))
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = cfg.to_device(tensor)
        assert result.device == cfg.device
        assert result.dtype == cfg.dtype

    def test_to_device_with_numpy_array(self):
        """Test converting numpy array to device."""
        cfg = DeviceCfg(device=torch.device("cpu"))
        array = np.array([1.0, 2.0, 3.0])
        result = cfg.to_device(array)
        assert result.device == cfg.device
        assert result.dtype == cfg.dtype
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_to_device_with_list(self):
        """Test converting list to device."""
        cfg = DeviceCfg(device=torch.device("cpu"))
        data = [1.0, 2.0, 3.0]
        result = cfg.to_device(data)
        assert result.device == cfg.device
        assert result.dtype == cfg.dtype
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_to_int8_device(self):
        """Test converting tensor to int8 on device."""
        cfg = DeviceCfg(device=torch.device("cpu"))
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = cfg.to_int8_device(tensor)
        assert result.device == cfg.device
        assert result.dtype == torch.int8
        assert torch.allclose(result, torch.tensor([1, 2, 3], dtype=torch.int8))

    def test_cpu(self):
        """Test creating CPU version of DeviceCfg."""
        cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float64)
        cpu_cfg = cfg.cpu()
        assert cpu_cfg.device == torch.device("cpu")
        assert cpu_cfg.dtype == torch.float64

    def test_as_torch_dict(self):
        """Test getting torch dictionary representation."""
        cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float64)
        torch_dict = cfg.as_torch_dict()
        assert torch_dict["device"] == torch.device("cpu")
        assert torch_dict["dtype"] == torch.float64
        assert len(torch_dict) == 2

    def test_collision_dtypes(self):
        """Test collision-specific dtypes."""
        cfg = DeviceCfg(
            device=torch.device("cpu"),
            collision_geometry_dtype=torch.float64,
            collision_gradient_dtype=torch.float16,
            collision_distance_dtype=torch.float64,
        )
        assert cfg.collision_geometry_dtype == torch.float64
        assert cfg.collision_gradient_dtype == torch.float16
        assert cfg.collision_distance_dtype == torch.float64

    def test_frozen_dataclass(self):
        """Test that DeviceCfg is frozen."""
        cfg = DeviceCfg()
        # This should raise an error because the dataclass is frozen
        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.11+
            cfg.device = torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_preserves_data(self):
        """Test that to_device preserves data values."""
        cfg = DeviceCfg(device=torch.device("cuda", 0))
        data = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        result = cfg.to_device(data)
        assert torch.allclose(result.cpu(), data)

    def test_as_torch_dict_usage(self):
        """Test using as_torch_dict for tensor creation."""
        cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float64)
        tensor = torch.zeros((3, 3), **cfg.as_torch_dict())
        assert tensor.device == cfg.device
        assert tensor.dtype == cfg.dtype

