# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the per-block shared-memory limit the self-collision kernels are sized against."""

# Standard Library
from types import SimpleNamespace

# Third Party
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _driver_optin_limit() -> int:
    from cuda.core import Device

    return int(Device(torch.cuda.current_device()).properties.max_shared_memory_per_block_optin)


def test_device_max_shared_memory_is_the_opt_in_limit():
    """The limit is the device's opt-in maximum, as the driver reports it."""
    from curobo._src.curobolib.backends.cuda_core_backend import geometry

    assert geometry.get_device_max_shared_memory(torch.device("cuda")) == _driver_optin_limit()


def test_device_max_shared_memory_without_torch_properties(monkeypatch: pytest.MonkeyPatch):
    """PyTorch before 2.7 reports neither shared-memory property; the driver's value is used."""
    from curobo._src.curobolib.backends.cuda_core_backend import geometry

    real = torch.cuda.get_device_properties(torch.cuda.current_device())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device=None: SimpleNamespace(name=real.name, total_memory=real.total_memory),
    )
    assert geometry.get_device_max_shared_memory(torch.device("cuda:0")) == _driver_optin_limit()
