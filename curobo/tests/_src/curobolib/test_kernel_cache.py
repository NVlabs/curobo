# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cuda.core kernel cache helpers."""

from __future__ import annotations

from typing import Any

from curobo._src.curobolib.backends.cuda_core_backend.kernel_cache import (
    CudaCoreKernelCache,
)


class _FakeCudaDevice:
    """Record external stream wrappers created by the cache."""

    def __init__(self) -> None:
        self.created_streams: list[Any] = []

    def create_stream(self, stream_wrapper: Any) -> object:
        """Return a unique stand-in cuda.core stream."""
        self.created_streams.append(stream_wrapper)
        return object()


class _FakeTorchStream:
    """Provide the torch stream attributes consumed by the cache."""

    def __init__(self, device: str, cuda_stream: int) -> None:
        self.device = device
        self.cuda_stream = cuda_stream


def test_get_stream_wrapper_reuses_stream_for_same_device_and_handle() -> None:
    """Repeated lookup of one CUDA stream creates one cuda.core wrapper."""
    cache = CudaCoreKernelCache()
    device = _FakeCudaDevice()
    cache.device = device

    first = cache.get_stream_wrapper(_FakeTorchStream("cuda:0", 123))
    second = cache.get_stream_wrapper(_FakeTorchStream("cuda:0", 123))

    assert second is first
    assert len(device.created_streams) == 1


def test_get_stream_wrapper_separates_handles_and_devices() -> None:
    """Distinct CUDA stream identities receive distinct cuda.core wrappers."""
    cache = CudaCoreKernelCache()
    device = _FakeCudaDevice()
    cache.device = device

    streams = (
        cache.get_stream_wrapper(_FakeTorchStream("cuda:0", 123)),
        cache.get_stream_wrapper(_FakeTorchStream("cuda:0", 456)),
        cache.get_stream_wrapper(_FakeTorchStream("cuda:1", 123)),
    )

    assert len({id(stream) for stream in streams}) == 3
    assert len(device.created_streams) == 3
