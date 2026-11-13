# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from curobo._src.curobolib.backends.cuda_core_backend.kernel_cache import CudaCoreKernelCache


class _TestDevice:
    def __init__(self) -> None:
        self.created_streams: list[tuple[Any, object]] = []

    def create_stream(self, stream_adapter: Any) -> object:
        stream = object()
        self.created_streams.append((stream_adapter, stream))
        return stream


class _TestTorchStream:
    def __init__(self, handle: int) -> None:
        self.device = torch.device("cuda:0")
        self.cuda_stream = handle


def test_kernel_key_tracks_source_contents_and_compile_options(tmp_path: Path) -> None:
    """Distinct source boundaries, flags, and architectures cannot alias."""
    cache = CudaCoreKernelCache()
    cache.arch = "sm_90"
    first_source = tmp_path / "first.cuh"
    second_source = tmp_path / "second.cuh"
    first_source.write_bytes(b"ab")
    second_source.write_bytes(b"c")
    sources = [first_source, second_source]
    original = cache.get_kernel_key(sources, "kernel", ["a_b", "c"])
    assert original == cache.get_kernel_key(sources, "kernel", ["a_b", "c"])
    assert original != cache.get_kernel_key(sources, "other_kernel", ["a_b", "c"])
    assert original != cache.get_kernel_key(sources, "kernel", ["a", "b_c"])
    first_source.write_bytes(b"a")
    second_source.write_bytes(b"bc")
    assert original != cache.get_kernel_key(sources, "kernel", ["a_b", "c"])
    first_source.write_bytes(b"ab")
    second_source.write_bytes(b"c")
    cache.arch = "sm_120"
    assert original != cache.get_kernel_key(sources, "kernel", ["a_b", "c"])


def test_prepared_kernel_cache_skips_key_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = CudaCoreKernelCache()
    cache.device = _TestDevice()
    cache.arch = "sm_90"
    kernel = object()
    get_kernel_key = Mock(return_value="content_key")
    compile_kernel = Mock(return_value=kernel)
    monkeypatch.setattr(cache, "get_kernel_key", get_kernel_key)
    monkeypatch.setattr(cache, "_compile_kernel", compile_kernel)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())

    source_files = [tmp_path / "kernel.cuh"]
    include_dirs = [tmp_path]
    compile_flags = ["-O3"]
    first = cache.get_or_compile_kernel(source_files, "kernel<float>", include_dirs, compile_flags)
    second = cache.get_or_compile_kernel(
        source_files, "kernel<float>", include_dirs, compile_flags
    )

    assert first is kernel
    assert second is kernel
    get_kernel_key.assert_called_once()
    compile_kernel.assert_called_once()


def test_kernel_cache_clear_forces_key_rebuild_and_compile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = CudaCoreKernelCache()
    cache.device = _TestDevice()
    cache.arch = "sm_90"
    get_kernel_key = Mock(side_effect=["first_key", "second_key"])
    compile_kernel = Mock(side_effect=[object(), object()])
    monkeypatch.setattr(cache, "get_kernel_key", get_kernel_key)
    monkeypatch.setattr(cache, "_compile_kernel", compile_kernel)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())

    args = ([tmp_path / "kernel.cuh"], "kernel<float>", [tmp_path], ["-O3"])
    first = cache.get_or_compile_kernel(*args)
    cache.clear()
    second = cache.get_or_compile_kernel(*args)

    assert first is not second
    assert get_kernel_key.call_count == 2
    assert compile_kernel.call_count == 2


def test_stream_wrappers_are_cached_by_device_and_handle() -> None:
    cache = CudaCoreKernelCache()
    device = _TestDevice()
    cache.device = device
    cache.arch = "sm_90"
    first_stream = _TestTorchStream(10)
    same_native_stream = _TestTorchStream(10)
    other_stream = _TestTorchStream(20)

    first_wrapper = cache.get_stream_wrapper(first_stream)
    repeated_wrapper = cache.get_stream_wrapper(same_native_stream)
    other_wrapper = cache.get_stream_wrapper(other_stream)

    assert repeated_wrapper is first_wrapper
    assert other_wrapper is not first_wrapper
    assert len(device.created_streams) == 2

    cache.clear()
    cleared_wrapper = cache.get_stream_wrapper(first_stream)
    assert cleared_wrapper is not first_wrapper
    assert len(device.created_streams) == 3
