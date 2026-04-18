# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""CUDA utilities for stream management and synchronization."""

# Standard Library
from contextlib import contextmanager, nullcontext
from typing import Dict, Optional

# Third Party
import torch

from curobo import runtime as curobo_runtime


@contextmanager
def cuda_stream_context(
    stream_name: str,
    streams_dict: Dict[str, torch.cuda.Stream],
    events_dict: Dict[str, torch.cuda.Event],
    device: torch.device,
    enabled: Optional[bool] = None,
):
    """Context manager for computation with optional CUDA streams.

    Handles stream synchronization and event recording when streams are enabled.
    When streams are disabled, acts as a no-op context manager.

    Args:
        stream_name: Name/key of the stream to use
        streams_dict: Dictionary mapping names to CUDA streams
        events_dict: Dictionary mapping names to CUDA events
        device: CUDA device
        enabled: Whether to use streams. If None, checks config.cuda_streams

    Yields:
        Context for the computation

    Example:
        >>> with cuda_stream_context("my_cost", self._costs_streams,
        ...                          self._costs_events, self.device_cfg.device):
        ...     result = compute_something()
    """
    if enabled is None:
        enabled = curobo_runtime.cuda_streams

    use_stream = enabled and stream_name in streams_dict

    # Wait for current stream before switching
    if use_stream:
        streams_dict[stream_name].wait_stream(torch.cuda.current_stream(device=device))

    # Create appropriate context
    stream_ctx = torch.cuda.stream(streams_dict[stream_name]) if use_stream else nullcontext()

    try:
        with stream_ctx:
            yield
    finally:
        # Record event after computation
        if use_stream:
            streams_dict[stream_name].record_event(events_dict[stream_name])


def synchronize_cuda_streams(
    events_dict: Dict[str, torch.cuda.Event],
    device: torch.device,
    enabled: Optional[bool] = None,
):
    """Synchronize all CUDA streams to the current stream.

    Waits for all events in the events dictionary to complete before continuing.
    This ensures all parallel stream computations have finished.

    Args:
        events_dict: Dictionary mapping names to CUDA events
        device: CUDA device
        enabled: Whether streams are enabled. If None, checks config.cuda_streams

    Example:
        >>> synchronize_cuda_streams(self._costs_events, self.device_cfg.device)
    """
    if enabled is None:
        enabled = curobo_runtime.cuda_streams

    if enabled:
        current_stream = torch.cuda.current_stream(device=device)
        for event in events_dict.values():
            current_stream.wait_event(event)


def create_cuda_stream_pair(device: torch.device, enabled: Optional[bool] = None) -> tuple:
    """Create a CUDA stream and event pair if enabled.

    Args:
        device: CUDA device
        enabled: Whether to create actual CUDA objects. If None, checks config

    Returns:
        Tuple of (stream, event), or (None, None) if disabled

    Example:
        >>> stream, event = create_cuda_stream_pair(self.device_cfg.device)
    """
    if enabled is None:
        # CuRobo

        enabled = curobo_runtime.cuda_streams

    if enabled:
        return torch.cuda.Stream(device=device), torch.cuda.Event()
    return None, None
