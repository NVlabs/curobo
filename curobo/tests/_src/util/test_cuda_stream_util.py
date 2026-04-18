# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.util.cuda_stream_util import (
    create_cuda_stream_pair,
    cuda_stream_context,
    synchronize_cuda_streams,
)


class TwoStreamCostComputer:
    """Example class that computes two costs in parallel streams."""

    def __init__(self, device: torch.device, enabled: bool = True):
        self.device = device
        self.enabled = enabled

        # Create streams and events
        self._streams = {}
        self._events = {}

        if enabled and torch.cuda.is_available():
            self._streams["cost1"], self._events["cost1"] = create_cuda_stream_pair(
                device, enabled=True
            )
            self._streams["cost2"], self._events["cost2"] = create_cuda_stream_pair(
                device, enabled=True
            )

    def compute_two_costs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute two different costs in parallel streams and add them.

        Args:
            x: First input tensor
            y: Second input tensor

        Returns:
            Total cost (cost1 + cost2)
        """
        cost1 = None
        cost2 = None

        # Compute first cost in stream 1
        with cuda_stream_context("cost1", self._streams, self._events, self.device, self.enabled):
            cost1 = (x ** 2).sum()

        # Compute second cost in stream 2
        with cuda_stream_context("cost2", self._streams, self._events, self.device, self.enabled):
            cost2 = (y ** 3).mean()

        # Synchronize both streams before combining results
        synchronize_cuda_streams(self._events, self.device, self.enabled)

        # Combine costs
        total_cost = cost1 + cost2

        return total_cost


class TestCudaStreamContext:
    @pytest.fixture
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_cuda_stream_context_disabled(self, device):
        """Test stream context when disabled."""
        streams = {}
        events = {}

        with cuda_stream_context("test", streams, events, device, enabled=False):
            x = torch.tensor([1.0, 2.0], device=device)
            result = x + 1

        assert torch.allclose(result, torch.tensor([2.0, 3.0], device=device))

    def test_cuda_stream_context_enabled(self, device):
        """Test stream context when enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        streams = {"test": torch.cuda.Stream(device=device)}
        events = {"test": torch.cuda.Event()}

        with cuda_stream_context("test", streams, events, device, enabled=True):
            x = torch.tensor([1.0, 2.0], device=device)
            result = x + 1

        assert torch.allclose(result, torch.tensor([2.0, 3.0], device=device))

    def test_cuda_stream_context_stream_not_in_dict(self, device):
        """Test when stream name is not in dictionary."""
        streams = {}
        events = {}

        with cuda_stream_context("nonexistent", streams, events, device, enabled=True):
            x = torch.tensor([1.0, 2.0], device=device)
            result = x + 1

        assert torch.allclose(result, torch.tensor([2.0, 3.0], device=device))


class TestSynchronizeCudaStreams:
    @pytest.fixture
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_synchronize_disabled(self, device):
        """Test synchronization when disabled."""
        events = {}
        # Should not raise any errors
        synchronize_cuda_streams(events, device, enabled=False)

    def test_synchronize_enabled(self, device):
        """Test synchronization when enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        events = {
            "event1": torch.cuda.Event(),
            "event2": torch.cuda.Event(),
        }
        events["event1"].record()
        events["event2"].record()

        # Should not raise any errors
        synchronize_cuda_streams(events, device, enabled=True)

    def test_synchronize_default_enabled(self, device):
        """Test synchronization with default enabled setting."""
        events = {}
        # Should not raise any errors
        synchronize_cuda_streams(events, device, enabled=None)


class TestCreateCudaStreamPair:
    @pytest.fixture
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_create_pair_disabled(self, device):
        """Test creating stream pair when disabled."""
        stream, event = create_cuda_stream_pair(device, enabled=False)
        assert stream is None
        assert event is None

    def test_create_pair_enabled(self, device):
        """Test creating stream pair when enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        stream, event = create_cuda_stream_pair(device, enabled=True)
        assert stream is not None
        assert event is not None
        assert isinstance(stream, torch.cuda.Stream)
        assert isinstance(event, torch.cuda.Event)

    def test_create_pair_default(self, device):
        """Test creating stream pair with default setting."""
        stream, event = create_cuda_stream_pair(device, enabled=None)
        # Result depends on runtime config
        assert (stream is None and event is None) or (stream is not None and event is not None)


class TestTwoStreamCostComputer:
    """Test the example parallel cost computation class."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_two_stream_computation_disabled(self, device):
        """Test two-stream computation when disabled."""
        computer = TwoStreamCostComputer(device, enabled=False)

        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        y = torch.tensor([1.0, 2.0, 3.0], device=device)

        total_cost = computer.compute_two_costs(x, y)

        # cost1 = (1^2 + 2^2 + 3^2) = 14
        # cost2 = (1^3 + 2^3 + 3^3) / 3 = 36 / 3 = 12
        # total = 14 + 12 = 26
        expected = torch.tensor(26.0, device=device)
        assert torch.allclose(total_cost, expected)

    def test_two_stream_computation_enabled(self, device):
        """Test two-stream computation when enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        computer = TwoStreamCostComputer(device, enabled=True)

        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        y = torch.tensor([1.0, 2.0, 3.0], device=device)

        total_cost = computer.compute_two_costs(x, y)

        # Same expected result regardless of streams
        expected = torch.tensor(26.0, device=device)
        assert torch.allclose(total_cost, expected)

    def test_different_inputs(self, device):
        """Test with different inputs."""
        computer = TwoStreamCostComputer(device, enabled=False)

        x = torch.tensor([2.0, 3.0], device=device)
        y = torch.tensor([1.0, 2.0], device=device)

        total_cost = computer.compute_two_costs(x, y)

        # cost1 = (2^2 + 3^2) = 13
        # cost2 = (1^3 + 2^3) / 2 = 9 / 2 = 4.5
        # total = 17.5
        expected = torch.tensor(17.5, device=device)
        assert torch.allclose(total_cost, expected)

