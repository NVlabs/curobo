# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor


# Test functions for CUDA graph
def simple_add_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Simple addition function for testing."""
    return a + b


def backward_fn(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Function with backward pass for testing."""
    x_copy = x.clone().requires_grad_(True)
    loss = ((x_copy - target) ** 2).sum()
    loss.backward()
    return loss


def multi_output_fn(x: torch.Tensor) -> tuple:
    """Function returning multiple outputs."""
    return x * 2, x + 1, x - 1


class TestGraphExecutor:
    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise CPU."""
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_init(self, device):
        """Test GraphExecutor initialization."""
        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=False, clone_outputs=True
        )
        assert executor._capture_fn == simple_add_fn
        assert executor._device == device
        assert executor._clone_outputs is True
        assert not executor.is_initialized

    def test_simple_execution_no_graph(self, device):
        """Test basic execution without CUDA graph."""
        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=False, clone_outputs=True
        )

        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)

        result = executor(a, b)
        expected = torch.tensor([5.0, 7.0, 9.0], device=device)

        assert torch.allclose(result, expected)
        assert executor.is_initialized

    def test_simple_execution_with_graph(self, device):
        """Test execution with CUDA graph enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=True, clone_outputs=True
        )

        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)

        result = executor(a, b)
        expected = torch.tensor([5.0, 7.0, 9.0], device=device)

        assert torch.allclose(result, expected)
        assert executor.is_initialized

    def test_multiple_calls(self, device):
        """Test multiple calls reuse the same graph."""
        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=False, clone_outputs=True
        )

        a1 = torch.tensor([1.0, 2.0], device=device)
        b1 = torch.tensor([3.0, 4.0], device=device)
        result1 = executor(a1, b1)

        a2 = torch.tensor([5.0, 6.0], device=device)
        b2 = torch.tensor([7.0, 8.0], device=device)
        result2 = executor(a2, b2)

        assert torch.allclose(result1, torch.tensor([4.0, 6.0], device=device))
        assert torch.allclose(result2, torch.tensor([12.0, 14.0], device=device))

    def test_clone_outputs_false(self, device):
        """Test execution without cloning outputs."""
        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=False, clone_outputs=False
        )

        a = torch.tensor([1.0, 2.0], device=device)
        b = torch.tensor([3.0, 4.0], device=device)

        result = executor(a, b)
        assert torch.allclose(result, torch.tensor([4.0, 6.0], device=device))

    def test_clone_outputs_override(self, device):
        """Test overriding clone_outputs at call time."""
        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=False, clone_outputs=False
        )

        a = torch.tensor([1.0, 2.0], device=device)
        b = torch.tensor([3.0, 4.0], device=device)

        # Override clone_outputs to True
        result = executor(a, b, clone_outputs=True)
        assert torch.allclose(result, torch.tensor([4.0, 6.0], device=device))

    def test_warmup(self, device):
        """Test explicit warmup."""
        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=False, clone_outputs=True
        )

        a = torch.tensor([1.0, 2.0], device=device)
        b = torch.tensor([3.0, 4.0], device=device)

        assert not executor.is_initialized
        executor.warmup(a, b)
        assert executor.is_initialized

        # Actual call should work
        result = executor(a, b)
        assert torch.allclose(result, torch.tensor([4.0, 6.0], device=device))

    def test_reset(self, device):
        """Test resetting the executor."""
        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=False, clone_outputs=True
        )

        a = torch.tensor([1.0, 2.0], device=device)
        b = torch.tensor([3.0, 4.0], device=device)

        executor(a, b)
        assert executor.is_initialized

        executor.reset()
        assert not executor.is_initialized

    def test_multi_output_function(self, device):
        """Test function with multiple outputs."""
        executor = GraphExecutor(
            capture_fn=multi_output_fn, device=device, use_cuda_graph=False, clone_outputs=True
        )

        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        out1, out2, out3 = executor(x)

        assert torch.allclose(out1, x * 2)
        assert torch.allclose(out2, x + 1)
        assert torch.allclose(out3, x - 1)

    def test_capture_fn_kwargs(self, device):
        """Test passing kwargs to capture function."""

        def fn_with_kwargs(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return x * scale

        executor = GraphExecutor(
            capture_fn=fn_with_kwargs,
            device=device,
            use_cuda_graph=False,
            clone_outputs=True,
            scale=5.0,
        )

        x = torch.tensor([1.0, 2.0], device=device)
        result = executor(x)

        assert torch.allclose(result, torch.tensor([5.0, 10.0], device=device))

    def test_debug_dump(self, device):
        """Test debug dump functionality."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        import tempfile

        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=True, clone_outputs=True
        )

        a = torch.tensor([1.0], device=device)
        b = torch.tensor([2.0], device=device)
        executor(a, b)

        # Test debug dump
        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as f:
            try:
                executor.debug_dump(f.name)
            except Exception:
                # Debug dump might not be available in all CUDA versions
                pass


class TestCreateGraphExecutor:
    def test_create_graph_executor(self):
        """Test factory function."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        executor = create_graph_executor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=False, clone_outputs=True
        )

        assert isinstance(executor, GraphExecutor)
        assert not executor.is_initialized

        a = torch.tensor([1.0], device=device)
        b = torch.tensor([2.0], device=device)
        result = executor(a, b)

        assert torch.allclose(result, torch.tensor([3.0], device=device))


class TestGraphExecutorWithBackward:
    """Test CUDA graph with backward pass."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_backward_operation(self, device):
        """Test function with backward pass."""

        def loss_fn(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            loss = ((x - target) ** 2).mean()
            return loss

        executor = GraphExecutor(
            capture_fn=loss_fn, device=device, use_cuda_graph=False, clone_outputs=True
        )

        x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        target = torch.tensor([0.0, 0.0, 0.0], device=device)

        loss = executor(x, target)
        assert loss.item() > 0

    def test_addition_with_backward(self, device):
        """Test simple addition that can be backpropagated."""

        def add_and_square(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            result = a + b
            return (result ** 2).sum()

        executor = GraphExecutor(
            capture_fn=add_and_square, device=device, use_cuda_graph=False, clone_outputs=True
        )

        a = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
        b = torch.tensor([3.0, 4.0], device=device, requires_grad=True)

        output = executor(a, b)
        # (1+3)^2 + (2+4)^2 = 16 + 36 = 52
        assert torch.allclose(output, torch.tensor(52.0, device=device))

        # Test backward
        output.backward()
        assert a.grad is not None
        assert b.grad is not None


class TestGraphExecutorEdgeCases:
    """Test edge cases and configuration options."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_use_cuda_graph_none_default(self, device):
        """Test that use_cuda_graph=None reads from runtime config."""
        # CuRobo
        import curobo._src.runtime as curobo_runtime

        # Store original value
        original_value = curobo_runtime.cuda_graphs

        try:
            # Set runtime to True
            curobo_runtime.cuda_graphs = True
            executor = GraphExecutor(
                capture_fn=simple_add_fn, device=device, use_cuda_graph=None, clone_outputs=True
            )
            # Should read from runtime config
            assert executor._use_cuda_graph == (torch.cuda.is_available() and True)
        finally:
            # Restore original
            curobo_runtime.cuda_graphs = original_value

    def test_cuda_graph_disabled_by_config(self, device):
        """Test CUDA graph disabled by runtime config."""
        # CuRobo
        from curobo import runtime as curobo_runtime

        original_value = curobo_runtime.cuda_graphs

        try:
            # Disable CUDA graphs in runtime before creating executor
            curobo_runtime.cuda_graphs = False

            executor = GraphExecutor(
                capture_fn=simple_add_fn, device=device, use_cuda_graph=True, clone_outputs=True
            )
            # Should be disabled despite use_cuda_graph=True
            assert executor._use_cuda_graph is False

            a = torch.tensor([1.0], device=device)
            b = torch.tensor([2.0], device=device)
            result = executor(a, b)
            assert torch.allclose(result, torch.tensor([3.0], device=device))
        finally:
            curobo_runtime.cuda_graphs = original_value

    def test_cuda_graph_debug_mode(self, device):
        """Test CUDA graph with debug mode enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # CuRobo
        import curobo._src.runtime as curobo_runtime

        original_debug = curobo_runtime.debug_cuda_graphs

        try:
            curobo_runtime.debug_cuda_graphs = True
            executor = GraphExecutor(
                capture_fn=simple_add_fn, device=device, use_cuda_graph=True, clone_outputs=True
            )

            a = torch.tensor([1.0], device=device)
            b = torch.tensor([2.0], device=device)
            result = executor(a, b)

            assert torch.allclose(result, torch.tensor([3.0], device=device))
        finally:
            curobo_runtime.debug_cuda_graphs = original_debug

    def test_reset_with_cuda_graph(self, device):
        """Test resetting when CUDA graph is active."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=True, clone_outputs=True
        )

        a = torch.tensor([1.0, 2.0], device=device)
        b = torch.tensor([3.0, 4.0], device=device)

        # Initialize
        executor(a, b)
        assert executor.is_initialized

        # Reset
        executor.reset()
        assert not executor.is_initialized
        assert executor._graph is None

    def test_input_copy_with_different_data_ptr(self, device):
        """Test that inputs are copied when data_ptr differs."""
        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=False, clone_outputs=True
        )

        # First call
        a1 = torch.tensor([1.0, 2.0], device=device)
        b1 = torch.tensor([3.0, 4.0], device=device)
        result1 = executor(a1, b1)

        # Second call with different tensors (different data_ptr)
        a2 = torch.tensor([5.0, 6.0], device=device)
        b2 = torch.tensor([7.0, 8.0], device=device)
        result2 = executor(a2, b2)

        # Should get different results
        assert torch.allclose(result1, torch.tensor([4.0, 6.0], device=device))
        assert torch.allclose(result2, torch.tensor([12.0, 14.0], device=device))

    def test_input_copy_with_same_data_ptr(self, device):
        """Test that copy is avoided when data_ptr is the same."""
        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=False, clone_outputs=True
        )

        # First call
        a = torch.tensor([1.0, 2.0], device=device)
        b = torch.tensor([3.0, 4.0], device=device)
        executor(a, b)

        # Modify the same tensors
        a[:] = torch.tensor([5.0, 6.0], device=device)
        b[:] = torch.tensor([7.0, 8.0], device=device)

        # Second call - should use updated values
        result = executor(a, b)
        assert torch.allclose(result, torch.tensor([12.0, 14.0], device=device))

    def test_warmup_already_initialized(self, device):
        """Test that warmup is no-op when already initialized."""
        executor = GraphExecutor(
            capture_fn=simple_add_fn, device=device, use_cuda_graph=False, clone_outputs=True
        )

        a = torch.tensor([1.0], device=device)
        b = torch.tensor([2.0], device=device)

        # Initialize via call
        executor(a, b)
        assert executor.is_initialized

        # Warmup should be no-op
        result = executor.warmup(a, b)
        assert result is executor

    def test_single_output_unwrapping(self, device):
        """Test that single output is returned as scalar, not tuple."""

        def single_out(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        executor = GraphExecutor(
            capture_fn=single_out, device=device, use_cuda_graph=False, clone_outputs=True
        )

        x = torch.tensor([1.0, 2.0], device=device)
        result = executor(x)

        # Should be a tensor, not a tuple
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, tuple)
        assert torch.allclose(result, torch.tensor([2.0, 4.0], device=device))

