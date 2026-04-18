# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional, Tuple, Union

import torch

import curobo._src.runtime as curobo_runtime
from curobo import runtime as curobo_runtime
from curobo._src.util.logging import log_info, log_warn
import gc


class GraphExecutor:
    """Executes a function with or without CUDA graph acceleration.

    Graph is recorded lazily on first call, allowing executor to be created
    early (e.g., in __init__) before inputs are available.

    Fully encapsulated: manages all internal state, copying, and execution.
    """

    def __init__(
        self,
        capture_fn: Callable,
        device: torch.device,
        use_cuda_graph: Optional[bool] = None,
        clone_outputs: bool = True,
        **capture_fn_kwargs,
    ):
        """Initialize executor (graph recorded lazily on first call).

        Args:
            capture_fn: Function to execute.
            device: Device to run on.
            use_cuda_graph: Whether to use CUDA graphs. If None, reads from
                           curobo_runtime.cuda_graphs. Defaults to None.
            clone_outputs: If True, always return cloned outputs. Defaults to False.
            **capture_fn_kwargs: Additional kwargs for capture_fn during graph recording.
        """
        self._capture_fn = capture_fn
        self._device = device
        # Use config value if not explicitly specified
        if use_cuda_graph is None:
            use_cuda_graph = curobo_runtime.cuda_graphs
        if not curobo_runtime.cuda_graphs:
            use_cuda_graph = False
            log_warn("CUDA Graph is disabled by config")
        self._use_cuda_graph = use_cuda_graph
        self._clone_outputs = clone_outputs
        self._capture_fn_kwargs = capture_fn_kwargs

        # Lazily initialized on first call
        self._graph_input: Optional[Tuple[torch.Tensor, ...]] = None
        self._graph_output: Optional[Tuple[torch.Tensor, ...]] = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None

    def __call__(
        self,
        *inputs,
        clone_outputs: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Execute function with given inputs.

        On first call: Records graph (if enabled) and logs a warning.
        Subsequent calls: Fast path with graph replay or direct execution.

        Note: On first call with CUDA graphs, recording adds ~100-500ms overhead.
              A warning is logged. Subsequent calls are fast (graph replay).

        Args:
            *inputs: Input tensors to the function.
            clone_outputs: If True, return cloned outputs. If False, return internal
                          tensors (faster but caller must not modify). If None (default),
                          uses the value from constructor.

        Returns:
            Single tensor if function returns one output, tuple otherwise.
            Outputs are either clones or internal tensors based on clone_outputs flag.
        """
        # Lazy initialization on first call
        if self._graph_input is None:
            if self._use_cuda_graph:
                log_info(
                    f"Recording CUDA Graph for {self._capture_fn.__name__ if hasattr(self._capture_fn, '__name__') else 'function'}"
                )
            self._initialize(inputs)

        # Copy inputs to internal tensors (only if different memory location)
        for i, inp in enumerate(inputs):
            if hasattr(self._graph_input[i], "copy_"):
                if hasattr(self._graph_input[i], "data_ptr") and callable(
                    self._graph_input[i].data_ptr
                ):
                    if self._graph_input[i].data_ptr() != inp.data_ptr():
                        self._graph_input[i].copy_(inp)
                else:
                    self._graph_input[i].copy_(inp)

        # Execute
        if self._use_cuda_graph:
            self._graph.replay()
            output = self._graph_output
        else:
            output = self._execute_direct()

        # Return outputs (clone if requested)
        should_clone = clone_outputs if clone_outputs is not None else self._clone_outputs
        if should_clone:
            outputs = tuple(t.clone() if hasattr(t, "clone") else t for t in output)
            return outputs[0] if len(outputs) == 1 else outputs
        else:
            return output[0] if len(output) == 1 else output

    def warmup(self, *sample_inputs):
        """Explicitly record graph with sample inputs (optional).

        Useful for:
        - Benchmarking (exclude graph recording from timing)
        - Early error detection
        - Predictable first-call latency

        If already initialized, this is a no-op.

        Args:
            *sample_inputs: Sample input tensors for graph recording.

        Returns:
            Self for chaining.
        """
        if self._graph_input is None:
            log_info(
                f"Warming up GraphExecutor for {self._capture_fn.__name__ if hasattr(self._capture_fn, '__name__') else 'function'}"
            )
            self._initialize(sample_inputs)
        return self

    def _initialize(self, inputs: Tuple[torch.Tensor, ...]):
        """Initialize executor with input shapes."""
        if self._use_cuda_graph:
            self._initialize_cuda_graph(inputs)
        else:
            self._initialize_direct(inputs)

    def _initialize_cuda_graph(self, inputs: Tuple[torch.Tensor, ...]):
        """Record CUDA graph."""
        # Flush any pending graph destructions (e.g. from GC of previous
        # GraphExecutors) so that cuGraphExecDestroy does not collide with
        # the upcoming stream capture.

        gc.collect()
        torch.cuda.synchronize(self._device)

        # Clone inputs for graph capture
        self._graph_input = tuple(t.clone() if hasattr(t, "clone") else t for t in inputs)

        mem_pool = (
            torch.cuda.graph_pool_handle() if hasattr(torch.cuda, "graph_pool_handle") else None
        )
        stream = torch.cuda.Stream(device=self._device)
        stream.wait_stream(torch.cuda.current_stream(device=self._device))

        # Warmup runs
        with torch.cuda.stream(stream):
            for _ in range(3):
                graph_output = self._capture_fn(*self._graph_input, **self._capture_fn_kwargs)

        torch.cuda.current_stream(self._device).wait_stream(stream)

        # Record graph
        self._graph = torch.cuda.CUDAGraph()
        if curobo_runtime.debug_cuda_graphs:
            self._graph.enable_debug_mode()
            log_warn("CUDA Graph Debug Mode enabled")

        with torch.cuda.graph(self._graph, pool=mem_pool, stream=stream):
            self._graph_output = self._capture_fn(*self._graph_input, **self._capture_fn_kwargs)

        # Normalize to tuple
        if not isinstance(self._graph_output, tuple):
            self._graph_output = (self._graph_output,)

    def _initialize_direct(self, inputs: Tuple[torch.Tensor, ...]):
        """Initialize for direct (non-graph) execution."""
        self._graph_input = tuple(t.clone() if hasattr(t, "clone") else t for t in inputs)

        output = self._capture_fn(*self._graph_input, **self._capture_fn_kwargs)
        if not isinstance(output, tuple):
            output = (output,)
        self._graph_output = output

    def _execute_direct(self):
        """Execute function directly (without graph)."""
        output = self._capture_fn(*self._graph_input, **self._capture_fn_kwargs)
        if not isinstance(output, tuple):
            output = (output,)
        return output

    def reset(self):
        """Reset executor, clearing cached graph and tensors.

        Next call will re-initialize with new input shapes.
        """
        if self._graph is not None:
            self._graph.reset()
        self._graph = None
        self._graph_input = None
        self._graph_output = None

    def debug_dump(self, file_path: str):
        """Dump CUDA graph to file for debugging (CUDA graph mode only).

        Args:
            file_path: Path to dump the graph visualization.
        """
        if self._use_cuda_graph and self._graph is not None:
            self._graph.debug_dump(file_path)

    @property
    def is_initialized(self) -> bool:
        """Check if executor has been initialized."""
        return self._graph_input is not None


def create_graph_executor(
    capture_fn: Callable,
    device: torch.device,
    use_cuda_graph: Optional[bool] = None,
    clone_outputs: bool = False,
    **capture_fn_kwargs,
) -> GraphExecutor:
    """Create a GraphExecutor for a function.

    The executor will lazily initialize on first call.

    Args:
        capture_fn: Function to execute.
        device: Device to run on.
        use_cuda_graph: Whether to use CUDA graphs. If None, reads from
                       curobo_runtime.cuda_graphs. Defaults to None.
        clone_outputs: If True, always return cloned outputs (safer but slower).
                      Defaults to False.
        **capture_fn_kwargs: Additional kwargs for capture_fn during graph recording.

    Returns:
        GraphExecutor that can be called to execute the function.
    """
    return GraphExecutor(
        capture_fn=capture_fn,
        device=device,
        use_cuda_graph=use_cuda_graph,
        clone_outputs=clone_outputs,
        **capture_fn_kwargs,
    )
