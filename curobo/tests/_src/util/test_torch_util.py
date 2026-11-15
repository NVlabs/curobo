# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
from unittest.mock import patch

# Third Party
import pytest
import torch
from packaging import version

# CuRobo
from curobo._src.util.torch_util import (
    disable_torch_compile_global,
    empty_decorator,
    get_profiler_context,
    get_profiler_decorator,
    get_torch_compile_options,
    get_torch_jit_decorator,
    is_cuda_graph_available,
    is_cuda_graph_reset_available,
    is_torch_compile_available,
    set_torch_compile_global_options,
)


class TestCudaGraph:
    def test_is_cuda_graph_available(self):
        result = is_cuda_graph_available()
        assert isinstance(result, bool)

    @patch("torch.__version__", "1.9.0")
    def test_is_cuda_graph_available_old_pytorch(self):
        result = is_cuda_graph_available()
        assert result is False

    def test_is_cuda_graph_reset_available(self):
        result = is_cuda_graph_reset_available()
        assert isinstance(result, bool)

    def test_is_cuda_graph_reset_available_when_disabled(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        original_value = curobo_runtime.cuda_graph_reset
        try:
            curobo_runtime.cuda_graph_reset = False
            result = is_cuda_graph_reset_available()
            assert result is False
        finally:
            curobo_runtime.cuda_graph_reset = original_value

    def test_is_cuda_graph_reset_available_when_enabled(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        original_value = curobo_runtime.cuda_graph_reset
        try:
            curobo_runtime.cuda_graph_reset = True
            result = is_cuda_graph_reset_available()
            # Result depends on CUDA version (12.0+), so it should be a boolean
            assert isinstance(result, bool)
            # If CUDA >= 12.0, should be True, otherwise False
        finally:
            curobo_runtime.cuda_graph_reset = original_value

    @patch("torch.version.cuda", "12.0")
    def test_is_cuda_graph_reset_available_with_cuda_12(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        original_value = curobo_runtime.cuda_graph_reset
        try:
            curobo_runtime.cuda_graph_reset = True
            result = is_cuda_graph_reset_available()
            assert result is True
        finally:
            curobo_runtime.cuda_graph_reset = original_value

    @patch("torch.version.cuda", "11.8")
    def test_is_cuda_graph_reset_available_with_cuda_11(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        original_value = curobo_runtime.cuda_graph_reset
        try:
            curobo_runtime.cuda_graph_reset = True
            result = is_cuda_graph_reset_available()
            assert result is False
        finally:
            curobo_runtime.cuda_graph_reset = original_value


class TestTorchCompile:
    def test_is_torch_compile_available(self):
        result = is_torch_compile_available()
        assert isinstance(result, bool)

    def test_is_torch_compile_available_when_torch_jit_disabled(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        original_value = curobo_runtime.torch_jit
        try:
            curobo_runtime.torch_jit = False
            result = is_torch_compile_available()
            assert result is False
        finally:
            curobo_runtime.torch_jit = original_value

    def test_is_torch_compile_available_when_explicitly_disabled(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        original_jit = curobo_runtime.torch_jit
        original_disable = curobo_runtime.torch_compile
        try:
            curobo_runtime.torch_jit = True
            curobo_runtime.torch_compile = False
            result = is_torch_compile_available()
            assert result is False
        finally:
            curobo_runtime.torch_jit = original_jit
            curobo_runtime.torch_compile = original_disable

    @patch("torch.__version__", "1.13.0")
    def test_is_torch_compile_available_old_pytorch(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        original_jit = curobo_runtime.torch_jit
        original_disable = curobo_runtime.torch_compile
        try:
            curobo_runtime.torch_jit = True
            curobo_runtime.torch_compile = True
            result = is_torch_compile_available()
            assert result is False
        finally:
            curobo_runtime.torch_jit = original_jit
            curobo_runtime.torch_compile = original_disable

    def test_is_torch_compile_available_with_torch_2(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        # This test only runs if torch >= 2.0 is available
        if version.parse(torch.__version__) >= version.parse("2.0"):
            original_jit = curobo_runtime.torch_jit
            original_disable = curobo_runtime.torch_compile
            try:
                curobo_runtime.torch_jit = True
                curobo_runtime.torch_compile = True
                result = is_torch_compile_available()
                assert isinstance(result, bool)
            finally:
                curobo_runtime.torch_jit = original_jit
                curobo_runtime.torch_compile = original_disable

    def test_get_torch_compile_options(self):
        options = get_torch_compile_options()
        assert isinstance(options, dict)

    def test_get_torch_compile_options_when_available(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        # Only test if torch.compile is actually available
        if version.parse(torch.__version__) >= version.parse("2.0") and hasattr(
            torch, "compile"
        ):
            original_jit = curobo_runtime.torch_jit
            original_disable = curobo_runtime.torch_compile
            try:
                curobo_runtime.torch_jit = True
                curobo_runtime.torch_compile = True
                options = get_torch_compile_options()
                assert isinstance(options, dict)
                # When available, should have some options set
                if options:
                    # Check that options are valid types
                    for key, value in options.items():
                        assert isinstance(key, str)
            finally:
                curobo_runtime.torch_jit = original_jit
                curobo_runtime.torch_compile = original_disable

    def test_get_torch_compile_options_when_not_available(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        original_jit = curobo_runtime.torch_jit
        try:
            curobo_runtime.torch_jit = False
            options = get_torch_compile_options()
            assert isinstance(options, dict)
            assert len(options) == 0  # Should be empty when not available
        finally:
            curobo_runtime.torch_jit = original_jit

    def test_disable_torch_compile_global(self):
        result = disable_torch_compile_global()
        assert isinstance(result, bool)

    def test_disable_torch_compile_global_when_available(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        # Only test if torch.compile is actually available
        if version.parse(torch.__version__) >= version.parse("2.0") and hasattr(
            torch, "_dynamo"
        ):
            original_jit = curobo_runtime.torch_jit
            original_disable = curobo_runtime.torch_compile
            try:
                curobo_runtime.torch_jit = True
                curobo_runtime.torch_compile = True
                result = disable_torch_compile_global()
                assert isinstance(result, bool)
                # When available, should return True
                if result:
                    assert torch._dynamo.config.disable is True
            finally:
                curobo_runtime.torch_jit = original_jit
                curobo_runtime.torch_compile = original_disable

    def test_set_torch_compile_global_options(self):
        result = set_torch_compile_global_options()
        assert isinstance(result, bool)

    def test_set_torch_compile_global_options_when_available(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        # Only test if torch.compile is actually available
        if version.parse(torch.__version__) >= version.parse("2.0") and hasattr(
            torch, "_dynamo"
        ):
            original_jit = curobo_runtime.torch_jit
            original_disable = curobo_runtime.torch_compile
            try:
                curobo_runtime.torch_jit = True
                curobo_runtime.torch_compile = True
                result = set_torch_compile_global_options()
                assert isinstance(result, bool)
                # When available, should return True and set options
                if result:
                    # Third Party

                    assert torch._dynamo.config.suppress_errors is True
            finally:
                curobo_runtime.torch_jit = original_jit
                curobo_runtime.torch_compile = original_disable


class TestDecorators:
    def test_empty_decorator(self):
        def test_func():
            return 42

        decorated = empty_decorator(test_func)
        assert decorated() == 42
        assert decorated is test_func

    def test_get_torch_jit_decorator(self):
        decorator = get_torch_jit_decorator()
        assert callable(decorator)

    def test_get_torch_jit_decorator_force_jit(self):
        decorator = get_torch_jit_decorator(force_jit=True)
        assert callable(decorator)

    def test_get_torch_jit_decorator_only_valid_for_compile(self):
        decorator = get_torch_jit_decorator(only_valid_for_compile=True)
        assert callable(decorator)

    def test_get_profiler_decorator_disabled(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        original_value = curobo_runtime.profiler
        try:
            curobo_runtime.profiler = False
            decorator = get_profiler_decorator("test_function")
            # Should return empty_decorator when disabled
            assert decorator is empty_decorator
        finally:
            curobo_runtime.profiler = original_value

    def test_get_profiler_decorator_enabled(self):
        # CuRobo
        from curobo import runtime as curobo_runtime

        original_value = curobo_runtime.profiler
        try:
            curobo_runtime.profiler = True
            decorator = get_profiler_decorator("test_function")
            # Should return a profiler context manager
            assert callable(decorator)
            assert decorator is not empty_decorator
        finally:
            curobo_runtime.profiler = original_value


@pytest.mark.parametrize("enabled", [False, True])
def test_profiler_decorator_records_only_when_enabled(enabled: bool) -> None:
    """Decorators preserve results and record labels only when enabled at creation."""
    from curobo import runtime as curobo_runtime

    label = "test/profiler_decorator"

    with patch.object(curobo_runtime, "profiler", enabled):

        @get_profiler_decorator(label)
        def double(value: int) -> int:
            """Return twice the input."""
            return value * 2

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        assert double(3) == 6

    assert (label in {event.key for event in trace.key_averages()}) is enabled
    assert double.__name__ == "double"


def test_profiler_context_checks_flag_for_each_scope() -> None:
    """Context scopes honor flag changes without retaining enabled annotations."""
    from curobo import runtime as curobo_runtime

    labels = ["test/context_disabled", "test/context_enabled", "test/context_disabled_again"]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        for label, enabled in zip(labels, [False, True, False]):
            with patch.object(curobo_runtime, "profiler", enabled):
                with get_profiler_context(label):
                    torch.ones(1, dtype=torch.float32).add_(1)

    events = {event.key for event in trace.key_averages()}
    assert labels[0] not in events
    assert labels[1] in events
    assert labels[2] not in events


def test_disabled_profiler_helpers_do_not_call_torch() -> None:
    """Disabled helpers avoid invoking the PyTorch annotation machinery."""
    from curobo import runtime as curobo_runtime

    def identity(value: int) -> int:
        """Return the input."""
        return value

    with patch.object(curobo_runtime, "profiler", False):
        with patch("curobo._src.util.torch_util.profiler.record_function") as record:
            wrapped = get_profiler_decorator("test/disabled")(identity)
            assert wrapped is identity
            with get_profiler_context("test/disabled_context"):
                assert wrapped(7) == 7
            record.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_profiler_helpers_preserve_exceptions(enabled: bool) -> None:
    """Enabled and disabled scopes propagate application exceptions."""
    from curobo import runtime as curobo_runtime

    with patch.object(curobo_runtime, "profiler", enabled):

        @get_profiler_decorator("test/raising_decorator")
        def fail() -> None:
            """Raise an application error."""
            raise ValueError("application error")

        with pytest.raises(ValueError, match="application error"):
            fail()
        with pytest.raises(ValueError, match="application error"):
            with get_profiler_context("test/raising_context"):
                raise ValueError("application error")
