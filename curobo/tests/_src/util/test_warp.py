# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
from pathlib import Path
from unittest.mock import MagicMock

# Third Party
import numpy as np
import pytest
import warp as wp

# CuRobo
from curobo._src.util.warp import warp_constant_suffix, warp_func, warp_kernel


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_constant_suffix_preserves_kernel_specializations(device: str, tmp_path: Path) -> None:
    """Long configuration names compile and keep separate constants on each backend."""
    wp.init()
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is not available")

    def make_kernel(value: float) -> wp.Kernel:
        suffix = warp_constant_suffix(value, tuple(range(64)), (0.1, -1.0, 1.0e-8))
        constant = wp.constant(value)

        @warp_func(f"constant_suffix_function_{suffix}")
        def specialized_value() -> float:
            return constant

        @warp_kernel(f"constant_suffix_specialization_{suffix}", enable_backward=False)
        def specialized_kernel(output: wp.array(dtype=wp.float32)) -> None:
            output[wp.tid()] = specialized_value()

        return specialized_kernel

    output = wp.zeros(4, dtype=wp.float32, device=device)
    first = make_kernel(2.0)
    second = make_kernel(3.0)
    for kernel, value in [(first, 2.0), (second, 3.0), (first, 2.0)]:
        wp.launch(kernel, dim=4, inputs=[output], device=device)
        np.testing.assert_array_equal(output.numpy(), np.full(4, value, dtype=np.float32))
    if device.startswith("cuda"):
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(first, dim=4, inputs=[output], device=device)
        wp.launch(second, dim=4, inputs=[output], device=device)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(output.numpy(), np.full(4, 2.0, dtype=np.float32))
        wp.compile_aot_module(
            first.module,
            device="cuda",
            arch=wp.get_device(device).arch,
            module_dir=str(tmp_path),
            use_ptx=False,
            strip_hash=True,
        )
        assert list(tmp_path.glob("*.cubin"))


class TestWarp:
    def test_init_warp(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import init_warp

            result = init_warp(quiet=True)
            assert result is True
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_sdf_struct(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import warp_support_sdf_struct

            result = warp_support_sdf_struct()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_kernel_key(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import warp_support_kernel_key

            result = warp_support_kernel_key()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_bvh_constructor_type(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import warp_support_bvh_constructor_type

            result = warp_support_bvh_constructor_type()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_sdf_struct_with_old_version(self):
        try:
            # CuRobo
            from curobo._src.util.warp import warp_support_sdf_struct

            mock_wp = MagicMock()
            mock_wp.config.version = "0.9.0"
            result = warp_support_sdf_struct(mock_wp)
            assert result is False
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_kernel_key_with_old_version(self):
        try:
            # CuRobo
            from curobo._src.util.warp import warp_support_kernel_key

            mock_wp = MagicMock()
            mock_wp.config.version = "1.0.0"
            result = warp_support_kernel_key(mock_wp)
            assert result is False
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_bvh_constructor_type_with_old_version(self):
        try:
            # CuRobo
            from curobo._src.util.warp import warp_support_bvh_constructor_type

            mock_wp = MagicMock()
            mock_wp.config.version = "1.5.0"
            result = warp_support_bvh_constructor_type(mock_wp)
            assert result is False
        except ImportError:
            pytest.skip("Warp not installed")

