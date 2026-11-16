# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AOT dispatch parity without querying the build host's CUDA device."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cpp_runtime.robot.tools import export_robot_aot as exporter
from cpp_runtime.robot.tools.export_device import ExportDevice


@pytest.mark.parametrize(
    "history,dimension,enabled,expected,expected_shared",
    [
        (1, 1, True, True, False),
        (15, 128, True, True, False),
        (16, 128, True, True, True),
        (17, 128, True, True, True),
        (18, 128, True, True, False),
        (24, 112, True, True, False),
        (25, 112, True, True, True),
        (26, 112, True, True, True),
        (28, 112, True, True, True),
        (29, 112, True, True, False),
        (30, 128, True, True, False),
        (18, 129, True, True, False),
        (19, 129, True, True, True),
        (20, 256, True, True, True),
        (21, 256, True, True, False),
        (29, 256, True, True, False),
        (30, 256, True, True, True),
        (31, 112, True, False, None),
        (30, 257, True, False, None),
        (26, 112, False, False, None),
    ],
)
@pytest.mark.parametrize("arch", ["sm_87", "sm_110", "sm_120"])
def test_lbfgs_aot_selection(
    history: int,
    dimension: int,
    enabled: bool,
    expected: bool,
    expected_shared: bool | None,
    arch: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the same shape/storage selection on the target without dispatching compact."""
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", Mock(side_effect=AssertionError("host device"))
    )

    def compile_kernels(*args: object) -> tuple[str, dict[str, str]]:
        """Return symbols without invoking a device compiler."""
        return "test.cubin", {name: name for name in args[-1]}

    monkeypatch.setattr(exporter, "_compile_cuda_specializations", compile_kernels)
    config = SimpleNamespace(
        history=history,
        line_search_scale=[0.01, 0.5, 1],
        use_geometric_scaling=True,
        use_compact_operator=enabled,
    )
    result = exporter.compile_lbfgs_kernels(
        tmp_path, arch, "test", dimension, 4, config, ExportDevice("target", arch, 16, 101376)
    )["lbfgs"]
    assert (result["source"] == "curobo_lbfgs_vector_free_kernel") is expected
    if expected:
        assert result["block_dim"] == 128 and result["shared_memory"] == 0
        assert "kernel_lbfgs_step_vector_free" in result["symbol"]
        assert result["symbol"].endswith(f",{str(expected_shared).lower()}>")
    else:
        assert "kernel_lbfgs_step_circular" in result["symbol"]


def test_vector_free_export_does_not_query_unused_two_loop_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compile a bounded target specialization without build-host CUDA properties."""
    from curobo._src.curobolib.backends.cuda_core_backend.optimization_config import LBFGSLaunchCfg

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", Mock(side_effect=AssertionError("host device"))
    )
    monkeypatch.setattr(
        LBFGSLaunchCfg, "calculate_config", Mock(side_effect=AssertionError("unused two-loop"))
    )
    monkeypatch.setattr(
        exporter,
        "_compile_cuda_specializations",
        lambda *args: ("test.cubin", {name: name for name in args[-1]}),
    )
    config = SimpleNamespace(
        history=26,
        line_search_scale=[0.01, 0.5, 1],
        use_geometric_scaling=True,
        use_compact_operator=True,
    )
    result = exporter.compile_lbfgs_kernels(tmp_path, "sm_87", "test", 112, 4, config)
    assert result["lbfgs"]["source"] == "curobo_lbfgs_vector_free_kernel"
