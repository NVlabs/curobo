# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check device descriptions and cross-device export without loading foreign kernels."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cpp_runtime.robot.tools.export_device import ExportDevice


def test_device_description_round_trip(tmp_path: Path) -> None:
    """A saved target retains the limits used for kernel specialization."""
    device = ExportDevice("NVIDIA Thor", "sm_110", 20, 101376)
    path = tmp_path / "thor.json"
    path.write_text(json.dumps(device.to_dict()))
    assert ExportDevice.from_file(path) == device


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arch", "compute_110"),
        ("arch", "sm_110a"),
        ("sm_count", 0),
        ("sm_count", True),
        ("max_shared_memory_per_block", -1),
        ("name", ""),
    ],
)
def test_invalid_device_description(field: str, value: object, tmp_path: Path) -> None:
    """Reject invalid limits before generating any artifacts."""
    description = ExportDevice("NVIDIA Thor", "sm_110", 20, 101376).to_dict()
    description[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(description))
    with pytest.raises(ValueError):
        ExportDevice.from_file(path)


def test_cross_compile_does_not_load_foreign_cubin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NVRTC compilation for Thor must not load its cubin on an RTX GPU."""
    from cpp_runtime.robot.tools import export_robot_aot as exporter

    local_device = Mock(return_value=SimpleNamespace(arch=120, set_current=lambda: None))
    monkeypatch.setattr(exporter, "Device", local_device)
    load = Mock(side_effect=AssertionError("cannot load foreign cubin"))
    monkeypatch.setattr(exporter, "ObjectCode", SimpleNamespace(from_cubin=load))
    exporter.validate_native_cubin(tmp_path / "foreign.cubin", "sm_110", {"fk": "fk"})
    load.assert_not_called()


def test_native_compile_still_loads_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the existing load validation for artifacts targeting the local GPU."""
    from cpp_runtime.robot.tools import export_robot_aot as exporter

    monkeypatch.setattr(
        exporter, "Device", Mock(return_value=SimpleNamespace(arch=120, set_current=lambda: None))
    )
    cubin = Mock()
    load = Mock(return_value=cubin)
    monkeypatch.setattr(exporter, "ObjectCode", SimpleNamespace(from_cubin=load))
    exporter.validate_native_cubin(tmp_path / "native.cubin", "sm_120", {"fk": "_fk"})
    cubin.get_kernel.assert_called_once_with("fk")


def test_unsupported_warp_target_fails_before_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA 12 Warp build must explain why Thor cubins cannot be produced."""
    from cuda.bindings import nvrtc

    from cpp_runtime.robot.tools import export_robot_aot as exporter

    monkeypatch.setattr(nvrtc, "nvrtcGetSupportedArchs", lambda: (0, [110, 120]))
    monkeypatch.setattr(exporter.wp, "init", lambda: None)
    monkeypatch.setattr(exporter.wp, "get_cuda_supported_archs", lambda: [120])
    with pytest.raises(ValueError, match=r"Warp.*sm_110.*CUDA 13"):
        exporter.validate_export_target(ExportDevice("Thor", "sm_110", 20, 232448))


@pytest.mark.parametrize(("shared_limit", "uses_shared"), [(49152, False), (101376, True)])
def test_lbfgs_uses_target_shared_memory(
    shared_limit: int, uses_shared: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Specialization must follow the target limit rather than the export host's GPU."""
    import torch

    from cpp_runtime.robot.tools import export_robot_aot as exporter

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", Mock(side_effect=AssertionError("host properties"))
    )
    symbols = []

    def compile_kernels(*args: object) -> tuple[str, dict[str, str]]:
        """Record requested template symbols without invoking NVRTC."""
        symbols.extend(args[-1])
        return "test.cubin", {name: name for name in args[-1]}

    monkeypatch.setattr(exporter, "_compile_cuda_specializations", compile_kernels)
    optimizer = SimpleNamespace(
        history=10, line_search_scale=[0.1, 1.0], use_geometric_scaling=True
    )
    exporter.compile_lbfgs_kernels(
        tmp_path,
        "sm_110",
        "trajopt",
        700,
        4,
        optimizer,
        ExportDevice("target", "sm_110", 20, shared_limit),
    )
    assert f"<float, false, 10, {str(uses_shared).lower()}," in symbols[0]
