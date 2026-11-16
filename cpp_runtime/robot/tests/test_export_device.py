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
    device = ExportDevice("NVIDIA Thor", "sm_110", 20, 101376, 228 * 1024)
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
        ("shared_memory_per_multiprocessor", 0),
        ("shared_memory_per_multiprocessor", True),
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


def test_older_device_description_retains_shared_memory_fallback(tmp_path: Path) -> None:
    """Existing target descriptions remain readable with their historical SM budget."""
    description = ExportDevice("target", "sm_110", 20, 101376).to_dict()
    del description["shared_memory_per_multiprocessor"]
    path = tmp_path / "older.json"
    path.write_text(json.dumps(description))
    assert ExportDevice.from_file(path).shared_memory_per_multiprocessor == 100 * 1024


def test_current_device_records_sm_shared_memory() -> None:
    """The standalone driver probe records the same capacity as PyTorch."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    ordinal = torch.cuda.current_device()
    expected = torch.cuda.get_device_properties(ordinal).shared_memory_per_multiprocessor
    assert ExportDevice.current(ordinal).shared_memory_per_multiprocessor == expected


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


@pytest.mark.parametrize("target_capacity", [None, 100 * 1024, 228 * 1024])
def test_rnea_export_uses_device_capacity(
    target_capacity: int | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the model device for native export and the target description for cross-export."""
    import torch

    from cpp_runtime.robot.tools import export_robot_aot as exporter
    from curobo._src.curobolib.backends.cuda_core_backend.dynamics_config import DynamicsLaunchCfg
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo._src.types.device_cfg import DeviceCfg

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for robot preparation")
    config = KinematicsCfg.from_robot_yaml_file(
        "franka.yml", device_cfg=DeviceCfg()
    )
    properties = Mock(return_value=SimpleNamespace(
        multi_processor_count=170, shared_memory_per_multiprocessor=164 * 1024
    ))
    target = None
    if target_capacity is not None:
        properties.side_effect = AssertionError("cross-export must not query host properties")
        target = ExportDevice("target", "sm_110", 170, 101376, target_capacity)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)

    def compile_kernels(*args: object) -> tuple[str, dict[str, str]]:
        """Skip compilation while retaining the complete profile-generation path."""
        return "test.cubin", {name: name for name in args[-1]}

    monkeypatch.setattr(exporter, "_compile_cuda_specializations", compile_kernels)
    forward = Mock(wraps=DynamicsLaunchCfg.calculate_forward_config)
    backward = Mock(wraps=DynamicsLaunchCfg.calculate_backward_config)
    monkeypatch.setattr(DynamicsLaunchCfg, "calculate_forward_config", forward)
    monkeypatch.setattr(DynamicsLaunchCfg, "calculate_backward_config", backward)
    _, profiles, metadata = exporter.build_rollout_artifact(
        tmp_path, "sm_110" if target else "sm_120", config.kinematics_config,
        {"ik": 512, "trajopt": 8192}, target,
    )
    assert set(profiles) == {"ik", "trajopt"}
    assert metadata["forward_profiles"]["count"] > 0
    expected = target_capacity if target_capacity is not None else 164 * 1024
    for calculate in (forward, backward):
        assert calculate.call_count >= 2
        assert all(
            call.kwargs["sm_shared_mem_capacity"] == expected
            for call in calculate.call_args_list
        )
    if target is None:
        properties.assert_called_once_with(config.kinematics_config.fixed_transforms.device)
    else:
        properties.assert_not_called()
