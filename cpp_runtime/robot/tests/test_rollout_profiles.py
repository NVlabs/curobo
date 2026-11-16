# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check exported forward-only RNEA dispatch against the shared runtime policy."""
from pathlib import Path
from typing import Any

import pytest

from cpp_runtime.robot.tools import export_robot_aot as exporter
from curobo._src.curobolib.backends.cuda_core_backend.dynamics_config import DynamicsLaunchCfg
from curobo._src.robot.dynamics.dynamics import _compute_runtime_threads_per_batch


@pytest.mark.parametrize("topology", [(1, 7, 1), (2, 12, 1), (3, 29, 2)])
@pytest.mark.parametrize("sm_count", [16, 32, 170])
@pytest.mark.parametrize("sm_capacity", [100 * 1024, 164 * 1024, 228 * 1024])
def test_forward_profiles_match_python_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    topology: tuple[int, int, int],
    sm_count: int,
    sm_capacity: int,
) -> None:
    """Dispatch preserves the Python width and packing at batch-policy boundaries."""
    def compile_stub(
        output_dir: Path, arch: str, name: str, config: Any,
        kernel_types: list[str], symbols: list[str],
    ) -> tuple[str, dict[str, str]]:
        return f"{name}.{arch}.cubin", {symbol: symbol for symbol in symbols}

    monkeypatch.setattr(exporter, "_compile_cuda_specializations", compile_stub)
    width, links, levels = topology
    metadata = exporter.build_rollout_forward_profiles(
        tmp_path, "sm_120", links, links, width, levels, True, True, sm_count, sm_capacity
    )
    profiles = [metadata[f"profile_{i}"] for i in range(metadata["count"])]
    assert profiles[-1]["max_states"] == 2**31 - 1
    sizes = {32, 324, 972, 4000, 6912, 16384}
    for boundary in (512, 1024, 4096, 8192):
        scaled = boundary * sm_count // 170
        sizes.update((max(32, scaled - 1), max(32, scaled), max(32, scaled + 1)))
    sizes.update((4095, 4096, 4097))
    for states in sizes:
        selected = next(profile for profile in profiles if states <= profile["max_states"])
        expected_width = _compute_runtime_threads_per_batch(
            width, links, levels, states, sm_count
        )
        config = DynamicsLaunchCfg.calculate_forward_config(
            states, links, expected_width, policy_sm_count=sm_count,
            sm_shared_mem_capacity=sm_capacity,
        )
        assert selected["threads_per_batch"] == expected_width
        assert selected["kernel"]["block_dim"] == config.block[0]
        assert selected["kernel"]["shared_memory"] == config.shmem_size
