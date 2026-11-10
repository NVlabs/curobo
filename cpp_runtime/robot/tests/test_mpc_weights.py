# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native MPC cost weights come from the existing MPC configuration."""

import json
import struct
from pathlib import Path

import pytest
import yaml

from cpp_runtime.robot.tools.export_mpc_weights import export_mpc_weights


def test_mpc_weights_match_source(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    export_mpc_weights(tmp_path)
    source = root / "curobo/content/configs/task/mpc/lbfgs_mpc.yml"
    cfg = yaml.safe_load(source.read_text())["rollout"]
    cost = cfg["cost_cfg"]["cspace_cfg"]
    constraints = cfg["constraint_cfg"]
    expected = (
        cost["weight"]
        + cost["activation_distance"]
        + cost["squared_l2_regularization_weight"]
        + [
            cost["cspace_target_weight"],
            cost["cspace_non_terminal_weight_factor"],
            constraints["scene_collision_cfg"]["weight"],
            constraints["self_collision_cfg"]["weight"],
        ]
    )
    actual = struct.unpack("<19f", (tmp_path / "mpc_cost_weights.f32").read_bytes())
    assert actual == pytest.approx(expected)
    metadata = json.loads((tmp_path / "robot_mpc_manifest.json").read_text())
    assert metadata["swept_collision"]
    assert metadata["retime_weights"] is False
    assert metadata["retime_regularization_weights"] is True
    assert metadata["schema_version"] == 2
    assert metadata["source_yaml"] == source.read_bytes().decode("utf-8")
    assert metadata["weights"] == list(actual)


@pytest.mark.parametrize("setting", ["sweep", "retiming"])
def test_export_rejects_incompatible_cost_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load((root / "curobo/content/configs/task/mpc/lbfgs_mpc.yml").read_text())
    if setting == "sweep":
        cfg["rollout"]["constraint_cfg"]["scene_collision_cfg"]["use_sweep"] = False
    else:
        cfg["rollout"]["cost_cfg"]["cspace_cfg"]["retime_weights"] = True
    monkeypatch.setattr(yaml, "safe_load", lambda _: cfg)
    with pytest.raises(ValueError, match="Native MPC requires"):
        export_mpc_weights(tmp_path)
    assert not list(tmp_path.iterdir())
