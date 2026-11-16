#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export fixed-target MPC weights alongside an existing native robot artifact."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import yaml


def export_mpc_weights(output_dir: Path) -> None:
    """Write weights and their exact source configuration; compile no kernels."""
    root = Path(__file__).resolve().parents[3]
    source = root / "curobo/content/configs/task/mpc/lbfgs_mpc.yml"
    source_bytes = source.read_bytes()
    cfg = yaml.safe_load(source_bytes)["rollout"]
    cost = cfg["cost_cfg"]["cspace_cfg"]
    collision = cfg["constraint_cfg"]
    world = collision["scene_collision_cfg"]
    if not world["use_sweep"] or not world["use_sweep_kernel"]:
        raise ValueError("Native MPC requires swept collision checking")
    if cost["retime_weights"] or not cost["retime_regularization_weights"]:
        raise ValueError("Native MPC requires unscaled bounds and dt-normalized regularization")
    values = (
        cost["weight"]
        + cost["activation_distance"]
        + cost["squared_l2_regularization_weight"]
        + [
            cost["cspace_target_weight"],
            cost["cspace_non_terminal_weight_factor"],
            world["weight"],
            collision["self_collision_cfg"]["weight"],
        ]
    )
    if len(values) != 19 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Invalid MPC cost weights")
    if not output_dir.is_dir():
        raise ValueError("Output must be an existing robot artifact directory")
    payload = struct.pack("<19f", *values)
    (output_dir / "mpc_cost_weights.f32").write_bytes(payload)
    metadata = {
        "schema_version": 2,
        "source": str(source.relative_to(root)),
        "source_yaml": source_bytes.decode("utf-8"),
        "weights": list(struct.unpack("<19f", payload)),
        "swept_collision": True,
        "retime_weights": False,
        "retime_regularization_weights": True,
        "layout": {
            "state_bound_weight": [0, 5],
            "state_activation_distance": [5, 5],
            "state_regularization_weight": [10, 5],
            "cspace_target_weight": [15, 1],
            "cspace_non_terminal_weight": [16, 1],
            "world_collision_weight": [17, 1],
            "self_collision_weight": [18, 1],
        },
    }
    (output_dir / "robot_mpc_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    export_mpc_weights(parser.parse_args().out)
