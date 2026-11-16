# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export existing unsigned mesh-distance calibration kernels for a target SM."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

import warp as wp


def export_calibration(output: Path, arch: int) -> None:
    """Compile the original query/reduction kernels without changing their math."""
    if arch < 75:
        raise ValueError("Calibration requires an SM75 or newer target")
    source = Path(__file__).resolve().parents[3] / "curobo/_src/perception/pose_estimation/wp_mesh_sdf_alignment.py"
    spec = importlib.util.spec_from_file_location("curobo_calibration_mesh_distance", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    query = module.mesh_surface_distance_query_kernel
    reduce = module.jacobian_reduce_kernel
    query.module.options["block_dim"] = 256
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="curobo_calibration_") as temporary:
        wp.compile_aot_module(query.module, arch=arch, module_dir=temporary,
                              use_ptx=False, strip_hash=True)
        compiled = Path(temporary)
        metadata = json.loads(next(compiled.glob("*.meta")).read_text())
        shutil.copyfile(next(compiled.glob("*.cubin")), output / f"pose_calibration.sm_{arch}.cubin")
        lines = ["curobo-pose-calibration 1", wp.__version__, str(arch)]
        for kernel in (query, reduce):
            symbol = f"{kernel.get_mangled_name()}_cuda_kernel_forward"
            lines.append(f"{symbol} {int(metadata.get(symbol + '_smem_bytes', 0))}")
        (output / "pose_calibration.meta").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Export a fixed target architecture; the export host may use another GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--arch", type=int, help="CUDA SM number, for example 89 or 110")
    target.add_argument("--device-json", type=Path, help="Existing export_device.py probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arch = args.arch
    if args.device_json:
        record = json.loads(args.device_json.read_text())
        if record.get("schema_version") != 1:
            raise ValueError("Unsupported target-device schema")
        arch = int(record["arch"].removeprefix("sm_"))
    export_calibration(args.output, arch)


if __name__ == "__main__":
    main()
