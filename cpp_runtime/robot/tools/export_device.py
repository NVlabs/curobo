#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe a CUDA export target using only Python's standard library and the driver."""

from __future__ import annotations

import argparse
import ctypes
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExportDevice:
    """GPU properties used to compile portable C++ runtime artifacts."""

    name: str
    """Reported GPU name."""
    arch: str
    """CUDA cubin architecture, for example sm_110."""
    sm_count: int
    """Number of streaming multiprocessors visible to the CUDA device."""
    max_shared_memory_per_block: int
    """Opt-in per-block shared-memory limit, in bytes."""

    def __post_init__(self) -> None:
        """Reject malformed architectures and unusable hardware limits."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("device name must be a nonempty string")
        if (
            not isinstance(self.arch, str)
            or re.fullmatch(r"sm_[1-9][0-9]{1,2}", self.arch) is None
        ):
            raise ValueError("device arch must use the form sm_110")
        for field in ("sm_count", "max_shared_memory_per_block"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"device {field} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON description recorded with exported artifacts."""
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_file(cls, path: Path) -> ExportDevice:
        """Load a device description produced by this tool."""
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("unsupported export device schema; expected schema_version 1")
        fields = ("name", "arch", "sm_count", "max_shared_memory_per_block")
        if any(field not in data for field in fields):
            raise ValueError(f"export device description requires {', '.join(fields)}")
        return cls(**{field: data[field] for field in fields})

    @classmethod
    def current(cls, ordinal: int = 0) -> ExportDevice:
        """Query a visible CUDA device without importing cuRobo, PyTorch, or Warp."""
        driver = ctypes.CDLL("libcuda.so.1")

        def check(status: int) -> None:
            if status != 0:
                raise RuntimeError(f"CUDA driver query failed with error {status}")

        check(driver.cuInit(0))
        device = ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(device), ordinal))
        name = ctypes.create_string_buffer(256)
        check(driver.cuDeviceGetName(name, len(name), device))

        def attribute(identifier: int) -> int:
            value = ctypes.c_int()
            check(driver.cuDeviceGetAttribute(ctypes.byref(value), identifier, device))
            return value.value

        # CU_DEVICE_ATTRIBUTE values from cuda.h. No CUDA Python package is needed.
        major, minor = attribute(75), attribute(76)
        return cls(name.value.decode(), f"sm_{major}{minor}", attribute(16), attribute(97))


def main() -> None:
    """Write a target description on the deployment machine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="visible CUDA device ordinal")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be non-negative")
    try:
        device = ExportDevice.current(args.device)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    args.out.write_text(json.dumps(device.to_dict(), indent=2) + "\n")


if __name__ == "__main__":
    main()
