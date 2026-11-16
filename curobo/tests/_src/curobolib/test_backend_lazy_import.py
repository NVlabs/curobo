# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for CUDA-independent configuration imports."""

# Standard Library
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def test_configuration_imports_do_not_load_cuda_package() -> None:
    """Keep documentation and configuration imports independent of CUDA packages."""
    import_script = dedent(
        """
        import sys

        class CudaPackageBlocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cuda" or fullname.startswith("cuda."):
                    raise ModuleNotFoundError(
                        f"blocked CUDA package import: {fullname}", name=fullname
                    )
                return None

        sys.meta_path.insert(0, CudaPackageBlocker())

        import curobo
        import curobo.runtime
        from curobo.kinematics import KinematicsCfg
        from curobo.motion_planner import MotionPlannerCfg
        from curobo._src.curobolib.backends import kinematics

        assert kinematics._module is None
        try:
            kinematics.launch_kinematics_forward
        except ImportError as error:
            assert "requires cuda.core" in str(error)
        else:
            raise AssertionError("kernel access should require cuda.core")
        assert not any(
            name == "cuda" or name.startswith("cuda.") for name in sys.modules
        )
        """
    )
    repository_root = Path(__file__).parents[4]
    subprocess.run(
        [sys.executable, "-c", import_script],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
