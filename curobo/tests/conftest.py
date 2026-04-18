# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared pytest fixtures and configuration for cuRobo tests."""

# Standard Library
import random

import numpy as np
import pytest

# Third Party
import torch

# CuRobo
import curobo._src.runtime as runtime

# Runtime configuration for tests
runtime.torch_compile = False
runtime.cuda_graph_reset = False


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--full-params",
        action="store_true",
        default=False,
        help="Use full parameter ranges for parametric tests (e.g., history=[1..31] instead of [1,10,31])",
    )


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Add a hook to skip test_robot_world_model.py
def pytest_ignore_collect(collection_path, config):
    """Skip collection of specific test files."""
    if collection_path.name == "test_robot_world_model.py":
        return True
    return False


# Common fixtures
@pytest.fixture(scope="session")
def cuda_device_cfg():
    """Provide CUDA tensor device configuration.

    Returns:
        DeviceCfg: CUDA tensor device configuration for testing.
    """
    # CuRobo
    from curobo._src.types.device_cfg import DeviceCfg

    return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)


@pytest.fixture(scope="session")
def cpu_device_cfg():
    """Provide CPU tensor device configuration.

    Returns:
        DeviceCfg: CPU tensor device configuration for testing.
    """
    # CuRobo
    from curobo._src.types.device_cfg import DeviceCfg

    return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)


@pytest.fixture(scope="session", params=["cpu", "cuda:0"])
def cpu_cuda_device_cfg(request):
    """Provide tensor device configuration for both CPU and CUDA.

    This fixture is parametrized to run tests on both CPU and CUDA devices.

    Returns:
        DeviceCfg: Tensor device configuration for testing.
    """
    # CuRobo
    from curobo._src.types.device_cfg import DeviceCfg

    return DeviceCfg(device=torch.device(request.param), dtype=torch.float32)

