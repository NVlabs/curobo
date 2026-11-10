# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for RNEA launch configuration and shared-memory-per-SM discovery."""

# Standard Library
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend import dynamics
from curobo._src.curobolib.backends.cuda_core_backend.dynamics_config import DynamicsLaunchCfg

#: Shared memory per SM for architectures cuRobo runs on: 100 KB (compute
#: capability 8.6 / 12.x), 164 KB (8.0), 228 KB (9.0 / 10.x / 11.0).
SM_CAPACITIES = [100 * 1024, 164 * 1024, 228 * 1024]

#: (num_links, batch_size, threads_per_batch) spanning a 7-DoF arm, a 29-DoF
#: humanoid, and tree-parallel launches.
ROBOT_CASES = [(9, 512, 1), (9, 512, 4), (21, 512, 4), (31, 512, 1), (31, 512, 4), (60, 64, 8)]


class TestTensorDeviceCapacity:
    """Pass the tensor device's cached properties into both RNEA launch paths."""

    @pytest.mark.parametrize("direction,tensor_count", [("forward", 16), ("backward", 18)])
    @pytest.mark.parametrize("capacity", [None, 0, 228 * 1024])
    def test_uses_tensor_properties(
        self,
        monkeypatch: pytest.MonkeyPatch,
        direction: str,
        tensor_count: int,
        capacity: int | None,
    ) -> None:
        """Use the tensor device's capacity, falling back when the property is unavailable."""
        properties = SimpleNamespace(multi_processor_count=80)
        if capacity is not None:
            properties.shared_memory_per_multiprocessor = capacity
        get_properties = Mock(return_value=properties)
        current_device = Mock(return_value=0)
        monkeypatch.setattr(torch.cuda, "get_device_properties", get_properties)
        monkeypatch.setattr(torch.cuda, "current_device", current_device)
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(dynamics, "get_runtime", Mock())
        kernel_launch = Mock()
        monkeypatch.setattr(dynamics, "launch_kernel", kernel_launch)
        calculate_name = f"calculate_{direction}_config"
        calculate = Mock(wraps=getattr(DynamicsLaunchCfg, calculate_name))
        monkeypatch.setattr(DynamicsLaunchCfg, calculate_name, calculate)

        tensor = Mock(device=torch.device("cuda:1"))
        tensor.data_ptr.return_value = 0
        launch = getattr(dynamics, f"launch_rnea_{direction}")
        # Each required tensor argument can use the same placeholder; no CUDA kernel runs.
        launch(
            *[tensor] * tensor_count,
            batch_size=512,
            num_links=7,
            num_dof=6,
            n_levels=7,
        )
        get_properties.assert_called_once_with(tensor.device)
        current_device.assert_not_called()
        expected = capacity or DynamicsLaunchCfg.DEFAULT_SM_SHARED_MEM_CAPACITY
        assert calculate.call_args.kwargs["sm_shared_mem_capacity"] == expected
        kernel_launch.assert_called_once()

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_configuration_without_cuda_queries(
        self, monkeypatch: pytest.MonkeyPatch, direction: str
    ) -> None:
        """Standalone launch sizing keeps a fallback without requiring a CUDA device."""
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", Mock(side_effect=AssertionError("unexpected query"))
        )
        calculate = getattr(DynamicsLaunchCfg, f"calculate_{direction}_config")
        default = calculate(512, 7, 32)
        explicit = calculate(
            512, 7, 32, sm_shared_mem_capacity=DynamicsLaunchCfg.DEFAULT_SM_SHARED_MEM_CAPACITY
        )
        assert default.grid == explicit.grid
        assert default.block == explicit.block
        assert default.shmem_size == explicit.shmem_size


class TestLaunchConfigValidity:
    """Launch configs stay within hardware limits at every SM capacity."""

    @pytest.mark.parametrize("capacity", SM_CAPACITIES)
    @pytest.mark.parametrize("num_links,batch_size,threads_per_batch", ROBOT_CASES)
    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_config_within_limits(
        self, capacity: int, num_links: int, batch_size: int, threads_per_batch: int, direction: str
    ) -> None:
        calculate = getattr(DynamicsLaunchCfg, f"calculate_{direction}_config")
        config = calculate(
            batch_size, num_links, threads_per_batch, sm_shared_mem_capacity=capacity
        )

        threads_per_block = config.block[0]
        batches_per_block = threads_per_block // threads_per_batch

        assert 0 < threads_per_block <= 1024
        assert threads_per_block % threads_per_batch == 0
        assert 0 < config.shmem_size <= DynamicsLaunchCfg.DEFAULT_MAX_SHARED_MEM
        assert config.grid[0] * batches_per_block >= batch_size


class TestWarpAlignBatches:
    """The occupancy heuristic responds to the SM shared-memory budget."""

    def test_capacity_changes_the_choice(self) -> None:
        """A wider budget truncates less, which can change the winning block."""
        kwargs = {
            "batches_per_block": 2,
            "threads_per_batch": 32,
            "smem_per_block_fn": lambda b: b * 26000,
        }
        narrow = DynamicsLaunchCfg._warp_align_batches(sm_shared_mem_capacity=100 * 1024, **kwargs)
        wide = DynamicsLaunchCfg._warp_align_batches(sm_shared_mem_capacity=228 * 1024, **kwargs)
        assert narrow == 1
        assert wide == 2

    @pytest.mark.parametrize("capacity", SM_CAPACITIES)
    def test_never_exceeds_requested_batches(self, capacity: int) -> None:
        """The heuristic only ever narrows the caller's batch count."""
        for batches in range(1, 65):
            aligned = DynamicsLaunchCfg._warp_align_batches(
                batches_per_block=batches,
                threads_per_batch=4,
                smem_per_block_fn=lambda b: b * 1024,
                sm_shared_mem_capacity=capacity,
            )
            assert 1 <= aligned <= batches


class TestCapacityAffectsRealRobots:
    """The queried capacity changes launch shapes for real robot sizes."""

    @pytest.mark.parametrize(
        "batch_size,num_links,threads_per_batch,direction",
        [
            (512, 7, 32, "forward"),
            (512, 21, 32, "forward"),
            (8192, 7, 16, "backward"),
        ],
    )
    def test_block_shape_differs_between_architectures(
        self, batch_size: int, num_links: int, threads_per_batch: int, direction: str
    ) -> None:
        calculate = getattr(DynamicsLaunchCfg, f"calculate_{direction}_config")

        narrow = calculate(
            batch_size, num_links, threads_per_batch, sm_shared_mem_capacity=100 * 1024
        )
        wide = calculate(
            batch_size, num_links, threads_per_batch, sm_shared_mem_capacity=228 * 1024
        )

        assert narrow.block[0] != wide.block[0]
