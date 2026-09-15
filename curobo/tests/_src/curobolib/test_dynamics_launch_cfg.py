# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for RNEA launch configuration and shared-memory-per-SM discovery."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend import dynamics_config, util
from curobo._src.curobolib.backends.cuda_core_backend.dynamics_config import DynamicsLaunchCfg

#: Shared memory per SM for architectures cuRobo runs on: 100 KB (compute
#: capability 8.6 / 12.x), 164 KB (8.0), 228 KB (9.0 / 10.x / 11.0).
SM_CAPACITIES = [100 * 1024, 164 * 1024, 228 * 1024]

#: (num_links, batch_size, threads_per_batch) spanning a 7-DoF arm, a 29-DoF
#: humanoid, and tree-parallel launches.
ROBOT_CASES = [(9, 512, 1), (9, 512, 4), (21, 512, 4), (31, 512, 1), (31, 512, 4), (60, 64, 8)]


@pytest.fixture
def sm_capacity(monkeypatch):
    """Override the shared-memory-per-SM that the launch heuristic sees."""

    def _set(capacity: int):
        monkeypatch.setattr(dynamics_config, "get_sm_shared_memory_capacity", lambda *_: capacity)

    return _set


class TestSharedMemoryCapacityQuery:
    """Shared memory per SM is read from the device, with a safe fallback."""

    def test_falls_back_without_cuda(self, monkeypatch):
        """Hosts without a CUDA device keep the historical assumption."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert util.get_sm_shared_memory_capacity() == util.FALLBACK_SM_SHARED_MEM_CAPACITY

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_queries_real_device(self):
        """The queried value is a plausible shared-memory-per-SM figure."""
        capacity = util.get_sm_shared_memory_capacity()
        assert 48 * 1024 <= capacity <= 256 * 1024

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_agrees_with_torch_device_properties(self):
        """The CUDA attribute matches what torch reports for the same device."""
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        expected = getattr(props, "shared_memory_per_multiprocessor", None)
        if expected is None:
            pytest.skip("torch build does not expose shared_memory_per_multiprocessor")
        assert util.get_sm_shared_memory_capacity() == expected

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_query_is_cached(self):
        """Repeated lookups do not re-enter the driver on the launch hot path."""
        device_id = torch.cuda.current_device()
        util._query_sm_shared_memory_capacity(device_id)
        hits_before = util._query_sm_shared_memory_capacity.cache_info().hits
        util._query_sm_shared_memory_capacity(device_id)
        assert util._query_sm_shared_memory_capacity.cache_info().hits == hits_before + 1


class TestLaunchConfigValidity:
    """Launch configs stay within hardware limits at every SM capacity."""

    @pytest.mark.parametrize("capacity", SM_CAPACITIES)
    @pytest.mark.parametrize("num_links,batch_size,threads_per_batch", ROBOT_CASES)
    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_config_within_limits(
        self, sm_capacity, capacity, num_links, batch_size, threads_per_batch, direction
    ):
        sm_capacity(capacity)
        calculate = getattr(DynamicsLaunchCfg, f"calculate_{direction}_config")
        config = calculate(batch_size, num_links, threads_per_batch)

        threads_per_block = config.block[0]
        batches_per_block = threads_per_block // threads_per_batch

        assert 0 < threads_per_block <= 1024
        assert threads_per_block % threads_per_batch == 0
        assert 0 < config.shmem_size <= DynamicsLaunchCfg.DEFAULT_MAX_SHARED_MEM
        assert config.grid[0] * batches_per_block >= batch_size


class TestWarpAlignBatches:
    """The occupancy heuristic responds to the SM shared-memory budget."""

    def test_capacity_changes_the_choice(self):
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
    def test_never_exceeds_requested_batches(self, capacity):
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
        "num_links,threads_per_batch,direction", [(9, 1, "forward"), (21, 4, "backward")]
    )
    def test_block_shape_differs_between_architectures(
        self, sm_capacity, num_links, threads_per_batch, direction
    ):
        calculate = getattr(DynamicsLaunchCfg, f"calculate_{direction}_config")

        sm_capacity(100 * 1024)
        narrow = calculate(512, num_links, threads_per_batch)
        sm_capacity(228 * 1024)
        wide = calculate(512, num_links, threads_per_batch)

        assert narrow.block[0] != wide.block[0]
