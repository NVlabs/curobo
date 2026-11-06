# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the hybrid RRT proposal stream."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.graph_planner.rrt.sampler_rrt import RRTSampler
from curobo._src.types.device_cfg import DeviceCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _sampler(seed: int = 0) -> RRTSampler:
    """Create a sampler with synthetic, unequal joint ranges."""
    joints = torch.arange(5, device="cuda", dtype=torch.float32)
    lows = -1.0 + 0.25 * joints
    highs = lows + 1.0 + 0.5 * joints
    return RRTSampler(
        lows, highs, 1, seed, DeviceCfg(), lambda q: q[:, 0] == q[:, 0], torch.ones_like(lows)
    )


def test_rrt_global_stream_continues_exploring_after_cached_samples() -> None:
    """A long four-lane search must not repeat its proposals every 20 steps."""
    sampler = _sampler()
    waves = [sampler.generate_action_samples(96).clone() for _ in range(100)]
    assert not torch.equal(waves[0], waves[20])
    samples = torch.cat(waves)
    assert torch.unique(samples, dim=0).shape[0] == samples.shape[0]
    buffer = sampler.action_sample_generator
    assert torch.all(samples >= buffer.low_bounds)
    assert torch.all(samples <= buffer.up_bounds)
    sampler.reset_seed()
    repeated = torch.cat([sampler.generate_action_samples(96).clone() for _ in range(100)])
    torch.testing.assert_close(repeated, samples, rtol=0, atol=0)


def test_rrt_global_stream_stream_is_independent_of_request_partition() -> None:
    """Crossing the cache boundary or using large batches must not drop/repeat rows."""
    sampler = _sampler()
    small = torch.cat([sampler.generate_action_samples(96).clone() for _ in range(40)])
    sampler.reset_seed()
    large = sampler.generate_action_samples(len(small)).clone()
    torch.testing.assert_close(large, small, rtol=1e-6, atol=1e-6)
    sampler.reset_seed()
    unit = sampler.generate_action_samples(len(small), bounded=False).clone()
    buffer = sampler.action_sample_generator
    torch.testing.assert_close(
        large, buffer.low_bounds + unit * buffer.range_b, rtol=1e-6, atol=1e-6
    )


def test_rrt_global_stream_reuses_storage_consumed_by_cuda_graph() -> None:
    """Captured growth inputs see fresh samples without reallocating their source."""
    sampler = _sampler()
    sampler.generate_action_samples(2000)
    source = sampler.generate_action_samples(96)
    destination = torch.empty_like(source)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        destination.copy_(source)
    first = source.clone()
    updated = sampler.generate_action_samples(96)
    assert updated.data_ptr() == source.data_ptr()
    assert not torch.equal(updated, first)
    graph.replay()
    torch.testing.assert_close(destination, updated, rtol=0, atol=0)


@pytest.mark.parametrize("offset", (0, 2016))
def test_rrt_hybrid_selects_close_pairs_and_preserves_global_samples(offset: int) -> None:
    """Guide toward close endpoints while retaining fresh global rows and cache data."""
    sampler = _sampler()
    sampler.generate_action_samples(offset)
    uniform = sampler.generate_proposals(1, 24)
    original = uniform.clone()
    buffer = sampler.action_sample_generator
    center = (buffer.low_bounds + buffer.up_bounds) * 0.5
    active = center.expand_as(uniform).clone()
    active[0, :, 0] += torch.arange(24, device="cuda") * 0.01
    passive = active.clone()
    passive[0, :, 1] += torch.arange(24, 0, -1, device="cuda") * 0.01
    sampler.reset_seed()
    sampler.generate_action_samples(offset)
    result = sampler.generate_proposals(1, 24, active, passive, scale=0.0)
    indices = torch.tensor([23, 22, 21, 20] * 4, device="cuda")
    expected = (active[:, indices] + passive[:, indices]) * 0.5
    torch.testing.assert_close(result[:, :16], expected)
    torch.testing.assert_close(result[:, 16:], original[:, 16:], rtol=0, atol=0)
    torch.testing.assert_close(uniform, original, rtol=0, atol=0)


def test_rrt_hybrid_bounds_noise_and_reuses_graph_storage() -> None:
    """Later growth steps refresh fixed storage; reset reproduces the same stream."""
    sampler = _sampler(seed=7)
    buffer = sampler.action_sample_generator
    lower, upper = buffer.low_bounds, buffer.up_bounds
    active = lower.expand(4, 24, lower.numel()).contiguous()
    passive = upper.expand_as(active).contiguous()
    initial = sampler.generate_proposals(4, 24).clone()
    first = sampler.generate_proposals(4, 24, active, passive)
    original = first.clone()
    destination = torch.empty_like(first)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        destination.copy_(first)
    second = sampler.generate_proposals(4, 24, active, passive)
    assert second.data_ptr() == first.data_ptr()
    assert not torch.equal(second[:, :16], original[:, :16])
    assert torch.all(second >= lower) and torch.all(second <= upper)
    graph.replay()
    torch.testing.assert_close(destination, second, rtol=0, atol=0)
    sampler.reset_seed()
    torch.testing.assert_close(sampler.generate_proposals(4, 24), initial, rtol=0, atol=0)
    repeated = sampler.generate_proposals(4, 24, active, passive)
    torch.testing.assert_close(repeated, original, rtol=0, atol=0)
