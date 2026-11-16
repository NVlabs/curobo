# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for rigid-body dynamics CUDA launch configuration."""

from curobo._src.curobolib.backends.cuda_core_backend.dynamics_config import (
    DynamicsLaunchCfg,
)


def test_batch_256_forward_targets_grid_coverage() -> None:
    """Batch-256 forward launches should expose two warp-width states per block."""
    for num_links in (7, 12, 29):
        config = DynamicsLaunchCfg.calculate_forward_config(
            batch_size=256,
            num_links=num_links,
            threads_per_batch=32,
        )
        assert config.grid == (128, 1, 1)
        assert config.block == (2 * 32, 1, 1)
        assert config.shmem_size == 2 * num_links * 24 * 4


def test_batch_256_backward_uses_topology_specific_grid() -> None:
    """Batch-256 backward should expose two warp-width states per block."""
    for num_links in (7, 12, 29):
        cooperative_config = DynamicsLaunchCfg.calculate_backward_config(
            batch_size=256,
            num_links=num_links,
            threads_per_batch=32,
        )
        assert cooperative_config.grid == (128, 1, 1)
        assert cooperative_config.block == (2 * 32, 1, 1)
        assert cooperative_config.shmem_size == num_links * (12 + 2 * 44) * 4


def test_batch_256_mass_matrix_uses_expected_shared_memory() -> None:
    """CRBA forward and backward should use their per-state shared layouts."""
    for num_links, threads_per_batch in ((7, 8), (12, 16), (49, 32)):
        forward = DynamicsLaunchCfg.calculate_mass_matrix_config(
            batch_size=256,
            num_links=num_links,
            threads_per_batch=threads_per_batch,
        )
        backward = DynamicsLaunchCfg.calculate_mass_matrix_backward_config(
            batch_size=256,
            num_links=num_links,
            threads_per_batch=threads_per_batch,
        )

        assert forward.grid == (128, 1, 1)
        assert forward.block == (2 * threads_per_batch, 1, 1)
        assert forward.shmem_size == 2 * num_links * 48 * 4
        assert backward.grid == (256, 1, 1)
        assert backward.block == (threads_per_batch, 1, 1)
        assert backward.shmem_size == num_links * 84 * 4


def test_explicit_batch_cap_overrides_default_policy() -> None:
    """Explicit caps should remain available for controlled experiments."""
    forward = DynamicsLaunchCfg.calculate_forward_config(
        batch_size=256,
        num_links=13,
        threads_per_batch=1,
        max_batches_per_block=4,
    )
    backward = DynamicsLaunchCfg.calculate_backward_config(
        batch_size=256,
        num_links=13,
        threads_per_batch=1,
        max_batches_per_block=4,
    )
    assert forward.grid == (64, 1, 1)
    assert forward.block == (4, 1, 1)
    assert backward.grid == (64, 1, 1)
    assert backward.block == (4, 1, 1)


def test_small_sm_serial_policy_packs_eight_states_per_block() -> None:
    """Serial RNEA should expose more blocks under the small-SM policy."""
    forward = DynamicsLaunchCfg.calculate_forward_config(
        batch_size=976,
        num_links=7,
        threads_per_batch=1,
        policy_sm_count=16,
    )
    backward = DynamicsLaunchCfg.calculate_backward_config(
        batch_size=976,
        num_links=7,
        threads_per_batch=1,
        policy_sm_count=16,
    )

    assert forward.grid == (122, 1, 1)
    assert forward.block == (8, 1, 1)
    assert forward.shmem_size == 8 * 7 * 24 * 4
    assert backward.grid == (122, 1, 1)
    assert backward.block == (8, 1, 1)
    assert backward.shmem_size == 7 * (12 + 8 * 30) * 4


def test_default_policy_uses_runtime_traversal_width() -> None:
    """Default state packing should follow the selected RNEA traversal width."""
    cases = (
        # batch_size, threads_per_batch, forward cap, backward cap
        (256, 32, 2, 2),
        (2048, 16, 4, 2),
        (8192, 16, 8, 8),
        (2048, 4, 4, 4),
        (4096, 4, 4, 8),
        (8192, 4, 4, 4),
        (8192, 2, 8, 8),
        (16384, 1, 16, 16),
    )
    for batch_size, threads_per_batch, forward_cap, backward_cap in cases:
        assert (
            DynamicsLaunchCfg._default_forward_batches_per_block(
                batch_size,
                threads_per_batch,
            )
            == forward_cap
        )
        assert (
            DynamicsLaunchCfg._default_backward_batches_per_block(
                batch_size,
                threads_per_batch,
            )
            == backward_cap
        )
