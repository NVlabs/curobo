# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for ``get_matching_feature_voxels`` and ``MatchedVoxels``.

These tests focus on the post-integration matching path: shape and dtype
of the returned dataclass, descending-by-score ordering, the
parallel-arrays invariant between ``block_pool_idx`` and ``block_scores``,
``scores_per_voxel`` correctness, and the empty / over-clamped /
invalid-arg edge cases.

To isolate matching from the integration pipeline, blocks are allocated
by integrating a synthetic depth observation, after which the per-block
feature accumulators are overwritten with deterministic patterns. This
keeps the cosine math hermetic without depending on the full feature
integration kernel.
"""

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.storage import (
    MatchedVoxels,
    OccupiedVoxels,
)
from curobo._src.util.warp import init_warp
from curobo.tests._src.perception.mapper.conftest import make_observation


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def warp_init():
    init_warp()
    return True


@pytest.fixture
def device():
    return "cuda:0"


def _make_integrator(
    device: str,
    feature_dim: int,
    max_blocks: int = 256,
    voxel_size: float = 0.02,
) -> BlockSparseTSDFIntegrator:
    feature_grid_kwargs = {}
    if feature_dim > 0:
        feature_grid_kwargs = {
            "feature_grid_height": 1,
            "feature_grid_width": 1,
        }
    config = BlockSparseTSDFIntegratorCfg(
        voxel_size=voxel_size,
        origin=torch.tensor([0.0, 0.0, 0.0]),
        grid_shape=(512, 512, 512),
        max_blocks=max_blocks,
        device=device,
        image_height=64,
        image_width=64,
        feature_dim=feature_dim,
        **feature_grid_kwargs,
    )
    return BlockSparseTSDFIntegrator(config)


def _allocate_blocks(integrator: BlockSparseTSDFIntegrator, device: str) -> int:
    """Integrate one synthetic frame to allocate blocks. Returns num_alloc."""
    img_H, img_W = 64, 64
    depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
    rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
    position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    intrinsics = torch.tensor(
        [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))
    return int(integrator._tsdf.data.num_allocated.item())


def _stamp_orthogonal_features(
    integrator: BlockSparseTSDFIntegrator,
    num_alloc: int,
    feature_dim: int,
) -> torch.Tensor:
    """Overwrite per-block features with deterministic orthogonal patterns.

    Each active block ``i`` gets a unit-magnitude feature equal to
    ``e_{i mod feature_dim}`` (one-hot in feature space). Block weight
    is set to 1, so the integrator reads back the unit vector exactly
    after dividing by weight. Returns the ``(num_alloc, feature_dim)``
    fp32 ground-truth feature matrix for use by tests.
    """
    device = integrator._tsdf.data.block_features.device
    gt_features = torch.zeros((num_alloc, feature_dim), dtype=torch.float32, device=device)
    for i in range(num_alloc):
        gt_features[i, i % feature_dim] = 1.0
    integrator._tsdf.data.block_features[:num_alloc] = gt_features.to(torch.float16)
    integrator._tsdf.data.block_feature_weight[:num_alloc] = 1.0
    return gt_features


# =============================================================================
# Tests
# =============================================================================


class TestMatchedVoxelsShape:
    """Return-type shape, dtype, and ordering invariants."""

    def test_returns_matched_voxels_dataclass(self, warp_init, device):
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        assert num_alloc > 0
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        query = torch.zeros(feature_dim, device=device)
        query[0] = 1.0
        result = integrator.get_matching_feature_voxels(query, top_k=5)

        assert isinstance(result, MatchedVoxels)
        assert isinstance(result.voxels, OccupiedVoxels)
        assert result.block_pool_idx.dtype == torch.int32
        assert result.block_scores.dtype == torch.float32
        assert result.block_pool_idx.device.type == "cuda"
        assert result.block_scores.device.type == "cuda"

    def test_parallel_arrays_same_length(self, warp_init, device):
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        query = torch.randn(feature_dim, device=device)
        result = integrator.get_matching_feature_voxels(query, top_k=7)

        k_expected = min(7, num_alloc)
        assert result.block_pool_idx.shape == (k_expected,)
        assert result.block_scores.shape == (k_expected,)

    def test_scores_descending(self, warp_init, device):
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        query = torch.randn(feature_dim, device=device)
        result = integrator.get_matching_feature_voxels(query, top_k=min(20, num_alloc))

        scores = result.block_scores
        if scores.numel() > 1:
            assert torch.all(scores[:-1] >= scores[1:]), (
                f"block_scores must be descending; got {scores.tolist()}"
            )

    def test_parallel_arrays_match_recomputed_cosine(self, warp_init, device):
        """For each rank ``i``, ``block_scores[i]`` must equal cosine of the
        query against the block at ``block_pool_idx[i]``.

        This is the load-bearing invariant for ``scores_per_voxel`` and
        any downstream filtering: if scores and pool ids ever drift out of
        alignment, callers silently get wrong scores for the right blocks.
        """
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        gt_features = _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        query = torch.randn(feature_dim, device=device)
        result = integrator.get_matching_feature_voxels(query, top_k=min(15, num_alloc))

        q_norm = query / query.norm().clamp(min=1e-6)
        # gt_features rows are already unit-magnitude one-hots.
        expected = (gt_features[result.block_pool_idx.long()] * q_norm).sum(dim=-1)
        assert torch.allclose(result.block_scores, expected, atol=1e-4)


class TestMatchedVoxelsScoresPerVoxel:
    """``scores_per_voxel`` gather correctness."""

    def test_scores_per_voxel_matches_block_scores(self, warp_init, device):
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        query = torch.randn(feature_dim, device=device)
        result = integrator.get_matching_feature_voxels(query, top_k=min(10, num_alloc))

        per_voxel = result.scores_per_voxel()
        assert per_voxel.shape == (len(result.voxels),)
        assert per_voxel.dtype == torch.float32

        # Build the expected lookup independently and compare element-wise.
        block_idx = result.voxels.block_idx_per_voxel.long()
        lookup = result.block_scores.new_full(
            (result.voxels.block_data.num_allocated,), float("nan")
        )
        lookup[result.block_pool_idx.long()] = result.block_scores
        expected = lookup[block_idx]
        assert torch.equal(per_voxel, expected)

        # Every voxel must come from a matched block, so no NaNs leak through.
        if per_voxel.numel() > 0:
            assert not torch.isnan(per_voxel).any()

    def test_scores_per_voxel_empty(self, warp_init, device):
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        # No integration -> num_allocated == 0.
        query = torch.zeros(feature_dim, device=device)
        result = integrator.get_matching_feature_voxels(query, top_k=5)

        per_voxel = result.scores_per_voxel()
        assert per_voxel.shape == (0,)
        assert per_voxel.dtype == torch.float32


class TestMatchedVoxelsEdgeCases:
    """Empty-map, over-clamped top-k, and invalid-argument paths."""

    def test_empty_map_returns_empty_dataclass(self, warp_init, device):
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        # Skip _allocate_blocks; num_alloc remains 0.

        query = torch.zeros(feature_dim, device=device)
        result = integrator.get_matching_feature_voxels(query, top_k=10)

        assert isinstance(result, MatchedVoxels)
        assert len(result) == 0
        assert result.block_pool_idx.shape == (0,)
        assert result.block_scores.shape == (0,)
        assert result.block_pool_idx.dtype == torch.int32
        assert result.block_scores.dtype == torch.float32

    def test_top_k_exceeds_num_alloc_clamps(self, warp_init, device):
        feature_dim = 4
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        oversized = num_alloc + 100
        query = torch.zeros(feature_dim, device=device)
        query[0] = 1.0
        result = integrator.get_matching_feature_voxels(query, top_k=oversized)
        assert result.block_pool_idx.shape == (num_alloc,)
        assert result.block_scores.shape == (num_alloc,)

    def test_inactive_recycled_slots_are_not_scored(self, warp_init, device):
        feature_dim = 4
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        assert num_alloc >= 2
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        inactive_pool_idx = 0
        integrator._tsdf.data.block_to_hash_slot[inactive_pool_idx] = -1
        integrator._tsdf.data.block_features[inactive_pool_idx].zero_()
        integrator._tsdf.data.block_features[inactive_pool_idx, 0] = 100.0
        integrator._tsdf.data.block_feature_weight[inactive_pool_idx] = 1.0

        query = torch.zeros(feature_dim, device=device)
        query[0] = 1.0
        result = integrator.get_matching_feature_voxels(query, top_k=num_alloc)

        assert inactive_pool_idx not in result.block_pool_idx.cpu().tolist()
        active_count = int((integrator._tsdf.data.block_to_hash_slot[:num_alloc] >= 0).sum().item())
        assert result.block_pool_idx.shape == (active_count,)

    def test_top_k_zero_or_negative_raises(self, warp_init, device):
        feature_dim = 4
        integrator = _make_integrator(device, feature_dim=feature_dim)
        query = torch.zeros(feature_dim, device=device)
        with pytest.raises(ValueError, match="top_k must be positive"):
            integrator.get_matching_feature_voxels(query, top_k=0)
        with pytest.raises(ValueError, match="top_k must be positive"):
            integrator.get_matching_feature_voxels(query, top_k=-3)

    def test_no_features_raises(self, warp_init, device):
        integrator = _make_integrator(device, feature_dim=0)
        query = torch.zeros(1, device=device)
        with pytest.raises(RuntimeError, match="requires feature_dim"):
            integrator.get_matching_feature_voxels(query, top_k=5)

    def test_wrong_feature_vector_shape_raises(self, warp_init, device):
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        bad_2d = torch.zeros((2, feature_dim), device=device)
        with pytest.raises(ValueError, match="feature_vector must be shape"):
            integrator.get_matching_feature_voxels(bad_2d, top_k=5)
        wrong_dim = torch.zeros(feature_dim + 1, device=device)
        with pytest.raises(ValueError, match="feature_vector must be shape"):
            integrator.get_matching_feature_voxels(wrong_dim, top_k=5)


class TestMatchedVoxelsClearBlocksIntegration:
    """``block_pool_idx`` should drop straight into ``clear_blocks``."""

    def test_block_pool_idx_clears_blocks(self, warp_init, device):
        feature_dim = 4
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        query = torch.zeros(feature_dim, device=device)
        query[0] = 1.0
        result = integrator.get_matching_feature_voxels(query, top_k=min(5, num_alloc))
        # No casting / no torch.unique() should be required: block_pool_idx
        # is already int32 and free of duplicates by construction.
        assert result.block_pool_idx.dtype == torch.int32
        n_cleared = integrator.clear_blocks(result.block_pool_idx)
        assert n_cleared == result.block_pool_idx.numel()


class TestMatchedVoxelsMinimumScore:
    """``minimum_score`` filters blocks after topk and prunes voxel work."""

    def test_threshold_keeps_only_high_scoring_blocks(self, warp_init, device):
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        # Orthogonal one-hots: a query aligned with axis 0 produces
        # cosine = 1.0 for blocks whose feature is e_0 and cosine = 0.0
        # for the rest.
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        query = torch.zeros(feature_dim, device=device)
        query[0] = 1.0
        # Threshold sits strictly between the two score levels (0 and 1)
        # so only the e_0 blocks survive.
        result = integrator.get_matching_feature_voxels(query, top_k=num_alloc, minimum_score=0.5)
        assert result.block_pool_idx.numel() > 0
        assert torch.all(result.block_scores >= 0.5)
        # Every kept block should have unit score (perfect alignment).
        assert torch.allclose(
            result.block_scores,
            torch.ones_like(result.block_scores),
            atol=1e-4,
        )

    def test_threshold_filters_subset_of_topk(self, warp_init, device):
        """A non-trivial threshold should yield fewer blocks than top_k."""
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        query = torch.zeros(feature_dim, device=device)
        query[0] = 1.0
        unfiltered = integrator.get_matching_feature_voxels(query, top_k=num_alloc)
        # Threshold above the lowest unfiltered score should strictly
        # shrink the result; the relative ordering of survivors is
        # preserved (descending).
        cutoff = float(unfiltered.block_scores.median().item())
        filtered = integrator.get_matching_feature_voxels(
            query, top_k=num_alloc, minimum_score=cutoff + 1e-3
        )
        assert filtered.block_pool_idx.numel() < unfiltered.block_pool_idx.numel()
        assert torch.all(filtered.block_scores >= cutoff + 1e-3)
        if filtered.block_scores.numel() > 1:
            assert torch.all(filtered.block_scores[:-1] >= filtered.block_scores[1:])

    def test_threshold_above_all_scores_returns_empty(self, warp_init, device):
        """No blocks above threshold -> empty MatchedVoxels (not crash)."""
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        query = torch.zeros(feature_dim, device=device)
        query[0] = 1.0
        result = integrator.get_matching_feature_voxels(
            query,
            top_k=num_alloc,
            minimum_score=2.0,  # impossible cosine
        )
        assert isinstance(result, MatchedVoxels)
        assert len(result) == 0
        assert result.block_pool_idx.shape == (0,)
        assert result.block_scores.shape == (0,)
        assert result.block_pool_idx.dtype == torch.int32

    def test_threshold_none_matches_no_threshold(self, warp_init, device):
        """``minimum_score=None`` must match the no-threshold path exactly."""
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        query = torch.randn(feature_dim, device=device)
        baseline = integrator.get_matching_feature_voxels(query, top_k=10)
        explicit = integrator.get_matching_feature_voxels(query, top_k=10, minimum_score=None)
        assert torch.equal(baseline.block_pool_idx, explicit.block_pool_idx)
        assert torch.allclose(baseline.block_scores, explicit.block_scores)

    def test_threshold_extracts_fewer_voxels(self, warp_init, device):
        """The kernel below topk should see a strictly smaller mask, so
        the extracted voxel count must drop monotonically with
        ``minimum_score``. This is the load-bearing perf claim — if the
        threshold ever stopped pruning the extraction stage, this test
        fails."""
        feature_dim = 8
        integrator = _make_integrator(device, feature_dim=feature_dim)
        num_alloc = _allocate_blocks(integrator, device)
        _stamp_orthogonal_features(integrator, num_alloc, feature_dim)

        query = torch.zeros(feature_dim, device=device)
        query[0] = 1.0
        unfiltered = integrator.get_matching_feature_voxels(query, top_k=num_alloc)
        cutoff = float(unfiltered.block_scores.median().item()) + 1e-3
        filtered = integrator.get_matching_feature_voxels(
            query, top_k=num_alloc, minimum_score=cutoff
        )
        assert len(filtered.voxels) <= len(unfiltered.voxels)
        # When the threshold actually drops blocks, voxel count must
        # shrink (each pruned block removes block_size**3 voxels).
        if filtered.block_pool_idx.numel() < unfiltered.block_pool_idx.numel():
            assert len(filtered.voxels) < len(unfiltered.voxels)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
