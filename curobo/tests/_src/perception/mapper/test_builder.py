# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""BlockSparseKernels functional builder and multi-block-size coexistence.

These tests lock in the functional-builder refactor from
``refactor_notes/mapper_block_sparse_kernel_builder_refactor.md``:

- The Python bundle is not cached.
- ESDF gather and scatter seed kernels are both present in the same bundle.
- Two TSDFs at different ``block_size`` coexist in one process with
  distinct kernel specializations and sensible numerical behavior.
"""

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.kernel.builder.builder_block_sparse_kernel import (
    make_block_sparse_kernels,
)
from curobo._src.perception.mapper.storage import (
    BlockSparseTSDF,
    BlockSparseTSDFCfg,
)
from curobo._src.util.warp import init_warp
from curobo.tests._src.perception.mapper.conftest import make_observation


@pytest.fixture(scope="module")
def warp_init():
    init_warp()
    return True


@pytest.fixture
def device():
    return "cuda:0"


# =============================================================================
# Bundle construction
# =============================================================================


class TestKernelBuilderConstruction:
    """The Python bundle is rebuilt on each call."""

    def test_same_cfg_returns_fresh_instance(self, warp_init):
        k1 = make_block_sparse_kernels(block_size=8, seeding_method="gather")
        k2 = make_block_sparse_kernels(block_size=8, seeding_method="gather")
        assert k1 is not k2
        assert k1.block_size == k2.block_size == 8

    def test_block_size_specializes_bundle(self, warp_init):
        k1 = make_block_sparse_kernels(block_size=1, seeding_method="gather")
        k4 = make_block_sparse_kernels(block_size=4, seeding_method="gather")
        k8 = make_block_sparse_kernels(block_size=8, seeding_method="gather")
        assert k1 is not k4
        assert k4 is not k8
        assert k1.block_size == 1
        assert k4.block_size == 4
        assert k8.block_size == 8

    @pytest.mark.parametrize("bad_value", [0, -1, 3, 6, True, "4", 4.0, 64])
    def test_block_size_validation(self, warp_init, bad_value):
        with pytest.raises(ValueError, match="block_size"):
            make_block_sparse_kernels(block_size=bad_value)

    def test_feature_channel_grouping_specializes_feature_kernel(self, warp_init):
        k3 = make_block_sparse_kernels(block_size=8, feature_channels_per_thread=3)
        k4 = make_block_sparse_kernels(block_size=8, feature_channels_per_thread=4)
        k5 = make_block_sparse_kernels(block_size=8, feature_channels_per_thread=5)

        assert k3.feature_channels_per_thread == 3
        assert k4.feature_channels_per_thread == 4
        assert k5.feature_channels_per_thread == 5
        assert (
            k3.integrate_features_grouped_kernel.key
            != k4.integrate_features_grouped_kernel.key
        )
        assert (
            k4.integrate_features_grouped_kernel.key
            != k5.integrate_features_grouped_kernel.key
        )
        assert "fcpt3" in k3.integrate_features_grouped_kernel.key
        assert "fcpt4" in k4.integrate_features_grouped_kernel.key
        assert "fcpt5" in k5.integrate_features_grouped_kernel.key

    @pytest.mark.parametrize("bad_value", [0, True, "3", 3.0])
    def test_feature_channel_grouping_validation(self, warp_init, bad_value):
        with pytest.raises(ValueError, match="feature_channels_per_thread"):
            make_block_sparse_kernels(
                block_size=8,
                feature_channels_per_thread=bad_value,
            )

    def test_prebuilt_kernel_grouping_must_match_tsdf_config(self, warp_init, device):
        cfg = BlockSparseTSDFCfg(
            voxel_size=0.01,
            max_blocks=100,
            device=device,
            grid_shape=(128, 128, 128),
            feature_channels_per_thread=3,
        )
        kernels = make_block_sparse_kernels(cfg, feature_channels_per_thread=4)

        with pytest.raises(AssertionError, match="feature_channels_per_thread"):
            BlockSparseTSDF(cfg, kernels=kernels)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("grid_shape", (64, 128, 128), "grid_shape"),
            ("origin", torch.tensor([-0.5, -1.0, 0.0]), "origin_xyz"),
            ("voxel_size", 0.02, "voxel_size"),
            ("truncation_distance", 0.08, "truncation_distance"),
        ],
    )
    def test_prebuilt_kernel_geometry_must_match_tsdf_config(
        self, warp_init, device, field, value, match
    ):
        base_kwargs = {
            "voxel_size": 0.01,
            "origin": torch.tensor([-1.0, -1.0, 0.0]),
            "truncation_distance": 0.04,
            "max_blocks": 100,
            "device": device,
            "grid_shape": (128, 128, 128),
        }
        kernel_cfg = BlockSparseTSDFCfg(**base_kwargs)
        kernels = make_block_sparse_kernels(kernel_cfg)

        mismatch_kwargs = dict(base_kwargs)
        mismatch_kwargs[field] = value
        cfg = BlockSparseTSDFCfg(**mismatch_kwargs)

        with pytest.raises(AssertionError, match=match):
            BlockSparseTSDF(cfg, kernels=kernels)

    def test_seeding_method_is_python_policy(self, warp_init):
        k_g = make_block_sparse_kernels(block_size=8, seeding_method="gather")
        k_s = make_block_sparse_kernels(block_size=8, seeding_method="scatter")
        assert k_g is not k_s
        assert hasattr(k_g, "seed_esdf_sites_gather_kernel")
        assert hasattr(k_g, "seed_esdf_sites_from_block_sparse_kernel")
        assert hasattr(k_s, "seed_esdf_sites_gather_kernel")
        assert hasattr(k_s, "seed_esdf_sites_from_block_sparse_kernel")

    def test_voxel_size_specializes_geometry_without_affecting_block_size(
        self, warp_init, device
    ):
        """``voxel_size`` specializes geometry while preserving block size."""
        cfg_a = BlockSparseTSDFCfg(
            voxel_size=0.01,
            max_blocks=100,
            device=device,
            grid_shape=(128, 128, 128),
        )
        cfg_b = BlockSparseTSDFCfg(
            voxel_size=0.02,
            max_blocks=100,
            device=device,
            grid_shape=(128, 128, 128),
        )
        t_a = BlockSparseTSDF(cfg_a)
        t_b = BlockSparseTSDF(cfg_b)
        assert t_a.kernels is not t_b.kernels
        assert t_a.kernels.block_size == t_b.kernels.block_size == 8
        assert t_a.kernels.voxel_size == 0.01
        assert t_b.kernels.voxel_size == 0.02
        assert t_a.kernels.block_size == t_b.kernels.block_size == 8


# =============================================================================
# ESDF seed kernel availability
# =============================================================================


class TestSeedKernelAvailability:
    """Both ESDF seed variants are always built."""

    def test_both_seed_variants_exist(self, warp_init):
        k = make_block_sparse_kernels(block_size=8, seeding_method="gather")
        assert k.seed_esdf_sites_from_block_sparse_kernel is not None
        assert k.seed_esdf_sites_gather_kernel is not None

    def test_unknown_attr_falls_through_to_attribute_error(self, warp_init):
        k = make_block_sparse_kernels(block_size=8)
        with pytest.raises(AttributeError):
            _ = k.totally_nonexistent_kernel_name


# =============================================================================
# Multi-block-size coexistence
# =============================================================================


class TestMultipleBlockSizesInOneProcess:
    """Validate coexistence for two block sizes.

    The mappers should produce distinct kernel specializations and show
    sensible numerical behavior on the same input frame.
    """

    def test_bs4_and_bs8_coexist(self, warp_init, device):
        integrators = {}
        for bs in (4, 8):
            cfg = BlockSparseTSDFIntegratorCfg(
                voxel_size=0.01,
                origin=torch.tensor([0.0, 0.0, 0.0]),
                grid_shape=(512, 512, 512),
                max_blocks=2000,
                device=device,
                block_size=bs,
                image_height=64,
                image_width=64,
            )
            integrators[bs] = BlockSparseTSDFIntegrator(cfg)

        # Distinct kernel specializations — different Warp modules.
        assert integrators[4].tsdf.kernels is not integrators[8].tsdf.kernels
        assert integrators[4].tsdf.block_size == 4
        assert integrators[8].tsdf.block_size == 8

        # Synthetic 1 m flat depth frame.
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
        obs = make_observation(depth, rgb, position, quaternion, intrinsics)

        allocated = {}
        for bs, integ in integrators.items():
            integ.integrate(obs)
            stats = integ.get_stats()
            assert stats["num_allocated"] > 0, f"BS={bs} allocated 0 blocks"
            allocated[bs] = stats["num_allocated"]

        # Smaller blocks tile the same surface with more blocks.
        # Exact ratio is workload-dependent; "strictly more" is the
        # load-bearing assertion.
        assert allocated[4] > allocated[8], (
            f"Expected BS=4 to allocate more blocks than BS=8; "
            f"got {allocated[4]} vs {allocated[8]}"
        )


class TestMapperCfgForwarding:
    """Validate MapperCfg forwarding into the integrator.

    Regressions here are silent: the mapper still works, but it runs at
    the ESDF integrator's default block_size instead of the user's
    requested one.
    """

    def test_mapper_cfg_block_size_reaches_tsdf(self, warp_init):
        """Check that ``MapperCfg(block_size=4)`` reaches the TSDF.

        A missing field in ``mapper.py``'s
        ``MapperCfg`` -> ``BlockSparseESDFIntegratorCfg`` conversion
        silently drops the user's value.
        """
        from curobo._src.perception.mapper.mapper import Mapper
        from curobo._src.perception.mapper.mapper_cfg import MapperCfg

        for bs in (4, 8):
            mapper = Mapper(
                MapperCfg(
                    extent_meters_xyz=(1.0, 1.0, 1.0),
                    voxel_size=0.02,
                    esdf_voxel_size=0.05,
                    block_size=bs,
                    image_height=64,
                    image_width=64,
                )
            )
            assert mapper.tsdf.block_size == bs, (
                f"MapperCfg(block_size={bs}) produced "
                f"mapper.tsdf.block_size={mapper.tsdf.block_size} — "
                f"plumbing from MapperCfg to ESDF integrator dropped the field"
            )
            assert mapper.tsdf.kernels.block_size == bs

    def test_mapper_cfg_visible_capacity_reaches_voxel_integrator(self, warp_init):
        """Visible-frame capacity must size voxel-project scratch, not TSDF storage."""
        from curobo._src.perception.mapper.mapper import Mapper
        from curobo._src.perception.mapper.mapper_cfg import MapperCfg

        mapper = Mapper(
            MapperCfg(
                extent_meters_xyz=(1.0, 1.0, 1.0),
                voxel_size=0.02,
                esdf_voxel_size=0.05,
                num_cameras=2,
                image_height=64,
                image_width=64,
                max_visible_blocks_per_integration=17,
                max_support_pixels_per_block_camera=9,
                feature_integration_kernel="grouped",
                profile_integration_kernel_timings=True,
            )
        )

        vp = mapper.integrator._tsdf_integrator._integrator
        max_blocks = mapper.tsdf.config.max_blocks

        assert mapper.integrator._tsdf_integrator.config.max_visible_blocks_per_integration == 17
        assert mapper.integrator._tsdf_integrator.config.max_support_pixels_per_block_camera == 9
        assert mapper.integrator._tsdf_integrator.config.feature_integration_kernel == "grouped"
        assert mapper.integrator._tsdf_integrator.config.profile_integration_kernel_timings is True
        assert vp.max_visible_blocks_per_integration == 17
        assert vp.max_support_pixels_per_block_camera == 9
        assert vp.use_tiled_feature_kernel is False
        assert vp.profile_kernel_timings is True
        assert vp.pool_indices.shape == (17,)
        assert vp.support_counts.shape == (17, 2)
        assert vp.support_pixels.shape == (17, 2, 9)
        assert vp.clear_pool_indices.shape == (max_blocks,)
        assert vp.visible_epoch.shape == (max_blocks,)
        assert vp.pool_to_visible_slot.shape == (max_blocks,)
        stats = mapper.get_stats(scan_pool=False)
        assert stats["last_integration"]["use_tiled_feature_kernel"] is False
        assert stats["last_integration_kernel_timings_ms"] == {}
