# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signed ESDF regressions for narrow-band static obstacle stamping."""

import pytest
import torch

from curobo._src.geom.data.data_scene import SceneData
from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegrator,
    BlockSparseESDFIntegratorCfg,
)
from curobo._src.types.camera import CameraObservation
from curobo.scene import Cuboid
from curobo.types import DeviceCfg, Pose


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("seeding_method", ["gather", "scatter"])
@pytest.mark.parametrize("esdf_voxel_size", [0.0025, 0.005, 0.01])
@pytest.mark.parametrize("layout", ["solid", "thin", "rotated", "overlap", "gap"])
@pytest.mark.parametrize(
    "truncation_distance, observed_band",
    [(0.01, None), (0.04, None), (0.08, None), (0.08, 0.025)],
    ids=["narrow_band", "wide_band", "wider_band", "missing_band_probe"],
)
def test_stamped_esdf_recovers_sign_and_surface_distance(
    seeding_method: str,
    esdf_voxel_size: float,
    layout: str,
    truncation_distance: float,
    observed_band: float | None,
) -> None:
    """Compare every resolved voxel to the analytic union of cuboids."""
    voxel_size = 0.005
    shape = (64, 64, 64)
    esdf_extent = 0.32
    esdf_size = round(esdf_extent / esdf_voxel_size)
    config = BlockSparseESDFIntegratorCfg(
        voxel_size=voxel_size,
        origin=torch.zeros(3),
        grid_shape=shape,
        esdf_voxel_size=esdf_voxel_size,
        esdf_grid_shape=(esdf_size,) * 3,
        truncation_distance=truncation_distance,
        max_blocks=512,
        hash_capacity=1024,
        enable_static=True,
        image_height=1,
        image_width=1,
        color_grid_size=1,
        feature_dim=0,
        instance_grid_size=0,
        use_cuda_graph=seeding_method == "gather",
        seeding_method=seeding_method,
        edt_solver="pba",
    )
    mapper = BlockSparseESDFIntegrator(config)
    device = DeviceCfg()
    scene = SceneData.create_cache(num_envs=1, device_cfg=device, cuboid_cache=2, mesh_cache=0)
    boxes = [([0.0, 0.0, 0.0], [0.2, 0.2, 0.2], 0.0)]
    if layout == "thin":
        boxes = [([0.0, 0.0, 0.0], [0.02, 0.2, 0.2], 0.0)]
    elif layout == "rotated":
        boxes = [([0.005, -0.005, 0.0], [0.16, 0.12, 0.2], 0.43)]
    elif layout == "overlap":
        boxes = [
            ([-0.03, 0.0, 0.0], [0.14, 0.2, 0.2], 0.0),
            ([0.03, 0.0, 0.0], [0.14, 0.2, 0.2], 0.0),
        ]
    elif layout == "gap":
        boxes = [
            ([-0.06, 0.0, 0.0], [0.08, 0.2, 0.2], 0.0),
            ([0.06, 0.0, 0.0], [0.08, 0.2, 0.2], 0.0),
        ]

    axis = (
        torch.arange(esdf_size, device="cuda", dtype=torch.float32) + 0.5 - esdf_size * 0.5
    ) * esdf_voxel_size
    points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    exact = torch.full(points.shape[:-1], torch.inf, device="cuda", dtype=torch.float32)
    for index, (position, dims, angle) in enumerate(boxes):
        angle_tensor = torch.tensor(angle, dtype=torch.float32, device="cuda")
        cosine, sine = angle_tensor.cos(), angle_tensor.sin()
        delta = points - torch.tensor(position, device="cuda", dtype=torch.float32)
        local = torch.stack(
            (
                cosine * delta[..., 0] + sine * delta[..., 1],
                -sine * delta[..., 0] + cosine * delta[..., 1],
                delta[..., 2],
            ),
            dim=-1,
        )
        q = local.abs() - torch.tensor(dims, device="cuda", dtype=torch.float32) * 0.5
        sdf = torch.linalg.vector_norm(q.clamp_min(0.0), dim=-1) + q.amax(-1).clamp_max(0.0)
        exact = torch.minimum(exact, sdf)
        quaternion = [
            float((angle_tensor * 0.5).cos()),
            0.0,
            0.0,
            float((angle_tensor * 0.5).sin()),
        ]
        scene.add_obstacle(
            Cuboid(name=f"box_{index}", dims=dims, pose=position + quaternion), env_idx=0
        )
    mapper.update_static_obstacles(scene)
    if observed_band is not None:
        # Leave the surface intact but remove samples at the preferred half-
        # truncation probe. The short local search must still recover the sign.
        values = mapper._tsdf_integrator.tsdf.data.static_block_data
        values.masked_fill_(values.abs() > observed_band, torch.inf)
    actual = mapper.compute_esdf().reshape(exact.shape).float().clone()
    # Replaying a captured gather pipeline must preserve the result.
    torch.testing.assert_close(mapper.compute_esdf().reshape(exact.shape).float(), actual)
    assert int(mapper._tsdf_integrator.tsdf.data.allocation_failures.item()) == 0

    # Gather deliberately dilates the surface seeds. Exclude only the resulting
    # discretization band when checking signs; verify distance separately.
    tolerance = 2.0 * esdf_voxel_size + voxel_size
    assert torch.all(actual[exact < -tolerance] < 0.0)
    assert torch.all(actual[exact > tolerance] > 0.0)
    # For overlap the union SDF is a sign oracle, but its interior magnitude is
    # not necessarily distance to the exposed union surface.
    if layout != "overlap":
        assert (actual.abs() - exact.abs()).abs().amax() <= tolerance


def _sensor_mapper(
    seeding_method: str,
    edt_solver: str = "pba",
    esdf_grid_shape: tuple[int, int, int] = (64, 64, 64),
) -> BlockSparseESDFIntegrator:
    """Create a small mapper with room for a depth plane and disjoint stamps."""
    return BlockSparseESDFIntegrator(
        BlockSparseESDFIntegratorCfg(
            voxel_size=0.005,
            origin=torch.zeros(3),
            grid_shape=(64, 64, 64),
            esdf_grid_shape=esdf_grid_shape,
            truncation_distance=0.04,
            max_blocks=512,
            hash_capacity=1024,
            enable_static=True,
            image_height=32,
            image_width=32,
            color_grid_size=1,
            feature_dim=0,
            instance_grid_size=0,
            use_cuda_graph=seeding_method == "gather",
            seeding_method=seeding_method,
            edt_solver=edt_solver,
        )
    )


def _plane_observation(camera_x: float = 0.0) -> CameraObservation:
    """Observe a finite plane at world Z=0 from a camera 15 cm in front."""
    return CameraObservation(
        depth_image=torch.full((1, 32, 32), 0.15, device="cuda", dtype=torch.float32),
        rgb_image=torch.zeros((1, 32, 32, 3), device="cuda", dtype=torch.uint8),
        pose=Pose.from_list([camera_x, 0.0, -0.15, 1.0, 0.0, 0.0, 0.0]),
        intrinsics=torch.tensor(
            [[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]],
            device="cuda",
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("seeding_method", ["gather", "scatter"])
@pytest.mark.parametrize("edt_solver", ["pba", "jfa"])
def test_sensor_esdf_does_not_extrapolate_sign_into_unknown(
    seeding_method: str, edt_solver: str
) -> None:
    """A sensor's negative observed band must not fill its occluded region."""
    mapper = _sensor_mapper(seeding_method, edt_solver)
    observation = _plane_observation()
    mapper.integrate(observation)
    actual = mapper.compute_esdf().clone()
    assert actual[32, 32, 34] < 0.0  # Observed, 12.5 mm behind the plane.
    assert actual[32, 32, 29] > 0.0  # Observed, 12.5 mm in front.
    assert actual[32, 32, 48] > 0.0  # Unobserved, 82.5 mm behind the plane.
    assert torch.count_nonzero(mapper._static_site_mask) == 0
    # Negative truncation sites keep the occupied/unknown boundary nearby.
    assert actual[32, 32, 48] < 0.06


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("seeding_method", ["gather", "scatter"])
def test_static_seed_owns_coincident_sensor_sites_and_mask_resets(
    seeding_method: str,
) -> None:
    """Static ownership wins collisions, including packed bit 31, and resets."""
    mapper = _sensor_mapper(seeding_method)
    scene = SceneData.create_cache(
        num_envs=1, device_cfg=DeviceCfg(), cuboid_cache=1, mesh_cache=0
    )
    scene.add_obstacle(
        Cuboid(name="solid", dims=[0.2, 0.2, 0.2], pose=[0, 0, 0, 1, 0, 0, 0]), env_idx=0
    )
    mapper.update_static_obstacles(scene)
    data = mapper.tsdf.data
    count = int(data.num_allocated.item())
    static = data.static_block_data[:count]
    surface = static.abs() <= 0.005 * 0.9
    data.block_data[:count, :, 0] = torch.where(surface, static, 0.0)
    data.block_data[:count, :, 1] = surface.to(torch.float16)
    mapper._seed_esdf_impl(mapper._origin, mapper._esdf_voxel_size)
    indices = torch.nonzero(mapper._site_index.reshape(-1) >= 0).reshape(-1)
    flags = (mapper._static_site_mask[indices >> 5] >> (indices & 31)) & 1
    assert torch.all(flags == 1)
    assert torch.any((indices & 31) == 31)
    assert mapper._static_site_mask.numel() == (64**3 + 31) // 32
    assert mapper._static_site_mask.dtype == torch.int32

    # Repeat with the same sites now belonging exclusively to the sensor.
    static.fill_(torch.inf)
    mapper._seed_esdf_impl(mapper._origin, mapper._esdf_voxel_size)
    assert torch.any(mapper._site_index >= 0)
    assert torch.count_nonzero(mapper._static_site_mask) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("seeding_method", ["gather", "scatter"])
def test_disjoint_stamp_and_sensor_keep_independent_sign_policies(
    seeding_method: str,
) -> None:
    """Static interiors stay occupied while sensor-occluded space stays free."""
    mapper = _sensor_mapper(seeding_method)
    scene = SceneData.create_cache(
        num_envs=1, device_cfg=DeviceCfg(), cuboid_cache=1, mesh_cache=0
    )
    scene.add_obstacle(
        Cuboid(name="solid", dims=[0.06, 0.1, 0.1], pose=[-0.10, 0, 0, 1, 0, 0, 0]), env_idx=0
    )
    mapper.update_static_obstacles(scene)
    observation = _plane_observation(camera_x=0.08)
    mapper.integrate(observation)
    actual = mapper.compute_esdf().clone()
    assert actual[12, 32, 32] < -0.015
    assert actual[48, 32, 34] < 0.0
    assert actual[48, 32, 48] > 0.0
    torch.testing.assert_close(mapper.compute_esdf(), actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("seeding_method", ["gather", "scatter"])
def test_static_sign_search_stays_within_tsdf_volume(seeding_method: str) -> None:
    """Clipped stamp faces cannot establish occupancy beyond map coverage."""
    mapper = _sensor_mapper(seeding_method, esdf_grid_shape=(96, 96, 96))
    scene = SceneData.create_cache(
        num_envs=1, device_cfg=DeviceCfg(), cuboid_cache=1, mesh_cache=0
    )
    scene.add_obstacle(
        Cuboid(name="clipped", dims=[0.4, 0.2, 0.2], pose=[0, 0, 0, 1, 0, 0, 0]), env_idx=0
    )
    mapper.update_static_obstacles(scene)
    actual = mapper.compute_esdf().clone()
    assert actual[48, 48, 48] < 0.0  # Interior query inside the TSDF volume.
    assert actual[84, 48, 48] > 0.0  # Outside map coverage, inside the full solid.
    assert actual[92, 48, 48] > 0.0  # Outside both coverage and the full solid.
