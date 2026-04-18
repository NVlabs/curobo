# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Comprehensive tests for sphere fitting functions."""

# Third Party
import numpy as np
import pytest

# CuRobo
from curobo._src.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo._src.geom.sphere_fit.fit_voxel import (
    sample_even_fit_mesh,
    voxel_fit_mesh,
)
from curobo._src.geom.types import Cuboid, Cylinder, Sphere


@pytest.fixture
def simple_cuboid_mesh():
    """Create a simple cuboid mesh for testing."""
    cuboid = Cuboid(name="test_cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.2, 0.2, 0.2])
    return cuboid.get_trimesh_mesh()


@pytest.fixture
def large_cuboid_mesh():
    """Create a larger cuboid mesh for testing."""
    cuboid = Cuboid(name="large_cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1.0, 0.5, 0.3])
    return cuboid.get_trimesh_mesh()


@pytest.fixture
def sphere_mesh():
    """Create a sphere mesh for testing."""
    sphere = Sphere(name="test_sphere", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.15)
    return sphere.get_trimesh_mesh()


@pytest.fixture
def cylinder_mesh():
    """Create a cylinder mesh for testing."""
    cylinder = Cylinder(name="test_cylinder", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.1, height=0.3)
    return cylinder.get_trimesh_mesh()


class TestSphereFitType:
    """Test SphereFitType enum."""

    def test_sphere_fit_type_values(self):
        """Test that all sphere fit types are accessible."""
        assert SphereFitType.SURFACE.value == "surface"
        assert SphereFitType.VOXEL.value == "voxel"
        assert SphereFitType.MORPHIT.value == "morphit"

    def test_sphere_fit_type_count(self):
        """Test that we have the expected number of fit types."""
        fit_types = list(SphereFitType)
        assert len(fit_types) == 3


class TestSampleEvenFitMesh:
    """Test sample_even_fit_mesh function."""

    def test_basic_sampling(self, simple_cuboid_mesh):
        """Test basic surface sampling."""
        num_spheres = 10
        sphere_radius = 0.01

        pts, radii = sample_even_fit_mesh(simple_cuboid_mesh, num_spheres, sphere_radius)

        assert pts is not None
        assert len(pts) > 0
        assert len(radii) == len(pts)
        assert all(r == sphere_radius for r in radii)

    def test_sampling_different_counts(self, simple_cuboid_mesh):
        """Test sampling with different sphere counts."""
        for num_spheres in [5, 20, 50]:
            pts, radii = sample_even_fit_mesh(simple_cuboid_mesh, num_spheres, 0.01)
            assert len(pts) > 0
            assert len(radii) > 0

    def test_sampling_different_radii(self, simple_cuboid_mesh):
        """Test sampling with different radii."""
        for radius in [0.005, 0.01, 0.02]:
            pts, radii = sample_even_fit_mesh(simple_cuboid_mesh, 10, radius)
            assert all(r == radius for r in radii)


class TestVoxelFitMesh:
    """Test voxel_fit_mesh function."""

    def test_basic_voxel_fit(self, large_cuboid_mesh):
        """Test basic voxel mesh fitting on larger mesh."""
        num_spheres = 50

        pts, radii = voxel_fit_mesh(large_cuboid_mesh, num_spheres)

        # voxel_fit_mesh can fail on some meshes, returning None
        if pts is not None:
            assert len(pts) > 0
            assert len(radii) == len(pts)

    def test_voxel_fit_preserves_count(self, large_cuboid_mesh):
        """Test that voxel fit returns requested number of spheres."""
        num_spheres = 40
        pts, radii = voxel_fit_mesh(large_cuboid_mesh, num_spheres)

        if pts is not None:
            # May not always match exactly due to voxelization
            assert len(pts) > 0


class TestFitSpheresToMesh:
    """Test fit_spheres_to_mesh function (main entry point)."""

    @pytest.mark.parametrize(
        "fit_type",
        [
            SphereFitType.SURFACE,
            SphereFitType.VOXEL,
            SphereFitType.MORPHIT,
        ],
    )
    def test_fit_spheres_all_types(self, large_cuboid_mesh, fit_type):
        """Test fitting spheres with all available fit types."""
        num_spheres = 30
        surface_radius = 0.01

        result = fit_spheres_to_mesh(
            large_cuboid_mesh, num_spheres, surface_radius, fit_type=fit_type
        )

        assert result.centers is not None
        assert result.num_spheres > 0
        assert len(result.radii) > 0
        assert len(result.centers) == len(result.radii)

    def test_fit_spheres_sample_surface(self, simple_cuboid_mesh):
        """Test SAMPLE_SURFACE fit type specifically."""
        result = fit_spheres_to_mesh(
            simple_cuboid_mesh, num_spheres=15, surface_radius=0.01, fit_type=SphereFitType.SURFACE
        )

        assert result.centers is not None
        assert result.num_spheres > 0
        assert (result.radii - 0.01).abs().max().item() < 1e-6

    def test_fit_spheres_voxel_volume(self, simple_cuboid_mesh):
        """Test VOXEL_VOLUME fit type."""
        result = fit_spheres_to_mesh(
            simple_cuboid_mesh,
            20,
            0.01,
            fit_type=SphereFitType.VOXEL,
        )

        assert result.centers is not None
        assert result.num_spheres > 0

    def test_fit_spheres_different_counts(self, simple_cuboid_mesh):
        """Test that fit_spheres_to_mesh handles a range of requested counts."""
        counts = [5, 15, 30, 50]
        results = []
        for num_spheres in counts:
            result = fit_spheres_to_mesh(
                simple_cuboid_mesh, num_spheres=num_spheres, surface_radius=0.01
            )
            assert result.centers is not None
            assert result.num_spheres > 0
            assert result.num_spheres <= num_spheres, (
                f"Returned more spheres ({result.num_spheres}) than requested ({num_spheres})"
            )
            results.append(result.num_spheres)

    def test_fit_spheres_different_radii(self, simple_cuboid_mesh):
        """Test fitting with different surface radii."""
        for radius in [0.005, 0.01, 0.015, 0.02]:
            result = fit_spheres_to_mesh(simple_cuboid_mesh, 20, radius)
            assert result.centers is not None
            assert result.num_spheres > 0

    def test_fit_spheres_sphere_mesh(self, sphere_mesh):
        """Test fitting spheres to a sphere mesh."""
        result = fit_spheres_to_mesh(sphere_mesh, 25, 0.01)

        assert result.centers is not None
        assert result.num_spheres > 0
        max_dist = result.centers.norm(dim=1).max().item()
        assert max_dist < 0.2

    def test_fit_spheres_cylinder_mesh(self, cylinder_mesh):
        """Test fitting spheres to a cylinder mesh."""
        result = fit_spheres_to_mesh(cylinder_mesh, 30, 0.01)

        assert result.centers is not None
        assert result.num_spheres > 0

    def test_fit_spheres_large_count(self, large_cuboid_mesh):
        """Test fitting large number of spheres."""
        result = fit_spheres_to_mesh(large_cuboid_mesh, 100, 0.01)

        assert result.centers is not None
        assert result.num_spheres > 0
        assert result.num_spheres >= 50

    def test_fit_spheres_fallback_mechanism(self, simple_cuboid_mesh):
        """Test that fallback to surface sampling works."""
        result = fit_spheres_to_mesh(simple_cuboid_mesh, 10, 0.01)

        assert result.centers is not None
        assert result.num_spheres > 0

    def test_fit_spheres_excess_points_trimmed(self, simple_cuboid_mesh):
        """Test that excess points are trimmed to requested count."""
        num_spheres = 15
        result = fit_spheres_to_mesh(
            simple_cuboid_mesh, num_spheres, 0.01, fit_type=SphereFitType.SURFACE
        )

        assert result.num_spheres <= num_spheres * 1.2

    def test_fit_spheres_voxel_volume(self, large_cuboid_mesh):
        """Test voxel volume fitting."""
        result = fit_spheres_to_mesh(
            large_cuboid_mesh,
            40,
            0.01,
            fit_type=SphereFitType.VOXEL,
        )
        assert result.num_spheres > 0


class TestSphereFitMetricsAndWeights:
    """Test compute_metrics and MorphIt weight parameters."""

    def test_compute_metrics_returns_metrics(self, large_cuboid_mesh):
        """Test that compute_metrics=True populates result.metrics."""
        result = fit_spheres_to_mesh(
            large_cuboid_mesh,
            num_spheres=20,
            surface_radius=0.01,
            fit_type=SphereFitType.MORPHIT,
            iterations=50,
            compute_metrics=True,
        )

        assert result.metrics is not None
        assert result.metrics.num_spheres == result.num_spheres
        assert 0.0 <= result.metrics.coverage <= 1.0
        assert 0.0 <= result.metrics.protrusion <= 1.0
        assert result.metrics.surface_gap_mean >= 0.0
        assert result.metrics.volume_ratio >= 0.0

    def test_no_metrics_by_default(self, large_cuboid_mesh):
        """Test that metrics are None when compute_metrics is not set."""
        result = fit_spheres_to_mesh(
            large_cuboid_mesh,
            num_spheres=20,
            surface_radius=0.01,
            fit_type=SphereFitType.MORPHIT,
            iterations=50,
        )

        assert result.metrics is None

    def test_coverage_weight_propagates(self, large_cuboid_mesh):
        """Test that coverage_weight affects the fit result."""
        result_low = fit_spheres_to_mesh(
            large_cuboid_mesh,
            num_spheres=15,
            fit_type=SphereFitType.MORPHIT,
            iterations=100,
            compute_metrics=True,
            coverage_weight=1.0,
        )
        result_high = fit_spheres_to_mesh(
            large_cuboid_mesh,
            num_spheres=15,
            fit_type=SphereFitType.MORPHIT,
            iterations=100,
            compute_metrics=True,
            coverage_weight=5000.0,
        )

        assert result_low.metrics is not None
        assert result_high.metrics is not None
        assert result_high.metrics.coverage >= result_low.metrics.coverage

    def test_protrusion_weight_propagates(self, large_cuboid_mesh):
        """Test that protrusion_weight affects the fit result."""
        result_low = fit_spheres_to_mesh(
            large_cuboid_mesh,
            num_spheres=15,
            fit_type=SphereFitType.MORPHIT,
            iterations=100,
            compute_metrics=True,
            protrusion_weight=0.1,
        )
        result_high = fit_spheres_to_mesh(
            large_cuboid_mesh,
            num_spheres=15,
            fit_type=SphereFitType.MORPHIT,
            iterations=100,
            compute_metrics=True,
            protrusion_weight=500.0,
        )

        assert result_low.metrics is not None
        assert result_high.metrics is not None
        assert result_high.metrics.protrusion <= result_low.metrics.protrusion

    def test_weights_ignored_for_non_morphit(self, large_cuboid_mesh):
        """Test that weight params don't break non-MORPHIT fit types."""
        for fit_type in [SphereFitType.SURFACE, SphereFitType.VOXEL]:
            result = fit_spheres_to_mesh(
                large_cuboid_mesh,
                num_spheres=15,
                surface_radius=0.01,
                fit_type=fit_type,
                coverage_weight=5000.0,
                protrusion_weight=500.0,
            )
            assert result.num_spheres > 0


class TestClipPlane:
    """Test half-plane clipping during sphere fitting."""

    def test_clip_plane_respects_boundary(self, simple_cuboid_mesh):
        """Spheres must not cross the clip plane (including buffer)."""
        clip_plane = ((0.0, 0.0, 1.0), 0.0)
        result = fit_spheres_to_mesh(
            simple_cuboid_mesh, num_spheres=15,
            fit_type=SphereFitType.MORPHIT,
            clip_plane=clip_plane,
        )
        assert result.num_spheres > 0
        buffer = 0.02
        z_clearance = result.centers[:, 2] - result.radii - buffer
        assert (z_clearance >= -1e-4).all(), (
            f"Spheres cross clip plane: min clearance = {z_clearance.min().item():.4f}"
        )

    def test_clip_plane_no_clip_when_none(self, simple_cuboid_mesh):
        """Without clip_plane, spheres may extend below z=0."""
        result = fit_spheres_to_mesh(
            simple_cuboid_mesh, num_spheres=15,
            fit_type=SphereFitType.MORPHIT,
        )
        assert result.num_spheres > 0
        z_bottom = result.centers[:, 2] - result.radii
        assert (z_bottom < 0.0).any(), "Expected some spheres below z=0 without clipping"

    def test_clip_plane_negative_axis(self, simple_cuboid_mesh):
        """Clip plane with negative normal direction."""
        clip_plane = ((0.0, 0.0, -1.0), 0.0)
        result = fit_spheres_to_mesh(
            simple_cuboid_mesh, num_spheres=15,
            fit_type=SphereFitType.MORPHIT,
            clip_plane=clip_plane,
        )
        assert result.num_spheres > 0
        buffer = 0.02
        neg_z_clearance = -result.centers[:, 2] - result.radii - buffer
        assert (neg_z_clearance >= -1e-4).all()

    def test_clip_plane_x_axis(self, large_cuboid_mesh):
        """Clip plane along x-axis."""
        clip_plane = ((1.0, 0.0, 0.0), 0.0)
        result = fit_spheres_to_mesh(
            large_cuboid_mesh, num_spheres=20,
            fit_type=SphereFitType.MORPHIT,
            clip_plane=clip_plane,
        )
        assert result.num_spheres > 0
        buffer = 0.02
        x_clearance = result.centers[:, 0] - result.radii - buffer
        assert (x_clearance >= -1e-4).all()

    def test_clip_plane_with_voxel_fallback(self, simple_cuboid_mesh):
        """Clip plane hard-clamp works for non-MORPHIT fit types."""
        clip_plane = ((0.0, 0.0, 1.0), 0.0)
        result = fit_spheres_to_mesh(
            simple_cuboid_mesh, num_spheres=15,
            fit_type=SphereFitType.VOXEL,
            clip_plane=clip_plane,
        )
        assert result.num_spheres > 0
        buffer = 0.02
        z_clearance = result.centers[:, 2] - result.radii - buffer
        assert (z_clearance >= -1e-4).all()

    def test_clip_plane_preserves_sphere_count(self, large_cuboid_mesh):
        """Clipping should not drastically reduce the number of spheres."""
        result_no_clip = fit_spheres_to_mesh(
            large_cuboid_mesh, num_spheres=20,
            fit_type=SphereFitType.MORPHIT,
        )
        clip_plane = ((0.0, 0.0, 1.0), -0.1)
        result_clip = fit_spheres_to_mesh(
            large_cuboid_mesh, num_spheres=20,
            fit_type=SphereFitType.MORPHIT,
            clip_plane=clip_plane,
        )
        assert result_clip.num_spheres >= result_no_clip.num_spheres * 0.5, (
            f"Clipping removed too many spheres: {result_clip.num_spheres} vs {result_no_clip.num_spheres}"
        )


class TestSphereFitEdgeCases:
    """Test edge cases and error handling."""

    def test_fit_very_small_count(self, simple_cuboid_mesh):
        """Test fitting very small number of spheres."""
        result = fit_spheres_to_mesh(simple_cuboid_mesh, 1, 0.01)

        assert result.centers is not None
        assert result.num_spheres >= 1

    def test_fit_different_mesh_scales(self):
        """Test fitting on meshes of different scales."""
        for scale in [0.05, 0.1, 0.5, 1.0]:
            cuboid = Cuboid(
                name="test_cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[scale, scale, scale]
            )
            mesh = cuboid.get_trimesh_mesh()
            result = fit_spheres_to_mesh(mesh, 15, 0.005)

            assert result.centers is not None
            assert result.num_spheres > 0

    def test_fit_non_uniform_cuboid(self):
        """Test fitting on non-uniform cuboid."""
        cuboid = Cuboid(name="test_cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.5, 0.2, 0.1])
        mesh = cuboid.get_trimesh_mesh()
        result = fit_spheres_to_mesh(mesh, 25, 0.01)

        assert result.centers is not None
        assert result.num_spheres > 0

    def test_sphere_fit_consistency(self, simple_cuboid_mesh):
        """Test that fitting produces consistent results."""
        r1 = fit_spheres_to_mesh(
            simple_cuboid_mesh, 20, 0.01, fit_type=SphereFitType.SURFACE
        )
        r2 = fit_spheres_to_mesh(
            simple_cuboid_mesh, 20, 0.01, fit_type=SphereFitType.SURFACE
        )

        assert r1.centers is not None
        assert r2.centers is not None
        assert r1.num_spheres > 0
        assert r2.num_spheres > 0


class TestSphereFitIntegration:
    """Integration tests combining multiple sphere fit operations."""

    def test_multiple_fit_types_on_same_mesh(self, simple_cuboid_mesh):
        """Test applying multiple fit types to the same mesh."""
        num_spheres = 20
        surface_radius = 0.01

        results = {}
        for fit_type in [
            SphereFitType.SURFACE,
            SphereFitType.VOXEL,
        ]:
            result = fit_spheres_to_mesh(
                simple_cuboid_mesh, num_spheres, surface_radius, fit_type=fit_type
            )
            results[fit_type.value] = result

        for fit_type, result in results.items():
            assert result.centers is not None, f"{fit_type} failed"
            assert result.num_spheres > 0, f"{fit_type} produced no points"

    def test_increasing_sphere_counts(self, large_cuboid_mesh):
        """Test fitting with progressively increasing sphere counts."""
        counts = [10, 20, 40, 80]

        for count in counts:
            result = fit_spheres_to_mesh(large_cuboid_mesh, count, 0.01)
            assert result.centers is not None
            assert result.num_spheres > 0

    def test_different_geometries_same_parameters(self):
        """Test same fitting parameters on different geometric shapes."""
        shapes = [
            Cuboid(name="cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.2, 0.2, 0.2]),
            Sphere(name="sphere", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.15),
            Cylinder(name="cylinder", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.1, height=0.3),
        ]

        for shape in shapes:
            mesh = shape.get_trimesh_mesh()
            result = fit_spheres_to_mesh(mesh, 20, 0.01)

            assert result.centers is not None, f"Failed for {shape.name}"
            assert result.num_spheres > 0, f"No points for {shape.name}"
            assert len(result.radii) == result.num_spheres, f"Mismatch for {shape.name}"


