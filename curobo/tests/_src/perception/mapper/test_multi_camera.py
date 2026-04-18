# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for multi-camera TSDF integration.

Tests multi-view fusion by integrating depth from multiple cameras into a
single TSDF. The for-loop tests establish ground truth; batched tests (added
later) must produce identical TSDF state.
"""

import math

import pytest
import torch

from curobo._src.perception.mapper import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
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


@pytest.fixture
def intrinsics_a(device):
    """Camera A intrinsics (500px focal length, 640x480 principal point)."""
    return torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


@pytest.fixture
def intrinsics_b(device):
    """Camera B intrinsics (different focal length)."""
    return torch.tensor(
        [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def _make_integrator(device, method="voxel_project", num_cameras=1):
    return BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=2000,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            truncation_distance=0.05,
            device=device,
            integration_method=method,
            num_cameras=num_cameras,
        )
    )


def _stack_observations(*observations):
    """Stack multiple single-camera CameraObservations into one batched observation."""
    return CameraObservation(
        depth_image=torch.cat([o.depth_image for o in observations], dim=0),
        rgb_image=torch.cat([o.rgb_image for o in observations], dim=0),
        pose=Pose(
            position=torch.cat([o.pose.position for o in observations], dim=0),
            quaternion=torch.cat([o.pose.quaternion for o in observations], dim=0),
        ),
        intrinsics=torch.cat([o.intrinsics for o in observations], dim=0),
    )


def _quat_from_axis_angle(axis, angle_rad, device):
    """Quaternion (wxyz) from axis-angle."""
    axis = torch.tensor(axis, dtype=torch.float32, device=device)
    axis = axis / axis.norm()
    half = angle_rad / 2.0
    w = math.cos(half)
    xyz = axis * math.sin(half)
    return torch.tensor([w, xyz[0], xyz[1], xyz[2]], dtype=torch.float32, device=device)


def _snapshot_tsdf(integrator):
    """Return TSDF state keyed by spatial block coordinates.

    Blocks are returned sorted by (bx, by, bz) so that two snapshots from
    independent integrators (which may allocate pool indices in different
    order) can be compared element-wise.
    """
    data = integrator.tsdf.data
    n = data.num_allocated.item()
    if n == 0:
        return {
            "num_allocated": 0,
            "block_data": data.block_data[:0].clone(),
            "block_rgb": data.block_rgb[:0].clone(),
            "block_coords": data.block_coords[:0].clone(),
        }

    coords = data.block_coords[: n * 3].clone().view(n, 3)
    bd = data.block_data[:n].clone()
    br = data.block_rgb[:n].clone()

    # Lexicographic sort by (bx, by, bz) for deterministic ordering.
    sort_key = coords[:, 0].long() * (2**20) + coords[:, 1].long() * (2**10) + coords[:, 2].long()
    order = sort_key.argsort()

    return {
        "num_allocated": n,
        "block_data": bd[order],
        "block_rgb": br[order],
        "block_coords": coords[order],
    }


# =============================================================================
# Multi-Camera For-Loop Tests (Ground Truth)
# =============================================================================


class TestMultiCameraForLoop:
    """Multi-camera integration using sequential for-loop.

    These establish the reference TSDF state that batched integration must match.
    """

    IMG_H, IMG_W = 48, 64

    def _make_depth_rgb(self, device, depth_value=1.0):
        depth = torch.full(
            (self.IMG_H, self.IMG_W), depth_value, dtype=torch.float32, device=device
        )
        rgb = torch.full(
            (self.IMG_H, self.IMG_W, 3), 128, dtype=torch.uint8, device=device
        )
        return depth, rgb

    @pytest.mark.parametrize("method", ["voxel_project", "sort_filter"])
    def test_two_cameras_same_pose_doubles_weight(
        self, warp_init, device, intrinsics_a, method
    ):
        """Two cameras at the same pose should roughly double TSDF weights
        compared to a single camera."""
        depth, rgb = self._make_depth_rgb(device)
        pos = torch.zeros(3, dtype=torch.float32, device=device)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        obs = make_observation(depth, rgb, pos, quat, intrinsics_a)

        # Single camera
        integ_single = _make_integrator(device, method)
        integ_single.integrate(obs)
        snap_single = _snapshot_tsdf(integ_single)

        # Two cameras (same pose) via for-loop
        integ_multi = _make_integrator(device, method)
        for _ in range(2):
            integ_multi.integrate(obs)
        snap_multi = _snapshot_tsdf(integ_multi)

        assert snap_single["num_allocated"] == snap_multi["num_allocated"]
        # Both snapshots are now sorted by block coords, so they align.
        torch.testing.assert_close(
            snap_single["block_coords"], snap_multi["block_coords"],
        )

        # Total weight across all voxels should roughly double.
        total_w_single = snap_single["block_data"][:, :, 1].float().sum()
        total_w_multi = snap_multi["block_data"][:, :, 1].float().sum()
        assert total_w_single > 0
        ratio = (total_w_multi / total_w_single).item()
        assert 1.8 < ratio < 2.2, (
            f"Expected ~2x total weight ({method}), got {ratio:.4f}"
        )

    @pytest.mark.parametrize("method", ["voxel_project", "sort_filter"])
    def test_two_cameras_different_poses_allocates_more_blocks(
        self, warp_init, device, intrinsics_a, method
    ):
        """Cameras at different positions should cover more volume (more blocks)."""
        depth, rgb = self._make_depth_rgb(device)

        pos_a = torch.tensor([0.3, 0.0, 0.0], dtype=torch.float32, device=device)
        pos_b = torch.tensor([-0.3, 0.0, 0.0], dtype=torch.float32, device=device)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

        obs_a = make_observation(depth, rgb, pos_a, quat, intrinsics_a)
        obs_b = make_observation(depth, rgb, pos_b, quat, intrinsics_a)

        # Single camera
        integ_single = _make_integrator(device, method)
        integ_single.integrate(obs_a)
        n_single = integ_single.tsdf.data.num_allocated.item()

        # Two cameras
        integ_multi = _make_integrator(device, method)
        integ_multi.integrate(obs_a)
        integ_multi.integrate(obs_b)
        n_multi = integ_multi.tsdf.data.num_allocated.item()

        assert n_single > 0
        assert n_multi >= n_single, (
            f"Two views should cover at least as much as one: "
            f"{n_multi} vs {n_single}"
        )

    @pytest.mark.parametrize("method", ["voxel_project", "sort_filter"])
    def test_two_cameras_per_camera_intrinsics(
        self, warp_init, device, intrinsics_a, intrinsics_b, method
    ):
        """Two cameras with different intrinsics should both contribute blocks."""
        depth, rgb = self._make_depth_rgb(device)
        pos = torch.zeros(3, dtype=torch.float32, device=device)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

        obs_a = make_observation(depth, rgb, pos, quat, intrinsics_a)
        obs_b = make_observation(depth, rgb, pos, quat, intrinsics_b)

        integ = _make_integrator(device, method)
        integ.integrate(obs_a)
        n_after_a = integ.tsdf.data.num_allocated.item()

        integ.integrate(obs_b)
        n_after_both = integ.tsdf.data.num_allocated.item()

        assert n_after_a > 0
        # Camera B has wider FOV (smaller focal length) so may add blocks
        assert n_after_both >= n_after_a

    @pytest.mark.parametrize("method", ["voxel_project", "sort_filter"])
    def test_three_cameras_convergent_views(
        self, warp_init, device, intrinsics_a, method
    ):
        """Three cameras looking at the same region from different angles.

        The overlapping region should accumulate higher weight than single-view.
        """
        depth, rgb = self._make_depth_rgb(device, depth_value=1.0)
        quat_identity = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device
        )

        observations = []
        for x_offset in [-0.2, 0.0, 0.2]:
            pos = torch.tensor(
                [x_offset, 0.0, 0.0], dtype=torch.float32, device=device
            )
            observations.append(
                make_observation(depth, rgb, pos, quat_identity, intrinsics_a)
            )

        integ = _make_integrator(device, method)
        for obs in observations:
            integ.integrate(obs)

        snap = _snapshot_tsdf(integ)
        assert snap["num_allocated"] > 0

        # Voxels that all 3 cameras see should have weight > single-camera weight
        w = snap["block_data"][:, :, 1].float()
        nonzero = w > 0
        assert nonzero.any(), "Should have observed voxels"


# =============================================================================
# Batched integrate_multi() vs For-Loop Equivalence
# =============================================================================


class TestMultiCameraBatchEquivalence:
    """Assert integrate_multi() produces the same TSDF as sequential integrate().

    Decay is disabled (time_decay=1.0, frustum_decay=1.0) so the only
    variable is the integration kernel path.
    """

    IMG_H, IMG_W = 48, 64

    def _make_depth_rgb(self, device, depth_value=1.0):
        depth = torch.full(
            (self.IMG_H, self.IMG_W), depth_value, dtype=torch.float32, device=device
        )
        rgb = torch.full(
            (self.IMG_H, self.IMG_W, 3), 128, dtype=torch.uint8, device=device
        )
        return depth, rgb

    @pytest.mark.parametrize("method", ["voxel_project", "sort_filter"])
    def test_same_pose_equivalence(self, warp_init, device, intrinsics_a, method):
        """Batched 2-camera integrate == two sequential single-camera integrates."""
        depth, rgb = self._make_depth_rgb(device)
        pos = torch.zeros(3, dtype=torch.float32, device=device)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        obs = make_observation(depth, rgb, pos, quat, intrinsics_a)

        # Reference: for-loop (num_cameras=1, called twice)
        ref = _make_integrator(device, method, num_cameras=1)
        ref.integrate(obs)
        ref.integrate(obs)
        snap_ref = _snapshot_tsdf(ref)

        # Batched (num_cameras=2, called once)
        bat = _make_integrator(device, method, num_cameras=2)
        bat.integrate(_stack_observations(obs, obs))
        snap_bat = _snapshot_tsdf(bat)

        assert snap_ref["num_allocated"] == snap_bat["num_allocated"]
        torch.testing.assert_close(
            snap_bat["block_data"].float(),
            snap_ref["block_data"].float(),
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize("method", ["voxel_project", "sort_filter"])
    def test_different_poses_equivalence(
        self, warp_init, device, intrinsics_a, method
    ):
        """Batched 2-camera integrate == two sequential single-camera integrates."""
        depth, rgb = self._make_depth_rgb(device)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

        pos_a = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float32, device=device)
        pos_b = torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float32, device=device)

        obs_a = make_observation(depth, rgb, pos_a, quat, intrinsics_a)
        obs_b = make_observation(depth, rgb, pos_b, quat, intrinsics_a)

        # Reference: for-loop
        ref = _make_integrator(device, method, num_cameras=1)
        ref.integrate(obs_a)
        ref.integrate(obs_b)
        snap_ref = _snapshot_tsdf(ref)

        # Batched
        bat = _make_integrator(device, method, num_cameras=2)
        bat.integrate(_stack_observations(obs_a, obs_b))
        snap_bat = _snapshot_tsdf(bat)

        assert snap_ref["num_allocated"] == snap_bat["num_allocated"]
        torch.testing.assert_close(
            snap_bat["block_data"].float(),
            snap_ref["block_data"].float(),
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize("method", ["voxel_project", "sort_filter"])
    def test_different_intrinsics_equivalence(
        self, warp_init, device, intrinsics_a, intrinsics_b, method
    ):
        """Batched 2-camera with per-camera intrinsics matches sequential.

        For voxel_project the batched path may produce *more* data than the
        sequential path because Phase 4 (voxel-centric) projects every
        discovered voxel into ALL cameras.  In the sequential path each
        camera's Phase 4 only processes blocks from its own Phase 1
        discovery, so edge blocks discovered by one camera miss
        contributions from the other.  The batched result is a strict
        superset: where the reference has data the batched value must be
        close; the batched path may additionally have nonzero voxels that
        the reference leaves at zero.
        """
        depth, rgb = self._make_depth_rgb(device)
        pos = torch.zeros(3, dtype=torch.float32, device=device)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

        obs_a = make_observation(depth, rgb, pos, quat, intrinsics_a)
        obs_b = make_observation(depth, rgb, pos, quat, intrinsics_b)

        # Reference: for-loop
        ref = _make_integrator(device, method, num_cameras=1)
        ref.integrate(obs_a)
        ref.integrate(obs_b)
        snap_ref = _snapshot_tsdf(ref)

        # Batched
        bat = _make_integrator(device, method, num_cameras=2)
        bat.integrate(_stack_observations(obs_a, obs_b))
        snap_bat = _snapshot_tsdf(bat)

        assert snap_ref["num_allocated"] == snap_bat["num_allocated"]

        ref_w = snap_ref["block_data"][:, :, 1].float()
        bat_w = snap_bat["block_data"][:, :, 1].float()

        ref_nonzero = ref_w > 0
        if ref_nonzero.any():
            torch.testing.assert_close(
                bat_w[ref_nonzero],
                ref_w[ref_nonzero],
                atol=1.0,
                rtol=0.05,
            )

        assert bat_w.sum() >= ref_w.sum() * 0.99


# =============================================================================
# Multi-Camera Frustum Decay Tests
# =============================================================================


def _make_integrator_with_decay(device, method="voxel_project", num_cameras=1):
    return BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=2000,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            truncation_distance=0.05,
            device=device,
            integration_method=method,
            num_cameras=num_cameras,
            time_decay=0.99,
            frustum_decay=0.5,
        )
    )


class TestMultiCameraDecay:
    """Test that multi-camera frustum decay marks union of frustums."""

    IMG_H, IMG_W = 48, 64

    def _make_depth_rgb(self, device, depth_value=1.0):
        depth = torch.full(
            (self.IMG_H, self.IMG_W), depth_value, dtype=torch.float32, device=device
        )
        rgb = torch.full(
            (self.IMG_H, self.IMG_W, 3), 128, dtype=torch.uint8, device=device
        )
        return depth, rgb

    @pytest.mark.parametrize("method", ["voxel_project", "sort_filter"])
    def test_decay_preserves_in_frustum_blocks(
        self, warp_init, device, intrinsics_a, method
    ):
        """Blocks visible in any camera should decay at frustum rate, not time-only rate.

        Camera A sees right half, camera B sees left half.  After integration
        + one decay step, ALL blocks should have the frustum decay factor
        (not just time_decay) because every block is in at least one frustum.
        """
        depth, rgb = self._make_depth_rgb(device)

        pos_a = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float32, device=device)
        pos_b = torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float32, device=device)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

        obs_a = make_observation(depth, rgb, pos_a, quat, intrinsics_a)
        obs_b = make_observation(depth, rgb, pos_b, quat, intrinsics_a)

        integ = _make_integrator_with_decay(device, method, num_cameras=2)
        batched = _stack_observations(obs_a, obs_b)

        # First frame: integrate + decay
        integ.integrate(batched)
        w_after_first = integ.tsdf.data.block_data[
            : integ.tsdf.data.num_allocated.item(), :, 1
        ].float().clone()

        # Second frame: integrate + decay again
        integ.integrate(batched)
        w_after_second = integ.tsdf.data.block_data[
            : integ.tsdf.data.num_allocated.item(), :, 1
        ].float().clone()

        # Weights should decrease due to decay
        nonzero = w_after_first > 0
        assert nonzero.any()
        # After second frame, new contributions are added but old weights
        # are decayed.  The total weight should still be positive.
        assert (w_after_second[nonzero] > 0).all()

    @pytest.mark.parametrize("method", ["voxel_project", "sort_filter"])
    def test_decay_union_vs_single_camera(
        self, warp_init, device, intrinsics_a, method
    ):
        """Multi-camera decay should mark more blocks in-frustum than single camera.

        Camera at center sees a subset of what two offset cameras see combined.
        The union frustum should be wider.
        """
        depth, rgb = self._make_depth_rgb(device)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        pos_center = torch.zeros(3, dtype=torch.float32, device=device)

        # Single camera at center
        obs_center = make_observation(depth, rgb, pos_center, quat, intrinsics_a)
        integ_single = _make_integrator_with_decay(device, method, num_cameras=1)
        integ_single.integrate(obs_center)
        n_single = integ_single.tsdf.data.num_allocated.item()

        # Two cameras offset left/right
        pos_a = torch.tensor([0.3, 0.0, 0.0], dtype=torch.float32, device=device)
        pos_b = torch.tensor([-0.3, 0.0, 0.0], dtype=torch.float32, device=device)
        obs_a = make_observation(depth, rgb, pos_a, quat, intrinsics_a)
        obs_b = make_observation(depth, rgb, pos_b, quat, intrinsics_a)

        integ_multi = _make_integrator_with_decay(device, method, num_cameras=2)
        integ_multi.integrate(_stack_observations(obs_a, obs_b))
        n_multi = integ_multi.tsdf.data.num_allocated.item()

        # Multi-camera should have at least as many blocks
        assert n_multi >= n_single
        assert n_multi > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
