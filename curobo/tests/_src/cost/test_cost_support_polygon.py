# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for CostSupportPolygon and CostSupportPolygonCfg."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_support_polygon import CostSupportPolygon
from curobo._src.cost.cost_support_polygon_cfg import CostSupportPolygonCfg
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def device_cfg():
    """Create tensor configuration for CPU."""
    return DeviceCfg(device=torch.device("cpu"))


@pytest.fixture
def device_cfg_cuda():
    """Create tensor configuration for GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device("cuda:0"))


class TestCostSupportPolygonCfg:
    """Test CostSupportPolygonCfg class."""

    def test_default_init(self, device_cfg):
        """Test default initialization."""
        cfg = CostSupportPolygonCfg(weight=1.0, device_cfg=device_cfg)
        assert cfg.class_type == CostSupportPolygon
        assert cfg.foot_sphere_indices is None
        assert cfg.foot_link_names is None
        assert cfg.inside_cost_weight == 0.001

    def test_init_with_foot_sphere_indices(self, device_cfg):
        """Test initialization with foot_sphere_indices."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_sphere_indices=indices
        )
        assert torch.equal(cfg.foot_sphere_indices, indices)

    def test_init_with_foot_link_names(self, device_cfg):
        """Test initialization with foot_link_names."""
        link_names = ["foot_1", "foot_2", "foot_3", "foot_4"]
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_link_names=link_names
        )
        assert cfg.foot_link_names == link_names

    def test_init_with_custom_inside_cost_weight(self, device_cfg):
        """Test initialization with custom inside_cost_weight."""
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, inside_cost_weight=0.01
        )
        assert cfg.inside_cost_weight == 0.01


class TestCostSupportPolygon:
    """Test CostSupportPolygon class."""

    def test_init(self, device_cfg):
        """Test CostSupportPolygon initialization."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_sphere_indices=indices
        )
        cost = CostSupportPolygon(cfg)
        assert cost is not None
        assert cost._polygon_helper is not None

    def test_build_convex_hull(self, device_cfg):
        """Test build_convex_hull method."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_sphere_indices=indices
        )
        cost = CostSupportPolygon(cfg)

        # Create vertices for a square
        batch_size = 2
        vertices = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            ],
            **device_cfg.as_torch_dict(),
        )

        cost.build_convex_hull(vertices)
        assert cost._polygon_helper._cached_convex_hulls is not None

    def test_build_convex_hull_with_padding(self, device_cfg):
        """Test build_convex_hull with padding."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_sphere_indices=indices
        )
        cost = CostSupportPolygon(cfg)

        # Create vertices for a square
        batch_size = 2
        vertices = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            ],
            **device_cfg.as_torch_dict(),
        )

        cost.build_convex_hull(vertices, padding=0.1)
        assert cost._polygon_helper._cached_convex_hulls is not None

    def test_forward_com_inside_polygon(self, device_cfg):
        """Test forward with CoM inside the support polygon."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_sphere_indices=indices
        )
        cost = CostSupportPolygon(cfg)

        batch_size = 2
        horizon = 5
        num_spheres = 10

        # Create robot spheres (first 4 are foot spheres defining a square)
        robot_spheres = torch.zeros(
            batch_size, horizon, num_spheres, 4, **device_cfg.as_torch_dict()
        )
        # Set foot positions to create a 1x1 square at origin
        robot_spheres[:, :, 0, :2] = torch.tensor([0.0, 0.0])
        robot_spheres[:, :, 1, :2] = torch.tensor([1.0, 0.0])
        robot_spheres[:, :, 2, :2] = torch.tensor([1.0, 1.0])
        robot_spheres[:, :, 3, :2] = torch.tensor([0.0, 1.0])
        robot_spheres[:, :, :, 3] = 0.05  # radius

        # Set CoM at center of square (inside polygon)
        robot_com = torch.zeros(batch_size, horizon, 3, **device_cfg.as_torch_dict())
        robot_com[:, :, 0] = 0.5  # x
        robot_com[:, :, 1] = 0.5  # y
        robot_com[:, :, 2] = 0.0  # z

        result = cost.forward(robot_com, robot_spheres)
        assert result.shape == (batch_size, horizon)
        # CoM inside polygon should have low or zero cost (when inside_cost_weight is small)
        assert torch.all(result <= 0.1)

    def test_forward_com_outside_polygon(self, device_cfg):
        """Test forward with CoM outside the support polygon."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_sphere_indices=indices
        )
        cost = CostSupportPolygon(cfg)

        batch_size = 2
        horizon = 5
        num_spheres = 10

        # Create robot spheres (first 4 are foot spheres defining a square)
        robot_spheres = torch.zeros(
            batch_size, horizon, num_spheres, 4, **device_cfg.as_torch_dict()
        )
        # Set foot positions to create a 1x1 square at origin
        robot_spheres[:, :, 0, :2] = torch.tensor([0.0, 0.0])
        robot_spheres[:, :, 1, :2] = torch.tensor([1.0, 0.0])
        robot_spheres[:, :, 2, :2] = torch.tensor([1.0, 1.0])
        robot_spheres[:, :, 3, :2] = torch.tensor([0.0, 1.0])
        robot_spheres[:, :, :, 3] = 0.05  # radius

        # Set CoM outside the square
        robot_com = torch.zeros(batch_size, horizon, 3, **device_cfg.as_torch_dict())
        robot_com[:, :, 0] = 2.0  # x (outside the square)
        robot_com[:, :, 1] = 2.0  # y (outside the square)
        robot_com[:, :, 2] = 0.0  # z

        result = cost.forward(robot_com, robot_spheres)
        assert result.shape == (batch_size, horizon)
        # CoM outside polygon should have positive cost
        assert torch.all(result > 0)

    def test_forward_with_inside_cost_weight(self, device_cfg):
        """Test forward with inside_cost_weight > 0."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_sphere_indices=indices, inside_cost_weight=0.1
        )
        cost = CostSupportPolygon(cfg)

        batch_size = 2
        horizon = 5
        num_spheres = 10

        # Create robot spheres (first 4 are foot spheres defining a square)
        robot_spheres = torch.zeros(
            batch_size, horizon, num_spheres, 4, **device_cfg.as_torch_dict()
        )
        # Set foot positions to create a 1x1 square at origin
        robot_spheres[:, :, 0, :2] = torch.tensor([0.0, 0.0])
        robot_spheres[:, :, 1, :2] = torch.tensor([1.0, 0.0])
        robot_spheres[:, :, 2, :2] = torch.tensor([1.0, 1.0])
        robot_spheres[:, :, 3, :2] = torch.tensor([0.0, 1.0])
        robot_spheres[:, :, :, 3] = 0.05  # radius

        # Set CoM at center of square (inside polygon)
        robot_com = torch.zeros(batch_size, horizon, 3, **device_cfg.as_torch_dict())
        robot_com[:, :, 0] = 0.5  # x
        robot_com[:, :, 1] = 0.5  # y
        robot_com[:, :, 2] = 0.0  # z

        result = cost.forward(robot_com, robot_spheres)
        assert result.shape == (batch_size, horizon)
        # With inside_cost_weight > 0, there may be some cost even inside

    def test_forward_different_batch_sizes(self, device_cfg):
        """Test forward with different batch sizes."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_sphere_indices=indices
        )

        for batch_size in [1, 2, 8]:
            cost = CostSupportPolygon(cfg)
            horizon = 5
            num_spheres = 10

            robot_spheres = torch.zeros(
                batch_size, horizon, num_spheres, 4, **device_cfg.as_torch_dict()
            )
            robot_spheres[:, :, 0, :2] = torch.tensor([0.0, 0.0])
            robot_spheres[:, :, 1, :2] = torch.tensor([1.0, 0.0])
            robot_spheres[:, :, 2, :2] = torch.tensor([1.0, 1.0])
            robot_spheres[:, :, 3, :2] = torch.tensor([0.0, 1.0])

            robot_com = torch.zeros(batch_size, horizon, 3, **device_cfg.as_torch_dict())
            robot_com[:, :, :2] = 0.5

            result = cost.forward(robot_com, robot_spheres)
            assert result.shape == (batch_size, horizon)

    def test_forward_different_horizons(self, device_cfg):
        """Test forward with different horizons."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_sphere_indices=indices
        )

        batch_size = 2
        num_spheres = 10

        for horizon in [1, 5, 20]:
            cost = CostSupportPolygon(cfg)

            robot_spheres = torch.zeros(
                batch_size, horizon, num_spheres, 4, **device_cfg.as_torch_dict()
            )
            robot_spheres[:, :, 0, :2] = torch.tensor([0.0, 0.0])
            robot_spheres[:, :, 1, :2] = torch.tensor([1.0, 0.0])
            robot_spheres[:, :, 2, :2] = torch.tensor([1.0, 1.0])
            robot_spheres[:, :, 3, :2] = torch.tensor([0.0, 1.0])

            robot_com = torch.zeros(batch_size, horizon, 3, **device_cfg.as_torch_dict())
            robot_com[:, :, :2] = 0.5

            result = cost.forward(robot_com, robot_spheres)
            assert result.shape == (batch_size, horizon)


class TestCostSupportPolygonCuda:
    """Test CostSupportPolygon on CUDA."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_cuda(self, device_cfg_cuda):
        """Test forward on CUDA."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device="cuda:0")
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg_cuda, foot_sphere_indices=indices
        )
        cost = CostSupportPolygon(cfg)

        batch_size = 4
        horizon = 10
        num_spheres = 10

        robot_spheres = torch.zeros(
            batch_size, horizon, num_spheres, 4, **device_cfg_cuda.as_torch_dict()
        )
        robot_spheres[:, :, 0, :2] = torch.tensor([0.0, 0.0], device="cuda:0")
        robot_spheres[:, :, 1, :2] = torch.tensor([1.0, 0.0], device="cuda:0")
        robot_spheres[:, :, 2, :2] = torch.tensor([1.0, 1.0], device="cuda:0")
        robot_spheres[:, :, 3, :2] = torch.tensor([0.0, 1.0], device="cuda:0")

        robot_com = torch.zeros(batch_size, horizon, 3, **device_cfg_cuda.as_torch_dict())
        robot_com[:, :, 0] = 0.5
        robot_com[:, :, 1] = 0.5

        result = cost.forward(robot_com, robot_spheres)
        assert result.shape == (batch_size, horizon)
        assert result.device.type == "cuda"


class TestConvexPolygon2DHelper:
    """Test ConvexPolygon2DHelper used by CostSupportPolygon."""

    def test_build_convex_hull_square(self, device_cfg):
        """Test convex hull computation for square vertices."""
        # CuRobo
        from curobo._src.geom.convex_polygon_helper import ConvexPolygon2DHelper

        helper = ConvexPolygon2DHelper(device_cfg)

        # Create a square
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            **device_cfg.as_torch_dict(),
        )

        helper.build_convex_hull(vertices)
        assert helper._cached_convex_hulls is not None
        assert helper._cached_convex_hulls.shape[0] == 1  # batch size

    def test_build_convex_hull_triangle(self, device_cfg):
        """Test convex hull computation for triangle vertices."""
        # CuRobo
        from curobo._src.geom.convex_polygon_helper import ConvexPolygon2DHelper

        helper = ConvexPolygon2DHelper(device_cfg)

        # Create a triangle
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]],
            **device_cfg.as_torch_dict(),
        )

        helper.build_convex_hull(vertices)
        assert helper._cached_convex_hulls is not None

    def test_compute_point_hull_distance_inside(self, device_cfg):
        """Test distance computation for point inside hull."""
        # CuRobo
        from curobo._src.geom.convex_polygon_helper import ConvexPolygon2DHelper

        helper = ConvexPolygon2DHelper(device_cfg)

        # Create a square
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            **device_cfg.as_torch_dict(),
        )

        helper.build_convex_hull(vertices)

        # Point inside the square
        points = torch.tensor([[[[0.5, 0.5]]]], **device_cfg.as_torch_dict())  # [1, 1, 1, 2]
        batch_indices = torch.tensor([0], device=device_cfg.device)

        distances = helper.compute_point_hull_distance(points, batch_indices)
        assert distances.shape == (1, 1, 1)
        # Inside point should have negative signed distance
        assert torch.all(distances < 0)

    def test_compute_point_hull_distance_outside(self, device_cfg):
        """Test distance computation for point outside hull."""
        # CuRobo
        from curobo._src.geom.convex_polygon_helper import ConvexPolygon2DHelper

        helper = ConvexPolygon2DHelper(device_cfg)

        # Create a square
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            **device_cfg.as_torch_dict(),
        )

        helper.build_convex_hull(vertices)

        # Point outside the square
        points = torch.tensor([[[[2.0, 2.0]]]], **device_cfg.as_torch_dict())  # [1, 1, 1, 2]
        batch_indices = torch.tensor([0], device=device_cfg.device)

        distances = helper.compute_point_hull_distance(points, batch_indices)
        assert distances.shape == (1, 1, 1)
        # Outside point should have positive signed distance
        assert torch.all(distances > 0)

    def test_compute_point_hull_distance_batched(self, device_cfg):
        """Test batched distance computation."""
        # CuRobo
        from curobo._src.geom.convex_polygon_helper import ConvexPolygon2DHelper

        helper = ConvexPolygon2DHelper(device_cfg)

        # Create two squares
        vertices = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            ],
            **device_cfg.as_torch_dict(),
        )

        helper.build_convex_hull(vertices)

        batch_size = 2
        horizon = 3
        n_points = 2
        # Create points
        points = torch.zeros(batch_size, horizon, n_points, 2, **device_cfg.as_torch_dict())
        points[0, :, 0, :] = torch.tensor([0.5, 0.5])  # Inside first square
        points[0, :, 1, :] = torch.tensor([3.0, 3.0])  # Outside first square
        points[1, :, 0, :] = torch.tensor([1.0, 1.0])  # Inside second square
        points[1, :, 1, :] = torch.tensor([5.0, 5.0])  # Outside second square

        batch_indices = torch.tensor([0, 1], device=device_cfg.device)

        distances = helper.compute_point_hull_distance(points, batch_indices)
        assert distances.shape == (batch_size, horizon, n_points)


class TestCostSupportPolygonGradients:
    """Test gradient computation for CostSupportPolygon.

    Note: CostSupportPolygon uses pure PyTorch operations for convex hull computation,
    so gradients flow naturally through the computation.
    """

    def test_support_polygon_gradient_com(self, device_cfg):
        """Test gradient computation for CostSupportPolygon w.r.t. robot_com."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_sphere_indices=indices
        )
        cost = CostSupportPolygon(cfg)

        batch_size = 4
        horizon = 10
        num_spheres = 10

        # Create robot spheres (fixed, not requiring grad)
        robot_spheres = torch.zeros(batch_size, horizon, num_spheres, 4, **device_cfg.as_torch_dict())
        robot_spheres[:, :, 0, :2] = torch.tensor([0.0, 0.0])
        robot_spheres[:, :, 1, :2] = torch.tensor([1.0, 0.0])
        robot_spheres[:, :, 2, :2] = torch.tensor([1.0, 1.0])
        robot_spheres[:, :, 3, :2] = torch.tensor([0.0, 1.0])

        # Create robot_com with requires_grad=True (outside polygon for non-zero cost)
        # Create as leaf tensor first, then add offset
        robot_com_base = torch.randn(
            batch_size, horizon, 3, dtype=torch.float32, device=device_cfg.device
        )
        robot_com = (robot_com_base + 2.0).requires_grad_(True)  # Offset to be outside the polygon
        robot_com.retain_grad()  # Retain grad for non-leaf tensor

        # Forward pass
        result = cost.forward(robot_com, robot_spheres)

        # Backward pass
        loss = result.sum()
        loss.backward()

        # Verify gradients exist and are finite
        assert robot_com.grad is not None, "CoM gradient should not be None"
        assert robot_com.grad.shape == robot_com.shape, "Gradient shape mismatch"
        assert torch.isfinite(robot_com.grad).all(), "Gradient contains non-finite values"

    def test_support_polygon_gradcheck(self, device_cfg):
        """Test numerical gradient check for CostSupportPolygon."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cfg = CostSupportPolygonCfg(
            weight=1.0, device_cfg=device_cfg, foot_sphere_indices=indices
        )
        cost = CostSupportPolygon(cfg)

        batch_size = 2
        horizon = 3
        num_spheres = 10

        # Create robot spheres (fixed)
        robot_spheres = torch.zeros(batch_size, horizon, num_spheres, 4, **device_cfg.as_torch_dict())
        robot_spheres[:, :, 0, :2] = torch.tensor([0.0, 0.0])
        robot_spheres[:, :, 1, :2] = torch.tensor([1.0, 0.0])
        robot_spheres[:, :, 2, :2] = torch.tensor([1.0, 1.0])
        robot_spheres[:, :, 3, :2] = torch.tensor([0.0, 1.0])

        # Pre-build convex hull
        cost.build_convex_hull(robot_spheres[:, 0, :4, :2].detach())

        # Define wrapper function for gradcheck
        def cost_fn(com):
            result = cost.forward(com, robot_spheres)
            return result.sum()

        # Create robot_com tensor (outside polygon for non-zero gradients)
        robot_com = torch.randn(
            batch_size, horizon, 3, dtype=torch.float32, device=device_cfg.device, requires_grad=True
        ) + 2.0

        # Numerical gradient check with relaxed tolerances for float32
        assert torch.autograd.gradcheck(
            cost_fn, robot_com, eps=1e-3, atol=1e-2, rtol=1e-2, raise_exception=True
        ), "Gradient check failed for CostSupportPolygon"

