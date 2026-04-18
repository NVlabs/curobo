# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Warp-accelerated mesh queries for the sphere fitting module.

Provides GPU-accelerated mesh queries used across all sphere fitting methods:

1. **Mesh containment** -- :meth:`WarpMeshQuery.query_outside_mask` returns a boolean
   mask indicating which query points lie outside the mesh.
2. **Signed distance with gradient** -- :meth:`WarpMeshQuery.query_sdf` computes the
   signed distance and analytic gradient from each query point to the mesh surface.
3. **Closest surface point** -- :meth:`WarpMeshQuery.query_closest_point` returns the
   nearest surface point and signed distance for each query point, replacing both
   ``trimesh.proximity.ProximityQuery.signed_distance`` and ``on_surface``.
4. **Differentiable SDF** -- :class:`WarpSphereSDFFunction` wraps :meth:`query_sdf`
   as a :class:`torch.autograd.Function` for use in MorphIt loss backpropagation.
"""

# Standard Library
from __future__ import annotations

# Third Party
import numpy as np
import torch
import trimesh
import warp as wp

# CuRobo
from curobo._src.util.warp import get_warp_device_stream, init_warp

# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _query_outside_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    outside_mask: wp.array(dtype=wp.bool),
    max_distance: wp.float32,
):
    """Mark each query point as *outside* (True) or *inside* (False) the mesh."""
    tid = wp.tid()
    point = query_points[tid]
    result = wp.mesh_query_point(mesh_id, point, max_distance)
    inside = False
    if result.result:
        inside = result.sign < 0.0  # negative sign == inside
    outside_mask[tid] = not inside


@wp.kernel
def _sdf_forward_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    out_sdf: wp.array(dtype=wp.float32),
    out_grad: wp.array(dtype=wp.vec3),
    max_distance: wp.float32,
):
    """Compute signed distance and gradient from each query point to the mesh surface.

    Convention: negative inside, positive outside (same as Warp ``mesh_query_point``).
    The gradient points toward the exterior (in the direction of increasing SDF).
    """
    tid = wp.tid()
    point = query_points[tid]
    result = wp.mesh_query_point(mesh_id, point, max_distance)

    if not result.result:
        out_sdf[tid] = max_distance
        out_grad[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    cl_pt = wp.mesh_eval_position(mesh_id, result.face, result.u, result.v)
    delta = point - cl_pt
    dist = wp.length(delta)
    signed_dist = dist * result.sign

    grad = wp.vec3(0.0, 0.0, 0.0)
    if dist > 1.0e-8:
        grad = delta / dist * result.sign

    out_sdf[tid] = signed_dist
    out_grad[tid] = grad


@wp.kernel
def _closest_point_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    out_closest: wp.array(dtype=wp.vec3),
    out_sdf: wp.array(dtype=wp.float32),
    max_distance: wp.float32,
):
    """For each query point, find the closest point on the mesh surface and signed distance.

    Convention: signed distance is negative inside, positive outside.
    """
    tid = wp.tid()
    point = query_points[tid]
    result = wp.mesh_query_point(mesh_id, point, max_distance)

    if not result.result:
        out_closest[tid] = point
        out_sdf[tid] = max_distance
        return

    cl_pt = wp.mesh_eval_position(mesh_id, result.face, result.u, result.v)
    dist = wp.length(point - cl_pt)
    out_closest[tid] = cl_pt
    out_sdf[tid] = dist * result.sign


# ---------------------------------------------------------------------------
# WarpMeshQuery
# ---------------------------------------------------------------------------

class WarpMeshQuery:
    """GPU-accelerated mesh queries backed by a Warp BVH.

    Builds a Warp :class:`wp.Mesh` from a :class:`trimesh.Trimesh` once, then
    provides fast batched queries on the GPU. All public methods accept and return
    :class:`torch.Tensor` on the same CUDA device.

    Args:
        mesh: Source triangle mesh.
        device: Torch device (must be CUDA).
    """

    def __init__(self, mesh: trimesh.Trimesh, device: torch.device):
        init_warp()

        if device.index is None:
            device = torch.device("cuda", 0)
        self.device = device

        verts = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)

        self._wp_device = wp.device_from_torch(device)
        self._wp_verts = wp.array(verts, dtype=wp.vec3, device=self._wp_device)
        self._wp_faces = wp.array(np.ravel(faces), dtype=wp.int32, device=self._wp_device)
        self._wp_mesh = wp.Mesh(points=self._wp_verts, indices=self._wp_faces)

        # Max query distance = bounding-box diagonal + margin
        box_dims = mesh.bounds[1] - mesh.bounds[0]
        self._max_distance = float(np.linalg.norm(box_dims)) + 0.01

        # Pre-allocated output buffers (resized lazily)
        self._batch_size = 0
        self._outside_buf: torch.Tensor | None = None
        self._sdf_buf: torch.Tensor | None = None
        self._grad_buf: torch.Tensor | None = None
        self._closest_buf: torch.Tensor | None = None

    # -- internal helpers --------------------------------------------------

    def _ensure_buffers(self, n: int) -> None:
        """Lazily (re)allocate output buffers when batch size changes."""
        if n != self._batch_size:
            self._batch_size = n
            self._outside_buf = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._sdf_buf = torch.zeros(n, dtype=torch.float32, device=self.device)
            self._grad_buf = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
            self._closest_buf = torch.zeros((n, 3), dtype=torch.float32, device=self.device)

    # -- public API --------------------------------------------------------

    def query_outside_mask(self, points: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask that is ``True`` for points outside the mesh.

        Args:
            points: Query points of shape ``(N, 3)`` on the same CUDA device.

        Returns:
            Boolean tensor of shape ``(N,)``.
        """
        points = points.contiguous()
        n = points.shape[0]
        self._ensure_buffers(n)
        wp_device, wp_stream = get_warp_device_stream(points)
        wp.launch(
            kernel=_query_outside_kernel,
            dim=n,
            inputs=[
                self._wp_mesh.id,
                wp.from_torch(points, dtype=wp.vec3),
                wp.from_torch(self._outside_buf, dtype=wp.bool),
                self._max_distance,
            ],
            stream=wp_stream,
            device=wp_device,
        )
        return self._outside_buf

    def query_sdf(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute signed distance and gradient for each query point.

        This is the *raw* (non-differentiable) query. For differentiable use inside
        a loss function, see :class:`WarpSphereSDFFunction`.

        Args:
            points: Query points of shape ``(N, 3)`` on the same CUDA device.

        Returns:
            Tuple ``(sdf, grad)`` where ``sdf`` has shape ``(N,)`` and ``grad`` has
            shape ``(N, 3)``.
        """
        points = points.contiguous()
        n = points.shape[0]
        self._ensure_buffers(n)
        wp_device, wp_stream = get_warp_device_stream(points)
        wp.launch(
            kernel=_sdf_forward_kernel,
            dim=n,
            inputs=[
                self._wp_mesh.id,
                wp.from_torch(points, dtype=wp.vec3),
                wp.from_torch(self._sdf_buf),
                wp.from_torch(self._grad_buf, dtype=wp.vec3),
                self._max_distance,
            ],
            stream=wp_stream,
            device=wp_device,
        )
        return self._sdf_buf.clone(), self._grad_buf.clone()

    def query_closest_point(
        self, points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the closest surface point and signed distance for each query point.

        Replaces both ``trimesh.proximity.ProximityQuery.signed_distance`` and
        ``on_surface`` in a single GPU kernel launch.

        Args:
            points: Query points of shape ``(N, 3)`` on the same CUDA device.

        Returns:
            Tuple ``(closest_points, sdf)`` where ``closest_points`` has shape
            ``(N, 3)`` and ``sdf`` has shape ``(N,)``.  Signed distance convention:
            negative inside, positive outside.
        """
        points = points.contiguous()
        n = points.shape[0]
        self._ensure_buffers(n)
        wp_device, wp_stream = get_warp_device_stream(points)
        wp.launch(
            kernel=_closest_point_kernel,
            dim=n,
            inputs=[
                self._wp_mesh.id,
                wp.from_torch(points, dtype=wp.vec3),
                wp.from_torch(self._closest_buf, dtype=wp.vec3),
                wp.from_torch(self._sdf_buf),
                self._max_distance,
            ],
            stream=wp_stream,
            device=wp_device,
        )
        return self._closest_buf.clone(), self._sdf_buf.clone()


# ---------------------------------------------------------------------------
# Differentiable autograd wrapper
# ---------------------------------------------------------------------------

class WarpSphereSDFFunction(torch.autograd.Function):
    """Differentiable signed-distance query via Warp.

    Forward pass launches the Warp kernel to compute per-point SDF values.
    Backward pass uses the analytic gradient (direction from closest surface
    point to query point, scaled by sign) to propagate gradients back to the
    query-point tensor.

    Usage::

        mesh_query = WarpMeshQuery(mesh, device)
        sdf = WarpSphereSDFFunction.apply(sphere_centers, mesh_query)
    """

    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,
        mesh_query: WarpMeshQuery,
    ) -> torch.Tensor:
        """Compute signed distance from *points* to the mesh surface.

        Args:
            points: ``(N, 3)`` query positions (requires_grad may be True).
            mesh_query: :class:`WarpMeshQuery` instance (not a Tensor, saved on *ctx*).

        Returns:
            ``(N,)`` signed-distance values (negative inside, positive outside).
        """
        sdf, grad = mesh_query.query_sdf(points.detach().contiguous())
        ctx.save_for_backward(grad)
        return sdf

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Propagate loss gradient through the SDF query.

        Args:
            grad_output: ``(N,)`` upstream gradient (dL/d_sdf).

        Returns:
            Tuple ``(grad_points, None)`` -- gradient w.r.t. *points* and ``None``
            for the non-Tensor *mesh_query* argument.
        """
        (grad_sdf_wrt_pts,) = ctx.saved_tensors  # (N, 3)
        # dL/d_points = dL/d_sdf * d_sdf/d_points
        grad_points = grad_output.unsqueeze(-1) * grad_sdf_wrt_pts  # (N, 3)
        return grad_points, None
