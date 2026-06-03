# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Mesh face triangulation utilities."""

# Standard Library
from typing import List, Tuple

# Third Party
import numpy as np
import torch
import warp as wp

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp
from curobo.logging import log_and_raise


@wp.kernel
def triangulate_quads_kernel(
    verts: wp.array(dtype=wp.vec3),
    quads: wp.array(dtype=wp.vec4i),
    tris_out: wp.array(dtype=wp.vec3i),
):
    ind = wp.tid()
    ind_out = 2 * ind
    quad_inds = quads[ind]
    q0, q1, q2, q3 = quad_inds[0], quad_inds[1], quad_inds[2], quad_inds[3]
    v0 = verts[q0]
    v1 = verts[q1]
    v2 = verts[q2]
    v3 = verts[q3]

    n_a1 = wp.normalize(wp.cross(v1 - v0, v2 - v0))
    n_a2 = wp.normalize(wp.cross(v2 - v0, v3 - v0))

    n_b1 = wp.normalize(wp.cross(v3 - v1, v0 - v1))
    n_b2 = wp.normalize(wp.cross(v2 - v1, v3 - v1))

    if wp.dot(n_a1, n_a2) > wp.dot(n_b1, n_b2):
        tris_out[ind_out] = wp.vec3i(q0, q1, q2)
        tris_out[ind_out + 1] = wp.vec3i(q0, q2, q3)
    else:
        tris_out[ind_out] = wp.vec3i(q1, q3, q0)
        tris_out[ind_out + 1] = wp.vec3i(q1, q2, q3)


def triangulate_quads_warp(
    vertices: List[List[float]],
    quads: List[List[int]],
    device_cfg: DeviceCfg,
) -> List[List[List[int]]]:
    init_warp()
    verts_np = np.asarray(vertices, dtype=np.float32)
    quads_np = np.asarray(quads, dtype=np.int32)
    torch_device = device_cfg.device
    if torch_device.type == "cuda" and torch_device.index is None:
        torch_device = torch.device("cuda", 0)
    wp_device = wp.device_from_torch(torch_device)

    verts_wp = wp.array(verts_np, dtype=wp.vec3, device=wp_device)
    quads_wp = wp.array(quads_np, dtype=wp.vec4i, device=wp_device)
    tris_wp = wp.empty(shape=(len(quads) * 2,), dtype=wp.vec3i, device=wp_device)

    wp.launch(
        triangulate_quads_kernel,
        dim=len(quads),
        inputs=[verts_wp, quads_wp],
        outputs=[tris_wp],
        device=wp_device,
    )

    return tris_wp.numpy().astype(np.int64).reshape(len(quads), 2, 3).tolist()


def triangulate_mesh_faces(
    vertices: List[List[float]],
    faces: List[int],
    face_counts: List[int],
    device_cfg: DeviceCfg = DeviceCfg(),
) -> List[List[int]]:
    """Convert triangle and quad mesh faces to triangle faces.

    Args:
        vertices: Mesh vertices used to validate face indices.
        faces: Flat face index buffer.
        face_counts: Number of vertices per face.
        device_cfg: Device configuration used to select the Warp device.

    Returns:
        List of triangle faces.

    Raises:
        ValueError: If face data is inconsistent or contains unsupported faces.
    """
    if len(faces) == 0:
        return []

    if sum(face_counts) != len(faces):
        log_and_raise(
            "Face index buffer length does not match face counts: "
            + str(len(faces))
            + " indices for "
            + str(sum(face_counts))
            + " counted vertices"
        )

    num_vertices = len(vertices)
    output_parts: List[Tuple[str, int]] = []
    triangle_faces: List[List[int]] = []
    quad_faces: List[List[int]] = []

    face_offset = 0
    for count in face_counts:
        face = faces[face_offset : face_offset + count]
        face_offset += count

        if count not in (3, 4):
            log_and_raise(
                "Unsupported mesh face with "
                + str(count)
                + " vertices; only triangles and quads are supported"
            )
        if any(x < 0 or x >= num_vertices for x in face):
            log_and_raise("Mesh face index is out of bounds")

        if count == 3:
            output_parts.append(("triangle", len(triangle_faces)))
            triangle_faces.append(face)
        else:
            output_parts.append(("quad", len(quad_faces)))
            quad_faces.append(face)

    quad_triangles = []
    if len(quad_faces) > 0:
        quad_triangles = triangulate_quads_warp(vertices, quad_faces, device_cfg=device_cfg)

    output_faces: List[List[int]] = []
    for face_type, face_idx in output_parts:
        if face_type == "triangle":
            output_faces.append(triangle_faces[face_idx])
        else:
            output_faces.extend(quad_triangles[face_idx])

    return output_faces
