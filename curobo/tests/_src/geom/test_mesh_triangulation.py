# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for mesh triangulation utilities."""

# Third Party
import pytest

# CuRobo
from curobo._src.geom.mesh_triangulation import triangulate_mesh_faces
from curobo._src.types.device_cfg import DeviceCfg


def test_triangle_faces_pass_through():
    """Test that triangle faces are returned unchanged."""
    faces = triangulate_mesh_faces(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[0, 1, 2],
        face_counts=[3],
        device_cfg=DeviceCfg().cpu(),
    )

    assert faces == [[0, 1, 2]]


def test_quad_faces_are_triangulated():
    """Test that quad faces are triangulated."""
    faces = triangulate_mesh_faces(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[0, 1, 2, 3],
        face_counts=[4],
        device_cfg=DeviceCfg().cpu(),
    )

    assert faces == [[1, 3, 0], [1, 2, 3]]


def test_mixed_triangle_and_quad_faces_preserve_order():
    """Test that mixed triangle and quad inputs preserve face order."""
    faces = triangulate_mesh_faces(
        vertices=[
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ],
        faces=[0, 1, 3, 1, 2, 4, 3],
        face_counts=[3, 4],
        device_cfg=DeviceCfg().cpu(),
    )

    assert faces == [[0, 1, 3], [2, 3, 1], [2, 4, 3]]


def test_mismatched_face_counts_raise():
    """Test that inconsistent flat faces and counts raise."""
    with pytest.raises(ValueError, match="Face index buffer length does not match face counts"):
        triangulate_mesh_faces(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[0, 1, 2],
            face_counts=[3, 3],
            device_cfg=DeviceCfg().cpu(),
        )


def test_unsupported_face_count_raises():
    """Test that unsupported polygon face counts raise."""
    with pytest.raises(ValueError, match="only triangles and quads are supported"):
        triangulate_mesh_faces(
            vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
            faces=[0, 1, 2, 3, 4],
            face_counts=[5],
            device_cfg=DeviceCfg().cpu(),
        )


def test_out_of_bounds_face_index_raises():
    """Test that face indices must reference existing vertices."""
    with pytest.raises(ValueError, match="Mesh face index is out of bounds"):
        triangulate_mesh_faces(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[0, 1, 3],
            face_counts=[3],
            device_cfg=DeviceCfg().cpu(),
        )
