# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot kinematic parsers.

Parsers turn robot description files into the link/joint data structures used by
:class:`curobo.kinematics.Kinematics`. Most users do not call parsers directly — the
high-level builders in :mod:`curobo.robot_builder` consume them internally. This module
is useful when you need direct access to the parsed representation, e.g. for inspecting
link meshes or joint limits.

Example:
    .. code-block:: python

        from curobo.robot_parser import UrdfRobotParser

        parser = UrdfRobotParser("robot.urdf", mesh_root="assets/")
        link_names = parser.get_link_names()
        mesh = parser.get_link_mesh(link_names[0])
"""
from curobo._src.robot.parser.parser_urdf import UrdfRobotParser

__all__ = [
    "UrdfRobotParser",
]
