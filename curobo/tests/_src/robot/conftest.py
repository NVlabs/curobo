# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared robot fixtures for the loader and dynamics tests."""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

ARM_MASS, ARM_COM, ARM_RPY = 2.0, np.array([0.3, 0.02, -0.05]), (0.3, -0.5, 0.8)
ARM_INERTIA = np.array([[0.02, 0.003, -0.002], [0.003, 0.05, 0.001], [-0.002, 0.001, 0.04]])
CAM_MASS, CAM_COM = 0.4, np.array([0.01, 0.0, 0.02])
CAM_INERTIA = np.diag([0.001, 0.002, 0.003])
# camera link in the arm link: two fixed joints, the second one rotated
MOUNT_XYZ, CAM_XYZ, CAM_RPY = np.array([0.4, 0.0, 0.0]), np.array([0.0, 0.0, 0.1]), (0.0, 0.0, 0.7)
# a passive bar on the arm link: a revolute joint nothing in the tree needs, lumped at zero
BAR_MASS, BAR_COM, BAR_INERTIA = 0.3, np.array([0.05, 0.0, 0.01]), np.diag([0.0002, 0.001, 0.001])
BAR_XYZ, BAR_RPY = np.array([0.2, 0.0, -0.05]), (0.5, 0.0, 0.3)


class LumpedInertialRobot(NamedTuple):
    """A URDF with links outside the kinematic tree, and the arm link's expected inertial.

    ``mass``, ``com`` and ``inertia`` are the arm link with the camera and the bar lumped in:
    the center of mass in the link frame and the inertia about it, in the link axes.
    """

    urdf_path: Path
    mass: float
    com: np.ndarray
    inertia: np.ndarray


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _vec(v):
    return " ".join(str(x) for x in v)


def _inertia_attr(i):
    return (
        f'ixx="{i[0, 0]}" ixy="{i[0, 1]}" ixz="{i[0, 2]}" iyy="{i[1, 1]}" iyz="{i[1, 2]}" '
        f'izz="{i[2, 2]}"'
    )


def _urdf() -> str:
    return f"""<?xml version="1.0"?>
<robot name="pendulum">
  <link name="base_link"><inertial><mass value="5.0"/>
    <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
  <link name="arm_link"><inertial><origin xyz="{_vec(ARM_COM)}" rpy="{_vec(ARM_RPY)}"/>
    <mass value="{ARM_MASS}"/><inertia {_inertia_attr(ARM_INERTIA)}/></inertial></link>
  <link name="mount_link"/>
  <link name="bar_link"><inertial><origin xyz="{_vec(BAR_COM)}"/>
    <mass value="{BAR_MASS}"/><inertia {_inertia_attr(BAR_INERTIA)}/></inertial></link>
  <link name="camera_link"><inertial><origin xyz="{_vec(CAM_COM)}"/>
    <mass value="{CAM_MASS}"/><inertia {_inertia_attr(CAM_INERTIA)}/></inertial></link>
  <joint name="swing" type="revolute"><origin xyz="0 0 1.0"/><parent link="base_link"/>
    <child link="arm_link"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" velocity="5" effort="100"/></joint>
  <joint name="mount" type="fixed"><origin xyz="{_vec(MOUNT_XYZ)}"/>
    <parent link="arm_link"/><child link="mount_link"/></joint>
  <joint name="bar" type="revolute"><origin xyz="{_vec(BAR_XYZ)}" rpy="{_vec(BAR_RPY)}"/>
    <parent link="arm_link"/><child link="bar_link"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="5" effort="10"/></joint>
  <joint name="camera" type="fixed"><origin xyz="{_vec(CAM_XYZ)}" rpy="{_vec(CAM_RPY)}"/>
    <parent link="mount_link"/><child link="camera_link"/></joint>
</robot>
"""


@pytest.fixture
def lumped_inertial_robot(tmp_path: Path) -> LumpedInertialRobot:
    """One revolute joint about y, with links the kinematic tree leaves out.

    A "camera" is fixed to the arm link through two fixed joints, and a passive "bar" hangs on a
    revolute joint. With ``arm_link`` as the only tool frame, the loader leaves both out of the
    tree and lumps their mass into the arm link (the bar at its joint's zero). The expected
    values come from the parallel-axis theorem, rotated into the link frame.
    """
    urdf_path = tmp_path / "pendulum.urdf"
    urdf_path.write_text(_urdf())

    r_arm = _rpy(*ARM_RPY)
    arm_inertia = r_arm @ ARM_INERTIA @ r_arm.T  # the URDF tensor is in the inertial frame
    r_cam = _rpy(*CAM_RPY)
    cam_com = MOUNT_XYZ + CAM_XYZ + r_cam @ CAM_COM
    cam_inertia = r_cam @ CAM_INERTIA @ r_cam.T
    r_bar = _rpy(*BAR_RPY)  # the bar's joint at zero: its origin
    bar_com = BAR_XYZ + r_bar @ BAR_COM
    bar_inertia = r_bar @ BAR_INERTIA @ r_bar.T
    mass = ARM_MASS + CAM_MASS + BAR_MASS
    com = (ARM_MASS * ARM_COM + CAM_MASS * cam_com + BAR_MASS * bar_com) / mass

    def about(c, m, inertia):
        d = c - com
        return inertia + m * (d @ d * np.eye(3) - np.outer(d, d))

    inertia = (
        about(ARM_COM, ARM_MASS, arm_inertia)
        + about(cam_com, CAM_MASS, cam_inertia)
        + about(bar_com, BAR_MASS, bar_inertia)
    )
    return LumpedInertialRobot(urdf_path, mass, com, inertia)
