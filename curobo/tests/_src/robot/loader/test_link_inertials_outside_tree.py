# SPDX-FileCopyrightText: Copyright (c) 2026 Mentee Robotics. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Link inertials from a URDF, including links the kinematic tree leaves out.

``KinematicsLoader`` keeps only the links on a chain to a tool frame, a collision link or an
extra link's parent. The mass of every other link (a camera, an IMU, the bars of a passive
linkage) is folded into the nearest tree link above it, so ``Dynamics`` (RNEA) still sees it.

The robot here is one revolute joint about y with a link whose inertial frame is rotated, a
"camera" fixed to that link through two fixed joints, and a passive "bar" on a revolute joint;
no tool frame or collision link needs them, so the loader leaves them out of the tree and lumps
them into the arm link (the bar at its joint's zero). Expected values, in closed form:

- ``test_link_inertials_follow_the_urdf`` (CPU): the arm link's mass, COM and inertia in the
  link frame, with the camera and the bar lumped in (parallel-axis theorem, rotated into the link
  frame).
- ``test_inverse_dynamics_match_the_closed_form`` (CUDA): the RNEA torque,
  ``(I_yy + m |c_xz|^2) qdd + dV/dq``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

GRAVITY = 9.81
ARM_MASS, ARM_COM, ARM_RPY = 2.0, np.array([0.3, 0.02, -0.05]), (0.3, -0.5, 0.8)
ARM_INERTIA = np.array([[0.02, 0.003, -0.002], [0.003, 0.05, 0.001], [-0.002, 0.001, 0.04]])
CAM_MASS, CAM_COM = 0.4, np.array([0.01, 0.0, 0.02])
CAM_INERTIA = np.diag([0.001, 0.002, 0.003])
# camera link in the arm link: two fixed joints, the second one rotated
MOUNT_XYZ, CAM_XYZ, CAM_RPY = np.array([0.4, 0.0, 0.0]), np.array([0.0, 0.0, 0.1]), (0.0, 0.0, 0.7)
# a passive bar on the arm link: a revolute joint nothing in the tree needs, lumped at zero
BAR_MASS, BAR_COM, BAR_INERTIA = 0.3, np.array([0.05, 0.0, 0.01]), np.diag([0.0002, 0.001, 0.001])
BAR_XYZ, BAR_RPY = np.array([0.2, 0.0, -0.05]), (0.5, 0.0, 0.3)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _inertia_attr(i):
    return (
        f'ixx="{i[0, 0]}" ixy="{i[0, 1]}" ixz="{i[0, 2]}" iyy="{i[1, 1]}" iyz="{i[1, 2]}" '
        f'izz="{i[2, 2]}"'
    )


def _urdf() -> str:
    xyz = " ".join(str(v) for v in ARM_COM)
    rpy = " ".join(str(v) for v in ARM_RPY)
    return f"""<?xml version="1.0"?>
<robot name="pendulum">
  <link name="base_link"><inertial><mass value="5.0"/>
    <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
  <link name="arm_link"><inertial><origin xyz="{xyz}" rpy="{rpy}"/>
    <mass value="{ARM_MASS}"/><inertia {_inertia_attr(ARM_INERTIA)}/></inertial></link>
  <link name="mount_link"/>
  <link name="bar_link"><inertial><origin xyz="{" ".join(str(v) for v in BAR_COM)}"/>
    <mass value="{BAR_MASS}"/><inertia {_inertia_attr(BAR_INERTIA)}/></inertial></link>
  <link name="camera_link"><inertial><origin xyz="{" ".join(str(v) for v in CAM_COM)}"/>
    <mass value="{CAM_MASS}"/><inertia {_inertia_attr(CAM_INERTIA)}/></inertial></link>
  <joint name="swing" type="revolute"><origin xyz="0 0 1.0"/><parent link="base_link"/>
    <child link="arm_link"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" velocity="5" effort="100"/></joint>
  <joint name="mount" type="fixed"><origin xyz="{" ".join(str(v) for v in MOUNT_XYZ)}"/>
    <parent link="arm_link"/><child link="mount_link"/></joint>
  <joint name="bar" type="revolute">
    <origin xyz="{" ".join(str(v) for v in BAR_XYZ)}" rpy="{" ".join(str(v) for v in BAR_RPY)}"/>
    <parent link="arm_link"/><child link="bar_link"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="5" effort="10"/></joint>
  <joint name="camera" type="fixed">
    <origin xyz="{" ".join(str(v) for v in CAM_XYZ)}" rpy="{" ".join(str(v) for v in CAM_RPY)}"/>
    <parent link="mount_link"/><child link="camera_link"/></joint>
</robot>
"""


def _expected():
    """Mass, COM and inertia (about the COM, link axes) of the arm link, camera and bar in it."""
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
    return mass, com, inertia


def _kinematics_cfg(tmp_path, device):
    from curobo.kinematics import KinematicsCfg
    from curobo.types import DeviceCfg

    urdf = tmp_path / "pendulum.urdf"
    urdf.write_text(_urdf())
    kin = {
        "urdf_path": str(urdf),
        "asset_root_path": str(tmp_path),
        "base_link": "base_link",
        "tool_frames": ["arm_link"],
        "collision_link_names": [],
        "collision_spheres": {},
        "cspace": {
            "joint_names": ["swing"],
            "default_joint_position": [0.0],
            "cspace_distance_weight": [1.0],
            "null_space_weight": [1.0],
            "max_acceleration": 10.0,
            "max_jerk": 500.0,
        },
    }
    return KinematicsCfg.from_robot_yaml_file(
        {"kinematics": kin}, device_cfg=DeviceCfg(device=torch.device(device))
    )


def test_link_inertials_follow_the_urdf(tmp_path):
    kp = _kinematics_cfg(tmp_path, "cpu").kinematics_config
    names = kp.link_name_to_idx_map
    assert not {"camera_link", "mount_link", "bar_link"} & set(names)  # outside the tree
    mc = kp.link_masses_com[names["arm_link"]].double().numpy()
    six = kp.link_inertias[names["arm_link"], :6].double().numpy()
    mass, com, inertia = _expected()
    assert mc[3] == pytest.approx(mass, rel=1e-6)
    np.testing.assert_allclose(mc[:3], com, atol=1e-6, err_msg="center of mass (link frame)")
    np.testing.assert_allclose(
        six,
        inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
        atol=1e-6,
        err_msg="inertia about the COM, link axes [ixx, iyy, izz, ixy, ixz, iyz]",
    )
    # a tree link with nothing fixed to it keeps its own URDF values
    base = kp.link_masses_com[names["base_link"]].double().numpy()
    np.testing.assert_allclose(base, [0.0, 0.0, 0.0, 5.0], atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_inverse_dynamics_match_the_closed_form(tmp_path):
    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
    from curobo.kinematics import Kinematics
    from curobo.types import JointState

    kin = Kinematics(_kinematics_cfg(tmp_path, "cuda"))
    dyn = Dynamics(
        DynamicsCfg(
            kinematics_config=kin.config.kinematics_config, device_cfg=kin.config.device_cfg
        )
    )
    q = torch.linspace(-2.5, 2.5, 11, device="cuda").unsqueeze(1)
    qdd = torch.linspace(-8.0, 8.0, 11, device="cuda").unsqueeze(1)
    dyn.setup_batch_size(q.shape[0])
    js = JointState.from_position(q, joint_names=kin.joint_names)
    js.velocity = torch.zeros_like(q)
    js.acceleration = qdd
    tau = dyn.compute_inverse_dynamics(js).double().cpu().numpy()[:, 0]

    mass, com, inertia = _expected()
    qs, qdds = q.double().cpu().numpy()[:, 0], qdd.double().cpu().numpy()[:, 0]
    # about y, COM c rotated by Ry(q): height z = -c_x sin q + c_z cos q, V = m g z
    d_potential = mass * GRAVITY * (-com[0] * np.cos(qs) - com[2] * np.sin(qs))
    expected = (inertia[1, 1] + mass * (com[0] ** 2 + com[2] ** 2)) * qdds + d_potential
    np.testing.assert_allclose(tau, expected, atol=2e-4, rtol=1e-5)
