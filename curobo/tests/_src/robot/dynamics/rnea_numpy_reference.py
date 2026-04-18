# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""NumPy reference implementation of body-frame RNEA (Recursive Newton-Euler Algorithm).

Uses curobo's KinematicsParams data layout. Intended for validation against the CUDA kernel.

Convention (Featherstone):
    - Spatial vectors are 6D: [angular (3); linear (3)]
    - Spatial transform X_{child<-parent} maps motion vectors from parent to child frame
    - Given homogeneous T = [R|p] where R rotates child->parent, p = child origin in parent:
        X = [ R^T,             0   ]
            [ -R^T * skew(p),  R^T ]
      (Featherstone: X = [E, 0; -skew(r)*E, E] with r = R^T*p)
    - Spatial inertia about link origin in body frame:
        I = [ Ic + m*(skew(c)^T * skew(c)),  m * skew(c)   ]
            [ m * skew(c)^T,                  m * I_3       ]
      where c = CoM offset from link origin in body frame
    - Joint types match curobo: FIXED=-1, X_PRISM=0..Z_PRISM=2, X_ROT=3..Z_ROT=5
"""

from typing import Optional, Tuple

import numpy as np

# Joint type constants (matching kinematics_constants.h)
FIXED = -1
X_PRISM = 0
Y_PRISM = 1
Z_PRISM = 2
X_ROT = 3
Y_ROT = 4
Z_ROT = 5


# ---------------------------------------------------------------------------
# Spatial algebra helpers
# ---------------------------------------------------------------------------


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector. skew(v) * x = v x x."""
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3D cross product a x b."""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def rotation_matrix(axis: int, angle: float) -> np.ndarray:
    """Build 3x3 rotation matrix for rotation about axis (0=X, 1=Y, 2=Z)."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:  # X
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 1:  # Y
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # Z
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_joint_transform(
    fixed_transform: np.ndarray, joint_type: int, q: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Build local homogeneous transform T = T_fixed * T_joint(q).

    Args:
        fixed_transform: [3, 4] static transform (R_fixed | p_fixed).
        joint_type: joint type enum value.
        q: joint position (angle for revolute, displacement for prismatic).

    Returns:
        R: [3, 3] rotation matrix (child axes in parent frame).
        p: [3] position of child origin in parent frame.
    """
    R_fixed = fixed_transform[:, :3]
    p_fixed = fixed_transform[:, 3]

    if joint_type == FIXED:
        return R_fixed, p_fixed

    if joint_type >= X_ROT:
        # Revolute: T = T_fixed * R_joint(q)
        axis = joint_type - X_ROT  # 0=X, 1=Y, 2=Z
        R_joint = rotation_matrix(axis, q)
        R = R_fixed @ R_joint
        p = p_fixed
    else:
        # Prismatic: T = T_fixed * T_prism(q)
        axis = joint_type - X_PRISM  # 0=X, 1=Y, 2=Z
        d = np.zeros(3)
        d[axis] = q
        R = R_fixed
        p = p_fixed + R_fixed @ d

    return R, p


def homogeneous_to_spatial_transform(
    R: np.ndarray, p: np.ndarray
) -> np.ndarray:
    """Build 6x6 spatial transform from homogeneous components.

    X_{child<-parent} maps motion vectors from parent frame to child frame.

    Given T = [R|p] where R rotates child->parent and p = child origin in parent frame:
        E = R^T (parent->child rotation)
        r = O_parent - O_child in parent coords = -p
        X = [E, 0; -E*skew(r), E] = [E, 0; E*skew(p), E]

    Args:
        R: [3,3] rotation (child axes in parent frame, i.e. rotates child->parent).
        p: [3] child origin in parent frame.

    Returns:
        X: [6,6] spatial motion transform.
    """
    E = R.T  # parent->child rotation
    X = np.zeros((6, 6))
    X[:3, :3] = E
    X[3:, :3] = -E @ skew(p)
    X[3:, 3:] = E
    return X


def spatial_transform_multiply(
    R: np.ndarray, p: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Compute X * v without forming the 6x6 matrix.

    X = [E, 0; -E*skew(p), E] where E = R^T.
    X * [omega; v_lin] = [E*omega; E*(v_lin + omega x p)]

    Args:
        R: [3,3] rotation (child axes in parent frame).
        p: [3] child origin in parent frame.
        v: [6] spatial motion vector [angular; linear].

    Returns:
        result: [6] = X * v.
    """
    E = R.T
    omega = v[:3]
    v_lin = v[3:]

    result = np.zeros(6)
    result[:3] = E @ omega
    result[3:] = E @ (v_lin + cross3(omega, p))
    return result


def spatial_transform_transpose_multiply(
    R: np.ndarray, p: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """Compute X^T * f without forming the 6x6 matrix.

    X^T = [R, skew(p)*R; 0, R]  (since X = [E, 0; -E*p_tilde, E], E=R^T)
    X^T * [f_ang; f_lin] = [R*f_ang + p x (R*f_lin); R*f_lin]

    Args:
        R: [3,3] rotation (child axes in parent frame).
        p: [3] child origin in parent frame.
        f: [6] spatial force vector [torque; force].

    Returns:
        result: [6] = X^T * f.
    """
    f_ang = f[:3]
    f_lin = f[3:]

    Rf_lin = R @ f_lin
    result = np.zeros(6)
    result[:3] = R @ f_ang + cross3(p, Rf_lin)
    result[3:] = Rf_lin
    return result


def build_spatial_inertia(
    mass: float, com: np.ndarray, inertia: np.ndarray
) -> np.ndarray:
    """Build 6x6 spatial inertia about link origin in body frame.

    Args:
        mass: link mass.
        com: [3] center of mass in body (link) frame.
        inertia: [6] = [ixx, iyy, izz, ixy, ixz, iyz] at CoM in body frame.

    Returns:
        I_spatial: [6,6] spatial inertia matrix.
    """
    # 3x3 rotational inertia at CoM
    Ic = np.array(
        [
            [inertia[0], inertia[3], inertia[4]],
            [inertia[3], inertia[1], inertia[5]],
            [inertia[4], inertia[5], inertia[2]],
        ]
    )

    cx = skew(com)  # skew(c)

    I_spatial = np.zeros((6, 6))
    # Top-left: Ic + m * skew(c)^T * skew(c)  (parallel axis theorem)
    I_spatial[:3, :3] = Ic + mass * (cx.T @ cx)
    # Top-right: m * skew(c)  (angular momentum contribution from linear velocity)
    I_spatial[:3, 3:] = mass * cx
    # Bottom-left: m * skew(c)^T  (linear momentum contribution from angular velocity)
    I_spatial[3:, :3] = mass * cx.T
    # Bottom-right: m * I_3
    I_spatial[3:, 3:] = mass * np.eye(3)

    return I_spatial


def spatial_inertia_multiply(
    mass: float, com: np.ndarray, inertia: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Compute I_spatial * v without forming the 6x6 matrix.

    Derivation from I = [Ic + m*cx^T*cx, m*cx; m*cx^T, m*I3]:
        h = v_lin + omega x c       (= v_lin + omega × com)
        result_lin = m * h           (= m*cx^T*omega + m*v_lin)
        result_ang = Ic*omega + m*(c x h)  (= (Ic + m*cx^T*cx)*omega + m*cx*v_lin)

    Args:
        mass: link mass.
        com: [3] center of mass in body frame.
        inertia: [6] = [ixx, iyy, izz, ixy, ixz, iyz] at CoM.
        v: [6] spatial vector [angular; linear].

    Returns:
        result: [6] = I_spatial * v.
    """
    omega = v[:3]
    v_lin = v[3:]

    # 3x3 inertia at CoM (symmetric)
    Ic = np.array(
        [
            [inertia[0], inertia[3], inertia[4]],
            [inertia[3], inertia[1], inertia[5]],
            [inertia[4], inertia[5], inertia[2]],
        ]
    )

    h = v_lin + cross3(omega, com)  # v_lin + omega x c

    result = np.zeros(6)
    result[3:] = mass * h
    result[:3] = Ic @ omega + mass * cross3(com, h)
    return result


def spatial_force_cross(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Spatial force cross product: v x* f = crf(v) * f.

    crf(v) * f = [omega x f_ang + v_lin x f_lin; omega x f_lin]

    where v = [omega; v_lin] and f = [f_ang; f_lin].

    Args:
        v: [6] spatial motion vector.
        f: [6] spatial force vector.

    Returns:
        result: [6] = v x* f.
    """
    omega = v[:3]
    v_lin = v[3:]
    f_ang = f[:3]
    f_lin = f[3:]

    result = np.zeros(6)
    result[:3] = cross3(omega, f_ang) + cross3(v_lin, f_lin)
    result[3:] = cross3(omega, f_lin)
    return result


def motion_cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Spatial motion cross product: v1 x_m v2 = crm(v1) * v2.

    crm(v) = [omega_tilde, 0; v_lin_tilde, omega_tilde]
    crm(v) * v2 = [omega x v2_ang; v_lin x v2_ang + omega x v2_lin]

    Args:
        v1: [6] first spatial motion vector.
        v2: [6] second spatial motion vector.

    Returns:
        result: [6] = v1 x_m v2.
    """
    omega = v1[:3]
    v_lin = v1[3:]
    v2_ang = v2[:3]
    v2_lin = v2[3:]

    result = np.zeros(6)
    result[:3] = cross3(omega, v2_ang)
    result[3:] = cross3(v_lin, v2_ang) + cross3(omega, v2_lin)
    return result


def get_joint_motion_subspace(joint_type: int) -> np.ndarray:
    """Get the 6x1 joint motion subspace vector S.

    Args:
        joint_type: joint type enum value.

    Returns:
        S: [6] motion subspace vector (unit vector with 1 at the DOF index).
    """
    S = np.zeros(6)
    if joint_type == FIXED:
        return S
    if joint_type >= X_ROT:
        # Revolute: angular component
        S[joint_type - X_ROT] = 1.0
    else:
        # Prismatic: linear component
        S[3 + joint_type - X_PRISM] = 1.0
    return S


def get_joint_s_index(joint_type: int) -> int:
    """Get the index of the 1 in the S vector.

    Args:
        joint_type: joint type enum value.

    Returns:
        Index into the 6D spatial vector.
    """
    if joint_type >= X_ROT:
        return joint_type - X_ROT
    else:
        return 3 + joint_type - X_PRISM


# ---------------------------------------------------------------------------
# RNEA: Body-frame Inverse Dynamics
# ---------------------------------------------------------------------------


def rnea(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: Optional[np.ndarray],
    fixed_transforms: np.ndarray,
    link_map: np.ndarray,
    joint_map: np.ndarray,
    joint_map_type: np.ndarray,
    joint_offset_map: np.ndarray,
    link_masses_com: np.ndarray,
    link_inertias: np.ndarray,
    gravity: float = -9.81,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute inverse dynamics using body-frame RNEA.

    Algorithm (Featherstone Chapter 5):
        1. Forward pass (base -> tip): compute v[k], a[k] for each link
        2. Force computation: f[k] = I[k]*a[k] + v[k] x* (I[k]*v[k])
        3. Backward pass (tip -> base): f[parent] += X[k]^T * f[k], tau[k] = S^T * f[k]

    Args:
        q: [num_dof] joint positions.
        qd: [num_dof] joint velocities.
        qdd: [num_dof] joint accelerations (None = zero).
        fixed_transforms: [num_links, 3, 4] static transforms.
        link_map: [num_links] parent link index (-1 or 0 for base link's parent).
        joint_map: [num_links] joint index for each link (-1 for fixed).
        joint_map_type: [num_links] joint type for each link.
        joint_offset_map: [num_links, 2] joint offset [offset, multiplier].
        link_masses_com: [num_links, 4] = [com_x, com_y, com_z, mass].
        link_inertias: [num_links, 6] = [ixx, iyy, izz, ixy, ixz, iyz].
        gravity: gravity constant (default -9.81, applied as upward acceleration on base).

    Returns:
        tau: [num_dof] joint torques.
        v: [num_links, 6] spatial velocities per link.
        a: [num_links, 6] spatial accelerations per link.
        f: [num_links, 6] spatial forces per link (after backward pass).
    """
    num_links = len(link_map)
    num_dof = len(q)

    v = np.zeros((num_links, 6))
    a = np.zeros((num_links, 6))
    f = np.zeros((num_links, 6))
    tau = np.zeros(num_dof)

    # Gravity as spatial acceleration of the base (Featherstone trick:
    # instead of adding gravity to each link, we set a_base = -gravity)
    gravity_spatial = np.zeros(6)
    gravity_spatial[5] = -gravity  # linear Z = +9.81 (upward acceleration)

    # Store per-link R, p for backward pass reuse
    R_local = [None] * num_links
    p_local = [None] * num_links

    # -----------------------------------------------------------------------
    # Step 1 + 2: Forward pass, compute v[k], a[k], f[k]
    # -----------------------------------------------------------------------
    for k in range(num_links):
        jtype = int(joint_map_type[k])
        jidx = int(joint_map[k])

        # Get joint angle (apply mimic: q_actual = multiplier * q[jidx] + offset)
        # joint_offset_map[k] = [multiplier, offset] (matches curobo convention)
        if jtype != FIXED and jidx >= 0:
            multiplier = joint_offset_map[k, 0]
            offset = joint_offset_map[k, 1]
            q_k = multiplier * q[jidx] + offset
            qd_k = multiplier * qd[jidx]
            qdd_k = multiplier * qdd[jidx] if qdd is not None else 0.0
        else:
            q_k = 0.0
            qd_k = 0.0
            qdd_k = 0.0

        # Build local transform
        R, p = build_joint_transform(fixed_transforms[k], jtype, q_k)
        R_local[k] = R
        p_local[k] = p

        parent = int(link_map[k])

        # Joint motion subspace
        S = get_joint_motion_subspace(jtype)

        # Detect root link: parent < 0 or parent == self (curobo convention)
        is_root = parent < 0 or parent == k

        # Velocity: v[k] = X * v[parent] + S * qd
        if is_root:
            # Base link: parent is fixed world (v_parent = 0)
            v[k] = S * qd_k
            a[k] = spatial_transform_multiply(R, p, gravity_spatial) + S * qdd_k
        else:
            v[k] = spatial_transform_multiply(R, p, v[parent]) + S * qd_k
            a[k] = spatial_transform_multiply(R, p, a[parent]) + S * qdd_k

        # Coriolis: a += v x_m (S * qd)  (applies even for base link)
        a[k] += motion_cross_product(v[k], S * qd_k)

        # Force: f[k] = I * a[k] + v[k] x* (I * v[k])
        mass = link_masses_com[k, 3]
        com = link_masses_com[k, :3]
        inertia = link_inertias[k]

        Iv = spatial_inertia_multiply(mass, com, inertia, v[k])
        Ia = spatial_inertia_multiply(mass, com, inertia, a[k])
        f[k] = Ia + spatial_force_cross(v[k], Iv)

    # -----------------------------------------------------------------------
    # Step 3: Backward pass, propagate forces, extract torques
    # -----------------------------------------------------------------------
    for k in range(num_links - 1, -1, -1):
        jtype = int(joint_map_type[k])
        jidx = int(joint_map[k])
        parent = int(link_map[k])

        # Extract torque: tau = S^T * f
        if jtype != FIXED and jidx >= 0:
            s_idx = get_joint_s_index(jtype)
            multiplier = joint_offset_map[k, 0]  # [multiplier, offset]
            # For mimic joints: tau_q += multiplier * S^T * f
            tau[jidx] += multiplier * f[k, s_idx]

        # Propagate force to parent: f[parent] += X^T * f[k]
        # Skip if root link (parent < 0 or parent == self)
        if parent >= 0 and parent != k:
            f[parent] += spatial_transform_transpose_multiply(
                R_local[k], p_local[k], f[k]
            )

    return tau, v, a, f


# ---------------------------------------------------------------------------
# RNEA Backward: VJP (Vector-Jacobian Product) of Inverse Dynamics
# ---------------------------------------------------------------------------


def rnea_backward(
    tau_bar: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: Optional[np.ndarray],
    v: np.ndarray,
    a: np.ndarray,
    f: np.ndarray,
    fixed_transforms: np.ndarray,
    link_map: np.ndarray,
    joint_map: np.ndarray,
    joint_map_type: np.ndarray,
    joint_offset_map: np.ndarray,
    link_masses_com: np.ndarray,
    link_inertias: np.ndarray,
    gravity: float = -9.81,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute VJP (backprop) of RNEA: given dL/dtau, compute dL/dq, dL/dqd, dL/dqdd.

    This is reverse-mode AD of the RNEA forward pass. Two passes:
        Pass 1 (root->leaves): adjoint of RNEA backward sweep -> f_bar
        Pass 2 (leaves->root): adjoint of RNEA forward sweep -> v_bar, a_bar, grad_q/qd/qdd

    Key identities used:
        - dX/dq * m = -S × (X*m)  (since X = X_joint*X_fixed, only X_joint depends on q)
        - VJP of crm(v)*w w.r.t. v: crf(w) * bar  (bilinear adjoint)
        - VJP of crf(v)*Iv w.r.t. v: -crf(bar)*Iv - I*crm(v)*bar

    Args:
        tau_bar: [num_dof] upstream gradient dL/dtau.
        q, qd, qdd: joint positions, velocities, accelerations.
        v, a, f: [num_links, 6] spatial velocities, accelerations, forces from rnea().
        fixed_transforms, link_map, ...: robot model (same as rnea()).
        gravity: gravity constant.

    Returns:
        grad_q: [num_dof] dL/dq.
        grad_qd: [num_dof] dL/dqd.
        grad_qdd: [num_dof] dL/dqdd.
    """
    num_links = len(link_map)
    num_dof = len(q)

    # Output gradients
    grad_q = np.zeros(num_dof)
    grad_qd = np.zeros(num_dof)
    grad_qdd = np.zeros(num_dof)

    # Adjoint variables
    f_bar = np.zeros((num_links, 6))
    a_bar = np.zeros((num_links, 6))
    v_bar = np.zeros((num_links, 6))

    gravity_spatial = np.zeros(6)
    gravity_spatial[5] = -gravity

    # Recompute per-link transforms (cheap in NumPy; avoids changing rnea API)
    R_local = [None] * num_links
    p_local = [None] * num_links
    S_vec = [None] * num_links
    q_local = np.zeros(num_links)

    for k in range(num_links):
        jtype = int(joint_map_type[k])
        jidx = int(joint_map[k])
        if jtype != FIXED and jidx >= 0:
            multiplier = joint_offset_map[k, 0]
            offset = joint_offset_map[k, 1]
            q_local[k] = multiplier * q[jidx] + offset
        R_local[k], p_local[k] = build_joint_transform(
            fixed_transforms[k], jtype, q_local[k]
        )
        S_vec[k] = get_joint_motion_subspace(jtype)

    # ===================================================================
    # Pass 1: Adjoint of RNEA backward sweep (root -> leaves)
    #
    # Original backward (k = n-1..0):
    #   tau[k] = S^T * f[k]
    #   f[parent] += X^T * f[k]
    #
    # Adjoint (k = 0..n-1):
    #   f_bar[k] += S * tau_bar[k]           (adj of tau = S^T * f)
    #   f_bar[k] += X * f_bar[parent]        (adj of f[parent] += X^T * f)
    #   grad_q from dX^T/dq * f[k]
    # ===================================================================
    for k in range(num_links):
        jtype = int(joint_map_type[k])
        jidx = int(joint_map[k])
        parent = int(link_map[k])
        is_root = parent < 0 or parent == k

        # adj of tau[k] = S^T * f[k]
        if jtype != FIXED and jidx >= 0:
            s_idx = get_joint_s_index(jtype)
            multiplier = joint_offset_map[k, 0]
            f_bar[k, s_idx] += multiplier * tau_bar[jidx]

        # adj of f[parent] += X^T * f[k]
        if not is_root:
            f_bar[k] += spatial_transform_multiply(
                R_local[k], p_local[k], f_bar[parent]
            )

            # dX/dq contribution from f[parent] += X^T * f[k]:
            # dX^T/dq * f = X^T * crf(S) * f  (since dX/dq = -crm(S)*X)
            # q_bar += f_bar[parent]^T * X^T * crf(S) * f[k]
            #        = (X * f_bar[parent])^T * crf(S) * f[k]
            if jtype != FIXED and jidx >= 0:
                X_fbar_p = spatial_transform_multiply(
                    R_local[k], p_local[k], f_bar[parent]
                )
                grad_q[jidx] += multiplier * np.dot(
                    X_fbar_p, spatial_force_cross(S_vec[k], f[k])
                )

    # ===================================================================
    # Pass 2: Adjoint of RNEA forward sweep (leaves -> root)
    #
    # Original forward (k = 0..n-1):
    #   [1] v[k] = X * v[parent] + S * qd
    #   [2] a[k] = X * a[parent] + S * qdd + crm(v) * S * qd
    #   [3] f[k] = I * a[k] + crf(v) * I * v[k]
    #
    # Adjoint (k = n-1..0):
    #   [3]': f_bar -> a_bar, v_bar (via I and Coriolis)
    #   [2]': a_bar -> grad_qdd, grad_qd, v_bar, a_bar[parent], grad_q
    #   [1]': v_bar -> grad_qd, v_bar[parent], grad_q
    # ===================================================================
    for k in range(num_links - 1, -1, -1):
        jtype = int(joint_map_type[k])
        jidx = int(joint_map[k])
        parent = int(link_map[k])
        is_root = parent < 0 or parent == k

        mass = link_masses_com[k, 3]
        com = link_masses_com[k, :3]
        inertia = link_inertias[k]

        # --- [3]' adj of f[k] = I*a + crf(v)*I*v ---

        # a_bar[k] += I * f_bar[k]   (I is symmetric)
        a_bar[k] += spatial_inertia_multiply(mass, com, inertia, f_bar[k])

        # v_bar from Coriolis: -crf(f_bar)*Iv - I*crm(v)*f_bar
        Iv = spatial_inertia_multiply(mass, com, inertia, v[k])
        v_bar[k] -= spatial_force_cross(f_bar[k], Iv)
        v_bar[k] -= spatial_inertia_multiply(
            mass, com, inertia, motion_cross_product(v[k], f_bar[k])
        )

        # --- [2]' adj of a[k] = X*a_parent + S*qdd + crm(v)*S*qd ---

        if jtype != FIXED and jidx >= 0:
            s_idx = get_joint_s_index(jtype)
            multiplier = joint_offset_map[k, 0]
            qd_k = multiplier * qd[jidx]

            # grad_qdd += S^T * a_bar  (from S * qdd)
            grad_qdd[jidx] += multiplier * a_bar[k, s_idx]

            # From crm(v) * S * qd:
            #   grad_qd += S^T * crm(v)^T * a_bar   (w.r.t. qd)
            #   v_bar += crf(S*qd) * a_bar           (w.r.t. v)
            Sqd = S_vec[k] * qd_k
            # crm(v)^T * a_bar = -crf(v) * a_bar
            neg_crf_v_abar = -spatial_force_cross(v[k], a_bar[k])
            grad_qd[jidx] += multiplier * neg_crf_v_abar[s_idx]
            v_bar[k] += spatial_force_cross(Sqd, a_bar[k])

        # Propagate a_bar to parent and dX/dq
        if not is_root:
            # a_bar[parent] += X^T * a_bar[k]
            a_bar[parent] += spatial_transform_transpose_multiply(
                R_local[k], p_local[k], a_bar[k]
            )

            # dX/dq: dX/dq * a_parent = -S × (X * a_parent)
            # X * a_parent = a[k] - S*qdd_k - crm(v)*S*qd_k
            if jtype != FIXED and jidx >= 0:
                qd_k_loc = joint_offset_map[k, 0] * qd[jidx]
                qdd_k_loc = (
                    joint_offset_map[k, 0] * qdd[jidx]
                    if qdd is not None
                    else 0.0
                )
                Sqd_loc = S_vec[k] * qd_k_loc
                Xa_parent = (
                    a[k] - S_vec[k] * qdd_k_loc
                    - motion_cross_product(v[k], Sqd_loc)
                )
                grad_q[jidx] -= multiplier * np.dot(
                    a_bar[k], motion_cross_product(S_vec[k], Xa_parent)
                )
        else:
            # Root: dX/dq from a = X * gravity_spatial
            # X * gravity = a[k] - S*qdd_k - crm(v)*S*qd_k  (same formula)
            if jtype != FIXED and jidx >= 0:
                qd_k_loc = joint_offset_map[k, 0] * qd[jidx]
                qdd_k_loc = (
                    joint_offset_map[k, 0] * qdd[jidx]
                    if qdd is not None
                    else 0.0
                )
                Sqd_loc = S_vec[k] * qd_k_loc
                Xa_grav = (
                    a[k] - S_vec[k] * qdd_k_loc
                    - motion_cross_product(v[k], Sqd_loc)
                )
                grad_q[jidx] -= multiplier * np.dot(
                    a_bar[k], motion_cross_product(S_vec[k], Xa_grav)
                )

        # --- [1]' adj of v[k] = X*v[parent] + S*qd ---

        if jtype != FIXED and jidx >= 0:
            # grad_qd += S^T * v_bar   (from S * qd)
            grad_qd[jidx] += multiplier * v_bar[k, s_idx]

        # Propagate v_bar to parent and dX/dq
        if not is_root:
            # v_bar[parent] += X^T * v_bar[k]
            v_bar[parent] += spatial_transform_transpose_multiply(
                R_local[k], p_local[k], v_bar[k]
            )

            # dX/dq: dX/dq * v_parent = -S × (X * v_parent)
            # X * v_parent = v[k] - S*qd_k
            if jtype != FIXED and jidx >= 0:
                qd_k_loc = joint_offset_map[k, 0] * qd[jidx]
                Xv_parent = v[k] - S_vec[k] * qd_k_loc
                grad_q[jidx] -= multiplier * np.dot(
                    v_bar[k], motion_cross_product(S_vec[k], Xv_parent)
                )

    return grad_q, grad_qd, grad_qdd


def rnea_backward_from_kinematics_params(
    tau_bar: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: Optional[np.ndarray],
    v: np.ndarray,
    a: np.ndarray,
    f: np.ndarray,
    kinematics_params,
    gravity: float = -9.81,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run RNEA backward using a KinematicsParams object.

    Args:
        tau_bar: [num_dof] upstream gradient dL/dtau.
        q, qd, qdd: joint positions, velocities, accelerations.
        v, a, f: [num_links, 6] from rnea_from_kinematics_params().
        kinematics_params: curobo KinematicsParams instance.
        gravity: gravity constant.

    Returns:
        grad_q, grad_qd, grad_qdd: [num_dof] each.
    """
    kp = kinematics_params

    offset_flat = kp.joint_offset_map.cpu().numpy()
    num_links = len(kp.link_map)
    joint_offset_map = offset_flat.reshape(num_links, 2)

    return rnea_backward(
        tau_bar=tau_bar,
        q=q,
        qd=qd,
        qdd=qdd,
        v=v,
        a=a,
        f=f,
        fixed_transforms=kp.fixed_transforms.cpu().numpy(),
        link_map=kp.link_map.cpu().numpy(),
        joint_map=kp.joint_map.cpu().numpy(),
        joint_map_type=kp.joint_map_type.cpu().numpy(),
        joint_offset_map=joint_offset_map,
        link_masses_com=kp.link_masses_com.cpu().numpy(),
        link_inertias=kp.link_inertias.cpu().numpy(),
        gravity=gravity,
    )


# ---------------------------------------------------------------------------
# Convenience: run RNEA from KinematicsParams tensors
# ---------------------------------------------------------------------------


def rnea_from_kinematics_params(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: Optional[np.ndarray],
    kinematics_params,
    gravity: float = -9.81,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run RNEA using a KinematicsParams object (tensors converted to numpy).

    Args:
        q: [num_dof] joint positions.
        qd: [num_dof] joint velocities.
        qdd: [num_dof] joint accelerations.
        kinematics_params: curobo KinematicsParams instance.
        gravity: gravity constant.

    Returns:
        tau, v, a, f: see rnea().
    """
    kp = kinematics_params

    # joint_offset_map in curobo is flat [num_links*2] with pairs [multiplier, offset].
    # Reshape to [num_links, 2] for the RNEA function.
    offset_flat = kp.joint_offset_map.cpu().numpy()
    num_links = len(kp.link_map)
    joint_offset_map = offset_flat.reshape(num_links, 2)

    return rnea(
        q=q,
        qd=qd,
        qdd=qdd,
        fixed_transforms=kp.fixed_transforms.cpu().numpy(),
        link_map=kp.link_map.cpu().numpy(),
        joint_map=kp.joint_map.cpu().numpy(),
        joint_map_type=kp.joint_map_type.cpu().numpy(),
        joint_offset_map=joint_offset_map,
        link_masses_com=kp.link_masses_com.cpu().numpy(),
        link_inertias=kp.link_inertias.cpu().numpy(),
        gravity=gravity,
    )
