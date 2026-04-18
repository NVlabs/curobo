# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import warp as wp

# =============================================================================
# Speed Metric Post-Processing Kernel
# =============================================================================


@wp.kernel
def apply_speed_metric(
    spheres: wp.array(dtype=wp.vec4),
    distance: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    speed_dt: wp.array(dtype=wp.float32),
    batch_size: wp.int32,
    horizon: wp.int32,
    num_spheres: wp.int32,
):
    """Apply speed metric scaling as post-processing.

    This kernel scales collision cost by sphere velocity for motion-aware
    collision checking. It uses central differences to estimate velocity
    and adjusts gradients to account for trajectory curvature.

    Should be called once after all obstacles have accumulated their costs.

    Args:
        spheres: Sphere positions and radii.
        distance: Accumulated distance buffer.
        gradient: Accumulated gradient buffer.
        speed_dt: Time delta between trajectory steps.
        batch_size: Number of batch elements.
        horizon: Trajectory horizon.
        num_spheres: Number of spheres per timestep.
    """
    tid = wp.tid()

    b_idx = tid / (horizon * num_spheres)
    h_idx = (tid - b_idx * horizon * num_spheres) / num_spheres

    if b_idx >= batch_size:
        return

    # Skip boundary timesteps (need neighbors for velocity)
    if h_idx == 0 or h_idx >= horizon - 1:
        return

    # Get neighboring sphere positions
    prev_sphere = spheres[tid - num_spheres]
    curr_sphere = spheres[tid]
    next_sphere = spheres[tid + num_spheres]

    dt = speed_dt[0]
    if dt < 1e-6:
        dt = 1e-6

    # Central difference velocity
    prev_pt = wp.vec3(prev_sphere[0], prev_sphere[1], prev_sphere[2])
    curr_pt = wp.vec3(curr_sphere[0], curr_sphere[1], curr_sphere[2])
    next_pt = wp.vec3(next_sphere[0], next_sphere[1], next_sphere[2])

    vel = 0.5 / dt * (next_pt - prev_pt)
    sph_vel = wp.length(vel)

    if sph_vel < 1e-3:
        return

    # Read accumulated values
    d = distance[tid]
    if d <= 0.0:
        return
    g = wp.vec3(gradient[tid * 4], gradient[tid * 4 + 1], gradient[tid * 4 + 2])

    # Compute curvature from acceleration
    acc = (1.0 / (dt * dt)) * (prev_pt + next_pt - 2.0 * curr_pt)
    norm_vel = vel / sph_vel
    curv = acc / (sph_vel * sph_vel)

    # Orthogonal projection: I - v*v^T
    orth_g = g - wp.dot(norm_vel, g) * norm_vel
    orth_curv = curv - wp.dot(norm_vel, curv) * norm_vel

    # Apply speed metric
    new_g = sph_vel * (orth_g - d * orth_curv)
    new_d = sph_vel * d

    # Write back
    distance[tid] = new_d
    gradient[tid * 4 + 0] = new_g[0]
    gradient[tid * 4 + 1] = new_g[1]
    gradient[tid * 4 + 2] = new_g[2]

