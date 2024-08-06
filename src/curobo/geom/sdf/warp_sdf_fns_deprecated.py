#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Deprecated warp kernels that use API compatibile with warp-lang < 1.0.0"""

# Third Party
import warp as wp


@wp.kernel
def get_swept_closest_pt_batch_env(
    pt: wp.array(dtype=wp.vec4),
    distance: wp.array(dtype=wp.float32),  # this stores the output cost
    closest_pt: wp.array(dtype=wp.float32),  # this stores the gradient
    sparsity_idx: wp.array(dtype=wp.uint8),
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),  # eta threshold
    speed_dt: wp.array(dtype=wp.float32),
    mesh: wp.array(dtype=wp.uint64),
    mesh_pose: wp.array(dtype=wp.float32),
    mesh_enable: wp.array(dtype=wp.uint8),
    n_env_mesh: wp.array(dtype=wp.int32),
    max_dist: wp.array(dtype=wp.float32),
    write_grad: wp.uint8,
    batch_size: wp.int32,
    horizon: wp.int32,
    nspheres: wp.int32,
    max_nmesh: wp.int32,
    sweep_steps: wp.uint8,
    enable_speed_metric: wp.uint8,
    env_query_idx: wp.array(dtype=wp.int32),
    use_batch_env: wp.uint8,
):
    """Kernel to compute swept closest point to mesh."""
    # we launch nspheres kernels
    # compute gradient here and return
    # distance is negative outside and positive inside
    tid = int(0)
    tid = wp.tid()

    b_idx = int(0)
    h_idx = int(0)
    sph_idx = int(0)
    # read horizon

    b_idx = tid / (horizon * nspheres)

    h_idx = (tid - (b_idx * (horizon * nspheres))) / nspheres
    sph_idx = tid - (b_idx * horizon * nspheres) - (h_idx * nspheres)
    if b_idx >= batch_size or h_idx >= horizon or sph_idx >= nspheres:
        return
    uint_zero = wp.uint8(0)
    uint_one = wp.uint8(1)
    env_idx = int(0)
    n_mesh = int(0)
    # read sphere
    sphere_0_distance = float(0.0)
    sphere_2_distance = float(0.0)

    sphere_0 = wp.vec3(0.0)
    sphere_2 = wp.vec3(0.0)
    sphere_int = wp.vec3(0.0)
    sphere_temp = wp.vec3(0.0)
    grad_vec = wp.vec3(0.0)
    eta = float(0.0)
    dt = float(0.0)
    k0 = float(0.0)
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    dist = float(0.0)
    dist_metric = float(0.0)
    euclidean_distance = float(0.0)
    cl_pt = wp.vec3(0.0)
    local_pt = wp.vec3(0.0)
    in_sphere = pt[b_idx * horizon * nspheres + (h_idx * nspheres) + sph_idx]
    in_rad = in_sphere[3]
    if in_rad < 0.0:
        distance[tid] = 0.0
        if write_grad == 1 and sparsity_idx[tid] == uint_one:
            sparsity_idx[tid] = uint_zero
            closest_pt[tid * 4] = 0.0
            closest_pt[tid * 4 + 1] = 0.0
            closest_pt[tid * 4 + 2] = 0.0

        return
    dt = speed_dt[0]
    eta = activation_distance[0]
    in_rad += eta
    max_dist_buffer = float(0.0)
    max_dist_buffer = max_dist[0]
    if (in_rad) > max_dist_buffer:
        max_dist_buffer += in_rad

    in_pt = wp.vec3(in_sphere[0], in_sphere[1], in_sphere[2])
    # read in sphere 0:
    if h_idx > 0:
        in_sphere = pt[b_idx * horizon * nspheres + ((h_idx - 1) * nspheres) + sph_idx]
        sphere_0 += wp.vec3(in_sphere[0], in_sphere[1], in_sphere[2])
        sphere_0_distance = wp.length(sphere_0 - in_pt) / 2.0
    if h_idx < horizon - 1:
        in_sphere = pt[b_idx * horizon * nspheres + ((h_idx + 1) * nspheres) + sph_idx]
        sphere_2 += wp.vec3(in_sphere[0], in_sphere[1], in_sphere[2])
        sphere_2_distance = wp.length(sphere_2 - in_pt) / 2.0

    # read in sphere 2:
    closest_distance = float(0.0)
    closest_point = wp.vec3(0.0)
    i = int(0)
    dis_length = float(0.0)
    jump_distance = float(0.0)
    mid_distance = float(0.0)
    if use_batch_env:
        env_idx = env_query_idx[b_idx]
    i = max_nmesh * env_idx
    n_mesh = i + n_env_mesh[env_idx]
    obj_position = wp.vec3()

    while i < n_mesh:
        if mesh_enable[i] == uint_one:
            # transform point to mesh frame:
            # mesh_pt = T_inverse @ w_pt
            obj_position[0] = mesh_pose[i * 8 + 0]
            obj_position[1] = mesh_pose[i * 8 + 1]
            obj_position[2] = mesh_pose[i * 8 + 2]
            obj_quat = wp.quaternion(
                mesh_pose[i * 8 + 4],
                mesh_pose[i * 8 + 5],
                mesh_pose[i * 8 + 6],
                mesh_pose[i * 8 + 3],
            )

            obj_w_pose = wp.transform(obj_position, obj_quat)
            obj_w_pose_t = wp.transform_inverse(obj_w_pose)
            local_pt = wp.transform_point(obj_w_pose, in_pt)

            if wp.mesh_query_point(
                mesh[i], local_pt, max_dist_buffer, sign, face_index, face_u, face_v
            ):
                cl_pt = wp.mesh_eval_position(mesh[i], face_index, face_u, face_v)
                delta = cl_pt - local_pt
                dis_length = wp.length(delta)
                dist = (-1.0 * dis_length * sign) + in_rad
                if dist > 0:
                    if dist == in_rad:
                        cl_pt = sign * (delta) / (dist)
                    else:
                        cl_pt = sign * (delta) / dis_length
                    euclidean_distance = dist
                    if dist > eta:
                        dist_metric = dist - 0.5 * eta
                    elif dist <= eta:
                        dist_metric = (0.5 / eta) * (dist) * dist
                        cl_pt = (1.0 / eta) * dist * cl_pt
                    grad_vec = wp.transform_vector(obj_w_pose_t, cl_pt)

                    closest_distance += dist_metric
                    closest_point += grad_vec
                else:
                    dist = -1.0 * dist
                    euclidean_distance = dist
            else:
                dist = max_dist_buffer
                euclidean_distance = dist
            dist = max(euclidean_distance - in_rad, in_rad)

            mid_distance = euclidean_distance
            # transform sphere -1
            if h_idx > 0 and mid_distance < sphere_0_distance:
                jump_distance = mid_distance
                j = int(0)
                sphere_temp = wp.transform_point(obj_w_pose, sphere_0)
                while j < sweep_steps:
                    k0 = (
                        1.0 - 0.5 * jump_distance / sphere_0_distance
                    )  # dist could be greater than sphere_0_distance here?
                    sphere_int = k0 * local_pt + ((1.0 - k0) * sphere_temp)
                    if wp.mesh_query_point(
                        mesh[i], sphere_int, max_dist_buffer, sign, face_index, face_u, face_v
                    ):
                        cl_pt = wp.mesh_eval_position(mesh[i], face_index, face_u, face_v)
                        delta = cl_pt - sphere_int
                        dis_length = wp.length(delta)
                        dist = (-1.0 * dis_length * sign) + in_rad
                        if dist > 0:
                            if dist == in_rad:
                                cl_pt = sign * (delta) / (dist)
                            else:
                                cl_pt = sign * (delta) / dis_length
                            euclidean_distance = dist
                            if dist > eta:
                                dist_metric = dist - 0.5 * eta
                            elif dist <= eta:
                                dist_metric = (0.5 / eta) * (dist) * dist
                                cl_pt = (1.0 / eta) * dist * cl_pt

                            closest_distance += dist_metric
                            grad_vec = wp.transform_vector(obj_w_pose_t, cl_pt)

                            closest_point += grad_vec
                            dist = max(euclidean_distance - in_rad, in_rad)
                            jump_distance += euclidean_distance
                        else:
                            dist = max(-dist - in_rad, in_rad)
                            euclidean_distance = dist
                            jump_distance += euclidean_distance
                    else:
                        jump_distance += max_dist_buffer
                    j += 1
                    if jump_distance >= sphere_0_distance:
                        j = int(sweep_steps)
            # transform sphere -1
            if h_idx < horizon - 1 and mid_distance < sphere_2_distance:
                jump_distance = mid_distance
                j = int(0)
                sphere_temp = wp.transform_point(obj_w_pose, sphere_2)
                while j < sweep_steps:
                    k0 = (
                        1.0 - 0.5 * jump_distance / sphere_2_distance
                    )  # dist could be greater than sphere_0_distance here?
                    sphere_int = k0 * local_pt + (1.0 - k0) * sphere_temp

                    if wp.mesh_query_point(
                        mesh[i], sphere_int, max_dist_buffer, sign, face_index, face_u, face_v
                    ):
                        cl_pt = wp.mesh_eval_position(mesh[i], face_index, face_u, face_v)
                        delta = cl_pt - sphere_int
                        dis_length = wp.length(delta)
                        dist = (-1.0 * dis_length * sign) + in_rad
                        if dist > 0:
                            euclidean_distance = dist
                            if dist == in_rad:
                                cl_pt = sign * (delta) / (dist)
                            else:
                                cl_pt = sign * (delta) / dis_length
                            # cl_pt = sign * (delta) / dis_length
                            if dist > eta:
                                dist_metric = dist - 0.5 * eta
                            elif dist <= eta:
                                dist_metric = (0.5 / eta) * (dist) * dist
                                cl_pt = (1.0 / eta) * dist * cl_pt
                            closest_distance += dist_metric
                            grad_vec = wp.transform_vector(obj_w_pose_t, cl_pt)

                            closest_point += grad_vec
                            dist = max(euclidean_distance - in_rad, in_rad)
                            jump_distance += dist

                        else:
                            dist = max(-dist - in_rad, in_rad)
                            jump_distance += dist
                    else:
                        jump_distance += max_dist_buffer

                    j += 1
                    if jump_distance >= sphere_2_distance:
                        j = int(sweep_steps)
        i += 1

    # return
    if closest_distance <= 0.0:
        if sparsity_idx[tid] == uint_zero:
            return
        sparsity_idx[tid] = uint_zero
        distance[tid] = 0.0
        if write_grad == 1:
            closest_pt[tid * 4 + 0] = 0.0
            closest_pt[tid * 4 + 1] = 0.0
            closest_pt[tid * 4 + 2] = 0.0

        return
    if enable_speed_metric == 1 and (h_idx > 0 and h_idx < horizon - 1):
        # calculate sphere velocity and acceleration:
        norm_vel_vec = wp.vec3(0.0)
        sph_acc_vec = wp.vec3(0.0)
        sph_vel = wp.float(0.0)

        # use central difference
        norm_vel_vec = (0.5 / dt) * (sphere_2 - sphere_0)
        sph_vel = wp.length(norm_vel_vec)
        if sph_vel > 1e-3:
            sph_acc_vec = (1.0 / (dt * dt)) * (sphere_0 + sphere_2 - 2.0 * in_pt)

            norm_vel_vec = norm_vel_vec * (1.0 / sph_vel)

            curvature_vec = sph_acc_vec / (sph_vel * sph_vel)

            orth_proj = wp.mat33(0.0)
            for i in range(3):
                for j in range(3):
                    orth_proj[i, j] = -1.0 * norm_vel_vec[i] * norm_vel_vec[j]

            orth_proj[0, 0] = orth_proj[0, 0] + 1.0
            orth_proj[1, 1] = orth_proj[1, 1] + 1.0
            orth_proj[2, 2] = orth_proj[2, 2] + 1.0

            orth_curv = wp.vec3(
                0.0, 0.0, 0.0
            )  # closest_distance * (orth_proj @ curvature_vec) #wp.matmul(orth_proj, curvature_vec)
            orth_pt = wp.vec3(0.0, 0.0, 0.0)  # orth_proj @ closest_point
            for i in range(3):
                orth_pt[i] = (
                    orth_proj[i, 0] * closest_point[0]
                    + orth_proj[i, 1] * closest_point[1]
                    + orth_proj[i, 2] * closest_point[2]
                )

                orth_curv[i] = closest_distance * (
                    orth_proj[i, 0] * curvature_vec[0]
                    + orth_proj[i, 1] * curvature_vec[1]
                    + orth_proj[i, 2] * curvature_vec[2]
                )

            closest_point = sph_vel * (orth_pt - orth_curv)

            closest_distance = sph_vel * closest_distance

    distance[tid] = weight[0] * closest_distance
    sparsity_idx[tid] = uint_one
    if write_grad == 1:
        # compute gradient:
        closest_distance = weight[0]
        closest_pt[tid * 4 + 0] = closest_distance * closest_point[0]
        closest_pt[tid * 4 + 1] = closest_distance * closest_point[1]
        closest_pt[tid * 4 + 2] = closest_distance * closest_point[2]


@wp.kernel
def get_closest_pt_batch_env(
    pt: wp.array(dtype=wp.vec4),
    distance: wp.array(dtype=wp.float32),  # this stores the output cost
    closest_pt: wp.array(dtype=wp.float32),  # this stores the gradient
    sparsity_idx: wp.array(dtype=wp.uint8),
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),  # eta threshold
    mesh: wp.array(dtype=wp.uint64),
    mesh_pose: wp.array(dtype=wp.float32),
    mesh_enable: wp.array(dtype=wp.uint8),
    n_env_mesh: wp.array(dtype=wp.int32),
    max_dist: wp.array(dtype=wp.float32),
    write_grad: wp.uint8,
    batch_size: wp.int32,
    horizon: wp.int32,
    nspheres: wp.int32,
    max_nmesh: wp.int32,
    env_query_idx: wp.array(dtype=wp.int32),
    use_batch_env: wp.uint8,
    compute_esdf: wp.uint8,
):
    """Kernel to compute closest point to mesh."""
    # we launch nspheres kernels
    # compute gradient here and return
    # distance is negative outside and positive inside
    tid = wp.tid()
    n_mesh = int(0)
    b_idx = int(0)
    h_idx = int(0)
    sph_idx = int(0)

    env_idx = int(0)

    b_idx = tid / (horizon * nspheres)

    h_idx = (tid - (b_idx * (horizon * nspheres))) / nspheres
    sph_idx = tid - (b_idx * horizon * nspheres) - (h_idx * nspheres)
    if b_idx >= batch_size or h_idx >= horizon or sph_idx >= nspheres:
        return

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    dist = float(0.0)
    grad_vec = wp.vec3(0.0)
    eta = float(0.0)
    dist_metric = float(0.0)
    max_dist_buffer = float(0.0)
    uint_zero = wp.uint8(0)
    uint_one = wp.uint8(1)
    cl_pt = wp.vec3(0.0)
    local_pt = wp.vec3(0.0)
    in_sphere = pt[tid]
    in_pt = wp.vec3(in_sphere[0], in_sphere[1], in_sphere[2])
    in_rad = in_sphere[3]
    if in_rad < 0.0:
        distance[tid] = 0.0
        if write_grad == 1 and sparsity_idx[tid] == uint_one:
            sparsity_idx[tid] = uint_zero
            closest_pt[tid * 4] = 0.0
            closest_pt[tid * 4 + 1] = 0.0
            closest_pt[tid * 4 + 2] = 0.0

        return
    eta = activation_distance[0]
    if compute_esdf != 1:
        in_rad += eta
    max_dist_buffer = max_dist[0]
    if (in_rad) > max_dist_buffer:
        max_dist_buffer += in_rad

    # TODO: read vec4 and use first 3 for sphere position and last one for radius
    # in_pt = pt[tid]
    closest_distance = float(0.0)
    closest_point = wp.vec3(0.0)
    dis_length = float(0.0)

    # read env index:
    if use_batch_env:
        env_idx = env_query_idx[b_idx]

    # read number of boxes in current environment:

    # get start index
    i = int(0)
    i = max_nmesh * env_idx
    n_mesh = i + n_env_mesh[env_idx]
    obj_position = wp.vec3()
    max_dist_value = -1.0 * max_dist_buffer
    if compute_esdf == 1:
        closest_distance = max_dist_value
    while i < n_mesh:
        if mesh_enable[i] == uint_one:
            # transform point to mesh frame:
            # mesh_pt = T_inverse @ w_pt
            obj_position[0] = mesh_pose[i * 8 + 0]
            obj_position[1] = mesh_pose[i * 8 + 1]
            obj_position[2] = mesh_pose[i * 8 + 2]
            obj_quat = wp.quaternion(
                mesh_pose[i * 8 + 4],
                mesh_pose[i * 8 + 5],
                mesh_pose[i * 8 + 6],
                mesh_pose[i * 8 + 3],
            )

            obj_w_pose = wp.transform(obj_position, obj_quat)

            local_pt = wp.transform_point(obj_w_pose, in_pt)

            if wp.mesh_query_point(
                mesh[i], local_pt, max_dist_buffer, sign, face_index, face_u, face_v
            ):
                cl_pt = wp.mesh_eval_position(mesh[i], face_index, face_u, face_v)
                delta = cl_pt - local_pt
                dis_length = wp.length(delta)
                dist = -1.0 * dis_length * sign

                if compute_esdf == 1:
                    if dist > max_dist_value:
                        max_dist_value = dist
                        closest_distance = dist
                        if write_grad == 1:
                            cl_pt = sign * (delta) / dis_length
                            grad_vec = wp.transform_vector(wp.transform_inverse(obj_w_pose), cl_pt)
                            closest_point = grad_vec

                else:
                    dist = dist + in_rad

                    if dist > 0:
                        if dist == in_rad:
                            cl_pt = sign * (delta) / (dist)
                        else:
                            cl_pt = sign * (delta) / dis_length
                        if dist > eta:
                            dist_metric = dist - 0.5 * eta
                        elif dist <= eta:
                            dist_metric = (0.5 / eta) * (dist) * dist
                            cl_pt = (1.0 / eta) * dist * cl_pt
                        grad_vec = wp.transform_vector(wp.transform_inverse(obj_w_pose), cl_pt)

                        closest_distance += dist_metric
                        closest_point += grad_vec
        i += 1

    if closest_distance == 0 and compute_esdf != 1:
        if sparsity_idx[tid] == uint_zero:
            return
        sparsity_idx[tid] = uint_zero
        distance[tid] = 0.0
        if write_grad == 1:
            closest_pt[tid * 4 + 0] = 0.0
            closest_pt[tid * 4 + 1] = 0.0
            closest_pt[tid * 4 + 2] = 0.0
    else:
        distance[tid] = weight[0] * closest_distance
        sparsity_idx[tid] = uint_one
        if write_grad == 1:
            # compute gradient:
            closest_distance = weight[0]
            closest_pt[tid * 4 + 0] = closest_distance * closest_point[0]
            closest_pt[tid * 4 + 1] = closest_distance * closest_point[1]
            closest_pt[tid * 4 + 2] = closest_distance * closest_point[2]
