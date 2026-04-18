/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */





namespace curobo{
    namespace kinematics{

// Singularity damping constants (hardcoded for prototyping)
constexpr float SINGULARITY_THRESHOLD = 1.0f;  // ~8.5 degrees
constexpr float MIN_DAMPING = 0.0f;

/**
 * Compute singularity damping factor for a revolute joint.
 * Uses scale-invariant normalized column norm = sin(angle between axis and lever arm).
 *
 * @param cumul_mat_joint Pointer to joint's 3x4 transform matrix [12 floats]
 * @param point_pos World position of the point being affected
 * @param joint_type Joint type (X_ROT, Y_ROT, Z_ROT, or prismatic)
 * @return Damping factor in [MIN_DAMPING, 1.0]
 */
__device__ __forceinline__ float compute_singularity_damping(
    const float* cumul_mat_joint,
    float3 point_pos,
    int joint_type
) {
    return 1.0f;
    // Prismatic joints don't have singularity issues
    if (joint_type < X_ROT || joint_type > Z_ROT) {
        return 1.0f;
    }

    // Extract joint position from transform (translation part)
    float3 joint_pos = make_float3(
        cumul_mat_joint[3],
        cumul_mat_joint[7],
        cumul_mat_joint[11]
    );

    int axis_idx = joint_type - X_ROT;
    float3 axis = make_float3(
        cumul_mat_joint[axis_idx],      // row 0
        cumul_mat_joint[4 + axis_idx],  // row 1
        cumul_mat_joint[8 + axis_idx]   // row 2
    );

    // Lever arm vector
    float3 r = point_pos - joint_pos;
    float r_len = length(r);
    if (r_len < 1e-6f) return 1.0f;  // Point at joint, no damping needed

    // Get joint axis from transform matrix (column corresponding to rotation axis)

    // Normalized column norm = ||axis × r|| / ||r|| = sin(angle between axis and r)
    float3 Jv = cross(axis, r);
    float normalized_col_norm = length(Jv) / (r_len);// / r_len;

    // Smoothstep ramp for C1 continuity
    float x = fminf(normalized_col_norm / SINGULARITY_THRESHOLD, 1.0f);
    return MIN_DAMPING + (1.0f - MIN_DAMPING) * (x * x * (3.0f - 2.0f * x));
}


// Device functions for common computations
template<typename scalar_t, typename psum_t>
__device__ void compute_sphere_gradients(
    psum_t* psum_grad,
    const scalar_t* grad_spheres,
    const float* cumul_mat,
    const float* robotSpheres,
    const int16_t* linkSphereMap,
    const int32_t* env_query_idx,
    const int16_t* linkChainData,
    const int16_t* linkChainOffsets,
    const int8_t* jointMapType,
    const int16_t* jointMap,
    const float* jointOffset,
    int batch, int nspheres, int num_envs, int horizon, int nlinks, int njoints,
    int thread_idx, int threads_per_batch, int matAddrBase)
{
    const int env_idx = (num_envs > 1) ? env_query_idx[batch / horizon] : 0;
    const int sphere_config_offset = env_idx * nspheres * 4;
    int read_cumul_idx = -1;
    const int spheres_perthread = (nspheres + threads_per_batch - 1) / threads_per_batch;

    for (int i = 0; i < spheres_perthread; i++)
    {
        const int sph_idx = thread_idx + i * threads_per_batch;

        if (sph_idx >= nspheres)
        {
            break;
        }
        const int sphAddrs          = sph_idx * 4;
        const int batchAddrs        = batch * nspheres * 4;
        float4    loc_grad_sphere_t = *(float4 *)&grad_spheres[batchAddrs + sphAddrs];

        // Sparsity-based optimization: Skip zero computation
        if ((loc_grad_sphere_t.x == 0) && (loc_grad_sphere_t.y == 0) &&
            (loc_grad_sphere_t.z == 0))
        {
                continue;
        }
        float3 loc_grad_sphere = make_float3(
            loc_grad_sphere_t.x, loc_grad_sphere_t.y, loc_grad_sphere_t.z);

        // read cumul idx:
        read_cumul_idx = linkSphereMap[sph_idx];
        float spheres_mem[4] = {0.0,0.0,0.0,0.0};

        transform_sphere(&cumul_mat[matAddrBase + read_cumul_idx * 12],
                         &robotSpheres[sphere_config_offset + sphAddrs], &spheres_mem[0]);

        // NEW: Use precomputed actual link indices instead of full matrix lookup
        const int16_t chain_start = linkChainOffsets[read_cumul_idx];
        const int16_t chain_end = linkChainOffsets[read_cumul_idx + 1];

        // OPTIMIZED LOOP: Iterate only over links that actually affect this link's gradient
        // Note: iterate backwards to maintain same order as original loop
        for (int chain_idx = chain_end - 1; chain_idx >= chain_start; chain_idx--)
        {
            const int j = linkChainData[chain_idx];  // This is now the actual link index!

            float axis_sign = jointOffset[j*2];

            int j_type = jointMapType[j];
            const int16_t j_idx = jointMap[j];

            float result = 0.0;
            if (j_type >= int(JointType::XRevolute) && j_type <= int(JointType::ZRevolute))
            {
                float3 sphere_position = *(float3 *)&spheres_mem[0];
                xyz_rot_backward_translation(&cumul_mat[matAddrBase + j * 12],
                                             sphere_position,
                                                loc_grad_sphere,
                                                result,
                                                j_type - int(JointType::XRevolute),
                                                axis_sign);
                // Apply singularity damping
                float sing_damp = compute_singularity_damping(
                    &cumul_mat[matAddrBase + j * 12], sphere_position, j_type);
                psum_grad[j_idx] += (psum_t)(result * sing_damp);
            }
            else if ((j_type >= int(JointType::XPrismatic)) && (j_type <= int(JointType::ZPrismatic)))
            {
                xyz_prism_backward(&cumul_mat[matAddrBase + j * 12],
                                   loc_grad_sphere, result, j_type, axis_sign);
                // No damping for prismatic (singularity not applicable)
                psum_grad[j_idx] += (psum_t)result;
            }
        }
    }
}

// Unified link gradients computation with strided thread indexing for better memory coalescing
template<typename psum_t>
__device__ void compute_link_gradients(
    psum_t* psum_grad,
    const float* grad_nlinks_pos,
    const float* grad_nlinks_quat,
    const float* cumul_mat,
    const int16_t* toolFrameMap,
    const int16_t* linkChainData,     // NEW: Packed actual link indices
    const int16_t* linkChainOffsets,  // NEW: Offset for each link's chain
    const int8_t* jointMapType,
    const int16_t* jointMap,
    const float* jointOffset,
    int batch, int n_tool_frames, int nlinks, int njoints,
    int thread_idx, int threads_per_batch, int matAddrBase)
{
    for (int16_t i = 0; i < n_tool_frames; i++)
    {
        const int batchAddrs      = batch * n_tool_frames;
        float3    g_position      = *(float3 *)&grad_nlinks_pos[batchAddrs * 3 + i * 3];

        __align__(16) float g_quaternion[4] = {0.0, 0.0, 0.0, 0.0};

        *(float4 *)&g_quaternion[0] =
            *(float4 *)&grad_nlinks_quat[batchAddrs * 4 + i * 4];

        // Sparsity check
            if ((g_position.x == 0.0f) && (g_position.y == 0.0f) && (g_position.z == 0.0f) &&
                (g_quaternion[0] == 0.0f) && (g_quaternion[1] == 0.0f) &&
                (g_quaternion[2] == 0.0f) && (g_quaternion[3] == 0.0f))
            {
                continue;
            }

        const int16_t l_map = toolFrameMap[i];
        common::CuPose pose = common::CuPose::from_transform_matrix(&cumul_mat[matAddrBase + l_map * 12]);


        // current quaternion:

        // convert quaternion delta to omega:
        float3 g_orientation = make_float3(0.0, 0.0, 0.0);
        common::quaternion_gradient_to_angular_velocity(pose.quaternion, &g_quaternion[0], g_orientation);

        // NEW: Use precomputed actual link indices instead of full matrix lookup
        const int16_t chain_start = linkChainOffsets[l_map];
        const int16_t chain_end = linkChainOffsets[l_map + 1];
        const int16_t chain_length = chain_end - chain_start;

        // Distribute chain processing across threads for better parallelization
        const int16_t links_per_thread = curobo::common::ceil_div(chain_length, threads_per_batch);

        for (int16_t k = 0; k < links_per_thread; k++)
        {
            // Use strided indexing within the chain for better memory coalescing
            int16_t chain_idx = chain_start + k * threads_per_batch + thread_idx;

            if (chain_idx >= chain_end)
                break;

            const int16_t j = linkChainData[chain_idx];  // This is now the actual link index!

            int16_t j_idx  = jointMap[j];
            int     j_type = jointMapType[j];
            float axis_sign = jointOffset[j*2];
            float result = 0.0f;
            if (j_type >= X_ROT && j_type <= Z_ROT)
            {
                xyz_rot_backward(&cumul_mat[matAddrBase + (j) * 12], pose.position,
                               g_position, g_orientation,
                                result, j_type - X_ROT, axis_sign);
                // Apply singularity damping
                float sing_damp = compute_singularity_damping(
                    &cumul_mat[matAddrBase + j * 12], pose.position, j_type);
                psum_grad[j_idx] += (psum_t)(result * sing_damp);
            }
            else if (j_type >= X_PRISM && j_type <= Z_PRISM)
            {
                xyz_prism_backward(&cumul_mat[matAddrBase + j * 12],
                                               g_position, result,
                                               j_type, axis_sign);
                // No damping for prismatic (singularity not applicable)
                psum_grad[j_idx] += (psum_t)result;
            }

        }
    }
}

// Center of mass gradient computation using existing gradient functions
template<typename psum_t>
__device__ void compute_center_of_mass_gradients(
    psum_t* psum_grad,                // Joint gradients accumulator
    const float* grad_center_of_mass, // [batch_size * 4] - xyz=pos grad, w=mass grad (ignored)
    const float* cumul_mat,           // Link transformations
    const float* link_masses_com,     // [n_links * 4] - xyz=local CoM, w=mass
    const float total_mass,           // Total robot mass for this batch
    const int16_t* linkChainData,     // Packed actual link indices
    const int16_t* linkChainOffsets,  // Offset for each link's chain
    const int8_t* jointMapType,       // Joint types
    const int16_t* jointMap,          // Joint indices
    const float* jointOffset,         // Joint offsets
    int batch, int n_links, int n_joints,
    int thread_idx, int threads_per_batch, int matAddrBase
) {
    const int links_per_thread = (n_links + threads_per_batch - 1) / threads_per_batch;

    // Get input gradient (ignore w component) - read from float array
    const int grad_addr = batch * 4;
    float3 grad_com = make_float3(
        grad_center_of_mass[grad_addr + 0],  // x gradient
        grad_center_of_mass[grad_addr + 1],  // y gradient
        grad_center_of_mass[grad_addr + 2]   // z gradient
        // ignore grad_center_of_mass[grad_addr + 3] (w gradient)
    );

    // Sparsity check (same as existing kernels)
    if (grad_com.x == 0.0f && grad_com.y == 0.0f && grad_com.z == 0.0f) {
        return;
    }

    // Each thread processes a subset of links
    for (int i = 0; i < links_per_thread; i++) {
        const int link_idx = thread_idx + i * threads_per_batch;

        if (link_idx >= n_links) {
            break;
        }

        // Load link mass and local center of mass from float array (4 floats per link)
        const int link_addr = link_idx * 4;
        float4 mass_com = make_float4(
            link_masses_com[link_addr + 0],  // x - local CoM x
            link_masses_com[link_addr + 1],  // y - local CoM y
            link_masses_com[link_addr + 2],  // z - local CoM z
            link_masses_com[link_addr + 3]   // w - mass
        );
        float mass = mass_com.w;

        if (mass <= 0.0f) {
            continue;  // Skip massless links
        }

        // Gradient contribution from this link to overall CoM
        // d(CoM)/d(com_link) = mass / total_mass
        float3 link_grad_contribution = make_float3(
            grad_com.x * mass / total_mass,
            grad_com.y * mass / total_mass,
            grad_com.z * mass / total_mass
        );

        // Transform local CoM to world for gradient computation
        float4 com_world;
        transform_sphere_float4(&cumul_mat[matAddrBase + link_idx * 12], mass_com, com_world);
        float3 com_world_pos = make_float3(com_world);

        // Use existing chain traversal pattern (same as compute_sphere_gradients)
        const int16_t chain_start = linkChainOffsets[link_idx];
        const int16_t chain_end = linkChainOffsets[link_idx + 1];

        // Same backwards traversal as existing gradients
        for (int chain_idx = chain_end - 1; chain_idx >= chain_start; chain_idx--) {
            const int j = linkChainData[chain_idx];

            float axis_sign = jointOffset[j * 2];
            int j_type = jointMapType[j];
            const int16_t j_idx = jointMap[j];

            float result = 0.0;

            if (j_type >= X_ROT && j_type <= Z_ROT) {
                // Use existing function directly!
                xyz_rot_backward_translation(
                    &cumul_mat[matAddrBase + j * 12],
                    com_world_pos,                    // Position as float3
                    link_grad_contribution,           // Gradient as float3
                    result,                          // Output gradient
                    j_type - X_ROT,                  // Axis index
                    axis_sign                        // Sign
                );
                // Apply singularity damping
                float sing_damp = compute_singularity_damping(
                    &cumul_mat[matAddrBase + j * 12], com_world_pos, j_type);
                psum_grad[j_idx] += (psum_t)(result * sing_damp);
            }
            else if (j_type >= X_PRISM && j_type <= Z_PRISM) {
                // Use existing function directly!
                xyz_prism_backward(
                    &cumul_mat[matAddrBase + j * 12],
                    link_grad_contribution,          // Gradient as float3
                    result,                         // Output gradient
                    j_type,               // Axis index
                    axis_sign                       // Sign
                );
                // No damping for prismatic (singularity not applicable)
                psum_grad[j_idx] += (psum_t)result;
            }
        }
    }
}
    }
}
