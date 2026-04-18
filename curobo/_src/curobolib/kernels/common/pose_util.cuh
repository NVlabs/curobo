/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "third_party/helper_math.h"
#include "quaternion_util.cuh"

namespace curobo {
namespace common {


    /**
     * @brief Pose representation with position and orientation
     *
     * Represents a 6-DOF pose in 3D space with:
     * - position: 3D translation vector
     * - quaternion: 4D rotation quaternion in x y z w format
     */
    struct CuPose
    {
        float3 position;    ///< Translation component (x, y, z)
        float4 quaternion;  ///< Rotation quaternion stored as (x, y, z, w) in float4(x, y, z, w)

        /**
         * @brief Default constructor - initializes to identity pose
         */
        __host__ __device__ __forceinline__ CuPose()
            : position(make_float3(0.0f, 0.0f, 0.0f))
            , quaternion(make_float4(0.0f, 0.0f, 0.0f, 1.0f)) // x=y=z=0, w=1
        {
        }

        /**
         * @brief Constructor with position and quaternion
         * @param pos Position vector
         * @param quat Quaternion in x y z w format
         */
        __host__ __device__ __forceinline__ CuPose(const float3& pos, const float4& quat)
            : position(pos), quaternion(quat)
        {
        }

        /**
         * @brief Constructor with individual position and quaternion components
         * @param px, py, pz Position components
         * @param qw, qx, qy, qz Quaternion components (w, x, y, z) - stored as (x, y, z, w)
         */
        __host__ __device__ __forceinline__ CuPose(float px, float py, float pz,
                                                  float qw, float qx, float qy, float qz)
            : position(make_float3(px, py, pz))
            , quaternion(make_float4(qx, qy, qz, qw))
        {
        }

        /**
         * @brief Transform a point by this pose
         *
         * Applies the pose transformation (rotation + translation) to a 3D point.
         * Formula: transformed_point = R * point + translation
         *
         * @param point Input point to transform
         * @return Transformed point
         */
        __host__ __device__ __forceinline__ float3
        transform_point(const float3& point) const
        {
            float3 rotated_point = quaternion_transform_vector(quaternion, point);
            rotated_point += position;
            return rotated_point;
        }


        __host__ __device__ __forceinline__ float4
        transform_sphere(const float4& sphere) const
        {
            float3 sphere_position = make_float3(sphere.x, sphere.y, sphere.z);
            float3 transformed_position = transform_point(sphere_position);
            return make_float4(transformed_position.x, transformed_position.y, transformed_position.z, sphere.w);
        }
        __host__ __device__ __forceinline__ float3
        transform_vector(const float3& vector) const
        {
            float3 transformed_vector = quaternion_transform_vector(quaternion, vector);
            return transformed_vector;
        }

        __host__ __device__ __forceinline__ float3
        inverse_transform_vector(const float3& vector) const
        {
            float3 transformed_vector = quaternion_transform_vector(quaternion_conjugate(quaternion), vector);
            return transformed_vector;
        }
        /**
         * @brief Compute inverse of this pose
         *
         * For a pose T = [R, t], the inverse is T^-1 = [R^T, -R^T * t]
         * where R^T is the transpose (conjugate for quaternions).
         *
         * @return Inverse pose
         */
        __host__ __device__ __forceinline__ CuPose
        inverse() const
        {
            CuPose result;

            // Inverse rotation: R^-1 = R^T (conjugate for unit quaternions)
            result.quaternion = quaternion_conjugate(quaternion);

            // Inverse translation: t^-1 = -R^T * t
            result.position = quaternion_transform_vector(result.quaternion, -1.0f * position);



            return result;
        }

        /**
         * @brief Multiply this pose with another pose (compose transformations)
         *
         * Computes result = this * other, which applies other first, then this.
         * This is equivalent to: T_result = T_this * T_other
         *
         * @param other Second pose (applied first)
         * @return Composed pose
         */
        __host__ __device__ __forceinline__ CuPose
        multiply(const CuPose& other) const
        {
            CuPose result;

            // Rotation composition: R_result = R_this * R_other
            result.quaternion = quaternion_multiply(quaternion, other.quaternion);
            result.position = quaternion_transform_vector(quaternion, other.position) + position;
            return result;
        }

        /**
         * @brief Convert this pose to a 3x4 transformation matrix
         *
         * @tparam T Scalar type for matrix elements
         * @param transform_mat Output 3x4 transformation matrix [R|t]
         */
        template<typename T>
        __host__ __device__ __forceinline__ void
        to_transform_matrix(T *transform_mat) const
        {
            // Quaternion to rotation matrix conversion
            float4 normalized_quaternion = quaternion_normalize(quaternion);

            const float nw = normalized_quaternion.w;
            const float nx = normalized_quaternion.x;
            const float ny = normalized_quaternion.y;
            const float nz = normalized_quaternion.z;

            // Compute rotation matrix from normalized quaternion
            // R = I + 2*[q_v]×*[q_v]× + 2*qw*[q_v]×
            const float x2 = nx + nx;
            const float y2 = ny + ny;
            const float z2 = nz + nz;
            const float xx = nx * x2;
            const float xy = nx * y2;
            const float xz = nx * z2;
            const float yy = ny * y2;
            const float yz = ny * z2;
            const float zz = nz * z2;
            const float wx = nw * x2;
            const float wy = nw * y2;
            const float wz = nw * z2;

            // Fill the 3x4 transformation matrix [R|t]
            // Row 0
            transform_mat[0*4+0] = static_cast<T>(1.0f - (yy + zz));
            transform_mat[0*4+1] = static_cast<T>(xy - wz);
            transform_mat[0*4+2] = static_cast<T>(xz + wy);
            transform_mat[0*4+3] = static_cast<T>(position.x);

            // Row 1
            transform_mat[1*4+0] = static_cast<T>(xy + wz);
            transform_mat[1*4+1] = static_cast<T>(1.0f - (xx + zz));
            transform_mat[1*4+2] = static_cast<T>(yz - wx);
            transform_mat[1*4+3] = static_cast<T>(position.y);

            // Row 2
            transform_mat[2*4+0] = static_cast<T>(xz - wy);
            transform_mat[2*4+1] = static_cast<T>(yz + wx);
            transform_mat[2*4+2] = static_cast<T>(1.0f - (xx + yy));
            transform_mat[2*4+3] = static_cast<T>(position.z);

        }

        /**
         * @brief Construct CuPose from a 3x4 transformation matrix
         *
         * Extracts position and orientation from a transformation matrix [R|t]
         * where R is the 3x3 rotation matrix and t is the 3x1 translation vector.
         *
         * @tparam T Scalar type of the transformation matrix elements
         * @param transform_mat 3x4 transformation matrix [R|t]
         * @return CuPose constructed from the transformation matrix
         */
        template<typename T>
        __host__ __device__ __forceinline__ static CuPose
        from_transform_matrix(const T* transform_mat)
        {
            CuPose result;

            // Extract translation (last column)
            result.position = make_float3(
                static_cast<float>(transform_mat[0*4+3]),
                static_cast<float>(transform_mat[1*4+3]),
                static_cast<float>(transform_mat[2*4+3])
            );

            quaternion_from_transform_matrix(transform_mat, result.quaternion);


            return result;
        }
        __host__ __device__ __forceinline__ float4
        get_quaternion_as_wxyz()
        {
            return make_float4(quaternion.w, quaternion.x, quaternion.y, quaternion.z);
        }
    };

    /**
     * @brief Multiply two poses using operator* (compose transformations)
     *
     * Computes pose_result = pose1 * pose2, which applies pose2 first, then pose1.
     * This is equivalent to: T_result = T_pose1 * T_pose2
     *
     * @param pose1 First pose (applied second)
     * @param pose2 Second pose (applied first)
     * @return Composed pose
     */
    __host__ __device__ __forceinline__ CuPose
    operator*(const CuPose& pose1, const CuPose& pose2)
    {
        return pose1.multiply(pose2);
    }

    /**
     * @brief Interpolate between two poses
     *
     * @param pose1 First pose
     * @param pose2 Second pose
     * @param interpolation_factor Interpolation factor (0.0 to 1.0)
     * @return Interpolated pose
     */
    __host__ __device__ __forceinline__ CuPose
    pose_interpolate(const CuPose& pose1, const CuPose& pose2, const float interpolation_factor)
    {
        CuPose result;

        // Clamp interpolation factor to [0.0, 1.0] range
        const float t = fmaxf(0.0f, fminf(1.0f, interpolation_factor));

        // Linear interpolation for position: p = p1 + t * (p2 - p1)
        result.position = make_float3(
            pose1.position.x + t * (pose2.position.x - pose1.position.x),
            pose1.position.y + t * (pose2.position.y - pose1.position.y),
            pose1.position.z + t * (pose2.position.z - pose1.position.z)
        );

        // SLERP (Spherical Linear Interpolation) for quaternions
        float4 q1 = pose1.quaternion;
        float4 q2 = pose2.quaternion;

        // Compute dot product of quaternions
        const float dot_product = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;

        // If dot product is negative, flip q2 to take shorter arc
        if (dot_product < 0.0f)
        {
            q2.x = -q2.x;
            q2.y = -q2.y;
            q2.z = -q2.z;
            q2.w = -q2.w;
        }

        const float abs_dot = fabsf(dot_product);
        const float SLERP_THRESHOLD = 0.9995f; // Threshold for switching to linear interpolation

        if (abs_dot > SLERP_THRESHOLD)
        {
            // Quaternions are nearly identical, use linear interpolation and normalize
            result.quaternion = make_float4(
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z),
                q1.w + t * (q2.w - q1.w)
            );

            // Normalize the result quaternion
            result.quaternion = quaternion_normalize(result.quaternion);

        }
        else
        {
            // Use SLERP formula
            const float theta = acosf(abs_dot);           // Angle between quaternions
            const float sin_theta = sinf(theta);         // sin(theta)
            const float inv_sin_theta = 1.0f / sin_theta; // 1/sin(theta)

            // Compute SLERP weights
            const float weight1 = sinf((1.0f - t) * theta) * inv_sin_theta;
            const float weight2 = sinf(t * theta) * inv_sin_theta;

            result.quaternion = make_float4(
                weight1 * q1.x + weight2 * q2.x,
                weight1 * q1.y + weight2 * q2.y,
                weight1 * q1.z + weight2 * q2.z,
                weight1 * q1.w + weight2 * q2.w
            );
        }

        return result;
    }

    __host__ __device__ __forceinline__ void load_pose_from_memory(
        const float *pose_vector,
        common::CuPose& pose)
      {
        pose.position = *(const float3 *)&pose_vector[0];
        pose.quaternion = make_float4(pose_vector[4], pose_vector[5], pose_vector[6], pose_vector[3]);
      }


} // namespace common
} // namespace curobo
