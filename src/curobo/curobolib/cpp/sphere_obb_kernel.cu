/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "helper_math.h"
#include <cuda_fp16.h>
#include <vector>
#define M 4

// #define MAX_WARP_THREADS 512 // warp level batching. 8 x M = 32
#define MAX_BOX_SHARED 256 // maximum number of boxes we can store distance and closest point
#define DEBUG false
namespace Curobo
{
  namespace Geometry
  {
    /**
     * @brief Compute length of sphere
     *
     * @param v1
     * @param v2
     * @return float
     */
    __device__ __forceinline__ float sphere_length(const float4& v1,
                                                   const float4& v2)
    {
      return norm3df(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
    }

    __device__ __forceinline__ float sphere_distance(const float4& v1,
                                                     const float4& v2)
    {
      return max(0.0f, sphere_length(v1, v2) - v1.w - v2.w);
    }

    template<typename scalar_t>
    __device__ __forceinline__ void load_obb_pose(const scalar_t *obb_mat,
                                                  float3& position, float4& quat)
    { // obb_mat has x,y,z, qw, qx, qy, qz, 0 with an extra 0 padding for better use of memory
      float4 temp = *(float4 *)&obb_mat[0];

      position.x = temp.x;
      position.y = temp.y;
      position.z = temp.z;
      quat.w     = temp.w;
      temp       = *(float4 *)&obb_mat[4];
      quat.x     = temp.x;
      quat.y     = temp.y;
      quat.z     = temp.z;
    }

    template<typename scalar_t>
    __device__ __forceinline__ void load_obb_bounds(const scalar_t *obb_bounds,
                                                    float3        & bounds)
    { // obb_bounds has x,y,z, 0 with an extra 0 padding.
      float4 loc_bounds = *(float4 *)&obb_bounds[0];

      bounds.x = loc_bounds.x / 2;
      bounds.y = loc_bounds.y / 2;
      bounds.z = loc_bounds.z / 2;
    }

    template<typename scalar_t>
    __device__ __forceinline__ void
    transform_sphere_quat(const scalar_t *transform_mat, // x,y,z, qw, qx,qy,qz
                          const float4& sphere_pos, float4& C)
    {
      // do dot product:
      // new_p = q * p * q_inv + obs_p
      const float3   p_arr = *(float3 *)&transform_mat[0];
      const scalar_t w     = transform_mat[3];
      const scalar_t x     = transform_mat[4];
      const scalar_t y     = transform_mat[5];
      const scalar_t z     = transform_mat[6];

      if ((x != 0) || (y != 0) || (z != 0))
      {
        C.x = p_arr.x + w * w * sphere_pos.x + 2 * y * w * sphere_pos.z -
              2 * z * w * sphere_pos.y + x * x * sphere_pos.x +
              2 * y * x * sphere_pos.y + 2 * z * x * sphere_pos.z -
              z * z * sphere_pos.x - y * y * sphere_pos.x;
        C.y = p_arr.y + 2 * x * y * sphere_pos.x + y * y * sphere_pos.y +
              2 * z * y * sphere_pos.z + 2 * w * z * sphere_pos.x -
              z * z * sphere_pos.y + w * w * sphere_pos.y - 2 * x * w * sphere_pos.z -
              x * x * sphere_pos.y;
        C.z = p_arr.z + 2 * x * z * sphere_pos.x + 2 * y * z * sphere_pos.y +
              z * z * sphere_pos.z - 2 * w * y * sphere_pos.x - y * y * sphere_pos.z +
              2 * w * x * sphere_pos.y - x * x * sphere_pos.z + w * w * sphere_pos.z;
      }
      else
      {
        C.x = p_arr.x + sphere_pos.x;
        C.y = p_arr.y + sphere_pos.y;
        C.z = p_arr.z + sphere_pos.z;
      }
      C.w = sphere_pos.w;
    }

    __device__ __forceinline__ void transform_sphere_quat(const float3  p,
                                                          const float4  q,
                                                          const float4& sphere_pos,
                                                          float4      & C)
    {
      // do dot product:
      // new_p = q * p * q_inv + obs_p

      if ((q.x != 0) || (q.y != 0) || (q.z != 0))
      {
        C.x = p.x + q.w * q.w * sphere_pos.x + 2 * q.y * q.w * sphere_pos.z -
              2 * q.z * q.w * sphere_pos.y + q.x * q.x * sphere_pos.x +
              2 * q.y * q.x * sphere_pos.y + 2 * q.z * q.x * sphere_pos.z -
              q.z * q.z * sphere_pos.x - q.y * q.y * sphere_pos.x;
        C.y = p.y + 2 * q.x * q.y * sphere_pos.x + q.y * q.y * sphere_pos.y +
              2 * q.z * q.y * sphere_pos.z + 2 * q.w * q.z * sphere_pos.x -
              q.z * q.z * sphere_pos.y + q.w * q.w * sphere_pos.y - 2 * q.x * q.w * sphere_pos.z -
              q.x * q.x * sphere_pos.y;
        C.z = p.z + 2 * q.x * q.z * sphere_pos.x + 2 * q.y * q.z * sphere_pos.y +
              q.z * q.z * sphere_pos.z - 2 * q.w * q.y * sphere_pos.x - q.y * q.y * sphere_pos.z +
              2 * q.w * q.x * sphere_pos.y - q.x * q.x * sphere_pos.z + q.w * q.w * sphere_pos.z;
      }
      else
      {
        C.x = p.x + sphere_pos.x;
        C.y = p.y + sphere_pos.y;
        C.z = p.z + sphere_pos.z;
      }
      C.w = sphere_pos.w;
    }

    __device__ __forceinline__ void
    inv_transform_vec_quat(
      const float3 p,
      const float4 q,
      const float4& sphere_pos, float3& C)
    {
      // do dot product:
      // new_p = q * p * q_inv + obs_p
      if ((q.x != 0) || (q.y != 0) || (q.z != 0))
      {
        C.x =  q.w *  q.w * sphere_pos.x - 2 * q.y *  q.w * sphere_pos.z +
              2 * q.z *  q.w * sphere_pos.y + q.x * q.x * sphere_pos.x +
              2 * q.y * q.x * sphere_pos.y + 2 * q.z * q.x * sphere_pos.z -
              q.z * q.z * sphere_pos.x - q.y * q.y * sphere_pos.x;
        C.y = 2 * q.x * q.y * sphere_pos.x + q.y * q.y * sphere_pos.y +
              2 * q.z * q.y * sphere_pos.z - 2 *  q.w * q.z * sphere_pos.x -
              q.z * q.z * sphere_pos.y +  q.w *  q.w * sphere_pos.y + 2 * q.x *  q.w *
              sphere_pos.z -
              q.x * q.x * sphere_pos.y;
        C.z = 2 * q.x * q.z * sphere_pos.x + 2 * q.y * q.z * sphere_pos.y +
              q.z * q.z * sphere_pos.z + 2 *  q.w * q.y * sphere_pos.x - q.y * q.y * sphere_pos.z -
              2 *  q.w * q.x * sphere_pos.y - q.x * q.x * sphere_pos.z +  q.w *  q.w * sphere_pos.z;
      }
      else
      {
        C.x = sphere_pos.x;
        C.y = sphere_pos.y;
        C.z = sphere_pos.z;
      }
    }

    __device__ __forceinline__ void
    inv_transform_vec_quat_add(const float3 p,
                               const float4 q, // x,y,z, qw, qx,qy,qz
                               const float4& sphere_pos, float3& C)
    {
      // do dot product:
      // new_p = q * p * q_inv + obs_p
      float3 temp_C = make_float3(0.0);

      inv_transform_vec_quat(p, q, sphere_pos, temp_C);
      C = C + temp_C;
    }

    template<typename scalar_t>
    __device__ __forceinline__ void
    inv_transform_vec_quat(const scalar_t *transform_mat, // x,y,z, qw, qx,qy,qz
                           const float4& sphere_pos, float3& C)
    {
      // do dot product:
      // new_p = q * p * q_inv + obs_p
      const scalar_t w = transform_mat[3];
      const scalar_t x = -1 * transform_mat[4];
      const scalar_t y = -1 * transform_mat[5];
      const scalar_t z = -1 * transform_mat[6];

      if ((x != 0) || (y != 0) || (z != 0))
      {
        C.x = w * w * sphere_pos.x + 2 * y * w * sphere_pos.z -
              2 * z * w * sphere_pos.y + x * x * sphere_pos.x +
              2 * y * x * sphere_pos.y + 2 * z * x * sphere_pos.z -
              z * z * sphere_pos.x - y * y * sphere_pos.x;
        C.y = 2 * x * y * sphere_pos.x + y * y * sphere_pos.y +
              2 * z * y * sphere_pos.z + 2 * w * z * sphere_pos.x -
              z * z * sphere_pos.y + w * w * sphere_pos.y - 2 * x * w * sphere_pos.z -
              x * x * sphere_pos.y;
        C.z = 2 * x * z * sphere_pos.x + 2 * y * z * sphere_pos.y +
              z * z * sphere_pos.z - 2 * w * y * sphere_pos.x - y * y * sphere_pos.z +
              2 * w * x * sphere_pos.y - x * x * sphere_pos.z + w * w * sphere_pos.z;
      }
      else
      {
        C.x = sphere_pos.x;
        C.y = sphere_pos.y;
        C.z = sphere_pos.z;
      }
    }

    template<typename scalar_t>
    __device__ __forceinline__ void
    inv_transform_vec_quat_add(const scalar_t *transform_mat, // x,y,z, qw, qx,qy,qz
                               const float4& sphere_pos, float3& C)
    {
      // do dot product:
      // new_p = q * p * q_inv + obs_p
      const scalar_t w = transform_mat[3];
      const scalar_t x = -1 * transform_mat[4];
      const scalar_t y = -1 * transform_mat[5];
      const scalar_t z = -1 * transform_mat[6];

      if ((x != 0) || (y != 0) || (z != 0))
      {
        C.x += w * w * sphere_pos.x + 2 * y * w * sphere_pos.z -
               2 * z * w * sphere_pos.y + x * x * sphere_pos.x +
               2 * y * x * sphere_pos.y + 2 * z * x * sphere_pos.z -
               z * z * sphere_pos.x - y * y * sphere_pos.x;
        C.y += 2 * x * y * sphere_pos.x + y * y * sphere_pos.y +
               2 * z * y * sphere_pos.z + 2 * w * z * sphere_pos.x -
               z * z * sphere_pos.y + w * w * sphere_pos.y -
               2 * x * w * sphere_pos.z - x * x * sphere_pos.y;
        C.z += 2 * x * z * sphere_pos.x + 2 * y * z * sphere_pos.y +
               z * z * sphere_pos.z - 2 * w * y * sphere_pos.x -
               y * y * sphere_pos.z + 2 * w * x * sphere_pos.y -
               x * x * sphere_pos.z + w * w * sphere_pos.z;
      }
      {
        C.x += sphere_pos.x;
        C.y += sphere_pos.y;
        C.z += sphere_pos.z;
      }
    }

    /**
     * @brief Scales the Collision across the trajectory by sphere velocity. This is
     * implemented from CHOMP motion planner (ICRA 2009). We use central difference
     * to compute the velocity and acceleration of the sphere.
     *
     * @param sphere_0_cache
     * @param sphere_1_cache
     * @param sphere_2_cache
     * @param dt
     * @param transform_back
     * @param max_dist
     * @param max_grad
     * @return void
     */
    __device__ __forceinline__ void
    scale_speed_metric(const float4& sphere_0_cache, const float4& sphere_1_cache,
                       const float4& sphere_2_cache, const float& dt,
                       const bool& transform_back, float& max_dist,
                       float3& max_grad)
    {
      float3 norm_vel_vec = make_float3(sphere_2_cache.x - sphere_0_cache.x,
                                        sphere_2_cache.y - sphere_0_cache.y,
                                        sphere_2_cache.z - sphere_0_cache.z);

      norm_vel_vec = (0.5 / dt) * norm_vel_vec;
      const float sph_vel = length(norm_vel_vec);

      if (transform_back)
      {
        float3 sph_acc_vec = make_float3(
          sphere_0_cache.x + sphere_2_cache.x - 2 * sphere_1_cache.x,
          sphere_0_cache.y + sphere_2_cache.y - 2 * sphere_1_cache.y,
          sphere_0_cache.z + sphere_2_cache.z - 2 * sphere_1_cache.z);

        sph_acc_vec  = (1 / (dt * dt)) * sph_acc_vec;
        norm_vel_vec = norm_vel_vec * (1 / sph_vel);

        const float3 curvature_vec = (sph_acc_vec) / (sph_vel * sph_vel);

        // compute orthogonal projection:
        float orth_proj[9] = { 0.0 };

        // load float3 into array for easier matmul later:
        float vel_arr[3];
        vel_arr[0] = norm_vel_vec.x;
        vel_arr[1] = norm_vel_vec.y;
        vel_arr[2] = norm_vel_vec.z;

        // calculate projection ( I - (v * v^T)):
#pragma unroll 3

        for (int i = 0; i < 3; i++)
        {
#pragma unroll 3

          for (int j = 0; j < 3; j++)
          {
            orth_proj[i * 3 + j] = -1 * vel_arr[i] * vel_arr[j];
          }
        }
        orth_proj[0] += 1;
        orth_proj[4] += 1;
        orth_proj[8] += 1;

        // curvature vec:

        // multiply by orth projection:
        // two matmuls:
        float orth_pt[3];    // orth_proj(3x3) * max_grad(3x1)
        float orth_curve[3]; // max_dist(1) * orth_proj (3x3) * curvature_vec (3x1)

#pragma unroll 3

        for (int i = 0; i < 3; i++) // matrix - vector product
        {
          orth_pt[i] = orth_proj[i * 3 + 0] * max_grad.x +
                       orth_proj[i * 3 + 1] * max_grad.y +
                       orth_proj[i * 3 + 2] * max_grad.z;

          orth_curve[i] = max_dist * (orth_proj[i * 3 + 0] * curvature_vec.x +
                                      orth_proj[i * 3 + 1] * curvature_vec.y +
                                      orth_proj[i * 3 + 2] * curvature_vec.z);
        }

        // max_grad =  sph_vel * ((orth_proj * max_grad) - max_dist *  orth_proj *
        // curvature)

        max_grad.x = sph_vel * (orth_pt[0] - orth_curve[0]); // orth_proj[0];// * (orth_pt[0] -
                                                             // orth_curve[0]);
        max_grad.y = sph_vel * (orth_pt[1] - orth_curve[1]);
        max_grad.z = sph_vel * (orth_pt[2] - orth_curve[2]);
      }
      max_dist = sph_vel * max_dist;
    }

    //

    /**
     * @brief check if sphere is inside. For numerical stability, we assume that if
     * sphere is exactly at bound of cuboid, we are not in collision. Note: this is
     * not warp safe.
     *
     * @param bounds bounds of cuboid
     * @param sphere sphere as float4 (in cuboid frame of reference)
     * @return bool
     */
    __device__ __forceinline__ bool
    check_sphere_aabb(const float3 bounds, const float4 sphere)
    {
      // if((fabs(sphere.x) - bounds.x) >= sphere.w || fabs(sphere.y) - bounds.y >=
      // sphere.w || (fabs(sphere.z) - bounds.z) >= sphere.w)
      if (max(max(fabs(sphere.x) - bounds.x, fabs(sphere.y) - bounds.y),
              fabs(sphere.z) - bounds.z) >= (sphere.w))
      {
        return false;
      }
      return true;
    }

    __device__ __forceinline__ void
    compute_closest_point(const float3& bounds, const float4& sphere, float4& pt)
    {
      // assuming the cuboid is centered at origin
      // Find the closest face to the sphere position:
      // If sphere is within bounds of obstacle,
      float min_val, curr_val;
      int   min_idx;

      // We check the distance to each face and store the closest face
      // All we want is the index of the closest face
      min_val = bounds.x - fabsf(sphere.x);

      if (min_val < 0) // it's outside limits, clamp:
      {
        pt.x = copysignf(bounds.x, sphere.x);
      }

      // check if bounds.x - sphere.x > bounds.x + sphere.x
      min_val = fabsf(min_val);

      if (sphere.x > 0)
      {
        min_idx = 0;
      }
      else
      {
        min_idx = 1;
      }

      curr_val = bounds.y - fabsf(sphere.y); // check if sphere-y is outside y-lim

      if (curr_val < 0)                      // outside obb-y, we clamp point to bounds
      {
        pt.y = copysignf(bounds.y, sphere.y);
      }
      curr_val = fabsf(curr_val);

      if (curr_val < min_val)
      {
        min_val = curr_val;

        if (sphere.y > 0)
        {
          min_idx = 2;
        }
        else
        {
          min_idx = 3;
        }
      }
      curr_val = bounds.z - fabsf(sphere.z); // distance to -ve bound

      if (curr_val < 0)
      {
        pt.y = copysignf(bounds.z, sphere.z);
      }
      curr_val = fabsf(curr_val);

      if (curr_val < min_val)
      {
        min_val = curr_val;

        if (sphere.z > 0)
        {
          min_idx = 4;
        }
        else
        {
          min_idx = 5;
        }
      }

      // we move pt's value to bounds on the closest face dimension:
      switch (min_idx)
      {
      case 0:
        pt.x = bounds.x;
        break;

      case 1:
        pt.x = -1 * bounds.x;
        break;

      case 2:
        pt.y = bounds.y;
        break;

      case 3:
        pt.y = -1 * bounds.y;
        break;

      case 4:
        pt.z = bounds.z;
        break;

      case 5:
        pt.z = -1 * bounds.z;
        break;

      default:
        break;
      }
    }

    __device__ __forceinline__ float
    compute_distance(const float3& bounds_w_radius,
                     const float4& sphere) // pass in cl_pt
    {// compute closest point:
      float4 cl_pt = make_float4(sphere.x, sphere.y, sphere.z, 0);

      compute_closest_point(bounds_w_radius, sphere, cl_pt);

      // clamp:

      // compute distance:
      return sphere_length(sphere, cl_pt); // cl_pt includes radius, and sphere also has radius
    }

    __device__ __forceinline__ void scale_eta_metric(const float4& sphere, const float4& cl_pt,
                                                     const float eta,
                                                     float4& sum_pt)
    {
      // compute distance:
      float sph_dist = 0;

      sph_dist = sphere_length(sphere, cl_pt);

      if (sph_dist == 0)
      {
        sum_pt.x = 0;
        sum_pt.y = 0;
        sum_pt.z = 0;
        sum_pt.w = 0;

        return;
      }
      sum_pt.x = (sphere.x - cl_pt.x) / sph_dist;
      sum_pt.y = (sphere.y - cl_pt.y) / sph_dist;
      sum_pt.z = (sphere.z - cl_pt.z) / sph_dist;

      if (eta > 0.0)
      {
        // sph_dist = sph_dist - eta;

        if (sph_dist > eta)
        {
          sum_pt.w = sph_dist - 0.5 * eta;
        }
        else if (sph_dist <= eta)
        {
          sum_pt.w = (0.5 / eta) * (sph_dist) * (sph_dist);
          const float scale = (1 / eta) * (sph_dist);
          sum_pt.x = scale * sum_pt.x;
          sum_pt.y = scale * sum_pt.y;
          sum_pt.z = scale * sum_pt.z;
        }
      }
      else
      {
        sum_pt.w = sph_dist;
      }
    }

    /**
     * @brief Compute distance and gradient, with a smooth l2 loss when distance >0
     * and < eta
     *
     * @param bounds_w_radius
     * @param sphere
     * @param sum_pt
     * @param eta
     * @return __device__
     */
    __device__ __forceinline__ void
    compute_sphere_gradient(const float3& bounds_w_radius, const float4& sphere,
                            float4& sum_pt, const float eta = 0.0)
    {
      // compute closest point:
      float4 cl_pt = make_float4(sphere.x, sphere.y, sphere.z, 0.0);

      compute_closest_point(bounds_w_radius, sphere, cl_pt);

      scale_eta_metric(sphere, cl_pt, eta, sum_pt);
    }

    __device__ __forceinline__ void
    compute_sphere_gradient_add(const float3& bounds_w_radius, const float4& sphere,
                                float4& sum_pt, const float k0 = 1.0,
                                const float eta = 0.0)
    {
      // compute closest point:
      float4 cl_pt = make_float4(sphere.x, sphere.y, sphere.z, 0.0);

      compute_closest_point(bounds_w_radius, sphere, cl_pt);

      // We need to check if sphere is colliding or outside
      // if distance > 0.0, then it's colliding
      // We can achieve this by adding eta to bounds_w_radius and then subtracting
      // it
      float4 local_sum_pt = make_float4(0.0, 0.0, 0.0, 0.0);
      scale_eta_metric(sphere, cl_pt, eta, local_sum_pt);
      sum_pt += local_sum_pt;
    }

    __device__ __forceinline__ void check_jump_distance(
      const float4& loc_sphere_1, const float4 loc_sphere_0, const float k0,
      const float eta, const float3& loc_bounds, const float3& grad_loc_bounds,
      float4& sum_pt,
      float& curr_jump_distance) // we can pass in interpolated sphere here, also
                                 // need to pass cl_pt for use in
                                 // compute_sphere_gradient & compute_distance
    {
      const float4 interpolated_sphere =
        (k0) * loc_sphere_1 + (1 - k0) * loc_sphere_0;

      if (check_sphere_aabb(loc_bounds, interpolated_sphere))
      {
        compute_sphere_gradient_add(grad_loc_bounds, interpolated_sphere, sum_pt,
                                    k0, eta);  // true, loc_sphere_1 works better
        curr_jump_distance +=  loc_sphere_1.w; // move by diameter of sphere
      }
      else
      {
        const float dist = compute_distance(grad_loc_bounds, interpolated_sphere);
        curr_jump_distance +=  max(dist - 2 * loc_sphere_1.w, loc_sphere_1.w);
      }
    }

    ///////////////////////////////////////////////////////////////
    // We write out the distance and gradients for all the spheres.
    // So we do not need to initize the output tensor to 0.
    // Each thread computes the max distance and gradients per sphere.
    // This version should be faster if we have enough spheres/threads
    // to fill the GPU as it avoid inter-thread communication and the
    // use of shared memory.
    ///////////////////////////////////////////////////////////////

    template<typename scalar_t>
    __device__ __forceinline__ void sphere_obb_collision_fn(
      const scalar_t *sphere_position,
      const int env_idx,
      const int bn_sph_idx,
      const int sph_idx,
      scalar_t *out_distance, const scalar_t *weight,
      const scalar_t *activation_distance, const scalar_t *obb_accel,
      const scalar_t *obb_bounds, const scalar_t *obb_mat,
      const uint8_t *obb_enable, const int max_nobs, const int nboxes)
    {
      float max_dist            = 0;
      const int   start_box_idx = max_nobs * env_idx;
      const float eta           = activation_distance[0];

      // Load sphere_position input
      float4 sphere_cache = *(float4 *)&sphere_position[bn_sph_idx * 4];

      if (sphere_cache.w <= 0.0)
      {
        // write zeros for cost:
        out_distance[bn_sph_idx] = 0;

        return;
      }
      sphere_cache.w += eta;

      float4 loc_sphere = make_float4(0.0);
      float4 obb_quat   = make_float4(0.0);
      float3 obb_pos    = make_float3(0.0);
      float3 loc_bounds = make_float3(0.0);

      for (int box_idx = 0; box_idx < nboxes; box_idx++)
      {
        if (obb_enable[start_box_idx + box_idx] == 0) // disabled obstacle
        {
          continue;
        }
        load_obb_pose(&obb_mat[(start_box_idx + box_idx) * 8], obb_pos,
                      obb_quat);
        load_obb_bounds(&obb_bounds[(start_box_idx + box_idx) * 4], loc_bounds);

        transform_sphere_quat(obb_pos, obb_quat, sphere_cache, loc_sphere);


        // first check if point is inside or outside box:

        if (check_sphere_aabb(loc_bounds, loc_sphere))
        {
          max_dist = 1;
          break; // we exit without checking other cuboids if we found a collision.
        }
      }
      out_distance[bn_sph_idx] = weight[0] * max_dist;
    }

    template<typename scalar_t>
    __device__ __forceinline__ void sphere_obb_distance_fn(
      const scalar_t *sphere_position,
      const int32_t env_idx,
      const int bn_sph_idx,
      const int sph_idx,
      scalar_t *out_distance, scalar_t *closest_pt, uint8_t *sparsity_idx,
      const scalar_t *weight, const scalar_t *activation_distance,
      const scalar_t *obb_accel, const scalar_t *obb_bounds,
      const scalar_t *obb_mat, const uint8_t *obb_enable, const int max_nobs,
      const int nboxes, const bool transform_back)
    {
      float max_dist  = 0.0;
      const float eta = activation_distance[0];
      float3 max_grad = make_float3(0.0, 0.0, 0.0);

      // Load sphere_position input
      float4 sphere_cache = *(float4 *)&sphere_position[bn_sph_idx * 4];

      if (sphere_cache.w <= 0.0)
      {
        // write zeros for cost:
        out_distance[bn_sph_idx] = 0;

        // write zeros for gradient if not zero:
        if (sparsity_idx[bn_sph_idx] != 0)
        {
          sparsity_idx[bn_sph_idx]               = 0;
          *(float4 *)&closest_pt[bn_sph_idx * 4] = make_float4(0.0);
        }
        return;
      }
      sphere_cache.w += eta;
      const int start_box_idx = max_nobs * env_idx;

      float4 loc_sphere = make_float4(0.0);
      float4 obb_quat   = make_float4(0.0);
      float3 obb_pos    = make_float3(0.0);
      float3 loc_bounds = make_float3(0.0);

      for (int box_idx = 0; box_idx < nboxes; box_idx++)
      {
        if (obb_enable[start_box_idx + box_idx] == 0) // disabled obstacle
        {
          continue;
        }

        load_obb_pose(&obb_mat[(start_box_idx + box_idx) * 8], obb_pos,
                      obb_quat);
        load_obb_bounds(&obb_bounds[(start_box_idx + box_idx) * 4], loc_bounds);

        transform_sphere_quat(obb_pos, obb_quat, sphere_cache, loc_sphere);

        // first check if point is inside or outside box:
        if (check_sphere_aabb(loc_bounds, loc_sphere))
        {
          // compute closest point:
          loc_bounds = loc_bounds + loc_sphere.w;

          // using same primitive functions:
          float4 cl;
          compute_sphere_gradient(loc_bounds, loc_sphere, cl, eta);

          max_dist += cl.w;

          if (transform_back)
          {
            inv_transform_vec_quat(obb_pos, obb_quat, cl, max_grad);

            // inv_transform_vec_quat_add(&obb_mat[(start_box_idx + box_idx) * 7], cl,
            //                           max_grad);
          }
        }
      }

      // sparsity opt:
      if (max_dist == 0)
      {
        if (sparsity_idx[bn_sph_idx] == 0)
        {
          return;
        }
        sparsity_idx[bn_sph_idx] = 0;

        if (transform_back)
        {
          *(float3 *)&closest_pt[bn_sph_idx * 4] = max_grad; // max_grad is all zeros
        }
        out_distance[bn_sph_idx] = 0.0;
        return;
      }

      // else max_dist != 0
      max_dist = weight[0] * max_dist;

      if (transform_back)
      {
        *(float3 *)&closest_pt[bn_sph_idx * 4] = weight[0] * max_grad;
      }
      out_distance[bn_sph_idx] = max_dist;
      sparsity_idx[bn_sph_idx] = 1;
    }

    template<typename scalar_t, bool enable_speed_metric>
    __device__ __forceinline__ void swept_sphere_obb_distance_fn(
      const scalar_t *sphere_position,
      const int env_idx, const int b_idx,
      const int h_idx, const int sph_idx,
      scalar_t *out_distance,
      scalar_t *closest_pt,
      uint8_t *sparsity_idx,
      const scalar_t *weight,
      const scalar_t *activation_distance, const scalar_t *speed_dt,
      const scalar_t *obb_accel, const scalar_t *obb_bounds,
      const scalar_t *obb_mat,
      const uint8_t *obb_enable,
      const int max_nobs,
      const int nboxes, const int batch_size, const int horizon,
      const int nspheres, const int sweep_steps,
      const bool transform_back)
    {
      // create shared memory to do warp wide reductions:
      // warp wide reductions should only happen across nspheres in same batch and horizon
      //
      // extern __shared__ float psum[];

      const int   sw_steps      = sweep_steps;
      const float fl_sw_steps   = sw_steps;
      const int   start_box_idx = max_nobs * env_idx;
      const int   b_addrs       =
        b_idx * horizon * nspheres; // + h_idx * n_spheres + sph_idx;

      // We read the same obstacles across

      // Load sphere_position input
      // if h_idx == horizon -1, we just read the same index
      const int bhs_idx = b_addrs + h_idx * nspheres + sph_idx;

      float4 sphere_1_cache = *(float4 *)&sphere_position[bhs_idx * 4];


      if (sphere_1_cache.w <= 0.0)
      {
        // write zeros for cost:
        out_distance[bhs_idx] = 0;

        // write zeros for gradient if not zero:
        if (sparsity_idx[bhs_idx] != 0)
        {
          sparsity_idx[b_addrs + h_idx * nspheres + sph_idx] = 0;
          *(float4 *)&closest_pt[bhs_idx * 4]                = make_float4(0.0);
        }

        return;
      }
      bool  sweep_back = false;
      bool  sweep_fwd = false;
      float sphere_0_distance, sphere_2_distance, sphere_0_len, sphere_2_len;

      float  max_dist = 0.0;
      float3 max_grad = make_float3(0.0, 0.0, 0.0);

      const float dt  = speed_dt[0];
      const float eta = activation_distance[0];
      sphere_1_cache.w += eta;
      float4 sphere_0_cache, sphere_2_cache;

      if (h_idx > 0)
      {
        sphere_0_cache =
          *(float4 *)&sphere_position[b_addrs * 4 + (h_idx - 1) * nspheres * 4 + sph_idx * 4];
        sphere_0_cache.w  = sphere_1_cache.w;
        sphere_0_distance = sphere_distance(sphere_0_cache, sphere_1_cache);
        sphere_0_len      = sphere_0_distance + sphere_0_cache.w * 2;

        if (sphere_0_distance > 0.0)
        {
          sweep_back = true;
        }
      }

      if (h_idx < horizon - 1)
      {
        sphere_2_cache =
          *(float4 *)&sphere_position[b_addrs * 4 + (h_idx + 1) * nspheres * 4 + sph_idx * 4];
        sphere_2_cache.w  = sphere_1_cache.w;
        sphere_2_distance = sphere_distance(sphere_2_cache, sphere_1_cache);
        sphere_2_len      = sphere_2_distance + sphere_2_cache.w * 2;

        if (sphere_2_distance > 0.0)
        {
          sweep_fwd = true;
        }
      }
      float4 loc_sphere_0, loc_sphere_1, loc_sphere_2;
      float  k0 = 0.0;


      // float4 loc_sphere = make_float4(0.0);
      float4 obb_quat   = make_float4(0.0);
      float3 obb_pos    = make_float3(0.0);
      float3 loc_bounds = make_float3(0.0);

      for (int box_idx = 0; box_idx < nboxes; box_idx++)
      {
        if (obb_enable[start_box_idx + box_idx] == 0) // disabled obstacle
        {
          continue;
        }

        // read position and quaternion:
        load_obb_pose(&obb_mat[(start_box_idx + box_idx) * 8], obb_pos,
                      obb_quat);
        load_obb_bounds(&obb_bounds[(start_box_idx + box_idx) * 4], loc_bounds);
        float curr_jump_distance = 0.0;

        const float3 grad_loc_bounds = loc_bounds + sphere_1_cache.w; // assuming sphere radius
                                                                      // doesn't change

        transform_sphere_quat(obb_pos, obb_quat, sphere_1_cache, loc_sphere_1);


        // assuming sphere position is in box frame:
        // read data:
        float4 sum_pt = make_float4(0.0, 0.0, 0.0, 0.0);

        // check at exact timestep:
        if (check_sphere_aabb(loc_bounds, loc_sphere_1))
        {
          compute_sphere_gradient(grad_loc_bounds, loc_sphere_1, sum_pt, eta);
          curr_jump_distance = loc_sphere_1.w; // TODO: check if this is better
        }
        else if (sweep_back || sweep_fwd)
        {
          // there is no collision, compute the distance to obstacle:
          curr_jump_distance = compute_distance(grad_loc_bounds, loc_sphere_1) -
                               loc_sphere_1.w; // - eta;
        }
        const float jump_mid_distance = curr_jump_distance;

        // compute distance between sweep spheres:
        if (sweep_back && (jump_mid_distance < sphere_0_distance / 2))
        {
          transform_sphere_quat(obb_pos, obb_quat, sphere_0_cache, loc_sphere_0);

          // get unit vector:
          // loc_sphere_0 = (loc_sphere_0 - loc_sphere_1)/(sphere_0_len);

          // loop over sweep steps and accumulate distance:
          for (int j = 0; j < sw_steps; j++)
          {
            // jump by current jump distance:

            // when sweep_steps == 0, then we only check at loc_sphere_1.
            // do interpolation from t=1 to t=0 (sweep backward)

            if (curr_jump_distance >= (sphere_0_len / 2))
            {
              break;
            }
            k0 = 1 - (curr_jump_distance / sphere_0_len);
            check_jump_distance(loc_sphere_1, loc_sphere_0, k0, eta, loc_bounds,
                                grad_loc_bounds, sum_pt, curr_jump_distance);
          }
        }

        if (sweep_fwd && (jump_mid_distance < (sphere_2_len / 2)))
        {
          curr_jump_distance = jump_mid_distance;
          transform_sphere_quat(obb_pos, obb_quat, sphere_2_cache, loc_sphere_2);


          for (int j = 0; j < sw_steps; j++)
          {
            if (curr_jump_distance >= (sphere_2_len / 2))
            {
              break;
            }
            k0 = 1 - curr_jump_distance / sphere_2_len;
            check_jump_distance(loc_sphere_1, loc_sphere_2, k0, eta, loc_bounds,
                                grad_loc_bounds, sum_pt, curr_jump_distance);
          }
        }

        if (sum_pt.w > 0)
        {
          max_dist += sum_pt.w;

          // transform point back if required:
          if (transform_back)
          {
            inv_transform_vec_quat(obb_pos, obb_quat, sum_pt, max_grad);
          }

          // break;// break after first obstacle collision
        }
      }


      // sparsity opt:
      if (max_dist == 0)
      {
        if (sparsity_idx[bhs_idx] == 0)
        {
          return;
        }
        sparsity_idx[bhs_idx] = 0;

        if (transform_back)
        {
          *(float3 *)&closest_pt[bhs_idx * 4] = max_grad; // max_grad is all zeros
        }
        out_distance[bhs_idx] = 0.0;
        return;
      }

      // computer speed metric here:
      if (enable_speed_metric)
      {
        if (sweep_back && sweep_fwd)
        {
          scale_speed_metric(sphere_0_cache, sphere_1_cache, sphere_2_cache, dt,
                             transform_back, max_dist, max_grad);
        }
      }
      max_dist = weight[0] * max_dist;

      if (transform_back)
      {
        *(float3 *)&closest_pt[bhs_idx * 4] = weight[0] * max_grad;
      }
      sparsity_idx[bhs_idx] = 1;

      out_distance[bhs_idx] = max_dist;
    }

    /**
     * @brief Swept Collision checking. Note: This function currently does not
     * implement skipping computation based on distance (which is done in
     * swept_sphere_obb_distance_fn).
     *
     * @tparam scalar_t
     * @param sphere_position
     * @param env_idx
     * @param b_idx
     * @param h_idx
     * @param sph_idx
     * @param out_distance
     * @param weight
     * @param activation_distance
     * @param obb_accel
     * @param obb_bounds
     * @param obb_mat
     * @param obb_enable
     * @param max_nobs
     * @param nboxes
     * @param batch_size
     * @param horizon
     * @param nspheres
     * @param sweep_steps
     * @return __device__
     */
    template<typename scalar_t>
    __device__ __forceinline__ void swept_sphere_obb_collision_fn(
      const scalar_t *sphere_position,
      const int env_idx, const int b_idx,
      const int h_idx, const int sph_idx, scalar_t *out_distance,
      const scalar_t *weight, const scalar_t *activation_distance,
      const scalar_t *obb_accel, const scalar_t *obb_bounds,
      const scalar_t *obb_mat, const uint8_t *obb_enable, const int max_nobs,
      const int nboxes, const int batch_size, const int horizon,
      const int nspheres, const int sweep_steps)
    {
      const int   sw_steps    = sweep_steps;
      const float fl_sw_steps = 2 * sw_steps + 1;
      float max_dist          = 0.0;
      const float eta         = activation_distance[0];
      const int   b_addrs     =
        b_idx * horizon * nspheres; // + h_idx * n_spheres + sph_idx;
      const int start_box_idx = max_nobs * env_idx;
      const int bhs_idx       = b_addrs + h_idx * nspheres + sph_idx;

      // We read the same obstacles across

      // Load sphere_position input
      // if h_idx == horizon -1, we just read the same index
      float4 sphere_1_cache = *(float4 *)&sphere_position[bhs_idx * 4];

      if (sphere_1_cache.w <= 0.0)
      {
        out_distance[b_addrs + h_idx * nspheres + sph_idx] = 0.0;
        return;
      }
      sphere_1_cache.w += eta;

      float4 sphere_0_cache, sphere_2_cache;

      if (h_idx > 0)
      {
        sphere_0_cache = *(float4 *)&sphere_position[b_addrs * 4 + (h_idx - 1) * nspheres * 4 +
                                                     sph_idx * 4];
        sphere_0_cache.w += eta;
      }

      if (h_idx < horizon - 1)
      {
        sphere_2_cache = *(float4 *)&sphere_position[b_addrs * 4 + (h_idx + 1) * nspheres * 4 +
                                                     sph_idx * 4];
        sphere_2_cache.w += eta;
      }
      float4 loc_sphere_0, loc_sphere_1, loc_sphere_2;
      float4 interpolated_sphere;
      float  k0, k1;
      float  in_obb_mat[7];

      for (int box_idx = 0; box_idx < nboxes; box_idx++)
      {
        // read position and quaternion:
        if (obb_enable[start_box_idx + box_idx] == 0) // disabled obstacle
        {
          continue;
        }

#pragma unroll

        for (int i = 0; i < 7; i++)
        {
          in_obb_mat[i] = obb_mat[(start_box_idx + box_idx) * 7 + i];
        }

        float3 loc_bounds =
          *(float3 *)&obb_bounds[(start_box_idx + box_idx) * 3]; // /2
        loc_bounds = loc_bounds / 2;

        transform_sphere_quat(&in_obb_mat[0], sphere_1_cache, loc_sphere_1);

        max_dist += box_idx;

        if (check_sphere_aabb(loc_bounds, loc_sphere_1))
        {
          max_dist = 1;
        }

        if (h_idx > 0)
        {
          transform_sphere_quat(&in_obb_mat[0], sphere_0_cache, loc_sphere_0);

          // loop over sweep steps and accumulate distance:
          for (int j = 0; j < sw_steps; j++)
          {
            // when sweep_steps == 0, then we only check at loc_sphere_1.
            // do interpolation from t=1 to t=0 (sweep backward)
            k0                  = (j + 1) / (fl_sw_steps);
            k1                  = 1 - k0;
            interpolated_sphere = k0 * loc_sphere_1 + (k1) * loc_sphere_0;

            if (check_sphere_aabb(loc_bounds, interpolated_sphere))
            {
              max_dist = 1;
              break;
            }
          }
        }

        if (h_idx < horizon - 1)
        {
          transform_sphere_quat(&in_obb_mat[0], sphere_2_cache, loc_sphere_2);

          for (int j = 0; j < sw_steps; j++)
          {
            // do interpolation from t=1 to t=2 (sweep forward):

            k0                  = (j + 1) / (fl_sw_steps);
            k1                  = 1 - k0;
            interpolated_sphere = k0 * loc_sphere_1 + (k1) * loc_sphere_2;

            if (check_sphere_aabb(loc_bounds, interpolated_sphere))
            {
              max_dist = 1;
              break;
            }
          }
        }

        if (max_dist > 0)
        {
          break;
        }
      }
      out_distance[b_addrs + h_idx * nspheres + sph_idx] = weight[0] * max_dist;
    }

    template<typename scalar_t, bool batch_env_t>
    __global__ void sphere_obb_distance_kernel(
      const scalar_t *sphere_position,
      scalar_t *out_distance,
      scalar_t *closest_pt, uint8_t *sparsity_idx, const scalar_t *weight,
      const scalar_t *activation_distance, const scalar_t *obb_accel,
      const scalar_t *obb_bounds, const scalar_t *obb_mat,
      const uint8_t *obb_enable, const int32_t *n_env_obb,
      const int32_t *env_query_idx,
      const int max_nobs,
      const int batch_size, const int horizon, const int nspheres,
      const bool transform_back)
    {
      // spheres_per_block is number of spheres in a thread
      // compute local sphere batch by first dividing threadidx/nboxes
      // const int sph_idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int t_idx   = blockIdx.x * blockDim.x + threadIdx.x;
      const int b_idx   = t_idx / (horizon * nspheres);
      const int h_idx   = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
      const int sph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);

      if ((sph_idx >= nspheres) || (b_idx >= batch_size) || (h_idx >= horizon))
      {
        return;
      }
      const int bn_sph_idx =
        b_idx * horizon * nspheres + h_idx * nspheres + sph_idx;

      int env_idx    = 0;
      int env_nboxes = n_env_obb[0];

      if (batch_env_t)
      {
        env_idx =
          env_query_idx[b_idx]; // read env idx from current batch idx
        env_nboxes =
          n_env_obb[env_idx];   // read nboxes in current environment
      }

      sphere_obb_distance_fn(sphere_position, env_idx, bn_sph_idx, sph_idx, out_distance,
                             closest_pt, sparsity_idx, weight, activation_distance,
                             obb_accel, obb_bounds, obb_mat, obb_enable, max_nobs,
                             env_nboxes, transform_back);

      // return the sphere distance here:
      // sync threads and do block level reduction:
    }

    template<typename scalar_t, bool enable_speed_metric>
    __global__ void swept_sphere_obb_distance_jump_kernel(
      const scalar_t *sphere_position,
      scalar_t *out_distance,
      scalar_t *closest_pt, uint8_t *sparsity_idx, const scalar_t *weight,
      const scalar_t *activation_distance, const scalar_t *speed_dt,
      const scalar_t *obb_accel, const scalar_t *obb_bounds,
      const scalar_t *obb_pose, const uint8_t *obb_enable,
      const int32_t *n_env_obb, const int max_nobs, const int batch_size,
      const int horizon, const int nspheres, const int sweep_steps, const bool transform_back)
    {
      // This kernel jumps by sdf to only get gradients at collision points.

      // spheres_per_block is number of spheres in a thread
      // compute local sphere batch by first dividing threadidx/nboxes
      const int t_idx   = blockIdx.x * blockDim.x + threadIdx.x;
      const int b_idx   = t_idx / (horizon * nspheres);
      const int h_idx   = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
      const int sph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);

      if ((sph_idx >= nspheres) || (b_idx >= batch_size) || (h_idx >= horizon))
      {
        return;
      }

      const int env_idx = 0;


      const int env_nboxes = n_env_obb[env_idx];
      swept_sphere_obb_distance_fn<scalar_t, enable_speed_metric>(
        sphere_position, env_idx, b_idx, h_idx, sph_idx, out_distance, closest_pt,
        sparsity_idx, weight, activation_distance, speed_dt, obb_accel,
        obb_bounds, obb_pose, obb_enable, max_nobs, env_nboxes, batch_size,
        horizon, nspheres, sweep_steps, transform_back);
    }

    template<typename scalar_t, bool enable_speed_metric>
    __global__ void swept_sphere_obb_distance_jump_batch_env_kernel(
      const scalar_t *sphere_position,
      scalar_t *out_distance,
      scalar_t *closest_pt, uint8_t *sparsity_idx, const scalar_t *weight,
      const scalar_t *activation_distance, const scalar_t *speed_dt,
      const scalar_t *obb_accel, const scalar_t *obb_bounds,
      const scalar_t *obb_pose, const uint8_t *obb_enable,
      const int32_t *n_env_obb, const int32_t *env_query_idx, const int max_nobs,
      const int batch_size, const int horizon, const int nspheres,
      const int sweep_steps,
      const bool transform_back)
    {
      // This kernel jumps by sdf to only get gradients at collision points.

      // spheres_per_block is number of spheres in a thread
      // compute local sphere batch by first dividing threadidx/nboxes
      const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int b_idx = t_idx / (horizon * nspheres);

      // const int sph_idx = (t_idx - b_idx * (horizon * nspheres)) / horizon;
      // const int h_idx = (t_idx - b_idx * horizon * nspheres - sph_idx * horizon);
      const int h_idx   = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
      const int sph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);

      if ((sph_idx >= nspheres) || (b_idx >= batch_size) || (h_idx >= horizon))
      {
        return;
      }

      const int env_idx    = env_query_idx[b_idx];
      const int env_nboxes = n_env_obb[env_idx];

      swept_sphere_obb_distance_fn<scalar_t, enable_speed_metric>(
        sphere_position,  env_idx, b_idx, h_idx, sph_idx, out_distance, closest_pt,
        sparsity_idx, weight, activation_distance, speed_dt, obb_accel,
        obb_bounds, obb_pose, obb_enable, max_nobs, env_nboxes, batch_size,
        horizon, nspheres, sweep_steps, transform_back);
    }

    template<typename scalar_t>
    __global__ void swept_sphere_obb_collision_kernel(
      const scalar_t *sphere_position,
      scalar_t *out_distance,
      const scalar_t *weight, const scalar_t *activation_distance,
      const scalar_t *obb_accel, const scalar_t *obb_bounds,
      const scalar_t *obb_pose, const uint8_t *obb_enable,
      const int32_t *n_env_obb, const int max_nobs, const int batch_size,
      const int horizon, const int nspheres, const int sweep_steps)
    {
      // spheres_per_block is number of spheres in a thread
      // compute local sphere batch by first dividing threadidx/nboxes
      const int t_idx   = blockIdx.x * blockDim.x + threadIdx.x;
      const int b_idx   = t_idx / (horizon * nspheres);
      const int h_idx   = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
      const int sph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);

      if ((sph_idx >= nspheres) || (b_idx >= batch_size) || (h_idx >= horizon))
      {
        return;
      }

      const int env_idx    = 0;
      const int env_nboxes = n_env_obb[env_idx];

      swept_sphere_obb_collision_fn(
        sphere_position, env_idx, b_idx, h_idx, sph_idx, out_distance, weight,
        activation_distance, obb_accel, obb_bounds, obb_pose, obb_enable,
        max_nobs, env_nboxes, batch_size, horizon, nspheres, sweep_steps);
    }

    template<typename scalar_t>
    __global__ void swept_sphere_obb_collision_batch_env_kernel(
      const scalar_t *sphere_position,
      scalar_t *out_distance,
      const scalar_t *weight, const scalar_t *activation_distance,
      const scalar_t *obb_accel, const scalar_t *obb_bounds,
      const scalar_t *obb_pose, const uint8_t *obb_enable,
      const int32_t *n_env_obb, const int32_t *env_query_idx, const int max_nobs,
      const int batch_size, const int horizon, const int nspheres,
      const int sweep_steps)
    {
      // spheres_per_block is number of spheres in a thread
      // compute local sphere batch by first dividing threadidx/nboxes
      const int t_idx   = blockIdx.x * blockDim.x + threadIdx.x;
      const int b_idx   = t_idx / (horizon * nspheres);
      const int h_idx   = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
      const int sph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);

      if ((sph_idx >= nspheres) || (b_idx >= batch_size) || (h_idx >= horizon))
      {
        return;
      }

      const int env_idx    = env_query_idx[b_idx];
      const int env_nboxes = n_env_obb[env_idx];

      swept_sphere_obb_collision_fn(
        sphere_position, env_idx, b_idx, h_idx, sph_idx, out_distance, weight,
        activation_distance, obb_accel, obb_bounds, obb_pose, obb_enable,
        max_nobs, env_nboxes, batch_size, horizon, nspheres, sweep_steps);
    }

    template<typename scalar_t>
    __global__ void sphere_obb_collision_kernel(
      const scalar_t *sphere_position,
      scalar_t *out_distance,
      const scalar_t *weight, const scalar_t *activation_distance,
      const scalar_t *obb_accel, const scalar_t *obb_bounds,
      const scalar_t *obb_mat, const uint8_t *obb_enable,
      const int32_t *n_env_obb, const int max_nobs, const int batch_size,
      const int horizon, const int nspheres)
    {
      // spheres_per_block is number of spheres in a thread
      // compute local sphere batch by first dividing threadidx/nboxes
      const int t_idx   = blockIdx.x * blockDim.x + threadIdx.x;
      const int b_idx   = t_idx / (horizon * nspheres);
      const int h_idx   = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
      const int sph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);

      if ((sph_idx >= nspheres) || (b_idx >= batch_size) || (h_idx >= horizon))
      {
        return;
      }
      const int env_idx    = 0;
      const int env_nboxes = n_env_obb[env_idx];
      const int bn_sph_idx =
        b_idx * horizon * nspheres + h_idx * nspheres + sph_idx;
      sphere_obb_collision_fn(sphere_position,
                              env_idx, bn_sph_idx, sph_idx,  out_distance,
                              weight, activation_distance, obb_accel, obb_bounds,
                              obb_mat, obb_enable, max_nobs, env_nboxes);
    }

    template<typename scalar_t>
    __global__ void sphere_obb_collision_batch_env_kernel(
      const scalar_t *sphere_position,
      scalar_t *out_distance,
      const scalar_t *weight, const scalar_t *activation_distance,
      const scalar_t *obb_accel, const scalar_t *obb_bounds,
      const scalar_t *obb_mat, const uint8_t *obb_enable,
      const int32_t *n_env_obb, const int32_t *env_query_idx, const int max_nobs,
      const int batch_size, const int horizon, const int nspheres)
    {
      // spheres_per_block is number of spheres in a thread
      // compute local sphere batch by first dividing threadidx/nboxes
      const int t_idx   = blockIdx.x * blockDim.x + threadIdx.x;
      const int b_idx   = t_idx / (horizon * nspheres);
      const int h_idx   = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
      const int sph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);

      if ((sph_idx >= nspheres) || (b_idx >= batch_size) || (h_idx >= horizon))
      {
        return;
      }
      const int env_idx    = env_query_idx[b_idx];
      const int env_nboxes = n_env_obb[env_idx];
      const int bn_sph_idx =
        b_idx * horizon * nspheres + h_idx * nspheres + sph_idx;
      sphere_obb_collision_fn(sphere_position, env_idx, bn_sph_idx, sph_idx, out_distance,
                              weight, activation_distance, obb_accel, obb_bounds,
                              obb_mat, obb_enable, max_nobs, env_nboxes);
    }
  } // namespace Geometry
}   // namespace Curobo


std::vector<torch::Tensor>
sphere_obb_clpt(const torch::Tensor sphere_position, // batch_size, 3
                torch::Tensor distance,
                torch::Tensor closest_point,         // batch size, 3
                torch::Tensor sparsity_idx, const torch::Tensor weight,
                const torch::Tensor activation_distance,
                const torch::Tensor obb_accel,       // n_boxes, 4, 4
                const torch::Tensor obb_bounds,      // n_boxes, 3
                const torch::Tensor obb_pose,        // n_boxes, 4, 4
                const torch::Tensor obb_enable,      // n_boxes, 4, 4
                const torch::Tensor n_env_obb,       // n_boxes, 4, 4
                const torch::Tensor env_query_idx,   // n_boxes, 4, 4
                const int max_nobs, const int batch_size, const int horizon,
                const int n_spheres, const bool transform_back,
                const bool compute_distance, const bool use_batch_env)
{
  using namespace Curobo::Geometry;
  cudaStream_t stream      = at::cuda::getCurrentCUDAStream();
  const int    bnh_spheres = n_spheres * batch_size * horizon; //

  int threadsPerBlock = bnh_spheres;

  if (threadsPerBlock > 128)
  {
    threadsPerBlock = 128;
  }
  int blocksPerGrid = (bnh_spheres + threadsPerBlock - 1) / threadsPerBlock;

  if (use_batch_env)
  {
    if (compute_distance)
    {
      // TODO: call kernel based on flag:
      AT_DISPATCH_FLOATING_TYPES(
        distance.scalar_type(), "SphereObb_clpt", ([&] {
        sphere_obb_distance_kernel<scalar_t, true>
          << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<scalar_t>(),
          closest_point.data_ptr<scalar_t>(),
          sparsity_idx.data_ptr<uint8_t>(),
          weight.data_ptr<scalar_t>(),
          activation_distance.data_ptr<scalar_t>(),
          obb_accel.data_ptr<scalar_t>(),
          obb_bounds.data_ptr<scalar_t>(),
          obb_pose.data_ptr<scalar_t>(),
          obb_enable.data_ptr<uint8_t>(),
          n_env_obb.data_ptr<int32_t>(),
          env_query_idx.data_ptr<int32_t>(), max_nobs, batch_size,
          horizon, n_spheres, transform_back);
      }));
    }
    else
    {
      // TODO: call kernel based on flag:
      AT_DISPATCH_FLOATING_TYPES(
        distance.scalar_type(), "SphereObb_collision", ([&] {
        sphere_obb_collision_batch_env_kernel<scalar_t>
          << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
          activation_distance.data_ptr<scalar_t>(),
          obb_accel.data_ptr<scalar_t>(),
          obb_bounds.data_ptr<scalar_t>(),
          obb_pose.data_ptr<scalar_t>(),
          obb_enable.data_ptr<uint8_t>(),
          n_env_obb.data_ptr<int32_t>(),
          env_query_idx.data_ptr<int32_t>(), max_nobs, batch_size,
          horizon, n_spheres);
      }));
    }
  }
  else
  {
    if (compute_distance)
    {
      // TODO: call kernel based on flag:
      AT_DISPATCH_FLOATING_TYPES(
        distance.scalar_type(), "SphereObb_clpt", ([&] {
        sphere_obb_distance_kernel<scalar_t, false>
          << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<scalar_t>(),
          closest_point.data_ptr<scalar_t>(),
          sparsity_idx.data_ptr<uint8_t>(),
          weight.data_ptr<scalar_t>(),
          activation_distance.data_ptr<scalar_t>(),
          obb_accel.data_ptr<scalar_t>(),
          obb_bounds.data_ptr<scalar_t>(),
          obb_pose.data_ptr<scalar_t>(),
          obb_enable.data_ptr<uint8_t>(),
          n_env_obb.data_ptr<int32_t>(),
          env_query_idx.data_ptr<int32_t>(), max_nobs, batch_size,
          horizon, n_spheres, transform_back);
      }));
    }
    else
    {
      // TODO: call kernel based on flag:
      AT_DISPATCH_FLOATING_TYPES(
        distance.scalar_type(), "SphereObb_collision", ([&] {
        sphere_obb_collision_kernel<scalar_t>
          << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
          activation_distance.data_ptr<scalar_t>(),
          obb_accel.data_ptr<scalar_t>(),
          obb_bounds.data_ptr<scalar_t>(),
          obb_pose.data_ptr<scalar_t>(),
          obb_enable.data_ptr<uint8_t>(),
          n_env_obb.data_ptr<int32_t>(), max_nobs, batch_size,
          horizon, n_spheres);
      }));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { distance, closest_point, sparsity_idx };
}

std::vector<torch::Tensor>swept_sphere_obb_clpt(
  const torch::Tensor sphere_position, // batch_size, 3

  torch::Tensor       distance,        // batch_size, 1
  torch::Tensor
  closest_point,                       // batch size, 4 -> written out as x,y,z,0 for gradient
  torch::Tensor sparsity_idx, const torch::Tensor weight,
  const torch::Tensor activation_distance, const torch::Tensor speed_dt,
  const torch::Tensor obb_accel,       // n_boxes, 4, 4
  const torch::Tensor obb_bounds,      // n_boxes, 3
  const torch::Tensor obb_pose,        // n_boxes, 4, 4
  const torch::Tensor obb_enable,      // n_boxes, 4,
  const torch::Tensor n_env_obb,       // n_boxes, 4, 4
  const torch::Tensor env_query_idx,   // n_boxes, 4, 4
  const int max_nobs, const int batch_size, const int horizon,
  const int n_spheres, const int sweep_steps, const bool enable_speed_metric,
  const bool transform_back, const bool compute_distance,
  const bool use_batch_env)
{
  using namespace Curobo::Geometry;


  // const int max_batches_per_block = 128;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // const int bh = batch_size * horizon;

  // const int warp_n_spheres = n_spheres + (n_spheres % 32);// make n_spheres a multiple of 32
  // int batches_per_block = (bh * warp_n_spheres) / max_batches_per_block;
  const int bnh_spheres = n_spheres * batch_size * horizon; //
  int threadsPerBlock   = bnh_spheres;

  // This block is for old kernels?
  if (threadsPerBlock > 128)
  {
    threadsPerBlock = 128;
  }
  int blocksPerGrid = (bnh_spheres + threadsPerBlock - 1) / threadsPerBlock;


  if (use_batch_env)
  {
    if (compute_distance)
    {
      if (enable_speed_metric)
      {
        // This is the best kernel for now
        AT_DISPATCH_FLOATING_TYPES(
          distance.scalar_type(), "Swept_SphereObb_clpt", ([&] {
          swept_sphere_obb_distance_jump_batch_env_kernel<scalar_t, true>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<scalar_t>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<scalar_t>(),
            activation_distance.data_ptr<scalar_t>(),
            speed_dt.data_ptr<scalar_t>(),
            obb_accel.data_ptr<scalar_t>(),
            obb_bounds.data_ptr<scalar_t>(),
            obb_pose.data_ptr<scalar_t>(),
            obb_enable.data_ptr<uint8_t>(),
            n_env_obb.data_ptr<int32_t>(),
            env_query_idx.data_ptr<int32_t>(), max_nobs, batch_size,
            horizon, n_spheres, sweep_steps,
            transform_back);
        }));
      }
      else
      {
        // This is the best kernel for now
        AT_DISPATCH_FLOATING_TYPES(
          distance.scalar_type(), "Swept_SphereObb_clpt", ([&] {
          swept_sphere_obb_distance_jump_batch_env_kernel<scalar_t, false>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<scalar_t>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<scalar_t>(),
            activation_distance.data_ptr<scalar_t>(),
            speed_dt.data_ptr<scalar_t>(),
            obb_accel.data_ptr<scalar_t>(),
            obb_bounds.data_ptr<scalar_t>(),
            obb_pose.data_ptr<scalar_t>(),
            obb_enable.data_ptr<uint8_t>(),
            n_env_obb.data_ptr<int32_t>(),
            env_query_idx.data_ptr<int32_t>(), max_nobs, batch_size,
            horizon, n_spheres, sweep_steps,
            transform_back);
        }));
      }
    }
    else
    {
      // TODO: implement this later

      // TODO: call kernel based on flag:
      AT_DISPATCH_FLOATING_TYPES(
        distance.scalar_type(), "SphereObb_collision", ([&] {
        swept_sphere_obb_collision_batch_env_kernel<scalar_t>
          << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
          activation_distance.data_ptr<scalar_t>(),
          obb_accel.data_ptr<scalar_t>(),
          obb_bounds.data_ptr<scalar_t>(),
          obb_pose.data_ptr<scalar_t>(),
          obb_enable.data_ptr<uint8_t>(),
          n_env_obb.data_ptr<int32_t>(),
          env_query_idx.data_ptr<int32_t>(), max_nobs, batch_size,
          horizon, n_spheres, sweep_steps);
      }));
    }
  }
  else
  {
    if (compute_distance)
    {
      if (enable_speed_metric)
      {
        // This is the best kernel for now
        AT_DISPATCH_FLOATING_TYPES(
          distance.scalar_type(), "Swept_SphereObb_clpt", ([&] {
          swept_sphere_obb_distance_jump_kernel<scalar_t, true>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<scalar_t>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<scalar_t>(),
            activation_distance.data_ptr<scalar_t>(),
            speed_dt.data_ptr<scalar_t>(),
            obb_accel.data_ptr<scalar_t>(),
            obb_bounds.data_ptr<scalar_t>(),
            obb_pose.data_ptr<scalar_t>(),
            obb_enable.data_ptr<uint8_t>(),
            n_env_obb.data_ptr<int32_t>(), max_nobs, batch_size,
            horizon, n_spheres, sweep_steps,
            transform_back);
        }));
      }
      else
      {
        // This is the best kernel for now
        AT_DISPATCH_FLOATING_TYPES(
          distance.scalar_type(), "Swept_SphereObb_clpt", ([&] {
          swept_sphere_obb_distance_jump_kernel<scalar_t, false>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<scalar_t>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<scalar_t>(),
            activation_distance.data_ptr<scalar_t>(),
            speed_dt.data_ptr<scalar_t>(),
            obb_accel.data_ptr<scalar_t>(),
            obb_bounds.data_ptr<scalar_t>(),
            obb_pose.data_ptr<scalar_t>(),
            obb_enable.data_ptr<uint8_t>(),
            n_env_obb.data_ptr<int32_t>(), max_nobs, batch_size,
            horizon, n_spheres, sweep_steps,
            transform_back);
        }));
      }
    }
    else
    {
      AT_DISPATCH_FLOATING_TYPES(
        distance.scalar_type(), "SphereObb_collision", ([&] {
        swept_sphere_obb_collision_kernel<scalar_t>
          << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
          activation_distance.data_ptr<scalar_t>(),
          obb_accel.data_ptr<scalar_t>(),
          obb_bounds.data_ptr<scalar_t>(),
          obb_pose.data_ptr<scalar_t>(),
          obb_enable.data_ptr<uint8_t>(),
          n_env_obb.data_ptr<int32_t>(), max_nobs, batch_size,
          horizon, n_spheres, sweep_steps);
      }));
    }
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { distance, closest_point, sparsity_idx }; // , debug_data};
}
