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

#include <vector>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include "check_cuda.h"
#include "cuda_precisions.h"

#define M 4
#define VOXEL_DEBUG true
#define VOXEL_UNOBSERVED_DISTANCE -1000.0

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

    __device__  __forceinline__ int3 robust_floor(const float3 f_grid, const float threshold=1e-04)
    {
      float3 nearest_grid = make_float3(round(f_grid.x), round(f_grid.y), round(f_grid.z));

      float3 abs_diff = (f_grid - nearest_grid);

      if (abs_diff.x >= threshold)
      {
        nearest_grid.x = floorf(f_grid.x);
      }
      if (abs_diff.y >= threshold)
      {
        nearest_grid.y = floorf(f_grid.y);
      }

      if (abs_diff.z >= threshold)
      {
        nearest_grid.z = floorf(f_grid.z);
      }
      return make_int3(nearest_grid);

    }

    #if CHECK_FP8
    __device__ __forceinline__ float
    get_array_value(const at::Float8_e4m3fn *grid_features, const int voxel_idx)
    {

      __nv_fp8_storage_t value_in = (reinterpret_cast<const __nv_fp8_storage_t*> (grid_features))[voxel_idx];

      float value = __half2float(__nv_cvt_fp8_to_halfraw(value_in, __NV_E4M3));

      return value;
    }

    __device__ __forceinline__ void
    get_array_value(const at::Float8_e4m3fn *grid_features, const int voxel_idx, float &value)
    {

      __nv_fp8_storage_t value_in = (reinterpret_cast<const __nv_fp8_storage_t*> (grid_features))[voxel_idx];

      value = __half2float(__nv_cvt_fp8_to_halfraw(value_in, __NV_E4M3));

    }

    #endif

    __device__ __forceinline__ float
    get_array_value(const at::BFloat16 *grid_features, const int voxel_idx)
    {

      __nv_bfloat16 value_in = (reinterpret_cast<const __nv_bfloat16*> (grid_features))[voxel_idx];

      float value = __bfloat162float(value_in);

      return value;
    }

    __device__ __forceinline__ float
    get_array_value(const at::Half *grid_features, const int voxel_idx)
    {

      __nv_half value_in = (reinterpret_cast<const __nv_half*> (grid_features))[voxel_idx];

      float value = __half2float(value_in);

      return value;
    }

    __device__ __forceinline__ float
    get_array_value(const float *grid_features, const int voxel_idx)
    {

      float value = grid_features[voxel_idx];

      return value;
    }

    __device__ __forceinline__ float
    get_array_value(const double *grid_features, const int voxel_idx)
    {

      float value = (float) grid_features[voxel_idx];

      return value;
    }


    __device__ __forceinline__ void
    get_array_value(const at::BFloat16 *grid_features, const int voxel_idx, float &value)
    {

      __nv_bfloat16 value_in = (reinterpret_cast<const __nv_bfloat16*> (grid_features))[voxel_idx];

      value = __bfloat162float(value_in);

    }

    __device__ __forceinline__ void
    get_array_value(const at::Half *grid_features, const int voxel_idx, float &value)
    {

      __nv_half value_in = (reinterpret_cast<const __nv_half*> (grid_features))[voxel_idx];

      value = __half2float(value_in);

    }

    __device__ __forceinline__ void
    get_array_value(const float *grid_features, const int voxel_idx, float &value)
    {

      value = grid_features[voxel_idx];

    }

    __device__ __forceinline__ void
    get_array_value(const double *grid_features, const int voxel_idx, float &value)
    {

      value = (float) grid_features[voxel_idx];

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
      if(x != 0 || y!= 0 || z!=0)
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
        C.x = sphere_pos.x ;
        C.y = sphere_pos.y ;
        C.z = sphere_pos.z ;
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
      if(sph_vel < 0.001)
      {
        return;
      }
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
#pragma unroll

        for (int i = 0; i < 3; i++)
        {
#pragma unroll

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

#pragma unroll

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



    __device__ __forceinline__ void
    compute_closest_point(const float3& bounds, const float4& sphere,
    float3& delta, float& distance, float& sph_distance)
    {
      float3 pt = make_float3(sphere.x, sphere.y, sphere.z);
      bool inside = true;

      if (max(max(fabs(sphere.x) - bounds.x, fabs(sphere.y) - bounds.y),
              fabs(sphere.z) - bounds.z) >= (0.0))
      {
        inside = false;
      }


      float3 val = make_float3(sphere.x,sphere.y,sphere.z);
      val = bounds - fabs(val);

      if(!inside)
      {


        if (val.x < 0) // it's outside limits, clamp:
        {
          pt.x = copysignf(bounds.x, sphere.x);
        }


        if (val.y < 0) // it's outside limits, clamp:
        {
          pt.y = copysignf(bounds.y, sphere.y);
        }

        if (val.z < 0) // it's outside limits, clamp:
        {
          pt.z = copysignf(bounds.z, sphere.z);
        }
      }
      else
      {


        val = fabs(val);




        if (val.y <= val.x && val.y <= val.z)
        {

          if(sphere.y > 0)
          {
            pt.y = bounds.y;
          }
          else
          {
            pt.y = -1 * bounds.y;
          }
        }

        else if (val.x <= val.y && val.x <= val.z)
        {
          if(sphere.x > 0)
          {
            pt.x = bounds.x;
          }
          else
          {
            pt.x = -1 * bounds.x;
          }
        }
        else if (val.z <= val.x && val.z <= val.y)
        {

          if(sphere.z > 0)
          {
            pt.z = bounds.z;
          }
          else
          {
            pt.z = -1 * bounds.z;
          }
        }




      }

      delta = make_float3(pt.x - sphere.x, pt.y - sphere.y, pt.z - sphere.z);

      distance = length(delta);
      if (distance == 0.0)
      {
        delta = -1.0 * make_float3(pt.x, pt.y, pt.z);
      }
      if (!inside) // outside
      {
        distance *= -1.0;
      }
      else // inside
      {
        delta = -1 * delta;
      }

      delta = normalize(delta);
      sph_distance = distance + sphere.w;
      //


    }

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
    check_sphere_aabb(const float3 bounds, const float4 sphere,  bool &inside,
    float3& delta, float& distance, float& sph_distance)
    {
      // if((fabs(sphere.x) - bounds.x) >= sphere.w || fabs(sphere.y) - bounds.y >=
      // sphere.w || (fabs(sphere.z) - bounds.z) >= sphere.w)

      inside = false;


      if (max(max(fabs(sphere.x) - bounds.x, fabs(sphere.y) - bounds.y),
              fabs(sphere.z) - bounds.z) >= (sphere.w))
      {
        return false;
      }
      // if it's within aabb, check more accurately:
      // compute closest point:

      compute_closest_point(bounds, sphere, delta, distance, sph_distance);
      if (sph_distance > 0)
      {
        inside = true;
      }

      return inside;
    }
    __device__ __forceinline__ float
    compute_distance_fn(
      const float3& bounds,
      const float4& sphere,
      const float max_distance,
      float3& delta,
      float& sph_dist,
      float& distance,
      bool& inside) // pass in cl_pt
    {

      // compute distance:
      float4 loc_sphere = sphere;
      loc_sphere.w = max_distance;
      distance = max_distance;
      check_sphere_aabb(bounds, loc_sphere, inside, delta, distance, sph_dist);

      //distance = fabsf(distance);
      return distance;
    }


template<bool SCALE_METRIC=true>
__device__ __forceinline__ void scale_eta_metric_vector(
const float eta,
float4 &sum_pt)
{
  float sph_dist = sum_pt.w;

  if (sph_dist == 0)
  {
    sum_pt.x = 0;
    sum_pt.y = 0;
    sum_pt.z = 0;
    sum_pt.w = 0;

    return;
  }
  sum_pt.w = sph_dist - eta;
  //sum_pt.x = sum_pt.x / sph_dist;
  //sum_pt.y = sum_pt.y / sph_dist;
  //sum_pt.z = sum_pt.z / sph_dist;

  if (SCALE_METRIC)
  {
    if (eta > 0.0 && sph_dist > 0)
    {
      //sum_pt.x = sum_pt.x * (1/sph_dist);
      //sum_pt.y = sum_pt.y * (1/sph_dist);
      //sum_pt.z = sum_pt.z * (1/sph_dist);

    if (sph_dist> eta)
    {
      sum_pt.w = sph_dist - 0.5 * eta;


    } else if (sph_dist <= eta)
    {

      sum_pt.w = (0.5 / eta) * (sph_dist) * (sph_dist);
      const float scale = (1 / eta) * (sph_dist);
      sum_pt.x = scale * sum_pt.x;
      sum_pt.y = scale * sum_pt.y;
      sum_pt.z = scale * sum_pt.z;
    }

    }

  }
}

    template<bool SCALE_METRIC=true>
    __device__ __forceinline__ void scale_eta_metric(
      const float3 delta,
      const float sph_dist,
      const float eta,
      float4& sum_pt)
    {
      // compute distance:
      //float sph_dist = 0;

      sum_pt.x =  delta.x;
      sum_pt.y =  delta.y;
      sum_pt.z =  delta.z;
      sum_pt.w = sph_dist;


      if(SCALE_METRIC)
      {

        if (sph_dist > 0)
        {

          if (sph_dist > eta)
          {
            sum_pt.w = sph_dist - 0.5 * eta;
          }
          else if (sph_dist <= eta)
          {
            sum_pt.w = (0.5 / eta) * (sph_dist) * (sph_dist);
            const float scale = (1.0 / eta) * (sph_dist);
            sum_pt.x = scale * sum_pt.x;
            sum_pt.y = scale * sum_pt.y;
            sum_pt.z = scale * sum_pt.z;
          }

        }
        else
        {
          sum_pt.x = 0.0;
          sum_pt.y = 0.0;
          sum_pt.z = 0.0;
          sum_pt.w = 0.0;

        }

      }





    }


    template<bool SCALE_METRIC=true>
    __device__ __forceinline__ void scale_eta_metric(const float4& sphere, const float4& cl_pt,
                                                     const float eta,
                                                     const bool inside,
                                                     float4& sum_pt)
    {
      // compute distance:
      float distance = 0;

      scale_eta_metric<SCALE_METRIC>(sphere, cl_pt, eta, inside, sum_pt, distance);


    }


    template<typename grid_scalar_t, bool INTERPOLATION=false>
    __device__ __forceinline__ void
    compute_voxel_index(
    const grid_scalar_t *grid_features,
    const float4& loc_grid_params,
    const float4& loc_sphere,
    int &voxel_idx,
    int3 &xyz_loc,
    int3 &xyz_grid,
    float &interpolated_distance)
    {




      const float3 loc_grid = make_float3(loc_grid_params.x, loc_grid_params.y, loc_grid_params.z);// - loc_grid_params.w;
      const  float3 sphere = make_float3(loc_sphere.x, loc_sphere.y, loc_sphere.z);
      const float inv_voxel_size = 1.0f / loc_grid_params.w;

      float3 f_grid = (loc_grid) * inv_voxel_size;


      xyz_grid = robust_floor(f_grid) + 1;


      xyz_loc = make_int3(((sphere.x + 0.5f * loc_grid.x) * inv_voxel_size),
                          ((sphere.y + 0.5f * loc_grid.y)* inv_voxel_size),
                          ((sphere.z + 0.5f * loc_grid.z) * inv_voxel_size));


      // check grid bounds:
      // 2 to catch numerical precision errors. 1 can be used when exact.
      // We need at least 1 as we
      // look at neighbouring voxels for finite difference
      const int offset = 2;
      if (xyz_loc.x >= xyz_grid.x - offset || xyz_loc.y >= xyz_grid.y - offset || xyz_loc.z >= xyz_grid.z - offset
          || xyz_loc.x <= offset || xyz_loc.y <= offset || xyz_loc.z <= offset
      )
      {
        voxel_idx = -1;
        return;
      }



      // find next nearest voxel to current point and then do weighted interpolation:
      voxel_idx = xyz_loc.x * xyz_grid.y * xyz_grid.z  + xyz_loc.y * xyz_grid.z + xyz_loc.z;


      // compute interpolation distance between voxel origin and sphere location:
      get_array_value(grid_features, voxel_idx, interpolated_distance);
      if(INTERPOLATION)
      {
      //
      float3 voxel_origin = (make_float3(xyz_loc) * loc_grid_params.w) - (loc_grid/2);


      float3 delta =  sphere - voxel_origin;
      int3 next_loc = make_int3(((make_float3(xyz_loc) + normalize(delta))));
      float ratio = length(delta) * inv_voxel_size;

      int next_voxel_idx = next_loc.x * xyz_grid.y * xyz_grid.z  + next_loc.y * xyz_grid.z + next_loc.z;

      interpolated_distance = ratio * interpolated_distance + (1 - ratio) * get_array_value(grid_features, next_voxel_idx) +
       max(0.0, (ratio * loc_grid_params.w) - loc_sphere.w);;

      }





    }





    template<typename grid_scalar_t, bool NORMALIZE=true, bool CENTRAL_DIFFERENCE=true, bool ADD_NOISE=false>
    __device__ __forceinline__ void
    compute_voxel_fd_gradient(
    const grid_scalar_t *grid_features,
    const int voxel_layer_start_idx,
    const int3& xyz_loc,
    const int3& xyz_grid,
    const float voxel_size,
    float4 &cl_pt)
    {

    float3 d_grad;
    if (CENTRAL_DIFFERENCE)
    {

    // x difference:
    int x_plus, x_minus, y_plus, y_minus, z_plus, z_minus;

    x_plus = (xyz_loc.x + 1) * xyz_grid.y * xyz_grid.z + xyz_loc.y * xyz_grid.z + xyz_loc.z;
    x_minus = (xyz_loc.x - 1)* xyz_grid.y * xyz_grid.z + xyz_loc.y * xyz_grid.z + xyz_loc.z;

    y_plus = (xyz_loc.x) * xyz_grid.y * xyz_grid.z + (xyz_loc.y + 1) * xyz_grid.z + xyz_loc.z;
    y_minus = (xyz_loc.x )* xyz_grid.y * xyz_grid.z + (xyz_loc.y -1) * xyz_grid.z + xyz_loc.z;

    z_plus = (xyz_loc.x) * xyz_grid.y * xyz_grid.z + xyz_loc.y * xyz_grid.z + xyz_loc.z + 1;
    z_minus = (xyz_loc.x)* xyz_grid.y * xyz_grid.z + xyz_loc.y * xyz_grid.z + xyz_loc.z - 1;


    float3 d_plus = make_float3(
      get_array_value(grid_features,voxel_layer_start_idx + x_plus),
      get_array_value(grid_features, voxel_layer_start_idx + y_plus),
      get_array_value(grid_features,voxel_layer_start_idx + z_plus));
    float3 d_minus = make_float3(
      get_array_value(grid_features,voxel_layer_start_idx + x_minus),
      get_array_value(grid_features, voxel_layer_start_idx + y_minus),
      get_array_value(grid_features,voxel_layer_start_idx + z_minus));


    d_grad = (d_plus - d_minus) * (1/(2*voxel_size));
    }
    if (!CENTRAL_DIFFERENCE)
    {
        // x difference:
    int x_minus,y_minus, z_minus;

    x_minus = (xyz_loc.x - 1)* xyz_grid.y * xyz_grid.z + xyz_loc.y * xyz_grid.z + xyz_loc.z;
    y_minus = (xyz_loc.x )* xyz_grid.y * xyz_grid.z + (xyz_loc.y -1) * xyz_grid.z + xyz_loc.z;
    z_minus = (xyz_loc.x)* xyz_grid.y * xyz_grid.z + xyz_loc.y * xyz_grid.z + xyz_loc.z - 1;


    float3 d_plus = make_float3(cl_pt.w, cl_pt.w, cl_pt.w);
    float3 d_minus = make_float3(
      get_array_value(grid_features,voxel_layer_start_idx + x_minus),
      get_array_value(grid_features, voxel_layer_start_idx + y_minus),
      get_array_value(grid_features,voxel_layer_start_idx + z_minus));


    d_grad = (d_plus - d_minus) * (1/voxel_size);
    }


    if (NORMALIZE)
    {
      if (!(d_grad.x ==0 && d_grad.y == 0 && d_grad.z == 0))
      {
        d_grad = normalize(d_grad);
      }
    }
    cl_pt.x = d_grad.x;
    cl_pt.y = d_grad.y;
    cl_pt.z = d_grad.z;
    if (ADD_NOISE)
    {
      if (cl_pt.z == 0 && cl_pt.x == 0 && cl_pt.y == 0)
      {
      cl_pt.x = 0.001;
      cl_pt.y = 0.001;
      }
    }

    }

    template<typename grid_scalar_t, bool SCALE_METRIC=true>
    __device__ __forceinline__ void
    compute_sphere_voxel_gradient(const grid_scalar_t *grid_features,
    const int voxel_layer_start_idx,
    const int num_voxels,
    const float4& loc_grid_params,
    const float4& loc_sphere,
    float4 &sum_pt,
    float &signed_distance,
    const float eta = 0.0,
    const float max_distance = -10.0,
    const bool transform_back = true)
    {
      int voxel_idx = 0;
      int3 xyz_loc = make_int3(0,0,0);
      int3 xyz_grid = make_int3(0,0,0);
      float interpolated_distance = 0.0;
      compute_voxel_index(grid_features, loc_grid_params, loc_sphere, voxel_idx, xyz_loc, xyz_grid, interpolated_distance);
      if (voxel_idx < 0 || voxel_idx >= num_voxels)
      {
        sum_pt.w = VOXEL_UNOBSERVED_DISTANCE;
        signed_distance = VOXEL_UNOBSERVED_DISTANCE;
        return;
      }


        //sum_pt.w = get_array_value(grid_features,voxel_layer_start_idx + voxel_idx);
        sum_pt.w = interpolated_distance;

        if ((!SCALE_METRIC && transform_back)|| (transform_back && sum_pt.w > -loc_sphere.w ))
        {
          // compute closest point:
          compute_voxel_fd_gradient(grid_features, voxel_layer_start_idx, xyz_loc, xyz_grid, loc_grid_params.w, sum_pt);
        }

        signed_distance = sum_pt.w;

        sum_pt.w += loc_sphere.w;


        scale_eta_metric_vector<SCALE_METRIC>(eta, sum_pt);


    }





    __device__ __forceinline__ void check_jump_distance(
      const float4& loc_sphere_1, const float4 loc_sphere_0, const float k0,
      const float3& bounds,
      const float max_distance,
      float3& delta,
      float& sph_dist,
      float& distance,
      bool& inside,
      const float eta,
      float4& sum_pt,
      float& curr_jump_distance) // we can pass in interpolated sphere here, also
                                 // need to pass cl_pt for use in
                                 // compute_sphere_gradient & compute_distance
    {
      const float4 interpolated_sphere =
        (k0) * loc_sphere_1 + (1 - k0) * loc_sphere_0;

      if (check_sphere_aabb(bounds, interpolated_sphere, inside, delta, distance, sph_dist))
      {
        float4 loc_grad = make_float4(0,0,0,0);
        scale_eta_metric<true>(delta, sph_dist, eta, loc_grad);
        sum_pt += loc_grad;

      }
      else
      {
        compute_distance_fn(
          bounds,
          interpolated_sphere,
          max_distance,
          delta,
          sph_dist,
          distance,
          inside
         );
      }
      curr_jump_distance += max(fabsf(distance), interpolated_sphere.w);

    }

    ///////////////////////////////////////////////////////////////
    // We write out the distance and gradients for all the spheres.
    // So we do not need to initize the output tensor to 0.
    // Each thread computes the max distance and gradients per sphere.
    // This version should be faster if we have enough spheres/threads
    // to fill the GPU as it avoid inter-thread communication and the
    // use of shared memory.
    ///////////////////////////////////////////////////////////////

    template<typename scalar_t, typename dist_scalar_t=float, bool SCALE_METRIC=true>
    __device__ __forceinline__ void sphere_obb_collision_fn(
      const scalar_t *sphere_position,
      const int env_idx,
      const int bn_sph_idx,
      const int sph_idx,
      dist_scalar_t *out_distance, const float *weight,
      const float *activation_distance, const float *obb_accel,
      const float *obb_bounds, const float *obb_mat,
      const uint8_t *obb_enable, const int max_nobs, const int nboxes)
    {
      float max_dist            = 0;
      const int   start_box_idx = max_nobs * env_idx;
      const float eta           = activation_distance[0];

      // Load sphere_position input
      float4 sphere_cache = *(float4 *)&sphere_position[bn_sph_idx * 4];

      if (sphere_cache.w < 0.0)
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
      bool inside = false;
      float distance = 0.0;
      float sph_dist = 0.0;
      float3 delta = make_float3(0,0,0);

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

        if (check_sphere_aabb(loc_bounds, loc_sphere, inside, delta, distance, sph_dist))
        {
          // using same primitive functions:
            max_dist = 1;
            break; // we exit without checking other cuboids if we found a collision.
        }

      }

      out_distance[bn_sph_idx] = weight[0] * max_dist;
    }

    template<typename scalar_t, typename dist_scalar_t=float, bool SCALE_METRIC=true, bool SUM_COLLISIONS=true>
    __device__ __forceinline__ void sphere_obb_distance_fn(
      const scalar_t *sphere_position,
      const int32_t env_idx,
      const int bn_sph_idx,
      const int sph_idx,
      dist_scalar_t *out_distance, scalar_t *closest_pt, uint8_t *sparsity_idx,
      const float *weight, const float *activation_distance,
      const float *obb_accel, const float *obb_bounds,
      const float *obb_mat, const uint8_t *obb_enable, const int max_nobs,
      const int nboxes, const bool transform_back)
    {
      float max_dist  = 0.0;

      const float eta = activation_distance[0];
      float3 max_grad = make_float3(0.0, 0.0, 0.0);

      // Load sphere_position input
      float4 sphere_cache = *(float4 *)&sphere_position[bn_sph_idx * 4];

      if (sphere_cache.w < 0.0)
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
      float4 loc_grad = make_float4(0,0,0,0);
      bool inside = false;
      float distance = 0.0;
      float sph_dist = 0.0;
      float3 delta = make_float3(0,0,0);
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
        if (check_sphere_aabb(loc_bounds, loc_sphere, inside, delta, distance, sph_dist))
        {
          // compute closest point:
          //loc_bounds = loc_bounds + loc_sphere.w;

          // using same primitive functions:
          scale_eta_metric<SCALE_METRIC>(delta, sph_dist, eta, loc_grad);


          if (SUM_COLLISIONS)
          {
            if (loc_grad.w > 0)
            {
              max_dist += loc_grad.w;

              if (transform_back)
              {

                inv_transform_vec_quat_add(obb_pos, obb_quat, loc_grad, max_grad);
              }
            }
          }
          else
          {
          if (loc_grad.w > max_dist)
          {
            max_dist = loc_grad.w;

            if (transform_back)
            {
              inv_transform_vec_quat(obb_pos, obb_quat, loc_grad, max_grad);

            }
          }

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


    template<typename scalar_t, typename dist_scalar_t=float,  bool SCALE_METRIC=false, bool SUM_COLLISIONS=false>
    __device__ __forceinline__ void sphere_obb_esdf_fn(
      const scalar_t *sphere_position,
      const int32_t env_idx,
      const int bn_sph_idx,
      const int sph_idx,
      dist_scalar_t *out_distance, scalar_t *closest_pt, uint8_t *sparsity_idx,
      const float *weight, const float *activation_distance,
      const float *max_distance,
      const float *obb_accel, const float *obb_bounds,
      const float *obb_mat, const uint8_t *obb_enable, const int max_nobs,
      const int nboxes, const bool transform_back)
    {

      const float eta = max_distance[0];
      float max_dist  = -1 * eta;

      float3 max_grad = make_float3(0.0, 0.0, 0.0);

      // Load sphere_position input
      float4 sphere_cache = *(float4 *)&sphere_position[bn_sph_idx * 4];
      if (sphere_cache.w < 0.0)
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

      //const float sphere_radius = sphere_cache.w + eta;

      const int start_box_idx = max_nobs * env_idx;

      float4 loc_sphere = make_float4(0.0);
      float4 obb_quat   = make_float4(0.0);
      float3 obb_pos    = make_float3(0.0);
      float3 loc_bounds = make_float3(0.0);
      bool inside = false;
      float distance = 0.0;
      float sph_dist = 0.0;
      float3 delta = make_float3(0,0,0);
      float4 loc_grad = make_float4(0,0,0,0);

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
        if (check_sphere_aabb(loc_bounds, loc_sphere, inside, delta, distance, sph_dist))
        {
          // compute closest point:


          // using same primitive functions:
          scale_eta_metric<SCALE_METRIC>(delta, sph_dist, eta, loc_grad);


          if (loc_grad.w > max_dist)
          {
            max_dist = loc_grad.w;

            if (transform_back)
            {
              inv_transform_vec_quat(obb_pos, obb_quat, loc_grad, max_grad);

            }
          }
        }
      }
      // subtract radius:
      max_dist = max_dist - sphere_cache.w;
      if (transform_back)
      {
        *(float3 *)&closest_pt[bn_sph_idx * 4] = max_grad;
      }
      out_distance[bn_sph_idx] = max_dist;
    }

    template<typename grid_scalar_t, typename geom_scalar_t=float, typename dist_scalar_t=float, typename grad_scalar_t=float, bool SCALE_METRIC=true, int NUM_LAYERS=100>
    __device__ __forceinline__ void sphere_voxel_distance_fn(
      const geom_scalar_t *sphere_position,
      const int32_t env_idx,
      const int bn_sph_idx,
      const int sph_idx,
      dist_scalar_t *out_distance,
      grad_scalar_t *closest_pt,
      uint8_t *sparsity_idx,
      const float *weight,
      const float *activation_distance,
      const float *max_distance,
      const grid_scalar_t *grid_features,
      const float *grid_params,
      const float *obb_mat,
      const uint8_t *obb_enable,
      const int max_nobs,
      const int num_voxels,
      const bool transform_back)
    {
      float max_dist  = 0.0;
      float max_distance_local = max_distance[0];
      max_distance_local = -1 * max_distance_local;
      const float eta = activation_distance[0];
      float3 max_grad = make_float3(0.0, 0.0, 0.0);

      // Load sphere_position input
      float4 sphere_cache = *(float4 *)&sphere_position[bn_sph_idx * 4];

      if (sphere_cache.w < 0.0)
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
      const int local_env_idx = max_nobs * env_idx;
      float signed_distance = 0;

      float4 loc_sphere = make_float4(0.0);
      float4 obb_quat   = make_float4(0.0);
      float3 obb_pos    = make_float3(0.0);
      float4 loc_grid_params = make_float4(0.0);

      if (NUM_LAYERS <= 4)
      {

      #pragma unroll
      for (int layer_idx=0; layer_idx < NUM_LAYERS; layer_idx++)
      {


      int local_env_layer_idx = local_env_idx + layer_idx;
      if (obb_enable[local_env_layer_idx] != 0) // disabled obstacle
      {

        load_obb_pose(&obb_mat[(local_env_layer_idx) * 8], obb_pos,
                      obb_quat);
        loc_grid_params = *(float4 *)&grid_params[local_env_layer_idx*4];

        transform_sphere_quat(obb_pos, obb_quat, sphere_cache, loc_sphere);
        int voxel_layer_start_idx = local_env_layer_idx * num_voxels;
        // check distance:
        float4 cl;
        compute_sphere_voxel_gradient<grid_scalar_t,SCALE_METRIC>(grid_features,
        voxel_layer_start_idx, num_voxels,
        loc_grid_params, loc_sphere, cl, signed_distance, eta,
        max_distance_local, transform_back);
        if (cl.w>0.0)
        {
          max_dist += cl.w;
          if (transform_back)
          {
            inv_transform_vec_quat_add(obb_pos, obb_quat, cl, max_grad);

          }
        }
      }
      }
      }
      else
      {




      for (int layer_idx=0; layer_idx < max_nobs; layer_idx++)
      {


      int local_env_layer_idx = local_env_idx + layer_idx;
      if (obb_enable[local_env_layer_idx] != 0) // disabled obstacle
      {

        load_obb_pose(&obb_mat[(local_env_layer_idx) * 8], obb_pos,
                      obb_quat);
        loc_grid_params = *(float4 *)&grid_params[local_env_layer_idx*4];

        transform_sphere_quat(obb_pos, obb_quat, sphere_cache, loc_sphere);
        int voxel_layer_start_idx = local_env_layer_idx * num_voxels;
        // check distance:
        float4 cl;
        compute_sphere_voxel_gradient<grid_scalar_t,SCALE_METRIC>(grid_features,
        voxel_layer_start_idx, num_voxels,
        loc_grid_params, loc_sphere, cl, signed_distance, eta,
        max_distance_local, transform_back);
        if (cl.w>0.0)
        {
          max_dist += cl.w;
          if (transform_back)
          {
            inv_transform_vec_quat_add(obb_pos, obb_quat, cl, max_grad);

          }
        }
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


    template<typename grid_scalar_t, typename scalar_t=float, typename dist_scalar_t=float,
    bool SCALE_METRIC=true, bool ENABLE_SPEED_METRIC=true, bool SUM_COLLISIONS=true,
    int NUM_LAYERS=100>
    __device__ __forceinline__ void swept_sphere_voxel_distance_fn(
      const scalar_t *sphere_position,
      const int env_idx,
      const int b_idx,
      const int h_idx,
      const int sph_idx,
      dist_scalar_t *out_distance,
      scalar_t *closest_pt,
      uint8_t *sparsity_idx,
      const float *weight,
      const float *activation_distance,
      const float *max_distance,
      const float *speed_dt,
      const grid_scalar_t *grid_features,
      const float *grid_params,
      const float *grid_pose,
      const uint8_t *grid_enable,
      const int max_nobs,
      const int env_ngrid,
      const int num_voxels,
      const int batch_size,
      const int horizon,
      const int nspheres,
      const int sweep_steps,
      const bool transform_back)
    {
      const int   sw_steps      = sweep_steps;
      const int   b_addrs       =
        b_idx * horizon * nspheres; // + h_idx * n_spheres + sph_idx;

      // We read the same obstacles across

      // Load sphere_position input
      // if h_idx == horizon -1, we just read the same index
      const int bhs_idx = b_addrs + h_idx * nspheres + sph_idx;





      float max_dist  = 0.0;
      float max_distance_local = max_distance[0];
      max_distance_local = -1 * max_distance_local;
      const float eta = activation_distance[0];
      float3 max_grad = make_float3(0.0, 0.0, 0.0);

      // Load sphere_position input
      float4 sphere_1_cache = *(float4 *)&sphere_position[bhs_idx * 4];

      if (sphere_1_cache.w < 0.0)
      {
        // write zeros for cost:
        out_distance[bhs_idx] = 0;

        // write zeros for gradient if not zero:
        if (sparsity_idx[bhs_idx] != 0)
        {
          sparsity_idx[bhs_idx]               = 0;
          *(float4 *)&closest_pt[bhs_idx * 4] = make_float4(0.0);
        }
        return;
      }
      sphere_1_cache.w += eta;
      float4 loc_sphere_0, loc_sphere_2;

      bool  sweep_back = false;
      bool  sweep_fwd = false;
      float sphere_0_distance, sphere_2_distance, sphere_0_len, sphere_2_len;


      const float dt  = speed_dt[0];
      float4 sphere_0_cache = make_float4(0,0,0,0);
      float4 sphere_2_cache = make_float4(0,0,0,0);

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

      float signed_distance = 0.0;
      const int local_env_idx = max_nobs * env_idx;

      float4 loc_sphere = make_float4(0.0);
      float4 obb_quat   = make_float4(0.0);
      float3 obb_pos    = make_float3(0.0);
      float4 loc_grid_params = make_float4(0.0);
      float4 sum_grad = make_float4(0.0, 0.0, 0.0, 0.0);
      float4 cl;
      float jump_mid_distance = 0.0;
      float k0;
      float temp_jump_distance = 0.0;

      if (NUM_LAYERS <= 4)
      {


      #pragma unroll
      for (int layer_idx=0; layer_idx < NUM_LAYERS; layer_idx++)
      {
      float curr_jump_distance = 0.0;

      int local_env_layer_idx = local_env_idx + layer_idx;
      sum_grad *= 0.0;
      if (grid_enable[local_env_layer_idx] != 0) // disabled obstacle
      {

        load_obb_pose(&grid_pose[(local_env_layer_idx) * 8], obb_pos,
                      obb_quat);
        loc_grid_params = *(float4 *)&grid_params[local_env_layer_idx*4];

        transform_sphere_quat(obb_pos, obb_quat, sphere_1_cache, loc_sphere);
        transform_sphere_quat(obb_pos, obb_quat, sphere_0_cache, loc_sphere_0);
        transform_sphere_quat(obb_pos, obb_quat, sphere_2_cache, loc_sphere_2);

        int voxel_layer_start_idx = local_env_layer_idx * num_voxels;
        // check distance:
        compute_sphere_voxel_gradient<grid_scalar_t,SCALE_METRIC>(grid_features,
        voxel_layer_start_idx, num_voxels,
        loc_grid_params, loc_sphere, cl, signed_distance, eta,
        max_distance_local, transform_back);
        if (cl.w>0.0)
        {
          sum_grad += cl;
          jump_mid_distance = signed_distance;
        }
        else if (signed_distance != VOXEL_UNOBSERVED_DISTANCE)
        {
          jump_mid_distance = -1 * signed_distance;
        }


        jump_mid_distance = max(jump_mid_distance, loc_sphere.w);
        curr_jump_distance = jump_mid_distance;
        if (sweep_back && curr_jump_distance < sphere_0_distance/2)
        {
          for (int j=0; j<sw_steps; j++)
          {
            if (curr_jump_distance >= sphere_0_len/2)
            {
              break;
            }
            temp_jump_distance = 0.0;
            k0 = 1 - (curr_jump_distance/sphere_0_len);
            // compute collision
            const float4 interpolated_sphere =
            (k0)*loc_sphere + (1 - k0) * loc_sphere_0;

            compute_sphere_voxel_gradient<grid_scalar_t,SCALE_METRIC>(
              grid_features,
              voxel_layer_start_idx, num_voxels,
              loc_grid_params, interpolated_sphere, cl, signed_distance, eta,
              max_distance_local, transform_back);
            if (cl.w>0.0)
            {
              sum_grad += cl;
              temp_jump_distance = signed_distance;
            }
            else if (signed_distance != VOXEL_UNOBSERVED_DISTANCE)
            {
              temp_jump_distance = -1 * signed_distance;
            }
            temp_jump_distance = max(temp_jump_distance, loc_sphere.w);
            curr_jump_distance += temp_jump_distance;


          }
        }
        curr_jump_distance = jump_mid_distance;
        if (sweep_fwd && curr_jump_distance < sphere_2_distance/2)
        {
          for (int j=0; j<sw_steps; j++)
          {
            if (curr_jump_distance >= sphere_2_len/2)
            {
              break;
            }
            temp_jump_distance = 0.0;
            k0 = 1 - (curr_jump_distance/sphere_2_len);
            // compute collision
            const float4 interpolated_sphere =
            (k0)*loc_sphere + (1 - k0) * loc_sphere_2;

             compute_sphere_voxel_gradient<grid_scalar_t,SCALE_METRIC>(
             grid_features,
             voxel_layer_start_idx, num_voxels,
             loc_grid_params, interpolated_sphere, cl, signed_distance, eta,
             max_distance_local, transform_back);
             if (cl.w>0.0)
            {
              sum_grad += cl;
              temp_jump_distance = signed_distance;
            }
            else if (signed_distance != VOXEL_UNOBSERVED_DISTANCE)
            {
              temp_jump_distance = -1 * signed_distance;
            }
            temp_jump_distance = max(temp_jump_distance, loc_sphere.w);
            curr_jump_distance += temp_jump_distance;

        }
        }
        if (sum_grad.w > 0.0 )
        {
          max_dist += sum_grad.w;
          if (transform_back)
          {
            inv_transform_vec_quat_add(obb_pos, obb_quat, sum_grad, max_grad);
          }

        }




      }
      }
      }
      else
      {



      for (int layer_idx=0; layer_idx < max_nobs; layer_idx++)
      {
      float curr_jump_distance = 0.0;

      int local_env_layer_idx = local_env_idx + layer_idx;
      sum_grad *= 0.0;
      if (grid_enable[local_env_layer_idx] != 0) // disabled obstacle
      {

        load_obb_pose(&grid_pose[(local_env_layer_idx) * 8], obb_pos,
                      obb_quat);
        loc_grid_params = *(float4 *)&grid_params[local_env_layer_idx*4];

        transform_sphere_quat(obb_pos, obb_quat, sphere_1_cache, loc_sphere);
        transform_sphere_quat(obb_pos, obb_quat, sphere_0_cache, loc_sphere_0);
        transform_sphere_quat(obb_pos, obb_quat, sphere_2_cache, loc_sphere_2);

        int voxel_layer_start_idx = local_env_layer_idx * num_voxels;
        // check distance:
        compute_sphere_voxel_gradient<grid_scalar_t,SCALE_METRIC>(grid_features,
        voxel_layer_start_idx, num_voxels,
        loc_grid_params, loc_sphere, cl, signed_distance, eta,
        max_distance_local, transform_back);
        if (cl.w>0.0)
        {
          sum_grad += cl;
          jump_mid_distance = signed_distance;
        }
        else if (signed_distance != VOXEL_UNOBSERVED_DISTANCE)
        {
          jump_mid_distance = -1 * signed_distance;
        }


        jump_mid_distance = max(jump_mid_distance, loc_sphere.w);
        curr_jump_distance = jump_mid_distance;
        if (sweep_back && curr_jump_distance < sphere_0_distance/2)
        {
          for (int j=0; j<sw_steps; j++)
          {
            if (curr_jump_distance >= sphere_0_len/2)
            {
              break;
            }
            temp_jump_distance = 0.0;
            k0 = 1 - (curr_jump_distance/sphere_0_len);
            // compute collision
            const float4 interpolated_sphere =
            (k0)*loc_sphere + (1 - k0) * loc_sphere_0;

            compute_sphere_voxel_gradient<grid_scalar_t,SCALE_METRIC>(
              grid_features,
              voxel_layer_start_idx, num_voxels,
              loc_grid_params, interpolated_sphere, cl, signed_distance, eta,
              max_distance_local, transform_back);
            if (cl.w>0.0)
            {
              sum_grad += cl;
              temp_jump_distance = signed_distance;
            }
            else if (signed_distance != VOXEL_UNOBSERVED_DISTANCE)
            {
              temp_jump_distance = -1 * signed_distance;
            }
            temp_jump_distance = max(temp_jump_distance, loc_sphere.w);
            curr_jump_distance += temp_jump_distance;


          }
        }
        curr_jump_distance = jump_mid_distance;
        if (sweep_fwd && curr_jump_distance < sphere_2_distance/2)
        {
          for (int j=0; j<sw_steps; j++)
          {
            if (curr_jump_distance >= sphere_2_len/2)
            {
              break;
            }
            temp_jump_distance = 0.0;
            k0 = 1 - (curr_jump_distance/sphere_2_len);
            // compute collision
            const float4 interpolated_sphere =
            (k0)*loc_sphere + (1 - k0) * loc_sphere_2;

             compute_sphere_voxel_gradient<grid_scalar_t,SCALE_METRIC>(
             grid_features,
             voxel_layer_start_idx, num_voxels,
             loc_grid_params, interpolated_sphere, cl, signed_distance, eta,
             max_distance_local, transform_back);
             if (cl.w>0.0)
            {
              sum_grad += cl;
              temp_jump_distance = signed_distance;
            }
            else if (signed_distance != VOXEL_UNOBSERVED_DISTANCE)
            {
              temp_jump_distance = -1 * signed_distance;
            }
            temp_jump_distance = max(temp_jump_distance, loc_sphere.w);
            curr_jump_distance += temp_jump_distance;

        }
        }
        if (sum_grad.w > 0.0 )
        {
          max_dist += sum_grad.w;
          if (transform_back)
          {
            inv_transform_vec_quat_add(obb_pos, obb_quat, sum_grad, max_grad);
          }

        }




      }
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
      if (ENABLE_SPEED_METRIC)
      {
        if (sweep_back && sweep_fwd)
        {
          scale_speed_metric(sphere_0_cache, sphere_1_cache, sphere_2_cache, dt,
                             transform_back, max_dist, max_grad);
        }
      }
      // else max_dist != 0
      max_dist = weight[0] * max_dist;

      if (transform_back)
      {
        *(float3 *)&closest_pt[bhs_idx * 4] = weight[0] * max_grad;
      }
      out_distance[bhs_idx] = max_dist;
      sparsity_idx[bhs_idx] = 1;

    }


    template<typename grid_scalar_t, typename scalar_t,   typename dist_scalar_t=float,
    bool BATCH_ENV_T=true,
    bool SCALE_METRIC=true,
    bool ENABLE_SPEED_METRIC=true,
    bool SUM_COLLISIONS=false,
    int NUM_LAYERS=100>
    __global__ void swept_sphere_voxel_distance_jump_kernel(
      const scalar_t *sphere_position,
      dist_scalar_t *out_distance,
      scalar_t *closest_pt, uint8_t *sparsity_idx, const float *weight,
      const float *activation_distance,
      const float *max_distance,
      const float *speed_dt,
      const grid_scalar_t *grid_features, const float *grid_params,
      const float *grid_pose, const uint8_t *grid_enable,
      const int32_t *n_env_grid, const int32_t *env_query_idx, const int max_nobs,
      const int num_voxels,
      const int batch_size, const int horizon, const int nspheres,
      const int sweep_steps,
      const bool transform_back)
      {
      const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int b_idx = t_idx / (horizon * nspheres);

      const int h_idx   = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
      const int sph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);

      if ((sph_idx >= nspheres) || (b_idx >= batch_size) || (h_idx >= horizon))
      {
        return;
      }

      int env_idx    = 0;

      if (BATCH_ENV_T)
      {
        env_idx = env_query_idx[b_idx];
      }

      const int env_nboxes = n_env_grid[env_idx];
      swept_sphere_voxel_distance_fn<grid_scalar_t, scalar_t,dist_scalar_t, SCALE_METRIC,
      ENABLE_SPEED_METRIC, SUM_COLLISIONS, NUM_LAYERS>(
        sphere_position,  env_idx, b_idx, h_idx, sph_idx,
        out_distance, closest_pt,
        sparsity_idx, weight, activation_distance, max_distance,
        speed_dt,
        grid_features,
        grid_params, grid_pose, grid_enable, max_nobs, env_nboxes, num_voxels, batch_size,
        horizon, nspheres, sweep_steps, transform_back);
      }

    template<typename grid_scalar_t, typename geom_scalar_t=float, typename dist_scalar_t=float, typename grad_scalar_t=float, bool SCALE_METRIC=true, int NUM_LAYERS=100>
    __device__ __forceinline__ void sphere_voxel_esdf_fn(
      const geom_scalar_t *sphere_position,
      const int32_t env_idx,
      const int bn_sph_idx,
      const int sph_idx,
      dist_scalar_t *out_distance,
      grad_scalar_t *closest_pt,
      uint8_t *sparsity_idx,
      const float *weight,
      const float *activation_distance,
      const float *max_distance,
      const grid_scalar_t *grid_features,
      const float *grid_params,
      const float *obb_mat,
      const uint8_t *obb_enable,
      const int max_nobs,
      const int num_voxels,
      const bool transform_back)
    {
      float max_distance_local = max_distance[0];
      const float eta = max_distance_local;
      float max_dist  = -1 * max_distance_local;
      max_distance_local = -1 * max_distance_local;

      float3 max_grad = make_float3(0.0, 0.0, 0.0);

      // Load sphere_position input
      float4 sphere_cache = *(float4 *)&sphere_position[bn_sph_idx * 4];

      if (sphere_cache.w < 0.0)
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
      const float sphere_radius = sphere_cache.w;

      sphere_cache.w += eta;
      const int local_env_idx = max_nobs * env_idx;

      float4 loc_sphere = make_float4(0.0);
      float4 obb_quat   = make_float4(0.0);
      float3 obb_pos    = make_float3(0.0);
      float4 loc_grid_params = make_float4(0.0);

      float signed_distance = 0;

      for (int layer_idx=0; layer_idx < max_nobs; layer_idx++)
      {


      int local_env_layer_idx = local_env_idx + layer_idx;
      if (obb_enable[local_env_layer_idx] != 0) // disabled obstacle
      {

        load_obb_pose(&obb_mat[(local_env_layer_idx) * 8], obb_pos,
                      obb_quat);
        loc_grid_params = *(float4 *)&grid_params[local_env_layer_idx*4];

        transform_sphere_quat(obb_pos, obb_quat, sphere_cache, loc_sphere);
        int voxel_layer_start_idx = local_env_layer_idx * num_voxels;
        // check distance:
        float4 cl;

        compute_sphere_voxel_gradient<grid_scalar_t,SCALE_METRIC>(grid_features,
        voxel_layer_start_idx, num_voxels,
        loc_grid_params, loc_sphere, cl, signed_distance, eta,
        max_distance_local, transform_back);
        if (cl.w>max_dist)
        {
          max_dist = cl.w;
          if (transform_back)
          {
            inv_transform_vec_quat(obb_pos, obb_quat, cl, max_grad);

          }
        }
      }
      }




      max_dist = max_dist - sphere_radius;
      if (transform_back)
      {
        *(float3 *)&closest_pt[bn_sph_idx * 4] = max_grad;
      }
      out_distance[bn_sph_idx] = max_dist;

    }


    template<typename scalar_t,  typename dist_scalar_t=float, bool ENABLE_SPEED_METRIC=true, bool SUM_COLLISIONS=true, int SWEEP_STEPS=-1>
    __device__ __forceinline__ void swept_sphere_obb_distance_fn(
      const scalar_t *sphere_position,
      const int env_idx, const int b_idx,
      const int h_idx, const int sph_idx,
      dist_scalar_t *out_distance,
      scalar_t *closest_pt,
      uint8_t *sparsity_idx,
      const float *weight,
      const float *activation_distance, const float *speed_dt,
      const float *obb_accel, const float *obb_bounds,
      const float *obb_mat,
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
      int   sw_steps      = SWEEP_STEPS;
      const float max_distance = 1000.0;

      if (SWEEP_STEPS == -1)
      {
        sw_steps = sweep_steps;
      }
      const int   start_box_idx = max_nobs * env_idx;
      const int   b_addrs       =
        b_idx * horizon * nspheres; // + h_idx * n_spheres + sph_idx;

      // We read the same obstacles across

      // Load sphere_position input
      // if h_idx == horizon -1, we just read the same index
      const int bhs_idx = b_addrs + h_idx * nspheres + sph_idx;

      float4 sphere_1_cache = *(float4 *)&sphere_position[bhs_idx * 4];


      if (sphere_1_cache.w < 0.0)
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
      float4 sphere_0_cache = make_float4(0,0,0,0);
      float4 sphere_2_cache = make_float4(0,0,0,0);
      float4 loc_grad = make_float4(0,0,0,0);
      bool inside = false;
      float distance = 0.0;
      float sph_dist = 0.0;
      float3 delta = make_float3(0,0,0);
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

        //const float3 grad_loc_bounds = loc_bounds + sphere_1_cache.w; // assuming sphere radius
                                                                      // doesn't change

        transform_sphere_quat(obb_pos, obb_quat, sphere_1_cache, loc_sphere_1);
        transform_sphere_quat(obb_pos, obb_quat, sphere_0_cache, loc_sphere_0);
        transform_sphere_quat(obb_pos, obb_quat, sphere_2_cache, loc_sphere_2);


        // assuming sphere position is in box frame:
        // read data:
        float4 sum_pt = make_float4(0.0, 0.0, 0.0, 0.0);
        curr_jump_distance = 0.0;
        // check at exact timestep:
        if (check_sphere_aabb(loc_bounds, loc_sphere_1, inside, delta, curr_jump_distance, sph_dist))
        {

          scale_eta_metric<true>(delta, sph_dist, eta, sum_pt);
        }
        else if (sweep_back || sweep_fwd)
        {
          // there is no collision, compute the distance to obstacle:
          curr_jump_distance = compute_distance_fn(loc_bounds, loc_sphere_1,
          max_distance, delta, sph_dist, distance, inside);

        }
        curr_jump_distance = fabsf(curr_jump_distance) - loc_sphere_1.w;
        curr_jump_distance = max(curr_jump_distance, loc_sphere_1.w);

        const float jump_mid_distance = curr_jump_distance;

        // compute distance between sweep spheres:
        if (sweep_back && (jump_mid_distance < sphere_0_distance / 2))
        {

          // get unit vector:
          // loc_sphere_0 = (loc_sphere_0 - loc_sphere_1)/(sphere_0_len);

          // loop over sweep steps and accumulate distance:
          #pragma unroll
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
            check_jump_distance(loc_sphere_1, loc_sphere_0,
            k0,
            loc_bounds,
            max_distance,
            delta,
            sph_dist,
            distance,
            inside,
            eta,
            sum_pt,
            curr_jump_distance);
          }
        }

        if (sweep_fwd && (jump_mid_distance < (sphere_2_len / 2)))
        {
          curr_jump_distance = jump_mid_distance;

          #pragma unroll
          for (int j = 0; j < sw_steps; j++)
          {
            if (curr_jump_distance >= (sphere_2_len / 2))
            {
              break;
            }
            k0 = 1 - curr_jump_distance / sphere_2_len;
            check_jump_distance(loc_sphere_1, loc_sphere_2,
            k0,
            loc_bounds,
            max_distance,
            delta,
            sph_dist,
            distance,
            inside,
            eta,
            sum_pt,
            curr_jump_distance);
          }
        }
        if (SUM_COLLISIONS)
        {
        if (sum_pt.w > 0) // max_dist starts at 0
        {
          max_dist += sum_pt.w;

          // transform point back if required:
          if (transform_back)
          {
            //inv_transform_vec_quat(obb_pos, obb_quat, sum_pt, max_grad);
            inv_transform_vec_quat_add(obb_pos, obb_quat, sum_pt, max_grad);

          }

          // break;// break after first obstacle collision
        }
        }
        else
        {

        if (sum_pt.w > max_dist) // max_dist starts at 0
        {
          max_dist = sum_pt.w;

          // transform point back if required:
          if (transform_back)
          {
            inv_transform_vec_quat(obb_pos, obb_quat, sum_pt, max_grad);
            //inv_transform_vec_quat_add(obb_pos, obb_quat, sum_pt, max_grad);

          }

          // break;// break after first obstacle collision
        }
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
      if (ENABLE_SPEED_METRIC)
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
    template<typename scalar_t, typename dist_scalar_t=float>
    __device__ __forceinline__ void swept_sphere_obb_collision_fn(
      const scalar_t *sphere_position,
      const int env_idx, const int b_idx,
      const int h_idx, const int sph_idx, dist_scalar_t *out_distance,
      const float *weight, const float *activation_distance,
      const float *obb_accel, const float *obb_bounds,
      const float *obb_mat, const uint8_t *obb_enable, const int max_nobs,
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
      float4 loc_grad = make_float4(0,0,0,0);
      float3 delta = make_float3(0,0,0);
      bool inside = false;
      float curr_jump_distance = 0.0;
      float sph_dist = 0.0;
      // We read the same obstacles across

      // Load sphere_position input
      // if h_idx == horizon -1, we just read the same index
      float4 sphere_1_cache = *(float4 *)&sphere_position[bhs_idx * 4];

      if (sphere_1_cache.w < 0.0)
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

        if (check_sphere_aabb(loc_bounds, loc_sphere_1, inside, delta, curr_jump_distance, sph_dist))
        {

            max_dist = 1;
            break;
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

            if (check_sphere_aabb(loc_bounds, interpolated_sphere, inside, delta, curr_jump_distance, sph_dist))
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
            if (check_sphere_aabb(loc_bounds, interpolated_sphere, inside, delta, curr_jump_distance, sph_dist))
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

    template<typename scalar_t, typename dist_scalar_t=float, bool BATCH_ENV_T=true, bool SCALE_METRIC=true, bool SUM_COLLISIONS=true, bool COMPUTE_ESDF=false>
    __global__ void sphere_obb_distance_kernel(
      const scalar_t *sphere_position,
      dist_scalar_t *out_distance,
      scalar_t *closest_pt, uint8_t *sparsity_idx, const float *weight,
      const float *activation_distance,
      const float *max_distance,
      const float *obb_accel,
      const float *obb_bounds, const float *obb_mat,
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

      if (BATCH_ENV_T)
      {
        env_idx =
          env_query_idx[b_idx]; // read env idx from current batch idx

      }
      const int env_nboxes = n_env_obb[env_idx];   // read nboxes in current environment
      if (COMPUTE_ESDF)
      {
          sphere_obb_esdf_fn<scalar_t, dist_scalar_t, SCALE_METRIC, SUM_COLLISIONS>(sphere_position, env_idx, bn_sph_idx, sph_idx, out_distance,
                             closest_pt, sparsity_idx, weight, activation_distance, max_distance,
                             obb_accel, obb_bounds, obb_mat, obb_enable, max_nobs,
                             env_nboxes, transform_back);

      }
      else
      {
        sphere_obb_distance_fn<scalar_t, dist_scalar_t, SCALE_METRIC, SUM_COLLISIONS>(sphere_position, env_idx, bn_sph_idx, sph_idx, out_distance,
                             closest_pt, sparsity_idx, weight, activation_distance,
                             obb_accel, obb_bounds, obb_mat, obb_enable, max_nobs,
                             env_nboxes, transform_back);

      }

      // return the sphere distance here:
      // sync threads and do block level reduction:
    }


    template<typename grid_scalar_t, typename geom_scalar_t=float, typename dist_scalar_t=float, typename grad_scalar_t=float, bool BATCH_ENV_T=false, bool SCALE_METRIC=true, bool COMPUTE_ESDF=false, int NUM_LAYERS=100>
    __global__ void sphere_voxel_distance_kernel(
      const geom_scalar_t *sphere_position,
      dist_scalar_t *out_distance,
      grad_scalar_t *closest_pt,
      uint8_t *sparsity_idx,
      const float *weight,
      const float *activation_distance,
      const float *max_distance,
      const grid_scalar_t *grid_features,
      const float *grid_params, const float *obb_mat,
      const uint8_t *obb_enable, const int32_t *n_env_obb,
      const int32_t *env_query_idx,
      const int max_nobs, const int num_voxels,
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

      if (BATCH_ENV_T)
      {
        env_idx =
          env_query_idx[b_idx]; // read env idx from current batch idx

      }
      if (COMPUTE_ESDF)
      {

        sphere_voxel_esdf_fn<grid_scalar_t, geom_scalar_t, dist_scalar_t, grad_scalar_t, SCALE_METRIC, NUM_LAYERS>(sphere_position, env_idx, bn_sph_idx, sph_idx, out_distance,
                             closest_pt, sparsity_idx, weight, activation_distance, max_distance,
                             grid_features, grid_params, obb_mat, obb_enable, max_nobs, num_voxels,
                             transform_back);

      }
      else
      {
         sphere_voxel_distance_fn<grid_scalar_t, geom_scalar_t, dist_scalar_t, grad_scalar_t, SCALE_METRIC, NUM_LAYERS>(sphere_position, env_idx, bn_sph_idx, sph_idx, out_distance,
                             closest_pt, sparsity_idx, weight, activation_distance, max_distance,
                             grid_features, grid_params, obb_mat, obb_enable, max_nobs, num_voxels,
                             transform_back);
      }

      // return the sphere distance here:
      // sync threads and do block level reduction:
    }


    template<typename scalar_t, typename dist_scalar_t, bool BATCH_ENV_T, bool ENABLE_SPEED_METRIC, bool SUM_COLLISIONS=false, int SWEEP_STEPS=-1>
    __global__ void swept_sphere_obb_distance_jump_kernel(
      const scalar_t *sphere_position,
      dist_scalar_t *out_distance,
      scalar_t *closest_pt, uint8_t *sparsity_idx, const float *weight,
      const float *activation_distance, const float *speed_dt,
      const float *obb_accel, const float *obb_bounds,
      const float *obb_pose, const uint8_t *obb_enable,
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

      int env_idx    = 0;

      if (BATCH_ENV_T)
      {
        env_idx = env_query_idx[b_idx];
      }

      const int env_nboxes = n_env_obb[env_idx];

      swept_sphere_obb_distance_fn<scalar_t, dist_scalar_t, ENABLE_SPEED_METRIC, SUM_COLLISIONS, SWEEP_STEPS>(
        sphere_position,  env_idx, b_idx, h_idx, sph_idx, out_distance, closest_pt,
        sparsity_idx, weight, activation_distance, speed_dt, obb_accel,
        obb_bounds, obb_pose, obb_enable, max_nobs, env_nboxes, batch_size,
        horizon, nspheres, sweep_steps, transform_back);
    }

    template<typename scalar_t, typename dist_scalar_t=float>
    __global__ void swept_sphere_obb_collision_kernel(
      const scalar_t *sphere_position,
      dist_scalar_t *out_distance,
      const float *weight, const float *activation_distance,
      const float *obb_accel, const float *obb_bounds,
      const float *obb_pose, const uint8_t *obb_enable,
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

    template<typename scalar_t, typename dist_scalar_t=float>
    __global__ void swept_sphere_obb_collision_batch_env_kernel(
      const scalar_t *sphere_position,
      dist_scalar_t *out_distance,
      const float *weight, const float *activation_distance,
      const float *obb_accel, const float *obb_bounds,
      const float *obb_pose, const uint8_t *obb_enable,
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


    template<typename scalar_t, typename dist_scalar_t, bool BATCH_ENV_T>
    __global__ void sphere_obb_collision_batch_env_kernel(
      const scalar_t *sphere_position,
      dist_scalar_t *out_distance,
      const float *weight, const float *activation_distance,
      const float *obb_accel, const float *obb_bounds,
      const float *obb_mat, const uint8_t *obb_enable,
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
      int env_idx    = 0;
      if(BATCH_ENV_T)
      {
        env_idx = env_query_idx[b_idx];
      }

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
                const torch::Tensor max_distance,
                const torch::Tensor obb_accel,       // n_boxes, 4, 4
                const torch::Tensor obb_bounds,      // n_boxes, 3
                const torch::Tensor obb_pose,        // n_boxes, 4, 4
                const torch::Tensor obb_enable,      // n_boxes, 4, 4
                const torch::Tensor n_env_obb,       // n_boxes, 4, 4
                const torch::Tensor env_query_idx,   // n_boxes, 4, 4
                const int max_nobs, const int batch_size, const int horizon,
                const int n_spheres, const bool transform_back,
                const bool compute_distance, const bool use_batch_env,
                const bool sum_collisions,
                const bool compute_esdf)
{
  using namespace Curobo::Geometry;
  cudaStream_t stream      = at::cuda::getCurrentCUDAStream();
  const int    bnh_spheres = n_spheres * batch_size * horizon; //
  const bool scale_metric = true;
  int threadsPerBlock = bnh_spheres;
  const bool sum_collisions_ = true;

  if (threadsPerBlock > 128)
  {
    threadsPerBlock = 128;
  }
  int blocksPerGrid = (bnh_spheres + threadsPerBlock - 1) / threadsPerBlock;

  if (!compute_distance)
  {

  AT_DISPATCH_FLOATING_TYPES(
    distance.scalar_type(), "SphereObb_clpt_collision", ([&]{
    auto collision_kernel = sphere_obb_collision_batch_env_kernel<scalar_t, float, false>;
    auto batch_collision_kernel = sphere_obb_collision_batch_env_kernel<scalar_t,float, true>;
    auto selected_k = collision_kernel;
    if (use_batch_env)
    {
        selected_k = batch_collision_kernel;
    }

      selected_k<< < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<float>(), weight.data_ptr<float>(),
          activation_distance.data_ptr<float>(),
          obb_accel.data_ptr<float>(),
          obb_bounds.data_ptr<float>(),
          obb_pose.data_ptr<float>(),
          obb_enable.data_ptr<uint8_t>(),
          n_env_obb.data_ptr<int32_t>(),
          env_query_idx.data_ptr<int32_t>(), max_nobs, batch_size,
          horizon, n_spheres);
    }));

  }
  else
  {

  // typename scalar_t, typename dist_scalar_t=float, bool BATCH_ENV_T=true, bool SCALE_METRIC=true, bool SUM_COLLISIONS=true, bool COMPUTE_ESDF=false
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kBFloat16, FP8_TYPE_MACRO,
      distance.scalar_type(), "SphereObb_clpt", ([&]{
    auto distance_kernel = sphere_obb_distance_kernel<scalar_t, scalar_t, false, scale_metric, sum_collisions_,false>;
    if (use_batch_env)
    {
      if (compute_esdf)
      {
              distance_kernel =  sphere_obb_distance_kernel<scalar_t, scalar_t, true, false, sum_collisions_, true>;

      }
      else
      {

      distance_kernel =  sphere_obb_distance_kernel<scalar_t,scalar_t,  true, scale_metric, sum_collisions_, false>;
      }

    }
    else if (compute_esdf)
    {
      distance_kernel = sphere_obb_distance_kernel<scalar_t, scalar_t, false, false, sum_collisions_, true>;
    }

    distance_kernel<< < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<scalar_t>(),
          closest_point.data_ptr<scalar_t>(),
          sparsity_idx.data_ptr<uint8_t>(),
          weight.data_ptr<float>(),
          activation_distance.data_ptr<float>(),
          max_distance.data_ptr<float>(),
          obb_accel.data_ptr<float>(),
          obb_bounds.data_ptr<float>(),
          obb_pose.data_ptr<float>(),
          obb_enable.data_ptr<uint8_t>(),
          n_env_obb.data_ptr<int32_t>(),
          env_query_idx.data_ptr<int32_t>(),
          max_nobs, batch_size,
          horizon, n_spheres, transform_back);


  }));
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
  const torch::Tensor activation_distance,
  const torch::Tensor speed_dt,
  const torch::Tensor obb_accel,       // n_boxes, 4, 4
  const torch::Tensor obb_bounds,      // n_boxes, 3
  const torch::Tensor obb_pose,        // n_boxes, 4, 4
  const torch::Tensor obb_enable,      // n_boxes, 4,
  const torch::Tensor n_env_obb,       // n_boxes, 4, 4
  const torch::Tensor env_query_idx,   // n_boxes, 4, 4
  const int max_nobs, const int batch_size, const int horizon,
  const int n_spheres, const int sweep_steps, const bool enable_speed_metric,
  const bool transform_back, const bool compute_distance,
  const bool use_batch_env,
  const bool sum_collisions)
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

  if (sum_collisions)
  {
    const bool sum_collisions_ = true;

  if (use_batch_env)
  {
    if (compute_distance)
    {
      if (enable_speed_metric)
      {
        // This is the best kernel for now
        AT_DISPATCH_FLOATING_TYPES(
          distance.scalar_type(), "Swept_SphereObb_clpt", ([&] {
          swept_sphere_obb_distance_jump_kernel<scalar_t,float,  true, true, sum_collisions_>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<float>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<float>(),
            activation_distance.data_ptr<float>(),
            speed_dt.data_ptr<float>(),
            obb_accel.data_ptr<float>(),
            obb_bounds.data_ptr<float>(),
            obb_pose.data_ptr<float>(),
            obb_enable.data_ptr<uint8_t>(),
            n_env_obb.data_ptr<int32_t>(),
            env_query_idx.data_ptr<int32_t>(), max_nobs, batch_size,
            horizon, n_spheres, sweep_steps,
            transform_back);
        }));
      }
      else
      {
        AT_DISPATCH_FLOATING_TYPES(
          distance.scalar_type(), "Swept_SphereObb_clpt", ([&] {
          swept_sphere_obb_distance_jump_kernel<scalar_t,float,  true, false, sum_collisions_>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<float>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<float>(),
            activation_distance.data_ptr<float>(),
            speed_dt.data_ptr<float>(),
            obb_accel.data_ptr<float>(),
            obb_bounds.data_ptr<float>(),
            obb_pose.data_ptr<float>(),
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
        swept_sphere_obb_collision_batch_env_kernel<scalar_t, float>
          << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<float>(), weight.data_ptr<float>(),
          activation_distance.data_ptr<float>(),
          obb_accel.data_ptr<float>(),
          obb_bounds.data_ptr<float>(),
          obb_pose.data_ptr<float>(),
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
        if (sweep_steps == 4)
        {
        AT_DISPATCH_FLOATING_TYPES(
          distance.scalar_type(), "Swept_SphereObb_clpt", ([&] {
          swept_sphere_obb_distance_jump_kernel<scalar_t, float, false, true, sum_collisions_,4>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<float>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<float>(),
            activation_distance.data_ptr<float>(),
            speed_dt.data_ptr<float>(),
            obb_accel.data_ptr<float>(),
            obb_bounds.data_ptr<float>(),
            obb_pose.data_ptr<float>(),
            obb_enable.data_ptr<uint8_t>(),
            n_env_obb.data_ptr<int32_t>(),
            env_query_idx.data_ptr<int32_t>(), max_nobs, batch_size,
            horizon, n_spheres, sweep_steps,
            transform_back);
        }));
        }
        else
        {


        AT_DISPATCH_FLOATING_TYPES(
          distance.scalar_type(), "Swept_SphereObb_clpt", ([&] {
          swept_sphere_obb_distance_jump_kernel<scalar_t, float, false, true, sum_collisions_>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<float>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<float>(),
            activation_distance.data_ptr<float>(),
            speed_dt.data_ptr<float>(),
            obb_accel.data_ptr<float>(),
            obb_bounds.data_ptr<float>(),
            obb_pose.data_ptr<float>(),
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
        // This is the best kernel for now
        AT_DISPATCH_FLOATING_TYPES(
          distance.scalar_type(), "Swept_SphereObb_clpt", ([&] {
          swept_sphere_obb_distance_jump_kernel<scalar_t,float, false, false, sum_collisions_>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<float>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<float>(),
            activation_distance.data_ptr<float>(),
            speed_dt.data_ptr<float>(),
            obb_accel.data_ptr<float>(),
            obb_bounds.data_ptr<float>(),
            obb_pose.data_ptr<float>(),
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
      AT_DISPATCH_FLOATING_TYPES(
        distance.scalar_type(), "SphereObb_collision", ([&] {
        swept_sphere_obb_collision_kernel<scalar_t, float>
          << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<float>(), weight.data_ptr<float>(),
          activation_distance.data_ptr<float>(),
          obb_accel.data_ptr<float>(),
          obb_bounds.data_ptr<float>(),
          obb_pose.data_ptr<float>(),
          obb_enable.data_ptr<uint8_t>(),
          n_env_obb.data_ptr<int32_t>(), max_nobs, batch_size,
          horizon, n_spheres, sweep_steps);
      }));
    }
  }
  }
  else
  {
    const bool sum_collisions_ = true;
  if (use_batch_env)
  {
    if (compute_distance)
    {
      if (enable_speed_metric)
      {
        // This is the best kernel for now
        AT_DISPATCH_FLOATING_TYPES(
          distance.scalar_type(), "Swept_SphereObb_clpt", ([&] {
          swept_sphere_obb_distance_jump_kernel<scalar_t, float, true, true, sum_collisions_>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<float>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<float>(),
            activation_distance.data_ptr<float>(),
            speed_dt.data_ptr<float>(),
            obb_accel.data_ptr<float>(),
            obb_bounds.data_ptr<float>(),
            obb_pose.data_ptr<float>(),
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
          swept_sphere_obb_distance_jump_kernel<scalar_t, float, true, false, sum_collisions_>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<float>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<float>(),
            activation_distance.data_ptr<float>(),
            speed_dt.data_ptr<float>(),
            obb_accel.data_ptr<float>(),
            obb_bounds.data_ptr<float>(),
            obb_pose.data_ptr<float>(),
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
        swept_sphere_obb_collision_batch_env_kernel<scalar_t, float>
          << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<float>(), weight.data_ptr<float>(),
          activation_distance.data_ptr<float>(),
          obb_accel.data_ptr<float>(),
          obb_bounds.data_ptr<float>(),
          obb_pose.data_ptr<float>(),
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
          swept_sphere_obb_distance_jump_kernel<scalar_t,float,  false, true, sum_collisions_>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<float>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<float>(),
            activation_distance.data_ptr<float>(),
            speed_dt.data_ptr<float>(),
            obb_accel.data_ptr<float>(),
            obb_bounds.data_ptr<float>(),
            obb_pose.data_ptr<float>(),
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
          swept_sphere_obb_distance_jump_kernel<scalar_t, float, false, false, sum_collisions_>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
            sphere_position.data_ptr<scalar_t>(),
            distance.data_ptr<float>(),
            closest_point.data_ptr<scalar_t>(),
            sparsity_idx.data_ptr<uint8_t>(),
            weight.data_ptr<float>(),
            activation_distance.data_ptr<float>(),
            speed_dt.data_ptr<float>(),
            obb_accel.data_ptr<float>(),
            obb_bounds.data_ptr<float>(),
            obb_pose.data_ptr<float>(),
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
      AT_DISPATCH_FLOATING_TYPES(
        distance.scalar_type(), "SphereObb_collision", ([&] {
        swept_sphere_obb_collision_kernel<scalar_t, float>
          << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
          sphere_position.data_ptr<scalar_t>(),
          distance.data_ptr<float>(), weight.data_ptr<float>(),
          activation_distance.data_ptr<float>(),
          obb_accel.data_ptr<float>(),
          obb_bounds.data_ptr<float>(),
          obb_pose.data_ptr<float>(),
          obb_enable.data_ptr<uint8_t>(),
          n_env_obb.data_ptr<int32_t>(), max_nobs, batch_size,
          horizon, n_spheres, sweep_steps);
      }));
    }
  }

  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { distance, closest_point, sparsity_idx }; // , debug_data};
}


std::vector<torch::Tensor>
sphere_voxel_clpt(const torch::Tensor sphere_position, // batch_size, 3
                torch::Tensor distance,
                torch::Tensor closest_point,         // batch size, 3
                torch::Tensor sparsity_idx, const torch::Tensor weight,
                const torch::Tensor activation_distance,
                const torch::Tensor max_distance,
                const torch::Tensor grid_features,       // n_boxes, 4, 4
                const torch::Tensor grid_params,      // n_boxes, 3
                const torch::Tensor obb_pose,        // n_boxes, 4, 4
                const torch::Tensor obb_enable,      // n_boxes, 4, 4
                const torch::Tensor n_env_obb,
                const torch::Tensor env_query_idx,   // n_boxes, 4, 4
                const int max_nobs, const int batch_size, const int horizon,
                const int n_spheres, const bool transform_back,
                const bool compute_distance, const bool use_batch_env,
                const bool sum_collisions,
                const bool compute_esdf)
{
  using namespace Curobo::Geometry;
  cudaStream_t stream      = at::cuda::getCurrentCUDAStream();
  const int    bnh_spheres = n_spheres * batch_size * horizon; //
  const bool scale_metric = true;
  const int num_voxels = grid_features.size(-2);
  int threadsPerBlock = bnh_spheres;

  if (threadsPerBlock > 128)
  {
    threadsPerBlock = 128;
  }
  // bfloat16
  //ScalarType::Float8_e4m3fn

  int blocksPerGrid = (bnh_spheres + threadsPerBlock - 1) / threadsPerBlock;


  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kBFloat16, FP8_TYPE_MACRO,
  grid_features.scalar_type(), "SphereVoxel_clpt", ([&]
  {

    auto kernel_esdf = sphere_voxel_distance_kernel<scalar_t, float, float, float, false, false, true, 100>;
    auto kernel_distance_1 =  sphere_voxel_distance_kernel<scalar_t, float, float, float, false, scale_metric, false, 1>;
    auto kernel_distance_2 =  sphere_voxel_distance_kernel<scalar_t, float, float, float, false, scale_metric, false, 2>;
    auto kernel_distance_3 =  sphere_voxel_distance_kernel<scalar_t, float, float, float, false, scale_metric, false, 3>;
    auto kernel_distance_4 =  sphere_voxel_distance_kernel<scalar_t, float, float, float, false, scale_metric, false, 4>;

    auto kernel_distance_n =  sphere_voxel_distance_kernel<scalar_t, float, float, float, false, scale_metric, false, 100>;
    auto selected_kernel = kernel_distance_n;
    if (compute_esdf)
    {
      selected_kernel = kernel_esdf;

    }
    else
    {
      switch (max_nobs){
        case 1:
          selected_kernel = kernel_distance_1;
          break;
        case 2:
          selected_kernel = kernel_distance_2;
          break;
        case 3:
          selected_kernel = kernel_distance_3;
          break;
        case 4:
          selected_kernel = kernel_distance_4;
          break;
        default:
          selected_kernel = kernel_distance_n;
          break;
      }
    }

      selected_kernel
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        sphere_position.data_ptr<float>(),
        distance.data_ptr<float>(),
        closest_point.data_ptr<float>(),
        sparsity_idx.data_ptr<uint8_t>(),
        weight.data_ptr<float>(),
        activation_distance.data_ptr<float>(),
        max_distance.data_ptr<float>(),
        grid_features.data_ptr<scalar_t>(),
        grid_params.data_ptr<float>(),
        obb_pose.data_ptr<float>(),
        obb_enable.data_ptr<uint8_t>(),
        n_env_obb.data_ptr<int32_t>(),
        env_query_idx.data_ptr<int32_t>(),
        max_nobs,
        num_voxels,
        batch_size,
        horizon, n_spheres, transform_back);
    }));





  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { distance, closest_point, sparsity_idx };
}



std::vector<torch::Tensor>
swept_sphere_voxel_clpt(const torch::Tensor sphere_position, // batch_size, 3
                torch::Tensor distance,
                torch::Tensor closest_point,         // batch size, 3
                torch::Tensor sparsity_idx, const torch::Tensor weight,
                const torch::Tensor activation_distance,
                const torch::Tensor max_distance,
                const torch::Tensor speed_dt,
                const torch::Tensor grid_features,       // n_boxes, 4, 4
                const torch::Tensor grid_params,      // n_boxes, 3
                const torch::Tensor grid_pose,        // n_boxes, 4, 4
                const torch::Tensor grid_enable,      // n_boxes, 4, 4
                const torch::Tensor n_env_grid,
                const torch::Tensor env_query_idx,   // n_boxes, 4, 4
                const int max_nobs,
                const int batch_size,
                const int horizon,
                const int n_spheres,
                const int sweep_steps,
                const bool enable_speed_metric,
                const bool transform_back,
                const bool compute_distance,
                const bool use_batch_env,
                const bool sum_collisions)
{
  using namespace Curobo::Geometry;

  const at::cuda::OptionalCUDAGuard guard(sphere_position.device());
  CHECK_INPUT(sphere_position);
  CHECK_INPUT(distance);
  CHECK_INPUT(closest_point);
  CHECK_INPUT(sparsity_idx);
  CHECK_INPUT(weight);
  CHECK_INPUT(activation_distance);
  CHECK_INPUT(max_distance);
  CHECK_INPUT(speed_dt);
  CHECK_INPUT(grid_features);
  CHECK_INPUT(grid_params);
  CHECK_INPUT(grid_pose);
  CHECK_INPUT(grid_enable);
  CHECK_INPUT(n_env_grid);
  CHECK_INPUT(env_query_idx);
  cudaStream_t stream      = at::cuda::getCurrentCUDAStream();
  const int    bnh_spheres = n_spheres * batch_size * horizon; //
  const bool scale_metric = true;
  const int num_voxels = grid_features.size(-2);
  int threadsPerBlock = bnh_spheres;

  if (threadsPerBlock > 128)
  {
    threadsPerBlock = 128;
  }
  // bfloat16
  //ScalarType::Float8_e4m3fn

  int blocksPerGrid = (bnh_spheres + threadsPerBlock - 1) / threadsPerBlock;

  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kBFloat16, FP8_TYPE_MACRO,
      grid_features.scalar_type(), "SphereVoxel_clpt", ([&] {

      auto collision_kernel_n = swept_sphere_voxel_distance_jump_kernel<scalar_t, float, float, false, scale_metric, true, false, 100>;
      auto collision_kernel_1 = swept_sphere_voxel_distance_jump_kernel<scalar_t, float, float, false, scale_metric, true, false, 1>;
      auto collision_kernel_2 = swept_sphere_voxel_distance_jump_kernel<scalar_t, float, float, false, scale_metric, true, false, 2>;
      auto collision_kernel_3 = swept_sphere_voxel_distance_jump_kernel<scalar_t, float, float, false, scale_metric, true, false, 3>;
      auto collision_kernel_4 = swept_sphere_voxel_distance_jump_kernel<scalar_t, float, float, false, scale_metric, true, false, 4>;
      auto selected_kernel = collision_kernel_n;
      switch (max_nobs){
        case 1:
          selected_kernel = collision_kernel_1;
          break;
        case 2:
          selected_kernel = collision_kernel_2;
          break;
        case 3:
          selected_kernel = collision_kernel_3;
          break;
        case 4:
          selected_kernel = collision_kernel_4;
          break;
        default:
          selected_kernel = collision_kernel_n;
      }

      selected_kernel<< < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        sphere_position.data_ptr<float>(),
        distance.data_ptr<float>(),
        closest_point.data_ptr<float>(),
        sparsity_idx.data_ptr<uint8_t>(),
        weight.data_ptr<float>(),
        activation_distance.data_ptr<float>(),
        max_distance.data_ptr<float>(),
        speed_dt.data_ptr<float>(),
        grid_features.data_ptr<scalar_t>(),
        grid_params.data_ptr<float>(),
        grid_pose.data_ptr<float>(),
        grid_enable.data_ptr<uint8_t>(),
        n_env_grid.data_ptr<int32_t>(),
        env_query_idx.data_ptr<int32_t>(),
        max_nobs,
        num_voxels,
        batch_size,
        horizon, n_spheres, sweep_steps, transform_back);
    }));



  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { distance, closest_point, sparsity_idx };
}

