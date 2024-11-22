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
#include <cmath>
#include <cuda_fp16.h>
#include <vector>

#define SINGLE_GOAL 0
#define BATCH_GOAL 1
#define GOALSET 2
#define BATCH_GOALSET 3

namespace Curobo
{
  namespace Geometry
  {
    template<typename scalar_t>__inline__ __device__ scalar_t relu(scalar_t var)
    {
      if (var < 0)
        return 0;
      else
        return var;
    }

    template<typename scalar_t>
    __global__ void self_collision_distance_kernel(
      scalar_t *out_distance,        // batch x 1
      scalar_t *out_vec,             // batch x nspheres x 4
      const scalar_t *robot_spheres, // batch x nspheres x 4
      const scalar_t *offsets,
      const uint8_t *coll_matrix,
      const int batch_size,
      const int nspheres, const scalar_t *weight, const bool write_grad = false)
    {
      const int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

      if (batch_idx >= batch_size)
      {
        return;
      }
      float  r_diff, distance;
      float  max_penetration = 0;
      float4 sph1, sph2;
      int    sph1_idx = -1;
      int    sph2_idx = -1;

      // iterate over spheres:
      for (int i = 0; i < nspheres; i++)
      {
        sph1 = *(float4 *)&robot_spheres[batch_idx * nspheres * 4 + i * 4];
        sph1.w += offsets[i];

        for (int j = i + 1; j < nspheres; j++)
        {
          if(coll_matrix[i * nspheres + j] == 1)
          {
            sph2 = *(float4 *)&robot_spheres[batch_idx * nspheres * 4 + j * 4];
            sph2.w += offsets[j];

            // compute sphere distance:
            r_diff = sph1.w + sph2.w;
            float d      = sqrt((sph1.x - sph2.x) * (sph1.x - sph2.x) +
                                  (sph1.y - sph2.y) * (sph1.y - sph2.y) +
                                  (sph1.z - sph2.z) * (sph1.z - sph2.z));
            distance = (r_diff - d);

            if (distance > max_penetration)
            {
              max_penetration = distance;
              sph1_idx        = i;
              sph2_idx        = j;
            }
          }
        }
      }

      // write out pose distance:
      if (max_penetration > 0)
      {
        out_distance[batch_idx] = weight[0] * max_penetration;

        if (write_grad)
        {
          float3 sph1_g =
              *(float3 *)&robot_spheres[4 * (batch_idx * nspheres + sph1_idx)];
          float3 sph2_g =
              *(float3 *)&robot_spheres[4 * (batch_idx * nspheres + sph2_idx)];
            float3 dist_vec = normalize(sph1_g - sph2_g);
          *(float3 *)&out_vec[batch_idx * nspheres * 4 + sph1_idx * 4] =
            weight[0] * -1 * dist_vec;
          *(float3 *)&out_vec[batch_idx * nspheres * 4 + sph2_idx * 4] =
            weight[0] * dist_vec;
        }
      }
    }

    typedef struct {
      float   d;
      int16_t i;
      int16_t j;
    } dist_t;


    ///////////////////////////////////////////////////////////
    // n warps per row
    // ndpt rows per warp
    ///////////////////////////////////////////////////////////
    template<typename scalar_t>
    __global__ void self_collision_distance_kernel4(
      scalar_t *out_distance,        // batch x 1
      scalar_t *out_vec,             // batch x nspheres x 3
      const scalar_t *robot_spheres, // batch x nspheres x 3
      const scalar_t *offsets, const uint8_t *coll_matrix, const int batch_size,
      const int nspheres,
      const int ndpt,                // number of distances to be computed per thread
      const int nwpr,                // number of warps per row
      const scalar_t *weight, uint8_t *sparse_index,
      const bool write_grad = false)
    {
      int batch_idx = blockIdx.x;
      int warp_idx  = threadIdx.x / 32;
      int i         = ndpt * (warp_idx / nwpr); // starting row number for this warp
      int j         = (warp_idx % nwpr) * 32;   // starting column number for this warp

      dist_t max_d = { 0.0, 0, 0 };// .d, .i, .j
      __shared__ dist_t max_darr[32];

      // Optimization: About 1/3 of the warps will have no work.
      // We compute distances only when i<j. If i is never <j in this warp, we have
      // no work
      if (i > j + 31) // this warp has no work
      {
        max_darr[warp_idx] = max_d;
        return;
      }

      // load robot_spheres to shared memory
      extern __shared__ float4 __rs_shared[];

      if (threadIdx.x < nspheres)
      {
        float4 sph = *(float4 *)&robot_spheres[4 * (batch_idx * nspheres + threadIdx.x)];

        // float4 sph = make_float4(robot_spheres[3 * (batch_idx * nspheres + threadIdx.x)],
        // robot_spheres[3 * (batch_idx * nspheres + threadIdx.x) + 1],
        // robot_spheres[3 * (batch_idx * nspheres + threadIdx.x) + 2],
        // robot_spheres_radius[threadIdx.x]) ;
        sph.w                   += offsets[threadIdx.x];
        __rs_shared[threadIdx.x] = sph;
      }
      __syncthreads();

      //////////////////////////////////////////////////////
      // Compute distances and store the maximum per thread
      // in registers (max_d).
      // Each thread computes up to ndpt distances.
      // two warps per row
      //////////////////////////////////////////////////////
      // int nspheres_2 = nspheres * nspheres;

      j = j + threadIdx.x % 32; // column number for this thread

      float4 sph2;

      if (j < nspheres)
      {
        sph2 = __rs_shared[j]; // we need not load sph2 in every iteration.

        for (int k = 0; k < ndpt; k++, i++) // increment i also here
        {
          if ((i < nspheres) && (j > i))
          {
            // check if self collision is allowed here:
            if (coll_matrix[i * nspheres + j] == 1)
            {
              float4 sph1 = __rs_shared[i];
              //
              //if ((sph1.w <= 0.0) || (sph2.w <= 0.0))
              //{
              //  continue;
              //}
              float r_diff = sph1.w + sph2.w;
              float d      = sqrt((sph1.x - sph2.x) * (sph1.x - sph2.x) +
                                  (sph1.y - sph2.y) * (sph1.y - sph2.y) +
                                  (sph1.z - sph2.z) * (sph1.z - sph2.z));

              // float distance = max((float)0.0, (float)(r_diff - d));
              float distance = r_diff - d;

              // printf("%d, %d: (%d, %d) %f new\n", blockIdx.x, threadIdx.x, i, j,
              // distance);
              if (distance > max_d.d)
              {
                max_d.d = distance;
                max_d.i = i;
                max_d.j = j;
              }
            }
          }
        }
      }

      // max_d has the result max for this thread

      //////////////////////////////////////////////////////
      // Reduce gridDim.x values using gridDim.x threads
      //////////////////////////////////////////////////////

      // Perform warp-wide reductions
      // Optimization: Skip the reduction if all the values are zero
      unsigned zero_mask = __ballot_sync(
        0xffffffff, max_d.d != 0.0); // we expect most values to be 0. So,

      // zero_mask should be 0 in the common case.
      if (zero_mask != 0)            // some of the values are non-zero
      {
        unsigned mask = __ballot_sync(0xffffffff, threadIdx.x < blockDim.x);

        if (threadIdx.x < blockDim.x)
        {
          // dist_t max_d = dist_sh[threadIdx.x];
#pragma unroll 4

          for (int offset = 16; offset > 0; offset /= 2)
          {
            uint64_t nd     = __shfl_down_sync(mask, *(uint64_t *)&max_d, offset);
            dist_t   d_temp = *(dist_t *)&nd;

            if (((threadIdx.x + offset) < blockDim.x)  && d_temp.d > max_d.d)
            {
              max_d = d_temp;
            }
          }
        }
      }

      // thread0 in the warp has the max_d for the warp
      if (threadIdx.x % 32 == 0)
      {
        max_darr[warp_idx] = max_d;

        // printf("threadIdx.x=%d, blockIdx.x=%d, max_d=%f\n", threadIdx.x,
        // blockIdx.x, max_d);
      }

      if (threadIdx.x < nspheres)
      {
        if (write_grad && (sparse_index[batch_idx * nspheres + threadIdx.x] != 0))
        {
          *(float4 *)&out_vec[batch_idx * nspheres * 4 + threadIdx.x * 4] =
            make_float4(0.0);
          sparse_index[batch_idx * nspheres + threadIdx.x] = 0;
        }
      }
      __syncthreads();

      if (threadIdx.x == 0)
      {
        dist_t max_d = max_darr[0];

        // TODO: This can be parallized
        for (int i = 1; i < (blockDim.x + 31) / 32; i++)
        {
          if (max_darr[i].d > max_d.d)
          {
            max_d = max_darr[i];
          }
        }

        //////////////////////////////////////////////////////
        // Write out the final results
        //////////////////////////////////////////////////////
        if (max_d.d != 0.0)
        {
          out_distance[batch_idx] = weight[0] * max_d.d;

          if (write_grad)
          {
            // NOTE: spheres can be read from rs_shared
            float3 sph1 =
              *(float3 *)&robot_spheres[4 * (batch_idx * nspheres + max_d.i)];
            float3 sph2 =
              *(float3 *)&robot_spheres[4 * (batch_idx * nspheres + max_d.j)];
            float3 dist_vec = normalize(sph1 - sph2);
            *(float3 *)&out_vec[batch_idx * nspheres * 4 + max_d.i * 4] =
              weight[0] * -1 * dist_vec;
            *(float3 *)&out_vec[batch_idx * nspheres * 4 + max_d.j * 4] =
              weight[0] *  dist_vec;
            sparse_index[batch_idx * nspheres + max_d.i] = 1;
            sparse_index[batch_idx * nspheres + max_d.j] = 1;
          }
        }
        else
        {
          out_distance[batch_idx] = 0;
        }
      }
    }

    template<typename scalar_t, int ndpt, int NBPB>
    __global__ void self_collision_distance_kernel7(
      scalar_t *out_distance,        // batch x 1
      scalar_t *out_vec,             // batch x nspheres x 3
      uint8_t *sparse_index,
      const scalar_t *robot_spheres, // batch x nspheres x 3
      const scalar_t *offsets,       // nspheres
      const scalar_t *weight, const int16_t *locations_, const int batch_size,
      const int nspheres, const bool write_grad = false)
    {
      uint32_t batch_idx = blockIdx.x * NBPB;
      uint8_t  nbpb      = min(NBPB, batch_size - batch_idx);

      if (nbpb == 0)
        return;

      // Layout in shared memory:
      //     sphere1[batch=0] sphere1[batch=1] sphere1[batch=2] sphere1[batch=4]
      //     sphere2[batch=0] sphere2[batch=1] sphere2[batch=2] sphere2[batch=4]
      //     ...
      extern __shared__ float4 __rs_shared[];

      if (threadIdx.x < nspheres) // threadIdx.x is sphere index
      {
#pragma unroll

        for (int l = 0; l < nbpb; l++)
        {
          float4 sph = *(float4 *)&robot_spheres[4 * ((batch_idx + l) * nspheres + threadIdx.x)];

          // float4 sph = make_float4(
          //  robot_spheres[3 * ((batch_idx + l) * nspheres + threadIdx.x)],
          //  robot_spheres[3 * ((batch_idx + l) * nspheres + threadIdx.x) + 1],
          //  robot_spheres[3 * ((batch_idx + l) * nspheres + threadIdx.x) + 2],
          //  robot_spheres_radius[threadIdx.x]
          // );

          sph.w                              += offsets[threadIdx.x];
          __rs_shared[NBPB * threadIdx.x + l] = sph;
        }
      }
      __syncthreads();

      //////////////////////////////////////////////////////
      // Compute distances and store the maximum per thread
      // in registers (max_d).
      // Each thread computes upto ndpt distances.
      //////////////////////////////////////////////////////
      dist_t  max_d[NBPB] = {{ 0.0, 0, 0}};
      int16_t indices[ndpt * 2];

      for (uint8_t i = 0; i < ndpt * 2; i++)
      {
        indices[i] = locations_[(threadIdx.x) * 2 * ndpt + i];
      }

#pragma unroll

      for (uint8_t k = 0; k < ndpt; k++)
      {
        // We are iterating through ndpt pair of spheres across batch
        // if we increase ndpt, then we can compute for more spheres?
        int i = indices[k * 2];
        int j = indices[k * 2 + 1];

        if ((i == -1) || (j == -1))
          continue;

#pragma unroll

        for (uint16_t l = 0; l < nbpb; l++) // iterate through nbpb batches
        {
          float4 sph1 = __rs_shared[NBPB * i + l];
          float4 sph2 = __rs_shared[NBPB * j + l];

          //if ((sph1.w <= 0.0) || (sph2.w <= 0.0))
          //{
          //  continue;
          //}
          float r_diff =
            sph1.w + sph2.w; // sum of two radii, radii include respective offsets
          float d = sqrt((sph1.x - sph2.x) * (sph1.x - sph2.x) +
                         (sph1.y - sph2.y) * (sph1.y - sph2.y) +
                         (sph1.z - sph2.z) * (sph1.z - sph2.z));
          float f_diff = r_diff - d;

          if (f_diff > max_d[l].d)
          {
            max_d[l].d = f_diff;
            max_d[l].i = i;
            max_d[l].j = j;
          }
        }
      }

      // max_d has the result max for this thread

      //////////////////////////////////////////////////////
      // Reduce gridDim.x values using gridDim.x threads
      //////////////////////////////////////////////////////
      // We find the sum across 32 threads. Hence, we are limited to running all our self collision
      // distances for a batch_idx to 32 threads.

      __shared__ dist_t max_darr[32 * NBPB];

#pragma unroll

      for (uint8_t l = 0; l < nbpb; l++)
      {
        // Perform warp-wide reductions
        // Optimization: Skip the reduction if all the values are zero
        unsigned zero_mask = __ballot_sync(
          0xffffffff,
          max_d[l].d != 0.0); // we expect most values to be 0. So, zero_mask

        // should be 0 in the common case.
        if (zero_mask != 0)   // some of the values are non-zero
        {
          unsigned mask = __ballot_sync(0xffffffff, threadIdx.x < blockDim.x);

          if (threadIdx.x < blockDim.x)
          {
            // dist_t max_d = dist_sh[threadIdx.x];
            for (int offset = 16; offset > 0; offset /= 2)
            {
              // the offset here is linked to ndpt?
              uint64_t nd     = __shfl_down_sync(mask, *(uint64_t *)&max_d[l], offset);
              dist_t   d_temp = *(dist_t *)&nd;

              if (((threadIdx.x + offset) < blockDim.x)  && d_temp.d > max_d[l].d)
              {
                max_d[l] = d_temp;
              }
            }
          }
        }

        // thread0 in the warp has the max_d for the warp
        if (threadIdx.x % 32 == 0)
        {
          max_darr[(threadIdx.x / 32) + 32 * l] = max_d[l];

          // printf("threadIdx.x=%d, blockIdx.x=%d, max_d=%f\n", threadIdx.x,
          // blockIdx.x, max_d);
        }
      }

      if (threadIdx.x < nspheres)
      {
        for (int l = 0; l < nbpb; l++)
        {
          if (write_grad &&
              (sparse_index[(batch_idx + l) * nspheres + threadIdx.x] != 0))
          {
            *(float4 *)&out_vec[(batch_idx + l) * nspheres * 4 + threadIdx.x * 4] =
              make_float4(0.0);
            sparse_index[(batch_idx + l) * nspheres + threadIdx.x] = 0;
          }
        }
      }
      __syncthreads();

      if (threadIdx.x == 0)
      {
#pragma unroll

        for (uint8_t l = 0; l < nbpb; l++)
        {
          dist_t max_d = max_darr[l * 32];

          // TODO: This can be parallized
          for (int i = 1; i < (blockDim.x + 31) / 32; i++)
          {
            if (max_darr[l * 32 + i].d > max_d.d)
            {
              max_d = max_darr[l * 32 + i];
            }
          }

          //////////////////////////////////////////////////////
          // Write out the final results
          //////////////////////////////////////////////////////
          if (max_d.d != 0.0)
          {
            out_distance[batch_idx + l] = weight[0] * max_d.d;

            if (write_grad)
            {
              // NOTE: spheres can also be read from rs_shared
              float3 sph1 =
                *(float3 *)&robot_spheres[4 *
                                          ((batch_idx + l) * nspheres + max_d.i)];
              float3 sph2 =
                *(float3 *)&robot_spheres[4 *
                                          ((batch_idx + l) * nspheres + max_d.j)];
              float3 dist_vec = normalize(sph1 - sph2);// / max_d.d;

              *(float3 *)&out_vec[(batch_idx + l) * nspheres * 4 + max_d.i * 4] =
                weight[0] * -1 * dist_vec;
              *(float3 *)&out_vec[(batch_idx + l) * nspheres * 4 + max_d.j * 4] =
                weight[0] *  dist_vec;
              sparse_index[(batch_idx + l) * nspheres + max_d.i] = 1;
              sparse_index[(batch_idx + l) * nspheres + max_d.j] = 1;
            }
          }
          else
          {
            out_distance[batch_idx + l] = 0;
          }
        }
      }
    }
  } // namespace Geometry
}   // namespace Curobo

// This is the best version so far.
// It precomputes the start addresses per thread on the cpu.
// The rest is similar to the version above.
std::vector<torch::Tensor>self_collision_distance(
  torch::Tensor out_distance, torch::Tensor out_vec,
  torch::Tensor sparse_index,
  const torch::Tensor robot_spheres,    // batch_size x n_spheres x 3
  const torch::Tensor collision_offset, // n_spheres x n_spheres
  const torch::Tensor weight, const torch::Tensor collision_matrix,
  const torch::Tensor thread_locations, const int thread_locations_size,
  const int batch_size, const int nspheres, const bool compute_grad = false,
  const int ndpt                 = 8, // Does this need to match template?
  const bool experimental_kernel = false)
{
  using namespace Curobo::Geometry;

  // use efficient kernel based on number of threads:
  const int nbpb = 1;

  assert(nspheres < 1024);


  int threadsPerBlock = ((thread_locations_size / 2) + ndpt - 1) /
                        ndpt; // location_size must be an even number. We store

  // i,j for each sphere pair.
  // assert(threadsPerBlock/nbpb <=32);
  if (threadsPerBlock < 32 * nbpb)
  {
    threadsPerBlock = 32 * nbpb;
  }

  if (threadsPerBlock < nspheres)
  {
    threadsPerBlock = nspheres;
  }
  int blocksPerGrid   = (batch_size + nbpb - 1) / nbpb; // number of batches per block
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // ((threadsPerBlock >= nspheres))&&
  if (experimental_kernel)
  {
    int smemSize = nbpb * nspheres * sizeof(float4);


    if (ndpt == 1)
    {
      AT_DISPATCH_FLOATING_TYPES(
        robot_spheres.scalar_type(), "self_collision_distance", ([&] {
        self_collision_distance_kernel7<scalar_t, 1, nbpb>
          << < blocksPerGrid, threadsPerBlock, smemSize, stream >> > (
          out_distance.data_ptr<scalar_t>(),
          out_vec.data_ptr<scalar_t>(),
          sparse_index.data_ptr<uint8_t>(),
          robot_spheres.data_ptr<scalar_t>(),
          collision_offset.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          thread_locations.data_ptr<int16_t>(), batch_size, nspheres,
          compute_grad);
      }));
    }

    else if (ndpt == 2)
    {
      AT_DISPATCH_FLOATING_TYPES(
        robot_spheres.scalar_type(), "self_collision_distance", ([&] {
        self_collision_distance_kernel7<scalar_t, 2, nbpb>
          << < blocksPerGrid, threadsPerBlock, smemSize, stream >> > (
          out_distance.data_ptr<scalar_t>(),
          out_vec.data_ptr<scalar_t>(),
          sparse_index.data_ptr<uint8_t>(),
          robot_spheres.data_ptr<scalar_t>(),
          collision_offset.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          thread_locations.data_ptr<int16_t>(), batch_size, nspheres,
          compute_grad);
      }));
    }

    else if (ndpt == 4)
    {
      AT_DISPATCH_FLOATING_TYPES(
        robot_spheres.scalar_type(), "self_collision_distance", ([&] {
        self_collision_distance_kernel7<scalar_t, 4, nbpb>
          << < blocksPerGrid, threadsPerBlock, smemSize, stream >> > (
          out_distance.data_ptr<scalar_t>(),
          out_vec.data_ptr<scalar_t>(),
          sparse_index.data_ptr<uint8_t>(),
          robot_spheres.data_ptr<scalar_t>(),
          collision_offset.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          thread_locations.data_ptr<int16_t>(), batch_size, nspheres,
          compute_grad);
      }));
    }
    else if (ndpt == 8)
    {
      AT_DISPATCH_FLOATING_TYPES(
        robot_spheres.scalar_type(), "self_collision_distance", ([&] {
        self_collision_distance_kernel7<scalar_t, 8, nbpb>
          << < blocksPerGrid, threadsPerBlock, smemSize, stream >> > (
          out_distance.data_ptr<scalar_t>(),
          out_vec.data_ptr<scalar_t>(),
          sparse_index.data_ptr<uint8_t>(),
          robot_spheres.data_ptr<scalar_t>(),
          collision_offset.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          thread_locations.data_ptr<int16_t>(), batch_size, nspheres,
          compute_grad);
      }));
    }
    else if (ndpt == 32)
    {
      AT_DISPATCH_FLOATING_TYPES(
        robot_spheres.scalar_type(), "self_collision_distance", ([&] {
        self_collision_distance_kernel7<scalar_t, 32, nbpb>
          << < blocksPerGrid, threadsPerBlock, smemSize, stream >> > (
          out_distance.data_ptr<scalar_t>(),
          out_vec.data_ptr<scalar_t>(),
          sparse_index.data_ptr<uint8_t>(),
          robot_spheres.data_ptr<scalar_t>(),
          collision_offset.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          thread_locations.data_ptr<int16_t>(), batch_size, nspheres,
          compute_grad);
      }));
    }
    else if (ndpt == 64)
    {
      AT_DISPATCH_FLOATING_TYPES(
        robot_spheres.scalar_type(), "self_collision_distance", ([&] {
        self_collision_distance_kernel7<scalar_t, 64, nbpb>
          << < blocksPerGrid, threadsPerBlock, smemSize, stream >> > (
          out_distance.data_ptr<scalar_t>(),
          out_vec.data_ptr<scalar_t>(),
          sparse_index.data_ptr<uint8_t>(),
          robot_spheres.data_ptr<scalar_t>(),
          collision_offset.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          thread_locations.data_ptr<int16_t>(), batch_size, nspheres,
          compute_grad);
      }));
    }
    else if (ndpt == 128)
    {
      AT_DISPATCH_FLOATING_TYPES(
        robot_spheres.scalar_type(), "self_collision_distance", ([&] {
        self_collision_distance_kernel7<scalar_t, 128, nbpb>
          << < blocksPerGrid, threadsPerBlock, smemSize, stream >> > (
          out_distance.data_ptr<scalar_t>(),
          out_vec.data_ptr<scalar_t>(),
          sparse_index.data_ptr<uint8_t>(),
          robot_spheres.data_ptr<scalar_t>(),
          collision_offset.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          thread_locations.data_ptr<int16_t>(), batch_size, nspheres,
          compute_grad);
      }));
    }
    else if (ndpt == 512)
    {
      AT_DISPATCH_FLOATING_TYPES(
        robot_spheres.scalar_type(), "self_collision_distance", ([&] {
        self_collision_distance_kernel7<scalar_t, 512, nbpb>
          << < blocksPerGrid, threadsPerBlock, smemSize, stream >> > (
          out_distance.data_ptr<scalar_t>(),
          out_vec.data_ptr<scalar_t>(),
          sparse_index.data_ptr<uint8_t>(),
          robot_spheres.data_ptr<scalar_t>(),
          collision_offset.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          thread_locations.data_ptr<int16_t>(), batch_size, nspheres,
          compute_grad);
      }));
    }
    else
    {
      assert(false);
    }
  }

  else
  {
    int ndpt_n        = 32; // number of distances to be computed per thread
    int nwpr          = (nspheres + 31) / 32;
    int warpsPerBlock = nwpr * ((nspheres + ndpt_n - 1) / ndpt_n);
    threadsPerBlock = warpsPerBlock * 32;
    blocksPerGrid   = batch_size;

    assert(collision_matrix.size(0) == nspheres * nspheres);
    int smemSize = nspheres * sizeof(float4);

    if (nspheres < 1024 && threadsPerBlock < 1024)
    {

    AT_DISPATCH_FLOATING_TYPES(
      robot_spheres.scalar_type(), "self_collision_distance", ([&] {
      self_collision_distance_kernel4<scalar_t>
        << < blocksPerGrid, threadsPerBlock, smemSize, stream >> > (
        out_distance.data_ptr<scalar_t>(),
        out_vec.data_ptr<scalar_t>(),
        robot_spheres.data_ptr<scalar_t>(),
        collision_offset.data_ptr<scalar_t>(),
        collision_matrix.data_ptr<uint8_t>(), batch_size, nspheres,
        ndpt_n, nwpr, weight.data_ptr<scalar_t>(),
        sparse_index.data_ptr<uint8_t>(), compute_grad);
    }));
    }
    else
    {
      threadsPerBlock = batch_size;
      if (threadsPerBlock > 128)
      {
        threadsPerBlock = 128;
      }
      blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

      AT_DISPATCH_FLOATING_TYPES(
      robot_spheres.scalar_type(), "self_collision_distance", ([&] {
      self_collision_distance_kernel<scalar_t>
        << < blocksPerGrid, threadsPerBlock, smemSize, stream >> > (
        out_distance.data_ptr<scalar_t>(),
        out_vec.data_ptr<scalar_t>(),
        robot_spheres.data_ptr<scalar_t>(),
        collision_offset.data_ptr<scalar_t>(),
        collision_matrix.data_ptr<uint8_t>(),
        batch_size, nspheres,
        weight.data_ptr<scalar_t>(),
        compute_grad);
    }));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { out_distance, out_vec, sparse_index };
}
