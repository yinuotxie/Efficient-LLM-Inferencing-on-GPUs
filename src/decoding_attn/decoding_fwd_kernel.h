// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding fwd kernel

#pragma once

#include "decoding_attn/block_info.h"
#include "decoding_attn/decoding.h"
#include "decoding_attn/kernel_traits.h"

template <typename KernelTraits, bool IsAlibi>
__global__ void mha_decoding_fwd_kernel(const DecodingParams params)
{
    // set up the block info
    const DecodingBlockInfo binfo(params, blockIdx.x, blockIdx.y);

    if (binfo.actual_seqlen_q != 1 || binfo.actual_seqlen_k == 0)
    {
        return;
    }

    constexpr size_t head_dim = KernelTraits::head_dim;
    constexpr size_t threads_per_block = KernelTraits::threads_per_block;
    constexpr size_t group_size = KernelTraits::group_size;

    constexpr size_t warp_size = KernelTraits::warp_size;
    constexpr size_t warps_per_block = KernelTraits::warps_per_block;

    constexpr size_t groups_per_warp = KernelTraits::groups_per_warp;
    constexpr size_t groups_per_block = KernelTraits::groups_per_block;

    constexpr size_t thread_copy_elem_nums = KernelTraits::thread_copy_elem_nums;

    constexpr size_t thread_elem_nums = KernelTraits::thread_elem_nums;
    constexpr size_t thread_iters = KernelTraits::thread_iters;

    constexpr unsigned int shfl_mask = KernelTraits::shfl_mask;

    /*
    lane_id: Within a warp, each thread has a unique 'lane' ID, ranging from 0 to the warp size minus one (0-31 in most cases).

    lane_id is calculated using threadIdx.x % warp_size. This determines the thread's position within its warp. It's akin to an index for the thread within the warp.

    group_id and group_size: These terms relate to an additional level of division within a warp. Sometimes, it's useful to divide a warp into smaller groups for more fine-grained control over thread execution. This is especially true for complex operations like reductions or when dealing with data structures that benefit from certain memory access patterns.

    group_size is a predefined constant that divides each warp into smaller groups. The choice of group size can depend on the specific requirements of the kernel's computation.

    group_id is calculated using lane_id / group_size. This gives the ID of the group within the warp to which the thread belongs.

    group_lane_id is the thread's position within its group, calculated using lane_id \% group_size.
    */
    const size_t warp_id = threadIdx.x / warp_size;
    const size_t lane_id = threadIdx.x % warp_size;
    const size_t group_id = lane_id / group_size;
    const size_t group_lane_id = lane_id % group_size;

    // S = Q * K^T
    // Loads segments of Q matrics into local arrays
    // thread_elem_nums: number of groups for the head dimension
    // for example, if the head dimension is 128 and the group size is 8,
    // then thread_elem_nums is 16
    half RQ[thread_elem_nums];

#pragma unroll
    /*
    int4: This means that for each iteration of the loop, four integers from the query matrix are loaded at once into RQ. This is significantly more efficient than loading each integer one by one, especially in a parallel processing environment like a GPU where such optimizations can lead to substantial performance improvements.
    */
    for (size_t i = 0; i < thread_iters; ++i)
    {
        // thread_copy_elem_nums: number of elements to copy for each iteration (8)
        // each integer is 2 bytes, so for one copy, only 4 integers can be copied
        *(int4 *)(&RQ[i * thread_copy_elem_nums]) = *(int4 *)(&params.q_ptr[binfo.q_offset(
            params.q_row_stride, params.q_head_stride, (i * group_size + group_lane_id) * thread_copy_elem_nums)]);
    }

    extern __shared__ float S_smem[];
    float S_max = -std::numeric_limits<float>::max();

#pragma unroll
    for (size_t base_seqlen_k = warp_id * groups_per_warp; base_seqlen_k < binfo.actual_seqlen_k;
         base_seqlen_k += groups_per_block)
    {
        size_t seqlen_k = base_seqlen_k + group_id;
        half RK[thread_elem_nums];

        float tmp = 0.0;
        if (seqlen_k >= binfo.actual_seqlen_k)
        {
            memset(RK, 0, sizeof(RK));
        }
        else
        {
#pragma unroll
            // Load segments of K matrics into local arrays
            for (size_t i = 0; i < thread_iters; ++i)
            {
                *(int4 *)(&RK[i * thread_copy_elem_nums]) =
                    *(int4 *)(&params.k_ptr[binfo.k_offset(seqlen_k, params.k_row_stride, params.k_head_stride,
                                                           (i * group_size + group_lane_id) * thread_copy_elem_nums)]);
            }

#pragma unroll
            // Compute the dot product of Q and K
            for (size_t i = 0; i < thread_elem_nums; ++i)
            {
                tmp += (__half2float(RQ[i]) * __half2float(RK[i]));
            }
        }

#pragma unroll
        // Intra-Warp Reduction: sum up the dot product of Q and K
        for (size_t i = group_size / 2; i >= 1; i /= 2)
        {
            tmp += __shfl_xor_sync(shfl_mask, tmp, i);
        }

        if (group_lane_id == 0 && seqlen_k < binfo.actual_seqlen_k)
        {
            tmp *= params.scale_softmax;

            if (IsAlibi)
            {
                tmp += (binfo.h_slope * (static_cast<int>(seqlen_k) - binfo.actual_seqlen_q - binfo.row_shift));
            }

            // Store the dot product of Q and K into shared memory
            S_smem[seqlen_k] = tmp;
            S_max = fmaxf(tmp, S_max);
        }
    }

    // P = Softmax(S)
    __shared__ float softmax_smem[warps_per_block];

#pragma unroll
    for (size_t i = warp_size / 2; i >= 1; i /= 2)
    {
        S_max = fmaxf(S_max, __shfl_xor_sync(shfl_mask, S_max, i));
    }

    // Find the max value of the dot product of Q[i] and K[i], where i is the warp id
    if (lane_id == 0)
    {
        softmax_smem[warp_id] = S_max;
    }

    __syncthreads();

    if (lane_id < warps_per_block)
    {
        S_max = softmax_smem[lane_id];
    }
    else
    {
        S_max = -std::numeric_limits<float>::max();
    }

#pragma unroll
    // Find the max value for all the warps (across warps)
    for (size_t i = warps_per_block / 2; i >= 1; i /= 2)
    {
        S_max = fmaxf(S_max, __shfl_xor_sync(shfl_mask, S_max, i));
    }

    S_max = __shfl_sync(shfl_mask, S_max, 0);

    float exp_sum = 0.0;
#pragma unroll
    for (size_t seqlen_k = threadIdx.x; seqlen_k < binfo.actual_seqlen_k; seqlen_k += threads_per_block)
    {
        // Softmax(S) = exp(S - max(S)) / sum(exp(S - max(S)))
        S_smem[seqlen_k] -= S_max;
        S_smem[seqlen_k] = exp(S_smem[seqlen_k]);
        exp_sum += S_smem[seqlen_k];
    }

#pragma unroll
    // Intra-Warp Reduction: sum up the exp(S - max(S))
    for (size_t i = warp_size / 2; i >= 1; i /= 2)
    {
        exp_sum += __shfl_xor_sync(shfl_mask, exp_sum, i);
    }

    if (lane_id == 0)
    {
        softmax_smem[warp_id] = exp_sum;
    }

    __syncthreads();

    if (lane_id < warps_per_block)
    {
        exp_sum = softmax_smem[lane_id];
    }

#pragma unroll
    // Across_Warp Reduction: sum up the sum(exp(S - max(S)))
    for (size_t i = warps_per_block / 2; i >= 1; i /= 2)
    {
        exp_sum += __shfl_xor_sync(shfl_mask, exp_sum, i);
    }
    exp_sum = __shfl_sync(shfl_mask, exp_sum, 0);

#pragma unroll
    // Normalization
    for (size_t seqlen_k = threadIdx.x; seqlen_k < binfo.actual_seqlen_k; seqlen_k += threads_per_block)
    {
        S_smem[seqlen_k] /= exp_sum;
    }

    __syncthreads();

    // O = P * V
    half RV[thread_elem_nums];
    float RO[thread_elem_nums];

    memset(RO, 0, sizeof(RO));

#pragma unroll
    for (size_t base_seqlen_k = warp_id * groups_per_warp; base_seqlen_k < binfo.actual_seqlen_k;
         base_seqlen_k += groups_per_block)
    {
        size_t seqlen_k = base_seqlen_k + group_id;

        if (seqlen_k < binfo.actual_seqlen_k)
        {
#pragma unroll
            // Load segments of V matrics into local arrays
            for (size_t i = 0; i < thread_iters; ++i)
            {
                *(int4 *)(&RV[i * thread_copy_elem_nums]) =
                    *(int4 *)(&params.v_ptr[binfo.k_offset(seqlen_k, params.v_row_stride, params.v_head_stride,
                                                           (i * group_size + group_lane_id) * thread_copy_elem_nums)]);
            }

#pragma unroll
            // Compute the dot product of P and V
            for (size_t i = 0; i < thread_elem_nums; ++i)
            {
                RO[i] += (S_smem[seqlen_k] * __half2float(RV[i]));
            }
        }
    }

#pragma unroll
    for (size_t i = 0; i < thread_elem_nums; ++i)
    {
#pragma unroll
        for (size_t j = group_size; j <= warp_size / 2; j *= 2)
        {
            RO[i] += __shfl_xor_sync(shfl_mask, RO[i], j);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = threadIdx.x; i < head_dim; i += threads_per_block)
    {
        S_smem[i] = 0.0;
    }

    __syncthreads();

    if (lane_id < group_size)
    {
#pragma unroll
        for (size_t i = 0; i < thread_iters; ++i)
        {
#pragma unroll
            for (size_t j = 0; j < thread_copy_elem_nums; ++j)
            {
                atomicAdd(S_smem + (i * group_size + lane_id) * thread_copy_elem_nums + j,
                          RO[i * thread_copy_elem_nums + j]);
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = threadIdx.x; i < head_dim; i += threads_per_block)
    {
        params.o_ptr[binfo.q_offset(params.o_row_stride, params.o_head_stride, i)] = __float2half(S_smem[i]);
    }
}
