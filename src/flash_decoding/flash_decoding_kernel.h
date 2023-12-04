#pragma once

#include "flash_decoding/block_info.h"
#include "flash_decoding/flash_decoding.h"
#include "flash_decoding/kernel_traits.h"

/**
 * @brief Kernel for the FlashAttention algorithm.
 * @tparam KernelTraits Traits for the kernel.
 * @param params Parameters for the kernel.
 */
template <typename KernelTraits>
__global__ void flashDecoding(const FlashDecodingParams &params)
{
    using Element = typename KernelTraits::Element;
    const int thid = threadIdx.x;

    constexpr int head_dim = KernelTraits::HeadDim;
    constexpr int threads_per_block = KernelTraits::numThreadsPerBlock;
    constexpr int group_size = KernelTraits::numThreadsPerGroup;
    constexpr int groups_per_warp = KernelTraits::numGroupsPerWarp;
    constexpr int groups_per_block = KernelTraits::numGroupsPerBlock;
    constexpr int warps_per_block = KernelTraits::numWarpsPerBlock;

    constexpr int numElemPerLoad = KernelTraits::numElemPerLoad;
    constexpr int numElemPerGroup = KernelTraits::numElemPerGroup;
    constexpr int groupIters = KernelTraits::groupIters;
    constexpr unsigned int shfl_mask = KernelTraits::shfl_mask;

    // Block information calculation for the current block
    const BlockInfo binfo(params, blockIdx.x, blockIdx.y);

    // Early exit if block is out of range
    if (binfo.actual_seqlen_k == 0)
    {
        return;
    }

    // Warp and lane identification for cooperative thread execution
    const int warp_size = KernelTraits::warp_size;
    const size_t warp_id = thid / warp_size;
    const size_t lane_id = thid % warp_size;
    const size_t group_id = lane_id / group_size;
    const size_t group_lane_id = lane_id % group_size;

    // Array for storing Q matrix data in half-precision
    half RQ[numElemPerGroup];

    // Load the data from the Q_ptr array into the RQ array
#pragma unroll
    for (size_t i = 0; i < groupIters; ++i)
    {
        size_t RQ_idx = i * numElemPerLoad;
        int4 *RQ_ptr = reinterpret_cast<int4 *>(&RQ[RQ_idx]);

        size_t Q_idx = binfo.q_offset(params.q_row_stride, params.q_head_stride, (i * group_size + group_lane_id) * numElemPerLoad);
        const int4 *Q_ptr = reinterpret_cast<const int4 *>(&params.q_ptr[Q_idx]);

        *RQ_ptr = *Q_ptr;
    }

    // Shared memory for storing intermediate results
    extern __shared__ float S_smem[];
    float S_max = -std::numeric_limits<float>::max();

    // Main computation loop for attention mechanism
#pragma unroll
    for (size_t base_seqlen_k = warp_id * groups_per_warp; base_seqlen_k < binfo.actual_seqlen_k; base_seqlen_k += groups_per_block)
    {
        size_t seqlen_k = base_seqlen_k + group_id;
        half RK[numElemPerGroup];

        float tmp = 0.0f;
        if (seqlen_k >= binfo.actual_seqlen_k)
        {
            memset(RK, 0, sizeof(RK));
        }
        else
        {
#pragma unroll
            for (size_t i = 0; i < groupIters; ++i)
            {
                size_t RK_idx = i * numElemPerLoad;
                int4 *RK_ptr = reinterpret_cast<int4 *>(&RK[RK_idx]);

                size_t K_idx = binfo.k_offset(seqlen_k, params.k_row_stride, params.k_head_stride, (i * group_size + group_lane_id) * numElemPerLoad);
                const int4 *K_ptr = reinterpret_cast<const int4 *>(&params.k_ptr[K_idx]);

                *RK_ptr = *K_ptr;
            }

            // Dot product calculation
#pragma unroll
            for (size_t i = 0; i < numElemPerGroup; ++i)
            {
                tmp += (__half2float(RQ[i]) * __half2float(RK[i]));
            }
        }

        // Intro-row reduction
        // Each row of the Q, K multiplied matrices is handles by a group of threads
        // For example, if the group size is 8, then the first 8 threads will handle the first row of the matrices
        // Here, we perform a reduction within the group to find the sum of the products of the Q and K matrices of each row
#pragma unroll
        for (size_t i = group_size / 2; i > 0; i /= 2)
        {
            tmp += __shfl_xor_sync(shfl_mask, tmp, i);
        }

        if (group_lane_id == 0 && seqlen_k < binfo.actual_seqlen_k)
        {
            tmp *= params.scale_softmax;
            // Store the result in the shared memory array (S_smem)
            S_smem[seqlen_k] = tmp;
            // Find the maximum value in the shared memory array (S_smem)
            S_max = fmaxf(S_max, tmp);
        }
    }

    // Warp-Level reduction to find maximum
    // Find the max across all the threads in the warp
    __shared__ float softmax_smem[warps_per_block];
#pragma unroll
    for (size_t i = warp_size / 2; i > 0; i /= 2)
    {
        S_max = fmaxf(S_max, __shfl_xor_sync(shfl_mask, S_max, i));
    }

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

    // Find the max value across the warps in the block
#pragma unroll
    for (size_t i = warps_per_block / 2; i > 0; i /= 2)
    {
        S_max = fmaxf(S_max, __shfl_xor_sync(shfl_mask, S_max, i));
    }

    S_max = __shfl_sync(shfl_mask, S_max, 0);

    // Calculation of exponential sum for softmax
    float exp_sum = 0.0f;
#pragma unroll
    for (size_t seqlen_k = thid; seqlen_k < binfo.actual_seqlen_k; seqlen_k += threads_per_block)
    {
        float val = expf(S_smem[seqlen_k] - S_max);
        S_smem[seqlen_k] = val;
        exp_sum += val;
    }

    // Reduction to calculate the sum of exponentials
#pragma unroll
    for (size_t i = warp_size / 2; i > 0; i /= 2)
    {
        exp_sum = __shfl_xor_sync(shfl_mask, exp_sum, i);
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
    for (size_t i = warps_per_block / 2; i > 0; i /= 2)
    {
        exp_sum += __shfl_xor_sync(shfl_mask, exp_sum, i);
    }
    exp_sum = __shfl_sync(shfl_mask, exp_sum, 0);

    // Normalization step of softmax
#pragma unroll
    for (size_t seqlen_k = thid; seqlen_k < binfo.actual_seqlen_k; seqlen_k += threads_per_block)
    {
        S_smem[seqlen_k] /= exp_sum;
    }

    __syncthreads();

    // Computation for the output values
    half RV[numElemPerGroup];
    float RO[numElemPerGroup];
    memset(RO, 0, sizeof(RO));

#pragma unroll
    for (size_t base_seqlen_k = warp_id * groups_per_warp; base_seqlen_k < binfo.actual_seqlen_k; base_seqlen_k += groups_per_block)
    {
        size_t seqlen_k = base_seqlen_k + group_id;
        if (seqlen_k < binfo.actual_seqlen_k)
        {
#pragma unroll
            for (size_t i = 0; i < groupIters; ++i)
            {
                size_t RV_idx = i * numElemPerLoad;
                int4 *RV_ptr = reinterpret_cast<int4 *>(&RV[RV_idx]);

                size_t V_idx = binfo.k_offset(seqlen_k, params.v_row_stride, params.v_head_stride, (i * group_size + group_lane_id) * numElemPerLoad);
                const int4 *V_ptr = reinterpret_cast<const int4 *>(&params.v_ptr[V_idx]);

                *RV_ptr = *V_ptr;
            }
        }

        // Dot product calculation for P and V
#pragma unroll
        for (size_t i = 0; i < numElemPerGroup; ++i)
        {
            RO[i] += S_smem[seqlen_k] * __half2float(RV[i]);
        }
    }

#pragma unroll
    for (size_t i = 0; i < numElemPerGroup; ++i)
    {
#pragma unroll
        for (size_t j = group_size; j <= warp_size / 2; j *= 2)
        {
            RO[i] += __shfl_xor_sync(shfl_mask, RO[i], j);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = thid; i < head_dim; i += threads_per_block)
    {
        S_smem[i] = 0.0f;
    }

    __syncthreads();

    if (lane_id < group_size)
    {
#pragma unroll
        for (size_t i = 0; i < groupIters; ++i)
        {
#pragma unroll
            for (size_t j = 0; j < numElemPerLoad; ++j)
            {
                // Calculate the index for the shared memory array (S_smem)
                size_t smem_index = (i * group_size + lane_id) * numElemPerLoad + j;

                // Calculate the index for the RO array
                size_t RO_index = i * numElemPerLoad + j;

                // Perform the atomic addition
                // This adds the value from the RO array to the shared memory (S_smem) at the calculated index.
                atomicAdd(&S_smem[smem_index], RO[RO_index]);
            }
        }
    }

    __syncthreads();

    for (size_t i = thid; i < head_dim; i += threads_per_block)
    {
        params.o_ptr[binfo.q_offset(params.o_row_stride, params.o_head_stride, i)] = __float2half(S_smem[i]);
    }
}