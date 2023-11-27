#pragma once

#include "decoding_attn_new/block_info.h"
#include "decoding_attn_new/decoding.h"
#include "decoding_attn_new/kernel_traits.h"

/**
 * @namespace flash
 * @brief Namespace containing functions and templates related to the FlashAttention algorithm.
 */
namespace flash
{
    /**
     * @brief Kernel for the FlashAttention algorithm.
     * @tparam KernelTraits Traits for the kernel.
     * @param params Parameters for the kernel.
     * @param m_block The current block index.
     */
    template <typename KernelTraits, bool is_even_MN>
    inline __device__ void decoding_kernel(const DecodingParams &params, const int m_block)
    {
        using Element = typename KernelTraits::Element;
        const int thid = threadIdx.x;
        constexpr int kBlockM = KernelTraits::kBlockM;
        constexpr int kBlockN = KernelTraits::kBlockN;
        constexpr int kHeadDim = KernelTraits::kHeadDim;
        constexpr int kNWarps = KernelTraits::kNWarpsPerBlock;
        constexpr int kNThreadsPerBlock = KernelTraits::kNThreadsPerBlock;
        constexpr int group_size = KernelTraits::group_size;
        constexpr int groups_per_warp = KernelTraits::groups_per_warp;
        constexpr int threadNElemPerLoad = KernelTraits::threadNElemPerLoad;
        constexpr int threadIters = KernelTraits::threadIters;
        constexpr unsigned int shfl_mask = KernelTraits::shfl_mask;

        // Block information calculation for the current block
        const BlockInfo<is_even_MN> binfo(params, blockIdx.y, blockIdx.z);

        // Early exit if block is out of range
        if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0)
        {
            return;
        }

        // Warp and lane identification for cooperative thread execution
        const int warp_size = 32;
        const size_t warp_id = thid / warp_size;
        const size_t lane_id = thid % warp_size;
        const size_t group_id = lane_id / group_size;
        const size_t group_lane_id = lane_id % group_size;

        // Array for storing Q matrix data in half-precision
        half RQ[kBlockN];

        // Load the data from the Q_ptr array into the RQ array
#pragma unroll
        for (size_t i = 0; i < threadIters; ++i)
        {
            size_t RQ_idx = i * threadNElemPerLoad;
            int4 *RQ_ptr = reinterpret_cast<int4 *>(&RQ[RQ_idx]);

            size_t Q_idx = binfo.q_offset(params.q_row_stride, params.q_head_stride, (i * group_size + group_lane_id) * threadNElemPerLoad);
            const int4 *Q_ptr = reinterpret_cast<const int4 *>(&params.q_ptr[Q_idx]);

            *RQ_ptr = *Q_ptr;
        }

        // Shared memory for storing intermediate results
        extern __shared__ float S_smem[];
        float S_max = -std::numeric_limits<float>::max();

        // Main computation loop for attention mechanism
#pragma unroll
        for (size_t base_seqlen_k = warp_id * groups_per_warp; base_seqlen_k < binfo.actual_seqlen_k; base_seqlen_k += groups_per_warp * warp_size)
        {
            size_t seqlen_k = base_seqlen_k + group_id;
            half RK[kBlockN];

            float tmp = 0.0f;
            if (seqlen_k < binfo.actual_seqlen_k)
            {
#pragma unroll
                for (size_t i = 0; i < threadIters; ++i)
                {
                    size_t RK_idx = i * threadNElemPerLoad;
                    int4 *RK_ptr = reinterpret_cast<int4 *>(&RK[RK_idx]);

                    size_t K_idx = binfo.k_offset(seqlen_k, params.k_row_stride, params.k_head_stride, (i * group_size + group_lane_id) * threadNElemPerLoad);
                    const int4 *K_ptr = reinterpret_cast<const int4 *>(&params.k_ptr[K_idx]);

                    *RK_ptr = *K_ptr;
                }

                // Dot product calculation
#pragma unroll
                for (size_t i = 0; i < kBlockN; ++i)
                {
                    tmp += (__half2float(RQ[i]) * __half2float(RK[i]));
                }
            }

            // Warp-level reduction to find maximum
#pragma unroll
            for (size_t i = group_size / 2; i > 0; i /= 2)
            {
                tmp = max(tmp, __shfl_xor_sync(0xFFFFFFFF, tmp, i));
            }

            if (group_lane_id == 0 && seqlen_k < binfo.actual_seqlen_k)
            {
                tmp *= params.scale_softmax;
                S_smem[seqlen_k] = tmp;
                S_max = fmaxf(S_max, tmp);
            }
        }

        // Block-level reduction to find maximum
        __shared__ float softmax_smem[kNWarps];
#pragma unroll
        for (size_t i = warp_size / 2; i > 0; i /= 2)
        {
            S_max = fmaxf(S_max, __shfl_xor_sync(0xFFFFFFFF, S_max, i));
        }

        if (lane_id == 0)
        {
            softmax_smem[warp_id] = S_max;
        }

        __syncthreads();

        if (lane_id < kNWarps)
        {
            S_max = softmax_smem[lane_id];
        }
        else
        {
            S_max = -std::numeric_limits<float>::max();
        }

        // Further reduction to find the global maximum
#pragma unroll
        for (size_t i = kNWarps / 2; i > 0; i /= 2)
        {
            S_max = fmaxf(S_max, __shfl_xor_sync(0xFFFFFFFF, S_max, i));
        }

        S_max = __shfl_sync(0xFFFFFFFF, S_max, 0);

        // Calculation of exponential sum for softmax
        float exp_sum = 0.0f;
#pragma unroll
        for (size_t seqlen_k = thid; seqlen_k < binfo.actual_seqlen_k; seqlen_k += blockDim.x)
        {
            float val = expf(S_smem[seqlen_k] - S_max);
            S_smem[seqlen_k] = val;
            exp_sum += val;
        }

        // Reduction to calculate the sum of exponentials
#pragma unroll
        for (size_t i = warp_size / 2; i > 0; i /= 2)
        {
            exp_sum = __shfl_xor_sync(0xFFFFFFFF, exp_sum, i);
        }

        if (lane_id == 0)
        {
            softmax_smem[warp_id] = exp_sum;
        }

        __syncthreads();

        if (lane_id < kNWarps)
        {
            exp_sum = softmax_smem[lane_id];
        }
        else
        {
            exp_sum = 0.0f;
        }

#pragma unroll
        for (size_t i = kNWarps / 2; i > 0; i /= 2)
        {
            exp_sum += __shfl_xor_sync(0xFFFFFFFF, exp_sum, i);
        }
        exp_sum = __shfl_sync(0xFFFFFFFF, exp_sum, 0);

        // Normalization step of softmax
#pragma unroll
        for (size_t seqlen_k = thid; seqlen_k < binfo.actual_seqlen_k; seqlen_k += blockDim.x)
        {
            S_smem[seqlen_k] /= exp_sum;
        }

        __syncthreads();

        // Computation for the output values
        half RV[kBlockN];
        float RO[kBlockN];
        memset(RO, 0, sizeof(RO));

#pragma unroll
        for (size_t base_seqlen_k = warp_id * groups_per_warp; base_seqlen_k < binfo.actual_seqlen_k; base_seqlen_k += groups_per_warp * warp_size)
        {
            size_t seqlen_k = base_seqlen_k + group_id;
            if (seqlen_k < binfo.actual_seqlen_k)
            {
#pragma unroll
                for (size_t i = 0; i < threadIters; ++i)
                {
                    size_t RV_idx = i * threadNElemPerLoad;
                    int4 *RV_ptr = reinterpret_cast<int4 *>(&RV[RV_idx]);

                    size_t V_idx = binfo.k_offset(seqlen_k, params.v_row_stride, params.v_head_stride, (i * group_size + group_lane_id) * threadNElemPerLoad);
                    const int4 *V_ptr = reinterpret_cast<const int4 *>(&params.v_ptr[V_idx]);

                    *RV_ptr = *V_ptr;
                }
            }

            // Dot product calculation for P and V
#pragma unroll
            for (size_t i = 0; i < kBlockN; ++i)
            {
                RO[i] += S_smem[seqlen_k] * __half2float(RV[i]);
            }
        }

#pragma unroll
        for (size_t i = 0; i < kBlockN; ++i)
        {
#pragma unroll
            for (size_t j = group_size; j <= warp_size / 2; j *= 2)
            {
                RO[i] += __shfl_xor_sync(shfl_mask, RO[i], j);
            }
        }

        __syncthreads();

#pragma unroll
        for (size_t i = thid; i < kHeadDim; i += kNThreadsPerBlock)
        {
            S_smem[i] = 0.0f;
        }

        __syncthreads();

        if (lane_id < group_size)
        {
#pragma unroll
            for (size_t i = 0; i < threadIters; ++i)
            {
#pragma unroll
                for (size_t j = 0; j < threadNElemPerLoad; ++j)
                {
                    // Calculate the index for the shared memory array (S_smem)
                    size_t smem_index = (i * group_size + lane_id) * threadNElemPerLoad + j;

                    // Calculate the index for the RO array
                    size_t RO_index = i * threadNElemPerLoad + j;

                    // Perform the atomic addition
                    // This adds the value from the RO array to the shared memory (S_smem) at the calculated index.
                    atomicAdd(&S_smem[smem_index], RO[RO_index]);
                }
            }
        }

        __syncthreads();

        for (size_t i = thid; i < kHeadDim; i += kNThreadsPerBlock)
        {
            params.o_ptr[binfo.q_offset(params.o_row_stride, params.o_head_stride, i)] = __float2half(S_smem[i]);
        }
    }

    /**
     * @brief Function to call the decoding kernel.
     * @tparam Kernel_traits Traits for the kernel.
     * @tparam Is_even_MN Boolean value indicating whether the block size is even or odd.
     * @param params Parameters for the kernel.
     */
    template <typename Kernel_traits, bool Is_even_MN>
    inline __device__ void compute_attn(const DecodingParams &params)
    {
        const int m_block = blockIdx.x;

        flash::decoding_kernel<Kernel_traits, Is_even_MN>(params, m_block);
    }
} // namespace flash