// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:33:18 on Tue, Nov 07, 2023
//
// Description: decoding int8 fwd kernel

#pragma once

#include "decoding_attn_int8/block_info.h"
#include "decoding_attn_int8/decoding_int8.h"
#include "decoding_attn_int8/kernel_traits.h"

template <typename KernelTraits>
__global__ void quantization_int8_kernel(const DecodingInt8Params params) {
    const QuantizatioInt8BlockInfo binfo(params, blockIdx.x, blockIdx.y);
    if (binfo.s_k >= binfo.actual_seqlen_k) {
        return;
    }

    constexpr size_t group_size = KernelTraits::group_size;

    constexpr size_t warp_size = KernelTraits::warp_size;

    constexpr size_t groups_per_warp = KernelTraits::groups_per_warp;
    constexpr size_t groups_per_block = KernelTraits::groups_per_block;

    constexpr size_t thread_copy_elem_nums = KernelTraits::thread_copy_elem_nums;

    constexpr size_t thread_elem_nums = KernelTraits::thread_elem_nums;
    constexpr size_t thread_iters = KernelTraits::thread_iters;

    constexpr unsigned int shfl_mask = KernelTraits::shfl_mask;

    const half max_int8 = 127.0;
    const half min_scale = 1e-5;

    const size_t warp_id = threadIdx.x / warp_size;
    const size_t lane_id = threadIdx.x % warp_size;
    const size_t group_id = lane_id / group_size;
    const size_t group_lane_id = lane_id % group_size;

#pragma unroll
    for (size_t base_h_idx = warp_id * groups_per_warp; base_h_idx < params.h_k; base_h_idx += groups_per_block) {
        size_t h_idx = base_h_idx + group_id;
        half RK[thread_elem_nums];
        half RV[thread_elem_nums];

        int8_t RQK[thread_elem_nums];
        int8_t RQV[thread_elem_nums];

        half k_scale = 0.0;
        half v_scale = 0.0;

        if (h_idx < params.h_k) {
#pragma unroll
            for (size_t i = 0; i < thread_iters; ++i) {
                *(int4 *)(&RK[i * thread_copy_elem_nums]) =
                    *(int4 *)(&params.k_ptr[binfo.k_offset(params.k_row_stride, h_idx, params.k_head_stride,
                                                           (i * group_size + group_lane_id) * thread_copy_elem_nums)]);
                *(int4 *)(&RV[i * thread_copy_elem_nums]) =
                    *(int4 *)(&params.v_ptr[binfo.k_offset(params.v_row_stride, h_idx, params.v_head_stride,
                                                           (i * group_size + group_lane_id) * thread_copy_elem_nums)]);
            }

#pragma unroll
            for (size_t i = 0; i < thread_elem_nums; ++i) {
                k_scale = (k_scale > __habs(RK[i])) ? k_scale : __habs(RK[i]);
                v_scale = (v_scale > __habs(RV[i])) ? v_scale : __habs(RV[i]);
            }
        }

#pragma unroll
        for (size_t i = group_size / 2; i >= 1; i /= 2) {
            k_scale = fmaxf(k_scale, __shfl_xor_sync(shfl_mask, k_scale, i));
            v_scale = fmaxf(v_scale, __shfl_xor_sync(shfl_mask, v_scale, i));
        }

        if (h_idx < params.h_k) {
            k_scale /= max_int8;
            v_scale /= max_int8;
            k_scale = (k_scale > min_scale) ? k_scale : min_scale;
            v_scale = (v_scale > min_scale) ? v_scale : min_scale;

#pragma unroll
            for (size_t i = 0; i < thread_elem_nums; ++i) {
                RQK[i] = static_cast<int8_t>(__half2short_rn(RK[i] / k_scale));
                RQV[i] = static_cast<int8_t>(__half2short_rn(RV[i] / v_scale));
            }

#pragma unroll
            for (size_t i = 0; i < thread_iters; ++i) {
                *(int2 *)(&params
                               .k_int8_ptr[binfo.k_offset(params.k_row_stride, h_idx, params.k_head_stride,
                                                          (i * group_size + group_lane_id) * thread_copy_elem_nums)]) =
                    *(int2 *)(&RQK[i * thread_copy_elem_nums]);
                *(int2 *)(&params
                               .v_int8_ptr[binfo.k_offset(params.v_row_stride, h_idx, params.v_head_stride,
                                                          (i * group_size + group_lane_id) * thread_copy_elem_nums)]) =
                    *(int2 *)(&RQV[i * thread_copy_elem_nums]);
            }

            params.k_scale_ptr[binfo.k_scale_offset(params.h_k, h_idx)] = k_scale;
            params.v_scale_ptr[binfo.k_scale_offset(params.h_k, h_idx)] = v_scale;
        }
    }
}

template <typename KernelTraits, bool IsAlibi>
__global__ void mha_decoding_int8_fwd_kernel(const DecodingInt8Params params) {
    const DecodingInt8BlockInfo binfo(params, blockIdx.x, blockIdx.y);
    if (binfo.actual_seqlen_q != 1 || binfo.actual_seqlen_k == 0) {
        return;
    }

    constexpr size_t head_dim = KernelTraits::head_dim;
    constexpr size_t threads_per_block = KernelTraits::threads_per_block;
    constexpr size_t group_size = KernelTraits::group_size;

    constexpr size_t warp_size = KernelTraits::warp_size;
    constexpr size_t warps_per_block = KernelTraits::warps_per_block;

    constexpr size_t groups_per_warp = KernelTraits::groups_per_warp;
    constexpr size_t groups_per_block = KernelTraits::groups_per_block;

    constexpr size_t thread_copy_q_elem_nums = KernelTraits::thread_copy_q_elem_nums;
    constexpr size_t thread_copy_k_elem_nums = KernelTraits::thread_copy_k_elem_nums;

    constexpr size_t thread_elem_nums = KernelTraits::thread_elem_nums;
    constexpr size_t thread_q_iters = KernelTraits::thread_q_iters;
    constexpr size_t thread_k_iters = KernelTraits::thread_k_iters;

    constexpr unsigned int shfl_mask = KernelTraits::shfl_mask;

    const size_t warp_id = threadIdx.x / warp_size;
    const size_t lane_id = threadIdx.x % warp_size;
    const size_t group_id = lane_id / group_size;
    const size_t group_lane_id = lane_id % group_size;

    // S = Q * K^T
    half RQ[thread_elem_nums];

#pragma unroll
    for (size_t i = 0; i < thread_q_iters; ++i) {
        *(int4 *)(&RQ[i * thread_copy_q_elem_nums]) = *(int4 *)(&params.q_ptr[binfo.q_offset(
            params.q_row_stride, params.q_head_stride, (i * group_size + group_lane_id) * thread_copy_q_elem_nums)]);
    }

    extern __shared__ float S_smem[];
    float S_max = -std::numeric_limits<float>::max();

#pragma unroll
    for (size_t base_seqlen_k = warp_id * groups_per_warp; base_seqlen_k < binfo.actual_seqlen_k;
         base_seqlen_k += groups_per_block) {
        size_t seqlen_k = base_seqlen_k + group_id;
        int8_t RQK[thread_elem_nums];
        float RK_scale = 0.0;

        float tmp = 0.0;
        if (seqlen_k >= binfo.actual_seqlen_k) {
            memset(RQK, 0, sizeof(RQK));
        } else {
#pragma unroll
            for (size_t i = 0; i < thread_k_iters; ++i) {
                *(int4 *)(&RQK[i * thread_copy_k_elem_nums]) = *(int4 *)(&params.k_int8_ptr[binfo.k_offset(
                    seqlen_k, params.k_row_stride, params.k_head_stride,
                    (i * group_size + group_lane_id) * thread_copy_k_elem_nums)]);
            }

            RK_scale = __half2float(params.k_scale_ptr[binfo.k_scale_offset(seqlen_k, params.h_k)]);

#pragma unroll
            for (size_t i = 0; i < thread_elem_nums; ++i) {
                tmp += (__half2float(RQ[i]) * (static_cast<float>(RQK[i]) * RK_scale));
            }
        }

#pragma unroll
        for (size_t i = group_size / 2; i >= 1; i /= 2) {
            tmp += __shfl_xor_sync(shfl_mask, tmp, i);
        }

        if (group_lane_id == 0 && seqlen_k < binfo.actual_seqlen_k) {
            tmp *= params.scale_softmax;

            if (IsAlibi) {
                tmp += (binfo.h_slope * (static_cast<int>(seqlen_k) - binfo.actual_seqlen_q - binfo.row_shift));
            }

            S_smem[seqlen_k] = tmp;
            S_max = fmaxf(tmp, S_max);
        }
    }

    // P = Softmax(S)
    __shared__ float softmax_smem[warps_per_block];

#pragma unroll
    for (size_t i = warp_size / 2; i >= 1; i /= 2) {
        S_max = fmaxf(S_max, __shfl_xor_sync(shfl_mask, S_max, i));
    }

    if (lane_id == 0) {
        softmax_smem[warp_id] = S_max;
    }

    __syncthreads();

    if (lane_id < warps_per_block) {
        S_max = softmax_smem[lane_id];
    } else {
        S_max = -std::numeric_limits<float>::max();
    }

#pragma unroll
    for (size_t i = warps_per_block / 2; i >= 1; i /= 2) {
        S_max = fmaxf(S_max, __shfl_xor_sync(shfl_mask, S_max, i));
    }

    S_max = __shfl_sync(shfl_mask, S_max, 0);

    float exp_sum = 0.0;
#pragma unroll
    for (size_t seqlen_k = threadIdx.x; seqlen_k < binfo.actual_seqlen_k; seqlen_k += threads_per_block) {
        S_smem[seqlen_k] -= S_max;
        S_smem[seqlen_k] = exp(S_smem[seqlen_k]);
        exp_sum += S_smem[seqlen_k];
    }

#pragma unroll
    for (size_t i = warp_size / 2; i >= 1; i /= 2) {
        exp_sum += __shfl_xor_sync(shfl_mask, exp_sum, i);
    }

    if (lane_id == 0) {
        softmax_smem[warp_id] = exp_sum;
    }

    __syncthreads();

    if (lane_id < warps_per_block) {
        exp_sum = softmax_smem[lane_id];
    }

#pragma unroll
    for (size_t i = warps_per_block / 2; i >= 1; i /= 2) {
        exp_sum += __shfl_xor_sync(shfl_mask, exp_sum, i);
    }
    exp_sum = __shfl_sync(shfl_mask, exp_sum, 0);

#pragma unroll
    for (size_t seqlen_k = threadIdx.x; seqlen_k < binfo.actual_seqlen_k; seqlen_k += threads_per_block) {
        S_smem[seqlen_k] /= exp_sum;
    }

    __syncthreads();

    // O = P * V
    int8_t RQV[thread_elem_nums];
    float RO[thread_elem_nums];

    memset(RO, 0, sizeof(RO));

#pragma unroll
    for (size_t base_seqlen_k = warp_id * groups_per_warp; base_seqlen_k < binfo.actual_seqlen_k;
         base_seqlen_k += groups_per_block) {
        size_t seqlen_k = base_seqlen_k + group_id;

        if (seqlen_k < binfo.actual_seqlen_k) {
#pragma unroll
            for (size_t i = 0; i < thread_k_iters; ++i) {
                *(int4 *)(&RQV[i * thread_copy_k_elem_nums]) = *(int4 *)(&params.v_int8_ptr[binfo.k_offset(
                    seqlen_k, params.v_row_stride, params.v_head_stride,
                    (i * group_size + group_lane_id) * thread_copy_k_elem_nums)]);
            }

            float RV_scale = __half2float(params.v_scale_ptr[binfo.k_scale_offset(seqlen_k, params.h_k)]);

#pragma unroll
            for (size_t i = 0; i < thread_elem_nums; ++i) {
                RO[i] += (S_smem[seqlen_k] * (static_cast<float>(RQV[i]) * RV_scale));
            }
        }
    }

#pragma unroll
    for (size_t i = 0; i < thread_elem_nums; ++i) {
#pragma unroll
        for (size_t j = group_size; j <= warp_size / 2; j *= 2) {
            RO[i] += __shfl_xor_sync(shfl_mask, RO[i], j);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = threadIdx.x; i < head_dim; i += threads_per_block) {
        S_smem[i] = 0.0;
    }

    __syncthreads();

    if (lane_id < group_size) {
#pragma unroll
        for (size_t i = 0; i < thread_k_iters; ++i) {
#pragma unroll
            for (size_t j = 0; j < thread_copy_k_elem_nums; ++j) {
                atomicAdd(S_smem + (i * group_size + lane_id) * thread_copy_k_elem_nums + j,
                          RO[i * thread_copy_k_elem_nums + j]);
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = threadIdx.x; i < head_dim; i += threads_per_block) {
        params.o_ptr[binfo.q_offset(params.o_row_stride, params.o_head_stride, i)] = __float2half(S_smem[i]);
    }
}
