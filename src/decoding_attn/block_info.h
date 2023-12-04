// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: block info

#pragma once

#include "cuda_runtime_api.h"

struct DecodingBlockInfo
{
    template <typename Params>
    __device__ DecodingBlockInfo(const Params &params, const int bidb, const int bidh)
        : b(bidb),
          h(bidh),
          h_k(h / params.h_h_k_ratio),
          sum_s_q(params.cu_seqlens_q[b]),
          sum_s_k(params.cu_seqlens_k[b]),
          actual_seqlen_q(params.cu_seqlens_q[b + 1] - sum_s_q),
          actual_seqlen_k(params.cu_seqlens_k[b + 1] - sum_s_k),
          row_shift(actual_seqlen_k - actual_seqlen_q),
          h_slope(1.0 / exp2f(8.0 * (h + 1) / params.h)) {}

    inline __device__ size_t q_offset(const int row_stride, const int head_stride, const int dim_idx) const
    {
        // row stride = head_q * dim = model_dim
        // head stride = dim
        return sum_s_q * row_stride + h * head_stride + dim_idx;
    }

    inline __device__ size_t k_offset(const size_t seqlen_k, const int row_stride, const int head_stride,
                                      const int dim_idx) const
    {
        return (sum_s_k + seqlen_k) * row_stride + h_k * head_stride + dim_idx;
    }

    const int b;               // batch size index
    const int h;               // number of heads for Q
    const int h_k;             // number of heads for K
    const int sum_s_q;         // start point of the sum of all the previous sequences length for Q
    const int sum_s_k;         // start point of the sum of all the previous sequences length for K
    const int actual_seqlen_q; // actual sequence length for Q
    const int actual_seqlen_k; // actual sequence length for K
    const int row_shift;       // difference between actual sequence length for K and Q, used for row index
    const float h_slope;
};
