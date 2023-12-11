// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:11:35 on Wed, Sep 13, 2023
//
// Description: flash alibi

#pragma once

#include <cute/tensor.hpp>

namespace flash {

using namespace cute;

template <int kBlockN, int kNWarps, typename Tensor>
inline __device__ void add_alibi(Tensor &S, const float h_slope, const int row_start, const int row_end,
                                 const int col_start, const int col_end, const int row_shift) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    static_assert(decltype(size<0>(S.layout()))::value == 4);
    const int threads_per_row = kBlockN / (size<2>(S) * 2);
    const int rows_per_warp = (32 / threads_per_row) * 2;
    const int rows_per_iter = kNWarps * rows_per_warp;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int i = 0; i < size<2>(S); ++i) {
#pragma unroll
        for (int j = 0; j < size<1>(S); ++j) {
#pragma unroll
            for (int k = 0; k < size<0>(S); ++k) {
                int row = row_start + j * rows_per_iter + warp_id * rows_per_warp + lane_id / 4 + (k / 2) * 8;
                int col = col_start + (lane_id % 4) * 2 + i * 8 + k % 2;
                if (col < row + row_shift && row < row_end && col < col_end) {
                    S(k, j, i) += (h_slope * (col - row - row_shift));
                }
            }
        }
    }
}

}  // namespace flash
