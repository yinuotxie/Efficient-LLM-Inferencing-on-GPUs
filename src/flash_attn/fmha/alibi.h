// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:11:35 on Wed, Sep 13, 2023
//
// Description: fmha alibi

#pragma once

#include <flash_attn/fmha/gemm.h>

namespace fmha {

template <int CTA_N, typename Acc, int M, int N>
inline __device__ void add_alibi(Acc (&S)[M][N], const float h_slope, const int row_start, const int row_end,
                                 const int col_start, const int col_end, const int row_shift) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int i = 0; i < M; ++i) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                int row = row_start + i * 16 + ((k % 4) / 2) * 8 + lane_id / 4;
                int col = col_start + j * CTA_N / N + warp_id * 16 + (k / 4) * 8 + (lane_id % 4) * 2 + k % 2;
                int warp_col_end = std::min(col_end, col_start + j * CTA_N / N + (warp_id + 1) * 16);
                if (col < row + row_shift && row < row_end && col < warp_col_end) {
                    S[i][j].elt(k) += (h_slope * (col - row - row_shift));
                }
            }
        }
    }
}

}  // namespace fmha
