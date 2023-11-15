// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding fwd launch template

#pragma once

#include "decoding_attn/decoding_fwd_kernel.h"
#include "decoding_attn/static_switch.h"

template <size_t HeadDim, size_t ThreadsPerBlock, size_t GroupSize>
void mha_decoding_fwd(const DecodingParams &params) {
    constexpr size_t warp_size = 32;
    constexpr size_t static_smem_size = ThreadsPerBlock / warp_size * sizeof(float);
    const size_t dynamic_smem_size = std::max(params.seqlen_k * sizeof(float), params.d * sizeof(float));
    FAI_CHECK_GT(params.props->sharedMemPerBlock, static_smem_size + dynamic_smem_size);

    dim3 block(ThreadsPerBlock);
    dim3 grid(params.b, params.h);

    BOOL_SWITCH(params.is_alibi, IsAlibi, [&] {
        mha_decoding_fwd_kernel<DecodingKernelTraits<HeadDim, ThreadsPerBlock, GroupSize>, IsAlibi>
            <<<grid, block, dynamic_smem_size, params.stream>>>(params);
        FAI_CHECK_CUDART_ERROR(cudaPeekAtLastError());
    });
}
