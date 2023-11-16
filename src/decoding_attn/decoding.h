// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding

#pragma once

#include "common.h"

struct DecodingParams
{
    // The QKV matrices.
    half *__restrict__ q_ptr;
    half *__restrict__ k_ptr;
    half *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    size_t q_row_stride;
    size_t k_row_stride;
    size_t v_row_stride;
    size_t q_head_stride;
    size_t k_head_stride;
    size_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,

    // The O matrix (output).
    half *__restrict__ o_ptr;

    // The stride between rows of O.
    size_t o_row_stride;
    size_t o_head_stride;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d;

    // The scaling factors for the kernel.
    float scale_softmax;

    // array of length b+1 holding starting offset of each sequence.
    int *__restrict__ cu_seqlens_q;
    int *__restrict__ cu_seqlens_k;

    cudaStream_t stream;
    cudaDeviceProp *props;

    bool is_alibi;
};

template <size_t HeadDim>
void run_mha_decoding_fwd_(const DecodingParams &params);
