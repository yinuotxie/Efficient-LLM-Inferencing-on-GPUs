// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:33:18 on Tue, Nov 07, 2023
//
// Description: decoding int8

#pragma once

#include "common.h"

struct DecodingInt8Params {
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
    int h_h_k_ratio;  // precompute h / h_k,

    // Quantization per head: half -> int8
    int8_t *__restrict__ k_int8_ptr;
    int8_t *__restrict__ v_int8_ptr;

    // Dequantization per head: int8 -> half
    half *k_scale_ptr;
    half *v_scale_ptr;

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
void run_quantization_int8_(const DecodingInt8Params &params);

template <size_t HeadDim>
void run_mha_decoding_int8_fwd_(const DecodingInt8Params &params);
