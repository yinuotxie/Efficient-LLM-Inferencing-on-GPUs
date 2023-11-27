#pragma once

#include <cstddef>
#include "common.h"

struct DecodingParams
/**
 * @brief Struct representing the parameters for decoding attention.
 *
 * This struct contains the necessary parameters for performing decoding attention.
 * It includes the Q, K, V matrices parameters, the row stride and head stride of the matrices,
 * the number of heads, the O matrix parameters, batch size, sequence lengths, head dimension,
 * and other necessary parameters for the decoding attention operation.
 */
{
    // The Q, K, V matrices parameters
    half *__restrict__ q_ptr;
    half *__restrict__ k_ptr;
    half *__restrict__ v_ptr;

    // The row stride of Q, K, V matrices equals to the embedding dim
    size_t q_row_stride;
    size_t k_row_stride;
    size_t v_row_stride;

    // The head stride of Q, K, V matrices equals to the head dim
    size_t q_head_stride;
    size_t k_head_stride;
    size_t v_head_stride;

    // The number of heads.
    int num_q_head, num_k_head;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be different from nheads (query).
    int q_k_head_ratio = num_q_head / num_k_head;

    // The O matrix parameters
    half *__restrict__ o_ptr;

    size_t o_row_stride;
    size_t o_head_stride;

    int batch_size, seqlen_q, seqlen_k, head_dim;

    float scale_softmax;

    // array of length b+1 holding starting offset of each sequence.
    int *__restrict__ cu_seqlens_q;
    int *__restrict__ cu_seqlens_k;

    cudaStream_t stream;
    cudaDeviceProp *props;
};

template <int HeadDim>
void run_mha_decoding_fwd_new_(const DecodingParams &params);