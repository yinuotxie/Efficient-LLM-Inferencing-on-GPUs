#pragma once

#include <cuda.h>
#include <vector>

struct QKV_params {
    using index_t = uint32_t;

    // The QKV matrices
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the QKV matrices
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;     // model dimension
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;    // head dimension
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads
    int h_q, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads_q (query)
    int h_q_h_k_ratio;  // precompute h_q / h_k
}

struct Flash_Fwd_params : public QKV_params {
    // The O matrix (output)
    void *__restrict__ o_ptr;

    // The stride between rows of O
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the softmax sum
    void *__restrict__ softmax_lse_ptr;

    // The scaling factor for softmax
    float scale_softmax;

    // array of sequence lengths
    const int *__restrict__ cu_seqlens_q;
    const int *__restrict__ cu_seqlens_k;

    // causual mask
    bool is_causal;

    cudaDeviceProp *device_prop;
    bool is_sm8x;
}