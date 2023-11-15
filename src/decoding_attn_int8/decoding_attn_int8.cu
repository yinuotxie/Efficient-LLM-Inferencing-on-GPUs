// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:33:18 on Tue, Nov 07, 2023
//
// Description: decoding attn int8

#include "decoding_attn_int8/static_switch.h"
#include "decoding_attn_int8/util.h"

DecodingInt8Params set_mha_decoding_int8_fwd_params(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V, Tensor<half> *O,
                                                    Tensor<int> *cu_seq_q, Tensor<int> *cu_seq_k, size_t max_seq_q,
                                                    size_t max_seq_k, cudaStream_t stream, cudaDeviceProp *dev_prop,
                                                    bool is_alibi) {
    size_t head_q = Q->getShape()[1];
    size_t dim = Q->getShape()[2];
    size_t total_k = K->getShape()[0];
    size_t head_k = K->getShape()[1];
    size_t batch = cu_seq_q->getShape()[0] - 1;

    FAI_CHECK_LE(dim, 256);
    FAI_CHECK_EQ(head_q % head_k, 0);

    DecodingInt8Params params;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = Q->getDevPtr();
    params.k_ptr = K->getDevPtr();
    params.v_ptr = V->getDevPtr();

    params.q_row_stride = head_q * dim;
    params.k_row_stride = head_k * dim;
    params.v_row_stride = head_k * dim;
    params.q_head_stride = dim;
    params.k_head_stride = dim;
    params.v_head_stride = dim;

    params.h = head_q;
    params.h_k = head_k;
    params.h_h_k_ratio = params.h / params.h_k;

    Tensor<int8_t> *k_int8 = new Tensor<int8_t>({total_k, head_k, dim}, "Tensor k_int8");
    FAI_CHECK(k_int8);
    params.k_int8_ptr = k_int8->getDevPtr();

    Tensor<int8_t> *v_int8 = new Tensor<int8_t>({total_k, head_k, dim}, "Tensor v_int8");
    FAI_CHECK(v_int8);
    params.v_int8_ptr = v_int8->getDevPtr();

    Tensor<half> *k_scale = new Tensor<half>({total_k, head_k}, "Tensor k_scale");
    FAI_CHECK(k_scale);
    params.k_scale_ptr = k_scale->getDevPtr();

    Tensor<half> *v_scale = new Tensor<half>({total_k, head_k}, "Tensor v_scale");
    FAI_CHECK(v_scale);
    params.v_scale_ptr = v_scale->getDevPtr();

    params.o_ptr = O->getDevPtr();

    params.o_row_stride = head_q * dim;
    params.o_head_stride = dim;

    // Set the dimensions.
    params.b = batch;
    params.seqlen_q = max_seq_q;
    params.seqlen_k = max_seq_k;
    params.d = dim;

    params.scale_softmax = 1.0 / std::sqrt(dim);

    params.cu_seqlens_q = cu_seq_q->getDevPtr();
    params.cu_seqlens_k = cu_seq_k->getDevPtr();

    params.stream = stream;
    params.props = dev_prop;

    params.is_alibi = is_alibi;

    return params;
}

void run_quantization_int8(const DecodingInt8Params &params) {
    DECODING_INT8_FWD_HEADDIM_SWITCH(params.d, [&] { run_quantization_int8_<HeadDim>(params); });
}

void run_mha_decoding_int8_fwd(const DecodingInt8Params &params) {
    DECODING_INT8_FWD_HEADDIM_SWITCH(params.d, [&] { run_mha_decoding_int8_fwd_<HeadDim>(params); });
}

void decoding_attn_int8(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V, Tensor<half> *O, Tensor<int> *cu_seq_q,
                        Tensor<int> *cu_seq_k, size_t max_seq_q, size_t max_seq_k, bool is_causal, int num_splits,
                        cudaStream_t stream, cudaDeviceProp *dev_prop, bool is_alibi) {
    static DecodingInt8Params params = set_mha_decoding_int8_fwd_params(Q, K, V, O, cu_seq_q, cu_seq_k, max_seq_q,
                                                                        max_seq_k, stream, dev_prop, is_alibi);
    run_quantization_int8(params);
    // check_quantization_int8(params, K, V, cu_seq_k);
    run_mha_decoding_int8_fwd(params);
}
