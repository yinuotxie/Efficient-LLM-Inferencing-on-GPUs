// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: flash attn v2.1.0

#include "cutlass/half.h"
#include "flash_attn_v2/flash.h"
#include "flash_attn_v2/static_switch.h"
#include "tensor.h"

#define M_LOG2E 1.4426950408889634074  // log_2 e

Flash_fwd_params set_mha_fwd_params(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V, Tensor<half> *O,
                                    Tensor<int> *cu_seq_q, Tensor<int> *cu_seq_k, size_t max_seq_q, size_t max_seq_k,
                                    bool is_causal, cudaDeviceProp *dev_prop, bool is_alibi) {
    size_t head_q = Q->getShape()[1];
    size_t dim = Q->getShape()[2];
    size_t head_k = K->getShape()[1];
    size_t batch = cu_seq_q->getShape()[0] - 1;

    FAI_CHECK_LE(dim, 256);
    FAI_CHECK_EQ(head_q % head_k, 0);

    Flash_fwd_params params;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = reinterpret_cast<void *>(Q->getDevPtr());
    params.k_ptr = reinterpret_cast<void *>(K->getDevPtr());
    params.v_ptr = reinterpret_cast<void *>(V->getDevPtr());

    // Calculate batch_stride using cu_seq
    params.q_batch_stride = 0;
    params.k_batch_stride = 0;
    params.v_batch_stride = 0;
    params.q_row_stride = head_q * dim;
    params.k_row_stride = head_k * dim;
    params.v_row_stride = head_k * dim;
    params.q_head_stride = dim;
    params.k_head_stride = dim;
    params.v_head_stride = dim;

    params.h = head_q;
    params.h_k = head_k;
    params.h_h_k_ratio = params.h / params.h_k;

    params.o_ptr = reinterpret_cast<void *>(O->getDevPtr());

    // Calculate batch_stride using cu_seq
    params.o_batch_stride = 0;
    params.o_row_stride = head_q * dim;
    params.o_head_stride = dim;

    // Softmax sum
    Tensor<float> *softmax_lse = new Tensor<float>({batch, head_q, max_seq_q}, "Tensor softmax_lse");
    FAI_CHECK(softmax_lse);
    params.softmax_lse_ptr = reinterpret_cast<void *>(softmax_lse->getDevPtr());

    // Set the dimensions.
    params.b = batch;
    params.seqlen_q = max_seq_q;
    params.seqlen_k = max_seq_k;
    params.d = dim;

    params.scale_softmax = 1.0 / std::sqrt(dim);
    params.scale_softmax_log2 = params.scale_softmax * M_LOG2E;

    params.cu_seqlens_q = cu_seq_q->getDevPtr();
    params.cu_seqlens_k = cu_seq_k->getDevPtr();

    params.is_causal = is_causal;

    params.props = dev_prop;
    params.is_sm8x = params.props->major == 8 && params.props->minor > 0;
    params.is_alibi = is_alibi;

    return params;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FWD_HEADDIM_SWITCH(params.d, [&] { run_mha_fwd_<cutlass::half_t, kHeadDim>(params, stream); });
}

void flash_attn_v2(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V, Tensor<half> *O, Tensor<int> *cu_seq_q,
                   Tensor<int> *cu_seq_k, size_t max_seq_q, size_t max_seq_k, bool is_causal, int num_splits,
                   cudaStream_t stream, cudaDeviceProp *dev_prop, bool is_alibi) {
    static Flash_fwd_params params =
        set_mha_fwd_params(Q, K, V, O, cu_seq_q, cu_seq_k, max_seq_q, max_seq_k, is_causal, dev_prop, is_alibi);
    run_mha_fwd(params, stream);
}
