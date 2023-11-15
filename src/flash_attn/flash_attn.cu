// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: flash attn v1.0.9

#include "flash_attn/fmha.h"
#include "flash_attn/static_switch.h"
#include "tensor.h"

Launch_params<FMHA_fprop_params> set_fmha_fwd_params(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V, Tensor<half> *O,
                                                     Tensor<int> *cu_seq_q, Tensor<int> *cu_seq_k, size_t max_seq_q,
                                                     size_t max_seq_k, bool is_causal, int num_splits,
                                                     cudaStream_t stream, cudaDeviceProp *dev_prop, bool is_alibi) {
    size_t total_q = Q->getShape()[0];
    size_t head_q = Q->getShape()[1];
    size_t dim = Q->getShape()[2];
    size_t head_k = K->getShape()[1];
    size_t batch = cu_seq_q->getShape()[0] - 1;

    FAI_CHECK_LE(dim, 128);
    FAI_CHECK_EQ(dim % 8, 0);
    FAI_CHECK_EQ(head_q % head_k, 0);

    Launch_params<FMHA_fprop_params> launch_params(dev_prop, stream);

    // Reset the parameters
    memset(&launch_params.params, 0, sizeof(launch_params.params));

    // Set the pointers and strides.
    launch_params.params.q_ptr = reinterpret_cast<void *>(Q->getDevPtr());
    launch_params.params.k_ptr = reinterpret_cast<void *>(K->getDevPtr());
    launch_params.params.v_ptr = reinterpret_cast<void *>(V->getDevPtr());

    launch_params.params.q_row_stride_in_elts = head_q * dim;
    launch_params.params.k_row_stride_in_elts = head_k * dim;
    launch_params.params.v_row_stride_in_elts = head_k * dim;
    launch_params.params.q_head_stride_in_elts = dim;
    launch_params.params.k_head_stride_in_elts = dim;
    launch_params.params.v_head_stride_in_elts = dim;

    launch_params.params.h = head_q;
    launch_params.params.h_k = head_k;
    launch_params.params.h_h_k_ratio = launch_params.params.h / launch_params.params.h_k;

    launch_params.params.o_ptr = reinterpret_cast<void *>(O->getDevPtr());

    launch_params.params.o_row_stride_in_elts = head_q * dim;
    launch_params.params.o_head_stride_in_elts = dim;
    launch_params.params.o_tmp_row_stride_in_elts = head_q * dim;
    launch_params.params.o_tmp_head_stride_in_elts = dim;

    int blocksize_c = dim > 64 ? 128 : 256;
    // Need to round max_seqlen_k to multiples of blocksize_c
    int round_max_seq_k = ((max_seq_k + blocksize_c - 1) / blocksize_c) * blocksize_c;
    if (round_max_seq_k <= 128) {
        round_max_seq_k = 128;
    } else if (round_max_seq_k <= 256) {
        round_max_seq_k = 256;
    }

    if (round_max_seq_k > blocksize_c) {
        Tensor<float> *o_tmp = new Tensor<float>({total_q, head_q, dim}, "Tensor o_tmp");
        FAI_CHECK(o_tmp);
        launch_params.params.o_tmp_ptr = reinterpret_cast<void *>(o_tmp->getDevPtr());
    } else {
        launch_params.params.o_tmp_ptr = nullptr;
    }

    int round_max_seq_q = ((max_seq_q + 16 - 1) / 16) * 16;

    // Softmax sum
    Tensor<float> *softmax_lse =
        new Tensor<float>({batch, head_q, static_cast<size_t>(round_max_seq_q)}, "Tensor softmax_lse");
    FAI_CHECK(softmax_lse);
    launch_params.params.softmax_lse_ptr = reinterpret_cast<void *>(softmax_lse->getDevPtr());

    // Set the dimensions.
    launch_params.params.b = batch;
    launch_params.params.seqlen_q = round_max_seq_q;
    launch_params.params.seqlen_k = round_max_seq_k;
    launch_params.params.d = dim;

    launch_params.params.scale_bmm1f = 1.0 / std::sqrt(dim);
    set_alpha(launch_params.params.scale_bmm1, launch_params.params.scale_bmm1f, DATA_TYPE_FP16);

    launch_params.params.cu_seqlens_q = cu_seq_q->getDevPtr();
    launch_params.params.cu_seqlens_k = cu_seq_k->getDevPtr();

    launch_params.params.is_causal = is_causal;

    launch_params.params.num_splits = num_splits;

    launch_params.params.is_alibi = is_alibi;

    return launch_params;
}

void run_fmha_fwd(Launch_params<FMHA_fprop_params> &launch_params) {
    if (launch_params.params.d <= 32) {
        run_fmha_fwd_hdim32(launch_params);
    } else if (launch_params.params.d <= 64) {
        run_fmha_fwd_hdim64(launch_params);
    } else if (launch_params.params.d <= 128) {
        run_fmha_fwd_hdim128(launch_params);
    }
}

void flash_attn(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V, Tensor<half> *O, Tensor<int> *cu_seq_q,
                Tensor<int> *cu_seq_k, size_t max_seq_q, size_t max_seq_k, bool is_causal, int num_splits,
                cudaStream_t stream, cudaDeviceProp *dev_prop, bool is_alibi) {
    static Launch_params<FMHA_fprop_params> launch_params = set_fmha_fwd_params(
        Q, K, V, O, cu_seq_q, cu_seq_k, max_seq_q, max_seq_k, is_causal, num_splits, stream, dev_prop, is_alibi);
    run_fmha_fwd(launch_params);
}
