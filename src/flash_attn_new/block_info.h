#pragma once 

#include "cuda_runtime_api.h"

struct BlockInfo {
    template <typename Params>

    /*
    * @brief Constructor for BlockInfo
    * @param params: Params object containing all the parameters
    * @param batch_idx: Batch index: 0 <= batch_idx < batch_size
    * @param head_idx: Head index: 0 <= head_idx < num_heads
    * @return BlockInfo object
    */
    __device__ BlockInfo(const Params params, const int batch_idx, const int head_idx) : 
        sum_seq_k(params.cu_seqlens_q[batch_idx])
        sum_seq_q(params.cu_seqlens_k[batch_idx])
        actual_seqlen_q(params.cu_seqlens_q[batch_idx + 1] - sum_seq_q)
        actual_seqlen_k(params.cu_seqlens_k[batch_idx + 1] - sum_seq_k)
        row_shift(actual_seqlen_k - actual_seqlen_q) {}
        //h_slope(1.0 / (exp2f(8.0 * (head_idx + 1) / params.h) * params.scale_softmax)) {}

    /*
    * @brief Returns the q_offset for the given batch index
    * @param batch_stride: Stride between batches
    * @param row_stride: Stride between rows (model dimension)
    * @param batch_idx: Batch index: 0 <= batch_idx < batch_size
    * @return q_offset
    */
    template <typename index_t>
    inline __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int batch_idx) const {
        return sum_seq_q == -1 ? batch_idx * batch_stride : uint32_t(sum_seq_q) * row_stride;
    }

    /*
    * @brief Returns the k_offset for the given batch index
    * @param batch_stride: Stride between batches
    * @param row_stride: Stride between rows (model dimension)
    * @param batch_idx: Batch index: 0 <= batch_idx < batch_size
    * @return k_offset
    */
    template <typename index_t>
    inline __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int batch_idx) const {
        return sum_seq_k == -1 ? batch_idx * batch_stride : uint32_t(sum_seq_k) * row_stride;
    }

    /*
    sum_seq_k: prefix sum of sequence lengths for k

    */
    const int sum_seq_k;
    const int sum_seq_q;
    const int actual_seqlen_q;
    const int actual_seqlen_k;
    const int row_shift;
    const float h_slope;
}
