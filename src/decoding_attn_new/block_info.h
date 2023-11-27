#pragma once

#include "cuda_runtime_api.h"

/**
 * @namespace flash
 * @brief Namespace for the FlashAttention library.
 */
namespace flash
{
    /**
     * @brief Template struct representing block information.
     * @tparam Varlen Boolean indicating whether variable length is enabled or not.
     */
    template <bool Varlen = true>
    struct BlockInfo
    {
        /**
         * @brief Constructor for BlockInfo struct.
         * @tparam Params Type of the parameters.
         * @param params The parameters.
         * @param bidb The batch index.
         * @param bidh The query head index.
         */
        template <typename Params>
        __device__ BlockInfo(const Params &params, const int bidb, const int bidh)
            : batch_idx(bidb),
              q_head_idx(bidh),
              k_head_idx(q_head_idx / params.q_k_head_ratio),
              sum_seq_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb]),
              sum_seq_k(!Varlen || params.cu_seqlens_k == nullptr ? -1 : params.cu_seqlens_k[bidb]),
              actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q
                                                                        : params.cu_seqlens_q[bidb + 1] - sum_seq_q),
              actual_seqlen_k(!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k
                                                                        : params.cu_seqlens_k[bidb + 1] - sum_seq_k),
              row_shift(actual_seqlen_k - actual_seqlen_q) {}

        /**
         * @brief Calculates the offset for the query tensor.
         * @param row_stride The row stride.
         * @param head_stride The head stride.
         * @param dim_idx The dimension index.
         * @return The offset for the query tensor.
         */
        inline __device__ size_t q_offset(const int row_stride, const int head_stride, const int dim_idx) const
        {
            return sum_seq_q * row_stride + q_head_idx * head_stride + dim_idx;
        }

        /**
         * @brief Calculates the offset for the key tensor.
         * @param seqlen_k The sequence length for key tensor.
         * @param row_stride The row stride.
         * @param head_stride The head stride.
         * @param dim_idx The dimension index.
         * @return The offset for the key tensor.
         */
        inline __device__ size_t k_offset(const size_t seqlen_k, const int row_stride, const int head_stride,
                                          const int dim_idx) const
        {
            return (sum_seq_k + seqlen_k) * row_stride + k_head_idx * head_stride + dim_idx;
        }

        const int batch_idx;       ///< The batch index.
        const int q_head_idx;      ///< The query head index.
        const int k_head_idx;      ///< The key head index.
        const int sum_seq_q;       ///< The sum of sequence lengths for query tensor.
        const int sum_seq_k;       ///< The sum of sequence lengths for key tensor.
        const int actual_seqlen_q; ///< The actual sequence length for query tensor.
        const int actual_seqlen_k; ///< The actual sequence length for key tensor.
        const int row_shift;       ///< The row shift.
    };
} // namespace flash
