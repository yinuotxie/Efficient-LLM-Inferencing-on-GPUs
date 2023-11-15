// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:33:18 on Tue, Nov 07, 2023
//
// Description: util

#pragma once

#include "decoding_attn_int8/decoding_int8.h"
#include "tensor.h"

inline void check_quantization_int8(const DecodingInt8Params &params, Tensor<half> *K, Tensor<half> *V,
                                    Tensor<int> *cu_seq_k) {
    size_t total_k = K->getShape()[0];
    size_t head_k = K->getShape()[1];
    size_t dim = K->getShape()[2];
    size_t k_elem_num = K->getElemNum();
    half *k_ptr = K->getHostPtr();
    half *v_ptr = V->getHostPtr();
    size_t batch = cu_seq_k->getShape()[0] - 1;
    int *cu_seq_k_ptr = cu_seq_k->getHostPtr();

    int8_t *k_int8_ptr = new int8_t[k_elem_num];
    FAI_CHECK(k_int8_ptr);
    FAI_CHECK_CUDART_ERROR(
        cudaMemcpy(k_int8_ptr, params.k_int8_ptr, k_elem_num * sizeof(int8_t), cudaMemcpyDeviceToHost));

    int8_t *v_int8_ptr = new int8_t[k_elem_num];
    FAI_CHECK(v_int8_ptr);
    FAI_CHECK_CUDART_ERROR(
        cudaMemcpy(v_int8_ptr, params.v_int8_ptr, k_elem_num * sizeof(int8_t), cudaMemcpyDeviceToHost));

    size_t k_scale_elem_num = total_k * head_k;
    half *k_scale_ptr = new half[k_scale_elem_num];
    FAI_CHECK(k_scale_ptr);
    FAI_CHECK_CUDART_ERROR(
        cudaMemcpy(k_scale_ptr, params.k_scale_ptr, k_scale_elem_num * sizeof(half), cudaMemcpyDeviceToHost));

    half *v_scale_ptr = new half[k_scale_elem_num];
    FAI_CHECK(v_scale_ptr);
    FAI_CHECK_CUDART_ERROR(
        cudaMemcpy(v_scale_ptr, params.v_scale_ptr, k_scale_elem_num * sizeof(half), cudaMemcpyDeviceToHost));

    half *k_dequantization_ptr = new half[k_elem_num];
    FAI_CHECK(k_dequantization_ptr);
    half *v_dequantization_ptr = new half[k_elem_num];
    FAI_CHECK(v_dequantization_ptr);

    double k_max_diff = 0.0;
    double v_max_diff = 0.0;
    double k_avg_diff = 0.0;
    double v_avg_diff = 0.0;
    for (size_t b = 0; b < batch; ++b) {
        size_t sum_seq_k = static_cast<size_t>(cu_seq_k_ptr[b]);
        size_t seq_k = static_cast<size_t>(cu_seq_k_ptr[b + 1]) - sum_seq_k;
        for (size_t h = 0; h < head_k; ++h) {
            for (size_t sk = 0; sk < seq_k; ++sk) {
                float k_scale = __half2float(k_scale_ptr[(sum_seq_k + sk) * head_k + h]);
                float v_scale = __half2float(v_scale_ptr[(sum_seq_k + sk) * head_k + h]);
                for (size_t d = 0; d < dim; ++d) {
                    k_dequantization_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d] = __float2half(
                        __half2float(static_cast<half>(k_int8_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d])) *
                        k_scale);
                    v_dequantization_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d] = __float2half(
                        __half2float(static_cast<half>(v_int8_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d])) *
                        v_scale);

                    double k_diff = static_cast<double>(
                        std::abs(__half2float(k_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d]) -
                                 __half2float(k_dequantization_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d])));
                    double v_diff = static_cast<double>(
                        std::abs(__half2float(v_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d]) -
                                 __half2float(v_dequantization_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d])));

                    k_max_diff = std::max(k_max_diff, k_diff);
                    v_max_diff = std::max(v_max_diff, v_diff);
                    k_avg_diff += k_diff;
                    v_avg_diff += v_diff;
                }
            }
        }
    }

    FLOG("Quantization: k_max_diff: %f, k_avg_diff: %f, v_max_diff: %f, v_avg_diff: %f", k_max_diff,
         k_avg_diff / k_elem_num, v_max_diff, v_avg_diff / k_elem_num);

    if (k_int8_ptr) {
        delete[] k_int8_ptr;
        k_int8_ptr = nullptr;
    }

    if (v_int8_ptr) {
        delete[] v_int8_ptr;
        v_int8_ptr = nullptr;
    }

    if (k_scale_ptr) {
        delete[] k_scale_ptr;
        k_scale_ptr = nullptr;
    }

    if (v_scale_ptr) {
        delete[] v_scale_ptr;
        v_scale_ptr = nullptr;
    }

    if (k_dequantization_ptr) {
        delete[] k_dequantization_ptr;
        k_dequantization_ptr = nullptr;
    }

    if (v_dequantization_ptr) {
        delete[] v_dequantization_ptr;
        v_dequantization_ptr = nullptr;
    }
}
