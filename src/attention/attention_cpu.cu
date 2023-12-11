#include "tensor.h"

void attention_cpu(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V, Tensor<half> *O, Tensor<int> *cu_seq_q,
                Tensor<int> *cu_seq_k, size_t max_seq_q, size_t max_seq_k, bool is_causal, bool is_alibi) {
    size_t total_q = Q->getShape()[0];
    size_t head_q = Q->getShape()[1];
    size_t dim = Q->getShape()[2];
    size_t head_k = K->getShape()[1];
    size_t batch = cu_seq_q->getShape()[0] - 1;

    FAI_CHECK_EQ(head_q % head_k, 0);   // head_q should be divisible by head_k, can be changed to MHA, MQA, GQA
    const size_t head_ratio = head_q / head_k;

    half *q_ptr = Q->getHostPtr();
    half *k_ptr = K->getHostPtr();
    half *v_ptr = V->getHostPtr();
    half *o_ptr = O->getHostPtr();

    int *cu_seq_q_ptr = cu_seq_q->getHostPtr();
    int *cu_seq_k_ptr = cu_seq_k->getHostPtr();

    // S = Q * K^T
    Tensor<float> *S = new Tensor<float>({total_q, head_q, max_seq_k}, "Tensor S");
    FAI_CHECK(S);
    float *s_ptr = S->getHostPtr();
    for (size_t b = 0; b < batch; ++b) {
        // starting point of q for batch b
        size_t sum_seq_q = static_cast<size_t>(cu_seq_q_ptr[b]);
        size_t sum_seq_k = static_cast<size_t>(cu_seq_k_ptr[b]);
        // q length and k length for batch b
        size_t seq_q = static_cast<size_t>(cu_seq_q_ptr[b + 1]) - sum_seq_q;
        size_t seq_k = static_cast<size_t>(cu_seq_k_ptr[b + 1]) - sum_seq_k;
        for (size_t h = 0; h < head_q; ++h) {
            size_t h_k = h / head_ratio;    // h_k is from 0 to head_q / (head_q / head_k) - 1
            for (size_t sq = 0; sq < seq_q; ++sq) {
                for (size_t sk = 0; sk < seq_k; ++sk) {
                    float tmp = 0.0;
                    for (size_t d = 0; d < dim; ++d) {
                        tmp += __half2float(q_ptr[(sum_seq_q + sq) * (head_q * dim) + h * dim + d]) *
                                __half2float(k_ptr[(sum_seq_k + sk) * (head_k * dim) + h_k * dim + d]);
                    }
                    s_ptr[sum_seq_q * (head_q * seq_k) + sq * (head_q * seq_k) + h * seq_k + sk] = tmp;
                }
            }
        }
    }

    // P = Softmax(S)
    Tensor<float> *P = new Tensor<float>({total_q, head_q, max_seq_k}, "Tensor P");
    FAI_CHECK(P);
    float *p_ptr = P->getHostPtr();
    float scale = 1.0 / std::sqrt(dim);
    for (size_t b = 0; b < batch; ++b) {
        size_t sum_seq_q = static_cast<size_t>(cu_seq_q_ptr[b]);
        size_t sum_seq_k = static_cast<size_t>(cu_seq_k_ptr[b]);
        size_t seq_q = static_cast<size_t>(cu_seq_q_ptr[b + 1]) - sum_seq_q;
        size_t seq_k = static_cast<size_t>(cu_seq_k_ptr[b + 1]) - sum_seq_k;
        size_t row_shift = seq_k - seq_q; // given seq_k >= seq_q ?
        for (size_t h = 0; h < head_q; ++h) {
            float h_slope = is_alibi ? (1.0 / exp2(8.0 * (h + 1) / head_q)) : 0.0;
            for (size_t sq = 0; sq < seq_q; ++sq) {
                size_t col_limit = is_causal ? std::min(seq_k, sq + row_shift + 1) : seq_k;

                // Max(S)
                std::vector<float> tmp_s(seq_k, 0.0);
                float max_s = -std::numeric_limits<float>::max();
                for (size_t sk = 0; sk < col_limit; ++sk) {
                    tmp_s[sk] = s_ptr[(sum_seq_q + sq) * (head_q * seq_k) + h * seq_k + sk] * scale;
                    if (is_alibi && sk < sq + row_shift) { // col_limit < seq_q + seq_k - seq_q
                        tmp_s[sk] +=
                            (h_slope * (static_cast<int>(sk) - static_cast<int>(sq) - static_cast<int>(row_shift)));
                    }
                    max_s = std::max(max_s, tmp_s[sk]);
                }

                // Sum(S)
                float sum_s = 0.0;
                for (size_t sk = 0; sk < col_limit; ++sk) {
                    tmp_s[sk] = std::exp(tmp_s[sk] - max_s);
                    sum_s += tmp_s[sk];
                }

                // Softmax(S)
                for (size_t sk = 0; sk < col_limit; ++sk) {
                    p_ptr[(sum_seq_q + sq) * (head_q * seq_k) + h * seq_k + sk] = tmp_s[sk] / sum_s; //  On chip, compute 𝑚˜𝑖𝑗 = rowmax(S𝑖𝑗), P𝑖𝑗 = exp(S𝑖𝑗 − 𝑚˜𝑖𝑗)(pointwise)

                }

                // Causal(S)
                if (is_causal) {
                    for (size_t sk = col_limit; sk < seq_k; ++sk) {
                        p_ptr[(sum_seq_q + sq) * (head_q * seq_k) + h * seq_k + sk] = 0.0;
                    }
                }
            }
        }
    }

    // O = P * V
    for (size_t b = 0; b < batch; ++b) {
        size_t sum_seq_q = static_cast<size_t>(cu_seq_q_ptr[b]);
        size_t sum_seq_k = static_cast<size_t>(cu_seq_k_ptr[b]);
        size_t seq_q = static_cast<size_t>(cu_seq_q_ptr[b + 1]) - sum_seq_q;
        size_t seq_k = static_cast<size_t>(cu_seq_k_ptr[b + 1]) - sum_seq_k;
        for (size_t h = 0; h < head_q; ++h) {
            size_t h_k = h / head_ratio;
            for (size_t sq = 0; sq < seq_q; ++sq) {
                for (size_t d = 0; d < dim; ++d) {
                    float tmp = 0.0;
                    for (size_t sk = 0; sk < seq_k; ++sk) {
                        tmp += p_ptr[(sum_seq_q + sq) * (head_q * seq_k) + h * seq_k + sk] *
                                __half2float(v_ptr[(sum_seq_k + sk) * (head_k * dim) + h_k * dim + d]);
                    }
                    o_ptr[(sum_seq_q + sq) * (head_q * dim) + h * dim + d] = __float2half(tmp);
                }
            }
        }
    }

    if (S) {
        delete S;
        S = nullptr;
    }

    if (P) {
        delete P;
        P = nullptr;
    }
}