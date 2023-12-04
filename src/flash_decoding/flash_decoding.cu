#include "cutlass/half.h"
#include "flash_decoding/flash_decoding.h"
#include "flash_decoding/static_switch.h"
#include "tensor.h"

/**
 * @brief Set up parameters for multi-head attention (MHA) decoding forward pass.
 *
 * @param Q Tensor representing the query matrix.
 * @param K Tensor representing the key matrix.
 * @param V Tensor representing the value matrix.
 * @param O Tensor representing the output matrix.
 * @param cu_seqlens_q Tensor holding sequence lengths for queries.
 * @param cu_seqlens_k Tensor holding sequence lengths for keys.
 * @param seqlens_q Size of the query sequence length.
 * @param seqlens_k Size of the key sequence length.
 * @param stream CUDA stream for asynchronous execution.
 * @param dev_prop CUDA device properties.
 *
 * @return FlashDecodingParams struct filled with necessary parameters for the decoding process.
 */
FlashDecodingParams set_flash_decoding_params(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V, Tensor<half> *O,
                                              Tensor<int> *cu_seqlens_q, Tensor<int> *cu_seqlens_k, size_t seqlens_q,
                                              size_t seqlens_k, cudaStream_t stream, cudaDeviceProp *dev_prop)
{
    size_t num_q_head = Q->getShape()[1];
    size_t head_dim = Q->getShape()[2];
    size_t num_k_head = K->getShape()[1];
    size_t batch_size = cu_seqlens_q->getShape()[0] - 1;

    FAI_CHECK_LE(head_dim, 256);
    FAI_CHECK_EQ(num_q_head % num_k_head, 0);

    FlashDecodingParams params;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = Q->getDevPtr();
    params.k_ptr = K->getDevPtr();
    params.v_ptr = V->getDevPtr();

    params.q_row_stride = num_q_head * head_dim;
    params.k_row_stride = num_k_head * head_dim;
    params.v_row_stride = num_k_head * head_dim;
    params.q_head_stride = head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;

    params.num_q_head = num_q_head;
    params.num_k_head = num_k_head;
    params.q_k_head_ratio = num_q_head / num_k_head;

    // Set the params for O = softmax(QK^T/sqrt(dk))V
    params.o_ptr = O->getDevPtr();
    params.o_row_stride = num_q_head * head_dim;
    params.o_head_stride = head_dim;

    // Set the dimensions
    params.batch_size = batch_size;
    params.seqlen_q = seqlens_q;
    params.seqlen_k = seqlens_k;
    params.head_dim = head_dim;

    params.scale_softmax = 1.0f / std::sqrt(static_cast<float>(head_dim));

    params.cu_seqlens_q = cu_seqlens_q->getDevPtr();
    params.cu_seqlens_k = cu_seqlens_k->getDevPtr();

    params.stream = stream;
    params.props = dev_prop;

    return params;
}

/**
 * @brief Execute the forward pass of multi-head attention decoding.
 *
 * @param params The FlashDecodingParams struct containing all necessary parameters for the forward pass.
 */
void run_flash_decoding_new(const FlashDecodingParams &params)
{
    DECODING_FWD_HEADDIM_SWITCH(params.head_dim, [&]
                                { run_flash_decoding_new_<kHeadDim>(params); });
}

/**
 * @brief High-level function to perform new decoding attention operation.
 *
 * @param Q Tensor representing the query matrix.
 * @param K Tensor representing the key matrix.
 * @param V Tensor representing the value matrix.
 * @param O Tensor representing the output matrix.
 * @param cu_seqlens_q Tensor holding sequence lengths for queries.
 * @param cu_seqlens_k Tensor holding sequence lengths for keys.
 * @param seqlens_q Size of the query sequence length.
 * @param seqlens_k Size of the key sequence length.
 * @param is_causal Boolean indicating if the operation is causal.
 * @param num_splits Number of splits in the operation.
 * @param stream CUDA stream for asynchronous execution.
 * @param dev_prop CUDA device properties.
 * @param is_alibi Boolean indicating if alibi is used.
 */
void flash_decoding(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V, Tensor<half> *O,
                    Tensor<int> *cu_seqlens_q, Tensor<int> *cu_seqlens_k, size_t seqlens_q, size_t seqlens_k,
                    bool is_causal, int num_splits, cudaStream_t stream, cudaDeviceProp *dev_prop, bool is_alibi)
{
    static FlashDecodingParams params = set_flash_decoding_params(Q, K, V, O, cu_seqlens_q, cu_seqlens_k, seqlens_q, seqlens_k, stream, dev_prop);

    FLOG("------------------Flash Decoding Params------------------");
    FLOG("seq_len_q: %d, seq_len_k: %d, batch_size: %d, head_dim: %d", params.seqlen_q, params.seqlen_k, params.batch_size, params.head_dim);
    FLOG("num_q_head: %d, num_k_head: %d, q_k_head_ratio: %d", params.num_q_head, params.num_k_head, params.q_k_head_ratio);
    FLOG("q_row_stride: %d, k_row_stride: %d, v_row_stride: %d, q_head_stride: %d, k_head_stride: %d, v_head_stride: %d", params.q_row_stride, params.k_row_stride, params.v_row_stride, params.q_head_stride, params.k_head_stride, params.v_head_stride);
    cu_seqlens_q->printTensor();
    cu_seqlens_k->printTensor();

    run_flash_decoding_new(params);
}
