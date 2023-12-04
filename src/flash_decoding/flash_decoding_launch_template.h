#pragma once

#include "flash_decoding/flash_decoding_kernel.h"
#include "flash_decoding/static_switch.h"

/**
 * @brief The entry point of the forward pass of the FLASH attention.
 * @tparam KernelTraits The kernel traits class.
 * @param params The decoding parameters.
 */
template <typename KernelTraits>
void run_flash_decoding(const FlashDecodingParams &params)
{
    // Static shared memory size.
    const size_t static_smem_size = KernelTraits::numWarpsPerBlock * sizeof(float);
    // Dynamic shared memory size calculation.
    const size_t dynamic_smem_size = max(params.seqlen_k * sizeof(float), params.head_dim * sizeof(float));
    // check if the shared memory size is within the limit.
    FAI_CHECK_GT(params.props->sharedMemPerBlock, static_smem_size + dynamic_smem_size);

    dim3 gridSize(params.batch_size, params.num_q_head);
    dim3 blockSize(KernelTraits::numThreadsPerBlock);

    // Launch the kernel.
    flashDecoding<KernelTraits><<<gridSize, blockSize, dynamic_smem_size, params.stream>>>(params);

    // Check for kernel launch errors.
    FAI_CHECK_CUDART_ERROR(cudaPeekAtLastError());
}