#pragma once

#include "decoding_attn_new/decoding_kernel.h"
#include "decoding_attn_new/static_switch.h"

/**
 * @brief The kernel for the forward pass of the FLASH attention.
 * @tparam KernelTraits The kernel traits class.
 * @tparam IsEvenMN Whether the sequence length is even divisible by the block size.
 * @param params The decoding parameters.
 */
template <typename KernelTraits, bool IsEvenMN>
__global__ void flash_fwd_kernel(DecodingParams params)
{
    flash::compute_attn<KernelTraits, IsEvenMN>(params);
}

/**
 * @brief The entry point of the forward pass of the FLASH attention.
 * @tparam KernelTraits The kernel traits class.
 * @param params The decoding parameters.
 */
template <typename KernelTraits>
void run_flash_fwd(const DecodingParams &params)
{
    const size_t warp_size = 32;
    // Static shared memory size calculation.
    const size_t static_smem_size = KernelTraits::kNThreadsPerBlock / warp_size * sizeof(float);
    // Dynamic shared memory size calculation.
    const size_t dynamic_smem_size = max(params.seqlen_k * sizeof(float), params.head_dim * sizeof(float));
    const size_t smem_size = static_smem_size + dynamic_smem_size;

    // Ensure an even division for optimal performance.
    const int num_m_blocks = (params.seqlen_q + KernelTraits::kBlockM - 1) / KernelTraits::kBlockM;

    dim3 gridSize(num_m_blocks, params.batch_size, params.num_q_head);
    dim3 blockSize(KernelTraits::kNThreadsPerBlock);

    // Check for even division and alignment.
    const bool is_even_MN = params.cu_seqlens_q != nullptr && params.cu_seqlens_k != nullptr &&
                            (*params.cu_seqlens_q % KernelTraits::kBlockM == 0) &&
                            (*params.cu_seqlens_k % KernelTraits::kBlockN == 0);

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&]
                {
    // Check shared memory size against the limit.
    if (smem_size >= params.props->sharedMemPerBlock) {
        FAI_CHECK_CUDART_ERROR(
            cudaFuncSetAttribute(&flash_fwd_kernel<KernelTraits, IsEvenMNConst>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    // Optimize the number of blocks per multiprocessor.
    int ctas_per_sm;
    FAI_CHECK_CUDART_ERROR(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas_per_sm, &flash_fwd_kernel<KernelTraits, IsEvenMNConst>, KernelTraits::kNThreadsPerBlock, smem_size));

    // Launch the kernel with optimized configurations.
    flash_fwd_kernel<KernelTraits, IsEvenMNConst><<<gridSize, blockSize, smem_size, params.stream>>>(params);

    // Check for kernel launch errors.
    FAI_CHECK_CUDART_ERROR(cudaPeekAtLastError()); });
}