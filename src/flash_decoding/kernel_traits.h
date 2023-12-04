#pragma once

#include "cutlass/cutlass.h"
#include <cutlass/numeric_types.h>

/**
 * @brief Defines traits for the Flash Decoding Kernel in CUDA using CUTLASS.
 *
 * This structure provides configuration constants required for the
 * implementation of a decoding kernel. It includes settings for
 * thread block size, warp size, and group size, among others.
 *
 * @tparam kHeadDim Dimension of the head in the decoding operation.
 * @tparam kNThreadsPerBlock Number of threads per block.
 * @tparam kNThreadsPerGroup Number of threads per group.
 * @tparam elem_type Type of the elements being processed (default is cutlass::half_t).
 */
template <int kHeadDim, int kNThreadsPerBlock, int kNThreadsPerGroup,
          typename elem_type = cutlass::half_t>
struct FlashDecodingKernelTraits
{
    using Element = elem_type; // Type of elements being processed.

    static constexpr int warp_size = 32;                         // Size of a warp in CUDA.
    static constexpr int numThreadsPerBlock = kNThreadsPerBlock; // Number of threads per block.
    static constexpr int numThreadsPerGroup = kNThreadsPerGroup; // Number of threads per group.
    static constexpr int HeadDim = kHeadDim;                     // Dimension of the head.

    static constexpr int numWarpsPerBlock = numThreadsPerBlock / warp_size; // Number of warps per block.
    static_assert(numThreadsPerBlock % warp_size == 0, "numThreadsPerBlock must be a multiple of warp_size");

    static constexpr int numGroupsPerWarp = warp_size / numThreadsPerGroup; // Number of groups per warp.
    static_assert(warp_size % numThreadsPerGroup == 0, "warp_size must be a multiple of numThreadsPerGroup");

    static constexpr int numGroupsPerBlock = numThreadsPerBlock / numThreadsPerGroup; // Number of groups per block.
    static_assert(numThreadsPerBlock % numThreadsPerGroup == 0, "numThreadsPerBlock must be a multiple of numThreadsPerGroup");

    static constexpr int numElemPerLoad = 16; // sizeof(cute::uint128_t) / sizeof(Element); // Number of elements each iteration can load.
    static_assert(numElemPerLoad > 0, "numElemPerLoad must be greater than zero");

    // Number of elements each group is responsible for
    static constexpr int numElemPerGroup = HeadDim / numThreadsPerGroup; // Number of elements each thread group handles.
    static_assert(numElemPerGroup > 0, "numElemPerGroup must be greater than zero");

    static constexpr int groupIters = numElemPerGroup / numElemPerLoad; // Iterations per group for processing elements.
    static_assert(groupIters > 0, "groupIters must be greater than zero");

    static constexpr unsigned int shfl_mask = 0xffffffff; // Shuffle mask used in thread communication.
};