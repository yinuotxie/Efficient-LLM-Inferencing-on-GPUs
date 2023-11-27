#pragma once

#include <cutlass/numeric_types.h>
#include "cute/algorithm/copy.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = cutlass::half_t>
struct DecodingKernelTraits
{
    using Element = elem_type;

    static constexpr size_t warp_size = 32;
    static constexpr size_t kNWarpsPerBlock = kNWarps_;
    static constexpr size_t kNThreadsPerBlock = kNWarps_ * warp_size;
    static constexpr size_t kBlockM = kBlockM_; // Br
    static constexpr size_t kBlockN = kBlockN_; // Bc
    static constexpr size_t kHeadDim = kHeadDim_;
    static constexpr size_t group_size = kHeadDim / kBlockN;
    static constexpr size_t groups_per_warp = warp_size / group_size;

    static constexpr size_t threadNElemPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kNThreadsPerBlock % threadNElemPerLoad == 0, "kNThreadsPerBlock must be a multiple of threadNElemPerLoad");

    // Number of elements repsonsible for each thread
    static constexpr size_t threadIters = kBlockN / threadNElemPerLoad;
    static_assert(kNThreadsPerBlock % threadIters == 0, "kNThreads must be a multiple of threadIters");

    static constexpr unsigned int shfl_mask = 0xffffffff;
};