// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: kernel traits

#pragma once

#include <cstddef>

template <size_t HeadDim, size_t ThreadsPerBlock, size_t GroupSize>
struct DecodingKernelTraits
{
    static constexpr size_t head_dim = HeadDim;
    static constexpr size_t threads_per_block = ThreadsPerBlock;
    static constexpr size_t group_size = GroupSize;

    static constexpr size_t warp_size = 32;
    static constexpr size_t warps_per_block = threads_per_block / warp_size;

    static constexpr size_t groups_per_warp = warp_size / group_size;
    static constexpr size_t groups_per_block = groups_per_warp * warps_per_block;

    // each thread in the CUDA kernel is designed to copy 16 bytes of data in each iteration of a particular loop or operation.
    static constexpr size_t thread_copy_bytes = 16;
    /*
    This line calculates the number of elements that can be copied by each thread in one operation, based on the size of each element.

    sizeof(half) gives the size of a half data type, which is typically 2 bytes (a half is a half-precision floating-point number).

    Therefore, thread_copy_elem_nums will be 16 / 2 = 8. This means each thread can handle 8 elements of type half in each copy operation.
    */
    static constexpr size_t thread_copy_elem_nums = 16 / sizeof(half);

    /*
    thread_elem_nums is calculated as the number of elements each thread (group) is responsible for, based on the division of the head dimension by the group size. This helps in distributing the workload evenly among the threads.
    */
    static constexpr size_t thread_elem_nums = head_dim / group_size;
    // This line calculates the number of iterations each thread will need to perform to process all its assigned
    // elements.
    static constexpr size_t thread_iters = thread_elem_nums / thread_copy_elem_nums;

    static constexpr unsigned int shfl_mask = 0xffffffff;
};
