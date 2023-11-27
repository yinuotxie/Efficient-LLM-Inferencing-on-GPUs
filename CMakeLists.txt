# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: cmake for flash attention inference

cmake_minimum_required (VERSION 3.12)

project (flash_attention_inference LANGUAGES C CXX CUDA)

set (CMAKE_VERBOSE_MAKEFILE ${FAI_VERBOSE_MAKEFILE})

set (CMAKE_C_FLAGS "-std=c17")
set (CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -g2 -ggdb")
set (CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -O3")

set (CMAKE_CXX_FLAGS "-std=c++17")
set (CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb")
set (CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

set (CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")

add_compile_options (
    -Wall
    -Werror
    -Wextra
    -Wswitch-default
    -Wfloat-equal
    -Wshadow
    -Wcast-qual
)

# Nvidia GPU
find_package (CUDA REQUIRED)
unset (CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
option (CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set (CUDA_VERBOSE_BUILD ${FAI_VERBOSE_MAKEFILE})
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -Xptxas=-v -Xcompiler -fopenmp --expt-relaxed-constexpr")
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES} -g -lineinfo -O0")
else ()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES} --use_fast_math -O3")
endif ()

set (SYS_CUDART_PATH "/usr/local/cuda")
set (SYS_CUDA_DRIVER_PATH "/usr/lib/x86_64-linux-gnu")

find_package(gflags REQUIRED)
find_package(OpenMP REQUIRED)

include_directories (
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    ${SYS_CUDART_PATH}/include
    ${GFLAGS_INCLUDE_DIR}
)

link_directories (
    ${SYS_CUDART_PATH}/lib64
    ${SYS_CUDA_DRIVER_PATH}
)

file (GLOB FAI_SRCS 
    ${PROJECT_SOURCE_DIR}/src/*.cu
    ${PROJECT_SOURCE_DIR}/src/flash_attn/*.cu
    ${PROJECT_SOURCE_DIR}/src/flash_attn_v2/*.cu
    ${PROJECT_SOURCE_DIR}/src/decoding_attn/*.cu
    ${PROJECT_SOURCE_DIR}/src/decoding_attn_new/*.cu
)

cuda_add_executable (flash_attention_inference ${FAI_SRCS})
target_link_libraries (flash_attention_inference OpenMP::OpenMP_CXX ${GFLAGS_LIBRARIES})

install (TARGETS flash_attention_inference RUNTIME DESTINATION bin)