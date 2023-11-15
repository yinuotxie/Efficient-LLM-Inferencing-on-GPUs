// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: common macro

#pragma once

#include "cuda_fp16.h"
#include "logging.h"
#include "util.h"

#define FAI_LIKELY(x) __builtin_expect(!!(x), 1)
#define FAI_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define FAI_CHECK(x)                      \
    do {                                  \
        if (FAI_UNLIKELY(!(x))) {         \
            FLOG("Check failed: %s", #x); \
            exit(EXIT_FAILURE);           \
        }                                 \
    } while (0)

#define FAI_CHECK_EQ(x, y) FAI_CHECK((x) == (y))
#define FAI_CHECK_NE(x, y) FAI_CHECK((x) != (y))
#define FAI_CHECK_LE(x, y) FAI_CHECK((x) <= (y))
#define FAI_CHECK_LT(x, y) FAI_CHECK((x) < (y))
#define FAI_CHECK_GE(x, y) FAI_CHECK((x) >= (y))
#define FAI_CHECK_GT(x, y) FAI_CHECK((x) > (y))

#define FAI_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &) = delete;       \
    void operator=(const TypeName &) = delete

#define FAI_CHECK_CUDART_ERROR(_expr_)                                                            \
    do {                                                                                          \
        cudaError_t _ret_ = _expr_;                                                               \
        if (FAI_UNLIKELY(_ret_ != cudaSuccess)) {                                                 \
            const char *_err_str_ = cudaGetErrorName(_ret_);                                      \
            int _rt_version_ = 0;                                                                 \
            cudaRuntimeGetVersion(&_rt_version_);                                                 \
            int _driver_version_ = 0;                                                             \
            cudaDriverGetVersion(&_driver_version_);                                              \
            FLOG("CUDA Runtime API error = %04d \"%s\", runtime version: %d, driver version: %d", \
                 static_cast<int>(_ret_), _err_str_, _rt_version_, _driver_version_);             \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)
