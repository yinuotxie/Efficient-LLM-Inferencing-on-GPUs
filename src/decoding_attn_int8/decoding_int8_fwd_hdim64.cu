// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:33:18 on Tue, Nov 07, 2023
//
// Description: decoding int8 fwd hdim64

#include "decoding_attn_int8/decoding_int8_fwd_launch_template.h"

template <>
void run_quantization_int8_<64>(const DecodingInt8Params &params) {
    quantization_int8<64, 256, 4>(params);
}

template <>
void run_mha_decoding_int8_fwd_<64>(const DecodingInt8Params &params) {
    mha_decoding_int8_fwd<64, 256, 4>(params);
}
