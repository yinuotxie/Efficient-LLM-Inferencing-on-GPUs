// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:33:18 on Tue, Nov 07, 2023
//
// Description: decoding int8 fwd hdim128

#include "decoding_attn_int8/decoding_int8_fwd_launch_template.h"

template <>
void run_quantization_int8_<128>(const DecodingInt8Params &params) {
    quantization_int8<128, 512, 16>(params);
}

template <>
void run_mha_decoding_int8_fwd_<128>(const DecodingInt8Params &params) {
    if (params.b <= 4) {
        mha_decoding_int8_fwd<128, 256, 8>(params);
    } else {
        mha_decoding_int8_fwd<128, 128, 8>(params);
    }
}
