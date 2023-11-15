// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding fwd hdim128

#include "decoding_attn/decoding_fwd_launch_template.h"

template <>
void run_mha_decoding_fwd_<128>(const DecodingParams &params) {
    if (params.b <= 4) {
        mha_decoding_fwd<128, 256, 8>(params);
    } else {
        mha_decoding_fwd<128, 128, 16>(params);
    }
}
