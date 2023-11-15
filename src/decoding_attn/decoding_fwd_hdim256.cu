// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding fwd hdim256

#include "decoding_attn/decoding_fwd_launch_template.h"

template <>
void run_mha_decoding_fwd_<256>(const DecodingParams &params) {
    mha_decoding_fwd<256, 256, 16>(params);
}
