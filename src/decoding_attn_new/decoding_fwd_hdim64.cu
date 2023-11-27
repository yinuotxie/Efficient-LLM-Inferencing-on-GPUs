#include "decoding_attn_new/decoding_launch_template.h"

template <>
void run_mha_decoding_fwd_new_<64>(const DecodingParams &params)
{
    run_flash_fwd<DecodingKernelTraits<64, 128, 32, 4>>(params);
}