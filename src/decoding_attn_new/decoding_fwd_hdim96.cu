#include "decoding_attn_new/decoding_launch_template.h"

template <>
void run_mha_decoding_fwd_new_<96>(const DecodingParams &params)
{
    run_flash_fwd<DecodingKernelTraits<96, 128, 32, 4>>(params);
}