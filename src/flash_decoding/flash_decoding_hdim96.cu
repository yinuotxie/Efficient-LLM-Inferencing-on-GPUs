#include "flash_decoding/flash_decoding_launch_template.h"

template <>
void run_flash_decoding_new_<96>(const FlashDecodingParams &params)
{
    run_flash_decoding<FlashDecodingKernelTraits<96, 256, 4>>(params);
}