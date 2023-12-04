#include "flash_decoding/flash_decoding_launch_template.h"

template <>
void run_flash_decoding_new_<64>(const FlashDecodingParams &params)
{
    run_flash_decoding<FlashDecodingKernelTraits<64, 256, 4>>(params);
}