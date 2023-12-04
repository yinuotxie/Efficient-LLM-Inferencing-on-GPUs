#include "flash_decoding/flash_decoding_launch_template.h"

template <>
void run_flash_decoding_new_<128>(const FlashDecodingParams &params)
{
    run_flash_decoding<FlashDecodingKernelTraits<128, 256, 8>>(params);
}