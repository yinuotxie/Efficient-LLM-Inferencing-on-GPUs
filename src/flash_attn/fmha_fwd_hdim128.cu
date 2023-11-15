// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_attn/fmha_fwd_launch_template.h"

void run_fmha_fwd_hdim128(Launch_params<FMHA_fprop_params> &launch_params) {
    using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 4, 0x08u, __half>;
    run_fmha_fwd_loop<Kernel_traits>(launch_params);
}