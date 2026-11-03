// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
namespace qfn_calibrated {
uint64_t workspace_bytes(int nt);
int prepare(void * stream);
// Exact current decode/verify capacity: one to four target rows, ten routes each.
uint64_t decode_workspace_bytes(int nt);
int run_decode(const float * x,const int32_t * physical_ids,const int32_t * logical_ids,
        const uint8_t * raw,uint32_t slots,const float * input_scale,int nt,
        float * out,void * workspace,void * stream);
int run(const float * x,const int32_t * ids,const uint8_t * raw,const float * input_scale,
        int nt,float * out,void * workspace,void * stream);
}
