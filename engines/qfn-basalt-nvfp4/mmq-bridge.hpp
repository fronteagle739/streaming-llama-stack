// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
namespace qfn_mmq {
uint64_t workspace_bytes(int nt);
// Records mode3 readiness without removing existing native/W4A4 availability.
void prepare(void * stream);
int run(const float * x,const int32_t * ids,const uint8_t * raw,int nt,float * out,
        void * workspace,uint64_t bytes,void * stream);
// Explicit device-routing adapters implemented alongside the unchanged existing kernels.
void group(const int32_t * ids,const uint8_t * raw,int nt,int32_t * tok,
           int32_t * scatter,int32_t * bounds,void * stream);
void scatter(const float * sorted,const int32_t * map,float * out,int64_t values,void * stream);
}
