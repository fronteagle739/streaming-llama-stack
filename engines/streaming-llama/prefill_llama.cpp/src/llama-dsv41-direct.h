#pragma once
#include "../ggml/include/ggml-dsv41-carry-copy.h"

// Read-only GPU checks for the existing F32/I32 carry contract. Integer bit
// tests preserve every finite/NaN/Inf distinction without floating arithmetic.
// This utility queues on the actual graph compute stream, writes only flags,
// and never changes model tensors or selects an alternate math kernel.
using llama_dsv41_validation=ggml_dsv41_validation;
// Returns the actual CUDA status. Caller must read exactly4 flag bytes and
// finish that stream before accepting a tile or starting dependent graph work.
inline int llama_dsv41_cuda_validate(const llama_dsv41_validation &p,void *flags,void *stream){return ggml_dsv41_cuda_validate(p,flags,stream);}
inline int llama_dsv41_cuda_copy_ram(bool to_device,const ggml_dsv41_ram_slice &slice,
    uint64_t offset,uint64_t width,void *device,void *stream){return ggml_dsv41_cuda_copy_ram(to_device,slice,offset,width,device,stream);}
