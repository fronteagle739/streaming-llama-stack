#pragma once
#include "../ggml/include/ggml.h"
#include "../ggml/include/ggml-backend.h"
#include "../ggml/include/ggml-cuda.h"
#include "../ggml/include/ggml-dsv41-native-fp4.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

// Host adapter only. Reuse existing CUDA D2D tensor copy and the existing
// disjoint source/decode/FP4 packer. Original CPU tensor descriptors and file
// bytes stay unchanged; this explicit ultra option replaces their leased GPU
// contents. Every source chunk is copied before an earlier/same-address packed
// result can overwrite it. No future unread source byte may be touched.
inline ggml_tensor *llama_dsv41_pack_q4k_inplace(
        ggml_context *metadata,ggml_backend_t backend,const ggml_tensor *source,
        ggml_backend_buffer_t workspace,uint64_t scratch_offset,uint64_t scratch_bytes,
        uint64_t *chunks=nullptr){
    auto need=[](bool ok,const char *why){if(!ok)throw std::runtime_error(why);};
    constexpr uint64_t mib=1024ull*1024,raw_bytes=64*mib;
    need(metadata&&backend&&source&&source->type==GGML_TYPE_Q4_K&&source->buffer&&source->data&&
        ggml_is_contiguous(source)&&source->ne[0]>=256&&source->ne[0]<=20480&&source->ne[0]%256==0,
        "In-place expert FP4 source must be an owned contiguous Q4_K tensor");
    need(workspace&&scratch_bytes==64*mib&&scratch_offset<=ggml_backend_buffer_get_size(workspace)&&
        scratch_bytes<=ggml_backend_buffer_get_size(workspace)-scratch_offset,
        "In-place expert FP4 requires its existing64MiB cast scratch");
    const auto original_start=reinterpret_cast<uintptr_t>(source->data);
    const uint64_t original_bytes=ggml_nbytes(source);
    const auto owner_start=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(source->buffer));
    need(original_start>=owner_start&&original_start-owner_start<=ggml_backend_buffer_get_size(source->buffer)&&
        original_bytes<=ggml_backend_buffer_get_size(source->buffer)-(original_start-owner_start),
        "In-place expert FP4 source escaped its leased GPU allocation");
    auto *scratch=static_cast<char*>(ggml_backend_buffer_get_base(workspace))+scratch_offset;
    const auto scratch_start=reinterpret_cast<uintptr_t>(scratch);
    need(original_start<=UINTPTR_MAX-original_bytes&&scratch_start<=UINTPTR_MAX-scratch_bytes&&
        (original_start+original_bytes<=scratch_start||scratch_start+scratch_bytes<=original_start),
        "In-place expert FP4 scratch overlaps original weights");
    auto *packed=ggml_new_tensor(metadata,GGML_TYPE_MXFP4,GGML_MAX_DIMS,source->ne);
    ggml_set_name(packed,source->name);
    need(ggml_nbytes(packed)<original_bytes&&packed->nb[1]<source->nb[1],"In-place FP4 must strictly shrink the row layout");
    // These are typed aliases of existing bytes. A fresh quantized allocation
    // may zero its padding, which would corrupt a future unread Q4 source row.
    // Use the backend's existing view initialization after explicit bounds.
    packed->view_src=const_cast<ggml_tensor*>(source);packed->view_offs=0;
    need(ggml_backend_view_init(packed)==GGML_STATUS_SUCCESS,"In-place FP4 descriptor view binding failed");
    auto *work_owner=ggml_new_tensor_1d(metadata,GGML_TYPE_I8,scratch_bytes);
    need(ggml_backend_tensor_alloc(workspace,work_owner,scratch)==GGML_STATUS_SUCCESS,"Existing byte scratch owner binding failed");
    const int64_t k=source->ne[0],rows=ggml_nrows(source);
    const int64_t chunk_rows=std::min<int64_t>(65535,raw_bytes/source->nb[1]);
    need(chunk_rows>0&&chunk_rows<=65535,"In-place expert FP4 chunk bound");
    auto *stream=ggml_backend_cuda_get_stream(backend);need(stream,"In-place expert FP4 requires its actual CUDA stream");
    // The existing copy/pack APIs snapshot addresses and dimensions while
    // queueing; CUDA never dereferences these host tensor descriptors. Reuse
    // one bounded metadata context per weight instead of allocating per chunk.
    ggml_context *chunk_context=ggml_init({65536,nullptr,true});need(chunk_context,"In-place FP4 chunk metadata failed");
    struct cleanup {ggml_context *p;~cleanup(){ggml_free(p);}} guard{chunk_context};
    auto *input=ggml_new_tensor_2d(chunk_context,GGML_TYPE_Q4_K,k,chunk_rows);
    auto *staged=ggml_new_tensor_2d(chunk_context,GGML_TYPE_Q4_K,k,chunk_rows);
    auto *output=ggml_new_tensor_2d(chunk_context,GGML_TYPE_MXFP4,k,chunk_rows);
    uint64_t completed=0;
    for(int64_t first=0;first<rows;first+=chunk_rows){
        const int64_t count=std::min(chunk_rows,rows-first);
        const uint64_t input_offset=uint64_t(first)*source->nb[1],input_bytes=uint64_t(count)*source->nb[1];
        const uint64_t output_offset=uint64_t(first)*packed->nb[1],output_bytes=uint64_t(count)*packed->nb[1];
        need(input_offset<=original_bytes&&input_bytes<=original_bytes-input_offset&&input_bytes<=raw_bytes&&
            output_offset<=input_offset&&output_bytes<=input_offset+input_bytes-output_offset,
            "In-place FP4 pack would overwrite a future unread source chunk");
        for(auto *tensor:{input,staged,output}){
            tensor->ne[1]=count;tensor->nb[2]=tensor->nb[1]*size_t(count);tensor->nb[3]=tensor->nb[2];
            tensor->buffer=nullptr;tensor->data=nullptr;
        }
        input->view_src=const_cast<ggml_tensor*>(source);input->view_offs=size_t(input_offset);
        staged->view_src=work_owner;staged->view_offs=0;
        output->view_src=packed;output->view_offs=size_t(output_offset);
        need(ggml_backend_view_init(input)==GGML_STATUS_SUCCESS&&
            ggml_backend_view_init(staged)==GGML_STATUS_SUCCESS&&
            ggml_backend_view_init(output)==GGML_STATUS_SUCCESS,
            "In-place expert FP4 chunk escaped its original owners");
        ggml_backend_tensor_copy_async(backend,backend,input,staged);
        need(ggml_backend_cuda_dsv41_pack_q4k_mxfp4_fused(staged,output,stream)==0,
            "Existing disjoint FP4 packer refused the staged expert chunk");
        ++completed;
    }
    ggml_backend_synchronize(backend);
    const uint64_t packed_bytes=ggml_nbytes(packed);
    ggml_backend_tensor_memset(const_cast<ggml_tensor*>(source),0,size_t(packed_bytes),size_t(std::min<uint64_t>(65536,original_bytes-packed_bytes)));
    if(chunks)*chunks=completed;
    return packed;
}
