#pragma once
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

// Native ds4.c:40887-40956: physical [position %128][layer37..39][5120]
// F32 mean rows, without another BF16 conversion. The caller owns the cache
// lock and synchronization; these helpers add only existing ggml CPY nodes.
namespace llama_dsv41_dspark_capture {
constexpr uint32_t width=5120,slots=3,capacity=128;
constexpr uint64_t bytes=uint64_t(width)*slots*capacity*sizeof(float);
struct span { uint32_t source_row=0,ring_row=0,rows=0; };
inline std::array<span,2> plan(uint32_t start,uint32_t count){
    if(!count||start>=1048576u||count>1048576u-start)
        throw std::runtime_error("DSpark capture position/count exceeds context bound");
    const uint32_t keep=std::min(count,capacity),skip=count-keep;
    const uint32_t slot=(start+skip)%capacity,head=std::min(keep,capacity-slot);
    return {{{skip,slot,head},{skip+head,0,keep-head}}};
}
inline void enqueue(ggml_context*ctx,ggml_cgraph*graph,ggml_tensor*ring,
                    ggml_tensor*mean,uint32_t slot,uint32_t start,uint32_t count){
    if(!ctx||!graph||!ring||!mean||slot>=slots||ring->type!=GGML_TYPE_F32||
       !ggml_is_contiguous(ring)||ring->ne[0]!=width||ring->ne[1]!=slots||
       ring->ne[2]!=capacity||ring->ne[3]!=1||ggml_nbytes(ring)!=bytes||
       mean->type!=GGML_TYPE_F32||!ggml_is_contiguous(mean)||
       mean->ne[0]!=width||mean->ne[1]!=count||mean->ne[2]!=1||mean->ne[3]!=1)
        throw std::runtime_error("DSpark capture requires canonical F32 mean and physical ring geometry");
    for(const auto&part:plan(start,count))if(part.rows){
        auto*src=ggml_view_2d(ctx,mean,width,part.rows,width*sizeof(float),
            uint64_t(part.source_row)*width*sizeof(float));
        auto*dst=ggml_view_2d(ctx,ring,width,part.rows,slots*width*sizeof(float),
            (uint64_t(part.ring_row)*slots+slot)*width*sizeof(float));
        ggml_build_forward_expand(graph,ggml_cpy(ctx,src,dst));
    }
}
struct completion {
    struct ticket {uint32_t slot=0,start=0,count=0;uint64_t serial=0;};
    std::array<uint32_t,slots> ends{};
    bool pending=false;
    ticket active{};
    uint64_t serial=0;
    ticket begin(uint32_t slot,uint32_t start,uint32_t count){
        if(pending||slot>=slots||ends[slot]!=start)
            throw std::runtime_error("DSpark capture must advance each layer contiguously");
        (void)plan(start,count);
        pending=true;active={slot,start,count,++serial};return active;
    }
    ticket begin_replay(uint32_t slot,uint32_t start,uint32_t count){
        if(pending||slot>=slots||ends[slot]>start||count<capacity)
            throw std::runtime_error("YOCO capture requires a complete replacement ring");
        (void)plan(start,count);
        pending=true;active={slot,start,count,++serial};return active;
    }
    void commit(const ticket&t){
        if(!pending||t.serial!=active.serial||t.slot!=active.slot||
           t.start!=active.start||t.count!=active.count)
            throw std::runtime_error("DSpark capture completion does not own the queued write");
        ends[t.slot]=t.start+t.count;pending=false;
    }
    bool ready(uint32_t position)const{
        return position&&!pending&&std::all_of(ends.begin(),ends.end(),[&](uint32_t x){return x==position;});
    }
    void clear(){ends={};pending=false;active={};++serial;}
};
}
