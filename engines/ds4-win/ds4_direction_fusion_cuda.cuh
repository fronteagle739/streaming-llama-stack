/* Target-weight fusion candidate. Hardware qualification is mandatory before
 * advertising this capability. Included after common dequantization/LoRA helpers.
 * Per-target A=r^T W is cached; W is always the original target quant.
 * Fused weights remain FP32 in a bounded VRAM residency cache. No base writes. */
#ifndef DS4_DIRECTION_FUSION_CUDA_CUH
#define DS4_DIRECTION_FUSION_CUDA_CUH
#include "cuda/mmq/ds4_fp4_packed_layout.cuh"
#include "ds4_direction_cache_memory.h"
#include "ds4_physical_vram.h"

static bool direction_cache_free_bytes(uint64_t &available,uint64_t &total) {
#if defined(_WIN32) && defined(DS4_EXTERNAL_DECODE)
    if(g_ds4x_external_mode) {
        if(!ds4_physical_vram::free_bytes(available,total)) {
            fprintf(stderr,"ds4: direction cache requires the producer's NVML physical VRAM measurement for imported DirectStorage buffers\n");return false;
        }
        return true;
    }
#endif
    size_t free_cuda=0,total_cuda=0;
    if(!cuda_ok(cudaMemGetInfo(&free_cuda,&total_cuda),"direction cache free VRAM"))return false;
    available=free_cuda;total=total_cuda;return true;
}

/* Internal device-source layout tag; the GGUF tensor type remains MXFP4/39. */
enum {DIRECTION_PACKED_MXFP4=0x10000u+39u};

__device__ static float direction_weight_value(const unsigned char *source,
        uint32_t type,uint32_t width,uint32_t output,uint32_t row,uint32_t col) {
    const uint64_t index=(uint64_t)row*width+col;
    if(type==DIRECTION_PACKED_MXFP4)return ds4_fp4p_weight(source,output,width,row,col);
    if(type==0u)return ((const float *)source)[index];
    if(type==1u)return __half2float(((const __half *)source)[index]);
    if(type==30u)return __uint_as_float((uint32_t)((const uint16_t *)source)[index]<<16);
    if(type==8u) {
        const unsigned char *block=source+((uint64_t)row*(width/32u)+col/32u)*34u;
        return __half2float(*(const __half *)block)*(float)((const int8_t *)(block+2))[col%32u];
    }
    if(type==39u) {
        /* MXFP4: one E8M0 scale byte, then 32 four-bit E2M1 values, low nibble first. */
        const unsigned char *block=source+((uint64_t)row*(width/32u)+col/32u)*17u;
        const uint32_t within=col%32u;
        const uint8_t pair=block[1u+within%16u];
        const uint32_t code=within<16u?(pair&15u):(pair>>4);
        const uint32_t e=block[0];
        const float scale=__uint_as_float(e?e<<23:0x00400000u);
        const uint32_t mag=code&7u;
        const float value=mag<4u?(float)mag*.5f:(float)(2u+((mag-4u)&1u))*(float)(1u<<((mag-4u)>>1));
        return scale*((code&8u)?-value:value);
    }
    if(type==10u) {
        const cuda_block_q2_K *block=(const cuda_block_q2_K *)source+(uint64_t)row*(width/256u)+col/256u;
        const uint32_t within=col%256u,chunk=within/128u,j=within%128u;
        const uint32_t group=j/32u,pair=(j%32u)/16u,lane=j%16u;
        const uint8_t scale=block->scales[chunk*8u+group*2u+pair];
        const uint32_t quant=(block->qs[chunk*32u+pair*16u+lane]>>(group*2u))&3u;
        return dev_f16_to_f32(block->d)*(float)(scale&15u)*(float)quant-
            dev_f16_to_f32(block->dmin)*(float)(scale>>4);
    }
    if(type==12u) {
        const cuda_block_q4_K *block=(const cuda_block_q4_K *)source+(uint64_t)row*(width/256u)+col/256u;
        const uint32_t within=col%256u,group=within/32u;
        uint8_t scale,minimum;dev_q4_K_get_scale_min(group,block->scales,&scale,&minimum);
        const uint8_t quant=block->qs[(group/2u)*32u+within%32u];
        return dev_f16_to_f32(block->d)*(float)scale*(float)((quant>>((group&1u)*4u))&15u)-
            dev_f16_to_f32(block->dmin)*(float)minimum;
    }
    return nanf("");
}

__global__ static void direction_target_a_kernel(float *a,const unsigned char *source,
        uint32_t type,uint32_t input,uint32_t output,const float *pack,
        uint32_t layer,uint32_t site,uint32_t count) {
    const uint32_t col=blockIdx.x*32u+threadIdx.x;
    const float *vectors=pack+((uint64_t)layer*2u+site)*count*(output+1u);
    float sums[8]={0};
    if(col<input)for(uint32_t row=threadIdx.y;row<output;row+=8u) {
        const float weight=direction_weight_value(source,type,input,output,row,col);
        for(uint32_t k=0;k<count;k++)sums[k]+=vectors[(uint64_t)k*(output+1u)+1u+row]*weight;
    }
    __shared__ float partial[8][8][32];
    for(uint32_t k=0;k<count;k++)partial[k][threadIdx.y][threadIdx.x]=sums[k];
    __syncthreads();
    if(!threadIdx.y && col<input)for(uint32_t k=0;k<count;k++) {
        float sum=0;for(uint32_t row=0;row<8;row++)sum+=partial[k][row][threadIdx.x];
        a[(uint64_t)k*input+col]=sum;
    }
}

__global__ static void direction_target_b_kernel(float *b,const float *pack,
        uint32_t layer,uint32_t site,uint32_t output,uint32_t count) {
    const uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=output)return;
    const float *vectors=pack+((uint64_t)layer*2u+site)*count*(output+1u);
    for(uint32_t k=0;k<count;k++)b[(uint64_t)row*count+k]=
        -vectors[(uint64_t)k*(output+1u)]*vectors[(uint64_t)k*(output+1u)+1u+row];
}

__global__ static void direction_target_dequant_kernel(float *weight,const unsigned char *source,
        uint32_t type,uint32_t input,uint32_t output) {
    const uint64_t index=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(index<(uint64_t)input*output)weight[index]=direction_weight_value(source,type,input,output,
        (uint32_t)(index/input),(uint32_t)(index%input));
}

/* Retain the complete 1017 target-factor closure, including all routed
 * experts. A 512-entry cache refused this fuse-once residency contract by
 * recomputing evicted A factors on every decode sweep. The much larger dense
 * fused-weight tensors obey the separately admitted dense writer ceiling. */
enum {DIRECTION_A_CACHE_COUNT=40*(384+2)};
typedef struct {
    const void *model_map;
    uint64_t offset,age;
    uint32_t type,input,output,layer,site,count;
    ds4_gpu_tensor *a,*weight;
} direction_a_entry;
static struct {
    const ds4_gpu_tensor *pack;
    direction_a_entry cache[DIRECTION_A_CACHE_COUNT];
    uint64_t clock,weight_bytes,fusions,hits;
    ds4_gpu_tensor *b,*status,*packed_a,*packed_dots,*packed_mid,*packed_ids,*host_packed,*packed_cpu_acc;
} g_direction_fused;

extern "C" void ds4_gpu_direction_fused_clear(void) {
    for(unsigned i=0;i<DIRECTION_A_CACHE_COUNT;i++) {
        ds4_gpu_tensor_free(g_direction_fused.cache[i].a);
        ds4_gpu_tensor_free(g_direction_fused.cache[i].weight);
    }
    ds4_gpu_tensor_free(g_direction_fused.b);
    ds4_gpu_tensor_free(g_direction_fused.packed_a);
    ds4_gpu_tensor_free(g_direction_fused.packed_dots);
    ds4_gpu_tensor_free(g_direction_fused.packed_mid);
    ds4_gpu_tensor_free(g_direction_fused.packed_ids);
    ds4_gpu_tensor_free(g_direction_fused.host_packed);
    ds4_gpu_tensor_free(g_direction_fused.packed_cpu_acc);
    ds4_gpu_tensor_free(g_direction_fused.status);memset(&g_direction_fused,0,sizeof(g_direction_fused));
}

extern "C" void ds4_gpu_direction_fused_stats(uint64_t *fusions,uint64_t *hits,uint64_t *bytes) {
    if(fusions)*fusions=g_direction_fused.fusions;
    if(hits)*hits=g_direction_fused.hits;
    if(bytes)*bytes=g_direction_fused.weight_bytes;
}

// Direction-owned buffers only; the graph's bank tensor is accounted by its
// owner. This query performs no allocation or device synchronization.
extern "C" uint64_t ds4_gpu_direction_allocated_bytes(void) {
    uint64_t bytes=g_direction_fused.weight_bytes;
    for(unsigned i=0;i<DIRECTION_A_CACHE_COUNT;i++)if(g_direction_fused.cache[i].a)bytes+=g_direction_fused.cache[i].a->bytes;
    const ds4_gpu_tensor *buffers[]={g_direction_fused.b,g_direction_fused.status,g_direction_fused.packed_a,
        g_direction_fused.packed_dots,g_direction_fused.packed_mid,g_direction_fused.packed_ids,
        g_direction_fused.host_packed,g_direction_fused.packed_cpu_acc};
    for(const auto *buffer:buffers)if(buffer)bytes+=buffer->bytes;
    return bytes;
}

static uint64_t direction_source_bytes(uint32_t type,uint32_t input,uint32_t output) {
    uint64_t row=0;
    if(type==0u)row=(uint64_t)input*4u;
    else if(type==1u || type==30u)row=(uint64_t)input*2u;
    else if(type==8u)row=(uint64_t)(input/32u)*34u;
    else if(type==39u || type==DIRECTION_PACKED_MXFP4)row=(uint64_t)(input/32u)*17u;
    else if(type==10u)row=(uint64_t)(input/256u)*sizeof(cuda_block_q2_K);
    else if(type==12u)row=(uint64_t)(input/256u)*sizeof(cuda_block_q4_K);
    return row*output;
}

static bool direction_quant_supported(uint32_t type,uint32_t width) {
    return width && (type==0u || type==1u || type==30u ||
        (type==DIRECTION_PACKED_MXFP4 && width%128u==0) || ((type==8u || type==39u) && width%32u==0) ||
        ((type==10u || type==12u) && width%256u==0));
}

/* Shared target-factor cache for fused float writers and packed-kernel epilogues. */
static direction_a_entry *direction_target_entry(const unsigned char *source,
        const void *model_map,uint64_t offset,uint32_t type,uint32_t input,uint32_t output,
        const ds4_gpu_tensor *pack,uint32_t layer,uint32_t site,uint32_t count,int logical_tier,
        bool source_on_host=false) {
    if(g_direction_fused.pack!=pack) {ds4_gpu_direction_fused_clear();g_direction_fused.pack=pack;}
    direction_a_entry *entry=NULL,*oldest=&g_direction_fused.cache[0];
    for(unsigned i=0;i<DIRECTION_A_CACHE_COUNT;i++) {
        direction_a_entry *candidate=&g_direction_fused.cache[i];
        if(!candidate->a)oldest=candidate;
        else if(oldest->a && candidate->age<oldest->age)oldest=candidate;
        if(candidate->a && candidate->model_map==model_map && candidate->offset==offset &&
            candidate->type==type && candidate->input==input && candidate->output==output &&
            candidate->layer==layer && candidate->site==site && candidate->count==count)entry=candidate;
    }
    const bool cached=entry && entry->a;
    if(!cached && !source)source=(const unsigned char *)cuda_resolve_weight_ptr(model_map,offset,
        direction_source_bytes(type,input,output),logical_tier,"direction:target-weight:read-only");
    if(!cached && !source)return NULL;
    if(source_on_host) {
        if(type!=DIRECTION_PACKED_MXFP4)return NULL;
        if(!cached) {
            // Borrowed RAM-tier bytes retain their packed format. Only target-A
            // construction runs here, on the GPU, once for this expert/bank.
            const uint64_t bytes=direction_source_bytes(type,input,output);
            if(!g_direction_fused.host_packed)g_direction_fused.host_packed=ds4_gpu_tensor_alloc(bytes);
            if(!g_direction_fused.host_packed || g_direction_fused.host_packed->bytes<bytes ||
                !cuda_ok(cudaMemcpyAsync(g_direction_fused.host_packed->ptr,source,bytes,cudaMemcpyHostToDevice,cuda_decode_stream()),"direction CPU-lane packed source upload") ||
                !cuda_ok(cudaStreamSynchronize(cuda_decode_stream()),"direction CPU-lane packed source retirement"))return NULL;
            source=(const unsigned char*)g_direction_fused.host_packed->ptr;
        }
    }
    if(!entry) {
        entry=oldest;ds4_gpu_tensor_free(entry->a);
        if(entry->weight) {
            g_direction_fused.weight_bytes-=entry->weight->bytes;
            ds4_gpu_tensor_free(entry->weight);
        }
        memset(entry,0,sizeof(*entry));
        entry->a=ds4_gpu_tensor_alloc((uint64_t)count*input*sizeof(float));
        if(!entry->a)return NULL;
        direction_target_a_kernel<<<(input+31u)/32u,dim3(32,8),0,cuda_decode_stream()>>>(
            (float *)entry->a->ptr,source,type,input,output,(const float *)pack->ptr,layer,site,count);
        if(!cuda_ok(cudaGetLastError(),"direction target A")) {ds4_gpu_tensor_free(entry->a);entry->a=NULL;return NULL;}
        entry->model_map=model_map;entry->offset=offset;entry->type=type;entry->input=input;
        entry->output=output;entry->layer=layer;entry->site=site;entry->count=count;
    }
    entry->age=++g_direction_fused.clock;
    if(type==DIRECTION_PACKED_MXFP4) {
        if(cached)++g_direction_fused.hits;else ++g_direction_fused.fusions;
    }
    return entry;
}

/* Exercise the SAME packed/Q8_1 down kernel and fused rank epilogue as the
 * serving path. One actual source expert, the first middle row nonzero; no
 * gate/up inference, six-matrix materialization or host layout conversion. */
__global__ static void direction_control_quarter_mid(float *mid,uint32_t width) {
    const uint32_t col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col<width)mid[width+col]=.25f*mid[col];
}
static int direction_packed_control(ds4_gpu_tensor *out,const void *model_map,uint64_t offset,
        const ds4_gpu_tensor *x,const ds4_gpu_tensor *pack,uint32_t layer,uint32_t site,uint32_t count,
        const ds4_gpu_tensor *cpu_base=nullptr) {
    const uint32_t input=2304u,output=5120u;
    const uint64_t bytes=direction_source_bytes(DIRECTION_PACKED_MXFP4,input,output);
    const unsigned char *source=(const unsigned char*)cuda_resolve_weight_ptr(model_map,offset,
        bytes,ds4_tensor_device_idx(out),"direction:packed-target:read-only");
    if(!source)return 0;
    direction_a_entry *entry=direction_target_entry(cpu_base?(const unsigned char*)model_map+offset:source,
        model_map,offset,DIRECTION_PACKED_MXFP4,input,output,pack,layer,site,count,ds4_tensor_device_idx(out),cpu_base!=nullptr);
    if(!entry){fprintf(stderr,"ds4: direction target factor refused layer=%u site=%u offset=%llu\n",layer,site,(unsigned long long)offset);return 0;}
    if(!g_direction_fused.packed_a)g_direction_fused.packed_a=ds4_gpu_tensor_alloc(6u*8u*input*sizeof(float));
    if(!g_direction_fused.packed_dots)g_direction_fused.packed_dots=ds4_gpu_tensor_alloc(6u*8u*sizeof(float));
    if(!g_direction_fused.packed_mid)g_direction_fused.packed_mid=ds4_gpu_tensor_alloc(6u*input*sizeof(float));
    if(!g_direction_fused.packed_ids)g_direction_fused.packed_ids=ds4_gpu_tensor_alloc(6u*sizeof(int32_t));
    if(!g_direction_fused.b)g_direction_fused.b=ds4_gpu_tensor_alloc(output*8u*sizeof(float));
    if(!g_direction_fused.packed_a || !g_direction_fused.packed_dots || !g_direction_fused.packed_mid ||
        !g_direction_fused.packed_ids || !g_direction_fused.b)return 0;
    for(unsigned a=0;a<6u;a++)if(!cuda_ok(cudaMemcpyAsync((float*)g_direction_fused.packed_a->ptr+(uint64_t)a*count*input,
        entry->a->ptr,(size_t)count*input*sizeof(float),cudaMemcpyDeviceToDevice,cuda_decode_stream()),"packed control A"))return 0;
    if(!cuda_ok(cudaMemsetAsync(g_direction_fused.packed_mid->ptr,0,6u*input*sizeof(float),cuda_decode_stream()),"packed control zeros") ||
        !cuda_ok(cudaMemsetAsync(g_direction_fused.packed_ids->ptr,0,6u*sizeof(int32_t),cuda_decode_stream()),"packed control IDs") ||
        !cuda_ok(cudaMemcpyAsync(g_direction_fused.packed_mid->ptr,x->ptr,input*sizeof(float),cudaMemcpyDeviceToDevice,cuda_decode_stream()),"packed control x"))return 0;
    direction_target_b_kernel<<<(output+255u)/256u,256,0,cuda_decode_stream()>>>(
        (float*)g_direction_fused.b->ptr,(const float*)pack->ptr,layer,site,output,count);
    if(!cuda_ok(cudaGetLastError(),"packed control B"))return 0;
    const ds4_mmq_writer_delta delta{(const float*)g_direction_fused.packed_a->ptr,
        (const float*)g_direction_fused.b->ptr,(float*)g_direction_fused.packed_dots->ptr,count};
    if(cpu_base) {
        if(!g_direction_fused.packed_cpu_acc)g_direction_fused.packed_cpu_acc=ds4_gpu_tensor_alloc(6u*output*sizeof(float));
        const int32_t ids[6]={-2,0,0,0,0,0};
        if(!g_direction_fused.packed_cpu_acc ||
            !cuda_ok(cudaMemsetAsync(g_direction_fused.packed_cpu_acc->ptr,0,6u*output*sizeof(float),cuda_decode_stream()),"mixed control CPU sums zero") ||
            !cuda_ok(cudaMemcpyAsync(g_direction_fused.packed_cpu_acc->ptr,cpu_base->ptr,output*sizeof(float),cudaMemcpyDeviceToDevice,cuda_decode_stream()),"mixed control CPU base") ||
            !ds4_gpu_tensor_write(g_direction_fused.packed_ids,0,ids,sizeof(ids)))return 0;
        direction_control_quarter_mid<<<(input+255u)/256u,256,0,cuda_decode_stream()>>>((float*)g_direction_fused.packed_mid->ptr,input);
        if(!cuda_ok(cudaGetLastError(),"mixed control GPU mid"))return 0;
        return ds4_mmq_mxfp4_packed_down_rank_mixed(source,bytes,(const float*)g_direction_fused.packed_mid->ptr,
            (const int32_t*)g_direction_fused.packed_ids->ptr,(float*)out->ptr,1u,
            (const float*)g_direction_fused.packed_cpu_acc->ptr,&delta,cuda_decode_stream())==0;
    }
    return ds4_mmq_mxfp4_packed_down_rank(source,bytes,(const float*)g_direction_fused.packed_mid->ptr,
        (const int32_t*)g_direction_fused.packed_ids->ptr,(float*)out->ptr,1u,&delta,cuda_decode_stream())==0;
}

/* Source can be the selected-expert cache, identified by the original file
 * offset. All factors are derived from that exact original source, never a
 * previously fused weight. Caller serializes the single native session. */
static int direction_fused_from_device(ds4_gpu_tensor *out,const unsigned char *source,
        const void *model_map,uint64_t offset,uint32_t type,uint32_t input,uint32_t output,
        const ds4_gpu_tensor *x,const ds4_gpu_tensor *pack,uint32_t layer,uint32_t site,uint32_t count) {
    const char *trace=getenv("DS4_DIRECTION_TRACE");
    if(trace && !strcmp(trace,"1")) {
        fprintf(stderr,"ds4: DIRECTION_ENTER layer=%u site=%u type=%u input=%u output=%u count=%u x_bytes=%llu out_bytes=%llu pack_bytes=%llu tiers=%d/%d/%d\n",
            layer,site,type,input,output,count,(unsigned long long)(x?x->bytes:0),
            (unsigned long long)(out?out->bytes:0),(unsigned long long)(pack?pack->bytes:0),
            out?ds4_tensor_device_idx(out):-99,x?ds4_tensor_device_idx(x):-99,pack?ds4_tensor_device_idx(pack):-99);
        fflush(stderr);
    }
    if(!out || !model_map || !x || !pack || !count || count>8 || layer>=64 || site>1 ||
        output!=5120u || input>32768u || !direction_quant_supported(type,input) ||
        x->bytes<(uint64_t)input*sizeof(float) || out->bytes<(uint64_t)output*sizeof(float) ||
        pack->bytes<((uint64_t)layer*2u+site+1u)*count*(output+1u)*sizeof(float) ||
        ds4_tensor_device_idx(out)!=ds4_tensor_device_idx(x) ||
        ds4_tensor_device_idx(out)!=ds4_tensor_device_idx(pack))return 0;
    direction_a_entry *entry=direction_target_entry(source,model_map,offset,type,input,output,
        pack,layer,site,count,ds4_tensor_device_idx(out));
    if(!entry)return 0;
    if(!entry->weight && !source)source=(const unsigned char *)cuda_resolve_weight_ptr(model_map,offset,
        direction_source_bytes(type,input,output),ds4_tensor_device_idx(out),"direction:fused-target:read-only");
    if(!entry->weight && !source){fprintf(stderr,"ds4: direction original target source unavailable layer=%u site=%u offset=%llu\n",layer,site,(unsigned long long)offset);return 0;}
    const uint64_t elements=(uint64_t)input*output,bytes=elements*sizeof(float);
    if(!entry->weight) {
        /* Fuse once per resident writer. Expert/source eviction never mutates this
         * independent target-derived tensor. The cache admits the full dense
         * writer working set instead of cycling a smaller LRU every token, and
         * keeps at least 256MiB free; evicted writers are re-fused from original W. */
        const uint64_t budget=ds4_direction_cache_memory::dense_capacity_bytes(),floor=256ull*1024u*1024u;
        uint64_t free_bytes=0,total_bytes=0;
        if(!direction_cache_free_bytes(free_bytes,total_bytes))return 0;
        if(trace && !strcmp(trace,"1"))fprintf(stderr,"ds4: DIRECTION_CACHE_MEMORY layer=%u site=%u measured_free=%llu measured_total=%llu needed=%llu floor=%llu resident=%llu budget=%llu\n",layer,site,(unsigned long long)free_bytes,(unsigned long long)total_bytes,(unsigned long long)bytes,(unsigned long long)floor,(unsigned long long)g_direction_fused.weight_bytes,(unsigned long long)budget);
        while(!ds4_direction_cache_memory::fits(g_direction_fused.weight_bytes,bytes,budget,free_bytes,floor)) {
            direction_a_entry *victim=NULL;
            for(unsigned i=0;i<DIRECTION_A_CACHE_COUNT;i++) {
                direction_a_entry *candidate=&g_direction_fused.cache[i];
                if(candidate!=entry && candidate->weight && (!victim || candidate->age<victim->age))victim=candidate;
            }
            if(!victim){fprintf(stderr,"ds4: direction dense cache has no evictable writer layer=%u site=%u measured_free=%llu needed=%llu floor=%llu resident=%llu\n",layer,site,(unsigned long long)free_bytes,(unsigned long long)bytes,(unsigned long long)floor,(unsigned long long)g_direction_fused.weight_bytes);return 0;}
            g_direction_fused.weight_bytes-=victim->weight->bytes;
            ds4_gpu_tensor_free(victim->weight);victim->weight=NULL;
            if(!direction_cache_free_bytes(free_bytes,total_bytes))return 0;
        }
        entry->weight=ds4_gpu_tensor_alloc(bytes);
        if(!entry->weight)return 0;
        g_direction_fused.weight_bytes+=bytes;
        if(!g_direction_fused.b)g_direction_fused.b=ds4_gpu_tensor_alloc(5120u*8u*sizeof(float));
        if(!g_direction_fused.status)g_direction_fused.status=ds4_gpu_tensor_alloc(sizeof(int));
        int invalid=0;
        if(!g_direction_fused.b || !g_direction_fused.status ||
            !ds4_gpu_tensor_write(g_direction_fused.status,0,&invalid,sizeof(invalid))) {
            g_direction_fused.weight_bytes-=entry->weight->bytes;
            ds4_gpu_tensor_free(entry->weight);entry->weight=NULL;return 0;
        }
        direction_target_b_kernel<<<(output+255u)/256u,256,0,cuda_decode_stream()>>>(
            (float *)g_direction_fused.b->ptr,(const float *)pack->ptr,layer,site,output,count);
        direction_target_dequant_kernel<<<(unsigned)((elements+255u)/256u),256,0,cuda_decode_stream()>>>(
            (float *)entry->weight->ptr,source,type,input,output);
        bool ok=cuda_ok(cudaGetLastError(),"direction target dequantization/B");
        /* Reuse the common LoRA fusion kernel with FP32 weight storage. */
        if(ok)output_lora_fuse_kernel<<<(unsigned)((elements+255u)/256u),256,0,cuda_decode_stream()>>>(
            (float *)entry->weight->ptr,(const float *)entry->a->ptr,
            (const float *)g_direction_fused.b->ptr,input,output,count,1.0f,(int *)g_direction_fused.status->ptr);
        ok=ok && cuda_ok(cudaGetLastError(),"direction target fusion") && ds4_gpu_synchronize() &&
            ds4_gpu_tensor_read(g_direction_fused.status,0,&invalid,sizeof(invalid)) && !invalid;
        if(!ok) {
            g_direction_fused.weight_bytes-=entry->weight->bytes;
            ds4_gpu_tensor_free(entry->weight);entry->weight=NULL;return 0;
        }
        ++g_direction_fused.fusions;
        const char *trace=getenv("DS4_DIRECTION_TRACE");
        if(trace && strcmp(trace,"1")==0) {
            fprintf(stderr,"ds4: DIRECTION_TARGET layer=%u site=%u offset=%llu type=%u input=%u output=%u count=%u\n",
                layer,site,(unsigned long long)offset,type,input,output,count);
            fflush(stderr);
        }
    } else ++g_direction_fused.hits;
    const float one=1,zero=0;
    return cublas_ok(cublasGemmEx(cuda_cublas_for_tier(ds4_tensor_device_idx(out)),
        CUBLAS_OP_T,CUBLAS_OP_N,(int)output,1,(int)input,&one,entry->weight->ptr,
        CUDA_R_32F,(int)input,x->ptr,CUDA_R_32F,(int)input,&zero,out->ptr,CUDA_R_32F,
        (int)output,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT),"direction fused FP32 writer");
}

extern "C" int ds4_gpu_direction_matmul(ds4_gpu_tensor *out,const void *model_map,
        uint64_t model_size,uint64_t offset,uint32_t type,uint32_t input,uint32_t output,
        const ds4_gpu_tensor *x,const ds4_gpu_tensor *pack,uint32_t layer,uint32_t site,uint32_t count) {
    if(!model_map || !direction_quant_supported(type,input) || !output)return 0;
    const uint64_t bytes=direction_source_bytes(type,input,output);
    if(offset>model_size || bytes>model_size-offset || !out)return 0;
    if(type==DIRECTION_PACKED_MXFP4) {
        if(input!=2304u || output!=5120u || !x || !pack || !count || count>8u || layer>=64u || site>1u ||
            x->bytes<(uint64_t)input*sizeof(float) || out->bytes<(uint64_t)output*sizeof(float) ||
            pack->bytes<((uint64_t)layer*2u+site+1u)*count*(output+1u)*sizeof(float))return 0;
        return direction_packed_control(out,model_map,offset,x,pack,layer,site,count);
    }
    return direction_fused_from_device(out,NULL,model_map,offset,type,input,output,
        x,pack,layer,site,count);
}
extern "C" int ds4_gpu_direction_packed_mixed_control(ds4_gpu_tensor *out,const void *model_map,
        uint64_t model_size,uint64_t offset,const ds4_gpu_tensor *x,const ds4_gpu_tensor *pack,
        const ds4_gpu_tensor *cpu_base,uint32_t layer,uint32_t site,uint32_t count) {
    const uint64_t bytes=direction_source_bytes(DIRECTION_PACKED_MXFP4,2304u,5120u);
    if(!out || !model_map || !x || !pack || !cpu_base || !count || count>8 || layer>=64 || site>1 ||
        offset>model_size || bytes>model_size-offset || x->bytes<2304u*sizeof(float) ||
        out->bytes<5120u*sizeof(float) || cpu_base->bytes<5120u*sizeof(float) ||
        pack->bytes<((uint64_t)layer*2u+site+1u)*count*5121u*sizeof(float))return 0;
    return direction_packed_control(out,model_map,offset,x,pack,layer,site,count,cpu_base);
}

/* Only active during the synchronous routed writer call. No request can inherit
 * a prior call's adapter. The pack itself belongs to the native session. */
static struct {
    const ds4_gpu_tensor *pack,*original_selected;
    uint32_t layer,count;
} g_direction_routed;
#endif
