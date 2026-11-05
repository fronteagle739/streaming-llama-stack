#pragma once
#include "ggml-dsv41-writer-bank.h"
#include "convert.cuh"
#include "quantize.cuh"
#include <array>

// All factors remain on the device for this immutable producer bank. Original
// GGUF bytes are only read on the GPU, once as each layer reaches the provider.
struct dsv41_writer_factors {float *a=nullptr;int input=0;bool ready=false;};
static float *dsv41_writer_pack=nullptr;
static std::array<dsv41_writer_factors,80> dsv41_writer_dense;
static std::array<dsv41_writer_factors,40> dsv41_writer_packed;
static uint64_t dsv41_writer_resident_bytes=0;
extern "C" uint64_t ggml_backend_cuda_dsv41_writer_resident_bytes(void) {return dsv41_writer_resident_bytes;}
static void dsv41_writer_upload(){
    const auto &bank=dsv41_writer_bank::current;
    if(dsv41_writer_pack||!bank.count)return;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dsv41_writer_pack),bank.pack.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dsv41_writer_pack,bank.pack.data(),bank.pack.size()*sizeof(float),cudaMemcpyHostToDevice));
    dsv41_writer_resident_bytes=bank.pack.size()*sizeof(float);
}
__global__ static void dsv41_writer_a_tile(float *a,const float *tile,const float *pack,int layer,int site,int rank,int input,int first,int rows){
    int col=int(blockIdx.x)*blockDim.x+threadIdx.x;if(col>=input)return;
    const float *vectors=pack+(size_t(layer)*2+site)*rank*5121;
    for(int d=0;d<rank;++d){float sum=0;
        for(int row=0;row<rows;++row)sum+=vectors[size_t(d)*5121+1+first+row]*tile[size_t(row)*input+col];
        size_t at=size_t(d)*input+col;a[at]=first?a[at]+sum:sum;
    }
}
static const float *dsv41_writer_dense_a(ggml_backend_cuda_context&ctx,const ggml_tensor*weight,int layer,int site){
    dsv41_writer_upload();auto &factor=dsv41_writer_dense[size_t(layer)*2+site];const int input=int(weight->ne[0]),rank=int(dsv41_writer_bank::current.count);
    if(factor.ready){GGML_ASSERT(factor.input==input);return factor.a;}
    GGML_ASSERT(ggml_is_contiguous(weight)&&weight->ne[1]==5120&&weight->ne[2]==1&&weight->ne[3]==1);
    auto convert=ggml_get_to_fp32_cuda(weight->type);GGML_ASSERT(convert&&"Writer target quant has no existing GPU dequantizer");
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&factor.a),size_t(rank)*input*sizeof(float)));
    factor.input=input;dsv41_writer_resident_bytes+=size_t(rank)*input*sizeof(float);
    ggml_cuda_pool_alloc<float> tile(ctx.pool(),size_t(64)*input);
    const size_t row_bytes=ggml_row_size(weight->type,input);
    for(int first=0;first<5120;first+=64){
        convert(static_cast<const char*>(weight->data)+size_t(first)*row_bytes,tile.get(),int64_t(64)*input,ctx.stream());
        dsv41_writer_a_tile<<<(input+127)/128,128,0,ctx.stream()>>>(factor.a,tile.get(),dsv41_writer_pack,layer,site,rank,input,first,64);
        CUDA_CHECK(cudaGetLastError());
    }
    factor.ready=true;return factor.a;
}
#include "dsv41-writer-packed-kernel.cuh"
static const float*dsv41_writer_packed_a(ggml_backend_cuda_context&ctx,const ggml_tensor*weight,int layer,bool slots=false){
    dsv41_writer_upload();auto &factor=dsv41_writer_packed[layer];const int rank=int(dsv41_writer_bank::current.count);
    if(factor.ready&&!slots)return factor.a;
    GGML_ASSERT(weight->type==GGML_TYPE_MXFP4&&weight->ne[0]==2304&&weight->ne[1]==5120&&
        (slots?(weight->ne[2]>0&&weight->ne[2]<=384):weight->ne[2]==384)&&ggml_is_contiguous(weight));
    if(!factor.a){const size_t bytes=size_t(384)*rank*2304*sizeof(float);CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&factor.a),bytes));dsv41_writer_resident_bytes+=bytes;}
    const auto *values=static_cast<const uint8_t*>(weight->data),*scales=values+size_t(slots?1:384)*5120*2304/2;
    dsv41_writer_packed_a_kernel<<<dim3((2304+31)/32,unsigned(weight->ne[2])),dim3(32,8),0,ctx.stream()>>>(factor.a,values,scales,dsv41_writer_pack,layer,rank,slots);CUDA_CHECK(cudaGetLastError());
    // Cache slots can be replaced at every layer/append; a cached projection
    // would silently steer a later expert using stale weights.
    factor.input=2304;factor.ready=!slots;return factor.a;
}
__device__ static float dsv41_writer_input(const void*input,int type,size_t i){
    return type==GGML_TYPE_F32?static_cast<const float*>(input)[i]:type==GGML_TYPE_F16?__half2float(static_cast<const half*>(input)[i]):__bfloat162float(static_cast<const nv_bfloat16*>(input)[i]);
}
__global__ static void dsv41_writer_dot(const float*a,const void*input,const block_q8_1_mmq*q,int type,int layout,int mode,float*dots,int width,int rank,int tokens){
    const int token=int(blockIdx.x),d=int(blockIdx.y);float sum=0;
    for(int k=threadIdx.x;k<width;k+=blockDim.x){float x;
        if(q){const auto &b=q[size_t(token)*(width/128)+k/128];const int at=k%128;
            float scale=layout==MMQ_Q8_1_DS_LAYOUT_D4?b.d4[at/32]:layout==MMQ_Q8_1_DS_LAYOUT_DS4?__low2float(b.ds4[at/32]):__half2float(b.d2s6[at/64]);x=scale*b.qs[at];
        }else {x=dsv41_writer_input(input,type,size_t(token)*width+k);if(mode==2)x=__half2float(__float2half_rn(x));}
        sum+=a[size_t(d)*width+k]*x;
    }
    __shared__ float partial[128];partial[threadIdx.x]=sum;__syncthreads();
    for(int step=64;step;step/=2){if(threadIdx.x<step)partial[threadIdx.x]+=partial[threadIdx.x+step];__syncthreads();}
    if(!threadIdx.x)dots[size_t(token)*rank+d]=partial[0];
}
__global__ static void dsv41_writer_add(const float*base,const float*dots,const float*pack,float*out,int layer,int site,int rank,int tokens){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(tokens)*5120)return;const int row=int(i%5120),token=int(i/5120);
    const float*vectors=pack+(size_t(layer)*2+site)*rank*5121;float value=base[i];
    for(int d=0;d<rank;++d)value+=(-vectors[size_t(d)*5121]*vectors[size_t(d)*5121+1+row])*dots[size_t(token)*rank+d];out[i]=value;
}
static void ggml_cuda_op_dsv41_writer(ggml_backend_cuda_context&ctx,ggml_tensor*t){
    GGML_ASSERT(dsv41_writer_bank::valid(t));const int layer=t->op_params[1],site=t->op_params[2],rank=int(dsv41_writer_bank::current.count),tokens=int(t->ne[1]);
    GGML_ASSERT(dsv41_writer_bank::current.active(layer,site));auto *weight=t->src[1],*input=t->src[2];
    GGML_ASSERT(input->type==GGML_TYPE_F32||input->type==GGML_TYPE_F16||input->type==GGML_TYPE_BF16);
    auto*a=dsv41_writer_dense_a(ctx,weight,layer,site);const int width=int(weight->ne[0]);
    ggml_cuda_pool_alloc<float> dots(ctx.pool(),size_t(tokens)*rank);
    ggml_cuda_pool_alloc<block_q8_1_mmq> quantized(ctx.pool());const block_q8_1_mmq*q=nullptr;int layout=0;
    if(t->op_params[3]==1){GGML_ASSERT(input->type==GGML_TYPE_F32&&width%128==0);q=quantized.alloc(size_t(tokens)*width/128);layout=int(mmq_get_q8_1_ds_layout(weight->type));
        quantize_mmq_q8_1_cuda(static_cast<const float*>(input->data),nullptr,const_cast<block_q8_1_mmq*>(q),weight->type,width,width,size_t(width)*tokens,size_t(width)*tokens,width,tokens,1,1,ctx.stream());}
    dsv41_writer_dot<<<dim3(tokens,rank),128,0,ctx.stream()>>>(a,input->data,q,int(input->type),layout,t->op_params[3],dots.get(),width,rank,tokens);
    dsv41_writer_add<<<unsigned((size_t(tokens)*5120+255)/256),256,0,ctx.stream()>>>(static_cast<const float*>(t->src[0]->data),dots.get(),dsv41_writer_pack,static_cast<float*>(t->data),layer,site,rank,tokens);CUDA_CHECK(cudaGetLastError());
}
