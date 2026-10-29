#include "dispatch-counter.cuh"
// Host-only reuse of the shared MUL_MAT_ID sort/gather/scatter fallback,
// existing Q4_K -> F16 conversion, and cuBLAS F16/F16/F32 GEMMs.
// No CUDA kernel bodies or full dequantized expert bank.
#include "dsv41-expert-cublas.cuh"
#include "ggml-dsv41-expert-cublas.h"
#include "convert.cuh"
#include "getrows.cuh"
#include <array>
#include <atomic>
#include <vector>
static ggml_cuda_dispatch_counter expert_cublas_queued{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_expert_cublas_queued(void){return expert_cublas_queued.load();}
void ggml_cuda_op_dsv41_expert_cublas(ggml_backend_cuda_context&ctx,ggml_tensor*dst){
 GGML_ASSERT(ggml_dsv41_expert_cublas_requested(dst)&&ggml_dsv41_expert_cublas_valid(dst));
 const auto*w=dst->src[0];const auto*x=dst->src[1];const auto*ids=dst->src[2];
 const int k=int(w->ne[0]),n=int(w->ne[1]),tokens=int(x->ne[2]),slots=tokens*6,channels=int(x->ne[1]);
 const size_t xb=size_t(slots)*k*sizeof(half),yb=size_t(slots)*n*sizeof(float),wb=size_t(k)*n*sizeof(half);
 GGML_ASSERT(xb+yb+wb+size_t(slots)*8<=ggml_dsv41_expert_cublas_workspace(tokens));
 auto stream=ctx.stream();std::vector<int32_t> host_ids(slots),maps(2*slots);std::array<int32_t,385> bounds{};
 CUDA_CHECK(cudaMemcpyAsync(host_ids.data(),ids->data,size_t(slots)*4,cudaMemcpyDeviceToHost,stream));
 CUDA_CHECK(cudaStreamSynchronize(stream));
 for(int t=0;t<tokens;++t)for(int rank=0;rank<6;++rank){const int e=host_ids[t*6+rank];
  GGML_ASSERT(e>=0&&e<384);for(int p=0;p<rank;++p)GGML_ASSERT(host_ids[t*6+p]!=e);++bounds[e+1];}
 for(int e=1;e<=384;++e)bounds[e]+=bounds[e-1];auto cursor=bounds;
 for(int t=0;t<tokens;++t)for(int rank=0;rank<6;++rank){const int original=t*6+rank,e=host_ids[original],sorted=cursor[e]++;
  maps[sorted]=t*channels+rank%channels;maps[slots+original]=sorted;}
 ggml_cuda_pool_alloc<int32_t> device_maps(ctx.pool(),size_t(2)*slots);
 ggml_cuda_pool_alloc<half> sorted_x(ctx.pool(),size_t(slots)*k),one_weight(ctx.pool(),size_t(k)*n);
 ggml_cuda_pool_alloc<float> sorted_y(ctx.pool(),size_t(slots)*n);
 CUDA_CHECK(cudaMemcpyAsync(device_maps.get(),maps.data(),size_t(slots)*8,cudaMemcpyHostToDevice,stream));
 // The host maps remain live through the explicit stream fence below.
 get_rows_cuda(x->data,GGML_TYPE_F32,device_maps.get(),sorted_x.get(),GGML_TYPE_F16,
  k,x->nb[1],x->nb[2],x->nb[3],slots,1,1,4,size_t(slots)*4,size_t(slots)*4,
  size_t(k)*2,size_t(slots)*k*2,size_t(slots)*k*2,stream);
 const auto convert=ggml_get_to_fp16_cuda(GGML_TYPE_Q4_K);GGML_ASSERT(convert);
 CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(),stream));const float alpha=1.f,beta=0.f;
 for(int e=0;e<384;++e){const int count=bounds[e+1]-bounds[e];if(!count)continue;
  convert((const char*)w->data+size_t(e)*w->nb[2],one_weight.get(),int64_t(k)*n,stream);
  CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(),CUBLAS_OP_T,CUBLAS_OP_N,n,count,k,
   &alpha,one_weight.get(),CUDA_R_16F,k,sorted_x.get()+size_t(bounds[e])*k,CUDA_R_16F,k,
   &beta,sorted_y.get()+size_t(bounds[e])*n,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 }
 get_rows_cuda(sorted_y.get(),GGML_TYPE_F32,device_maps.get()+slots,dst->data,GGML_TYPE_F32,
  n,size_t(n)*4,size_t(slots)*n*4,size_t(slots)*n*4,slots,1,1,4,size_t(slots)*4,size_t(slots)*4,
  dst->nb[1],dst->nb[2],dst->nb[3],stream);
 CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
 expert_cublas_queued.fetch_add(1);
}
