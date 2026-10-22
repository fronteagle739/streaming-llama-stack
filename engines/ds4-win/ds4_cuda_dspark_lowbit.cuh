/* Resident draft-only Q2_K/Q3_K adapters. Existing gate/down/sum math,
 * clamp and original gate probabilities; no target/drafter replacement. */
#include "ds4_dspark_q3.h"
template<unsigned Type>
__device__ static float dspark_lowbit_dot(const char *block,const float *x) {
    if constexpr(Type==10)return dev_q2_K_dot_f32((const cuda_block_q2_K *)block,x,1);
    else {
        __half half;memcpy(&half,block+108,2);const float d=__half2float(half);
        float sum=0;
        for(unsigned i=0;i<256;i++){int scale=0;int q=ds4_dspark_q3_value((const uint8_t *)block,i,&scale);sum+=d*(float)scale*(float)q*x[i];}
        return sum;
    }
}
template<unsigned Type>
__global__ static void dspark_lowbit_gate_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const char *gr = (const char *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const char *ur = (const char *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dspark_lowbit_dot<Type>(gr+(uint64_t)b*(Type==10?84u:110u),xr+(uint64_t)b*CUDA_QK_K);
        up += dspark_lowbit_dot<Type>(ur+(uint64_t)b*(Type==10?84u:110u),xr+(uint64_t)b*CUDA_QK_K);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

template<unsigned Type>
__global__ static void dspark_lowbit_down_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t nb = expert_mid_dim / CUDA_QK_K;
    const char *wr = (const char *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const float *xr = mid + (uint64_t)pair * expert_mid_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) acc += dspark_lowbit_dot<Type>(wr+(uint64_t)b*(Type==10?84u:110u),xr+(uint64_t)b*CUDA_QK_K);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}


static int dspark_lowbit_launch(ds4_gpu_tensor *out,ds4_gpu_tensor *gate,ds4_gpu_tensor *up,
    ds4_gpu_tensor *mid,ds4_gpu_tensor *down,const void *map,uint64_t size,
    uint64_t go,uint64_t uo,uint64_t doff,uint32_t type,uint64_t geb,uint64_t grb,
    uint64_t deb,uint64_t drb,uint32_t in,uint32_t hidden,uint32_t output,
    const ds4_gpu_tensor *selected,const ds4_gpu_tensor *weights,uint32_t experts,
    uint32_t used,float clamp,const ds4_gpu_tensor *x,uint32_t rows) {
    if(!out || !gate || !up || !mid || !down || !selected || !weights || !x ||
       experts!=128 || used!=3 || !rows || rows>16 || (type!=10 && type!=11) ||
       map!=g_support_host_base)return 0;
    const int tier=ds4_tensor_device_idx(out);
    const char *gw=(const char *)cuda_resolve_weight_ptr(map,go,(uint64_t)experts*geb,tier,"draft lowbit gate");
    const char *uw=(const char *)cuda_resolve_weight_ptr(map,uo,(uint64_t)experts*geb,tier,"draft lowbit up");
    const char *dw=(const char *)cuda_resolve_weight_ptr(map,doff,(uint64_t)experts*deb,tier,"draft lowbit down");
    if(!gw || !uw || !dw)return 0;
    dim3 gate_grid(hidden,rows*used),down_grid(output,rows*used);
#define DSPARK_LOWBITS(Type) \
    dspark_lowbit_gate_kernel<Type><<<gate_grid,256,0,cuda_decode_stream()>>>( \
        (float *)gate->ptr,(float *)up->ptr,(float *)mid->ptr,gw,uw,(const float *)x->ptr, \
        (const int32_t *)selected->ptr,(const float *)weights->ptr,geb,grb,in,hidden,used,clamp); \
    dspark_lowbit_down_kernel<Type><<<down_grid,256,0,cuda_decode_stream()>>>( \
        (float *)down->ptr,dw,(const float *)mid->ptr,(const int32_t *)selected->ptr,deb,drb,hidden,output,used)
    if(type==10){DSPARK_LOWBITS(10);}else{DSPARK_LOWBITS(11);}
#undef DSPARK_LOWBITS
    if(!cuda_ok(cudaGetLastError(),"draft lowbit expert projections"))return 0;
    const uint64_t n=(uint64_t)rows*output;
    moe_sum_kernel<<<(n+255)/256,256,0,cuda_decode_stream()>>>((float *)out->ptr,(const float *)down->ptr,output,used,rows);
    return cuda_ok(cudaGetLastError(),"draft lowbit sum");
}
