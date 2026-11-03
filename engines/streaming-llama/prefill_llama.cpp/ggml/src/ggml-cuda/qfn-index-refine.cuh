// SPDX-License-Identifier: MIT
#pragma once

// Retain the TC bulk scores, but restore the scalar warp's arithmetic for
// candidates whose rounding interval overlaps the causal top-k boundary.
// No host readback or pool truncation. Positions are explicit graph inputs.
static __global__ void qfn_index_visible_limit(const int32_t *positions,int nt,int np,int ratio,int *limit){
    const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nt)return;
    const int64_t n=(int64_t(positions[t])+1)/ratio;
    if(positions[t]<0 || n>np){printf("QFN_INDEX_REFINE invalid causal position\n");asm volatile("trap;");return;}
    atomicMax(limit,int(n));
}
template<typename K>
static __global__ void qfn_index_key_norm(const char *keys, unsigned long long *maximum,
        int np, const int *limit, uint64_t kp, uint64_t ks) {
    const int64_t p=int64_t(blockIdx.x)*4+threadIdx.x/32;const int lane=threadIdx.x%32,s=blockIdx.y;
    if(p>=np || p>=*limit)return;
    const K *row=(const K *)(keys+uint64_t(s)*ks+uint64_t(p)*kp);
    double sum=0;
    for(int j=0;j<4;++j){const double x=ggml_cuda_cast<float>(row[lane+j*32]);sum+=x*x;}
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(!lane)atomicMax(maximum+s,__double_as_longlong(sqrt(sum)*1.0001));
}

template<typename K>
static __global__ void qfn_index_refine_kernel(float *scores,const char *queries,
        const char *keys,const char *weights,const int32_t *positions,const unsigned long long *knorm,
        int np,int nt,int top,int ratio,uint64_t qh,uint64_t qt,uint64_t qs,
        uint64_t kp,uint64_t ks,uint64_t wt,uint64_t ws,float se,float sh) {
    const int t=blockIdx.x,s=blockIdx.y,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const int64_t visible=(int64_t(positions[t])+1)/ratio;
    if(positions[t]<0 || visible>np){if(!tid)printf("QFN_INDEX_REFINE invalid causal position\n");asm volatile("trap;");return;}
    const int nv=int(visible);
    if(nv<=top)return; // selector includes every visible pool
    float *row=scores+(uint64_t(s)*nt+t)*np;
    __shared__ float q[512],w[4];
    __shared__ double bound;
    __shared__ uint32_t hist[256],threshold,need;
    for(int i=tid;i<512;i+=256)q[i]=*(const float *)(queries+uint64_t(s)*qs+uint64_t(t)*qt+uint64_t(i/128)*qh+(i%128)*4);
    if(tid<4)w[tid]=*(const float *)(weights+uint64_t(s)*ws+uint64_t(t)*wt+tid*4);
    if(!tid){threshold=0;need=top;}
    __syncthreads();
    if(!tid){
        double norm=0;
        for(int h=0;h<4;++h){double v=0;for(int d=0;d<128;++d){const double x=q[h*128+d];v+=x*x;}norm+=sqrt(v)*fabs(double(w[h]));}
        // Cauchy-Schwarz bounds sum |q*k|. 1e-4 exceeds the FP32
        // gamma384 TC accumulation, scalar reduction and 3xTF32 residual budgets.
        // An approximate kth score is at most E from the true kth score;
        // therefore only approximate scores within 2E can change membership.
        bound=2.0e-4*norm*__longlong_as_double(knorm[s])*fabs(double(se))*fabs(double(sh))+1.0e-20;
        if(!isfinite(bound)){printf("QFN_INDEX_REFINE nonfinite error envelope\n");asm volatile("trap;");}
    }
    for(int pass=0;pass<4;++pass){
        hist[tid]=0;__syncthreads();
        const int shift=24-8*pass;
        const uint32_t mask=pass?0xffffffffu<<(shift+8):0,prefix=threshold;
        for(int p=tid;p<nv;p+=256){const uint32_t key=row[p]>0?__float_as_uint(row[p]):0;
            if((key&mask)==prefix)atomicAdd(hist+((key>>shift)&255),1u);}
        __syncthreads();
        if(!tid){uint32_t n=need;for(int d=255;d>=0;--d){if(hist[d]>=n){threshold=prefix|(uint32_t(d)<<shift);break;}n-=hist[d];}need=n;}
        __syncthreads();
    }
    const float cutoff=__uint_as_float(threshold);
    for(int p=warp;p<nv;p+=8){
        const float approx=fmaxf(row[p],0.f);
        if(fabs(double(approx)-double(cutoff))>bound)continue;
        const K *kr=(const K *)(keys+uint64_t(s)*ks+uint64_t(p)*kp);
        float kreg[4];for(int j=0;j<4;++j)kreg[j]=ggml_cuda_cast<float>(kr[lane+j*32]);
        float result=0;
        for(int h=0;h<4;++h){float part=0;
            // Same lane+j*32 products, FMA policy, shuffle tree and ordered
            // head sum as lightning_indexer_kernel. No dot reorder here.
            for(int j=0;j<4;++j)part+=q[h*128+lane+j*32]*kreg[j];
            part+=__shfl_down_sync(0xffffffff,part,16);part+=__shfl_down_sync(0xffffffff,part,8);
            part+=__shfl_down_sync(0xffffffff,part,4);part+=__shfl_down_sync(0xffffffff,part,2);part+=__shfl_down_sync(0xffffffff,part,1);
            if(!lane)result+=fmaxf(part*se,0.f)*w[h];
        }
        if(!lane)row[p]=result*sh;
    }
}

static void qfn_index_refine(ggml_backend_cuda_context &ctx,ggml_tensor *t){
    const auto *q=t->src[0],*k=t->src[1],*w=t->src[2],*pos=t->src[3];
    const int np=int(k->ne[2]),nt=int(q->ne[2]),ns=int(q->ne[3]);
    ggml_cuda_pool_alloc<unsigned long long> norms(ctx.pool(),ns);
    ggml_cuda_pool_alloc<int> visible(ctx.pool(),1);
    const auto stream=ctx.stream();CUDA_CHECK(cudaMemsetAsync(norms.get(),0,ns*8,stream));
    CUDA_CHECK(cudaMemsetAsync(visible.get(),0,4,stream));
    qfn_index_visible_limit<<<(nt+255)/256,256,0,stream>>>((const int32_t *)pos->data,nt,np,t->op_params[5],visible.get());
    if(k->type==GGML_TYPE_F16)qfn_index_key_norm<half><<<dim3(unsigned((int64_t(np)+3)/4),ns),128,0,stream>>>((const char *)k->data,norms.get(),np,visible.get(),k->nb[2],k->nb[3]);
    else qfn_index_key_norm<float><<<dim3(unsigned((int64_t(np)+3)/4),ns),128,0,stream>>>((const char *)k->data,norms.get(),np,visible.get(),k->nb[2],k->nb[3]);
    float se,sh;memcpy(&se,t->op_params+2,4);memcpy(&sh,t->op_params+3,4);
#define QFN_REFINE_ARGS (float *)t->data,(const char *)q->data,(const char *)k->data,(const char *)w->data,(const int32_t *)pos->data,norms.get(),np,nt,t->op_params[4],t->op_params[5],q->nb[1],q->nb[2],q->nb[3],k->nb[2],k->nb[3],w->nb[1],w->nb[3],se,sh
    if(k->type==GGML_TYPE_F16)qfn_index_refine_kernel<half><<<dim3(nt,ns),256,0,stream>>>(QFN_REFINE_ARGS);
    else qfn_index_refine_kernel<float><<<dim3(nt,ns),256,0,stream>>>(QFN_REFINE_ARGS);
#undef QFN_REFINE_ARGS
    CUDA_CHECK(cudaGetLastError());
}
