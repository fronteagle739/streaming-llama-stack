#include "dsv41-hc-collapse.cuh"
#include "dsv4-act-qat-kernels.cuh"
#include "../ggml-dsv41-ops.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
static void check(cudaError_t e) { if(e!=cudaSuccess) {std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(2);} }
// Separate launches model the original seven HC arithmetic nodes.
__global__ static void product(const float*r,const float*p,float*out,int w,int n,int c) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<w*n)out[i]=__fmul_rn(r[(i/w*4+c)*w+i%w],p[i/w*4+c]);
}
__global__ static void add(float*a,const float*b,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]=__fadd_rn(a[i],b[i]);
}
static void equal(const std::vector<float>&a,const std::vector<float>&b,const char*name) {
    for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],4)) {
        std::fprintf(stderr,"FAIL %s at %zu: %.9g %.9g\n",name,i,a[i],b[i]);std::exit(3);
    }
}
int main() {
    int cases=0;
    for(int w: {1,31,64,5120})for(int n: {1,3,65,4096}) {
        const int size=w*n;
        std::vector<float>r(size*4),p(n*4),a(size),b(size);
        for(size_t i=0;i<r.size();++i)r[i]=float(int((i*12347+99)%65521)-32760)/8192.f;
        for(size_t i=0;i<p.size();++i)p[i]=float(int((i*13+7)%127)-63)/32.f;
        if(r.size()>=16) {r[0]=-0.f;r[1]=0.f;r[2]=std::ldexp(1.f,-140);r[3]=std::ldexp(-1.f,-140);r[4]=1.00390625f;r[5]=1.01171875f;}
        float *dr,*dp,*da,*db,*temp;
        check(cudaMalloc(&dr,r.size()*4));check(cudaMalloc(&dp,p.size()*4));
        check(cudaMalloc(&da,size*4));check(cudaMalloc(&db,size*4));check(cudaMalloc(&temp,size*4));
        check(cudaMemcpy(dr,r.data(),r.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(dp,p.data(),p.size()*4,cudaMemcpyHostToDevice));
        product<<<(size+255)/256,256>>>(dr,dp,da,w,n,0);
        for(int c=1;c<4;++c) {product<<<(size+255)/256,256>>>(dr,dp,temp,w,n,c);add<<<(size+255)/256,256>>>(da,temp,size);}
        dsv41_act_qat_kernel<256,0><<<(size+255)/256,256>>>(da,da,size,1);
        dsv41_hc_collapse_kernel<<<(size+255)/256,256>>>(dr,dp,db,w,n);
        check(cudaMemcpy(a.data(),da,size*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),db,size*4,cudaMemcpyDeviceToHost));equal(a,b,"collapse");
        // Independent CPU oracle uses the portable quantization helpers.
        for(int i=0;i<size;++i) {float v=ggml_dsv41_rope_product(r[(i/w*4)*w+i%w],p[i/w*4]);
            for(int c=1;c<4;++c)v=ggml_dsv41_add_rn(v,ggml_dsv41_rope_product(r[(i/w*4+c)*w+i%w],p[i/w*4+c]));
            v=ggml_dsv41_bf16(v);if(std::memcmp(&v,&b[i],4))return 4;}
        for(auto ptr:{dr,dp,da,db,temp})check(cudaFree(ptr));++cases;
    }
    for(int width:{32,64,512,5120})for(int rows:{1,3,257,4096})for(int format:{1,2,3}) {
        const int size=width*rows,block=format==3?16:32;
        std::vector<float>src(size),a(size),b(size);
        for(int i=0;i<size;++i)src[i]=std::ldexp(float(int((uint64_t(i)*127+89)%65521)-32760),int(i%23)-20);
        if(size>=32){for(int i=0;i<16;++i)src[i]=0.f;src[16]=-0.f;src[17]=std::ldexp(1.f,-140);}
        float *ds,*da,*db;check(cudaMalloc(&ds,size*4));check(cudaMalloc(&da,size*4));check(cudaMalloc(&db,size*4));
        check(cudaMemcpy(ds,src.data(),size*4,cudaMemcpyHostToDevice));
        dim3 grid(width/block,rows);
        if(format==1){dsv41_act_qat_kernel<32,1><<<grid,32>>>(ds,da,width,rows);dsv41_act_qat_tiled_kernel<32,1><<<(size+255)/256,256>>>(ds,db,size);}
        if(format==2){dsv41_act_qat_kernel<32,2><<<grid,32>>>(ds,da,width,rows);dsv41_act_qat_tiled_kernel<32,2><<<(size+255)/256,256>>>(ds,db,size);}
        if(format==3){dsv41_act_qat_kernel<16,3><<<grid,16>>>(ds,da,width,rows);dsv41_act_qat_tiled_kernel<16,3><<<(size+255)/256,256>>>(ds,db,size);}
        check(cudaMemcpy(a.data(),da,size*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),db,size*4,cudaMemcpyDeviceToHost));equal(a,b,"qat");
        // The in-place case is required by the allocator's last-reader reuse.
        if(format==1)dsv41_act_qat_tiled_kernel<32,1><<<(size+255)/256,256>>>(ds,ds,size);
        if(format==2)dsv41_act_qat_tiled_kernel<32,2><<<(size+255)/256,256>>>(ds,ds,size);
        if(format==3)dsv41_act_qat_tiled_kernel<16,3><<<(size+255)/256,256>>>(ds,ds,size);
        check(cudaMemcpy(b.data(),ds,size*4,cudaMemcpyDeviceToHost));equal(a,b,"qat inplace");
        for(auto ptr:{ds,da,db})check(cudaFree(ptr));++cases;
    }
    check(cudaGetLastError());std::printf("PASS %d full-output bitwise cases, independent HC CPU oracle, QAT in-place\n",cases);return 0;
}
