#include "dsv41-hc-channels.cuh"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#define OK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);} }while(0)
__global__ void reference(float*out,const float*b,const float*r,const float*m,int w,int n,bool bf){
 uint64_t g=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(g>=uint64_t(w)*4*n)return;
 int d=g%w,h=g/w%4,t=g/w/4;float a=__fmul_rn(b[t*w+d],m[t*20+h]);
 for(int s=0;s<4;s++)a=__fadd_rn(a,__fmul_rn(m[t*20+4+s*4+h],r[(uint64_t(t)*4+s)*w+d]));
 out[g]=bf?ggml_dsv41_bf16(a):a;
}
static uint32_t seed=71;
float sample(){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return float(int(seed%200001)-100000)/32768.f;}
int main(){
 for(int w:{1,31,128,5120})for(int n:{1,17,4096})for(bool bf:{false,true}){
  size_t count=size_t(w)*4*n;std::vector<float> r(count),b(size_t(w)*n),m(size_t(n)*20);
  for(auto&v:r)v=sample();for(auto&v:b)v=sample();for(auto&v:m)v=sample();
  // Include cancellation, subnormals, BF16 halfway cases and signed zero.
  for(size_t i=0;i<r.size();i+=127)r[i]=(i&1)?-0.f:0.f;
  for(size_t i=1;i<r.size();i+=257)r[i]=ggml_dsv41_bits_float(0x00000001u);
  for(size_t i=0;i<b.size();i+=113)b[i]=ggml_dsv41_bits_float(0x3f808000u);
  float *dr,*db,*dm,*out,*ref;OK(cudaMalloc(&dr,r.size()*4));OK(cudaMalloc(&db,b.size()*4));OK(cudaMalloc(&dm,m.size()*4));OK(cudaMalloc(&out,count*4));OK(cudaMalloc(&ref,count*4));
  OK(cudaMemcpy(dr,r.data(),r.size()*4,cudaMemcpyHostToDevice));OK(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));OK(cudaMemcpy(dm,m.data(),m.size()*4,cudaMemcpyHostToDevice));
  reference<<<unsigned((count+255)/256),256>>>(ref,db,dr,dm,w,n,bf);
  if(bf)dsv41_hc_channels_kernel<true><<<unsigned((size_t(w)*n+255)/256),256>>>(out,db,dr,dm,w,n);
  else dsv41_hc_channels_kernel<false><<<unsigned((size_t(w)*n+255)/256),256>>>(out,db,dr,dm,w,n);
  OK(cudaDeviceSynchronize());std::vector<float>a(count),z(count);OK(cudaMemcpy(a.data(),out,count*4,cudaMemcpyDeviceToHost));OK(cudaMemcpy(z.data(),ref,count*4,cudaMemcpyDeviceToHost));
  if(memcmp(a.data(),z.data(),count*4)){fprintf(stderr,"MISMATCH w=%d n=%d bf=%d\n",w,n,bf);return 2;}
  // Separate CPU oracle samples across rows and channels without sharing the
  // candidate's channel grouping. Volatile stores retain each F32 boundary.
  for(size_t g=0;g<count;g+=4093){
   size_t d=g%w,h=g/w%4,t=g/w/4;
   volatile float acc=b[t*w+d]*m[t*20+h];
   for(int s=0;s<4;s++){volatile float p=m[t*20+4+s*4+h]*r[(t*4+s)*w+d];acc=acc+p;}
   float expected=bf?ggml_dsv41_bf16(acc):float(acc);
   if(ggml_dsv41_float_bits(expected)!=ggml_dsv41_float_bits(a[g])){fprintf(stderr,"CPU ORACLE mismatch at %zu\n",g);return 3;}
  }
  float ms[2];cudaEvent_t start,stop;OK(cudaEventCreate(&start));OK(cudaEventCreate(&stop));
  for(int v=0;v<2;v++){
   OK(cudaEventRecord(start));for(int j=0;j<10;j++){
    if(v==0)reference<<<unsigned((count+255)/256),256>>>(ref,db,dr,dm,w,n,bf);
    else if(bf)dsv41_hc_channels_kernel<true><<<unsigned((size_t(w)*n+255)/256),256>>>(out,db,dr,dm,w,n);
    else dsv41_hc_channels_kernel<false><<<unsigned((size_t(w)*n+255)/256),256>>>(out,db,dr,dm,w,n);
   }OK(cudaEventRecord(stop));OK(cudaEventSynchronize(stop));OK(cudaEventElapsedTime(&ms[v],start,stop));
  }
  printf("PASS w=%d rows=%d bf16=%d values=%zu old_ms=%.6f candidate_ms=%.6f\n",w,n,int(bf),count,ms[0]/10,ms[1]/10);fflush(stdout);
  OK(cudaEventDestroy(start));OK(cudaEventDestroy(stop));for(float*p:{dr,db,dm,out,ref})OK(cudaFree(p));
 }
}
