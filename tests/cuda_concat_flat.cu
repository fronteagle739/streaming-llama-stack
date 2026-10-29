#include "concat-flat.cuh"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <array>
#define OK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);} }while(0)
__global__ void old_concat(const float*x,const float*y,float*out,int a,int n0,int n1,int n2,int dim){
 int d=blockIdx.x*blockDim.x+threadIdx.x;if(d>=n0)return;
 int64_t i=d+int64_t(blockIdx.y)*n0+int64_t(blockIdx.z)*n0*n1;
 int64_t inner=dim==0?1:dim==1?n0:int64_t(n0)*n1;
 int64_t xs=inner*a,span=inner*(dim==0?n0:dim==1?n1:n2),o=i/span,j=i%span;
 out[i]=j<xs?x[o*xs+j]:y[o*(span-xs)+j-xs];
}
int main(){
 uint32_t seed=17;
 for(int dim:{0,1,2})for(int size:{3,64,512})for(int extra:{0,1,size})for(int offset:{0,1}){
  int n[3]={size,7,17};n[dim]=size*2+extra;int a=size;
  size_t inner=1;for(int i=0;i<dim;i++)inner*=n[i];
  size_t xs=inner*a,ys=inner*(size+extra),span=xs+ys,count=size_t(n[0])*n[1]*n[2];
  size_t xc=count/span*xs,yc=count/span*ys;
  std::vector<uint32_t>x(xc),y(yc),expected(count),actual(count);
  for(auto&v:x){seed=seed*1664525u+1013904223u;v=seed;}
  for(auto&v:y){seed=seed*1664525u+1013904223u;v=seed;}
  if(xc>=4){x[0]=0x80000000u;x[1]=0x7fc12345u;x[2]=0x7f800000u;x[3]=0x00000001u;}
  for(size_t i=0;i<count;i++){size_t o=i/span,j=i%span;expected[i]=j<xs?x[o*xs+j]:y[o*ys+j-xs];}
  float *dx,*dy,*dout;OK(cudaMalloc(&dx,(xc+1)*4));OK(cudaMalloc(&dy,(yc+1)*4));OK(cudaMalloc(&dout,(count+1)*4));
  OK(cudaMemcpy(dx+offset,x.data(),xc*4,cudaMemcpyHostToDevice));OK(cudaMemcpy(dy+offset,y.data(),yc*4,cudaMemcpyHostToDevice));
  bool vec=xs%4==0&&ys%4==0&&offset==0;
  if(vec)concat_f32_flat<true><<<unsigned((count+1023)/1024),256>>>(dx+offset,dy+offset,dout+offset,xs,ys,count);
  else concat_f32_flat<false><<<unsigned((count+255)/256),256>>>(dx+offset,dy+offset,dout+offset,xs,ys,count);
  OK(cudaDeviceSynchronize());OK(cudaMemcpy(actual.data(),dout+offset,count*4,cudaMemcpyDeviceToHost));
  if(memcmp(expected.data(),actual.data(),count*4)){fprintf(stderr,"MISMATCH dim=%d size=%d offset=%d\n",dim,size,offset);return 2;}
  printf("PASS concat dim=%d size=%d extra=%d offset=%d vector=%d values=%zu\n",dim,size,extra,offset,int(vec),count);
  OK(cudaFree(dx));OK(cudaFree(dy));OK(cudaFree(dout));
 }
 // Actual short-row geometry, original grid versus flattened vector grid.
 for(int dim:{0,2}){
  const int n0=dim==0?128:512,n1=dim==0?64:1,n2=4096,a=dim==0?64:2048;
  size_t count=size_t(n0)*n1*n2,xs=dim==0?64:size_t(n0)*n1*a,ys=xs;
  float*x,*y,*o;OK(cudaMalloc(&x,count*2));OK(cudaMalloc(&y,count*2));OK(cudaMalloc(&o,count*4));OK(cudaMemset(x,0,count*2));OK(cudaMemset(y,0,count*2));
  cudaEvent_t start,stop;OK(cudaEventCreate(&start));OK(cudaEventCreate(&stop));float ms[2];
  for(int variant=0;variant<2;variant++){
   OK(cudaEventRecord(start));for(int j=0;j<10;j++){
    if(variant==0)old_concat<<<dim3((n0+255)/256,n1,n2),256>>>(x,y,o,a,n0,n1,n2,dim);
    else concat_f32_flat<true><<<unsigned((count+1023)/1024),256>>>(x,y,o,xs,ys,count);
   }OK(cudaEventRecord(stop));OK(cudaEventSynchronize(stop));OK(cudaEventElapsedTime(&ms[variant],start,stop));
  }printf("TIMING concat dim=%d values=%zu old_ms=%.6f candidate_ms=%.6f\n",dim,count,ms[0]/10,ms[1]/10);
  OK(cudaEventDestroy(start));OK(cudaEventDestroy(stop));OK(cudaFree(x));OK(cudaFree(y));OK(cudaFree(o));
 }
}
