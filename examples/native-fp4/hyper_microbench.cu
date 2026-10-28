#include "mmq.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" void ggml_abort(const char *file,int line,const char *format,...) {
    std::fprintf(stderr,"GGML abort %s:%d ",file,line);
    va_list ap;va_start(ap,format);std::vfprintf(stderr,format,ap);va_end(ap);std::exit(2);
}
static void check(cudaError_t error) {if(error!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(error));std::exit(3);}}
__host__ __device__ static unsigned hash32(unsigned e,unsigned row,unsigned k) {
    unsigned x=(e+1)*0x9e3779b9u ^ (row+1)*0x85ebca6bu ^ (k+1)*0xc2b2ae35u;
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;
}
__host__ __device__ static int pattern(int block) {return block%4==3?0:block%4;}
__device__ static unsigned acode(unsigned e,unsigned row,unsigned k) {
    const unsigned codes[5]={12,10,0,2,4};return codes[hash32(e,row,k)%5];
}
__global__ static void weights(block_mxfp4 *out,long long blocks,int N,int K) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=blocks)return;
    int kb=i%(K/32),n=(i/(K/32))%N,e=i/((long long)N*(K/32));
    block_mxfp4 value;value.e=127+pattern(kb);
    for(int j=0;j<16;++j)value.qs[j]=(hash32(e,n,kb*32+j)&15)|((hash32(e,n,kb*32+j+16)&15)<<4);
    out[i]=value;
}
__global__ static void activations(block_fp4_mmq *out,long long blocks,int rows,int M,int K,int Kpad) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=blocks)return;
    int group=i/rows,flat=i%rows,e=flat/M,m=flat%M;
    block_fp4_mmq value={};
    for(int b=0;b<8;++b) {
        int kb=group*8+b;
        unsigned s=127+(pattern(kb)+(e+m)%3)%3;
        value.d4[b/2]|=s<<((b%2)*8);
        for(int j=0;j<16;++j) {
            int k=group*256+b*32+j;
            unsigned lo=k<K?acode(e,m,k):0,hi=k+16<K?acode(e,m,k+16):0;
            value.qs[b*16+j]=(lo|(hi<<4));
        }
    }
    out[i]=value;
}
template<int X> static void launch(const char *w,const int *a,const int *ids,const int *bounds,float *out,float *fix,
                                   int E,int M,int N,int K,int Kpad,int sms,int Y,int nw,int shared,cudaStream_t stream) {
    const int rows=E*M,channel_x=N*(K/32),tiles=((M+X-1)/X)*((N+Y-1)/Y)*E;
    mul_mat_q<GGML_TYPE_MXFP4,X,false><<<sms,dim3(32,nw),shared,stream>>>(w,a,ids,bounds,out,fix,
        K,N,M,K/32,rows,N,1,E,channel_x,0,0,1,1,0,0,0,M);
    if(tiles%sms)mul_mat_q_stream_k_fixup<GGML_TYPE_MXFP4,X,false><<<sms,dim3(32,nw),0,stream>>>(
        ids,bounds,out,fix,K,N,M,N,E,0,1,0,M);
}
int main(int argc,char **argv) {
    if(argc!=5)return 1;
    int E=std::atoi(argv[1]),M=std::atoi(argv[2]),N=std::atoi(argv[3]),K=std::atoi(argv[4]);
    if(E<1||E>384||M<1||M>1536||!((N==2304&&K==5120)||(N==5120&&K==2304)))return 1;
    cudaDeviceProp p;check(cudaGetDeviceProperties(&p,0));if(p.major!=12||p.minor!=0)return 1;
    int cc=p.major*100+p.minor*10,Y=get_mmq_y_host(cc),nw=mmq_get_nwarps_host(cc,32),X=M<=8?8:64;
    if(N%Y)return 1;
    int shared=X==8?mmq_get_nbytes_shared<GGML_TYPE_MXFP4>(8,Y,cc,32,nw):mmq_get_nbytes_shared<GGML_TYPE_MXFP4>(64,Y,cc,32,nw);
    if(X==8)check(cudaFuncSetAttribute(mul_mat_q<GGML_TYPE_MXFP4,8,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    else check(cudaFuncSetAttribute(mul_mat_q<GGML_TYPE_MXFP4,64,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    long long wb=(long long)E*N*K/32;int Kpad=(K+511)/512*512,rows=E*M;
    size_t wbytes=wb*sizeof(block_mxfp4),ablocks=(size_t)Kpad/256*rows;
    int banks=std::max(1,int(((128ull<<20)+wbytes-1)/wbytes));
    block_mxfp4 *w;block_fp4_mmq *a;float *out,*fix;int *ids,*bounds;
    check(cudaMalloc(&w,wbytes*banks));check(cudaMalloc(&a,(ablocks+128)*sizeof(block_fp4_mmq)));
    check(cudaMalloc(&out,(size_t)E*M*N*sizeof(float)));check(cudaMalloc(&fix,(size_t)p.multiProcessorCount*X*Y*sizeof(float)));
    check(cudaMalloc(&ids,(rows+128)*sizeof(int)));check(cudaMalloc(&bounds,(E+1)*sizeof(int)));
    weights<<<(wb+255)/256,256>>>(w,wb,N,K);
    activations<<<(ablocks+255)/256,256>>>(a,ablocks,rows,M,K,Kpad);
    check(cudaMemset(a+ablocks,0,128*sizeof(block_fp4_mmq)));
    for(int bank=1;bank<banks;++bank)check(cudaMemcpy((char*)w+bank*wbytes,w,wbytes,cudaMemcpyDeviceToDevice));
    std::vector<int> hi(rows+128,0),hb(E+1);for(int i=0;i<rows;++i)hi[i]=i;for(int i=0;i<=E;++i)hb[i]=i*M;
    check(cudaMemcpy(ids,hi.data(),hi.size()*sizeof(int),cudaMemcpyHostToDevice));check(cudaMemcpy(bounds,hb.data(),hb.size()*sizeof(int),cudaMemcpyHostToDevice));
    auto run=[&](int bank){if(X==8)launch<8>((char*)w+bank*wbytes,(int*)a,ids,bounds,out,fix,E,M,N,K,Kpad,p.multiProcessorCount,Y,nw,shared,0);
        else launch<64>((char*)w+bank*wbytes,(int*)a,ids,bounds,out,fix,E,M,N,K,Kpad,p.multiProcessorCount,Y,nw,shared,0);};
    run(0);check(cudaDeviceSynchronize());
    const float lut[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
    int checked=0;double maxerror=0;
    for(int ee:std::vector<int>{0,E/2,E-1})for(int mm:std::vector<int>{0,M/2,M-1})for(int nn:std::vector<int>{0,N/2,N-1}) {
        double reference=0;for(int k=0;k<K;++k) {
            double av=int(hash32(ee,mm,k)%5)-2,bv=lut[hash32(ee,nn,k)&15];int pp=pattern(k/32);
            reference+=av*std::ldexp(bv,pp+(pp+(ee+mm)%3)%3);
        }
        float actual;check(cudaMemcpy(&actual,out+((size_t)ee*M+mm)*N+nn,sizeof(float),cudaMemcpyDeviceToHost));
        double error=std::abs(actual-reference);maxerror=std::max(maxerror,error);++checked;
    }
    if(maxerror!=0){std::fprintf(stderr,"reference mismatch max_absolute_error=%g\n",maxerror);return 2;}
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    int repeats=100;check(cudaEventRecord(begin));for(int i=0;i<repeats;++i)run(i%banks);check(cudaEventRecord(end));check(cudaEventSynchronize(end));
    float ms;check(cudaEventElapsedTime(&ms,begin,end));ms/=repeats;
    double ops=2.*E*M*N*K,bytes=wbytes+(double)ablocks*sizeof(block_fp4_mmq)+(double)E*M*N*4;
    std::printf("{\"status\":\"PASS\",\"fixture\":\"hash32-v1\",\"experts\":%d,\"M_per_expert\":%d,\"N\":%d,\"K\":%d,\"mmq_x\":%d,\"mmq_y\":%d,\"milliseconds\":%.9g,\"TFLOPS\":%.9g,\"logical_effective_GB_s\":%.9g,\"sampled_reference_elements\":%d,\"max_absolute_error\":%.9g,\"banks\":%d}\n",E,M,N,K,X,Y,ms,ops/ms/1e9,bytes/ms/1e6,checked,maxerror,banks);
    check(cudaDeviceSynchronize());return 0;
}
