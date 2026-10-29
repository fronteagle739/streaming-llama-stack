// Small admission experiment: ordinary versus programmatic graph edges.
// It is not a model or a prediction of full-layer speedup.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static void check(cudaError_t e) {if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(2);}}
template<bool PDL> __global__ void chain(const float *input,float *out,int count) {
    const int at=blockIdx.x*blockDim.x+threadIdx.x;
    if constexpr(PDL) {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
    if(at<count) {float x=input[at];for(int i=0;i<16;++i)x=__fadd_rn(__fmul_rn(x,.999f),.001f);out[at]=x;}
}
template<bool PDL> static void launch(cudaStream_t stream,const float *in,float *out,int count) {
    cudaLaunchConfig_t config{};config.gridDim=dim3((count+255)/256);config.blockDim=dim3(256);config.stream=stream;
    cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed=1;
    if constexpr(PDL){config.attrs=&attribute;config.numAttrs=1;}
    check(cudaLaunchKernelEx(&config,chain<PDL>,in,out,count));
}
template<bool PDL> static float measure(cudaStream_t stream,float *a,float *b,int count,std::vector<float>&out) {
    check(cudaMemsetAsync(a,0,count*4,stream));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed));
    for(int step=0;step<16;++step){launch<PDL>(stream,a,b,count);auto tmp=a;a=b;b=tmp;}
    cudaGraph_t graph;cudaGraphExec_t exec;
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
    check(cudaEventRecord(begin,stream));
    for(int repeat=0;repeat<20;++repeat)check(cudaGraphLaunch(exec,stream));
    check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
    float ms;check(cudaEventElapsedTime(&ms,begin,end));
    check(cudaMemcpy(out.data(),a,count*4,cudaMemcpyDeviceToHost));
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
    return ms/20;
}
int main() {
    constexpr int count=4*1024*1024;
    float *a,*b;check(cudaMalloc(&a,count*4));check(cudaMalloc(&b,count*4));
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::vector<float>control(count),pdl(count);
    const float ordinary=measure<false>(stream,a,b,count,control),dependent=measure<true>(stream,a,b,count,pdl);
    if(std::memcmp(control.data(),pdl.data(),count*4))return 3;
    std::printf("PASS PDL graph-chain full output bitwise; ordinary_ms=%.6f pdl_ms=%.6f elements=%d kernels=16 repeats=20 scope=component_only\n",ordinary,dependent,count);
    check(cudaStreamDestroy(stream));check(cudaFree(a));check(cudaFree(b));return 0;
}
