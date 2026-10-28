/* External single-row hybrid FFN. The ordinary ds4 GPU path consumes the
 * VRAM experts while the existing ds4 CPU pool consumes borrowed RAM experts.
 * Only activations and the final output row cross PCIe. */
#ifdef DS4_EXTERNAL_DECODE
#include <array>
static double g_external_cpu_ms=0,g_external_wait_ms=0,g_external_transfer_ms=0,g_external_merge_ms=0;
static uint64_t g_external_cpu_experts=0,g_external_cpu_layers=0;
static double external_ms(std::chrono::steady_clock::time_point t){
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
}
struct external_cpu_worker {
    std::mutex mutex;std::condition_variable cv;std::thread thread;
    bool pending=false,done=false,stop=false;int ok=0;
    uint32_t count=0,in_dim=0,mid_dim=0,out_dim=0;float clamp=0;
    std::array<const char*,6>gate{},up{},down{};std::array<float,6>weights{};
    std::array<float,5120>x{},out{};double compute_ms=0;
    ~external_cpu_worker(){shutdown();}
    void shutdown(){
        if(!thread.joinable())return;
        {std::lock_guard<std::mutex>l(mutex);stop=true;cv.notify_one();}
        thread.join();stop=false;pending=false;done=false;
    }
    void start(){
        if(!thread.joinable())thread=std::thread([this]{
            std::unique_lock<std::mutex>l(mutex);
            for(;;){
                cv.wait(l,[this]{return stop||pending;});if(stop)return;
                pending=false;l.unlock();auto t=std::chrono::steady_clock::now();
                ok=ds4_external_cpu_q4(out.data(),x.data(),gate.data(),up.data(),down.data(),weights.data(),count,in_dim,mid_dim,out_dim,clamp);
                compute_ms=external_ms(t);l.lock();done=true;cv.notify_one();
            }
        });
        std::lock_guard<std::mutex>l(mutex);done=false;pending=true;cv.notify_one();
    }
    int finish(){std::unique_lock<std::mutex>l(mutex);cv.wait(l,[this]{return done;});return ok;}
};
static external_cpu_worker g_external_cpu;
static void external_cpu_shutdown(){g_external_cpu.shutdown();g_external_host_busy=false;}
static int external_cpu_mode(){
    static int mode=[]{const char*p=getenv("LLAMADECODE_HOST_MODE");return p&&!strcmp(p,"cpu")?1:p&&!strcmp(p,"mix")?2:0;}();
    return mode;
}
/* A CPU slot is represented by -1 only inside the qualified Q4 decode path.
 * Its gate result is unused and the existing down sum skips it. */
static std::vector<int32_t> g_external_cpu_ids;
static double g_external_cpu_cost_ms=0;
static uint32_t g_external_cpu_samples=0;
static bool external_cpu_choose(uint32_t count){
    if(external_cpu_mode()==1)return true;
    // Warm both choices, then compare measured service cost per expert.
    // Periodic CPU probes track changing load without running both paths.
    static uint64_t calls=0;++calls;
    if(g_external_cpu_samples<4)return true;
    if(g_external_upload_samples<4)return false;
    const double cpu=g_external_cpu_cost_ms/g_external_cpu_samples;
    const double upload=g_external_upload_cost_ms/g_external_upload_samples;
    return cpu<upload||calls%128==0;
}
extern "C" __declspec(dllexport) void ds4x_host_timing_v1(double *times,uint64_t *counts){
    if(times){times[0]=g_external_cpu_ms;times[1]=g_external_wait_ms;times[2]=g_external_transfer_ms;times[3]=g_external_merge_ms;}
    if(counts){counts[0]=g_external_cpu_experts;counts[1]=g_external_cpu_layers;}
}
__global__ static void external_cpu_add_kernel(float*out,const float*cpu,uint32_t n){
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]+=cpu[i];
}
#endif
