#include "qfn-vision.h"
#include "../mtmd/clip.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "cuda_runtime.h"
#include "../../../../ds4-mac/ds4_image.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace {
using clock_type=std::chrono::steady_clock;
double seconds(clock_type::time_point start){return std::chrono::duration<double>(clock_type::now()-start).count();}
void need(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
uint64_t free_vram(){size_t free=0,total=0;need(cudaDeviceSynchronize()==cudaSuccess&&cudaMemGetInfo(&free,&total)==cudaSuccess,"cannot measure CUDA memory");return free;}
}

bool qfn_vision_encode_file(const std::string &model,const std::string &image,
    uint32_t minimum,uint32_t maximum,int threads,const std::string &backend,bool flash_attention,
    qfn_vision_result &out,std::string &error) {
    ds4_image decoded{};ds4_image_patches patches{};clip_ctx *vision=nullptr;
    try {
        need(minimum>=1&&maximum>=minimum&&maximum<=16383&&threads>=1&&threads<=256,
             "invalid explicit image token/thread budget");
        need(!backend.empty(),"an explicit CUDA vision backend is required");
        auto start=clock_type::now();char reason[512]{};
        need(ds4_image_decode_file(&decoded,image.c_str(),reason,sizeof(reason))!=0,reason);
        out.image_width=decoded.width;out.image_height=decoded.height;
        need(ds4_image_preprocess_qwen4(&patches,&decoded,minimum,maximum,reason,sizeof(reason))!=0,reason);
        out.count=patches.image_token_count;out.grid_height=patches.grid_height/2;
        out.grid_width=patches.grid_width/2;std::copy(decoded.fingerprint,decoded.fingerprint+32,out.fingerprint.begin());
        // Reuse the Mac's exact decode/bicubic/normalization, then invert only
        // its patch ordering to the shared CLIP graph's normalized RGB input.
        const uint32_t width=patches.padded_width,height=patches.padded_height;
        std::vector<float> pixels(size_t(width)*height*3);size_t at=0;
        for(uint32_t by=0;by<patches.grid_height/2;by++)for(uint32_t bx=0;bx<patches.grid_width/2;bx++)
        for(uint32_t my=0;my<2;my++)for(uint32_t mx=0;mx<2;mx++)
        for(uint32_t c=0;c<3;c++)for(uint32_t y=0;y<16;y++)for(uint32_t x=0;x<16;x++)
            pixels[(size_t((by*2+my)*16+y)*width+(bx*2+mx)*16+x)*3+c]=patches.patches[at++];
        need(at==pixels.size(),"QFN image patch extent differs");
        ds4_image_patches_free(&patches);ds4_image_free(&decoded);
        out.preprocess_seconds=seconds(start);
        // Same isolated load/encode/unload lifecycle as slstack.vision_worker.
        // An explicit flag supplies the retained library's backend selection.
        need(_putenv_s("MTMD_BACKEND_DEVICE",backend.c_str())==0,"cannot select explicit vision backend");
        ggml_backend_register(ggml_backend_cuda_reg());
        auto *device=ggml_backend_dev_by_name(backend.c_str());
        need(device&&ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU,"selected vision device is not CUDA GPU");
        out.free_before=free_vram();start=clock_type::now();
        clip_context_params options{};options.use_gpu=true;options.warmup=false;
        need(!(flash_attention&&std::getenv("LLAMA_CLIP_DISABLE_FLASH")),"vision flash-attention flag conflicts with explicit disable option");
        options.flash_attn_type=flash_attention?CLIP_FLASH_ATTN_TYPE_ENABLED:CLIP_FLASH_ATTN_TYPE_DISABLED;
        options.image_min_tokens=int(minimum);options.image_max_tokens=int(maximum);
        auto loaded=clip_init(model.c_str(),options);vision=loaded.ctx_v;
        if(loaded.ctx_a)clip_free(loaded.ctx_a);
        need(vision&&clip_has_vision_encoder(vision),"QFN vision projector could not load");
        need(clip_use_qfn_reference_geometry(vision),"projector is not the validated QFN vision tower");
        out.width=uint32_t(clip_n_mmproj_embd(vision));need(out.width==2560,"QFN image projection width differs");
        out.free_after_load=free_vram();out.load_seconds=seconds(start);
        out.values.resize(size_t(out.count)*out.width);start=clock_type::now();
        need(clip_encode_float_image(vision,threads,pixels.data(),int(height),int(width),out.values.data()),"QFN CUDA image encoding failed");
        need(std::all_of(out.values.begin(),out.values.end(),[](float v){return std::isfinite(v);}),"nonfinite QFN image embeddings");
        need(cudaDeviceSynchronize()==cudaSuccess,"QFN image CUDA completion failed");out.encode_seconds=seconds(start);
        start=clock_type::now();clip_free(vision);vision=nullptr;
        out.free_after_unload=free_vram();out.unload_seconds=seconds(start);
        return true;
    }catch(const std::exception &e){error=e.what();if(vision)clip_free(vision);
        ds4_image_patches_free(&patches);ds4_image_free(&decoded);return false;}
}
