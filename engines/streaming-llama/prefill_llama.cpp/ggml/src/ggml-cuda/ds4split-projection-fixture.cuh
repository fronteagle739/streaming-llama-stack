#pragma once
// Private standalone TEST entry appended to the one reviewed CUDA host unit.
// No model/engine entry, no owner config IO, no producer/API registration.
#include <cmath>
namespace ds4split_projection_fixture {
namespace p=ds4split_projection_candidate;
using ds4split_mmid::require;
constexpr uint64_t device_ceiling=512ull*1024*1024,host_budget=3ull*1024*1024*1024;
constexpr size_t q4_bytes=6635520,arena_bytes=128*1024*1024,host_chunk=65536;
constexpr size_t rhs_bytes=655360,out_bytes=655360,row_bytes=3932160,workspace_bytes=32*1024*1024;
constexpr size_t explicit_device_bytes=q4_bytes+arena_bytes;
static_assert(explicit_device_bytes==140853248,"fixture explicit GPU accounting");
struct golden_row{int slot,expert,k,n;uint64_t offset;const char*q4_sha;const char*half_sha;};
#include "projection-fixture-goldens.inc"
struct backend_owner {
    ggml_backend_t value=nullptr;
    ~backend_owner(){if(value)ggml_backend_free(value);}
};
struct source_owner {
    ds4split_mmid::handle file;
    source_owner():file(CreateFileW(_wgetenv(L"SLSTACK_DIAGNOSTIC_MODEL"),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_RANDOM_ACCESS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr)){
        ds4split_mmid::ordinary(file.value,false);LARGE_INTEGER size{};require(GetFileSizeEx(file.value,&size)&&uint64_t(size.QuadPart)==518596067328ull,"fixture canonical D source size differs");
    }
};
struct buffer_owner {
    ggml_backend_buffer_t value=nullptr;cudaStream_t stream;
    explicit buffer_owner(cudaStream_t s):stream(s){}
    ~buffer_owner(){if(value){CUDA_CHECK(cudaStreamSynchronize(stream));ggml_backend_buffer_free(value);}}
};
struct run {
    std::string nonce;uint64_t before_gpu_used=0,max_observed_delta=0;ggml_backend_cuda_context*ctx=nullptr;
    ggml_backend_buffer_t original_buffer=nullptr;size_t actual_source_allocation_bytes=0;
    std::vector<unsigned char> readback,readback2,source_chunk;
    std::ostringstream golden_records,case_records;size_t proved=0,cases=0,gpu_calls=0;uint64_t source_read=0;
    explicit run(const char*n,uint64_t before):nonce(n),before_gpu_used(before){require(ds4split_mmid::hex(nonce,32),"fixture nonce");}
    void guard(){
        ds4split_mmid::physical_ram();size_t free=0,total=0;CUDA_CHECK(cudaMemGetInfo(&free,&total));uint64_t used=total-free,delta=used>before_gpu_used?used-before_gpu_used:0;max_observed_delta=std::max(max_observed_delta,delta);require(delta<=device_ceiling,"fixture observed GPU usage exceeds512MiB ceiling");
    }
    void write(const std::string&suffix,const std::string&bytes){
        require(bytes.size()<=1024*1024,"fixture receipt exceeds1MiB");auto filename=ds4split_mmid::path("projection-fixture-"+nonce+"."+suffix);ds4split_mmid::handle out(CreateFileW(filename.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL|FILE_FLAG_OPEN_REPARSE_POINT,nullptr));ds4split_mmid::ordinary(out.value,false);DWORD wrote=0;require(WriteFile(out.value,bytes.data(),DWORD(bytes.size()),&wrote,nullptr)&&wrote==bytes.size()&&FlushFileBuffers(out.value),"fixture exclusive receipt write");
    }
    std::string device_hash(const void*data,size_t n){
        ds4split_mmid::digest digest;
        for(size_t off=0;off<n;){guard();size_t take=std::min(n-off,readback.size());CUDA_CHECK(cudaMemcpyAsync(readback.data(),static_cast<const char*>(data)+off,take,cudaMemcpyDeviceToHost,ctx->stream()));CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));digest.update(readback.data(),take);off+=take;}return digest.finish();
    }
    void exact_device_bytes(const void*a,const void*b,size_t n){
        for(size_t off=0;off<n;){guard();size_t take=std::min(n-off,readback.size());CUDA_CHECK(cudaMemcpyAsync(readback.data(),static_cast<const char*>(a)+off,take,cudaMemcpyDeviceToHost,ctx->stream()));CUDA_CHECK(cudaMemcpyAsync(readback2.data(),static_cast<const char*>(b)+off,take,cudaMemcpyDeviceToHost,ctx->stream()));CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));require(std::memcmp(readback.data(),readback2.data(),take)==0,"actual GPU output gather/scatter differs bytewise");off+=take;}
    }
    void load(source_owner&source,const golden_row&g,void*raw){
        require(g.offset<=518596067328ull&&q4_bytes<=518596067328ull-g.offset,"fixture original D source range bounds");LARGE_INTEGER offset{};offset.QuadPart=g.offset;require(SetFilePointerEx(source.file.value,offset,nullptr,FILE_BEGIN),"fixture source positioned seek");ds4split_mmid::digest hash;
        for(size_t at=0;at<q4_bytes;){guard();const DWORD n=DWORD(std::min(q4_bytes-at,source_chunk.size()));DWORD got=0;require(ReadFile(source.file.value,source_chunk.data(),n,&got,nullptr)&&got==n,"fixture64KiB source read");hash.update(source_chunk.data(),n);CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(raw)+at,source_chunk.data(),n,cudaMemcpyHostToDevice,ctx->stream()));CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));at+=n;source_read+=n;}
        require(hash.finish()==g.q4_sha&&device_hash(raw,q4_bytes)==g.q4_sha,"actual original D/GPU Q4_K matrix SHA differs from native proof");
    }
    void operator_case(const golden_row&g,bool half,int nc,const std::vector<int>&columns,int rank,void*weights,float*source,float*rhs,float*out,float*scattered,float*expected,float*gathered,p::reference_handle&reference,p::reference_modes&modes,const std::string&initial_modes){
        guard();const int k=g.k,n=g.n,channels=g.slot==2?6:1;
        ggml_tensor wb{},xb{},yb{};wb.type=GGML_TYPE_Q4_K;wb.buffer=original_buffer;wb.data=ggml_backend_buffer_get_base(original_buffer);wb.ne[0]=k;wb.ne[1]=n;wb.ne[2]=wb.ne[3]=1;wb.nb[0]=144;wb.nb[1]=size_t(k/256)*144;wb.nb[2]=wb.nb[1]*n;wb.nb[3]=wb.nb[2];
        for(int i=0;i<4;++i){xb.ne[i]=yb.ne[i]=1;}
        for(int start=0;start<nc;start+=16){int count=std::min(16,nc-start);
            for(int j=0;j<count;++j){int slot=columns[start+j],sr=channels==1?slot/6:slot;CUDA_CHECK(cudaMemcpyAsync(rhs+size_t(j)*k,source+size_t(sr)*k,size_t(k)*4,cudaMemcpyDeviceToDevice,ctx->stream()));}
            if(half){auto w=p::matrix(&wb,GGML_TYPE_F16,k,n,weights),x=p::matrix(&xb,GGML_TYPE_F32,k,count,rhs),y=p::matrix(&yb,GGML_TYPE_F32,n,count,out);y.src[0]=&w;y.src[1]=&x;require(w.buffer==nullptr&&w.view_src==nullptr&&w.extra==nullptr&&x.buffer==nullptr&&y.buffer==nullptr,"fixture borrowed views unexpectedly own provider storage");require(ggml_cuda_should_use_mmf(GGML_TYPE_F16,ggml_cuda_info().devices[ctx->device].cc,ggml_cuda_info().devices[ctx->device].warp_size,w.ne,w.nb,count,false),"fixture actual MMF shape unsupported");ggml_cuda_mul_mat_f(*ctx,&w,&x,nullptr,&y);}
            else{p::verify_reference_environment(p::arm::down_f32);modes.observe(reference.value,ctx->stream());require(modes.json()==initial_modes,"fixture pedantic getter drift");const float alpha=1,beta=0;CUBLAS_CHECK(cublasGemmEx(reference.value,CUBLAS_OP_T,CUBLAS_OP_N,n,count,k,&alpha,weights,CUDA_R_32F,k,rhs,CUDA_R_32F,k,&beta,out,CUDA_R_32F,n,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));}
            ++gpu_calls;CUDA_CHECK(cudaGetLastError());
            // Independent expected output is the ACTUAL contiguous GPU result,
            // not any CPU matrix multiplication. Only byte movement is checked.
            CUDA_CHECK(cudaMemcpyAsync(expected+size_t(start)*n,out,size_t(count)*n*4,cudaMemcpyDeviceToDevice,ctx->stream()));
            for(int j=0;j<count;++j)CUDA_CHECK(cudaMemcpyAsync(scattered+size_t(columns[start+j])*n,out+size_t(j)*n,size_t(n)*4,cudaMemcpyDeviceToDevice,ctx->stream()));
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));guard();
        }
        for(int t=0;t<nc;++t)CUDA_CHECK(cudaMemcpyAsync(gathered+size_t(t)*g.n,scattered+size_t(t*6+rank)*g.n,size_t(g.n)*4,cudaMemcpyDeviceToDevice,ctx->stream()));CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));exact_device_bytes(expected,gathered,size_t(nc)*g.n*4);
        if(cases++)case_records<<",";case_records<<"{\"k\":"<<g.k<<",\"n\":"<<g.n<<",\"columns\":"<<nc<<",\"rank\":"<<rank<<",\"actual_operator\":\""<<(half?"direct_mmf_half2_f32_accumulator":"direct_cublas_compute32f_pedantic")<<"\",\"output_sha256\":\""<<device_hash(expected,size_t(nc)*g.n*4)<<"\",\"scatter_regather_byteexact\":true}";
    }
    int execute(){
        ds4split_mmid::physical_ram(host_budget);require(explicit_device_bytes<=device_ceiling,"fixture declared allocation budget");
        readback.resize(4*1024*1024);readback2.resize(4*1024*1024);source_chunk.resize(host_chunk);
        source_owner source;backend_owner backend;backend.value=ggml_backend_cuda_init(0);require(backend.value,"fixture CUDA backend initialization");ctx=static_cast<ggml_backend_cuda_context*>(backend.value->context);guard();
        buffer_owner original(ctx->stream());original.value=ggml_backend_alloc_buffer(backend.value,q4_bytes);require(original.value,"fixture original sized Q4 allocation");original_buffer=original.value;actual_source_allocation_bytes=ggml_backend_buffer_get_size(original.value);require(actual_source_allocation_bytes>=q4_bytes&&actual_source_allocation_bytes<=q4_bytes+65536,"fixture source allocation alignment exceeds64KiB allowance");void*raw=ggml_backend_buffer_get_base(original.value);
        p::device_arena arena(arena_bytes,ctx->stream());guard();char*weights=arena.p;float*rhs=reinterpret_cast<float*>(weights+p::float_matrix);float*out=reinterpret_cast<float*>(reinterpret_cast<char*>(rhs)+rhs_bytes);char*workspace=reinterpret_cast<char*>(out)+out_bytes;
        float*inputs=reinterpret_cast<float*>(workspace+workspace_bytes);float*scattered=reinterpret_cast<float*>(reinterpret_cast<char*>(inputs)+row_bytes);float*expected=reinterpret_cast<float*>(reinterpret_cast<char*>(scattered)+row_bytes);float*gathered=reinterpret_cast<float*>(reinterpret_cast<char*>(expected)+row_bytes);
        require(reinterpret_cast<char*>(gathered)+row_bytes<=arena.p+arena.bytes,"fixture exact arena partition overflow");
        for(const auto&g:goldens){load(source,g,raw);auto convert=ggml_get_to_fp16_cuda(GGML_TYPE_Q4_K);require(convert,"fixture actual Q4_K→half converter absent");convert(raw,reinterpret_cast<half*>(weights),int64_t(g.k)*g.n,ctx->stream());CUDA_CHECK(cudaGetLastError());auto actual=device_hash(weights,p::half_matrix);
            if(actual!=g.half_sha){CUDA_CHECK(cudaMemcpyAsync(readback.data(),weights,65536,cudaMemcpyDeviceToHost,ctx->stream()));CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));std::string sample(reinterpret_cast<const char*>(readback.data()),65536);write("half-prefix.bin",sample);std::ostringstream error;error<<"{\"status\":\"CUDA_NATIVE_HALF_BITS_MISMATCH\",\"slot\":"<<g.slot<<",\"expert\":"<<g.expert<<",\"source_offset\":"<<g.offset<<",\"actual_half_sha256\":\""<<actual<<"\",\"expected_half_sha256\":\""<<g.half_sha<<"\",\"half_prefix_bytes\":65536,\"half_prefix_sha256\":\""<<ds4split_mmid::sha(sample.data(),sample.size())<<"\",\"model_engine_initialized\":false,\"mmf_equivalence_claimed\":false}\n";write("mismatch.json",error.str());throw std::runtime_error("actual CUDA half bits differ from native Metal proof; no MMF continuation");}
            if(proved++)golden_records<<",";golden_records<<"{\"slot\":"<<g.slot<<",\"expert\":"<<g.expert<<",\"offset\":"<<g.offset<<",\"q4_sha256\":\""<<g.q4_sha<<"\",\"half_sha256\":\""<<actual<<"\"}";
        }
        require(proved==282,"fixture complete282 half proof required before operators");
        std::array<int32_t,192> ids{};for(int i=0;i<192;++i)ids[i]=i%6;auto columns=p::ordered_columns(ids.data());const golden_row*shapes[2]={nullptr,nullptr};for(const auto&g:goldens){if(g.slot==0&&!shapes[0])shapes[0]=&g;if(g.slot==2&&!shapes[1])shapes[1]=&g;}
        std::ostringstream reference_records;
        for(int shape=0;shape<2;++shape){require(shapes[shape],"fixture original geometries missing");const auto&g=*shapes[shape];load(source,g,raw);const int source_rows=g.slot==2?192:32;std::vector<float> host_inputs(size_t(g.k)*source_rows);for(size_t i=0;i<host_inputs.size();++i)host_inputs[i]=float(int((i*3+(i/g.k)*17)%257)-128)*0.0078125f;CUDA_CHECK(cudaMemcpyAsync(inputs,host_inputs.data(),host_inputs.size()*4,cudaMemcpyHostToDevice,ctx->stream()));CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));
            for(int mode=0;mode<2;++mode){bool half_mode=mode==0;p::reference_handle reference(ctx->stream());p::reference_modes modes;std::string initial;
                if(half_mode){auto convert=ggml_get_to_fp16_cuda(GGML_TYPE_Q4_K);convert(raw,reinterpret_cast<half*>(weights),int64_t(g.k)*g.n,ctx->stream());}
                else{p::verify_reference_environment(p::arm::down_f32);auto convert=ggml_get_to_fp32_cuda(GGML_TYPE_Q4_K);require(convert,"fixture actual Q4_K→float converter absent");convert(raw,reinterpret_cast<float*>(weights),int64_t(g.k)*g.n,ctx->stream());reference.create();CUBLAS_CHECK(cublasSetStream(reference.value,ctx->stream()));CUBLAS_CHECK(cublasSetMathMode(reference.value,CUBLAS_PEDANTIC_MATH));CUBLAS_CHECK(cublasSetAtomicsMode(reference.value,CUBLAS_ATOMICS_NOT_ALLOWED));CUBLAS_CHECK(cublasSetPointerMode(reference.value,CUBLAS_POINTER_MODE_HOST));CUBLAS_CHECK(cublasSetWorkspace(reference.value,workspace,workspace_bytes));modes.observe(reference.value,ctx->stream());initial=modes.json();if(shape)reference_records<<",";reference_records<<initial;}
                for(int nc=1;nc<=32;++nc){CUDA_CHECK(cudaMemsetAsync(scattered,0xcd,row_bytes,ctx->stream()));operator_case(g,half_mode,nc,columns[0],0,weights,inputs,rhs,out,scattered,expected,gathered,reference,modes,initial);}
                std::array<bool,192> seen{};for(int rank=0;rank<6;++rank){for(int dst:columns[rank]){require(!seen[dst],"fixture duplicate full scatter slot");seen[dst]=true;}operator_case(g,half_mode,32,columns[rank],rank,weights,inputs,rhs,out,scattered,expected,gathered,reference,modes,initial);}for(bool v:seen)require(v,"fixture full192-slot scatter hole");
                if(!half_mode){modes.observe(reference.value,ctx->stream());require(modes.json()==initial,"fixture reference getters changed after operations");}
            }
        }
        guard();std::ostringstream result;result<<"{\"status\":\"MODEL_FREE_CUDA_PROJECTION_FIXTURE_PASS\",\"nonce\":\""<<nonce<<"\",\"model_engine_initialized\":false,\"full_model_mapped\":false,\"native_golden_proof_sha256\":\"24231bd1932c637fd9344e7f780d9cf1352303d55a7af5a7c01f5933582a6025\",\"native_half_matrices_verified\":"<<proved<<",\"actual_operator_cases\":"<<cases<<",\"actual_gpu_projection_calls\":"<<gpu_calls<<",\"source_D_read_bytes\":"<<source_read<<",\"host_process_budget_bytes\":"<<host_budget<<",\"physical_ram_floor_bytes\":"<<ds4split_mmid::ram_floor<<",\"explicit_device_allocation_bytes\":"<<explicit_device_bytes<<",\"observed_gpu_ceiling_bytes\":"<<device_ceiling<<",\"maximum_observed_GPU_delta_bytes\":"<<max_observed_delta<<",\"stream_fenced_reuse\":true,\"scatter_192_slots_covered\":true,\"reference_actual_getters\":["<<reference_records.str()<<"],\"goldens\":["<<golden_records.str()<<"],\"cases\":["<<case_records.str()<<"],\"qualification_passed\":false,\"native_operator_equivalence_claimed\":false}\n";write("json",result.str());return 0;
    }
};
} // namespace ds4split_projection_fixture
extern "C" int ds4split_expert_projection_fixture(const char*nonce,unsigned long long before_gpu_used){
    try{ds4split_projection_fixture::run fixture(nonce,before_gpu_used);return fixture.execute();}
    catch(const std::exception&e){std::fprintf(stderr,"MODEL_FREE_CUDA_PROJECTION_FIXTURE_FAILED: %s\n",e.what());return 2;}
}
