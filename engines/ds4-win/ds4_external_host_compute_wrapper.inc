/* Called only on the external Q4_K scalar decode path. Other formats and
 * multi-row prefills retain their existing GPU dispatch. */
static int routed_moe_launch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map,
        uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint64_t down_offset, uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens,
        int allow_streaming, int owned_filtered) {
#define DS4_EXTERNAL_MOE_ARGS out,gate,up,mid,down,model_map,model_size,gate_offset,up_offset,down_offset,gate_type,down_type,gate_expert_bytes,gate_row_bytes,down_expert_bytes,down_row_bytes,expert_in_dim,expert_mid_dim,out_dim,selected,weights,n_total_expert,n_expert,clamp,x,layer_index,n_tokens,allow_streaming,owned_filtered
#ifdef DS4_EXTERNAL_DECODE
    const bool eligible=g_ds4x_expert_hook&&g_ds4x_residency_hook&&external_cpu_mode()&&
        n_tokens==1u&&(n_expert==6u||n_expert==3u)&&gate_type==12u&&down_type==12u&&
        expert_in_dim==5120u&&expert_mid_dim==2304u&&out_dim==5120u&&
        !getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6");
    if(!eligible)return routed_moe_launch_impl(DS4_EXTERNAL_MOE_ARGS);
    std::array<int32_t,6>ids{};std::array<float,6>w{};std::array<uint8_t,384>tier{};
    if(!ds4_gpu_tensor_read(selected,0,ids.data(),n_expert*sizeof(int32_t)))return 0;
    bool found=false;
    for(uint32_t i=0;i<n_expert;++i)if(ids[i]>=0&&external_host_find(layer_index,uint32_t(ids[i]))>=0)found=true;
    if(!found)return routed_moe_launch_impl(DS4_EXTERNAL_MOE_ARGS);
    if(!g_ds4x_residency_hook(g_ds4x_residency_user,layer_index,tier.data(),384u))return 0;
    auto &job=g_external_cpu;job.count=0;g_external_cpu_ids.clear();
    std::array<uint32_t,6>cpu_positions{};
    for(uint32_t i=0;i<n_expert;++i)if(ids[i]>=0&&ids[i]<384&&tier[ids[i]]==2){
        const int slot=external_host_find(layer_index,uint32_t(ids[i]));if(slot<0)return 0;
        const uint32_t k=job.count++;
        cpu_positions[k]=i;
        job.gate[k]=g_external_host_cache.gate_ptr+uint64_t(slot)*g_external_host_cache.gate_expert_bytes;
        job.up[k]=g_external_host_cache.up_ptr+uint64_t(slot)*g_external_host_cache.gate_expert_bytes;
        job.down[k]=g_external_host_cache.down_ptr+uint64_t(slot)*g_external_host_cache.down_expert_bytes;
        g_external_cpu_ids.push_back(ids[i]);
    }
    if(!job.count||!external_cpu_choose(job.count)){
        g_external_cpu_ids.clear();return routed_moe_launch_impl(DS4_EXTERNAL_MOE_ARGS);
    }
    const auto transfer=std::chrono::steady_clock::now();
    // The decode stream can be nonblocking; a legacy-stream memcpy does
    // not make an activation produced on that stream ready for the CPU.
    if(!cuda_ok(cudaStreamSynchronize(cuda_decode_stream()),"CPU expert input readiness"))return 0;
    if(!ds4_gpu_tensor_read(x,0,job.x.data(),expert_in_dim*sizeof(float))||
       !ds4_gpu_tensor_read(weights,0,w.data(),n_expert*sizeof(float))){g_external_cpu_ids.clear();return 0;}
    uint32_t k=0;for(uint32_t i=0;i<n_expert;++i)if(std::find(g_external_cpu_ids.begin(),g_external_cpu_ids.end(),ids[i])!=g_external_cpu_ids.end())job.weights[k++]=w[i];
    g_external_transfer_ms+=external_ms(transfer);
    job.in_dim=expert_in_dim;job.mid_dim=expert_mid_dim;job.out_dim=out_dim;job.clamp=clamp;
    g_external_host_busy=true;job.start();
    const auto gpu_start=std::chrono::steady_clock::now();
    int ok=job.count==n_expert?1:routed_moe_launch_impl(DS4_EXTERNAL_MOE_ARGS);
    const auto waited=std::chrono::steady_clock::now();
    ok=job.finish()&&ok;g_external_wait_ms+=external_ms(waited);
    g_external_cpu_ms+=job.compute_ms;g_external_cpu_experts+=job.count;++g_external_cpu_layers;
    // Both CPU service time and exposed wait are recorded; cost selection
    // includes the complete CPU service so it cannot over-credit overlap.
    g_external_cpu_cost_ms+=job.compute_ms/job.count;++g_external_cpu_samples;
    (void)gpu_start;
    g_external_cpu_ids.clear();g_external_host_busy=false;
    if(!ok)return 0;
    const auto merge=std::chrono::steady_clock::now();
    if(!cuda_ok(cudaStreamSynchronize(cuda_decode_stream()),"CPU expert scratch retirement"))return 0;
    static unsigned verify_count=0;
    const char*verify=getenv("LLAMADECODE_HOST_CPU_VERIFY");
    const bool verify_all=verify&&(!strcmp(verify,"all")||!strcmp(verify,"reference"));
    if(verify&&*verify&&strcmp(verify,"0")&&(verify_all||verify_count<16u)){
        ++verify_count;
        std::array<float,5120>hybrid{},reference{};
        if(job.count!=n_expert&&!ds4_gpu_tensor_read(out,0,hybrid.data(),out_dim*sizeof(float)))return 0;
        for(uint32_t i=0;i<out_dim;++i)hybrid[i]+=job.out[i];
        // All-RAM/GPU reference uses the same activation and original weights.
        if(!routed_moe_launch_impl(DS4_EXTERNAL_MOE_ARGS)||
           !cuda_ok(cudaStreamSynchronize(cuda_decode_stream()),"CPU expert verification")||
           !ds4_gpu_tensor_read(out,0,reference.data(),out_dim*sizeof(float)))return 0;
        double diff=0,norm=0,maximum=0,cpu_norm=0;
        for(uint32_t i=0;i<out_dim;++i){double d=double(hybrid[i])-reference[i];cpu_norm+=double(job.out[i])*job.out[i];diff+=d*d;norm+=double(reference[i])*reference[i];maximum=std::max(maximum,std::abs(d));}
        fprintf(stderr,"LLAMADECODE_HOST_CPU_VERIFY row=%u layer=%u experts=%u relative_rms=%.9g max_abs=%.9g cpu_norm=%.9g reference_norm=%.9g forward=%s\n",verify_count,layer_index,job.count,sqrt(diff/std::max(norm,1e-30)),maximum,sqrt(cpu_norm),sqrt(norm),!strcmp(verify,"reference")?"reference":"hybrid");
        if(verify_all&&sqrt(diff/std::max(norm,1e-30))>1e-6){
            std::array<cuda_block_q8_K,20>gx{};
            std::array<cuda_block_q8_K,54>gm{};
            std::array<float,13824>fm{};
            if(!ds4_gpu_tensor_read(down,0,gx.data(),sizeof(gx))||
               !ds4_gpu_tensor_read(gate,0,gm.data(),n_expert*9*sizeof(cuda_block_q8_K))||
               !ds4_gpu_tensor_read(mid,0,fm.data(),n_expert*expert_mid_dim*sizeof(float)))return 0;
            const auto*cx=(const cuda_block_q8_K*)ds4_external_cpu_trace(0);
            const auto*cm=(const cuda_block_q8_K*)ds4_external_cpu_trace(1);
            const auto*cf=(const float*)ds4_external_cpu_trace(2);
            unsigned xq_diff=0,midq_diff=0,xq_codes=0,midq_codes=0;
            double mid_diff=0,mid_norm=0,xq_scale=0,midq_scale=0;
            auto compare_q8=[](const cuda_block_q8_K&a,const cuda_block_q8_K&b,unsigned&codes,double&scale){
                for(unsigned j=0;j<256;++j)codes+=(a.d<0?-int(a.qs[j]):int(a.qs[j]))!=(b.d<0?-int(b.qs[j]):int(b.qs[j]));
                scale=std::max(scale,std::abs(double(fabsf(a.d))-fabsf(b.d))/std::max(double(fabsf(b.d)),1e-30));
            };
            for(unsigned j=0;j<20;++j)compare_q8(cx[j],gx[j],xq_codes,xq_scale);
            for(unsigned j=0;j<sizeof(gx);++j)xq_diff+=((const uint8_t*)cx)[j]!=((const uint8_t*)gx.data())[j];
            for(unsigned k=0;k<job.count;++k){
                for(unsigned j=0;j<9;++j)compare_q8(cm[k*9+j],gm[cpu_positions[k]*9+j],midq_codes,midq_scale);
                for(unsigned j=0;j<9*sizeof(cuda_block_q8_K);++j)
                    midq_diff+=((const uint8_t*)(cm+k*9))[j]!=((const uint8_t*)(gm.data()+cpu_positions[k]*9))[j];
                for(unsigned j=0;j<expert_mid_dim;++j){double a=cf[k*expert_mid_dim+j],b=fm[cpu_positions[k]*expert_mid_dim+j];mid_diff+=(a-b)*(a-b);mid_norm+=b*b;}
            }
            fprintf(stderr,"LLAMADECODE_HOST_CPU_QUANT row=%u xq_byte_diff=%u midq_byte_diff=%u xq_code_diff=%u midq_code_diff=%u xq_scale_relative=%.9g midq_scale_relative=%.9g mid_relative_rms=%.9g\n",verify_count,xq_diff,midq_diff,xq_codes,midq_codes,xq_scale,midq_scale,sqrt(mid_diff/std::max(mid_norm,1e-30)));
        }
        if(strcmp(verify,"reference")&&!ds4_gpu_tensor_write(out,0,hybrid.data(),out_dim*sizeof(float)))return 0;
        g_external_merge_ms+=external_ms(merge);return 1;
    }
    if(job.count==n_expert){
        if(!ds4_gpu_tensor_write(out,0,job.out.data(),out_dim*sizeof(float)))return 0;
    }else{
        // The stream fence above retires the gate scratch before row upload.
        if(!ds4_gpu_tensor_write(gate,0,job.out.data(),out_dim*sizeof(float)))return 0;
        external_cpu_add_kernel<<<(out_dim+255u)/256u,256,0,cuda_decode_stream()>>>((float*)out->ptr,(const float*)gate->ptr,out_dim);
        if(!cuda_ok(cudaGetLastError(),"hybrid expert output merge"))return 0;
    }
    g_external_merge_ms+=external_ms(merge);return 1;
#else
    return routed_moe_launch_impl(DS4_EXTERNAL_MOE_ARGS);
#endif
#undef DS4_EXTERNAL_MOE_ARGS
}
