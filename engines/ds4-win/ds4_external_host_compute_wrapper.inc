#ifdef DS4_EXTERNAL_DECODE
/* RAM-tier packed FP4 experts computed on the CPU (LLAMADECODE_HOST_MODE=cpu|mix).
 *
 * Per layer: the router ids, weights and FFN input come back in one D2H (the
 * same single sync the id read already cost). RAM-resident experts chosen for
 * the CPU are handed to a persistent AVX-512 pool and computed in place from
 * the locked tier, while this thread runs the expert hook (DirectStorage
 * misses, uploads) and queues the GPU experts. The packed launcher joins the
 * CPU lane just before the down projection: it waits only if the CPU job is
 * still running, uploads each CPU expert's per-row sums, and the down kernel
 * adds them in slot order (the GPU's own summation order). No per-expert
 * waits on the critical path; one join per layer. */
extern "C" int ds4_cpu_fp4_init(int threads);
extern "C" int ds4_cpu_fp4_submit(const float *x, const uint8_t *const *gate, const uint8_t *const *up,
                                  const uint8_t *const *down, const float *w, int k, float clamp, float *const *acc);
extern "C" double ds4_cpu_fp4_wait(void);
struct external_fp4_pinned { int32_t ids[8]; float w[8]; float x[5120]; float acc[6][5120]; };
static external_fp4_pinned *g_fp4_pin = nullptr;
static int g_fp4_state = 0;               // 0 unset, 1 ready, -1 refused
static uint32_t g_fp4_positions[6], g_fp4_count = 0;
static uint32_t g_fp4_borrowed[6];
static double g_fp4_wait_last = 0;
/* Mix cost model (ms per expert, exponential averages measured in place). */
static double g_fp4_t_cpu = 0.35, g_fp4_t_ds = 1.0, g_fp4_t_up = 1.0;
static uint64_t g_fp4_calls = 0, g_fp4_cpu_experts = 0, g_fp4_up_experts = 0, g_fp4_cpu_layers = 0, g_fp4_explore = 0;
static double g_fp4_service_ms = 0, g_fp4_wait_ms = 0, g_fp4_read_ms = 0, g_fp4_join_ms = 0;
extern "C" void ds4_cpu_fp4_set_epilogue(int (*fn)(void *, const float *, const float *, int, float, float *), void *ctx);

/* SwiGLU epilogue of ds4_fp4p_gate_up_kernel, same source expression: the
 * CPU's raw gate/up rows get the GPU's own expf (ex2.approx), so the CPU
 * lane's mid, and hence its output, matches the GPU bit for bit. */
struct fp4_lane_weights { float w[6]; };
__global__ static void fp4_lane_epilogue_kernel(const float *__restrict__ gu, float *__restrict__ mid,
                                                uint32_t n, float clamp, fp4_lane_weights weights) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t e = i / 2304u, row = i - e * 2304u;
    float gate = gu[(uint64_t)e * 4608u + row], up = gu[(uint64_t)e * 4608u + 2304u + row];
    if (!isfinite(gate)) gate = 0.0f;
    if (!isfinite(up)) up = 0.0f;
    if (clamp > 1.0e-6f) {
        if (gate > clamp) gate = clamp;
        if (up > clamp) up = clamp;
        if (up < -clamp) up = -clamp;
    }
    const float silu = gate / (1.0f + expf(-gate));
    mid[i] = silu * up * weights.w[e];
}
struct fp4_lane_epi { cudaStream_t stream; float *gu_h, *mid_h, *gu_d, *mid_d; };
static fp4_lane_epi g_fp4_epi = {};
/* Runs on CPU worker 0 while the decode thread is in the expert hook: its own
 * non-blocking stream, so it neither waits for nor delays the decode stream. */
static int fp4_lane_epilogue(void *, const float *gu, const float *w, int k, float clamp, float *mid) {
    auto &E = g_fp4_epi;
    const uint32_t n = (uint32_t)k * 2304u;
    std::memcpy(E.gu_h, gu, (size_t)k * 4608u * sizeof(float));
    fp4_lane_weights weights{};
    for (int e = 0; e < k; ++e) weights.w[e] = w[e];
    if (cudaMemcpyAsync(E.gu_d, E.gu_h, (size_t)k * 4608u * sizeof(float), cudaMemcpyHostToDevice, E.stream) != cudaSuccess) return 0;
    fp4_lane_epilogue_kernel<<<(n + 255u) / 256u, 256, 0, E.stream>>>(E.gu_d, E.mid_d, n, clamp, weights);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(E.mid_h, E.mid_d, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost, E.stream) != cudaSuccess ||
        cudaStreamSynchronize(E.stream) != cudaSuccess) return 0;
    std::memcpy(mid, E.mid_h, (size_t)n * sizeof(float));
    return 1;
}
static bool fp4_lane_epilogue_init() {
    const char *v = getenv("LLAMADECODE_CPU_FP4_SILU");
    if (v && !strcmp(v, "cpu")) { fprintf(stderr, "LLAMADECODE_CPU_FP4 silu=cpu (diagnostic: differs from the GPU in expf rounding)\n"); return true; }
    auto &E = g_fp4_epi;
    const size_t gu = 6u * 4608u * sizeof(float), md = 6u * 2304u * sizeof(float);
    if (cudaStreamCreateWithFlags(&E.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaHostAlloc((void **)&E.gu_h, gu, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void **)&E.mid_h, md, cudaHostAllocDefault) != cudaSuccess ||
        cudaMalloc((void **)&E.gu_d, gu) != cudaSuccess || cudaMalloc((void **)&E.mid_d, md) != cudaSuccess) {
        fprintf(stderr, "LLAMADECODE_CPU_FP4 epilogue setup refused: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    ds4_cpu_fp4_set_epilogue(fp4_lane_epilogue, nullptr);
    return true;
}

static int external_fp4_cpu_join(void *, cudaStream_t stream) {
    const auto t = std::chrono::steady_clock::now();
    const double service = ds4_cpu_fp4_wait();
    g_fp4_wait_last = external_ms(t);
    if (service < 0) return 0;
    g_fp4_service_ms += service;
    if (g_fp4_count) g_fp4_t_cpu += 0.05 * (service / g_fp4_count - g_fp4_t_cpu);
    for (uint32_t i = 0; i < g_fp4_count; ++i) {
        const uint32_t a = g_fp4_positions[i];
        if (!cuda_ok(cudaMemcpyAsync(g_fp4_cpu_acc_dev + (size_t)a * 5120u, g_fp4_pin->acc[a], 5120u * sizeof(float),
                                     cudaMemcpyHostToDevice, stream), "CPU expert sums upload")) return 0;
    }
    g_fp4_join_ms += external_ms(t);
    return 1;
}

static int external_fp4_threads() {
    const char *v = getenv("LLAMADECODE_CPU_FP4_THREADS");
    const long n = v && *v ? strtol(v, nullptr, 10) : 12;
    return (int)(n < 1 ? 1 : n > 32 ? 32 : n);
}

/* Returns -1 when this call is not for the CPU lane (caller runs the GPU path). */
static int external_fp4_cpu_launch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down,
        const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim,
        uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert,
        uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens,
        int allow_streaming, int owned_filtered) {
    const int mode = external_cpu_mode();
    if (!mode || !g_ds4_fp4_packed_experts || gate_type != 39u || down_type != 39u || n_tokens != 1u || n_expert != 6u ||
        expert_in_dim != 5120u || expert_mid_dim != 2304u || out_dim != 5120u || layer_index >= 40u ||
        !g_ds4x_expert_hook || !g_ds4x_residency_hook || !g_external_host_cache.valid ||
        gate_expert_bytes != 2304ull * 5120ull / 32ull * 17ull || down_expert_bytes != 5120ull * 2304ull / 32ull * 17ull)
        return -1;
    if (g_fp4_state == 0) {
        const int threads = external_fp4_threads();
        g_fp4_state = (ds4_cpu_fp4_init(threads) > 0 &&
                       cudaHostAlloc((void **)&g_fp4_pin, sizeof(external_fp4_pinned), cudaHostAllocDefault) == cudaSuccess &&
                       cudaMalloc((void **)&g_fp4_cpu_acc_dev, 6u * 5120u * sizeof(float)) == cudaSuccess &&
                       fp4_lane_epilogue_init()) ? 1 : -1;
        fprintf(stderr, "LLAMADECODE_CPU_FP4 %s threads=%d mode=%s arithmetic=packed-q8_1-replica+gpu-swiglu tier_slots=%u\n",
                g_fp4_state > 0 ? "ready" : "REFUSED", threads, mode == 1 ? "cpu" : "mix", (unsigned)g_external_host_cache.capacity);
        if (g_fp4_state < 0) fprintf(stderr, "LLAMADECODE_CPU_FP4 refused: RAM-tier experts stay on the upload path\n");
    }
    if (g_fp4_state < 0) return -1;
    if (g_external_host_borrowed.size() != g_external_host_cache.capacity) g_external_host_borrowed.assign(g_external_host_cache.capacity, 0);
    const cudaStream_t stream = cuda_decode_stream();
    const auto t_read = std::chrono::steady_clock::now();
    // One sync: ids, weights and the FFN input (the id read alone cost the same sync).
    if (!cuda_ok(cudaMemcpyAsync(g_fp4_pin->ids, selected->ptr, 6u * sizeof(int32_t), cudaMemcpyDeviceToHost, stream), "fp4 ids") ||
        !cuda_ok(cudaMemcpyAsync(g_fp4_pin->w, weights->ptr, 6u * sizeof(float), cudaMemcpyDeviceToHost, stream), "fp4 weights") ||
        !cuda_ok(cudaMemcpyAsync(g_fp4_pin->x, x->ptr, 5120u * sizeof(float), cudaMemcpyDeviceToHost, stream), "fp4 x") ||
        !cuda_ok(cudaStreamSynchronize(stream), "fp4 router readback")) return 0;
    g_fp4_read_ms += external_ms(t_read);
    ++g_fp4_calls;
    const int32_t *ids = g_fp4_pin->ids;
    std::memcpy(g_fp4_pre_ids, ids, sizeof(g_fp4_pre_ids));
    g_fp4_pre_ids_src = selected->ptr;
    bool any = false;
    for (uint32_t i = 0; i < 6u; ++i) {
        if (ids[i] < 0 || ids[i] >= 384) { g_fp4_pre_ids_src = nullptr; return -1; }
        if (external_host_find(layer_index, (uint32_t)ids[i]) >= 0) any = true;
    }
#define FP4_ARGS out,gate,up,mid,down,model_map,model_size,gate_offset,up_offset,down_offset,gate_type,down_type,gate_expert_bytes,gate_row_bytes,down_expert_bytes,down_row_bytes,expert_in_dim,expert_mid_dim,out_dim,selected,weights,n_total_expert,n_expert,clamp,x,layer_index,n_tokens,allow_streaming,owned_filtered
    std::array<uint8_t, 384> tier{};
    uint32_t cand[6], n_cand = 0, n_ssd = 0;
    if (any) {
        if (!g_ds4x_residency_hook(g_ds4x_residency_user, layer_index, tier.data(), 384u)) return 0;
        for (uint32_t i = 0; i < 6u; ++i) {
            if (tier[ids[i]] == 2) cand[n_cand++] = i;
            else if (tier[ids[i]] == 0) ++n_ssd;
        }
    }
    // The hook always serves at least one expert (an empty request is not a
    // provider case), so six RAM hits keep one on the upload path.
    uint32_t n_cpu = n_cand > 5u ? 5u : n_cand;
    if (mode == 2 && n_cpu) {
        // Choose how many RAM hits the CPU takes: minimize the later of the two
        // lanes, CPU (n_cpu * t_cpu) vs hook (misses * t_ds + uploads * t_up).
        // 1 in 16 such layers explores (all CPU or all upload) to keep both measured.
        if ((g_fp4_calls & 15u) == 0) { n_cpu = (g_fp4_explore++ & 1u) ? 0u : n_cpu; }
        else {
            uint32_t best = 0; double best_t = 1e30;
            for (uint32_t c = 0; c <= n_cpu; ++c) {
                const double tc = c * g_fp4_t_cpu, th = n_ssd * g_fp4_t_ds + (n_cand - c) * g_fp4_t_up;
                const double t = tc > th ? tc : th;
                if (t < best_t - 1e-9) { best_t = t; best = c; }
            }
            n_cpu = best;
        }
    }
    if (!n_cpu) {
        const int ok = routed_moe_launch_impl(FP4_ARGS);
        // Misses and uploads are measured from layers without a CPU lane.
        if (n_cand == 0 && n_ssd) g_fp4_t_ds += 0.05 * (g_last_hook_ms / n_ssd - g_fp4_t_ds);
        else if (n_cand) { const double r = (g_last_hook_ms - n_ssd * g_fp4_t_ds) / n_cand; if (r > 0) g_fp4_t_up += 0.05 * (r - g_fp4_t_up); }
        g_fp4_up_experts += n_cand;
        return ok;
    }
    const uint8_t *pg[6], *pu[6], *pd[6]; float w[6]; float *acc[6];
    g_external_cpu_ids.clear();
    for (uint32_t c = 0; c < n_cpu; ++c) {
        const uint32_t a = cand[c];
        const int slot = external_host_find(layer_index, (uint32_t)ids[a]);
        if (slot < 0) return 0;
        pg[c] = (const uint8_t *)external_host_payload((uint32_t)slot, 0);
        pu[c] = (const uint8_t *)external_host_payload((uint32_t)slot, 1);
        pd[c] = (const uint8_t *)external_host_payload((uint32_t)slot, 2);
        if (!pg[c] || !pu[c] || !pd[c]) return 0;
        g_external_host_borrowed[(uint32_t)slot] = 1; g_fp4_borrowed[c] = (uint32_t)slot;
        w[c] = g_fp4_pin->w[a]; acc[c] = g_fp4_pin->acc[a]; g_fp4_positions[c] = a;
        g_external_cpu_ids.push_back(ids[a]);
    }
    g_fp4_count = n_cpu;
    if (!ds4_cpu_fp4_submit(g_fp4_pin->x, pg, pu, pd, w, (int)n_cpu, clamp, acc)) {
        for (uint32_t c = 0; c < n_cpu; ++c) g_external_host_borrowed[g_fp4_borrowed[c]] = 0;
        g_external_cpu_ids.clear(); g_fp4_count = 0; return 0;
    }
    g_fp4_cpu_active = true;
    g_fp4_wait_last = 0;
    int ok = routed_moe_launch_impl(FP4_ARGS);
    g_fp4_cpu_active = false;
    (void)ds4_cpu_fp4_wait();            // no-op after the join; covers a failed launch
    for (uint32_t c = 0; c < n_cpu; ++c) g_external_host_borrowed[g_fp4_borrowed[c]] = 0;
    g_external_cpu_ids.clear();
    g_fp4_wait_ms += g_fp4_wait_last;
    g_fp4_cpu_experts += n_cpu; g_fp4_up_experts += n_cand - n_cpu; ++g_fp4_cpu_layers;
    g_external_cpu_ms = g_fp4_service_ms; g_external_wait_ms = g_fp4_wait_ms;
    g_external_transfer_ms = g_fp4_read_ms; g_external_merge_ms = g_fp4_join_ms;
    g_external_cpu_experts = g_fp4_cpu_experts; g_external_cpu_layers = g_fp4_cpu_layers;
    if ((g_fp4_cpu_layers & 1023u) == 1u)
        fprintf(stderr, "LLAMADECODE_CPU_FP4_STATS calls=%llu cpu_layers=%llu cpu_experts=%llu upload_experts=%llu t_cpu=%.3f t_ds=%.3f t_up=%.3f service_ms=%.1f wait_ms=%.1f read_ms=%.1f join_ms=%.1f borrow_skips=%llu\n",
                (unsigned long long)g_fp4_calls, (unsigned long long)g_fp4_cpu_layers, (unsigned long long)g_fp4_cpu_experts,
                (unsigned long long)g_fp4_up_experts, g_fp4_t_cpu, g_fp4_t_ds, g_fp4_t_up, g_fp4_service_ms, g_fp4_wait_ms,
                g_fp4_read_ms, g_fp4_join_ms, (unsigned long long)g_external_host_borrow_skips);
    if (!ok) return 0;
    // Diagnostic (LLAMADECODE_HOST_CPU_VERIFY=1|all): rerun this layer with every
    // expert on the GPU (RAM hits uploaded) on the same activation and compare;
    // the hybrid result is kept.
    static const char *verify = getenv("LLAMADECODE_HOST_CPU_VERIFY");
    static unsigned verify_count = 0;
    if (verify && *verify && strcmp(verify, "0") && (!strcmp(verify, "all") || verify_count < 64u)) {
        ++verify_count;
        std::vector<float> hybrid(5120), reference(5120);
        if (!cuda_ok(cudaStreamSynchronize(stream), "fp4 verify") ||
            !ds4_gpu_tensor_read(out, 0, hybrid.data(), 5120u * sizeof(float))) return 0;
        std::memcpy(g_fp4_pre_ids, ids, sizeof(g_fp4_pre_ids)); g_fp4_pre_ids_src = selected->ptr;
        if (!routed_moe_launch_impl(FP4_ARGS) || !cuda_ok(cudaStreamSynchronize(stream), "fp4 verify ref") ||
            !ds4_gpu_tensor_read(out, 0, reference.data(), 5120u * sizeof(float))) return 0;
        double diff = 0, norm = 0, mx = 0; unsigned same = 0;
        for (uint32_t i = 0; i < 5120u; ++i) {
            const double d = (double)hybrid[i] - reference[i];
            diff += d * d; norm += (double)reference[i] * reference[i]; mx = std::max(mx, std::abs(d));
            same += std::memcmp(&hybrid[i], &reference[i], 4) == 0;
        }
        fprintf(stderr, "LLAMADECODE_CPU_FP4_VERIFY n=%u layer=%u cpu_experts=%u bit_identical=%u/5120 relative_rms=%.3e max_abs=%.3e\n",
                verify_count, layer_index, n_cpu, same, sqrt(diff / std::max(norm, 1e-30)), mx);
        if (!ds4_gpu_tensor_write(out, 0, hybrid.data(), 5120u * sizeof(float))) return 0;
    }
    return 1;
#undef FP4_ARGS
}
#endif
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
    {const int fp4=external_fp4_cpu_launch(DS4_EXTERNAL_MOE_ARGS);if(fp4>=0)return fp4;}
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
