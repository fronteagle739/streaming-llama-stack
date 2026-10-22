#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-new-mma.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

static bool ggml_cuda_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value && value[0] != '\0' && value[0] != '0';
}

static int ggml_cuda_env_int(const char * name, int fallback = 0) {
    const char * value = getenv(name);
    return value && value[0] != '\0' ? atoi(value) : fallback;
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if ((turing_mma_available(cc) || amd_wmma_available(cc)) && Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    // sm_120 (RTX 5090 consumer Blackwell) nominally has 99 KB max shared memory per block.
    // MLA FA (DKQ=576) at ncols=64 needs ~99 KB — very close. Historically capped at
    // ncols=32 to be safe, but that halves tile size and ~2x attention cost for MLA.
    // Iter 14d: attempt ncols=64 with cudaFuncSetAttribute raising the opt-in limit
    // to 100 KB. Set LLAMA_FA_SM120_NCOLS32=1 to revert to the safe cap.
    static const bool sm120_force_cap = []() {
        const char * env = std::getenv("LLAMA_FA_SM120_NCOLS32");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    const bool sm120_shmem_limit = sm120_force_cap &&
        (cc >= GGML_CUDA_CC_BLACKWELL) && (DKQ == 576);

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || amd_wmma_available(cc) || Q->ne[1] <= 32/ncols2 || sm120_shmem_limit) {
        if (sm120_shmem_limit) {
            // Log only once per session to avoid 36M-line log spam
            static bool logged_once = false;
            if (!logged_once) {
                fprintf(stderr, "*** FA_NCOLS: sm120 MLA cap — ncols1=%d ncols2=%d (DKQ=%d cc=%d Q.ne1=%lld)\n",
                        32/ncols2, ncols2, DKQ, cc, (long long)Q->ne[1]);
                logged_once = true;
            }
        }
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    // MLA (dkq=576): force GQA opt even with unaligned K->ne[1].
    // Only GQA instances exist for dkq=576. The mask tensor is padded to 256
    // in the graph builder, and out-of-range K entries are zero in the KV cache.
    // NOMASK(2026-05-30): drop the `mask` requirement so the MLA (dkq=576) GQA path (ncols2>1) is
    // selected even with a null mask (direct_causal inline-mask, >160k unlock). Behavior-identical when
    // mask present. The ncols2>1 selection is required so the kernel's mask-APPLY sites run + use inline m.
    if (K->ne[0] == 576 && max_bias == 0.0f) {
        use_gqa_opt = true;
    }
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // QUERY-TILE CAP (env LLAMA_FA_QCHUNK=N, e.g. 2048). FIX for the multi-ubatch
    // streaming-prefill bug (2026-05-29): the NCOLS=32 MLA FA kernel mishandles the
    // (correct) causal mask when a call's query count > ~2048 AND the KV cache has
    // unfilled cells (non-last ubatch). Splitting the FA into <=N-query launches makes
    // each launch q<=N (the regime that is always correct), and is mathematically
    // identical since queries are independent. KQV->ne={DV,n_head,n_q(=ne[2]),n_batch}
    // (ggml.c:5380); Q & mask query dim = ne[1]. Recurse per chunk (each chunk q<=N
    // skips this guard -> normal dispatch). Touches ONLY the FA launch (gallocr-safe).
    {
        static const int qchunk = ggml_cuda_env_int("LLAMA_FA_QCHUNK");
        if (qchunk > 0 && mask && Q->ne[1] > (int64_t) qchunk && dst->ne[2] == Q->ne[1]) {
            ggml_tensor * Qm = dst->src[0];
            ggml_tensor * Mm = dst->src[3];
            const int64_t total_q = Qm->ne[1];
            char * const q0 = (char *) Qm->data;  const size_t qnb1 = Qm->nb[1];
            char * const o0 = (char *) dst->data;  const size_t onb2 = dst->nb[2];
            char * const m0 = (char *) Mm->data;  const size_t mnb1 = Mm->nb[1];
            const int64_t sq = Qm->ne[1], so = dst->ne[2], sm = Mm->ne[1];
            static bool logged = false;
            if (!logged) {
                fprintf(stderr, "*** FA_QCHUNK: total_q=%lld chunk=%d (KQV.ne2=%lld mask.ne1=%lld)\n",
                        (long long) total_q, qchunk, (long long) so, (long long) sm);
                fflush(stderr); logged = true;
            }
            for (int64_t c0 = 0; c0 < total_q; c0 += qchunk) {
                const int64_t cs = (total_q - c0 < (int64_t) qchunk) ? (total_q - c0) : (int64_t) qchunk;
                Qm->ne[1] = cs;  Qm->data = q0 + (size_t) c0 * qnb1;
                dst->ne[2] = cs; dst->data = o0 + (size_t) c0 * onb2;
                Mm->ne[1] = cs;  Mm->data = m0 + (size_t) c0 * mnb1;
                ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);  // recurse: cs <= qchunk -> normal path
            }
            Qm->ne[1] = sq;  Qm->data = q0;
            dst->ne[2] = so; dst->data = o0;
            Mm->ne[1] = sm;  Mm->data = m0;
            return;
        }
    }

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 512:
            // DeepSeek V4 raw SWA attention has 64 Q heads per K/V head. Keep
            // ncols2 fixed at 8 so only the query-count variants needed by
            // prefill and decode are instantiated.
            GGML_ASSERT(V->ne[0] == 512);
            GGML_ASSERT(mask);
            GGML_ASSERT(Q->ne[1] > 4);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            GGML_ASSERT((Q->ne[2] / K->ne[2]) % 8 == 0);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<512, 512, 8>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            const int force_ncols1 = ggml_cuda_env_int("LLAMA_FA_MLA_MMA_NCOLS1");
            switch (force_ncols1) {
                case 4:
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                case 16:
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                    break;
                case 32:
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                    break;
                default:
                    break;
            }
            if (force_ncols1 == 4 || force_ncols1 == 16 || force_ncols1 == 32) {
                break;
            }
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512 && V->ne[0] != 576) {
                return BEST_FATTN_KERNEL_NONE;
            }
            // Relax stride requirement: mask is padded to 256 in graph builder.
            // NOMASK(2026-05-30): allow a NULL mask for MLA under the opt-in env (direct_causal
            // inline-mask path, >160k unlock). Env-only gate (stable across sched_reserve where
            // op_params[4] may be 0); the graph only nulls the mask for the gated MLA+direct path.
            {
                static const bool nomask_env = (std::getenv("LLAMA_FA_MLA_CAUSAL_DIRECT_NOMASK") != nullptr);
                if (gqa_ratio < 2 || (!mask && !nomask_env) || max_bias != 0.0f) {
                    return BEST_FATTN_KERNEL_NONE;
                }
            }
            gqa_opt_applies = true;
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    // The full-width 576/576 shape has no MMA specialization.
    if (K->ne[0] == 576 && V->ne[0] == 576) {
        return BEST_FATTN_KERNEL_TILE;
    }
    if (K->ne[0] == 512 && V->ne[0] == 512) {
        // DeepSeek V4 raw SWA attention: the dense MMA path supports its
        // strided F32 Q and F32 sinks, but the compact instance set assumes
        // eight-way GQA tiles. Keep all other 512/512 variants on tile.
        const bool dsv4_mma_applies =
            turing_mma_available(cc) &&
            K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 &&
            gqa_opt_applies && gqa_ratio % 8 == 0 &&
            Q->ne[1] > 4; // Tile is faster for single-query DSV4 decode.
        if (!dsv4_mma_applies || ggml_cuda_env_enabled("LLAMA_FA_DSV4_FORCE_TILE")) {
            return BEST_FATTN_KERNEL_TILE;
        }
    }
    if (K->ne[0] == 576 && ggml_cuda_env_enabled("LLAMA_FA_MLA_FORCE_TILE")) {
        return BEST_FATTN_KERNEL_TILE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 576) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            } else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 8) {
            return BEST_FATTN_KERNEL_TILE; // AMD WMMA is only faster if the full tile width of 16 can be utilized.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    // Phase 97 probe: dump mask shape + first/last fp16 elements for the
    // first N FA calls (= layer 0..N-1) when LLAMA_FA_MASK_PROBE=1.
    {
        static int fa_probe_count = 0;
        static const char * fa_probe_env = std::getenv("LLAMA_FA_MASK_PROBE");
        static const bool fa_probe = fa_probe_env && fa_probe_env[0] != '\0' && fa_probe_env[0] != '0';
        // [2026-08-28] Gate on the SHAPE, not just a call count, and print the tensor NAME.
        //
        // A bare "first 5 calls" cap samples whatever runs first, which is the n_ubatch-sized warmup
        // or max graph -- three separate diagnostics were confounded this way in one session
        // (COMP_DUMP, MASKFILL_ECHO, and this one). The real streaming-prefill passes are the SMALL
        // ones, so `LLAMA_FA_MASK_PROBE=<max_q>` fires only when Q->ne[1] <= max_q (bare "1" keeps
        // the old behaviour via the 4096 default).
        //
        // The NAME carries the layer suffix, which is the point: DSV4's raw and compressed attention
        // both reach this kernel in the SAME graph, so printing `il` lets them be compared WITHIN ONE
        // RUN. That is the only controlled comparison available -- comparing across runs assumes both
        // sampled the same graph, and nothing establishes that.
        //
        // Reminder for whoever reads the output: the SHAPES here are reliable, the mask BYTES are
        // not (this fires at dispatch and the cudaMemcpy below is unordered against ggml's streams).
        // Use LLAMA_DLM_NODE_NAN_SCAN for mask CONTENT.
        static const int64_t fa_probe_max_q = []() {
            const char * e = std::getenv("LLAMA_FA_MASK_PROBE");
            const int64_t v = (e && e[0]) ? (int64_t) atoll(e) : 0;
            return (v > 1) ? v : 4096;   // "1" (or unset-but-enabled) = no shape restriction
        }();
        static const int fa_probe_cap = []() {
            const char * e = std::getenv("LLAMA_FA_MASK_PROBE_N");
            const int v = (e && e[0]) ? atoi(e) : 0;
            return v > 0 ? v : 5;
        }();
        if (fa_probe && fa_probe_count < fa_probe_cap &&
                dst->src[0] && dst->src[0]->ne[1] <= fa_probe_max_q) {
            const ggml_tensor * Q = dst->src[0];
            const ggml_tensor * K = dst->src[1];
            const ggml_tensor * mask = dst->src[3];
            fprintf(stderr, "*** FA_PROBE call=%d dst='%s' Q.ne=[%lld,%lld,%lld,%lld] K.ne=[%lld,%lld,%lld,%lld]",
                    fa_probe_count, ggml_get_name(dst),
                    (long long)Q->ne[0], (long long)Q->ne[1], (long long)Q->ne[2], (long long)Q->ne[3],
                    (long long)K->ne[0], (long long)K->ne[1], (long long)K->ne[2], (long long)K->ne[3]);
            if (mask && mask->data) {
                fprintf(stderr, " mask.ne=[%lld,%lld,%lld,%lld] mask.type=%d data=%p",
                        (long long)mask->ne[0], (long long)mask->ne[1], (long long)mask->ne[2], (long long)mask->ne[3],
                        (int)mask->type, mask->data);
                // Read first 4 + last 4 fp16 values from mask
                size_t ne = (size_t)mask->ne[0] * (size_t)mask->ne[1] * (size_t)mask->ne[2] * (size_t)mask->ne[3];
                if (ne >= 8) {
                    uint16_t buf_first[4] = {0};
                    uint16_t buf_last[4] = {0};
                    cudaMemcpy(buf_first, mask->data, sizeof(buf_first), cudaMemcpyDeviceToHost);
                    cudaMemcpy(buf_last, (const uint8_t*)mask->data + (ne - 4)*sizeof(uint16_t), sizeof(buf_last), cudaMemcpyDeviceToHost);
                    fprintf(stderr, " first4=%04x,%04x,%04x,%04x last4=%04x,%04x,%04x,%04x",
                            buf_first[0], buf_first[1], buf_first[2], buf_first[3],
                            buf_last[0], buf_last[1], buf_last[2], buf_last[3]);
                }
            } else {
                fprintf(stderr, " mask=NULL");
            }
            fprintf(stderr, "\n");
            fflush(stderr);
            fa_probe_count++;
        }
    }
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16: {
            // LLAMA_FA_NEW_MMA=1: route DKQ=576 MLA cases to the ik_llama new-MMA kernel.
            static const bool use_new_mma = ggml_cuda_env_enabled("LLAMA_FA_NEW_MMA");
            const ggml_tensor * K = dst->src[1];
            if (use_new_mma && K->ne[0] == 576) {
                ggml_cuda_flash_attn_ext_mma_new_case(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            }
            break;
        }
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
