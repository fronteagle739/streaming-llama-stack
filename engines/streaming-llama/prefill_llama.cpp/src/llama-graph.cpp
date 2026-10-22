#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <unordered_set>

static void llm_set_compact_2d_shape(struct ggml_tensor * t, int64_t ne0, int64_t ne1) {
    if (t == nullptr || ne0 <= 0 || ne1 <= 0) {
        return;
    }

    t->ne[0] = ne0;
    t->ne[1] = ne1;

    t->nb[0] = ggml_type_size(t->type);
    t->nb[1] = ggml_row_size(t->type, t->ne[0]);
    t->nb[2] = t->nb[1] * t->ne[1];
    t->nb[3] = t->nb[2] * t->ne[2];
}

static void llm_set_attn_mask_shape(struct ggml_tensor * mask, struct ggml_tensor * mask_cnv, int64_t n_kv, int64_t n_tokens) {
    if (mask == nullptr) {
        return;
    }

    const int64_t n_stream = mask->ne[3];
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens % n_stream == 0);

    const int64_t n_mask_rows = n_tokens / n_stream;

    // The leaf and the GPU input_cpy can be allocated at different ne by
    // the scheduler (per-ubatch graphs in streaming-prefill produce src/dst
    // pairs with different graph-time max sizes). Setting ne here for the
    // leaf alone makes the scheduler's tensor_copy assert "layout mismatch"
    // when src.nb (from leaf alloc) != dst.nb (from input_cpy alloc). We
    // leave both leaves at their graph-time shape and let the downstream
    // FA mask reshape (arm_fa_patch in llama-context.cpp) shrink only the
    // FA-side mask tensor (after the CAST). set_input_kq_mask fills the
    // entire leaf with proper causal + padding values; the unused padding
    // beyond the visible region is harmless because FA reads only the
    // compact region.
    (void) n_kv;
    (void) n_mask_rows;
    (void) mask_cnv;
}

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        // GLM-Image image-token GENERATION provides FULL 4D M-RoPE positions per token (a 2D
        // image grid: distinct h/w per token). The default text-token path below collapses all
        // sections to the temporal position (correct for text where t=h=w=pos, but it would
        // discard the 2D grid h/w). When LLAMA_MROPE_FULL_POS is set, use the caller's full 4D
        // positions verbatim (section-major [t,h,w,extra]).
        static const bool mrope_full_pos = (getenv("LLAMA_MROPE_FULL_POS") != nullptr);
        if (ubatch->token && n_pos_per_embd == 4 && !mrope_full_pos) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    // 2026-04-21 (iter 10): When the mega-dispatch path skips get_rows for
    // external per-layer tensors, inp_out_ids is not referenced by any graph
    // node and the scheduler does not allocate a buffer for it.  In that case
    // the tensor is unused — no host-side initialization is needed.
    if (!out_ids->buffer) {
        return;
    }

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN) {
        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && arch == LLM_ARCH_QWEN3) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= s_copy_extra->ne[0] == mctx->get_n_rs() - params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

static void print_mask(const float * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swq_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = data[i * n_kv + j];
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](float * data, int n_swa, llama_swa_type swa_type) {
        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f;
            }
        }
    };

    {
        GGML_ASSERT(self_kq_mask);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));

        float * data = (float *) self_kq_mask->data;

        std::fill(data, data + ggml_nelements(self_kq_mask), -INFINITY);

        fill_mask(data, 0, LLAMA_SWA_TYPE_NONE);

        if (debug) {
            print_mask(data, n_tokens, n_kv, 0, LLAMA_SWA_TYPE_NONE);
        }
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));

        float * data = (float *) self_kq_mask_swa->data;

        std::fill(data, data + ggml_nelements(self_kq_mask_swa), -INFINITY);

        fill_mask(data, hparams.n_swa, hparams.swa_type);

        if (debug) {
            print_mask(data, n_tokens, n_kv, hparams.n_swa, hparams.swa_type);
        }
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    llm_set_attn_mask_shape(self_kq_mask, self_kq_mask_cnv, mctx->get_n_kv(), ubatch->n_tokens);
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn,
                             cparams.wir_soft_mask_alpha, cparams.wir_soft_mask_tau);
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_kq_mask->ne[0] == mctx->get_n_kv();
    res &= self_kq_mask->ne[1] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    llm_set_attn_mask_shape(self_kq_mask, self_kq_mask_cnv, mctx->get_n_kv(), ubatch->n_tokens);
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn,
                             cparams.wir_soft_mask_alpha, cparams.wir_soft_mask_tau);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= self_kq_mask->ne[0] == mctx->get_n_kv();
    res &= self_kq_mask->ne[1] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    llm_set_attn_mask_shape(self_kq_mask, self_kq_mask_cnv, mctx->get_base()->get_n_kv(), ubatch->n_tokens);
    if (self_kq_mask_swa) {
        llm_set_attn_mask_shape(self_kq_mask_swa, self_kq_mask_swa_cnv, mctx->get_swa()->get_n_kv(), ubatch->n_tokens);
    }
    mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);

    mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);

    mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_kq_mask->ne[0] == mctx->get_base()->get_n_kv();
    res &= self_kq_mask->ne[1] == params.ubatch.n_tokens;

    res &= self_kq_mask_swa->ne[0] == mctx->get_swa()->get_n_kv();
    res &= self_kq_mask_swa->ne[1] == params.ubatch.n_tokens;

    return res;
}

//
// DeepSeek-V4 compressed-KV / lightning-indexer graph input (ported from upstream dsv4).
//

static bool dsv4_remainder_timers_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("LLAMA_DLM_REMAINDER_TIMERS");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

static thread_local llama_dlm_dsv4_input_stats g_dsv4_input_stats;

void llama_dlm_dsv4_input_timers_reset() {
    g_dsv4_input_stats = {};
}

llama_dlm_dsv4_input_stats llama_dlm_dsv4_input_timers_get() {
    return g_dsv4_input_stats;
}

static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) plan.n_visible.size());
    GGML_ASSERT((int64_t) n_tokens <= dst->ne[1]);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == 1);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    float * data = (float *) dst->data;

    for (int64_t i = 0; i < dst->ne[1]; ++i) {
        const int32_t n_visible = plan.n_visible[i];

        for (int64_t j = 0; j < dst->ne[0]; ++j) {
            data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
        }
    }
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}

static bool dsv4_suffix_write_diag_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("LLAMA_DSV4_SUFFIX_WRITE_DIAG");
        return env && env[0] != '\0' && env[0] != '0';
    }();

    return enabled;
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        bool debug,
        uint32_t n_tokens) {
    const bool timers = dsv4_remainder_timers_enabled();
    const auto vectors_t0 = timers ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    dsv4_set_i32(inp.state_idxs, plan.state_idxs);
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_i32(inp.state_write_end, plan.state_write_end);
    dsv4_set_i32(inp.n_visible,  plan.n_visible);
    if (dsv4_suffix_write_diag_enabled()) {
        const long long first_write = plan.state_write_idxs.empty()
                ? -1LL : (long long) plan.state_write_idxs.front();
        LLAMA_LOG_INFO("DSV4 suffix-write diag: set-input %s n_tokens=%u write=%zu "
                       "first_write=%lld leaf=%p buffer=%p\n",
                name, n_tokens, plan.state_write_idxs.size(), first_write,
                (void *) inp.state_write_idxs,
                inp.state_write_idxs ? (void *) inp.state_write_idxs->buffer : nullptr);
    }
    if (timers) {
        g_dsv4_input_stats.comp_vector_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - vectors_t0).count();
    }

    const auto mask_t0 = timers ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens);
    if (timers) {
        g_dsv4_input_stats.comp_mask_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - mask_t0).count();
        if (inp.kq_mask) {
            g_dsv4_input_stats.comp_mask_elems += (uint64_t) ggml_nelements(inp.kq_mask);
        }
    }

    if (debug || dsv4_compress_debug()) {
        LLAMA_LOG_DEBUG("%s: %s ratio=%u, n_tokens=%u, state_persist_dst=%s, state_write_end=%s\n",
                __func__, name, plan.ratio, n_tokens,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_end).c_str());
    }
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens &&
           t->ne[2] == 1 &&
           t->ne[3] == 1;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_idxs, plan.state_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_end, plan.state_write_end.size());
    res &= dsv4_can_reuse_tensor_1d(inp.n_visible,  plan.n_visible.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

static void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name) {
    inp.state_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_idxs.size(), std::string("dsv4_") + name + "_state_idxs");
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");
    inp.state_write_end = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_end.size(), std::string("dsv4_") + name + "_state_write_end");
    inp.n_visible  = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.n_visible.size(), std::string("dsv4_") + name + "_n_visible");

    if (plan.n_kv > 0) {
        inp.kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, plan.n_kv, plan.n_visible.size(), 1, 1);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4::set_input(const llama_ubatch * ubatch) {
    const bool timers = dsv4_remainder_timers_enabled();
    const auto total_t0 = timers ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (timers) {
        ++g_dsv4_input_stats.calls;
    }

    // DSV4 deliberately makes the raw ISWA cache K-only, so build_inp_dsv4()
    // does not create self_v_idxs/self_v_idxs_swa.  Do not use the generic
    // ISWA set_input() here: it feeds both V-index tensors unconditionally and
    // would dereference those null DSV4 inputs on the layer-major staging path.
    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;

    llm_set_attn_mask_shape(inp_raw->self_kq_mask, inp_raw->self_kq_mask_cnv,
            raw_ctx->get_base()->get_n_kv(), ubatch->n_tokens);
    if (inp_raw->self_kq_mask_swa) {
        llm_set_attn_mask_shape(inp_raw->self_kq_mask_swa, inp_raw->self_kq_mask_swa_cnv,
                raw_ctx->get_swa()->get_n_kv(), ubatch->n_tokens);
    }

    raw_ctx->get_base()->set_input_k_idxs(inp_raw->self_k_idxs, ubatch);
    const auto raw_mask_t0 = timers ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    raw_ctx->get_base()->set_input_kq_mask(inp_raw->self_kq_mask, ubatch, cparams.causal_attn);

    raw_ctx->get_swa()->set_input_k_idxs(inp_raw->self_k_idxs_swa, ubatch);
    raw_ctx->get_swa()->set_input_kq_mask(inp_raw->self_kq_mask_swa, ubatch, cparams.causal_attn);

    // Fill the DSV4-owned copy with the SAME routine, so the causal+SWA rule is never
    // reimplemented -- only WHERE the tensor is consumed differs (see the declaration).
    //
    // set_input_kq_mask writes through a raw host pointer and asserts the buffer is host-resident.
    // Report residency for BOTH masks before filling: attempt 2 hit that assert and the message
    // does not say WHICH tensor, and my change also removed `_cnv`'s only consumer, which could
    // have left the leaf unconsumed and reallocated. Name them rather than guess.
    {
        static int dsv4_maskres_dbg = 0;
        if (dsv4_maskres_dbg < 4) {
            dsv4_maskres_dbg++;
            auto residency = [](const ggml_tensor * t) {
                if (!t)         return "ABSENT";
                if (!t->buffer) return "NO-BUFFER";
                return ggml_backend_buffer_is_host(t->buffer) ? "HOST" : "DEVICE";
            };
            fprintf(stderr, "*** DSV4_MASKRES: leaf=%s own=%s csa=%s\n",
                    residency(inp_raw->self_kq_mask_swa),
                    residency(raw_kq_mask),
                    residency(inp_csa.kq_mask));
            fflush(stderr);
        }
    }
    // Measured: leaf=HOST own=DEVICE csa=HOST. The scheduler places this tensor on the DEVICE
    // (its only consumer is a GPU op), so set_input_kq_mask -- which writes through a raw
    // `float *` and asserts host residency -- cannot fill it directly. That assert is what
    // crashed attempt 2.
    //
    // The leaf is host-resident and, at this point, already filled CORRECTLY by the call above
    // (verified on device: -inf=40896, == ds4's rule 256/256). Same dims, same dtype, so upload
    // its bytes with ggml_backend_tensor_set, which handles host->device. This copies the mask
    // ONCE PER set_input -- i.e. per ubatch, inside the staging the executor does run -- which is
    // exactly what the head-of-graph cast fails to do.
    // NOTE the `raw_kq_mask->buffer` check: when the consumer knob is OFF nothing reads this tensor,
    // so the allocator never assigns it a buffer (measured: `own=NO-BUFFER`) and
    // ggml_backend_tensor_set asserts "tensor buffer not set" -- which broke the BASELINE, not just
    // the experiment. A diagnostic must be inert when its feature is disabled.
    if (raw_kq_mask && raw_kq_mask->buffer &&
            inp_raw->self_kq_mask_swa &&
            inp_raw->self_kq_mask_swa->buffer &&
            ggml_backend_buffer_is_host(inp_raw->self_kq_mask_swa->buffer) &&
            ggml_nbytes(raw_kq_mask) == ggml_nbytes(inp_raw->self_kq_mask_swa)) {
        ggml_backend_tensor_set(raw_kq_mask, inp_raw->self_kq_mask_swa->data, 0,
                                ggml_nbytes(raw_kq_mask));

        // [MASKFILL_ECHO 2026-08-28] Read the tensor straight back and count -inf.
        //
        // Measured 2026-08-28: the fill above RUNS (MASKFILL_SKIP never fires, so every precondition
        // held) and the destination HAS a buffer (DSV4_MASKRES own=DEVICE) -- yet at compute time a
        // ggml_cont of this very tensor reads inf=0, while a ggml_cont of the LEAF it was copied
        // from, in the same layer range of the same graph, reads inf=40896.
        //
        // Two different bugs are consistent with that and they need different fixes:
        //     echo == 40896 -> the write LANDED and something later overwrites or re-points the
        //                      buffer (allocator reuse of a graph input, or a stale pointer from a
        //                      prior per-layer graph). Fix = protect/repoint the destination.
        //     echo == 0     -> the write never reached this buffer at all despite returning
        //                      normally. Fix = the copy itself (wrong buffer, wrong stream,
        //                      async-not-synchronised).
        // Counting -inf rather than dumping: the expected value is known exactly and independently
        // (a 128 window over 256x256 masks 65536-24640 = 40896), so one integer settles it.
        // Fire on the first several calls, not once: a single firing catches WARMUP, whose empty KV
        // cache legitimately masks everything (measured: n=1048576 inf=1048576 on a [1024,1024]
        // warmup graph). Print the shape so warmup and the real request can be told apart -- the
        // real one is [n_kv, n_tokens] with n_kv stepping 256 -> 512 across passes.
        // Gate on the SHAPE, not a call count. A plain "first N calls" cap is exhausted by warmup
        // and the max-ubatch graphs before the real request is ever reached: 8 firings all came back
        // ne=[1024,1024] (n_ubatch-sized) while the mask actually consumed at layer 42 is [256,256].
        // The real streaming-prefill passes are the small ones (n_kv 256 then 512), so fire only on
        // those. Third time today that a count-based cap captured warmup instead of the request.
        static int echoed = 0;
        if (echoed < 8 && raw_kq_mask->ne[0] > 0 && raw_kq_mask->ne[0] <= 512) {
            echoed++;
            const size_t n = ggml_nelements(raw_kq_mask);
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(raw_kq_mask, tmp.data(), 0, ggml_nbytes(raw_kq_mask));
            size_t n_inf = 0, n_nan = 0;
            for (size_t i = 0; i < n; ++i) {
                if (std::isinf(tmp[i])) n_inf++;
                else if (std::isnan(tmp[i])) n_nan++;
            }
            LLAMA_LOG_WARN("MASKFILL_ECHO: ne=[%lld,%lld] wrote %zu bytes to dsv4_raw_kq_mask, read back n=%zu "
                    "inf=%zu nan=%zu (leaf inf is the reference; equal => the write landed and is "
                    "lost later, 0 => the write never reached this buffer)\n",
                    (long long) raw_kq_mask->ne[0], (long long) raw_kq_mask->ne[1],
                    ggml_nbytes(raw_kq_mask), n, n_inf, n_nan);
        }
    } else if (raw_kq_mask && !raw_kq_mask->buffer) {
        // [MASKFILL_NOBUF 2026-08-28] The THIRD silent path. Both the fill and the MASKFILL_SKIP
        // warn below are guarded on `raw_kq_mask->buffer`, so a NULL destination buffer skips the
        // copy AND says nothing -- the same silence that hid the original defect for two months.
        //
        // With the consumer knob OFF this is the normal, expected state (nothing reads the tensor,
        // so the allocator never assigns it a buffer) and warning would be noise. So warn only when
        // the knob is ON, i.e. when someone is relying on this fill to happen.
        static const bool own_knob = []() {
            const char * e = std::getenv("LLAMA_DSV4_RAW_MASK_OWN");
            return e && e[0] != '\0' && e[0] != '0';
        }();
        static int nobuf_warned = 0;
        if (own_knob && nobuf_warned < 4) {
            nobuf_warned++;
            LLAMA_LOG_WARN("MASKFILL_NOBUF: LLAMA_DSV4_RAW_MASK_OWN is ON but dsv4_raw_kq_mask has "
                    "NO BUFFER on this graph (ne=[%lld,%lld]) -- the fill is skipped silently and "
                    "the tensor will read as ZEROS wherever it is consumed\n",
                    (long long) raw_kq_mask->ne[0], (long long) raw_kq_mask->ne[1]);
        }
    } else if (raw_kq_mask && raw_kq_mask->buffer) {
        // [MASKFILL_SKIP 2026-08-28] The knob is ON (the tensor has a buffer, so something reads it)
        // but a precondition failed and the copy above did NOT happen -- leaving the DSV4-owned mask
        // at whatever its buffer held, i.e. ZEROS. That is exactly the state measured on 2026-08-28:
        // with LLAMA_DSV4_RAW_MASK_OWN=1, NODE_SCAN read `csa_raw_mask-42 inf=0` while the leaf it
        // was supposed to be copied FROM read `inf=40896`. The fill had silently not run.
        //
        // This mattered far beyond one experiment: the standing conclusion in deepseek-v4.cpp
        // ("supplying the CORRECT mask produces DEGENERATE output, so the empty raw half is NOT a
        // defect") was drawn from runs on this path -- runs that never delivered a correct mask.
        // A guard that skips in silence turned a broken experiment into a false refutation.
        //
        // Warn ONCE with every quantity needed to tell the four failure modes apart. Deliberately
        // NOT fatal: this path is opt-in and must not take down a baseline that has the knob off.
        static bool warned = false;
        if (!warned) {
            warned = true;
            const ggml_tensor * leaf = inp_raw->self_kq_mask_swa;
            LLAMA_LOG_WARN("MASKFILL_SKIP: dsv4_raw_kq_mask NOT filled -- it will read as ZEROS. "
                    "leaf=%s leaf_buf=%s host=%s own_ne=[%lld,%lld] own_bytes=%zu "
                    "leaf_ne=[%lld,%lld] leaf_bytes=%zu\n",
                    leaf ? "present" : "NULL",
                    (leaf && leaf->buffer) ? "present" : "NULL",
                    (leaf && leaf->buffer) ? (ggml_backend_buffer_is_host(leaf->buffer) ? "yes" : "NO") : "n/a",
                    (long long) raw_kq_mask->ne[0], (long long) raw_kq_mask->ne[1],
                    ggml_nbytes(raw_kq_mask),
                    leaf ? (long long) leaf->ne[0] : -1, leaf ? (long long) leaf->ne[1] : -1,
                    leaf ? ggml_nbytes(leaf) : (size_t) 0);
        }
    }
    if (timers) {
        g_dsv4_input_stats.raw_mask_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - raw_mask_t0).count();
        if (inp_raw->self_kq_mask) {
            g_dsv4_input_stats.raw_mask_elems += (uint64_t) ggml_nelements(inp_raw->self_kq_mask);
        }
        if (inp_raw->self_kq_mask_swa) {
            g_dsv4_input_stats.raw_mask_elems += (uint64_t) ggml_nelements(inp_raw->self_kq_mask_swa);
        }
    }

    // fork adaptation: the raw iswa set_input does not handle the QK Hadamard rotation,
    // so fill the raw base/swa rotation matrices here (no-op when nullptr).
    if (inp_raw->self_k_rot) {
        raw_ctx->get_base()->set_input_k_rot(inp_raw->self_k_rot);
    }
    if (inp_raw->self_v_rot) {
        raw_ctx->get_base()->set_input_v_rot(inp_raw->self_v_rot);
    }
    if (inp_raw->self_k_rot_swa) {
        raw_ctx->get_swa()->set_input_k_rot(inp_raw->self_k_rot_swa);
    }
    if (inp_raw->self_v_rot_swa) {
        raw_ctx->get_swa()->set_input_v_rot(inp_raw->self_v_rot_swa);
    }

    dsv4_set_comp_inputs(inp_csa, mctx->get_csa_plan(), "csa", debug > 0, ubatch->n_tokens);
    dsv4_set_comp_inputs(inp_hca, mctx->get_hca_plan(), "hca", debug > 0, ubatch->n_tokens);
    dsv4_set_comp_inputs(inp_lid, mctx->get_lid_plan(), "lid", debug > 0, ubatch->n_tokens);

    if (inp_lid.k_rot && inp_lid.k_rot->buffer) {
        mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);
    }
    if (timers) {
        g_dsv4_input_stats.total_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - total_t0).count();
    }
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx_cur;
    inp_raw->mctx = mctx_cur->get_raw();

    bool res = true;

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == params.ubatch.n_tokens;
        res &= inp_raw->self_kq_mask &&
               inp_raw->self_kq_mask->ne[0] == mctx_cur->get_raw()->get_base()->get_n_kv() &&
               inp_raw->self_kq_mask->ne[1] == params.ubatch.n_tokens;
    }

    if (inp_raw->self_k_idxs_swa && inp_raw->self_k_idxs_swa->buffer) {
        res &= inp_raw->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
        res &= inp_raw->self_kq_mask_swa &&
               inp_raw->self_kq_mask_swa->ne[0] == mctx_cur->get_raw()->get_swa()->get_n_kv() &&
               inp_raw->self_kq_mask_swa->ne[1] == params.ubatch.n_tokens;
    }

    res &= dsv4_can_reuse_comp_input(inp_csa, mctx_cur->get_csa_plan(), params.ubatch.n_tokens);
    res &= dsv4_can_reuse_comp_input(inp_hca, mctx_cur->get_hca_plan(), params.ubatch.n_tokens);
    res &= dsv4_can_reuse_comp_input(inp_lid, mctx_cur->get_lid_plan(), params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    float * data = (float *) cross_kq_mask->data;

    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < n_enc; ++j) {
            float f = -INFINITY;

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = ubatch->seq_id[i][s];

                if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                    f = 0.0f;
                }
            }

            data[i*n_enc + j] = f;
        }
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    llm_set_attn_mask_shape(inp_attn->self_kq_mask, inp_attn->self_kq_mask_cnv, mctx->get_attn()->get_n_kv(), ubatch->n_tokens);
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= inp_attn->self_kq_mask->ne[0] == mctx->get_attn()->get_n_kv();
    res &= inp_attn->self_kq_mask->ne[1] == params.ubatch.n_tokens;

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    llm_set_attn_mask_shape(inp_attn->self_kq_mask, inp_attn->self_kq_mask_cnv, mctx->get_attn()->get_n_kv(), ubatch->n_tokens);
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= inp_attn->self_kq_mask->ne[0] == mctx->get_attn()->get_n_kv();
    res &= inp_attn->self_kq_mask->ne[1] == params.ubatch.n_tokens;

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        llm_set_attn_mask_shape(inp_attn->self_kq_mask, inp_attn->self_kq_mask_cnv,
                attn_ctx->get_base()->get_n_kv(), ubatch->n_tokens);
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

        attn_ctx->get_base()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        llm_set_attn_mask_shape(inp_attn->self_kq_mask_swa, inp_attn->self_kq_mask_swa_cnv,
                attn_ctx->get_swa()->get_n_kv(), ubatch->n_tokens);
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);

        attn_ctx->get_swa()->set_input_kq_mask(inp_attn->self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

        res &= inp_attn->self_kq_mask->ne[0] == attn_ctx->get_base()->get_n_kv();
        res &= inp_attn->self_kq_mask->ne[1] == params.ubatch.n_tokens;
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

        res &= inp_attn->self_kq_mask_swa->ne[0] == attn_ctx->get_swa()->get_n_kv();
        res &= inp_attn->self_kq_mask_swa->ne[1] == params.ubatch.n_tokens;
    }

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs() {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    for (auto & [seq_id, t] : t_sampled) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_probs) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_logits) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_candidates) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer),
    n_rot            (hparams.n_rot),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}

ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
    // bug#2 precision partition (env LLAMA_DSV4_DECODE_F32_MOE, default OFF): also force F32 on the DSV4
    // nt==1 REGULAR matmuls (attention wq_a/wq_b/wkv/wo Q8_0 + the SHARED expert ffn_*_shexp Q2_K/Q3_K),
    // completing the all-matmul F32-decode test alongside the mul_mat_id force in build_lora_mm_id. Regular
    // mul_mat honors GGML_PREC_F32 via force_cublas_prec (ggml-cuda.cu:2974) and does NOT crash (needs
    // GGML_CUDA_DISABLE_GRAPHS=1 to avoid the capture conflict). Still-wrong with EVERYTHING F32 => bug#2 is
    // STRUCTURAL, not any quantized-matmul precision.
    static const bool dsv4_decode_f32_reg = []() {
        const char * e = std::getenv("LLAMA_DSV4_DECODE_F32_MOE");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    if (dsv4_decode_f32_reg && arch == LLM_ARCH_DEEPSEEK4 && n_tokens == 1) {
        ggml_mul_mat_set_prec(res, GGML_PREC_F32);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    // bug#2 stream_local confirm-via-fix (env LLAMA_DSV4_DECODE_F32_MOE=1, default OFF): force F32
    // accumulation on the DSV4 nt==1 (decode) quantized expert matmul. At decode (1 token = 1 column) the
    // Q2_K/Q3_K MUL_MAT_ID CUDA path can select a different single-column kernel than the multi-token
    // prefill matmul, drifting the decode hidden state (sol-downstream strongest lead). If recall returns
    // with this ON, the MoE decode kernel is the mechanism. Covers gate/up/down (all via this helper).
    // See .planning/debug/stream-local-recall-debug-downstream.md.
    static const bool dsv4_decode_f32_moe = []() {
        const char * e = std::getenv("LLAMA_DSV4_DECODE_F32_MOE");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    if (dsv4_decode_f32_moe && arch == LLM_ARCH_DEEPSEEK4 && n_tokens == 1) {
        // res is a GGML_OP_MUL_MAT_ID, so ggml_mul_mat_set_prec() (which ASSERTs op==GGML_OP_MUL_MAT and
        // aborts otherwise) cannot be used here. Set the prec op-param directly; it is read by
        // ggml_cuda_mul_mat_id's force_cublas_prec_id gate to skip MMVQ/MMQ and take the F32 cuBLAS path.
        res->op_params[0] = (int32_t) GGML_PREC_F32;
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_token_sidecars(
          ggml_tensor * ffn_input,
                  int   il) const {
    ggml_tensor * sum = nullptr;

    for (const auto & active : *loras) {
        const float adapter_scale = active.second;
        if (adapter_scale == 0.0f) {
            continue;
        }
        const auto * experts = active.first->get_token_sidecars(il);
        if (experts == nullptr) {
            continue;
        }

        for (const auto & expert : *experts) {
            ggml_tensor * normalized = ggml_rms_norm(ctx0, ffn_input, norm_rms_eps);
            normalized = ggml_mul(ctx0, normalized, expert.norm);

            // Full-rank SwiGLU branch. The exporter deliberately stores gate
            // and up separately even though training fuses them, avoiding a
            // runtime view whose row stride depends on the adapter width.
            ggml_tensor * ffn_gate = ggml_silu(ctx0, ggml_mul_mat(ctx0, expert.ffn_gate, normalized));
            ggml_tensor * ffn_up   = ggml_mul_mat(ctx0, expert.ffn_up, normalized);
            ggml_tensor * branch   = ggml_mul_mat(ctx0, expert.ffn_down, ggml_mul(ctx0, ffn_gate, ffn_up));

            // Shape [1, n_tokens], then broadcast over the hidden dimension.
            ggml_tensor * token_gate = ggml_mul_mat(ctx0, expert.gate, normalized);
            token_gate = ggml_add(ctx0, token_gate, expert.gate_b);
            token_gate = ggml_sigmoid(ctx0, token_gate);
            branch = ggml_mul(ctx0, branch, ggml_repeat(ctx0, token_gate, branch));
            branch = ggml_scale(ctx0, branch, adapter_scale);

            sum = sum == nullptr ? branch : ggml_add(ctx0, sum, branch);
        }
    }

    return sum;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    // P3-alt test (2026-05-30, LLAMA_RMS_EPS_OVERRIDE): increase RMS-norm eps to dampen the
    // near-zero-RMS amplification (1/sqrt(eps)) that is the suspected amplifier turning the
    // ~1-ULP multi-ub batch-shape seed into catastrophic divergence. Affects all RMS norms.
    // Gated/default-off: only overrides when the env is set, so production is untouched.
    static const float rms_eps_override = []() {
        const char * e = std::getenv("LLAMA_RMS_EPS_OVERRIDE");
        return (e && e[0]) ? (float) atof(e) : -1.0f;
    }();
    const float rms_eps = rms_eps_override > 0.0f ? rms_eps_override : hparams.f_norm_rms_eps;
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, rms_eps);                break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                // Step35: HF clamps gate (after SiLU) and up before multiplication
                // DeepSeek-V4 also clamps but uses swiglu_split gate ordering.
                if ((arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_DEEPSEEK4) && il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                        cb(tmp, "ffn_up_clamped", il);

                        if (arch == LLM_ARCH_DEEPSEEK4) {
                            cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
                            cb(cur, "ffn_gate_clamped", il);
                            cur = ggml_swiglu_split(ctx0, cur, tmp);
                        } else {
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, tmp);
                        }
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_SITU:
            {
                // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
                //
                // DENSE sibling of the same case in build_moe_ffn. The two functions name things
                // differently: there `up` is the up-projection RESULT and the guard is `has_gate`;
                // here `up` is the up-projection WEIGHT, its result is `tmp`, and the gate weight
                // is `gate`. The case must not be copied between them verbatim.
                GGML_ASSERT(gate);
                const float beta = hparams.situ_beta;
                const float lb   = hparams.situ_linear_beta;

                ggml_tensor * act = ggml_scale(ctx0,
                        ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, cur));

                ggml_tensor * lin = tmp;
                if (lb > 0.0f) {
                    lin = ggml_scale(ctx0,
                            ggml_tanh(ctx0, ggml_scale(ctx0, lin, 1.0f/lb)), lb);
                }
                cur = ggml_mul(ctx0, act, lin);
                cb(cur, "ffn_situ", il);
                // SITU has already consumed the up projection; leaving type_gate as PAR would make
                // the post-switch `cur = ggml_mul(cur, tmp)` multiply it in a second time.
                type_gate = LLM_FFN_SEQ;
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    // ---- Per-group expert dispatch for sub-layer pipelining ----
    // When n_expert_splits > 1, split the MoE into N groups of experts.
    // Each group becomes a separate sub-layer for weight streaming.
    // This is a top-level call (probs_in == nullptr); recursive per-group
    // calls pass probs_in != nullptr and skip this block.
    const int32_t n_splits = cparams.n_expert_splits;
    if (n_splits > 1 && probs_in == nullptr && gate_inp != nullptr) {
        const int64_t experts_per_group = (n_expert + n_splits - 1) / n_splits;

        // Step 1: Full router computation (logits + softmax + argsort)
        // This MUST complete in sub-layer 0 so the post_layer_cb can read
        // the selected expert IDs before sub-layer 1 streams expert weights.
        ggml_tensor * global_logits = build_lora_mm(gate_inp, cur);
        if (exp_probs_b) {
            global_logits = ggml_add(ctx0, global_logits, exp_probs_b);
        }
        cb(global_logits, "ffn_moe_logits_global", il);

        // Compute full routing probabilities and top-K expert selection
        // (same logic as the unsplit path at lines 1345-1411)
        ggml_tensor * route_probs = nullptr;
        switch (gating_op) {
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
                route_probs = ggml_soft_max(ctx0, global_logits);
                break;
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
                route_probs = ggml_sigmoid(ctx0, global_logits);
                break;
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
                route_probs = global_logits;
                break;
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
                // DeepSeek-V4 expert scoring (moe.score_func="sqrtsoftplus"): sqrt(softplus(logits)).
                // Mirrors the unsplit path (llama-graph.cpp ~1842). Without this case the
                // LAYER_SPLITS>1 router (required for WF_MOE_FETCH_ACTIVE) aborted for DSV4.
                route_probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, global_logits));
                break;
            default:
                GGML_ABORT("fatal error");
        }

        ggml_tensor * selection_probs = route_probs;
        // Group-based expert selection (DeepSeek V3 / GLM-5 style)
        if (hparams.n_expert_groups > 1 && n_tokens > 0) {
            const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;
            ggml_tensor * sel_groups = ggml_reshape_3d(ctx0, selection_probs,
                n_exp_per_group, hparams.n_expert_groups, n_tokens);
            ggml_tensor * grp_scores = ggml_argsort_top_k(ctx0, sel_groups, 2);
            grp_scores = ggml_get_rows(ctx0,
                ggml_reshape_4d(ctx0, sel_groups, 1, sel_groups->ne[0], sel_groups->ne[1], sel_groups->ne[2]),
                grp_scores);
            grp_scores = ggml_sum_rows(ctx0,
                ggml_reshape_3d(ctx0, grp_scores, grp_scores->ne[1], grp_scores->ne[2], grp_scores->ne[3]));
            grp_scores = ggml_reshape_2d(ctx0, grp_scores, grp_scores->ne[1], grp_scores->ne[2]);
            ggml_tensor * exp_groups = ggml_argsort_top_k(ctx0, grp_scores, hparams.n_group_used);
            selection_probs = ggml_get_rows(ctx0, sel_groups, exp_groups);
            selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, sel_groups, -INFINITY), selection_probs, exp_groups);
            selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens);
        }

        // Top-K expert selection — this produces the tensor that post_layer_cb reads
        ggml_tensor * selected_experts_global = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
        cb(selected_experts_global, "ffn_moe_topk", il);

        // Step 2: Per-group expert dispatch
        ggml_tensor * moe_out = nullptr;

        for (int g = 0; g < n_splits; g++) {
            const int64_t exp_start = g * experts_per_group;
            const int64_t exp_end   = std::min(exp_start + experts_per_group, n_expert);
            const int64_t n_exp_g   = exp_end - exp_start;
            if (n_exp_g <= 0) continue;

            // Slice router logits for this group
            ggml_tensor * group_logits = ggml_view_2d(ctx0, global_logits,
                n_exp_g, global_logits->ne[1],
                global_logits->nb[1],
                exp_start * ggml_type_size(global_logits->type));
            ggml_tensor * group_logits_cont = ggml_cont(ctx0, group_logits);
            char gname[48];
            snprintf(gname, sizeof(gname), "ffn_moe_gstart_%d", g);
            cb(group_logits_cont, gname, il);

            // Expert tensor views for this group
            ggml_tensor * up_v = up_exps ? ggml_view_3d(ctx0, up_exps,
                up_exps->ne[0], up_exps->ne[1], n_exp_g,
                up_exps->nb[1], up_exps->nb[2],
                exp_start * up_exps->nb[2]) : nullptr;
            ggml_tensor * gate_v = gate_exps ? ggml_view_3d(ctx0, gate_exps,
                gate_exps->ne[0], gate_exps->ne[1], n_exp_g,
                gate_exps->nb[1], gate_exps->nb[2],
                exp_start * gate_exps->nb[2]) : nullptr;
            ggml_tensor * down_v = down_exps ? ggml_view_3d(ctx0, down_exps,
                down_exps->ne[0], down_exps->ne[1], n_exp_g,
                down_exps->nb[1], down_exps->nb[2],
                exp_start * down_exps->nb[2]) : nullptr;
            ggml_tensor * gate_up_v = gate_up_exps ? ggml_view_3d(ctx0, gate_up_exps,
                gate_up_exps->ne[0], gate_up_exps->ne[1], n_exp_g,
                gate_up_exps->nb[1], gate_up_exps->nb[2],
                exp_start * gate_up_exps->nb[2]) : nullptr;

            // Bias views (if present)
            ggml_tensor * up_b_v = up_exps_b ? ggml_view_3d(ctx0, up_exps_b,
                up_exps_b->ne[0], up_exps_b->ne[1], n_exp_g,
                up_exps_b->nb[1], up_exps_b->nb[2],
                exp_start * up_exps_b->nb[2]) : nullptr;
            ggml_tensor * gate_b_v = gate_exps_b ? ggml_view_3d(ctx0, gate_exps_b,
                gate_exps_b->ne[0], gate_exps_b->ne[1], n_exp_g,
                gate_exps_b->nb[1], gate_exps_b->nb[2],
                exp_start * gate_exps_b->nb[2]) : nullptr;
            ggml_tensor * down_b_v = down_exps_b ? ggml_view_3d(ctx0, down_exps_b,
                down_exps_b->ne[0], down_exps_b->ne[1], n_exp_g,
                down_exps_b->nb[1], down_exps_b->nb[2],
                exp_start * down_exps_b->nb[2]) : nullptr;
            ggml_tensor * gate_up_b_v = gate_up_exps_b ? ggml_view_3d(ctx0, gate_up_exps_b,
                gate_up_exps_b->ne[0], gate_up_exps_b->ne[1], n_exp_g,
                gate_up_exps_b->nb[1], gate_up_exps_b->nb[2],
                exp_start * gate_up_exps_b->nb[2]) : nullptr;

            const int64_t n_used_g = std::max<int64_t>(1,
                std::min(n_exp_g, (n_expert_used + n_splits - 1) / n_splits));

            // Temporarily disable grouped routing for split dispatch
            uint32_t saved_expert_groups = hparams.n_expert_groups;
            const_cast<llama_hparams &>(hparams).n_expert_groups = 0;

            // Recursive call with group's expert slice and pre-computed logits
            ggml_tensor * group_out = build_moe_ffn(cur,
                nullptr, nullptr,       // gate_inp = nullptr (using probs_in)
                up_v, up_b_v, gate_v, gate_b_v, down_v, down_b_v,
                nullptr,                // probs_b already applied to global_logits
                n_exp_g, n_used_g,
                type_op, norm_w, w_scale, gating_op, il,
                group_logits_cont,      // probs_in = pre-computed biased logits
                gate_up_v, gate_up_b_v);

            const_cast<llama_hparams &>(hparams).n_expert_groups = saved_expert_groups;

            if (moe_out == nullptr) {
                moe_out = group_out;
            } else {
                moe_out = ggml_add(ctx0, moe_out, group_out);
            }
        }
        cb(moe_out, "ffn_moe_out", il);
        return moe_out;
    }

    // ---- Standard (unsplit) MoE dispatch ----
    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                // DeepSeek-V4 expert scoring (moe.score_func="sqrtsoftplus"): sqrt(softplus(logits)).
                // Raw positive scores; the top-k selection (noaux_tc) + expert_weights_norm below
                // handle group-topk and renormalization. The enum value (4) was defined + passed by
                // deepseek-v4.cpp but this case was missing, so DSV4 aborted here at graph build.
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    // DSV4 hash-routed layers pass an explicit token->expert-id map (selected_experts_in) that
    // OVERRIDES the learned top-k selection. Default (nullptr) keeps the top-k path byte-identical
    // for every existing arch. Ported from upstream origin/dsv4.
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        // apply per-expert scale2 to merged gate_up (use up_exps_s since gate and up are fused)
        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            gate_up = ggml_mul(ctx0, gate_up, s);
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        // apply per-expert scale2 to up
        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            up = ggml_mul(ctx0, up, s);
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }

        // apply per-expert scale2 to gate
        if (gate_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, gate_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            cur = ggml_mul(ctx0, cur, s);
            cb(cur, "ffn_moe_gate_scaled", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                // Step35: per-layer clamp for routed experts
                // DeepSeek-V4 also clamps but uses swiglu_split gate ordering.
                if ((arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_DEEPSEEK4) && il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        up = ggml_clamp(ctx0, up, -limit, limit);
                        cb(up, "ffn_moe_up_clamped", il);

                        if (arch == LLM_ARCH_DEEPSEEK4) {
                            cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
                            cb(cur, "ffn_moe_gate_clamped", il);
                            cur = ggml_swiglu_split(ctx0, cur, up);
                        } else {
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_moe_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_moe_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, up);
                        }
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SITU:
            {
                GGML_ASSERT(has_gate);
                const float beta = hparams.situ_beta;
                const float lb   = hparams.situ_linear_beta;
                ggml_tensor * act = ggml_scale(ctx0,
                        ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, cur));
                if (lb > 0.0f) {
                    up = ggml_scale(ctx0,
                            ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/lb)), lb);
                }
                cur = ggml_mul(ctx0, act, up);
                cb(cur, "ffn_moe_situ", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    // apply per-expert scale2 to down
    if (down_exps_s) {
        ggml_tensor * s = ggml_reshape_3d(ctx0, down_exps_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
        experts = ggml_mul(ctx0, experts, s);
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // Use the smaller of hparams.n_expert_used and the function parameter n_expert_used.
    // In per-group split dispatch, n_expert_used is smaller than hparams.n_expert_used,
    // and the experts tensor was created with the function parameter size.
    const uint32_t n_used_loop = (uint32_t)std::min((int64_t)hparams.n_expert_used, n_expert_used);

    // order the views before the adds
    for (uint32_t i = 0; i < n_used_loop; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < n_used_loop; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    // select one of the 2 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 2> inps;

    // token embeddings path (ubatch.token != nullptr)
    {
        auto & cur = inps[0];

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }
    }

    // vector embeddings path (ubatch.embd != nullptr)
    {
        auto & cur = inps[1];

        cur = inp->embd;
    }

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    // Named so downstream code (split-decode Phase 2d.C1 arena-offset lookup,
    // find_named_tensor_any("inp_pos") debug path) can locate it by name.
    ggml_set_name(cur, "inp_pos");
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interere with
    //       features that require constant topology such as pipline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    // D-FIX (2026-07-17, bug#2 stream_local far-recall): the CUDA 512-wide single-query FA tile
    // (fattn-tile.cuh) does NOT read ggml_flash_attn_ext_get_prec(), so the GGML_PREC_F32 request
    // below is silently ignored and the tile accumulates half2 VKQ (half-staged probabilities) over
    // all selected pooled rows. That half accumulation erases the weak, distributed needle recall
    // signal even when the correct pooled row ranks well (rank 95) => "selected but not recalled".
    // MLX accumulates the pooled value sum in F32. So for DSV4 single-token (nt==1) 512-wide decode,
    // route through the DECOMPOSED path (which forces F32 QK softmax + identical sink semantics) and
    // additionally force F32 on the V*prob (kqv) multiply. Fires ONLY on nt==1 (decode) => production
    // streaming->Mac prefill (nt>1, decode runs on MLX) is untouched. Env LLAMA_DSV4_DECODE_F32_ATTN=0
    // reverts to the FA tile (A/B control). Sourced: .planning/debug/stream-local-recall-debug.md (D).
    // NOTE (2026-07-17): recall A/B FALSIFIED this as the sole bug#2 fix — forcing F32 decode attention
    // changed the (still-wrong) answer 5->826 but did NOT restore far-recall => the root cause is downstream
    // of attention. Kept as an env-gated LEVER (default OFF) for combined testing once the downstream fix lands.
    static const bool dsv4_decode_f32_attn = []() {
        const char * e = std::getenv("LLAMA_DSV4_DECODE_F32_ATTN");
        return e && e[0] != '\0' && e[0] != '0';   // default OFF; =1 enables (F32 decode attention lever)
    }();
    // [DSV4_PREFILL_DECOMP 2026-08-28] The PREFILL counterpart of the lever above.
    //
    // `fa=off` RECALLS the needle and `fa=on` does not, with the mask now verified correct on both
    // compressed paths (inf=40896, zero NaN) and the CSA sink excluded -- so the FA KERNEL itself is
    // the last standing candidate. But global `fa=off` is not usable as a fix: per AGENTS.md
    // known-issue #3 it needs ~34GB for QK^T at 16k tokens and OOMs.
    //
    // This routes ONLY the DSV4 compressed attention through the decomposed path, at prefill. That
    // is cheap precisely because the compressed path is small: `k_all` is raw+compressed, i.e. 512
    // rows at a 267-token prompt, so the materialised QK^T is [512, n_tokens] per head -- nothing
    // like the full-context matrix that makes global fa=off blow up. The `k->ne[0] == 512` guard
    // (shared with the decode lever) is what keeps this off the RAW layers, which are the expensive
    // ones.
    //
    // So this is a TEST and a candidate FIX at once: if recall returns, the FA kernel on the
    // compressed path is the defect, and confining that path to the decomposed implementation keeps
    // FA -- and its speed -- everywhere it actually matters.
    static const bool dsv4_prefill_decomp = []() {
        const char * e = std::getenv("LLAMA_DSV4_PREFILL_DECOMP");
        return e && e[0] != '\0' && e[0] != '0';   // default OFF
    }();
    // Guard on the layer being COMPRESSED, not on the head dim. The first version used
    // `k->ne[0] == 512`, which is DSV4's MLA head dim generally and therefore also matched the RAW
    // layers -- the marker fired at il=0 and il=1. That made the arm "decomposed attention for all
    // DSV4 layers", which broke the model into the degenerate attractor instead of isolating the
    // kernel on the compressed path, and it removed the cost argument too (on raw layers the
    // materialised QK^T scales with the full raw cache, which is the ~34GB OOM that makes global
    // fa=off unusable).
    //
    // `hparams.dsv4_compress_ratios[il] != 0` is the same predicate deepseek-v4.cpp itself uses to
    // decide a layer is compressed, so this now selects exactly the CSA/HCA layers.
    const bool dsv4_layer_is_compressed =
        arch == LLM_ARCH_DEEPSEEK4 && il >= 0 && il < (int) LLAMA_MAX_LAYERS &&
        hparams.dsv4_compress_ratios[il] != 0;
    // [DECOMP_RAW 2026-08-28] The inverted guard, for the RAW layers (0, 1, 43).
    //
    // Decomposing the COMPRESSED layers gave a clean negative: guard verified at il=2,3,4 only,
    // output stayed COHERENT (finish=stop, not the degenerate attractor), and the reply CHANGED
    // ('west vault' -> 'WEST-VAULT-8492', the model now confabulating a code in the right FORMAT) --
    // so the arm was live and had real numerical effect, and recall still failed. The FA kernel on
    // the compressed path is therefore excluded.
    //
    // Every other variable fa=off changes is already excluded (mask fixed, sinks, PREC_F32; the f16
    // cast is not independent). By elimination the defect is the FA kernel on the RAW layers.
    //
    // **This arm CANNOT SHIP.** On raw layers the materialised QK^T scales with the full raw cache,
    // which is the ~34GB OOM at 16k that makes global fa=off unusable (AGENTS.md known-issue #3). It
    // is a LOCALISER only -- at 267 tokens the raw cache is 512 rows so it is cheap here, and if it
    // recalls, the fix has to be a correct FA kernel on that path, not this.
    static const bool dsv4_prefill_decomp_raw = []() {
        const char * e = std::getenv("LLAMA_DSV4_PREFILL_DECOMP_RAW");
        return e != nullptr && e[0] != 0 && e[0] != '0';
    }();
    const bool force_dsv4_decomp_raw =
        dsv4_prefill_decomp_raw && arch == LLM_ARCH_DEEPSEEK4 && n_tokens > 1 &&
        !dsv4_layer_is_compressed && kq_b == nullptr && v_mla == nullptr;
    if (force_dsv4_decomp_raw) {
        static int dpr_dbg = 0;
        if (dpr_dbg < 3) {
            dpr_dbg++;
            fprintf(stderr, "*** DSV4_DECOMP_RAW: decomposed RAW attn il=%d nt=%d\n",
                    il, (int) n_tokens);
            fflush(stderr);
        }
    }

    const bool force_dsv4_decomp_prefill =
        (dsv4_prefill_decomp && arch == LLM_ARCH_DEEPSEEK4 && n_tokens > 1 &&
         dsv4_layer_is_compressed && kq_b == nullptr && v_mla == nullptr)
        || force_dsv4_decomp_raw;
    if (force_dsv4_decomp_prefill) {
        static int dpd_dbg = 0;
        if (dpd_dbg < 3) {
            dpd_dbg++;
            fprintf(stderr, "*** DSV4_PREFILL_DECOMP: decomposed CSA attn il=%d nt=%d k.ne0=%lld\n",
                    il, (int) n_tokens, (long long) k->ne[0]);
            fflush(stderr);
        }
    }

    const bool force_dsv4_decomp_f32 =
        (dsv4_decode_f32_attn && arch == LLM_ARCH_DEEPSEEK4 && n_tokens == 1 &&
         k->ne[0] == 512 && kq_b == nullptr && v_mla == nullptr)
        || force_dsv4_decomp_prefill;

    ggml_tensor * cur;

    if (cparams.flash_attn && kq_b == nullptr && !force_dsv4_decomp_f32) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        // NOMASK (2026-05-30, >160k unlock — graph-side DEFERRED): the plan was to pass a null mask
        // here so the FA uses its inline causal mask + the O(ub*n_kv) kq_mask is freed. The kernel +
        // dispatch groundwork is in place (fattn-mma-f16.cuh loads gated on mask_h, direct_causal flag
        // relaxed; fattn.cu GQA-path + selector allow null mask under LLAMA_FA_MLA_CAUSAL_DIRECT_NOMASK).
        // BUT nulling src[3] here leaves self_kq_mask DANGLING (not allocated) and set_input_kq_mask then
        // hits a NULL-buffer GGML_ASSERT. The clean fix is to SKIP BUILDING the KV-cache self_kq_mask +
        // its f16 CAST + set_input + shape + the build_*_attn shape-validation under the env (multi-site,
        // shared decode/prefill input path). Reverted to the working mask path until that is done.
        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        cb(cur, LLAMA_TENSOR_NAME_FATTN, il);

        ggml_flash_attn_ext_add_sinks(cur, sinks);

        // [NO_PREC_F32 2026-08-28] LLAMA_FA_NO_PREC_F32 (default off) skips the GGML_PREC_F32
        // request, leaving the kernel at its default precision.
        //
        // This is the LAST independent variable separating fa=on from fa=off for the DSV4 needle.
        // fa=off recalls (n=2, byte-identical replies); fa=on does not (n=6). Of the five things
        // fa=off changes: the SWA _cnv identity is fixed and excluded, the attention sink is
        // excluded, and the f16 mask turned out NOT to be independent -- the CUDA kernel asserts
        // `mask->type == GGML_TYPE_F16` (fattn-common.cuh:826), so the cast is a mandatory
        // consequence of using FA rather than a separate choice. That leaves this.
        //
        // Read the comment at the top of this function before interpreting a null result: the
        // 512-wide single-query tile does NOT read ggml_flash_attn_ext_get_prec(), so this request
        // is already ignored on that path. If it is also ignored on the prefill path, flipping it
        // will change nothing and that is a fact about the plumbing, not about precision -- which is
        // itself worth establishing rather than assuming.
        static const bool no_prec_f32 = []() {
            const char * e = std::getenv("LLAMA_FA_NO_PREC_F32");
            return e != nullptr && e[0] != 0 && e[0] != '0';
        }();
        if (!no_prec_f32) {
            ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);
        } else {
            static int npf_dbg = 0;
            if (npf_dbg < 3) {
                npf_dbg++;
                fprintf(stderr, "*** NO_PREC_F32: skipped GGML_PREC_F32 request at il=%d\n", il);
                fflush(stderr);
            }
        }

        if (v_mla) {
            const char * safe_mla = std::getenv("LLAMA_FA_MLA_SAFE_DECOMP");
            if (n_stream > 1) {
                // Sparse prefill represents each query token as its own FA stream.
                // Flatten streams back into the token axis before MLA decompression so
                // wv_b sees the same [kv_lora_rank, n_tokens, n_head, 1] layout as dense prefill.
                cur = ggml_permute(ctx0, cur, 0, 2, 3, 1);
                cb(cur, "fattn_stream_flat", il);
                cur = ggml_mul_mat(ctx0, v_mla, cur);
                cb(cur, "fattn_mla", il);
                cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
                cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
            } else if (safe_mla && safe_mla[0] != '\0' && safe_mla[0] != '0') {
                // Conservative path: apply MLA decompression without relying on
                // the FA output's permute-only reinterpretation.
                cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
                cur = ggml_mul_mat(ctx0, v_mla, cur);
                cb(cur, "fattn_mla_safe", il);
            } else {
                // Fast path: treat the FA output as an MxN matrix by reinterpreting
                // dimensions, then decompress via matrix-matrix multiplication.
                cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
                const char * cont_perm = std::getenv("LLAMA_FA_MLA_CONT_PERMUTE");
                if (cont_perm && cont_perm[0] != '\0' && cont_perm[0] != '0') {
                    cur = ggml_cont(ctx0, cur);
                    cb(cur, "fattn_perm_cont", il);
                } else {
                    cb(cur, "fattn_perm", il);
                }
                cur = ggml_mul_mat(ctx0, v_mla, cur);
                cb(cur, "fattn_mla", il);
                cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
                cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
            }
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        // bug#2 (2026-07-18): pin the ACTUAL post-softmax attention weights at DSV4 DECODE (nt==1) under
        // LLAMA_DSV4_LID_PROBE so the clean dump can raw-byte them - the earlier attempt pinned csa_lid_kq_mask
        // (the pre-top-k position mask, all-finite). This kq IS the applied weight distribution (top-k -inf
        // already baked in). Requires the DECOMPOSED path (LLAMA_DSV4_DECODE_F32_ATTN=1 + GGML_CUDA_DISABLE_GRAPHS=1).
        {
            static const bool lid_probe_kq = []() {
                const char * e = std::getenv("LLAMA_DSV4_LID_PROBE");
                return e && e[0] != '\0';
            }();
            if (lid_probe_kq && arch == LLM_ARCH_DEEPSEEK4 && n_tokens == 1) {
                ggml_set_output(kq);
            }
        }

        // [DIAG_SPARSE_NAN 2026-06-25] env-gated probe: name post-softmax kq so the cb_eval
        // LLAMA_LAYER_CMP_DUMP hook captures it (localizes whether the softmax NaNs).
        static const bool diag_sparse_nan = []() {
            const char * e = std::getenv("LLAMA_DIAG_SPARSE_NAN");
            return e && e[0] != '\0' && e[0] != '0';
        }();
        if (diag_sparse_nan && n_stream > 1) {
            char dn[64]; snprintf(dn, sizeof(dn), "diagsoftmax-%d", il);
            ggml_set_name(kq, dn);  // cb_eval observes by name; no set_output (would bloat alloc)
        }

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        // D-FIX (2026-07-17): force F32 accumulation for the DSV4 nt==1 512-wide decode V*prob multiply.
        // Disabling FA alone is insufficient — the decomposed V*prob multiply otherwise keeps default
        // (F16-capable) precision, which is the exact half accumulation we are fixing. See top of fn.
        if (force_dsv4_decomp_f32) {
            ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);
        }
        cb(kqv, "kqv", il);
        if (diag_sparse_nan && n_stream > 1) {
            char dn[64]; snprintf(dn, sizeof(dn), "diagkqv-%d", il);
            ggml_set_name(kqv, dn);
        }

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            // [DIAG_KQV_MLA_SHAPES 2026-06-24] env-gated shape dump to root-cause the multi-ubatch
            // CUDA invalid-argument at this mul_mat (dst kqv_mla, src0 attn_v_b/wv_b). Zero cost off.
            static const bool diag_kqv_mla = []() {
                const char * e = std::getenv("LLAMA_DIAG_KQV_MLA");
                return e && e[0] != '\0' && e[0] != '0';
            }();
            if (diag_kqv_mla && il < 2) {
                FILE * df = fopen("diagnostics/kqv_mla_diag.txt", "a");
                if (df) {
                    fprintf(df,
                        "[DIAG_KQV_MLA] il=%d n_stream=%lld v_trans=%d | v_mla ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] | "
                        "kqv ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] cont=%d | "
                        "v ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] | kq ne=[%lld,%lld,%lld,%lld]\n",
                        il, (long long) n_stream, v_trans ? 1 : 0,
                        (long long) v_mla->ne[0], (long long) v_mla->ne[1], (long long) v_mla->ne[2], (long long) v_mla->ne[3],
                        v_mla->nb[0], v_mla->nb[1], v_mla->nb[2], v_mla->nb[3],
                        (long long) kqv->ne[0], (long long) kqv->ne[1], (long long) kqv->ne[2], (long long) kqv->ne[3],
                        kqv->nb[0], kqv->nb[1], kqv->nb[2], kqv->nb[3], ggml_is_contiguous(kqv) ? 1 : 0,
                        (long long) v->ne[0], (long long) v->ne[1], (long long) v->ne[2], (long long) v->ne[3],
                        v->nb[0], v->nb[1], v->nb[2], v->nb[3],
                        (long long) kq->ne[0], (long long) kq->ne[1], (long long) kq->ne[2], (long long) kq->ne[3]);
                    fflush(df);
                    fclose(df);
                }
            }
            // [KQV_MLA_MULTIUBATCH_FIX 2026-06-25] In multi-ubatch (n_stream>1) src1=kqv carries n_stream in
            // ne[3] while src0=v_mla (the quantized wv_b weight, ne[3]=1) must broadcast over ne[3]; the CUDA
            // mul_mat can't broadcast a QUANTIZED src0 over ne[3] -> "CUDA error: invalid argument". Dequant-cast
            // v_mla so it broadcasts fine. Use F32 (NOT F16): the ggml-cpu DUP op only supports quantized->F32
            // (ops.cpp:570 GGML_ABORT on quantized->F16), and the long-ctx KV-inject decode_layer_major +
            // conservative-multiubatch path can schedule this cast on the CPU. F32 works on both CPU+GPU and is
            // higher precision than the quant. Single-ubatch (n_stream==1, no ne[3] broadcast) keeps the original
            // quantized path byte-identical. See memory project_dsa_multiubatch_kqv_mla_rootcause.
            ggml_tensor * v_mla_eff = (n_stream > 1) ? ggml_cast(ctx0, v_mla, GGML_TYPE_F32) : v_mla;
            kqv = ggml_mul_mat(ctx0, v_mla_eff, kqv);
            cb(kqv, "kqv_mla", il);
            if (diag_sparse_nan && n_stream > 1) {
                char dn[64]; snprintf(dn, sizeof(dn), "diagkqvmla-%d", il);
                ggml_set_name(kqv, dn);
            }
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);
    ggml_set_name(inp->self_kq_mask, "self_kq_mask");  // required by split-decode CUDA graph kq_mask capture

    inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    ggml_set_name(inp->self_kq_mask_cnv, "self_kq_mask_cnv");
    // [MASK_SITE_DIAG 2026-08-27] env-gated (LLAMA_MASK_SITE_DIAG). Z1a/Z4: a 512-row staged
    // mask copy meets a 256-row self_kq_mask_cnv in cpy.cu:409. Static reading could not say
    // WHICH build makes the 256 one, so every site that can create this tensor names itself.
    if (std::getenv("LLAMA_MASK_SITE_DIAG")) {
        fprintf(stderr, "*** MASK_SITE %s mask=[%lld,%lld,%lld,%lld] cnv=[%lld,%lld,%lld,%lld] n_ubatch=%d n_tokens=%d\n",
            __func__,
            (long long) inp->self_kq_mask->ne[0], (long long) inp->self_kq_mask->ne[1],
            (long long) inp->self_kq_mask->ne[2], (long long) inp->self_kq_mask->ne[3],
            (long long) inp->self_kq_mask_cnv->ne[0], (long long) inp->self_kq_mask_cnv->ne[1],
            (long long) inp->self_kq_mask_cnv->ne[2], (long long) inp->self_kq_mask_cnv->ne[3],
            (int) cparams.n_ubatch, (int) ubatch.n_tokens);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);
        ggml_set_name(inp->self_kq_mask_swa, "self_kq_mask_swa");

        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
        ggml_set_name(inp->self_kq_mask_swa_cnv, "self_kq_mask_swa_cnv");
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        const auto n_kv     = mctx_cur->get_n_kv();
        const auto n_tokens = ubatch.n_tokens;
        const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        // FA + MLA: pad mask ne[0] to 256 for GQA kernel's mask-to-KV-max scan.
        auto n_kv_mask = n_kv;
        const char * no_mask_pad = std::getenv("LLAMA_FA_MLA_NO_MASK_PAD");
        const bool disable_mask_pad = no_mask_pad && no_mask_pad[0] != '\0' && no_mask_pad[0] != '0';
        if (!disable_mask_pad && cparams.flash_attn && hparams.is_mla() && (n_kv % 256) != 0) {
            n_kv_mask = (n_kv + 255) & ~(int64_t)255;
        }
        auto mask_rows = n_tokens / n_stream;
        // Opt-in (LLAMA_PREFILL_STREAMING_MASK_MAXKV): size the mask leaf at a CONSTANT
        // max [n_ctx, n_ubatch] so it is invariant across the streaming-prefill ubatches.
        // Per-ubatch get_n_kv() grows (4096..n_batch) and the sched reserves the cross-
        // backend copy buffer once -> "layout mismatch" assert on the wir_local/split CPU
        // mask copy. Constant sizing makes leaf==copy-buffer always. Safe: set_input_kq_mask
        // fills using dst->ne[0] and pads non-causal/padding keys with -inf (FA ignores them);
        // arm_fa_patch reshapes the FA-side view to the actual visible n_kv/n_tokens. Default
        // OFF -> unchanged behavior (Kimi production unaffected).
        static const bool stream_max_mask = [](){ const char * e = std::getenv("LLAMA_PREFILL_STREAMING_MASK_MAXKV"); return e && e[0] != '\0' && e[0] != '0'; }();
        if (stream_max_mask) {
            int64_t maxkv = (int64_t) cparams.n_ctx;
            if (cparams.flash_attn && (maxkv % 256) != 0) {
                maxkv = (maxkv + 255) & ~(int64_t)255;
            }
            if (maxkv > n_kv_mask) {
                n_kv_mask = maxkv;
            }
            const int64_t maxrows = (int64_t) cparams.n_ubatch / n_stream;
            if (maxrows > mask_rows) {
                mask_rows = maxrows;
            }
        }
        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_mask, mask_rows, 1, n_stream);
        ggml_set_input(inp->self_kq_mask);
        ggml_set_name(inp->self_kq_mask, "self_kq_mask");  // required by split-decode CUDA graph kq_mask capture

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
        ggml_set_name(inp->self_kq_mask_cnv, "self_kq_mask_cnv");
        // [MASK_SITE_DIAG 2026-08-27] env-gated (LLAMA_MASK_SITE_DIAG). Z1a/Z4: a 512-row staged
        // mask copy meets a 256-row self_kq_mask_cnv in cpy.cu:409. Static reading could not say
        // WHICH build makes the 256 one, so every site that can create this tensor names itself.
        if (std::getenv("LLAMA_MASK_SITE_DIAG")) {
            fprintf(stderr, "*** MASK_SITE %s mask=[%lld,%lld,%lld,%lld] cnv=[%lld,%lld,%lld,%lld] n_ubatch=%d n_tokens=%d\n",
                __func__,
                (long long) inp->self_kq_mask->ne[0], (long long) inp->self_kq_mask->ne[1],
                (long long) inp->self_kq_mask->ne[2], (long long) inp->self_kq_mask->ne[3],
                (long long) inp->self_kq_mask_cnv->ne[0], (long long) inp->self_kq_mask_cnv->ne[1],
                (long long) inp->self_kq_mask_cnv->ne[2], (long long) inp->self_kq_mask_cnv->ne[3],
                (int) cparams.n_ubatch, (int) ubatch.n_tokens);
        }
        // 2026-05-31 PIN_MASK (env LLAMA_PREFILL_PIN_MASK, default off): at chunk5 the ~419MB f16
        // self_kq_mask_cnv is the largest n_kv-scaled compute tensor; in the layer-major repeated-
        // slice prefill its gallocr placement reshuffles the frozen layout so a per-layer (MoE)
        // neighbor overruns the attn_norm/kv_cmpr_pe slot -> layer1+ Kcur=0/nan (ggml-alloc.c:26-35
        // reuse-collision). Pinning the mask_cnv as a graph output takes the reshuffler OUT of the
        // reuse pool (low VRAM, unlike GGML_ALLOC_NO_FREE_REUSE which OOMs at 69GB). Marker PIN_MASK.
        if (cparams.flash_attn && inp->self_kq_mask_cnv != inp->self_kq_mask) {
            static const bool pin_mask = std::getenv("LLAMA_PREFILL_PIN_MASK") != nullptr;  // PIN_MASK
            if (pin_mask) ggml_set_output(inp->self_kq_mask_cnv);
        }
    }

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    // For hybrid memory, extract the attention sub-context (kv_cache_context).
    // The mctx may be llama_memory_hybrid_context, not kv_cache_context.
    const llama_kv_cache_context * mctx_cur = dynamic_cast<const llama_kv_cache_context *>(mctx);
    if (!mctx_cur) {
        const auto * hybrid = dynamic_cast<const llama_memory_hybrid_context *>(mctx);
        if (hybrid) {
            mctx_cur = hybrid->get_attn();
            fprintf(stderr, "*** BUILD_ATTN: hybrid path, mctx_cur=%p\n", (const void*)mctx_cur);
        }
    } else {
        fprintf(stderr, "*** BUILD_ATTN: direct kv_cache path, mctx_cur=%p\n", (const void*)mctx_cur);
    }
    GGML_ASSERT(mctx_cur && "memory context must be kv_cache_context or hybrid_context with attn");

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn_m3_sparse(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * block_mask,
            float     kq_scale,
            int       il) const {
    // Mirror build_attn(llm_graph_input_attn_kv*): store K+V to the cache, view them, run
    // build_attn_mha. The ONLY difference: optionally fold an additive block-sparse mask into the
    // causal kq_mask (block_mask == nullptr => numerically identical to dense; that's the
    // coherence-gate baseline). M3 GQA: V is a separate cache (not a view of K like MLA).
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();
    if (block_mask != nullptr) {
        if (std::getenv("LLAMA_M3_SPARSE_DEBUG")) {
            fprintf(stderr, "*** M3_SPARSE il=%d kq_mask ne=[%lld,%lld,%lld,%lld] block_mask ne=[%lld,%lld,%lld,%lld]\n",
                il, (long long) kq_mask->ne[0], (long long) kq_mask->ne[1], (long long) kq_mask->ne[2], (long long) kq_mask->ne[3],
                (long long) block_mask->ne[0], (long long) block_mask->ne[1], (long long) block_mask->ne[2], (long long) block_mask->ne[3]);
        }
        if (block_mask->ne[2] > 1) {
            // PER-HEAD mask [n_kv, n_q, n_head]: fold with kq_mask ([n_kv, n_q, 1, n_stream]) broadcasting
            // over the head axis. soft_max_ext then applies it per-head (mask ne[2] == kq ne[2]). Requires
            // n_stream==1 (kv_unified); the streamed per-head layout is a later refinement.
            kq_mask = ggml_add(ctx0, ggml_cont(ctx0, block_mask), kq_mask);
        } else {
            // SHARED mask [n_kv, n_q]: the causal kq_mask may be STREAMED ([n_kv, mask_rows, 1, n_stream]
            // with n_q == mask_rows*n_stream — the fork splits a prefill into per-token streams). Reshape
            // block_mask to kq_mask's exact ne so the additive fold lines up (same n_kv + same nelements).
            if (block_mask->ne[0] == kq_mask->ne[0] && ggml_nelements(block_mask) == ggml_nelements(kq_mask)) {
                block_mask = ggml_reshape_4d(ctx0, ggml_cont(ctx0, block_mask),
                                             kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
            }
            kq_mask = ggml_add(ctx0, kq_mask, block_mask); // additive: 0 selected, -1e30 pruned
        }
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask, nullptr, nullptr, kq_scale, il);
    cb(cur, "kqv_out_m3_sparse", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }
    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        const auto n_kv     = mctx_cur->get_n_kv();
        const auto n_tokens = ubatch.n_tokens;
        const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        // FA + MLA: pad mask to 256 (same as build_attn_inp_kv_impl)
        auto n_kv_mask = n_kv;
        const char * no_mask_pad = std::getenv("LLAMA_FA_MLA_NO_MASK_PAD");
        const bool disable_mask_pad = no_mask_pad && no_mask_pad[0] != '\0' && no_mask_pad[0] != '0';
        if (!disable_mask_pad && cparams.flash_attn && hparams.is_mla() && (n_kv % 256) != 0) {
            n_kv_mask = (n_kv + 255) & ~(int64_t)255;
        }
        auto mask_rows = n_tokens / n_stream;
        // Opt-in (LLAMA_PREFILL_STREAMING_MASK_MAXKV): constant-max mask leaf [n_ctx, n_ubatch]
        // so it is invariant across the streaming-prefill ubatches. This is the MLA path
        // (build_attn_inp_k, used by deepseek2 is_mla); the cross-backend copy buffer (keyed
        // by the "self_kq_mask_cnv" name) is shared with the kv-path mask, so BOTH must size
        // consistently to avoid the sched copy-buffer layout-mismatch assert. Fill is safe
        // (set_input_kq_mask uses dst->ne[0] + pads -inf); FA reads the compact region via
        // arm_fa_patch. Default OFF -> unchanged (Kimi production unaffected).
        static const bool stream_max_mask_k = [](){ const char * e = std::getenv("LLAMA_PREFILL_STREAMING_MASK_MAXKV"); return e && e[0] != '\0' && e[0] != '0'; }();
        if (stream_max_mask_k) {
            int64_t maxkv = (int64_t) cparams.n_ctx;
            if (cparams.flash_attn && (maxkv % 256) != 0) {
                maxkv = (maxkv + 255) & ~(int64_t)255;
            }
            if (maxkv > n_kv_mask) {
                n_kv_mask = maxkv;
            }
            const int64_t maxrows = (int64_t) cparams.n_ubatch / n_stream;
            if (maxrows > mask_rows) {
                mask_rows = maxrows;
            }
        }
        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_mask, mask_rows, 1, n_stream);
        ggml_set_input(inp->self_kq_mask);
        ggml_set_name(inp->self_kq_mask, "self_kq_mask");  // required by split-decode CUDA graph kq_mask capture

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
        ggml_set_name(inp->self_kq_mask_cnv, "self_kq_mask_cnv");
        // [MASK_SITE_DIAG 2026-08-27] env-gated (LLAMA_MASK_SITE_DIAG). Z1a/Z4: a 512-row staged
        // mask copy meets a 256-row self_kq_mask_cnv in cpy.cu:409. Static reading could not say
        // WHICH build makes the 256 one, so every site that can create this tensor names itself.
        if (std::getenv("LLAMA_MASK_SITE_DIAG")) {
            fprintf(stderr, "*** MASK_SITE %s mask=[%lld,%lld,%lld,%lld] cnv=[%lld,%lld,%lld,%lld] n_ubatch=%d n_tokens=%d\n",
                __func__,
                (long long) inp->self_kq_mask->ne[0], (long long) inp->self_kq_mask->ne[1],
                (long long) inp->self_kq_mask->ne[2], (long long) inp->self_kq_mask->ne[3],
                (long long) inp->self_kq_mask_cnv->ne[0], (long long) inp->self_kq_mask_cnv->ne[1],
                (long long) inp->self_kq_mask_cnv->ne[2], (long long) inp->self_kq_mask_cnv->ne[3],
                (int) cparams.n_ubatch, (int) ubatch.n_tokens);
        }
        // 2026-05-31 PIN_MASK (env LLAMA_PREFILL_PIN_MASK, default off): at chunk5 the ~419MB f16
        // self_kq_mask_cnv is the largest n_kv-scaled compute tensor; in the layer-major repeated-
        // slice prefill its gallocr placement reshuffles the frozen layout so a per-layer (MoE)
        // neighbor overruns the attn_norm/kv_cmpr_pe slot -> layer1+ Kcur=0/nan (ggml-alloc.c:26-35
        // reuse-collision). Pinning the mask_cnv as a graph output takes the reshuffler OUT of the
        // reuse pool (low VRAM, unlike GGML_ALLOC_NO_FREE_REUSE which OOMs at 69GB). Marker PIN_MASK.
        if (cparams.flash_attn && inp->self_kq_mask_cnv != inp->self_kq_mask) {
            static const bool pin_mask = std::getenv("LLAMA_PREFILL_PIN_MASK") != nullptr;  // PIN_MASK
            if (pin_mask) ggml_set_output(inp->self_kq_mask_cnv);
        }
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

// Phase 3.5-lite (iter 29): sparse-MLA indexer K cache helpers.
ggml_tensor * llm_graph_context::build_inp_indexer_k(int il) const {
    const auto * mctx_cur = dynamic_cast<const llama_kv_cache_context *>(mctx);
    if (mctx_cur == nullptr) {
        return nullptr;
    }
    return mctx_cur->get_k_idx(ctx0, il);
}

ggml_tensor * llm_graph_context::build_inp_indexer_k_max(int il) const {
    const auto * mctx_cur = dynamic_cast<const llama_kv_cache_context *>(mctx);
    if (mctx_cur == nullptr) {
        return nullptr;
    }
    return mctx_cur->get_k_idx_max(ctx0, il);
}

ggml_tensor * llm_graph_context::build_sparse_store_indexer_k(ggml_tensor * k_idx_cur, ggml_tensor * k_idxs, int il) const {
    if (k_idx_cur == nullptr || k_idxs == nullptr) {
        return nullptr;
    }
    const auto * mctx_cur = dynamic_cast<const llama_kv_cache_context *>(mctx);
    if (mctx_cur == nullptr) {
        return nullptr;
    }
    return mctx_cur->cpy_k_idx(ctx0, k_idx_cur, k_idxs, il);
}

ggml_tensor * llm_graph_context::build_sparse_store_main_k(ggml_tensor * k_cur, ggml_tensor * k_idxs, int il) const {
    if (k_cur == nullptr || k_idxs == nullptr) {
        return nullptr;
    }
    const auto * mctx_cur = dynamic_cast<const llama_kv_cache_context *>(mctx);
    if (mctx_cur == nullptr) {
        return nullptr;
    }
    return mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il);
}

ggml_tensor * llm_graph_context::build_sparse_main_k_view(int il) const {
    const auto * mctx_cur = dynamic_cast<const llama_kv_cache_context *>(mctx);
    if (mctx_cur == nullptr) {
        return nullptr;
    }
    return mctx_cur->get_k(ctx0, il);
}

ggml_tensor * llm_graph_context::build_sparse_store_main_v(ggml_tensor * v_cur, ggml_tensor * v_idxs, int il) const {
    if (v_cur == nullptr || v_idxs == nullptr) {
        return nullptr;
    }
    const auto * mctx_cur = dynamic_cast<const llama_kv_cache_context *>(mctx);
    if (mctx_cur == nullptr) {
        return nullptr;
    }
    return mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il);
}

ggml_tensor * llm_graph_context::build_sparse_main_v_view(int il) const {
    const auto * mctx_cur = dynamic_cast<const llama_kv_cache_context *>(mctx);
    if (mctx_cur == nullptr) {
        return nullptr;
    }
    return mctx_cur->get_v(ctx0, il);
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const bool is_swa = hparams.is_swa(il);

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [LEAFCONT_SHARED 2026-08-28] The general form of the DSV4 mask fix, for EVERY ISWA arch.
    //
    // `get_kq_mask_swa()` returns `self_kq_mask_swa_cnv` = `ggml_cast(leaf, F16)` under FA, built
    // ONCE AT THE GRAPH HEAD. The layer-major streaming executor computes per-layer node ranges, and
    // **layer 0's range begins at the head** -- so the cast executes there and layer 0 reads
    // correctly, while every LATER layer reads it stale, as ZEROS. Measured on DSV4, same node one
    // layer apart, same build:
    //     raw_kq_mask-0   inf=958400 / 1015040     correct
    //     raw_kq_mask-1   inf=0                    no causal masking, no window
    // With fa=off the ternary returns the leaf itself, there is no node to go stale, and the same
    // prompts recall correctly -- which is why fa=off was the only working configuration.
    //
    // Fixing this in deepseek-v4.cpp (6e977c0c, c26d313d, 80d1ab03) restored DSV4's lost needle with
    // FA ON. But that fix is arch-local, and EVERY other ISWA arch reaches the mask through THIS
    // function: gemma2/3/3n/4, cohere2, exaone4, exaone-moe, phi3, olmo2, plamo3, afmoe, mimo2,
    // llama-iswa, openai-moe, smallthinker, step35. They have the same defect. (Non-ISWA archs such
    // as GLM-5.2 are unaffected -- they never touch the SWA mask.)
    //
    // Take a ggml_cont of the LEAF inside the consuming layer and cast back to the consumed type so
    // FA still receives f16 (fattn-common.cuh:826 asserts it). ggml_cont rather than the leaf itself:
    // handing the raw leaf to a GPU op forces it device-side and trips
    // GGML_ASSERT(ggml_backend_buffer_is_host) in set_input_kq_mask -- that one distinction is why
    // four earlier attempts at this route asserted.
    //
    // Opt-in for now (LLAMA_ISWA_MASK_LEAFCONT), because it changes output on every ISWA arch and
    // must clear the lossless gate per model. Prefill only: at nt==1 the mask is a single row and the
    // stale-buffer window does not arise the same way, and gating both knobs to the same graphs is
    // what avoids the mismatched-guard assert seen with the DSV4 levers.
    ggml_tensor * kq_mask_eff = kq_mask;
    {
        static const bool iswa_leafcont = []() {
            const char * e = std::getenv("LLAMA_ISWA_MASK_LEAFCONT");
            return e != nullptr && e[0] != 0 && e[0] != '0';
        }();
        ggml_tensor * leaf_src = is_swa ? inp->self_kq_mask_swa : inp->self_kq_mask;
        if (iswa_leafcont && cparams.flash_attn && leaf_src && kq_mask_eff &&
                kq_mask_eff->ne[1] > 1 && leaf_src != kq_mask_eff) {
            ggml_tensor * lm = leaf_src;
            if (lm->ne[0] != kq_mask_eff->ne[0] || lm->ne[1] != kq_mask_eff->ne[1]) {
                lm = ggml_view_2d(ctx0, lm, kq_mask_eff->ne[0], kq_mask_eff->ne[1], lm->nb[1], 0);
            }
            ggml_tensor * fixed = ggml_cont(ctx0, lm);
            if (fixed->type != kq_mask_eff->type) {
                fixed = ggml_cast(ctx0, fixed, kq_mask_eff->type);
            }
            kq_mask_eff = fixed;
            static int ilc_dbg = 0;
            if (ilc_dbg < 3) {
                ilc_dbg++;
                fprintf(stderr, "*** LEAFCONT_SHARED: iswa mask from leaf as %s at il=%d nt=%lld\n",
                        ggml_type_name(kq_mask_eff->type), il, (long long) kq_mask_eff->ne[1]);
                fflush(stderr);
            }
        }
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_eff, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->cross_kq_mask, GGML_TYPE_F16) : inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    {
        const auto n_kv = mctx_cur->get_base()->get_n_kv();

        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp->self_kq_mask);
        ggml_set_name(inp->self_kq_mask, "self_kq_mask");

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
        ggml_set_name(inp->self_kq_mask_cnv, "self_kq_mask_cnv");
        // [MASK_SITE_DIAG 2026-08-27] env-gated (LLAMA_MASK_SITE_DIAG). Z1a/Z4: a 512-row staged
        // mask copy meets a 256-row self_kq_mask_cnv in cpy.cu:409. Static reading could not say
        // WHICH build makes the 256 one, so every site that can create this tensor names itself.
        if (std::getenv("LLAMA_MASK_SITE_DIAG")) {
            fprintf(stderr, "*** MASK_SITE %s mask=[%lld,%lld,%lld,%lld] cnv=[%lld,%lld,%lld,%lld] n_ubatch=%d n_tokens=%d\n",
                __func__,
                (long long) inp->self_kq_mask->ne[0], (long long) inp->self_kq_mask->ne[1],
                (long long) inp->self_kq_mask->ne[2], (long long) inp->self_kq_mask->ne[3],
                (long long) inp->self_kq_mask_cnv->ne[0], (long long) inp->self_kq_mask_cnv->ne[1],
                (long long) inp->self_kq_mask_cnv->ne[2], (long long) inp->self_kq_mask_cnv->ne[3],
                (int) cparams.n_ubatch, (int) ubatch.n_tokens);
        }
        // 2026-05-31 PIN_MASK (env LLAMA_PREFILL_PIN_MASK, default off): at chunk5 the ~419MB f16
        // self_kq_mask_cnv is the largest n_kv-scaled compute tensor; in the layer-major repeated-
        // slice prefill its gallocr placement reshuffles the frozen layout so a per-layer (MoE)
        // neighbor overruns the attn_norm/kv_cmpr_pe slot -> layer1+ Kcur=0/nan (ggml-alloc.c:26-35
        // reuse-collision). Pinning the mask_cnv as a graph output takes the reshuffler OUT of the
        // reuse pool (low VRAM, unlike GGML_ALLOC_NO_FREE_REUSE which OOMs at 69GB). Marker PIN_MASK.
        if (cparams.flash_attn && inp->self_kq_mask_cnv != inp->self_kq_mask) {
            static const bool pin_mask = std::getenv("LLAMA_PREFILL_PIN_MASK") != nullptr;  // PIN_MASK
            if (pin_mask) ggml_set_output(inp->self_kq_mask_cnv);
        }
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        const auto n_kv = mctx_cur->get_swa()->get_n_kv();

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp->self_kq_mask_swa);
        ggml_set_name(inp->self_kq_mask_swa, "self_kq_mask_swa");

        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
        ggml_set_name(inp->self_kq_mask_swa_cnv, "self_kq_mask_swa_cnv");
        // 2026-08-28 PIN_MASK_SWA: the base-mask block above pins self_kq_mask_cnv under
        // LLAMA_PREFILL_PIN_MASK, but the SWA mask never got the same treatment -- and the SWA
        // _cnv is precisely what DSV4's CSA path consumes (get_kq_mask_swa() -> raw_mask in
        // build_csa_lid_attention). Measured 2026-08-28: with fa=ON the raw half of the consumed
        // mask carries 65536 zeros (everything visible) while its source leaf is exact
        // (-inf=40896 == a 128 window over 256x256); with fa=OFF, _cnv IS the leaf (the ternary's
        // else branch) and the needle is recalled at both 267 and 492 tokens.
        //
        // This pin DISCRIMINATES the two candidate mechanisms for those zeros:
        //   recall returns -> the cast ran and its buffer was RECYCLED (gallocr reuse pool), same
        //                     defect the base-mask pin was added for on 2026-05-31.
        //   still a miss   -> the cast NEVER RAN: it is built at the graph head and the layer-major
        //                     executor computes per-layer node ranges, so nothing in any range
        //                     evaluates it. Fix would be to rebuild the cast inside the consuming
        //                     layer range instead.
        // Same env var and same default (off) as the base pin, so this is inert unless asked for.
        if (cparams.flash_attn && inp->self_kq_mask_swa_cnv != inp->self_kq_mask_swa) {
            static const bool pin_mask_swa = std::getenv("LLAMA_PREFILL_PIN_MASK") != nullptr;  // PIN_MASK_SWA
            if (pin_mask_swa) {
                ggml_set_output(inp->self_kq_mask_swa_cnv);
                fprintf(stderr, "*** PIN_MASK_SWA: pinned self_kq_mask_swa_cnv [%lld,%lld]\n",
                        (long long) inp->self_kq_mask_swa_cnv->ne[0],
                        (long long) inp->self_kq_mask_swa_cnv->ne[1]);
            }
        }
    }

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

// DeepSeek-V4 compressed-KV / lightning-indexer graph input (ported from upstream dsv4,
// adapted to this fork's inline iswa kq-mask idiom in build_attn_inp_kv_iswa above).
llm_graph_input_dsv4 * llm_graph_context::build_inp_dsv4() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
    const auto * raw_ctx  = mctx_cur->get_raw();

    auto inp_raw = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, raw_ctx);

    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    {
        const auto n_kv = raw_ctx->get_base()->get_n_kv();

        inp_raw->self_k_idxs = raw_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);

        inp_raw->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp_raw->self_kq_mask);
        ggml_set_name(inp_raw->self_kq_mask, "self_kq_mask");

        inp_raw->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_raw->self_kq_mask, GGML_TYPE_F16) : inp_raw->self_kq_mask;
        ggml_set_name(inp_raw->self_kq_mask_cnv, "self_kq_mask_cnv");
        // [MASK_SITE_DIAG 2026-08-27] env-gated (LLAMA_MASK_SITE_DIAG). Z1a/Z4: a 512-row staged
        // mask copy meets a 256-row self_kq_mask_cnv in cpy.cu:409. Static reading could not say
        // WHICH build makes the 256 one, so every site that can create this tensor names itself.
        if (std::getenv("LLAMA_MASK_SITE_DIAG")) {
            fprintf(stderr, "*** MASK_SITE %s mask=[%lld,%lld,%lld,%lld] cnv=[%lld,%lld,%lld,%lld] n_ubatch=%d n_tokens=%d\n",
                __func__,
                (long long) inp_raw->self_kq_mask->ne[0], (long long) inp_raw->self_kq_mask->ne[1],
                (long long) inp_raw->self_kq_mask->ne[2], (long long) inp_raw->self_kq_mask->ne[3],
                (long long) inp_raw->self_kq_mask_cnv->ne[0], (long long) inp_raw->self_kq_mask_cnv->ne[1],
                (long long) inp_raw->self_kq_mask_cnv->ne[2], (long long) inp_raw->self_kq_mask_cnv->ne[3],
                (int) cparams.n_ubatch, (int) ubatch.n_tokens);
        }
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "DSV4 expects SWA raw cache");

        const auto n_kv = raw_ctx->get_swa()->get_n_kv();

        inp_raw->self_k_idxs_swa = raw_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);

        inp_raw->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp_raw->self_kq_mask_swa);
        ggml_set_name(inp_raw->self_kq_mask_swa, "self_kq_mask_swa");

        inp_raw->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_raw->self_kq_mask_swa, GGML_TYPE_F16) : inp_raw->self_kq_mask_swa;
        ggml_set_name(inp_raw->self_kq_mask_swa_cnv, "self_kq_mask_swa_cnv");
        // 2026-08-28 PIN_MASK_SWA_DSV4: THIS is the SWA _cnv that DSV4 actually consumes --
        // build_csa_lid_attention reads inp_attn->get_kq_mask_swa() off the inp_raw built HERE,
        // not off build_attn_inp_kv_iswa. Patching only the latter produced an inert arm (the
        // PIN_MASK_SWA marker never appeared in the server log), so both sites carry the pin and
        // DISTINCT markers -- whichever fires tells you which builder the arch went through.
        // Mechanism and predictions: see the PIN_MASK_SWA comment in build_attn_inp_kv_iswa.
        if (cparams.flash_attn && inp_raw->self_kq_mask_swa_cnv != inp_raw->self_kq_mask_swa) {
            static const bool pin_mask_swa_dsv4 = std::getenv("LLAMA_PREFILL_PIN_MASK") != nullptr;  // PIN_MASK_SWA_DSV4
            if (pin_mask_swa_dsv4) {
                ggml_set_output(inp_raw->self_kq_mask_swa_cnv);
                fprintf(stderr, "*** PIN_MASK_SWA_DSV4: pinned inp_raw->self_kq_mask_swa_cnv [%lld,%lld]\n",
                        (long long) inp_raw->self_kq_mask_swa_cnv->ne[0],
                        (long long) inp_raw->self_kq_mask_swa_cnv->ne[1]);
            }
        }
    }

    // QK Hadamard rotation matrices (nullptr unless attn_rot_k enabled for DEEPSEEK4)
    inp_raw->self_k_rot     = raw_ctx->get_base()->build_input_k_rot(ctx0);
    inp_raw->self_k_rot_swa = raw_ctx->get_swa()->build_input_k_rot(ctx0);

    auto inp = std::make_unique<llm_graph_input_dsv4>(cparams, std::move(inp_raw), mctx_cur);

    dsv4_build_comp_inputs(ctx0, inp->inp_csa, mctx_cur->get_csa_plan(), "csa");
    dsv4_build_comp_inputs(ctx0, inp->inp_hca, mctx_cur->get_hca_plan(), "hca");
    dsv4_build_comp_inputs(ctx0, inp->inp_lid, mctx_cur->get_lid_plan(), "lid");

    // The DSV4-owned raw-window mask (see the declaration for why). Same shape and dtype as the
    // ISWA leaf, but it is a DSV4 input consumed DIRECTLY by the compressed attention -- no
    // head-of-graph cast -- so the scheduler stages it within the range that reads it, exactly as
    // it already does for inp_csa.kq_mask.
    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        const auto n_kv_raw = raw_ctx->get_swa()->get_n_kv();
        inp->raw_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
                n_kv_raw, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp->raw_kq_mask);
        ggml_set_name(inp->raw_kq_mask, "dsv4_raw_kq_mask");
        // [MASKPIN_OWN 2026-08-28] Take this tensor OUT of the gallocr reuse pool.
        //
        // Measured, on the real 256-token streaming pass (not warmup -- the echo is shape-gated):
        //     MASKFILL_ECHO: ne=[256,256] wrote 262144 bytes, read back inf=40896   <- correct
        //     NODE_SCAN il=42 'csa_raw_mask-42' ne=[256,256]            inf=0       <- consumed
        // i.e. set_input writes a correct mask into this tensor and a ggml_cont of that same tensor
        // inside layer 42 reads ZEROS. The write lands (immediate read-back proves it) and the value
        // is lost before the consumer runs, which is the signature of the buffer being reused.
        //
        // Note this is NOT the pin that failed on self_kq_mask_swa_cnv (2026-08-28, marker fired 7x,
        // mask still inf=0). That one is a ggml_cast NODE which the layer-major executor never
        // evaluates, so pinning stabilised a buffer nothing ever writes. THIS tensor's content comes
        // from a host->device copy in set_input that provably runs, so protecting its buffer is the
        // right tool. Same API, different failure mode -- do not let the earlier null result veto it.
        //
        // Gated on the consumer knob so the tensor keeps its current no-buffer behaviour (and this
        // stays completely inert) whenever the feature is off.
        {
            static const bool own_knob = []() {
                const char * e = std::getenv("LLAMA_DSV4_RAW_MASK_OWN");
                return e && e[0] != '\0' && e[0] != '0';
            }();
            if (own_knob) {
                ggml_set_output(inp->raw_kq_mask);
                // Runtime marker: a comment cannot be grepped out of the DLL, so a negative result
                // would be indistinguishable from "the pin was never compiled in". Print once.
                static int pin_own_dbg = 0;
                if (pin_own_dbg < 3) {
                    pin_own_dbg++;
                    fprintf(stderr, "*** MASKPIN_OWN: pinned dsv4_raw_kq_mask ne=[%lld,%lld]\n",
                            (long long) inp->raw_kq_mask->ne[0],
                            (long long) inp->raw_kq_mask->ne[1]);
                    fflush(stderr);
                }
            }
        }
    }
    inp->inp_lid.k_rot = mctx_cur->get_lid()->build_input_k_rot(ctx0);

    return (llm_graph_input_dsv4 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, rs_size);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_1d(ctx0, s, state_size*(n_rs - n_seqs), (rs_head + n_seqs)*state_size*ggml_element_size(s))));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    {
        const auto n_kv = attn_ctx->get_base()->get_n_kv();

        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp_attn->self_kq_mask);
        ggml_set_name(inp_attn->self_kq_mask, "self_kq_mask");  // required by split-decode CUDA graph kq_mask capture

        inp_attn->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_attn->self_kq_mask, GGML_TYPE_F16) : inp_attn->self_kq_mask;
        ggml_set_name(inp_attn->self_kq_mask_cnv, "self_kq_mask_cnv");
    }

    {
        const auto n_kv = attn_ctx->get_swa()->get_n_kv();

        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp_attn->self_kq_mask_swa);

        inp_attn->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_attn->self_kq_mask_swa, GGML_TYPE_F16) : inp_attn->self_kq_mask_swa;
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    cur = ggml_tanh(ctx0, cur);
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, int32_t> seq_to_logit_row;
    int32_t logit_row_idx = 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (ubatch.output[i]) {
            llama_seq_id seq_id = ubatch.seq_id[i][0];
            seq_to_logit_row[seq_id] = logit_row_idx;
            logit_row_idx++;
        }
    }

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // add a dummy row of logits
    // this trick makes the graph static, regardless of which samplers are activated
    // this is important in order to minimize graph reallocations
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = seq_to_logit_row.find(seq_id);

        // inactive samplers always work on the first row
        const auto row_idx = it != seq_to_logit_row.end() ? it->second : 0;
        const int i_out    = it != seq_to_logit_row.end() ? 1          : 0;

        ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], row_idx * logits_t->nb[1]);
        ggml_format_name(logits_seq, "logits_seq_%d", seq_id);

        struct llama_sampler_data data = {
            /*.logits      =*/ logits_seq,
            /*.probs       =*/ nullptr,
            /*.sampled     =*/ nullptr,
            /*.candidates  =*/ nullptr,
        };

        assert(sampler->iface->backend_apply);
        sampler->iface->backend_apply(sampler, ctx0, gf, &data);

        if (data.sampled != nullptr) {
            res->t_sampled[seq_id] = data.sampled;
            outs[1] = data.sampled;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.probs != nullptr) {
            res->t_sampled_probs[seq_id] = data.probs;
            outs[1] = data.probs;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.logits != nullptr) {
            res->t_sampled_logits[seq_id] = data.logits;
            outs[1] = data.logits;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.candidates != nullptr) {
            res->t_candidates[seq_id] = data.candidates;
            outs[1] = data.candidates;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }
    }

    // TODO: Call llama_sampler_accept_ggml after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        if (auto it = res->t_sampled.find(seq_id); it != res->t_sampled.end()) {
            ggml_tensor * selected_token = it->second;
            if (selected_token != nullptr) {
                llama_sampler_accept_ggml(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
