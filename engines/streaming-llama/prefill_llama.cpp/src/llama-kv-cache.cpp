#include "llama-kv-cache.h"
#include "llama-kv-tier.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

//
// DSV4 / DeepSeek lightning-indexer QK Hadamard rotation (upstream PR 21038).
// Generates a normalized Hadamard matrix in-place. Ported verbatim from
// upstream llama.cpp origin/dsv4 (src/llama-kv-cache.cpp).
//
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(n > 0 && (n & (n - 1)) == 0);
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

// Diagnostic-only census of the exact, already-assembled bytes that are about
// to be appended to a KV state blob. There is deliberately no default-path
// payload scan: callers invoke this only when LLAMA_KV_BLOB_SCAN=1.
static void llama_kv_blob_scan_v1(
        const char * kind,
        uint32_t il,
        uint32_t strm,
        ggml_type type,
        const void * data,
        size_t total_bytes,
        size_t row_bytes,
        int64_t row_elements,
        const char * layout) {
    if (!data || row_bytes == 0 || total_bytes % row_bytes != 0) {
        fprintf(stderr,
                "*** KV_BLOB_SCAN_V1 ERROR: kind=%s layer=%u stream=%u type=%s "
                "bytes=%zu row_bytes=%zu divisible=%d\n",
                kind, il, strm, ggml_type_name(type), total_bytes, row_bytes,
                (int) (row_bytes != 0 && total_bytes % row_bytes == 0));
        fflush(stderr);
        return;
    }

    const uint8_t * raw = (const uint8_t *) data;
    const size_t rows = total_bytes / row_bytes;
    size_t zero_rows = 0;
    int64_t first_zero_row = -1;
    for (size_t row = 0; row < rows; ++row) {
        const uint8_t * row_ptr = raw + row * row_bytes;
        bool all_zero = true;
        for (size_t i = 0; i < row_bytes; ++i) {
            if (row_ptr[i] != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            if (first_zero_row < 0) {
                first_zero_row = (int64_t) row;
            }
            ++zero_rows;
        }
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    const bool packed_row_layout =
        row_elements > 0 && traits && traits->blck_size > 0 &&
        row_elements % traits->blck_size == 0 &&
        (size_t) (row_elements / traits->blck_size) * traits->type_size == row_bytes;
    const bool finite_supported =
        packed_row_layout &&
        (type == GGML_TYPE_F32 || type == GGML_TYPE_F64 ||
         (traits && traits->to_float != nullptr));
    size_t nan_values = 0;
    size_t inf_values = 0;
    size_t nonfinite_rows = 0;
    int64_t first_nonfinite_row = -1;
    std::vector<float> decoded;
    if (finite_supported) {
        decoded.resize((size_t) row_elements);
        for (size_t row = 0; row < rows; ++row) {
            const uint8_t * row_ptr = raw + row * row_bytes;
            bool row_nonfinite = false;
            if (type == GGML_TYPE_F32) {
                memcpy(decoded.data(), row_ptr, (size_t) row_elements * sizeof(float));
            } else if (type == GGML_TYPE_F64) {
                for (int64_t i = 0; i < row_elements; ++i) {
                    double value;
                    memcpy(&value, row_ptr + (size_t) i * sizeof(double), sizeof(value));
                    if (std::isnan(value)) {
                        ++nan_values;
                        row_nonfinite = true;
                    } else if (std::isinf(value)) {
                        ++inf_values;
                        row_nonfinite = true;
                    }
                }
            } else {
                traits->to_float(row_ptr, decoded.data(), row_elements);
            }

            if (type != GGML_TYPE_F64) {
                for (float value : decoded) {
                    if (std::isnan(value)) {
                        ++nan_values;
                        row_nonfinite = true;
                    } else if (std::isinf(value)) {
                        ++inf_values;
                        row_nonfinite = true;
                    }
                }
            }
            if (row_nonfinite) {
                if (first_nonfinite_row < 0) {
                    first_nonfinite_row = (int64_t) row;
                }
                ++nonfinite_rows;
            }
        }
    }

    fprintf(stderr,
            "*** KV_BLOB_SCAN_V1: kind=%s layer=%u stream=%u type=%s layout=%s "
            "rows=%zu row_bytes=%zu bytes=%zu zero_rows=%zu first_zero=%lld "
            "finite_scan=%s nonfinite_rows=%zu first_nonfinite=%lld nan_values=%zu inf_values=%zu\n",
            kind, il, strm, ggml_type_name(type), layout, rows, row_bytes,
            total_bytes, zero_rows, (long long) first_zero_row,
            finite_supported ? "yes" : "n/a", nonfinite_rows,
            (long long) first_nonfinite_row, nan_values, inf_values);
    fflush(stderr);
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                     bool   sparse_mla,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse) :
    llama_kv_cache(model, model.hparams, type_k, type_v, v_trans, offload, unified, sparse_mla,
            kv_size, n_seq_max, n_pad, n_swa, swa_type, filter, reuse) {
}

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
      const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                     bool   sparse_mla,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer_kv = hparams.n_layer_kv();

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                // 3u (not 2u) reserves tensor-struct headroom for K + V + the optional indexer-K cache
                // (k_idx, 1 base + n_stream views per layer); needed since M3 sparse / GLM-DSA allocate it.
                /*.mem_size   =*/ size_t(3u*(1 + n_stream)*n_layer_kv*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < hparams.n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);

            // In -ngl 0 streaming prefill, dev_layer(il) often resolves to CPU.
            // Prefer a real accelerator device for KV cache when available so
            // cache SET_ROWS/VIEW/PERMUTE ops can stay on GPU.
            if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && model.n_gpu_layers() == 0) {
                // First try model.devices
                for (auto * candidate : model.devices) {
                    if (candidate && ggml_backend_dev_type(candidate) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                        dev = candidate;
                        break;
                    }
                }
                // Fallback: scan the global device registry
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
                        auto * candidate = ggml_backend_dev_get(di);
                        if (candidate && ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                            dev = candidate;
                            break;
                        }
                    }
                }
            }

            LLAMA_LOG_DEBUG("*** KV layer %3d: offload=%d ngl=%u n_devs=%zu dev=%s type=%d ***\n",
                il, offload, model.n_gpu_layers(), model.devices.size(),
                ggml_backend_dev_name(dev), ggml_backend_dev_type(dev));

            if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                buft = ggml_backend_dev_buffer_type(dev);
                dev_name = ggml_backend_dev_name(dev);
            }
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        // Phase 3.5-lite (iter 29): allocate indexer K cache for GLM_DSA / sparse-MLA
        // models. Sized [indexer_head_size, kv_size, n_stream] in fp16. Skipped for
        // models without indexer tensors (hparams.indexer_n_head == 0).
        //
        // Iter 32: also skip when sparse MLA is disabled; the cache is ~8MB/layer
        // at 32k ctx (~624MB total for 78 layers), and with LLAMA_NO_HOST=1 this lands
        // in pinned host RAM, eating bandwidth from weight streaming. GLM-5 carries
        // indexer_n_head=32 in its metadata even though we only use sparse when the
        // caller opts in via runtime config.
        ggml_tensor * k_idx = nullptr;
        std::vector<ggml_tensor *> k_idx_stream;
        // minimax-m3 (GQA block-sparse) needs the indexer-K cache for its DECODE path, but it does NOT
        // use the deepseek2 MLA sparse path -> allocate when M3 sparse is enabled (LLAMA_M3_SPARSE) even
        // though cparams.sparse_mla stays false (setting that would wrongly turn on the MLA graph).
        static const bool m3_sparse_idx = [] {
            const char * e = std::getenv("LLAMA_M3_SPARSE");
            return e && e[0] != '\0' && e[0] != '0';
        }();
        if ((sparse_mla || m3_sparse_idx) && hparams.indexer_n_head > 0 && hparams.indexer_head_size > 0) {
            const uint32_t idx_head_size = hparams.indexer_head_size;
            // M3 uses an F32 indexer cache so its block selection is BIT-IDENTICAL to the direct-score
            // path (keeps the coherence gate bit-exact); deepseek2 / GLM-DSA keep F16.
            const enum ggml_type idx_type = m3_sparse_idx ? GGML_TYPE_F32 : GGML_TYPE_F16;
            k_idx = ggml_new_tensor_3d(ctx, idx_type, idx_head_size, kv_size, n_stream);
            ggml_format_name(k_idx, "cache_k_idx_l%d", il);
            for (uint32_t s = 0; s < n_stream; ++s) {
                k_idx_stream.push_back(ggml_view_2d(ctx, k_idx, idx_head_size, kv_size, k_idx->nb[1], s*k_idx->nb[2]));
            }
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, k_idx, k_idx_stream });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < hparams.n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // VIRTUAL KV (2026-09-10, LLAMA_KV_TIER_VIRTUAL_KV=1 with the NVMe tier, CPU placement only): the
    // streaming pass points every layer's K/V/k_idx at the GPU staging before it computes, the tier files
    // hold the rows, and nothing reads the cache's own buffer -- so it is not allocated. At 262144 cells
    // that buffer would be 27 GB of host RAM for GLM-5.3; this is what lets rtx.ctx reach the Mac's.
    const bool virtual_kv = !offload && llama_kv_tier::enabled() && []() {
        const char * e = std::getenv("LLAMA_KV_TIER_VIRTUAL_KV");
        return e && e[0] != 0 && e[0] != '0';
    }();
    lazy_host_buffers = !offload && !virtual_kv && []() {
        const auto kv_cfg = llama_prefill_resolve_kv_mode_from_env();
        const char * keep = std::getenv("LLAMA_PREFILL_KV_RAM_HOST_CLEAR");
        return kv_cfg.mode == LLAMA_KV_RAM && kv_cfg.forced && !(keep && keep[0] != 0 && keep[0] != '0');
    }();
    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (virtual_kv) {
            const size_t would_be = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
            LLAMA_LOG_INFO("%s: KV tensors are VIRTUAL: %.2f MiB of %s buffer not allocated (the streaming pass points them at the GPU staging; the NVMe tier holds the rows) [VIRTUAL_KV]\n",
                           __func__, would_be / (1024.0 * 1024.0), ggml_backend_buft_name(buft));
        } else if (model.hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        if (lazy_host_buffers && ggml_backend_buft_is_host(buft)) {
            // Fresh large allocations are demand-zero pages: same contents as a clear, no RAM until touched.
            LLAMA_LOG_INFO("%s: %10s KV buffer left demand-zero, %.2f MiB not made resident (KV-RAM streaming keeps the rows in its mirror) [KV_RAM_LAZY_HOST]\n",
                           __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        } else {
            ggml_backend_buffer_clear(buf, 0);
        }
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // DSV4 / DeepSeek lightning-indexer QK Hadamard rotation (upstream PR 21038).
    // This fork uses a uniform per-head key/value dimension (hparams.n_embd_head_k/v scalars),
    // so n_embd_head_k_all/v_all are just those scalars. attn_rot_k/v remain false unless enabled
    // below, keeping behavior byte-identical for all non-DEEPSEEK4 archs.
    n_embd_head_k_all = (int32_t) hparams.n_embd_head_k;
    n_embd_head_v_all = (int32_t) hparams.n_embd_head_v;

    {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if (model.arch == LLM_ARCH_DEEPSEEK4 &&
                hparams.n_embd_head_k == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v % 64 == 0;

        if (attn_rot_k || attn_rot_v) {
            LLAMA_LOG_INFO("%s: attn_rot_k = %d, attn_rot_v = %d, n_embd_head_k_all = %d, n_embd_head_v_all = %d\n",
                    __func__, attn_rot_k, attn_rot_v, n_embd_head_k_all, n_embd_head_v_all);

            // pre-compute the Hadamard matrices and keep them in host memory
            for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
                attn_rot_hadamard[n] = std::vector<float>(n*n);

                ggml_init_params hp = {
                    /* .mem_size   = */ 1*ggml_tensor_overhead(),
                    /* .mem_buffer = */ nullptr,
                    /* .no_alloc   = */ true,
                };

                ggml_context_ptr hctx { ggml_init(hp) };

                ggml_tensor * tmp = ggml_new_tensor_2d(hctx.get(), GGML_TYPE_F32, n, n);
                tmp->data = attn_rot_hadamard[n].data();

                ggml_gen_hadamard(tmp);
            }
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            if (lazy_host_buffers && ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buf.get()))) {
                continue;  // never read in KV-RAM streaming; clearing would make the whole context resident
            }
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // Diagnostic: verify sinfo is valid
        if (sinfo_new.s0 > 100 || sinfo_new.s1 > 100) {
            fprintf(stderr, "*** PREPARE_CORRUPT: s0=%u s1=%u ns=%u ubatch.n_tokens=%u\n",
                    sinfo_new.s0, sinfo_new.s1, sinfo_new.s1 - sinfo_new.s0 + 1,
                    ubatch.n_tokens);
        }

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                // Phase 320 (2026-05-05): when n_stream==1 and many seqs share
                // the unified stream, cells accumulate multi-seq ownership
                // (seq_count > 1) under continuous-batching workloads (NP>=64).
                // The original `seq_count == 1` constraint rejected these cells,
                // causing find_slot to fail despite ~99% of cells being empty
                // somewhere in the cache. Relax to also consider multi-seq cells
                // for SWA eviction when n_stream==1; the SWA check still
                // ensures only safe-to-recycle cells are reused.
                const bool relax_multi_seq = !can_use && (n_stream == 1) && (cells.seq_count(idx) > 1);
                if (!can_use && (cells.seq_count(idx) == 1 || relax_multi_seq)) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                break;  // Phase 332: drop out of inner while; second-pass scan below
            }
        }

        // Phase 332 (2026-05-05): if first-pass scan didn't find enough cells,
        // run a second pass that ONLY accepts is_empty cells. The first pass's
        // can_use logic (seq_count==1 + SWA-evictable, plus Phase 320's
        // multi-seq relaxation) can exhaust scan budget before reaching the
        // is_empty cells freed by Phase 329's release-clears-KV. The second
        // pass starts at idx 0 and looks specifically for empty cells, which
        // recently-released slots produced via prompt_clear -> seq_rm.
        if (n_stream == 1 && res.idxs[s].size() < n_tokens) {
            res.idxs[s].clear();  // discard partial first-pass result
            for (uint32_t idx = 0; idx < cells.size() && res.idxs[s].size() < n_tokens; ++idx) {
                if (cells.is_empty(idx)) {
                    res.idxs[s].push_back(idx);
                }
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // K-shift assumes a single global RoPE dimension. These ISWA models use
    // per-layer RoPE geometry, so shifting cached K values can corrupt them.
    if (model.arch == LLM_ARCH_STEP35 ||
        model.arch == LLM_ARCH_GEMMA4 ||
        model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false;
    }
    return true;
}

bool llama_kv_cache::has_layer(int32_t il) const {
    return map_layer_ids.find(il) != map_layer_ids.end();
}

void * llama_kv_cache::get_k_data(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    return layers[ikv].k->data;
}

void * llama_kv_cache::get_v_data(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    return layers[ikv].v ? layers[ikv].v->data : nullptr;
}

void llama_kv_cache::set_k_data(int32_t il, void * data, ggml_backend_buffer_t buf) {
    const int32_t ikv = map_layer_ids.at(il);
    layers[ikv].k->data = data;
    if (buf) layers[ikv].k->buffer = buf;
}

void llama_kv_cache::set_v_data(int32_t il, void * data, ggml_backend_buffer_t buf) {
    const int32_t ikv = map_layer_ids.at(il);
    if (!layers[ikv].v) return;  // MLA: no V tensor
    layers[ikv].v->data = data;
    if (buf) layers[ikv].v->buffer = buf;
}

size_t llama_kv_cache::get_k_layer_bytes(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    return ggml_nbytes(layers[ikv].k);
}

size_t llama_kv_cache::get_v_layer_bytes(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    return layers[ikv].v ? ggml_nbytes(layers[ikv].v) : 0;
}

size_t llama_kv_cache::get_k_idx_layer_bytes(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    return layers[ikv].k_idx ? ggml_nbytes(layers[ikv].k_idx) : 0;
}

ggml_tensor * llama_kv_cache::get_k_tensor(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return nullptr;
    return layers[it->second].k;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

ggml_tensor * llama_kv_cache::get_v_tensor(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return nullptr;
    return layers[it->second].v;
}

ggml_tensor * llama_kv_cache::get_k_idx_tensor(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return nullptr;
    return layers[it->second].k_idx;
}

void * llama_kv_cache::get_k_idx_data(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    return layers[ikv].k_idx ? layers[ikv].k_idx->data : nullptr;
}

void llama_kv_cache::set_k_data_all(int32_t il, void * data, ggml_backend_buffer_t buf) {
    const int32_t ikv = map_layer_ids.at(il);
    void * old_base = layers[ikv].k->data;
    layers[ikv].k->data = data;
    layers[ikv].k->buffer = buf;
    for (auto * view : layers[ikv].k_stream) {
        if (view) {
            ptrdiff_t off = (char *)view->data - (char *)old_base;
            view->data = (char *)data + off;
            view->buffer = buf;
        }
    }
}

void llama_kv_cache::set_v_data_all(int32_t il, void * data, ggml_backend_buffer_t buf) {
    const int32_t ikv = map_layer_ids.at(il);
    if (!layers[ikv].v) return;
    void * old_base = layers[ikv].v->data;
    layers[ikv].v->data = data;
    layers[ikv].v->buffer = buf;
    for (auto * view : layers[ikv].v_stream) {
        if (view) {
            ptrdiff_t off = (char *)view->data - (char *)old_base;
            view->data = (char *)data + off;
            view->buffer = buf;
        }
    }
}

void llama_kv_cache::set_k_idx_data_all(int32_t il, void * data, ggml_backend_buffer_t buf) {
    const int32_t ikv = map_layer_ids.at(il);
    if (!layers[ikv].k_idx) return;
    void * old_base = layers[ikv].k_idx->data;
    layers[ikv].k_idx->data = data;
    layers[ikv].k_idx->buffer = buf;
    for (auto * view : layers[ikv].k_idx_stream) {
        if (view) {
            ptrdiff_t off = (char *)view->data - (char *)old_base;
            view->data = (char *)data + off;
            view->buffer = buf;
        }
    }
}

ggml_backend_buffer_type_t llama_kv_cache::free_device_buffer() {
    ggml_backend_buffer_type_t buft_saved = nullptr;

    for (auto & [ctx, buf] : ctxs_bufs) {
        if (!buf) continue;

        auto * buft = ggml_backend_buffer_get_type(buf.get());
        if (ggml_backend_buft_is_host(buft)) continue;  // skip CPU/host buffers

        buft_saved = buft;

        LLAMA_LOG_INFO("%s: freeing device buffer %.1f MiB\n", __func__,
                       ggml_backend_buffer_get_size(buf.get()) / (1024.0 * 1024.0));

        // Create a size-0 dummy buffer of the same type. This keeps t->buffer
        // valid for the scheduler (which checks buffer type during compute) while
        // freeing the actual GPU memory. Same pattern as no_alloc mode.
        ggml_backend_buffer_t dummy = ggml_backend_buft_alloc_buffer(buft, 0);

        // Point all tensors to the dummy buffer, null data pointers
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t; t = ggml_get_next_tensor(ctx.get(), t)) {
            t->buffer = dummy;
            t->data   = nullptr;
        }

        // Replace real buffer with dummy — frees the GPU memory
        buf.reset(dummy);
        break;  // typically only one device buffer
    }

    return buft_saved;
}

bool llama_kv_cache::realloc_device_buffer(ggml_backend_buffer_type_t buft) {
    if (!buft) return false;

    for (auto & [ctx, buf] : ctxs_bufs) {
        if (!buf) continue;

        // Find the dummy (size-0) device buffer
        if (ggml_backend_buffer_get_size(buf.get()) > 0) continue;
        auto * cur_buft = ggml_backend_buffer_get_type(buf.get());
        if (ggml_backend_buft_is_host(cur_buft)) continue;

        // Null out all tensor buffer/data pointers (required by ggml_backend_tensor_alloc)
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t; t = ggml_get_next_tensor(ctx.get(), t)) {
            t->buffer = nullptr;
            t->data   = nullptr;
        }

        // Free dummy buffer, then allocate real one
        buf.reset();
        ggml_backend_buffer_t new_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!new_buf) {
            LLAMA_LOG_ERROR("%s: failed to reallocate device buffer\n", __func__);
            return false;
        }

        ggml_backend_buffer_clear(new_buf, 0);
        buf.reset(new_buf);

        LLAMA_LOG_INFO("%s: reallocated device buffer %.1f MiB\n", __func__,
                       ggml_backend_buffer_get_size(new_buf) / (1024.0 * 1024.0));
        return true;
    }

    LLAMA_LOG_ERROR("%s: no freed entry found in ctxs_bufs\n", __func__);
    return false;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

void llama_kv_cache::sync_view_pointers() {
    int repointed = 0;
    for (auto & layer : layers) {
        // Simpler: for each stream, recalculate data = k->data + s * k->nb[2]
        if (layer.k) {
            for (uint32_t s = 0; s < layer.k_stream.size(); s++) {
                auto * ks = layer.k_stream[s];
                if (ks) {
                    void * expected = (char*)layer.k->data + s * layer.k->nb[2];
                    if (ks->data != expected) {
                        ks->data = expected;
                        repointed++;
                    }
                }
            }
        }
        // Update v_stream views from parent v tensor
        if (layer.v) {
            for (uint32_t s = 0; s < layer.v_stream.size(); s++) {
                auto * vs = layer.v_stream[s];
                if (vs) {
                    void * expected = (char*)layer.v->data + s * layer.v->nb[2];
                    if (vs->data != expected) {
                        vs->data = expected;
                        repointed++;
                    }
                }
            }
        }
        // Phase 3.5-lite (iter 29): sync k_idx_stream views too. Missing sync caused
        // serialization to read from stale null data pointers, producing bit-identical
        // bytes across ALL layers at ALL positions — broke Mac-side sparse decode.
        if (layer.k_idx) {
            for (uint32_t s = 0; s < layer.k_idx_stream.size(); s++) {
                auto * ks = layer.k_idx_stream[s];
                if (ks) {
                    void * expected = (char*)layer.k_idx->data + s * layer.k_idx->nb[2];
                    if (ks->data != expected) {
                        ks->data = expected;
                        repointed++;
                    }
                }
            }
        }
    }
    if (repointed > 0) {
        fprintf(stderr, "*** KV_REPOINT: updated %d view data pointers\n", repointed);
        fflush(stderr);
    }
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    const uint32_t n_embd_head_k = hparams.n_embd_head_k_layer(il);

    if (ns > 1000 || n_kv > 4000000) {
        fprintf(stderr, "*** GET_K_CORRUPT: il=%d ikv=%d n_kv=%u ns=%u s0=%u s1=%u kv_size=%llu k->ne=[%lld,%lld,%lld,%lld]\n",
                il, ikv, n_kv, ns, sinfo.s0, sinfo.s1, (unsigned long long)kv_size,
                (long long)k->ne[0], (long long)k->ne[1], (long long)k->ne[2], (long long)k->ne[3]);
    }

    return ggml_view_4d(ctx, k,
            n_embd_head_k, hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, n_embd_head_k),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    const uint32_t n_embd_head_v = hparams.n_embd_head_v_layer(il);

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                n_embd_head_v, hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, n_embd_head_v),                  // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), n_embd_head_v, ns,
            ggml_row_size(v->type, kv_size*n_embd_head_v),          // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    // B1 (env LLAMA_PREFILL_KCUR_CONT, default off): at chunk5 (n_kv>=163840) the
    // set_rows src0 (this k_cur VIEW into the K-projection buffer) reads EXACTLY zero
    // for layers>=1 (SETROWS_POST WROTE=0 src0=0.00000), while layer0 is fine. If the
    // K-projection buffer is reused by gallocr BEFORE set_rows reads the view (WAR
    // hazard at deep n_kv), materializing the view into a fresh contiguous tensor that
    // set_rows reads live defeats the reuse. Marker KCUR_CONT for build-grep.
    {
        static const bool kcur_cont = std::getenv("LLAMA_PREFILL_KCUR_CONT") != nullptr;  // KCUR_CONT
        if (kcur_cont) {
            k_cur = ggml_cont(ctx, k_cur);
        }
    }

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    ggml_tensor * result = ggml_set_rows(ctx, k, k_cur, k_idxs);
    ggml_format_name(result, "kv_k_store-%d", il);

    // Phase 148 partial-UB diag: log K SET_ROWS dispatch shape per layer.
    // With KIMI_PARTIAL_UB=1 the last microbatch has fewer rows than the
    // n_ubatch the graph was built for; this dump localises whether
    // k_cur->ne[1] (token-count) tracks the per-ubatch n_tokens correctly
    // or stays at ubatch_max. Gated behind LLAMA_K_SET_ROWS_DIAG=1.
    {
        static const bool k_setrows_diag = []() {
            const char * env = std::getenv("LLAMA_K_SET_ROWS_DIAG");
            return env && env[0] != '\0' && env[0] != '0';
        }();
        if (k_setrows_diag) {
            fprintf(stderr,
                    "*** K_SET_ROWS: il=%d k_cur ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] "
                    "k_dst ne=[%lld,%lld] k_idxs ne[0]=%lld nb[0]=%zu n_stream=%lld result_ne=[%lld,%lld]\n",
                    il,
                    (long long) k_cur->ne[0], (long long) k_cur->ne[1],
                    (long long) k_cur->ne[2], (long long) k_cur->ne[3],
                    k_cur->nb[0], k_cur->nb[1], k_cur->nb[2], k_cur->nb[3],
                    (long long) k->ne[0], (long long) k->ne[1],
                    (long long) k_idxs->ne[0], k_idxs->nb[0],
                    (long long) n_stream,
                    (long long) result->ne[0], (long long) result->ne[1]);
            fflush(stderr);
        }
    }
    // Phase 145: tried ggml_set_output(result) and ggml_set_output(k)
    // to mark the K cache buffer as non-reusable by the gallocr pool.
    // Neither helped — same 7020/28672 zero rows in [1, 12286] pattern.
    // The K cache's actual data lives in kv_staging_buf (allocated
    // separately via ggml_backend_buft_alloc_buffer in
    // llama-context.cpp:5534, then attached to k via
    // cache->set_k_data_all). The staging buffer is OUTSIDE the
    // gallocr pool, so OUTPUT-flag is a no-op for it.
    //
    // The MoE intermediate that's clobbering K cells must therefore
    // be writing directly to the staging address, not via gallocr
    // pool reuse. Most likely: a MoE op's dst tensor was assigned the
    // SAME staging GPU address by the scheduler. Investigation
    // continues — needs ggml-alloc tensor→address dump for one MoE
    // layer.
    return result;
}

// Phase 3.5-lite (iter 29): indexer K cache get/set for GLM_DSA sparse MLA.
ggml_tensor * llama_kv_cache::get_k_idx(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k_idx = layers[ikv].k_idx;
    if (k_idx == nullptr) {
        return nullptr;  // model doesn't have indexer tensors
    }

    const uint64_t kv_size      = get_size();
    const uint64_t idx_head_sz  = k_idx->ne[0];

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    // View: [idx_head_size, n_kv, ns] starting at stream s0
    return ggml_view_3d(ctx, k_idx,
            idx_head_sz, n_kv, ns,
            ggml_row_size(k_idx->type, idx_head_sz),           // nb1 — one row of head_dim
            ggml_row_size(k_idx->type, idx_head_sz*kv_size),   // nb2 — one full stream slab
            ggml_row_size(k_idx->type, idx_head_sz*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k_idx(ggml_context * ctx, ggml_tensor * k_idx_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k_idx = layers[ikv].k_idx;
    if (k_idx == nullptr) {
        return nullptr;  // no-op when model lacks indexer tensors
    }

    // k_idx_cur shape: [idx_head_size, 1, n_tokens] or [idx_head_size, n_tokens]
    // Flatten any stray singleton dim to [idx_head_size, n_tokens] for set_rows.
    const int64_t idx_head_sz = k_idx_cur->ne[0];
    const int64_t n_tokens    = k_idx_cur->ne[k_idx_cur->ne[1] == 1 ? 2 : 1];
    k_idx_cur = ggml_view_2d(ctx, k_idx_cur, idx_head_sz, n_tokens,
                              ggml_row_size(k_idx_cur->type, idx_head_sz), 0);

    const int64_t n_stream = k_idx->ne[2];
    ggml_tensor * k_idx_dst = k_idx;
    if (n_stream > 1) {
        const int64_t kv_size = get_size();
        k_idx_dst = ggml_reshape_2d(ctx, k_idx, idx_head_sz, kv_size*n_stream);
    }

    ggml_tensor * result = ggml_set_rows(ctx, k_idx_dst, k_idx_cur, k_idxs);
    ggml_format_name(result, "kv_k_idx_store-%d", il);
    return result;
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        ggml_tensor * result = ggml_set_rows(ctx, v, v_cur, v_idxs);
        ggml_format_name(result, "kv_v_store-%d", il);
        return result;
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    ggml_tensor * result = ggml_set_rows(ctx, v_view, v_cur, v_idxs);
    ggml_format_name(result, "kv_v_store-%d", il);
    return result;
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);
    ggml_set_name(k_idxs, "k_idxs");  // required by split-decode CUDA-graph capture/replay (iter 19)

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);
    ggml_set_name(v_idxs, "v_idxs");  // required by split-decode CUDA-graph capture/replay (iter 19)

    return v_idxs;
}

// DSV4 / DeepSeek lightning-indexer QK Hadamard rotation (upstream PR 21038).
// Returns nullptr unless attn_rot_k/attn_rot_v is enabled (false for existing archs).
ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // use the largest power-of-two rotation that still divides the head dim
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(n_rot));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(n_rot));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::fill_input_k_idxs(std::vector<int64_t> & dst, int64_t n_total, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    // Fill unused index entries with a safe target (last valid cell + 1).
    // This prevents stale indices from a previous ubatch causing writes to wrong cells
    // when the current ubatch has fewer tokens than the tensor was sized for.
    // Writing to an empty padding cell (beyond used cells) is harmless.
    // Padding indices for graph-allocated tokens beyond this ubatch's real tokens.
    // Must use a cell BEYOND n_kv_max so flash attention never reads the garbage
    // K/V data that SET_ROWS writes there. Using last_valid+1 is WRONG because
    // that cell is within n_kv range and FA would attend to garbage data.
    const int64_t safe_idx = (int64_t)get_size() - 1;  // last cell, beyond n_kv_max
    dst.assign((size_t) n_total, safe_idx);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            const int64_t j = s*sinfo.size() + i;
            if (j < n_total) {
                dst[(size_t) j] = offs + sinfo.idxs[s][i];
            }
        }
    }
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    // 2026-07-02 (DSA-on multi-ubatch): a graph-input tensor that no live node
    // references is left unallocated by the scheduler (buffer==NULL). Mirrors the
    // llm_graph_input_out_ids::set_input guard: nothing to feed, so skip. Without
    // this, the DSA-on sparse path (where the dense mask/idx inputs can go dead)
    // crashed in ggml_backend_buffer_is_host(NULL) during layer-major staging.
    if (!dst->buffer) {
        return;
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    std::vector<int64_t> idxs;
    fill_input_k_idxs(idxs, ggml_nelements(dst), ubatch, sinfo);
    memcpy(data, idxs.data(), idxs.size()*sizeof(int64_t));

    if (std::getenv("LLAMA_K_IDXS_DUMP")) {
        const int64_t n_total = ggml_nelements(dst);
        fprintf(stderr, "*** K_IDXS_DUMP: n_tokens=%u sinfo.size=%u n_stream=%u n_total=%lld first4=[%lld,%lld,%lld,%lld] last4=[%lld,%lld,%lld,%lld]\n",
            ubatch->n_tokens, (uint32_t)sinfo.size(), (uint32_t)sinfo.n_stream(), (long long)n_total,
            (long long)data[0], (long long)data[std::min<int64_t>(1,n_total-1)], (long long)data[std::min<int64_t>(2,n_total-1)], (long long)data[std::min<int64_t>(3,n_total-1)],
            (long long)data[std::max<int64_t>(0,n_total-4)], (long long)data[std::max<int64_t>(0,n_total-3)], (long long)data[std::max<int64_t>(0,n_total-2)], (long long)data[n_total-1]);
        fflush(stderr);
    }

}

void llama_kv_cache::fill_input_v_idxs(std::vector<int64_t> & dst, int64_t n_total, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    // Pad unused V index entries with safe target (same approach as K)
    const int64_t safe_idx = !v_trans
        ? (int64_t)get_size() - 1
        : (int64_t)get_size() * (int64_t)hparams.n_embd_v_gqa_max() * (int64_t)std::max<uint32_t>(1, sinfo.n_stream()) - 1;
    dst.assign((size_t) n_total, std::max<int64_t>(0, safe_idx));

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                const int64_t j = s*sinfo.size() + i;
                if (j < n_total) {
                    dst[(size_t) j] = offs + sinfo.idxs[s][i];
                }
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    const int64_t o = s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j;
                    if (o < n_total) {
                        dst[(size_t) o] = offs + j*kv_size + sinfo.idxs[s][i];
                    }
                }
            }
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    // 2026-07-02 (DSA-on multi-ubatch): dead (unreferenced) input -> buffer==NULL. Skip.
    // See set_input_k_idxs note.
    if (!dst->buffer) {
        return;
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    std::vector<int64_t> idxs;
    fill_input_v_idxs(idxs, ggml_nelements(dst), ubatch, sinfo);
    memcpy(data, idxs.data(), idxs.size()*sizeof(int64_t));
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;

    // WIR soft mask parameters (0 = disabled)
    float soft_mask_alpha;
    float soft_mask_tau;
};

template<bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        // WIR soft mask: instead of hard -inf for future tokens,
                        // apply a spatial decay that lets nearby future tokens
                        // contribute attenuated attention weight.
                        if (args.soft_mask_alpha > 0.0f) {
                            float dist = (float)(p0 - p1);
                            float decay = args.soft_mask_tau > 0.0f
                                ? std::exp(-dist / args.soft_mask_tau)
                                : 1.0f;
                            float alpha_eff = args.soft_mask_alpha * decay;
                            // Lerp between -inf and 0: bias = -inf * (1 - alpha_eff)
                            // Use a large negative value instead of actual -inf
                            // to avoid NaN in softmax.
                            data[idst + j] = -1e9f * (1.0f - alpha_eff);
                            continue;
                        }
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = -std::abs(p0 - p1);
                } else {
                    data[idst + j] = 0.0f;
                }

                continue;
skip:
                data[idst + j] = -INFINITY;
            }
        }
    }
}

template<bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<causal, swa, is_2d, false>(args, data);
    }
}

template<bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<causal, swa, false>(args, data);
    }
}

template<bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<causal, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn,
                                        float soft_mask_alpha, float soft_mask_tau) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    // 2026-07-02 (DSA-on multi-ubatch NULL-buffer staging crash, ROOT CAUSE):
    // The GLM-5.2 DSA-on sparse path builds its own position-derived causal mask
    // (deepseek2.cpp [DSA_POS_CAUSAL_MASK]) and discards the gathered dense mask.
    // With fa=off, self_kq_mask_cnv == self_kq_mask, so the dense self_kq_mask
    // input leaf loses its only live consumer and the scheduler leaves it
    // unallocated (buffer==NULL). The per-layer set_inputs() in the layer-major
    // staging path then reached here with dst->buffer==NULL and crashed inside
    // ggml_backend_buffer_is_host (ggml-backend.cpp:198). A dead input has nothing
    // to feed; skip it, mirroring llm_graph_input_out_ids::set_input. The sparse
    // attention still gets its (live, correct) pos-causal mask.
    if (!dst->buffer) {
        return;
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    float * data = (float *) dst->data;

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
        /*.soft_mask_alpha  =*/ soft_mask_alpha,
        /*.soft_mask_tau    =*/ soft_mask_tau,
    };

    if (causal_attn) {
        set_input_kq_mask_impl<true> (args, data);
    } else {
        set_input_kq_mask_impl<false>(args, data);
    }

    // Phase 2.2 angle 21 (2026-05-12): trace the freshly written CPU mask
    // values to compare against the GPU-side FA mask sampled by MASK_TRACE.
    // If this shows correct causal triangular mask but the GPU FA mask is
    // wrong, the bug is in the CPU→GPU CPY+CAST pipeline. If THIS also
    // shows wrong values, the bug is in set_input_kq_mask_impl itself.
    {
        const char * dbg_cpu = std::getenv("LLAMA_PHASE1V7_MASK_TRACE");
        if (dbg_cpu && dbg_cpu[0] != '\0' && dbg_cpu[0] != '0' && ubatch->pos && ubatch->n_tokens > 0) {
            const float * row0 = data;
            const int64_t pos0 = ubatch->pos[0];
            // valid range for causal: row0 j ∈ [0..pos0], invalid: [pos0+1..n_kv-1]
            auto S = [&](int64_t c) -> double {
                if (c < 0 || c >= n_kv) return NAN;
                return (double) row0[c];
            };
            fprintf(stderr,
                "*** MASK_CPU_TRACE: pos0=%d n_tokens=%u n_kv=%lld n_stream=%lld "
                "c[0,0]=%g c[0,1]=%g c[0,pos0]=%g c[0,pos0+1]=%g c[0,n_kv/2]=%g c[0,n_kv-1]=%g\n",
                (int) pos0, (unsigned) n_tokens, (long long) n_kv, (long long) n_stream,
                S(0), S(1), S(pos0), S(pos0 + 1), S(n_kv / 2), S(n_kv - 1));
            fflush(stderr);
        }
    }

    // LLAMA_FA_KTRACE (2026-06-01 broadened): fire for EVERY batch (was gated to
    // pos0∈{0,512,4096} which never matched the image batch at pos0=19). Reports whether
    // the PADDING region [actual_kv, n_kv) — the 132 stale cells the image's 380->512
    // GGML_PAD creates — is correctly masked (-inf) or leaks finite (attended) values into
    // the image attention. Samples row 0 AND the LAST query row (pos n_tokens-1, where the
    // padding cells are at the smallest position-gap from the query => most likely exposed).
    const char * debug_mask = std::getenv("LLAMA_FA_KTRACE");
    if (debug_mask && debug_mask[0] != '\0' && debug_mask[0] != '0' &&
            ubatch->n_tokens > 0 && ubatch->pos) {
        int64_t actual_kv = 0;
        const auto & cells0 = v_cells[0];
        while (actual_kv < (int64_t) cells0.size() && !cells0.is_empty((uint32_t) actual_kv)) {
            ++actual_kv;
        }
        auto scan_row = [&](int64_t row, int64_t & tail_finite, int64_t & tail_zero, int64_t & first_unmasked) {
            tail_finite = 0; tail_zero = 0; first_unmasked = -1;
            if (row < 0 || row >= (int64_t) n_tps) return;
            for (int64_t j = actual_kv; j < n_kv; ++j) {
                const float v = data[row * n_kv + j];
                if (std::isfinite(v)) {
                    ++tail_finite;
                    if (v == 0.0f) ++tail_zero;
                    if (first_unmasked < 0) first_unmasked = j;
                }
            }
        };
        int64_t tf0 = 0, tz0 = 0, fu0 = -1, tfL = 0, tzL = 0, fuL = -1;
        scan_row(0, tf0, tz0, fu0);
        const int64_t last = (int64_t) n_tps - 1;
        scan_row(last, tfL, tzL, fuL);
        fprintf(stderr,
                "*** MASKTRACE: pos0=%d n_tokens=%u mask_n_kv=%lld actual_kv=%lld | "
                "row0 tail_finite=%lld tail_zero=%lld first_unmasked=%lld | "
                "rowL(%lld) tail_finite=%lld tail_zero=%lld first_unmasked=%lld\n",
                ubatch->pos[0], ubatch->n_tokens,
                (long long) n_kv, (long long) actual_kv,
                (long long) tf0, (long long) tz0, (long long) fu0,
                (long long) last, (long long) tfL, (long long) tzL, (long long) fuL);
        fflush(stderr);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

bool llama_kv_cache::fill_input_kq_mask_f16(std::vector<uint16_t> & dst, int64_t n_kv, const llama_ubatch * ubatch,
                                            bool causal_attn, float soft_mask_alpha, float soft_mask_tau) const {
    if (ubatch == nullptr || ubatch->n_tokens == 0 || n_kv <= 0 || n_stream == 0) {
        return false;
    }

    const uint32_t n_tokens = ubatch->n_tokens;
    if (n_tokens % n_stream != 0) {
        return false;
    }

    const int64_t n_tps = n_tokens / n_stream;

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
        /*.soft_mask_alpha  =*/ soft_mask_alpha,
        /*.soft_mask_tau    =*/ soft_mask_tau,
    };

    static thread_local std::vector<float> f32;
    f32.assign((size_t) n_kv * (size_t) n_tokens, -INFINITY);

    if (causal_attn) {
        set_input_kq_mask_impl<true> (args, f32.data());
    } else {
        set_input_kq_mask_impl<false>(args, f32.data());
    }

    dst.resize(f32.size());
    for (size_t i = 0; i < f32.size(); ++i) {
        dst[i] = (uint16_t) ggml_fp32_to_fp16(f32[i]);
    }

    return true;
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot;
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;

    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    const auto & n_embd_head_k = hparams.n_embd_head_k;
  //const auto & n_embd_head_v = hparams.n_embd_head_v;

    const auto & n_rot = hparams.n_rot;

    const auto n_embd_nope = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, rope_factors, freq_base_l, freq_scale_l);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    // Phase G diag (2026-05-06): trace state_write entry on streaming-prefill
    // MLA models to find the exact crash site.  No-op overhead unless env set.
    const char * dbg = std::getenv("LLAMA_HANDOFF_DEBUG");
    const bool trace = dbg && dbg[0] != '\0' && dbg[0] != '0';
    if (trace) {
        fprintf(stderr, "*** STATE_WRITE: enter seq_id=%d n_stream=%u v_cells.size()=%zu layers=%zu\n",
            (int) seq_id, n_stream, v_cells.size(), layers.size());
        fflush(stderr);
    }

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        if (trace) {
            fprintf(stderr, "*** STATE_WRITE: stream s=%u of %u\n", s, n_stream); fflush(stderr);
        }
        const auto & cells = v_cells[s];
        if (trace) {
            fprintf(stderr, "*** STATE_WRITE: cells.size()=%u\n", cells.size()); fflush(stderr);
        }

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && (seq_id == -1 || cells.seq_has(i, seq_id))) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // Debug: log cell count
        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read_to(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        // A wildly out-of-range value (e.g. 0x5241564B = ASCII "KVAR") means the blob isn't a raw
        // state_seq stream at all (wrong envelope fed to the reader), not a stream-count config gap.
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "n_stream mismatch (blob=%u 0x%08x vs cache=%u — a huge value means a non-state_seq "
                 "envelope, e.g. a KVARTIF1 artifact, was fed to state_read)",
                 n_stream_cur, n_stream_cur, n_stream);
        throw std::runtime_error(msg);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read_to(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);
        res = res && state_read_data(io, strm, cell_count, sinfo);

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];
    const bool write_cell_ext = hparams.n_pos_per_embd() > 1;

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            // TODO: we also need to save llama_kv_cell_ext when apply_ubatch() support loading it
            //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }

            if (write_cell_ext) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();
    const uint32_t state_ext_bit       = 0x80000000u;
    const uint32_t state_ext_has_k_idx = 0x00000001u;
    bool has_k_idx_state = false;
    for (const auto & layer : layers) {
        if (layer.k_idx && !layer.k_idx_stream.empty()) {
            has_k_idx_state = true;
            break;
        }
    }
    uint32_t n_layer_word = n_layer;
    uint32_t state_ext_flags = 0;
    if (has_k_idx_state) {
        n_layer_word |= state_ext_bit;
        state_ext_flags |= state_ext_has_k_idx;
    }

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer_word, sizeof(n_layer_word));
    if (state_ext_flags) {
        io.write(&state_ext_flags, sizeof(state_ext_flags));
    }

    // [KV_EXPORT_PIPELINE, stream-local-decode-perf.md §7.1(a)] The K/V
    // download (ggml_backend_tensor_get, GPU->host) and the CPU-side
    // serialization (copying the downloaded bytes into the io sink) used to
    // run strictly sequentially, layer by layer: nothing else happens while
    // one layer's D2H copy is in flight. On a multi-turn save of the
    // ~1.1 GB conversation state this dominates per-turn wall time.
    //
    // The pipelined path below double-buffers at layer granularity: a
    // worker thread downloads layer K+1 into the OTHER staging slot while
    // this thread serializes the already-downloaded layer K (copies staging
    // -> io). This overlaps GPU D2H wait time with CPU copy time. Staging
    // is bounded to 2 slots (never more than 2 layers of host memory
    // resident at once). A true ggml-backend async-get + event/stream
    // primitive exists in this fork (ggml_backend_tensor_get_async +
    // ggml_backend_event_*, ggml-backend.h) and the CUDA backend implements
    // get_tensor_async, but it is not usable here without much bigger,
    // unbuildable-blind surgery: (1) it needs a ggml_backend_t handle,
    // which llama_kv_cache::state_write_data() does not have — plumbing one
    // in means changing the llama_memory_i::state_write() virtual signature
    // across 6 overrides (llama_kv_cache, -iswa, -dsv4, -recurrent,
    // -hybrid, -hybrid-iswa) and every caller; (2) genuine D2H overlap also
    // needs a PINNED host destination — cudaMemcpyAsync to/from pageable
    // host memory is synchronous w.r.t. the host regardless of the stream
    // used (confirmed by reading ggml_backend_cuda_buffer_get_tensor()
    // itself, ggml-cuda.cu: it already issues cudaMemcpyAsync and only
    // conditionally syncs). The output buffers here (llama_io_write_buffer
    // / llama_io_write_file) are ordinary pageable heap memory, so without
    // new pinned staging the "async" call would just add stream/event
    // bookkeeping for no measurable overlap. The worker-thread approach
    // below gets the same overlap using only the existing cross-backend
    // synchronous ggml_backend_tensor_get() (already safe to call from any
    // host thread — it does its own device-select internally) plus
    // std::thread, with zero new backend plumbing and zero new pinned
    // allocations. It additionally coalesces each layer's (possibly
    // multi-range) reads into one staging buffer and one io.write() call.
    //
    // Output bytes are byte-for-byte identical to the sequential path: the
    // same header fields in the same order, the same payload bytes in the
    // same cr.data range order, just staged through one buffer instead of
    // N separate write_tensor() calls.
    //
    // LLAMA_KV_EXPORT_PIPELINE=0 forces the old strictly-sequential path
    // (kept verbatim below, byte-for-byte and behaviorally unchanged,
    // including its diagnostic prints) for A/B comparison / fallback.
    const char * pipeline_env = std::getenv("LLAMA_KV_EXPORT_PIPELINE");
    const bool pipeline_enabled = !(pipeline_env && pipeline_env[0] == '0' && pipeline_env[1] == '\0');
    const char * blob_scan_env = std::getenv("LLAMA_KV_BLOB_SCAN");
    const bool blob_scan_enabled =
        blob_scan_env && blob_scan_env[0] != '\0' && blob_scan_env[0] != '0';
    if (blob_scan_enabled) {
        fprintf(stderr,
                "*** KV_BLOB_SCAN_V1 ACTIVE: stream=%u layers=%u ranges=%zu pipeline=%d "
                "(scanning assembled payload bytes before io.write)\n",
                cr.strm, n_layer, cr.data.size(), (int) pipeline_enabled);
        fflush(stderr);
    }

    auto collect_ranges = [&](ggml_tensor * tensor, size_t row_bytes) {
        size_t total_bytes = 0;
        for (const auto & range : cr.data) {
            total_bytes += (range.second - range.first) * row_bytes;
        }
        std::vector<uint8_t> assembled(total_bytes);
        size_t offset = 0;
        for (const auto & range : cr.data) {
            const size_t range_bytes = (range.second - range.first) * row_bytes;
            ggml_backend_tensor_get(
                tensor, assembled.data() + offset, range.first * row_bytes, range_bytes);
            offset += range_bytes;
        }
        return assembled;
    };

    double copy_ms      = 0.0;
    double serialize_ms = 0.0;
    const auto t_wall_start = std::chrono::steady_clock::now();

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    if (pipeline_enabled && n_layer > 0) {
        struct kv_export_k_item {
            ggml_tensor * k;
            uint32_t      il;
            int64_t       row_elements;
            int32_t       k_type_i;
            uint64_t      k_size_row;
            size_t        total_bytes;
        };

        std::vector<kv_export_k_item> items;
        items.reserve(n_layer);
        size_t max_bytes = 0;
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;
            const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            auto * k = layer.k_stream[cr.strm];
            const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);

            size_t total = 0;
            for (const auto & range : cr.data) {
                total += (range.second - range.first) * k_size_row;
            }

            items.push_back({ k, il, n_embd_k_gqa, (int32_t) k->type, k_size_row, total });
            max_bytes = std::max(max_bytes, total);
        }

        // 2-slot staging ping-pong (hard bound: at most 2 layers resident).
        std::vector<uint8_t> slot[2];
        slot[0].resize(max_bytes);
        slot[1].resize(max_bytes);

        auto download = [&](uint32_t idx, uint8_t * dst) {
            const auto & it = items[idx];
            size_t off = 0;
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size   = range_size * it.k_size_row;
                ggml_backend_tensor_get(it.k, dst + off, range.first * it.k_size_row, buf_size);
                off += buf_size;
            }
        };

        std::thread worker;
        auto kick = [&](uint32_t idx) {
            worker = std::thread([&, idx] {
                const auto t0 = std::chrono::steady_clock::now();
                download(idx, slot[idx % 2].data());
                const auto t1 = std::chrono::steady_clock::now();
                copy_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            });
        };

        kick(0);
        for (uint32_t idx = 0; idx < (uint32_t) items.size(); ++idx) {
            worker.join(); // wait for idx's download to land in its slot
            const uint8_t * ptr = slot[idx % 2].data();
            if (idx + 1 < items.size()) {
                kick(idx + 1); // start next layer's download BEFORE serializing this one
            }

            const auto s0 = std::chrono::steady_clock::now();
            const auto & it = items[idx];
            if (blob_scan_enabled) {
                llama_kv_blob_scan_v1(
                    "K", it.il, cr.strm, (ggml_type) it.k_type_i, ptr,
                    it.total_bytes, (size_t) it.k_size_row, it.row_elements,
                    "cell-major");
            }
            io.write(&it.k_type_i,   sizeof(it.k_type_i));
            io.write(&it.k_size_row, sizeof(it.k_size_row));
            io.write(ptr, it.total_bytes);
            const auto s1 = std::chrono::steady_clock::now();
            serialize_ms += std::chrono::duration<double, std::milli>(s1 - s0).count();
        }
    } else {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

            auto * k = layer.k_stream[cr.strm];

            // Diagnostic: check K tensor data pointer for first 2 layers
            if (il < 2) {
                const void * kdata = k ? k->data : nullptr;
                const void * kbase = (k && k->view_src) ? k->view_src->data : kdata;
                uint32_t cksum = 0;
                // [2026-07-12] host-read only when the buffer is host-resident: with DSV4's GPU-resident
                // raw KV (REPOINT bypass), k->data is a CUDA device pointer — this raw loop was the
                // post-prefill 0xc0000005 (dump fault VA == cache_k_l0 GPU base). The real export writer
                // is safe (ggml_backend_tensor_get); only this diagnostic dereferenced directly.
                if (kdata && k->buffer && ggml_backend_buffer_is_host(k->buffer)) {
                    const uint8_t * p = (const uint8_t *)kdata;
                    for (int i = 0; i < 256 && i < (int)ggml_nbytes(k); i++) cksum = (cksum * 31) + p[i];
                }
                fprintf(stderr, "*** STATE_WRITE_K: il=%d k=%p k->data=%p view_src=%p base=%p buf=%p cksum=0x%08x\n",
                        il, (void*)k, kdata, k ? (void*)k->view_src : nullptr, kbase,
                        k ? (void*)k->buffer : nullptr, cksum);
                fflush(stderr);
            }
            // Write key type
            const int32_t k_type_i = (int32_t) k->type;
            io.write(&k_type_i, sizeof(k_type_i));

            // Write row size of key
            const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
            io.write(&k_size_row, sizeof(k_size_row));

            // Read each range of cells of k_size length and write out
            if (blob_scan_enabled) {
                auto assembled = collect_ranges(k, (size_t) k_size_row);
                llama_kv_blob_scan_v1(
                    "K", il, cr.strm, k->type, assembled.data(), assembled.size(),
                    (size_t) k_size_row, n_embd_k_gqa, "cell-major");
                io.write(assembled.data(), assembled.size());
            } else {
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t buf_size = range_size * k_size_row;
                    io.write_tensor(k, range.first * k_size_row, buf_size);
                }
            }
        }
    }

    if (has_k_idx_state) {
        for (const auto & layer : layers) {
            auto * k_idx = (!layer.k_idx_stream.empty() && cr.strm < layer.k_idx_stream.size()) ?
                layer.k_idx_stream[cr.strm] : nullptr;
            const uint32_t present = k_idx ? 1u : 0u;
            io.write(&present, sizeof(present));
            if (!present) {
                continue;
            }
            // DIAGNOSTIC (iter 29): print view + parent data pointers for first 3 layers
            if (layer.il < 3) {
                const void * view_data = k_idx ? k_idx->data : nullptr;
                const void * parent_data = layer.k_idx ? layer.k_idx->data : nullptr;
                const void * view_src_data = (k_idx && k_idx->view_src) ? k_idx->view_src->data : nullptr;
                const void * main_k_data = layer.k ? layer.k->data : nullptr;
                fprintf(stderr, "*** STATE_WRITE_K_IDX: il=%u k_idx_view=%p k_idx_parent=%p view_src=%p main_k=%p\n",
                        layer.il, view_data, parent_data, view_src_data, main_k_data);
                fflush(stderr);
            }

            const int32_t k_idx_type_i = (int32_t) k_idx->type;
            io.write(&k_idx_type_i, sizeof(k_idx_type_i));

            const uint64_t k_idx_size_row = ggml_row_size(k_idx->type, k_idx->ne[0]);
            io.write(&k_idx_size_row, sizeof(k_idx_size_row));

            if (blob_scan_enabled) {
                auto assembled = collect_ranges(k_idx, (size_t) k_idx_size_row);
                llama_kv_blob_scan_v1(
                    "K_IDX", layer.il, cr.strm, k_idx->type,
                    assembled.data(), assembled.size(), (size_t) k_idx_size_row,
                    k_idx->ne[0], "cell-major");
                io.write(assembled.data(), assembled.size());
            } else {
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t buf_size = range_size * k_idx_size_row;
                    io.write_tensor(k_idx, range.first * k_idx_size_row, buf_size);
                }
            }
        }
    }

    if (!v_trans) {
        if (pipeline_enabled) {
            struct kv_export_v_item {
                ggml_tensor * v;
                uint32_t      il;
                int64_t       row_elements;
                int32_t       v_type_i;
                uint64_t      v_size_row;
                size_t        total_bytes;
            };

            std::vector<kv_export_v_item> items;
            items.reserve(n_layer);
            size_t max_bytes = 0;
            for (const auto & layer : layers) {
                const uint32_t il = layer.il;
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

                auto * v = layer.v_stream[cr.strm];
                if (!v) {
                    continue;
                }

                const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);

                size_t total = 0;
                for (const auto & range : cr.data) {
                    total += (range.second - range.first) * v_size_row;
                }

                items.push_back({ v, il, n_embd_v_gqa, (int32_t) v->type, v_size_row, total });
                max_bytes = std::max(max_bytes, total);
            }

            if (!items.empty()) {
                // 2-slot staging ping-pong (hard bound: at most 2 layers resident).
                std::vector<uint8_t> slot[2];
                slot[0].resize(max_bytes);
                slot[1].resize(max_bytes);

                auto download = [&](uint32_t idx, uint8_t * dst) {
                    const auto & it = items[idx];
                    size_t off = 0;
                    for (const auto & range : cr.data) {
                        const size_t range_size = range.second - range.first;
                        const size_t buf_size   = range_size * it.v_size_row;
                        ggml_backend_tensor_get(it.v, dst + off, range.first * it.v_size_row, buf_size);
                        off += buf_size;
                    }
                };

                std::thread worker;
                auto kick = [&](uint32_t idx) {
                    worker = std::thread([&, idx] {
                        const auto t0 = std::chrono::steady_clock::now();
                        download(idx, slot[idx % 2].data());
                        const auto t1 = std::chrono::steady_clock::now();
                        copy_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                    });
                };

                kick(0);
                for (uint32_t idx = 0; idx < (uint32_t) items.size(); ++idx) {
                    worker.join(); // wait for idx's download to land in its slot
                    const uint8_t * ptr = slot[idx % 2].data();
                    if (idx + 1 < items.size()) {
                        kick(idx + 1); // start next layer's download BEFORE serializing this one
                    }

                    const auto s0 = std::chrono::steady_clock::now();
                    const auto & it = items[idx];
                    if (blob_scan_enabled) {
                        llama_kv_blob_scan_v1(
                            "V", it.il, cr.strm, (ggml_type) it.v_type_i, ptr,
                            it.total_bytes, (size_t) it.v_size_row, it.row_elements,
                            "cell-major");
                    }
                    io.write(&it.v_type_i,   sizeof(it.v_type_i));
                    io.write(&it.v_size_row, sizeof(it.v_size_row));
                    io.write(ptr, it.total_bytes);
                    const auto s1 = std::chrono::steady_clock::now();
                    serialize_ms += std::chrono::duration<double, std::milli>(s1 - s0).count();
                }
            }
        } else {
            for (const auto & layer : layers) {
                const uint32_t il = layer.il;

                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

                auto * v = layer.v_stream[cr.strm];
                if (!v) {
                    continue;
                }

                // Write value type
                const int32_t v_type_i = (int32_t) v->type;
                io.write(&v_type_i, sizeof(v_type_i));

                // Write row size of value
                const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
                io.write(&v_size_row, sizeof(v_size_row));

                // Read each range of cells of v_size length and write out
                if (blob_scan_enabled) {
                    auto assembled = collect_ranges(v, (size_t) v_size_row);
                    llama_kv_blob_scan_v1(
                        "V", il, cr.strm, v->type, assembled.data(), assembled.size(),
                        (size_t) v_size_row, n_embd_v_gqa, "cell-major");
                    io.write(assembled.data(), assembled.size());
                } else {
                    for (const auto & range : cr.data) {
                        const size_t range_size = range.second - range.first;
                        const size_t buf_size = range_size * v_size_row;
                        io.write_tensor(v, range.first * v_size_row, buf_size);
                    }
                }
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            if (blob_scan_enabled) {
                size_t selected_cells = 0;
                for (const auto & range : cr.data) {
                    selected_cells += range.second - range.first;
                }
                const size_t assembled_row_bytes = selected_cells * v_size_el;
                std::vector<uint8_t> assembled(
                    (size_t) n_embd_v_gqa * assembled_row_bytes);
                size_t dst_offset = 0;
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    for (const auto & range : cr.data) {
                        const size_t range_size = range.second - range.first;
                        const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                        const size_t buf_size = range_size * v_size_el;
                        ggml_backend_tensor_get(
                            v, assembled.data() + dst_offset, src_offset, buf_size);
                        dst_offset += buf_size;
                    }
                }
                llama_kv_blob_scan_v1(
                    "V", il, cr.strm, v->type, assembled.data(), assembled.size(),
                    assembled_row_bytes, (int64_t) selected_cells, "feature-major-transposed");
                io.write(assembled.data(), assembled.size());
            } else {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    // Read each range of cells of v_size_el length and write out
                    for (const auto & range : cr.data) {
                        const size_t range_size = range.second - range.first;
                        const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                        const size_t buf_size = range_size * v_size_el;
                        io.write_tensor(v, src_offset, buf_size);
                    }
                }
            }
        }
    }

    const auto t_wall_end = std::chrono::steady_clock::now();
    const double wall_ms  = std::chrono::duration<double, std::milli>(t_wall_end - t_wall_start).count();

    // Sequential mode fuses "download" and "serialize" into the same
    // ggml_backend_tensor_get() call (straight into the output buffer), so
    // there's nothing to separately attribute to serialize_ms — the whole
    // measured span is copy time. copy_ms/serialize_ms in pipelined mode
    // won't sum exactly to wall_ms either (the untouched k_idx pass, the
    // v_trans==1 pass, header writes, and thread-creation overhead all land
    // only in wall_ms) — that's expected; wall_ms is the number that
    // matters for the A/B comparison.
    if (!pipeline_enabled) {
        copy_ms      = wall_ms;
        serialize_ms = 0.0;
    }

    fprintf(stderr, "*** KV_EXPORT: %d layers, copy %.0f ms, serialize %.0f ms, wall %.0f ms (%s)\n",
            (int) n_layer, copy_ms, serialize_ms, wall_ms, pipeline_enabled ? "pipelined" : "sequential");
    fflush(stderr);
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];
    const bool read_cell_ext = hparams.n_pos_per_embd() > 1;

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);
        std::vector<llama_kv_cell_ext> cell_ext;
        if (read_cell_ext) {
            cell_ext.resize(cell_count);
        }

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;

            if (read_cell_ext) {
                io.read_to(&cell_ext[i], sizeof(cell_ext[i]));
            }
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
            if (read_cell_ext) {
                cells.ext_set(idx, cell_ext[i]);
            }
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);
            if (read_cell_ext) {
                llama_kv_cell_ext ext;
                io.read_to(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    // VIRTUAL KV (2026-09-10): the tensors have no buffer of their own; nothing can be restored into them.
    for (const auto & layer : layers) {
        if (layer.k && layer.k->data == nullptr) {
            LLAMA_LOG_ERROR("%s: the KV tensors are virtual (LLAMA_KV_TIER_VIRTUAL_KV); a state blob cannot be restored into them\n", __func__);
            return false;
        }
    }

    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer_word;

    io.read_to(&v_trans, sizeof(v_trans));
    io.read_to(&n_layer_word, sizeof(n_layer_word));

    const uint32_t state_ext_bit       = 0x80000000u;
    const uint32_t state_ext_has_k_idx = 0x00000001u;
    const bool has_state_ext = (n_layer_word & state_ext_bit) != 0;
    const uint32_t n_layer = n_layer_word & ~state_ext_bit;
    uint32_t state_ext_flags = 0;
    if (has_state_ext) {
        io.read_to(&state_ext_flags, sizeof(state_ext_flags));
    }
    const bool has_k_idx_state = (state_ext_flags & state_ext_has_k_idx) != 0;

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    fprintf(stderr, "*** STATE_READ_DATA: v_trans_blob=%u v_trans_cache=%d n_layer_blob=%u n_layer_cache=%zu cell_count=%u cells_size=%u\n",
            v_trans, (int)this->v_trans, n_layer, layers.size(), cell_count, (uint32_t)cells.size());
    fflush(stderr);
    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition (blob=%u cache=%d)\n", __func__, v_trans, (int)this->v_trans);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k_view = layer.k_stream[strm];
        auto * k_dst  = layer.k ? layer.k : k_view;
        const size_t k_stream_offset = layer.k ? strm * layer.k->nb[2] : 0;

        // Read type of key
        int32_t k_type_i_ref;
        io.read_to(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k_view->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read_to(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k_view->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Import into the base tensor so downstream attention views see
                // the updated bytes on every backend, including Metal.
                ggml_backend_tensor_set(
                        k_dst,
                        io.read(cell_count * k_size_row),
                        k_stream_offset + sinfo.head() * k_size_row,
                        cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                const void * src = io.read(cell_count * k_size_row);
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = k_stream_offset + sinfo.idxs[0][i] * k_size_row;
                    ggml_backend_tensor_set(k_dst, (const char*)src + i * k_size_row, dst_offset, k_size_row);
                }
            }
        }
    }

    if (has_k_idx_state) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            uint32_t present = 0;
            io.read_to(&present, sizeof(present));
            if (!present) {
                continue;
            }

            auto * k_idx_view = (!layer.k_idx_stream.empty() && strm < layer.k_idx_stream.size()) ?
                layer.k_idx_stream[strm] : nullptr;
            auto * k_idx_dst = layer.k_idx ? layer.k_idx : k_idx_view;
            if (!k_idx_view || !k_idx_dst) {
                LLAMA_LOG_ERROR("%s: blob has DSA k_idx data but layer %d has no k_idx cache\n", __func__, il);
                return false;
            }
            const size_t k_idx_stream_offset = layer.k_idx ? strm * layer.k_idx->nb[2] : 0;

            int32_t k_idx_type_i_ref;
            io.read_to(&k_idx_type_i_ref, sizeof(k_idx_type_i_ref));
            const int32_t k_idx_type_i = (int32_t) k_idx_view->type;
            if (k_idx_type_i != k_idx_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched DSA k_idx type (%d != %d, layer %d)\n", __func__, k_idx_type_i, k_idx_type_i_ref, il);
                return false;
            }

            uint64_t k_idx_size_row_ref;
            io.read_to(&k_idx_size_row_ref, sizeof(k_idx_size_row_ref));
            const size_t k_idx_size_row = ggml_row_size(k_idx_view->type, k_idx_view->ne[0]);
            if (k_idx_size_row != k_idx_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched DSA k_idx row size (%zu != %zu, layer %d)\n",
                        __func__, k_idx_size_row, (size_t) k_idx_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // DIAGNOSTIC (iter 29): when LLAMA_KV_DEBUG_DUMP_K_IDX=1, dump first 32 f16
                    // values of the just-imported k_idx data for layer 0. Checks if prefill's
                    // indexer-cache writes produced real data at early positions.
                    const void * src_bytes = io.read(cell_count * k_idx_size_row);
                    static const bool dump_k_idx = []() {
                        const char * e = std::getenv("LLAMA_KV_DEBUG_DUMP_K_IDX");
                        return e && e[0] != '\0' && e[0] != '0';
                    }();
                    if (dump_k_idx && (il == 0 || il == 1 || il == 40 || il == 77)) {
                        const uint16_t * p = (const uint16_t *) src_bytes;  // f16 raw
                        const size_t row_halfs = k_idx_size_row / 2;
                        const size_t total_halfs = cell_count * row_halfs;
                        // Dump position 0 (first row) and position cell_count/2 (middle row)
                        fprintf(stderr, "*** K_IDX_IMPORT layer=%u pos=0     f16=[", il);
                        for (int i = 0; i < 8 && (size_t)i < total_halfs; ++i) {
                            fprintf(stderr, " %04x", (unsigned) p[i]);
                        }
                        fprintf(stderr, " ]\n");
                        if (cell_count >= 2) {
                            const size_t mid_offset = (cell_count / 2) * row_halfs;
                            fprintf(stderr, "*** K_IDX_IMPORT layer=%u pos=%u  f16=[", il, (unsigned)(cell_count/2));
                            for (int i = 0; i < 8 && (mid_offset + i) < total_halfs; ++i) {
                                fprintf(stderr, " %04x", (unsigned) p[mid_offset + i]);
                            }
                            fprintf(stderr, " ]\n");
                        }
                        fflush(stderr);
                    }
                    ggml_backend_tensor_set(
                            k_idx_dst,
                            src_bytes,
                            k_idx_stream_offset + sinfo.head() * k_idx_size_row,
                            cell_count * k_idx_size_row);
                } else {
                    const void * src = io.read(cell_count * k_idx_size_row);
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = k_idx_stream_offset + sinfo.idxs[0][i] * k_idx_size_row;
                        ggml_backend_tensor_set(k_idx_dst, (const char*)src + i * k_idx_size_row, dst_offset, k_idx_size_row);
                    }
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v_view = layer.v_stream[strm];
            auto * v_dst  = layer.v ? layer.v : v_view;
            if (!v_view || !v_dst) {
                continue;
            }
            const size_t v_stream_offset = layer.v ? strm * layer.v->nb[2] : 0;

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v_view->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read_to(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v_view->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    ggml_backend_tensor_set(
                            v_dst,
                            io.read(cell_count * v_size_row),
                            v_stream_offset + sinfo.head() * v_size_row,
                            cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    const void * src = io.read(cell_count * v_size_row);
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = v_stream_offset + sinfo.idxs[0][i] * v_size_row;
                        ggml_backend_tensor_set(v_dst, (const char*)src + i * v_size_row, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v_view = layer.v_stream[strm];
            auto * v_dst  = layer.v ? layer.v : v_view;
            if (!v_view || !v_dst) {
                continue;
            }
            const size_t v_stream_offset = layer.v ? strm * layer.v->nb[2] : 0;

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v_view->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read_to(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v_view->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read_to(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Import into the base tensor with the stream stride baked
                    // into the destination offset.
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = v_stream_offset + (h + j * cells.size()) * v_size_el;
                        ggml_backend_tensor_set(v_dst, io.read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const void * src = io.read(cell_count * v_size_el);
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = v_stream_offset + (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            ggml_backend_tensor_set(v_dst, (const char*)src + i * v_size_el, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
    // Verify sinfos are valid after move
    for (size_t i = 0; i < this->sinfos.size(); i++) {
        if (this->sinfos[i].s0 > 100 || this->sinfos[i].s1 > 100) {
            fprintf(stderr, "*** KV_CTX_SINFO_CORRUPT: sinfos[%zu] s0=%u s1=%u (after move)\n",
                    i, this->sinfos[i].s0, this->sinfos[i].s1);
        }
    }
    // Compute per-ubatch n_kv values. Each ubatch should use the n_kv that
    // reflects only cells placed by ubatches 0..u. This reduces flash attention
    // numerical divergence from extra -inf tiles in layer-major mode.
    {
        const uint32_t pad = 256u;
        uint32_t max_cell_idx = 0;
        n_kv_per_ubatch.resize(this->ubatches.size());
        for (size_t u = 0; u < this->sinfos.size(); ++u) {
            for (uint32_t s = 0; s < this->sinfos[u].n_stream(); ++s) {
                for (const auto & idx : this->sinfos[u].idxs[s]) {
                    max_cell_idx = std::max(max_cell_idx, idx + 1);
                }
            }
            n_kv_per_ubatch[u] = std::min(kv->get_size(), std::max(pad, GGML_PAD(max_cell_idx, pad)));
        }
        // n_kv starts at the max (for graph build). seek() updates per-ubatch.
        n_kv = n_kv_per_ubatch.empty() ? 256 : n_kv_per_ubatch.back();
        n_kv_max = n_kv;
    }
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_k_idx(ggml_context * ctx, int32_t il) const {
    return kv->get_k_idx(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_k_idx_max(ggml_context * ctx, int32_t il) const {
    // Return a view sized to the FULL allocated cache (kv->get_size() = context size).
    // This is a TRUE constant — independent of n_kv (per-ubatch), n_kv_max
    // (per-batch, may be 0 during sched_reserve before apply()), or any other
    // dynamic state. Reservation and runtime always see the same shape.
    // Causal mask handles unfilled future positions, so attending against the
    // full allocated cache is safe (positions beyond actual fill have -inf score
    // and rank last in top_k).
    const uint32_t kv_size = kv->get_size();
    return kv->get_k_idx(ctx, il, kv_size, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k_idx(ggml_context * ctx, ggml_tensor * k_idx_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k_idx(ctx, k_idx_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

// DSV4 / DeepSeek lightning-indexer QK Hadamard rotation (upstream PR 21038).
ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::fill_input_k_idxs(std::vector<int64_t> & dst, int64_t n_total, const llama_ubatch * ubatch) const {
    kv->fill_input_k_idxs(dst, n_total, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::fill_input_v_idxs(std::vector<int64_t> & dst, int64_t n_total, const llama_ubatch * ubatch) const {
    kv->fill_input_v_idxs(dst, n_total, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn,
                                                float soft_mask_alpha, float soft_mask_tau) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn, soft_mask_alpha, soft_mask_tau);
}

bool llama_kv_cache_context::fill_input_kq_mask_f16(std::vector<uint16_t> & dst, int64_t n_kv, const llama_ubatch * ubatch,
                                                    bool causal_attn, float soft_mask_alpha, float soft_mask_tau) const {
    return kv->fill_input_kq_mask_f16(dst, n_kv, ubatch, causal_attn, soft_mask_alpha, soft_mask_tau);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::seek(size_t idx) {
    if (idx >= ubatches.size()) {
        fprintf(stderr, "*** SEEK_CRASH: idx=%zu ubatches.size()=%zu nkv_per_ub.size()=%zu\n",
                idx, ubatches.size(), n_kv_per_ubatch.size());
        fflush(stderr);
        GGML_ASSERT(false && "seek: idx >= ubatches.size()");
    }
    i_cur = idx;
    if (idx < n_kv_per_ubatch.size()) {
        n_kv = n_kv_per_ubatch[idx];
    }
}

bool llama_kv_cache::export_layout(uint32_t strm, std::vector<export_layer_layout> & out, bool & has_k_idx_state) const {
    out.clear();
    has_k_idx_state = false;
    if (strm >= n_stream) {
        return false;
    }
    for (const auto & layer : layers) {
        if (layer.k_idx && !layer.k_idx_stream.empty()) {
            has_k_idx_state = true;
            break;
        }
    }
    out.reserve(layers.size());
    for (const auto & layer : layers) {
        export_layer_layout L;
        L.il = layer.il;
        auto * k = strm < layer.k_stream.size() ? layer.k_stream[strm] : nullptr;
        if (!k) {
            return false;
        }
        L.k_type     = (int32_t) k->type;
        L.k_size_row = ggml_row_size(k->type, hparams.n_embd_k_gqa(layer.il));
        auto * k_idx = (!layer.k_idx_stream.empty() && strm < layer.k_idx_stream.size()) ? layer.k_idx_stream[strm] : nullptr;
        if (k_idx) {
            L.k_idx_present  = true;
            L.k_idx_type     = (int32_t) k_idx->type;
            L.k_idx_size_row = ggml_row_size(k_idx->type, k_idx->ne[0]);
        }
        auto * v = strm < layer.v_stream.size() ? layer.v_stream[strm] : nullptr;
        if (v) {
            L.v_present  = true;
            L.v_type     = (int32_t) v->type;
            L.v_size_row = ggml_row_size(v->type, hparams.n_embd_v_gqa(layer.il));
        }
        out.push_back(L);
    }
    return true;
}

bool llama_kv_cache::seq_cells_identity(llama_seq_id seq_id, uint32_t n, uint32_t & strm_out) const {
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return false;
    }
    const uint32_t s = seq_to_stream[seq_id];
    if (s >= v_cells.size()) {
        return false;
    }
    const auto & cells = v_cells[s];
    if (n > cells.size()) {
        return false;
    }
    for (uint32_t i = 0; i < cells.size(); ++i) {
        const bool has = !cells.is_empty(i) && cells.seq_has(i, seq_id);
        if (i < n) {
            if (!has || cells.pos_get(i) != (llama_pos) i) {
                return false;
            }
        } else if (has) {
            return false;
        }
    }
    strm_out = s;
    return true;
}

bool llama_kv_cache::reserve_prefix_cells(llama_seq_id seq_id, uint32_t n) {
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return false;
    }
    const uint32_t s = seq_to_stream[seq_id];
    if (s >= v_cells.size()) {
        return false;
    }
    auto & cells = v_cells[s];
    if (n > cells.size()) {
        return false;
    }
    uint32_t added = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (!cells.is_empty(i)) {
            if (cells.pos_get(i) != (llama_pos) i || !cells.seq_has(i, seq_id)) {
                LLAMA_LOG_WARN("%s: cell %u is used at pos %d by another seq; cannot reserve prefix of %u for seq %d\n",
                               __func__, i, (int) cells.pos_get(i), n, seq_id);
                return false;
            }
            continue;
        }
        cells.pos_set(i, (llama_pos) i);
        cells.seq_add(i, seq_id);
        added++;
    }
    if (v_heads[s] < n) {
        v_heads[s] = n;
    }
    LLAMA_LOG_INFO("%s: reserved %u prefix cells for seq %d (%u newly marked, head=%u)\n", __func__, n, seq_id, added, v_heads[s]);
    return true;
}
