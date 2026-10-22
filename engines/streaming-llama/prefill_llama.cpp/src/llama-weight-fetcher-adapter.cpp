#include "llama-weight-fetcher-adapter.h"
#include "llama-wf-bounded-policy.h"

#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>

#if defined(GGML_USE_CUDA) && __has_include(<cuda_runtime.h>)
#define WFA_HAS_CUDA 1
#include <cuda_runtime.h>
#endif

wf_context * wf_adapter_create(
    const std::vector<layer_tensor_info> & layer_infos,
    size_t                                  max_layer_bytes,
    const char *                            model_path,
    const char **                           split_paths,
    int                                     n_split_files,
    void *                                  gpu_backend,
    bool                                    trace,
    const std::unordered_map<std::string, wf_adapter_weight_loc> * weight_locs) {

    wf_params p = wf_params_default();
    p.model_path      = model_path;
    p.split_paths     = split_paths;
    p.n_split_files   = n_split_files;
    p.max_layer_bytes = max_layer_bytes;
    p.gpu_backend     = gpu_backend;
    p.n_layers        = (int)layer_infos.size();
    p.trace           = trace;

    // Optional transfer backend override for live A/Bs.  AUTO remains the
    // default so production keeps preferring DirectStorage on Windows.
    {
        const char * mode_env = std::getenv("LLAMA_WF_MODE");
        if (mode_env && mode_env[0] != '\0') {
            if (std::strcmp(mode_env, "auto") == 0) {
                p.mode = WF_MODE_AUTO;
            } else if (std::strcmp(mode_env, "dstorage") == 0 ||
                       std::strcmp(mode_env, "ds") == 0) {
                p.mode = WF_MODE_DSTORAGE;
            } else if (std::strcmp(mode_env, "direct_io") == 0 ||
                       std::strcmp(mode_env, "direct") == 0 ||
                       std::strcmp(mode_env, "ram") == 0) {
                p.mode = WF_MODE_DIRECT_IO;
            } else if (std::strcmp(mode_env, "mmap") == 0) {
                p.mode = WF_MODE_MMAP;
            } else {
                fprintf(stderr, "[weight-fetcher] ignoring unknown LLAMA_WF_MODE=%s\n", mode_env);
            }
            fprintf(stderr, "[weight-fetcher] transfer mode override: %s (env LLAMA_WF_MODE)\n", mode_env);
        }
    }

    {
        const char * direct_env = std::getenv("LLAMA_WF_IO_DIRECT");
        if (direct_env && direct_env[0] != '\0') {
            p.io_direct = !(direct_env[0] == '0' ||
                            std::strcmp(direct_env, "false") == 0 ||
                            std::strcmp(direct_env, "off") == 0);
            fprintf(stderr, "[weight-fetcher] IO direct: %s (env LLAMA_WF_IO_DIRECT)\n",
                    p.io_direct ? "on" : "off");
        }
    }

    {
        const char * threads_env = std::getenv("LLAMA_WF_IO_THREADS");
        if (threads_env && threads_env[0] != '\0') {
            int threads = std::atoi(threads_env);
            if (threads >= 1 && threads <= 64) {
                p.io_threads = threads;
                fprintf(stderr, "[weight-fetcher] IO threads override: %d (env LLAMA_WF_IO_THREADS)\n", threads);
            }
        }
    }

    {
        const char * qd_env = std::getenv("LLAMA_WF_IO_QD");
        if (qd_env && qd_env[0] != '\0') {
            int qd = std::atoi(qd_env);
            if (qd >= 1 && qd <= 64) {
                p.io_queue_depth = qd;
                fprintf(stderr, "[weight-fetcher] IO queue depth override: %d (env LLAMA_WF_IO_QD)\n", qd);
            }
        }
    }

    {
        const char * ram_env = std::getenv("LLAMA_WF_RAM_BUFS");
        if (ram_env && ram_env[0] != '\0') {
            int n = std::atoi(ram_env);
            if (n >= 1 && n <= 8) {
                p.n_ram_buffers = n;
                fprintf(stderr, "[weight-fetcher] RAM buffers override: %d (env LLAMA_WF_RAM_BUFS)\n", n);
            }
        }
    }

    // Auto-size DS block so that every layer fits in a single queue submission.
    // Larger blocks improve NVMe throughput (fewer per-request overheads).
    // Target ~1000 requests per wave (staging=3.5GB) for ~3.5MB per request.
    {
        const size_t auto_block = llama_wf_bounded::auto_io_block(max_layer_bytes);
        if (auto_block > p.io_block_size) {
            fprintf(stderr, "[weight-fetcher] auto-sized DS block: %zu KiB (layer=%.2f GB, ~%zu reqs/layer)\n",
                    auto_block / 1024, max_layer_bytes / (1024.0 * 1024.0 * 1024.0),
                    (max_layer_bytes + auto_block - 1) / auto_block);
            p.io_block_size = auto_block;
        }
    }

    // Optional manual DS block override for throughput tuning.
    {
        const char * block_mb_env = std::getenv("LLAMA_WF_IO_BLOCK_MB");
        if (block_mb_env && block_mb_env[0] != '\0') {
            int mb = std::atoi(block_mb_env);
            if (mb >= 1 && mb <= 64) {
                p.io_block_size = ((size_t) mb * 1024 * 1024 + 4095) & ~(size_t)4095;
                fprintf(stderr, "[weight-fetcher] IO block override: %d MB (env LLAMA_WF_IO_BLOCK_MB)\n", mb);
            }
        }
    }

    // Optional prefetch depth override.
    {
        const char * depth_env = std::getenv("LLAMA_WF_PREFETCH_DEPTH");
        if (depth_env && depth_env[0] != '\0') {
            int depth = std::atoi(depth_env);
            if (depth >= 1 && depth <= 8) {
                p.prefetch_depth = depth;
                fprintf(stderr, "[weight-fetcher] Prefetch depth override: %d (env LLAMA_WF_PREFETCH_DEPTH)\n", depth);
            }
        }
    }

    // Default to 3 GPU buffers when sub-layer splitting is active:
    // 1 computing, 1 GPU_READY (waiting), 1 DMA in-flight.
    // This eliminates compute stalls from waiting for DMA completion.
    {
        const char * splits_env = std::getenv("LLAMA_WF_LAYER_SPLITS");
        int n_sub = (splits_env && splits_env[0]) ? std::atoi(splits_env) : 1;
        if (n_sub > 1) {
            p.n_gpu_buffers = 3;
        }
    }

    // Allow overriding number of GPU buffers
    {
        const char * gpu_bufs_env = std::getenv("LLAMA_WF_GPU_BUFS");
        if (gpu_bufs_env && gpu_bufs_env[0] != '\0') {
            int n = std::atoi(gpu_bufs_env);
            if (n >= 1 && n <= 4) {
                p.n_gpu_buffers = n;
                fprintf(stderr, "[weight-fetcher] GPU buffers override: %d (env LLAMA_WF_GPU_BUFS)\n", n);
            }
        }
    }

    // MoE expert fetch mode: "all" (default) or "active" (selective streaming)
    // LLAMA_WF_MOE_MODE=active enables selective expert streaming.
    // Requires n_expert_splits >= 2 so the router runs as a separate sub-layer
    // before expert weights are fetched.
    {
        const char * moe_env = std::getenv("LLAMA_WF_MOE_MODE");
        if (moe_env && (std::strcmp(moe_env, "active") == 0 || std::strcmp(moe_env, "1") == 0)) {
            p.moe_mode = WF_MOE_FETCH_ACTIVE;
            fprintf(stderr, "[weight-fetcher] MoE mode: ACTIVE (selective expert streaming)\n");
        }
    }

    // Use MMAP mode when no model_path is provided (fall back to existing pointers)
    if (!model_path || model_path[0] == '\0') {
        p.mode = WF_MODE_MMAP;
    }

    wf_context * wf = wf_create(&p);
    if (!wf) return nullptr;

    // Register each layer's weight spans (skip empty sub-layers)
    for (int il = 0; il < (int)layer_infos.size(); il++) {
        auto & info = layer_infos[il];
        if (info.entries.empty() || info.total_bytes == 0) {
            // Empty sub-layer (dense layer's MoE slot) — register as empty
            wf_register_layer(wf, il, nullptr, 0, 0);
            continue;
        }
        std::vector<wf_weight_span> spans(info.entries.size());

        for (int si = 0; si < (int)info.entries.size(); si++) {
            auto & entry = info.entries[si];
            auto & span  = spans[si];

            span.tensor_name  = entry.tensor ? entry.tensor->name : nullptr;
            span.nbytes       = entry.nbytes;
            span.alloc_size   = entry.alloc_size;   // from prefill_compute_alloc_offsets
            span.alloc_offset = entry.alloc_offset; // ggml allocator offset (may differ from sequential)
            span.mmap_addr    = entry.orig_data;    // fallback: existing CPU pointer

            // Look up real file offset from the model's weight location map
            if (weight_locs && span.tensor_name) {
                auto it = weight_locs->find(span.tensor_name);
                if (it != weight_locs->end()) {
                    span.file_offset = it->second.file_offset + entry.slice_offset;
                    span.file_idx    = it->second.file_idx;
                } else {
                    // Tensor not found in weight map — fall back to mmap
                    span.file_offset = 0;
                    span.file_idx    = 0;
                }
            } else {
                span.file_offset = 0;
                span.file_idx    = 0;
            }

            // Detect MoE expert tensors from name pattern: "blk.N.ffn_*.N."
            span.is_expert  = false;
            span.expert_idx = -1;
            if (span.tensor_name) {
                // Pattern: "ffn_gate_exps", "ffn_down_exps", "ffn_up_exps"
                // or expert-specific: "ffn_gate_exp.N", etc.
                const char * exp_tag = strstr(span.tensor_name, "_exp");
                if (exp_tag) {
                    span.is_expert = true;
                    // Try to parse expert index from "exp.N" pattern
                    const char * dot = strchr(exp_tag + 4, '.');
                    if (dot) {
                        span.expert_idx = atoi(dot + 1);
                    }
                }
            }
        }

        wf_register_layer(wf, il, spans.data(), (int)spans.size(), info.total_bytes);
    }

    return wf;
}

void wf_adapter_remap_tensors(
    std::vector<tensor_remap_entry> & layer_entries,
    void *                            gpu_buf_ptr,
    void *                            gpu_buf_handle) {

    auto * gpu_buf = (ggml_backend_buffer_t)gpu_buf_handle;
    size_t offset = 0;

    // Ensure all GPU work is complete before modifying tensor pointers.
#ifdef WFA_HAS_CUDA
    cudaDeviceSynchronize();
#endif

    // Use pre-computed alloc_offset from prefill_compute_alloc_offsets.
    // These offsets mirror what ggml's GPU buffer allocator would use,
    // including backend-specific padding (MATRIX_ROW_PADDING for CUDA).

    for (auto & entry : layer_entries) {
        entry.tensor->buffer = gpu_buf;
        size_t tensor_offset = (entry.alloc_size > 0) ? entry.alloc_offset : offset;
        entry.tensor->data = (char *)gpu_buf_ptr + tensor_offset;

        ggml_backend_buffer_init_tensor(gpu_buf, entry.tensor);

        offset = tensor_offset + ((entry.alloc_size > 0) ? entry.alloc_size : entry.nbytes);
    }
}

void wf_adapter_restore_tensors(
    std::vector<tensor_remap_entry> & layer_entries) {

    for (auto & entry : layer_entries) {
        entry.tensor->buffer = entry.orig_buffer;
        entry.tensor->data   = entry.orig_data;
    }
}
