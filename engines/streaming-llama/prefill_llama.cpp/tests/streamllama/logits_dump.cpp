// Dump the logits of a deterministic prompt (+ a few single-token decode steps) for fork-vs-upstream comparison.
// Uses only the public llama.h API so the same source builds against the fork and against upstream.
//
// usage: logits-dump <model.gguf> <out.bin> <n_prompt> <n_ubatch> [n_decode=8] [flash_attn=0] [seed=1234] [gpu_layers=0] [restore=0]
// out.bin: int32 n_rows, int32 n_vocab, then n_rows*n_vocab float32 (prompt rows, then one row per decode step)

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.gguf out.bin n_prompt n_ubatch [n_decode] [flash_attn] [seed] [gpu_layers] [restore]\n", argv[0]);
        return 1;
    }
    const char * path_model = argv[1];
    const char * path_out   = argv[2];
    const int n_prompt = atoi(argv[3]);
    const int n_ubatch = atoi(argv[4]);
    const int n_decode = argc > 5 ? atoi(argv[5]) : 8;
    const int fa       = argc > 6 ? atoi(argv[6]) : 0;
    const uint32_t seed = argc > 7 ? (uint32_t) atoi(argv[7]) : 1234u;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = argc > 8 ? atoi(argv[8]) : 0;
    llama_model * model = llama_model_load_from_file(path_model, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 512;
    cparams.n_batch         = n_prompt;
    cparams.n_ubatch        = n_ubatch;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = 2;
    cparams.n_threads_batch = 2;
    cparams.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.no_perf         = true;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // deterministic tokens (LCG)
    uint32_t s = seed;
    auto next_tok = [&]() {
        s = s * 1664525u + 1013904223u;
        return (llama_token) ((s >> 8) % (uint32_t) n_vocab);
    };

    std::vector<float> out;

    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        batch.token[i]     = next_tok();
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = n_prompt;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "prompt decode failed\n");
        return 1;
    }
    for (int i = 0; i < n_prompt; ++i) {
        const float * l = llama_get_logits_ith(ctx, i);
        out.insert(out.end(), l, l + n_vocab);
    }

    // Restore into cleared caches so missing indexer/recurrent owners cannot pass
    // merely because their old bytes survived in the same context.
    if (argc > 9 && atoi(argv[9])) {
        std::vector<uint8_t> state(llama_state_get_size(ctx));
        const size_t saved = llama_state_get_data(ctx, state.data(), state.size());
        if (!saved) return 2;
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_state_set_data(ctx, state.data(), saved) != saved) return 3;
    }

    for (int j = 0; j < n_decode; ++j) {
        batch.token[0]     = next_tok();
        batch.pos[0]       = n_prompt + j;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        batch.n_tokens     = 1;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode step %d failed\n", j);
            return 1;
        }
        const float * l = llama_get_logits_ith(ctx, 0);
        out.insert(out.end(), l, l + n_vocab);
    }

    FILE * f = fopen(path_out, "wb");
    const int32_t hdr[2] = { n_prompt + n_decode, n_vocab };
    fwrite(hdr, sizeof(int32_t), 2, f);
    fwrite(out.data(), sizeof(float), out.size(), f);
    fclose(f);

    fprintf(stderr, "wrote %d rows x %d logits to %s\n", hdr[0], hdr[1], path_out);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
