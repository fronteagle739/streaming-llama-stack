// Standalone logits-dump harness for the MiniMax-M3 sparse graph COHERENCE GATE, bypassing the
// tokenizer (the tiny synthetic GGUF has only a dummy vocab). Loads the model on CPU, decodes a
// FIXED token sequence, and prints a logits checksum (sum + argmax + max) of the last position.
//
// Coherence gate: run twice on tiny_m3_sparse.gguf with LLAMA_M3_SPARSE=0 vs =1. With a sequence
// shorter than topk*block_size (full block coverage), the sparse graph selects every block, so the
// logits MUST be identical -> validates the §14a sparse graph end-to-end WITHOUT real weights.
//
// Build (from prefill_llama.cpp, with vcvars + the build3 import libs):
//   cl /std:c++17 /EHsc /I include tests\m3_msa\m3_logits_test.cpp /Fe:build3\bin\m3_logits_test.exe ^
//      /link build3\src\llama.lib
// Run:  build3\bin\m3_logits_test.exe MODEL.gguf
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [n_tokens | (MTP) token_id token_pos]\n", argv[0]); return 2; }
    const int n_tok = argc >= 3 ? atoi(argv[2]) : 12;
    // LLAMA_M3_TEST_MTP: run the NextN/MTP DRAFT block via llama_mtp_draft_logits from a FIXED synthetic
    // prev-hidden (hidden[i]=(i%11-5)*0.1) + the embedding of argv[2]=token_id at argv[3]=token_pos. Cross-
    // checks the RTX MTP graph vs the MLX M3MTPLayer (scripts/m3_crosscheck_mtp_mlx.py). Use token_pos>0
    // (the deepseek2-style RTX prelude does not mask pos 0); for 1 token the attention is trivial so the
    // rope position is irrelevant.
    const bool mtp = std::getenv("LLAMA_M3_TEST_MTP") != nullptr;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // CPU: bypass the params_fit GPU memory-fitter
    if (mtp) mp.mtp_mode = LLAMA_MTP_MODE_DRAFT; // load the native NextN/MTP layers

    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "FAIL: load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 256;
    cp.n_batch = 256;
    cp.kv_unified = true; // n_stream=1 => mask [n_kv, n_tokens, 1, 1] (not per-token streamed), so the
                          // block_mask [n_kv, n_q] the M3 sparse graph builds broadcasts onto it cleanly
    // M3 runs FA off by default (SM120 VKQ; the non-FA path applies the PER-HEAD block mask + keeps
    // kq_mask f32). LLAMA_M3_TEST_FA=1 enables FA to check the shared-mask production path (FA on ->
    // f16 kq_mask; the graph casts it to f32 before the causal add).
    cp.flash_attn_type = std::getenv("LLAMA_M3_TEST_FA") ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                                         : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (mtp) cp.mtp_mode = LLAMA_MTP_MODE_DRAFT; // execute the native NextN/MTP draft graph
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "FAIL: create context\n"); return 1; }

    if (mtp) {
        const int n_embd    = llama_model_n_embd(model);
        const int token_id  = argc >= 3 ? atoi(argv[2]) : 1;
        const int token_pos = argc >= 4 ? atoi(argv[3]) : 5;
        std::vector<float> hidden(n_embd);
        for (int i = 0; i < n_embd; ++i) hidden[i] = (float) ((i % 11) - 5) * 0.1f;
        std::vector<float> out(n_vocab);
        int rc = llama_mtp_draft_logits(ctx, hidden.data(), token_id, token_pos, 0, out.data());
        if (rc < 0) { fprintf(stderr, "FAIL: mtp_draft_logits rc=%d\n", rc); return 1; }
        double sum = 0.0; int argmax = 0; float mxv = out[0];
        for (int i = 0; i < n_vocab; ++i) { sum += (double) out[i]; if (out[i] > mxv) { mxv = out[i]; argmax = i; } }
        printf("MTP_LOGITS token=%d pos=%d n_vocab=%d sum=%.6f argmax=%d max=%.6f l0=%.6f\n",
               token_id, token_pos, n_vocab, sum, argmax, mxv, out[0]);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    }

    // deterministic, valid token ids (< n_vocab)
    std::vector<llama_token> toks;
    for (int i = 0; i < n_tok; ++i) toks.push_back((llama_token)((i * 7 + 1) % n_vocab));

    // ONE sequence of n_tok tokens (causal attention across them) — NOT llama_batch_get_one, which
    // in this fork makes n_tok independent 1-token sequences (no cross-token attention -> nothing for
    // the block-sparse mask to prune). The MSA coherence gate needs real multi-token causal attention.
    //
    // Decode a [lo, hi) slice of toks at absolute positions; only the last token requests logits when
    // want_logits. Returns the logits-index used for llama_get_logits_ith, or -1 on failure.
    auto decode_range = [&](int lo, int hi, bool want_logits) -> int {
        llama_batch b = llama_batch_init(hi - lo, 0, 1);
        b.n_tokens = hi - lo;
        for (int i = lo; i < hi; ++i) {
            b.token[i - lo]     = toks[i];
            b.pos[i - lo]       = i;
            b.n_seq_id[i - lo]  = 1;
            b.seq_id[i - lo][0] = 0;
            b.logits[i - lo]    = (want_logits && i == hi - 1) ? 1 : 0;
        }
        int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        return rc == 0 ? (hi - lo - 1) : -1;
    };

    // LLAMA_M3_TEST_INCR: validate the DECODE path by computing the LAST token's logits via
    // prefill(0..n-2) + decode(1) instead of a one-shot prefill. Both must match (within tolerance) ->
    // proves the indexer-K cache store/view + decode scoring are consistent with prefill.
    const bool incr = std::getenv("LLAMA_M3_TEST_INCR") != nullptr;
    int li;
    if (incr && n_tok >= 2) {
        if (decode_range(0, n_tok - 1, false) < 0) { fprintf(stderr, "FAIL: prefill\n"); return 1; }
        li = decode_range(n_tok - 1, n_tok, true);   // decode the final token
    } else {
        li = decode_range(0, n_tok, true);           // one-shot
    }
    if (li < 0) { fprintf(stderr, "FAIL: decode\n"); return 1; }

    const float * logits = llama_get_logits_ith(ctx, li);
    if (!logits) { fprintf(stderr, "FAIL: no logits\n"); return 1; }

    double sum = 0.0; int argmax = 0; float mx = logits[0];
    for (int i = 0; i < n_vocab; ++i) {
        sum += (double) logits[i];
        if (logits[i] > mx) { mx = logits[i]; argmax = i; }
    }
    // checksum is what the coherence-gate comparison diffs
    printf("LOGITS_OK n_vocab=%d n_tok=%d sum=%.6f argmax=%d max=%.6f l0=%.6f\n",
           n_vocab, n_tok, sum, argmax, mx, logits[0]);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
