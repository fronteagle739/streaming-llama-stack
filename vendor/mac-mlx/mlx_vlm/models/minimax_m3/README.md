# MiniMax-M3 (text-only) — MLX decoder

Block-sparse (MSA) GQA + DeepSeek-V3 MoE, vendored in mlx-vlm so it reaches the Macs. Auto-dispatched by
`model_type: "minimax_m3"` (`mlx_vlm.utils.get_model_and_args`). Parity target = the RTX gguf engine
`prefill_llama.cpp/src/models/minimax-m3.cpp` (the two impls mirror each other).

## Architecture (the M3 deltas vs mlx-lm `minimax.py`, which this adapts)
- **Attention:** GQA + **per-head** QK-RMSNorm (over head_dim, after reshape) + **partial RoPE** (`rotary_dim = n_rot`).
- **MSA block-sparse attention** (`il >= first_k_dense_replace`): an index branch (`index_q/k_proj` →
  per-head RMSNorm → partial RoPE) drives a block selection — block-max-pool over `sparse_block_size`,
  top-`indexer_top_k` blocks, plus forced `sparse_init_blocks` (sink) + `sparse_local_blocks` (recent) —
  built into a **per-head additive mask** (each kv-group's selection expanded to its `gs` query heads,
  because soft-max broadcasts the head axis interleaved while GQA is contiguous). `num_index_heads ==
  num_key_value_heads`.
- **MoE:** sigmoid gate + `e_score_correction_bias` routing bias + top-k routed experts (SwitchGLU) +
  shared experts, all **swiglu_oai** (`alpha=1.702, limit=7`, GGML `ggml_swiglu_oai` parity). 3 leading
  **dense** layers (swiglu_oai MLP).
- **Norm:** gemma-style — `sanitize()` folds `w → w+1` on every `*.norm.weight` when loading (the HF
  checkpoint is raw; the RTX converter folds it).

## Caching / decode
`M3Cache` subclasses `KVCache` and also holds the per-token index-K (`update_idx`, grows along n_kv) so a
**decode** query selects blocks over the cached prior index-K. `_msa_general_mask` unifies prefill & decode
(L queries at `q_offset..` over `n_kv` keys) and supports **B≥1**. `Model.make_cache()` returns `M3Cache`.

## Validation (no weights — runs on a Mac, Metal-only)
```
cd <mlx-vlm clone>;  python -m mlx_vlm.models.minimax_m3.test_synthetic
```
Random-inits a tiny minimax_m3 and asserts (independent of the random weight VALUES):
- **coherence gate:** `M3_MLX_SPARSE=1` vs `0` — n_tok ≤ `topk*block_size` ⇒ sparse == dense; longer ⇒ diverges.
- **decode-consistency:** one-shot prefill-of-N == prefill-(N-1)+incremental-decode-1 (also checks the RoPE offset).
- **B>1:** each batch row's argmax == that sequence's B=1 result.

### Cross-platform cross-check vs the RTX gguf engine (the strongest parity test)
`scripts/m3_crosscheck_gen.py` (in the RTX repo) writes ONE tiny M3 in BOTH formats from the SAME weights:
`cc.gguf` (fed to `build3/bin/m3_logits_test`) + `cc_mlx.npz` (the same arrays under THESE param names) +
`cc_config.json`. `scripts/m3_crosscheck_mlx.py` runs the MLX side here on a Mac; compare the logits
checksums on the same fixed token ids. Result (2026-06-24): the two independent impls AGREE within ~1e-4 at
BOTH full coverage (n_tok=6: argmax 27, sum −0.5204≈−0.5202) AND partial coverage (n_tok=12: argmax 33,
sum −1.07644≈−1.07644). This cross-check FOUND + fixed a real RTX-side bug — the in-graph selection used to
add init/local forced blocks ON TOP of top-k (→ topk+#forced) instead of injecting them into the budget
(→ topk total, as this MLX impl and the numpy oracle do). Partial-coverage divergence is the regression
signal for that bug; keep it green. Re-run the full matrix (prefill full/partial + decode + large-weight
swiglu_oai clamp, 9 regimes) with one command: `bash scripts/m3_crosscheck_run.sh` (in the RTX repo; it
runs this MLX side over ssh to the Mac and asserts matching argmax). The full matrix now covers 14 regimes
(adds: decode, large-weight swiglu_oai clamp, init-force, and the MTP draft head below).

## MTP (NextN draft head) — DONE + cross-validated
`M3MTPLayer` (DeepSeek-V3 NextN: enorm/hnorm → `eh_proj(cat[next-embed, prev-hidden])` → one M3 decoder layer
→ final_layernorm → shared lm_head; position-0 embed masked) is implemented + structurally tested
(`test_mtp.py`) AND cross-validated vs the RTX `llama_mtp_draft_logits` (`scripts/m3_crosscheck_mtp_mlx.py`):
fed an identical fixed prev-hidden, the two agree (~1e-5) at pos=0 (argmax 63, exercises the masking) and
pos>0 (argmax 46). Building the RTX draft path fixed 4 latent gguf-mtp bugs (M3 is the first arch to run it):
NEXTN op-map → LAYER_REPEATING; main graph uses `effective_n_layers`; the MTP layer keeps a KV slot (no
`n_layer_kv_from_start`); `MINIMAX_M3` added to the mtp arch allowlists.

## `sanitize()` — DONE + structurally validated (no weights)
`sanitize()` now does the gemma `+1` norm fold AND the HF→MLX MoE remap: `block_sparse_moe.*` → `mlp.*` with
the per-expert routed weights (`experts.{x}.w1=gate / w2=down / w3=up`) STACKED into SwitchGLU
`switch_mlp.{gate,up,down}_proj` (`[n_expert, out, in]`); the MTP head (`model.mtp.layers.*`) is dropped
(loaded by the separate drafter). The HF key names + the `w1/w2/w3` convention were confirmed against the
authoritative vLLM loader (`vllm/models/minimax_m3/*/model.py:704/817-822/273/392-393/520-523`).
`python -m mlx_vlm.models.minimax_m3.test_sanitize` validates it with NO weights: reverse the sanitize on the
model's own param tree → synthesize an HF-named checkpoint → `sanitize()` → assert it reproduces the exact
param tree (keys+shapes). PASS. Non-disruptive to the cross-check (`cc_mlx.npz` uses MLX names, so no HF
pattern fires).

## Weight-gated TODO (do when the ~900GB M3 HF checkpoint is available — validate vs the reference)
FP8 dequant in `sanitize()` if the checkpoint is FP8; then load → coherent decode → cross-check vs the RTX
gguf logits. MTP acceptance-quality (the real-weight speculative speedup) + quant also remain weight-gated —
the MTP head + draft path + the HF→MLX remap/stacking themselves are done (above).
