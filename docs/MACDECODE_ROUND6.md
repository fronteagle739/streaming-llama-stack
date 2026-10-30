# Mac decode Round6 upstream rebase

Core and Metal sources use upstream f48c5e2e20b1aaae1c79735a8457225d2469ae73.
Required native import/cone v2, sampler/bias/tools/vision, checkpoint and native
save, descriptor identity, wired allocation, DSpark and Studio integration remain.
The raw F32 native mode3 ring is converted to upstream stage tensors; BF16
rounding occurs at drafter input and mode2 export. Mode2 export now owns a Metal
command buffer for its staging copy. Its focused fixture enforces that contract.

The model and drafter are the owner's existing lossy Q4-to-MXFP4 speed proxies.
No model is downloaded. No published native-weight speed or accuracy is claimed.

## Retired tuning controls

The ROUND 6 DECISION authorizes retiring nonwinning pure experimental tuning
controls without upstream equivalents. Each retired name remains visible with
status `retired_round6`; setting its environment or debug value fails loudly.
Presets remove legacy tuning overrides and use upstream defaults. Markov phase
profiling is preserved as real measured telemetry, rather than retired as tuning.
All 120 legacy names and the retained adapters are in
[the control manifest](macdecode-r5/upstream-controls.json).

| Retired control | Rejected environment variable |
|---|---|
| narrow_weights | DS4_DS41_NARROW_WEIGHTS |
| engram_async | DS4_DS41_ENGRAM_ASYNC |
| round_fuse_norm | DS4_METAL_DISABLE_V41_ROUND_FUSE_NORM |
| round_fuse_hcsum | DS4_METAL_DISABLE_V41_ROUND_FUSE_HCSUM |
| round_fuse_hcexpand | DS4_METAL_DISABLE_V41_ROUND_FUSE_HCEXPAND |
| q4_wide | DS4_DS41_Q4_WIDE |
| q4_gu_nr1 | DS4_DS41_Q4_GU_NR1 |
| router_fused | DS4_DS41_ROUTER_FUSED |
| router_fused_w | DS4_DS41_ROUTER_FUSED_W |
| shared_swiglu | DS4_DS41_SHARED_SWIGLU |
| hc_norm_mix | DS4_DS41_HC_NORM_MIX |
| mtp_down_tail_pack | DS4_DS41_MTP_DOWN_TAIL_PACK |
| mtp_hc_epilogue | DS4_DS41_MTP_HC_EPILOGUE |
| down_tail_pack | DS4_DS41_DOWN_TAIL_PACK |
| router_shared_hc | DS4_DS41_ROUTER_SHARED_HC |
| q8_virtual_down | DS4_DS41_Q8_VIRTUAL_DOWN |
| concurrent_down | DS4_DS41_CONCURRENT_DOWN |
| attn_low_static | DS4_DS41_ATTN_LOW_STATIC |
| verify_mxfp4 | DS4_DS41_VERIFY_MXFP4 |
| concurrent_ffn | DS4_DS41_CONCURRENT_FFN |
| engram_lookup | DS4_DS41_ENGRAM_LOOKUP |
| verify_output | DS4_DS41_VERIFY_OUTPUT |
| engram_gate | DS4_DS41_ENGRAM_GATE |
| shared_down_hc | DS4_DS41_SHARED_DOWN_HC |
| fixed_dense | DS4_DS41_FIXED_DENSE |
| query_norm_pair | DS4_DS41_QUERY_NORM_PAIR |
| query_rope | DS4_DS41_QUERY_ROPE |
| heads_rope | DS4_DS41_HEADS_ROPE |
| engram_persistent | DS4_DS41_ENGRAM_PERSISTENT |
| q8_virtual_query | DS4_DS41_Q8_VIRTUAL_QUERY |
| kv_prepare | DS4_DS41_KV_PREPARE |
| kv_query_mix | DS4_DS41_KV_QUERY_MIX |
| attn_out_hc | DS4_DS41_ATTN_OUT_HC |
| ffn_producer | DS4_DS41_FFN_PRODUCER |
| qa_kv_flat | DS4_DS41_QA_KV_FLAT |
| hc_stream_layout | DS4_DS41_HC_STREAM_LAYOUT |
| hc_stream | DS4_DS41_HC_STREAM |
| hc_tail | DS4_DS41_HC_TAIL |
| hc_expand_fold | DS4_DS41_HC_EXPAND_FOLD |
| mv_round | DS4_DS41_MV_ROUND |
| producer_round | DS4_DS41_PRODUCER_ROUND |
| ffn_add_fold | DS4_DS41_FFN_ADD_FOLD |
| kv_stage_f16 | DS4_DS41_KV_STAGE_F16 |
| attn_cohort4 | DS4_DS41_ATTN_COHORT4 |
| gather_wide | DS4_DS41_GATHER_WIDE |
| index_topk_radix | DS4_DS41_INDEX_TOPK_RADIX |
| index_compact_score | DS4_DS41_INDEX_COMPACT_SCORE |
| mtp_q8_trio_mask | DS4_DS41_MTP_Q8_TRIO_MASK |
| markov_cache | DS4_DS41_MARKOV_CACHE |
| router_simd | DS4_DS41_ROUTER_SIMD |
| router_hier | DS4_DS41_ROUTER_HIER |
| mtp_down_wide | DS4_DS41_MTP_DOWN_WIDE |
| mtp_gu_sort_simd | DS4_DS41_MTP_GU_SORT_SIMD |
| mtp_swiglu_round | DS4_DS41_MTP_SWIGLU_ROUND |
| mtp_q8_round | DS4_DS41_MTP_Q8_ROUND |
| mtp_f32_rows | DS4_DS41_MTP_F32_ROWS |
| mtp_pointwise_batch | DS4_DS41_MTP_POINTWISE_BATCH |
| mtp_f16_trio | DS4_DS41_MTP_F16_TRIO |
| mtp_gu_expert_sort | DS4_DS41_MTP_GU_EXPERT_SORT |
| mtp_q8_swizzle | DS4_DS41_MTP_Q8_SWIZZLE |
| mtp_gu_swizzle | DS4_DS41_MTP_GU_SWIZZLE |
| index_score_stream | DS4_DS41_INDEX_SCORE_STREAM |
| hc_join_static | DS4_DS41_HC_JOIN_STATIC |
| topk_short | DS4_DS41_TOPK_SHORT |
| index_candidates | DS4_DS41_INDEX_CANDIDATES |
| index_schedule | DS4_DS41_INDEX_SCHEDULE |
| index_mask_score | DS4_DS41_INDEX_MASK_SCORE |
| stage_gather | DS4_DS41_STAGE_GATHER |
| q4_dn_nr1 | DS4_DS41_Q4_DN_NR1 |
| hc_expand_nth | DS4_DS41_HC_EXPAND_NTH |
| routed_tail_cull | DS4_METAL_DISABLE_V41_ROUTED_TAIL_CULL |
| prefill_tail_absorb | DS4_DS41_PREFILL_TAIL_ABSORB |
| engram_readers | DS4_DS41_ENGRAM_READERS |
| prefill_attn_rb16 | DS4_METAL_DISABLE_V41_PREFILL_ATTN_RB16 |
| mtp_async_chunks | DS4_DS41_MTP_ASYNC_CHUNKS |
| mtp_engram_rows6 | DS4_DS41_MTP_ENGRAM_ROWS6 |
| mtp_capture_warmup | DS4_DS41_MTP_CAPTURE_WARMUP |
| mtp_state_fix | DS4_DS41_MTP_STATE_FIX |
| mtp_q8_stream6 | DS4_DS41_MTP_Q8_STREAM6 |
| mtp_q8_stream6_mask | DS4_DS41_MTP_Q8_STREAM6_MASK |
| mtp_q8_pair6 | DS4_DS41_MTP_Q8_PAIR6 |
| prefill_f16_rows2 | DS4_DS41_PREFILL_F16_ROWS2 |
| prefill_embed_init | DS4_DS41_PREFILL_EMBED_INIT |
| prefill_hc_sum_round | DS4_DS41_PREFILL_HC_SUM_ROUND |
| prefill_hc_expand_round | DS4_DS41_PREFILL_HC_EXPAND_ROUND |
| prefill_hc_norm_round | DS4_DS41_PREFILL_HC_NORM_ROUND |
| prefill_ffn_add_round | DS4_DS41_PREFILL_FFN_ADD_ROUND |
| attn_diag | DS4_DS41_ATTN_DIAG |
| prefill_attn_lean_rows | DS4_METAL_DISABLE_V41_PREFILL_ATTN_LEAN_ROWS |
| prefill_engram_xsweep | DS4_DS41_PREFILL_ENGRAM_XSWEEP |
| attn_geom | DS4_DS41_ATTN_GEOM |
| dspark_capture | DS4_DS41_DSPARK_CAPTURE |
| verify_wide_prefill | DS4_DS41_VERIFY_WIDE_PREFILL |
| dspark_loss_budget | DS4_DS41_DSPARK_LOSS_BUDGET |
| dspark_adaptive | DS4_DS41_DSPARK_ADAPTIVE |
| dspark_serve | DS4_DS41_DSPARK_SERVE |
| dspark_reasoning_serial | DS4_DS41_DSPARK_REASONING_SERIAL |
| verify_batch_core | DS4_DS41_VERIFY_BATCH_CORE |

## Qualification

Three complete-vocabulary exact proofs PASS: eight serial continuation rows
at 8K and 131K prose, and three actual DSpark verifier frames at 8K code.
The code proof generates 32 tokens to cross upstream's unchanged first-16-serial
admission policy; rejected draft rows and the pre-cycle frontier are included.
All captured float words and frame metadata match upstream bit for bit.

One normal 128-token timing per workload/arm, same lossy weights, allocated
context 393216, drafter and vision loaded, restored RAM prefix, greedy decode:

| Workload | Upstream tok/s | Ours tok/s | Difference |
|---|---:|---:|---:|
| prose8k | 39.1842 | 39.0665 | -0.300% |
| prose131k | 37.7922 | 37.8919 | +0.264% |
| code8k | 52.3186 | 52.6538 | +0.641% |

The prose measurements/proofs use e3efa445; code uses ac55f925 after the sole
HTTP diagnostic capture-bound validation changed from 8 to 32. Engine, Metal,
sampler, native and arithmetic sources are identical across those two builds.
Initial failed snapshot/capture requests are recorded separately and are not
qualification. The 101-test CPU run passes with nine existing platform skips;
no source checks are deselected. Native representation and executor fixtures
pass under explicit 2 GiB/16 GiB guards. Model runs use 400 GiB/16 GiB guards,
with no guard violation or swap. Source and exact reports are in
[the Round6 evidence directory](macdecode-r6/summary.json).

The pinned optimized packed-FP4 producer from experimental 44928ad and the
canonical receiver cdeb1d1d pass the eight-row full-vocabulary KL gate (mean
0.0238444, max 0.0530931 nats; limits 0.1/0.5). A fresh 199952-token release
prompt recalls the planted password. Primary FP4-to-FP4 results:

| Configuration | End-to-end prefill tok/s | Decode tok/s |
|---|---:|---:|
| Original 5 GiB graph floor | 1905.762325 | 36.305599 |
| 3.5 GiB graph floor, one requested followup | 1962.500875 | 35.146325 |

The graph-floor option moves 1.5 GiB within the existing device budget to the
exact hidden bank. Its graph allocator still refuses any measured graph that
does not fit; host limits, 1M context and DSpark capture remain. RTX prompt work
fell from 79.215 to 75.354 seconds. This single trial does not establish a stable
percentage gain. The ULTRA cone example uses it and the current automatic fold
policy; other profiles keep the 5 GiB floor unless explicitly selected.

The followup's engine initialization plus prompt was 81.812 s; export preparation
4.360 s; transfer/import/hash/ack 6.366 s; rewind save 5.015 s; other request work
4.333 s, totaling 101.886 s. Transfer and Mac import share one streamed interval;
they are not independently timed. RTX prompt work itself was 75.354 s, rather
than the target 38 s. Both runs use the same optimized producer and native FFN
bridge, packed FP4, exact cone2541/v2 and single-stream policy, with RAM tier and
CPU lane off. The 1M native path has fewer resident hidden rows than the 200K
prefill-only control. Two sweeps remain: a full 200K single sweep needs 13.39 GiB
of host workspace before the physical floor; current free RAM is about 12.8 GiB.
The reservation improvement does not close the remaining native-path gap.

DSpark is enabled at 200K. Its unchanged upstream confidence/controller policy
accepted one five-token block; 122 cycles had no draft. The measured prose
continuation therefore remains mostly serial. No forced confidence, sampler or
quality reduction is used. **The 5240 prefill / 40-60 decode release targets are
not met**, despite Mac-only upstream parity and the passing import/recall gate.

Studio adapter 483405fcfb is merged to the Studio integration branch and deployed to RTX,
Mac Studio and MacBook. Final GUI/restore/cleanup receipts are in the task RESULT.


## Capability reductions

The 98 explicitly listed tuning controls are retired by owner decision.
Obsolete preset-only gates DS4_V41_DSPARK_DECODE_READY and DS4_KV_TURN_FRONTIER
are removed from presets; their DSpark and checkpoint capabilities remain.
No user-facing sampler, bias, tools, vision, checkpoints, native save, cone,
descriptor identity, wired allocation, DSpark, KV/state import or Studio adapter
capability is authorized for removal.
