# Mac decode upstream ports, Round 4

The Round 4 execution paths are available without removing native
state import, cone, disk KV, vision, sampling, bias or tool support. Upstream
`v41-m3ultra` was fetched and still resolves to
`f48c5e2e20b1aaae1c79735a8457225d2469ae73`. Its
[published reference](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/f48c5e2e20b1aaae1c79735a8457225d2469ae73/README.md)
uses native MXFP4 weights; our existing Q4-to-MXFP4 target and drafter remain
explicitly lossy speed proxies. No model was downloaded.

## Execution paths and controls

| Port | Upstream source | Selection |
|---|---|---|
| Read-only model views without hazard tracking | da48ab1c | V4.1 default; `DS4_METAL_MODEL_TRACKED=1` retains tracking. Historical explicit `MODEL_UNTRACKED` semantics remain. |
| Aligned exact uint4 copies in the open compute encoder | 1df25c6f | Default for eligible disjoint ranges; `DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY=1` retains blits. Bounds, misalignment, concurrent and overlap cases retain their original behavior. |
| Literal V4.1 attention low projection, including the fork's BF16 store | e06a0b7f | `DS4_DS41_ATTN_LOW_STATIC=1` |
| Shared down beside routed gate/up, retaining the combined router/shared/HC producer | adaptation of 555e258f | `DS4_DS41_CONCURRENT_DOWN=1` |
| Full BF16 shared/routed two-stage FFN | 555e258f | `DS4_DS41_CONCURRENT_FFN=1`; uses the retained separate HC/router producers so both stages can overlap. The original producer remains selectable. |
| Six-to-eight-row MXFP4 verify kernels | 04830a71 | `DS4_DS41_VERIFY_MXFP4=1`; only actual DSpark verify blocks gain the new admission. Ordinary batches retain their old arithmetic. |
| Distinct MXFP4 expert planning and shared reads | a5635c52 | `DS4_DS41_MXFP4_DEDUP=1`; two to eight fused rows, original per-pair arithmetic. The 4 KiB plan uses the owned/wired allocator (16 KiB physical page on this Mac). |
| Persistent per-row Engram workers | bb7e488f/current reader | `DS4_DS41_ENGRAM_PERSISTENT=1`; 24 bounded workers on Apple, both uncached tables, no table cache. Existing batch, prefill and synchronous paths remain. |
| Held next-token layer 0 | 67dcb001 | `DS4_DS41_PREENCODE=1`; only commits after the real sampled token is published. Full lever image, graph generation/position and host latches must match; reset, restore, foreign commands and failures discard it. |
| Exact step RoPE table, index-query quantization and transforming attention store | b1ba3164 | `DS4_DS41_HEADS_ROPE=1`; eligible scalar split reductions retain their arithmetic and write BF16/inverse-RoPE heads directly. Index queries combine their existing RoPE/quantization passes. Existing producer-round, projection, batched and generic paths remain. |
| Literal QA/KV and attention/HC shapes | e06a0b7f | `DS4_DS41_FIXED_DENSE=1`; retains the fork's producer epilogues and general alternatives. |
| Exact Unicode JoyAI splits | 1d28e1ff | Corrects the old non-ASCII-as-letter approximation. Existing GLM tokenization is preserved. No model download or added-token policy change. |
| Exact Engram code/scale format table | 962ac191 | `DS4_DS41_ENGRAM_LOOKUP=0` retains the scalar reference; default on. The 128 KiB table caches format values, never model rows. Invalid and overflowing values still fail with EDOM. |
| Shared-down / HC / carry join | 0ba56e9c, 73f1bdb1, e06a0b7f | `DS4_DS41_SHARED_DOWN_HC=1`; after the routed sum, one short72 projection writes all observable shared/block/HC/pre values with the fork's ordered FMA epilogue. Existing overlap and producer controls remain. |
| Segmented Engram gate | dbe5d51d | `DS4_DS41_ENGRAM_GATE=1`; 256 threads retain the original serial accumulation order for each lane. Masks and other shapes retain their original path. |
| Exact Q8 row weight reuse | 7924b68a | `DS4_DS41_ROW_MATMULS=1`; eligible two-to-eight-row products reuse weight cache lines and retain each scalar reduction. Explicit pair/trio/stream schedules take priority. |
| DSpark Q8 matrix rows | f8e36424 | `DS4_DS41_ROWS_MMA=1`, with `MTP_Q8_PAIR6=0` and `MTP_Q8_STREAM6=0`; conflicting six-row schedules fail explicitly. Optional upstream reduction for three to eight DSpark rows can change last bits and speculative continuation. All ordinary rows and serial decode retain their reductions, and scope restores on failure. |

The thirteen new scheduling levers start at zero; the format lookup starts at one.
All are exposed by the existing debug
lever mechanism. Their environment variables are registered and documented.
Internal schedule selection computes the same model operations; it does not
remove user capabilities. Read/worker failures fail explicitly. Reader cleanup
joins workers before table descriptors close, row views do not inherit reader
ownership, and actual completion timestamps preserve the existing diagnostics.

## Verification so far

`tests/native_metal_copy.m` executes the real Metal implementation and checks:

- Compute versus blit dispatch selection, exact copy bytes, views, disjoint
  ranges, alignment, zero length, bounds, concurrent fallback and ordering.
- Model-view policy controls, including historical override precedence.
- Static versus generic low projection for both BF16 modes.
- Down-only overlap, consumer ordering and abort cleanup.
- MXFP4 rows 2–8 versus independent scalar rows; dedup gate/up/middle/output
  bits for shared and disjoint experts; ordinary-batch behavior.
- Full shared/routed overlap versus serial middle/down/routed words with both
  fixed-shape settings, including early error cleanup.
- Persistent Engram generations versus serial rows, real timestamps,
  EINVAL/EIO/EDOM handling and cleanup during in-flight work.
- Held-buffer commit with the real embedding token word, no premature execution,
  explicit/foreign drops, and host-write/flush error cleanup.
- Attention reduction plus BF16/inverse RoPE versus standalone kernels at
  positions 0, 8192, 131072 and 1048575, raw and compressed frequencies.
- Literal/general QA, KV, attention block and every HC stream word.
- Every one of the 65,536 Engram code/scale pairs, including signed zero,
  underflow, invalid/overflow EDOM and null EINVAL handling.
- Step-table index queries versus separate RoPE/quantization at the same four
  positions, both frequency choices and all three quantization formats.
- Shared-down/HC join versus the retained down/expansion/carry chain, including
  every observable output word and invalid-buffer refusal.
- Segmented Engram gate at one, six and sixteen rows, every word exact.
- Exact Q8 row reuse at two to eight rows, three input widths, with both store
  modes; DSpark matrix-row numerical bounds and dispatch witnesses, plus exact
  ordinary rows outside the DSpark scope.

The standalone `tests/native_deepseek_tokenizer.py` compiles the actual C
splitting fragment and intercepts only its BPE-piece emission. Against the
existing released tokenizer JSON it checks 22,181 texts and 8,721,151 pieces,
including every Unicode scalar and both benchmark files, with complete byte
coverage. Set `DS4_TEST_SCRATCH` and `DS4_TEST_TOKENIZER_JSON` explicitly and run
under a 2 GiB budget/16 GiB floor. No engine or GGUF is opened by that test.

A separate existing-JSON BPE audit confirms that the prose fixture changes:
the legacy 8K/131K token-prefix hashes are `3cd4d4bba0ab9986` /
`94835d300d960e5c`, while the released splits give `676274b21242e429` /
`52097139496b33b8`, matching the fresh upstream control. Code 8K remains
`77f0c7f9c5bd40c6`. New prose logits therefore use a new reference; they are
not presented as bit-identical to the old tokenizer's different prefix.

The synthetic model is in owned wired RAM, with no multi-GB scratch file.
Run this test alone under an outer 20 GiB host budget and 16 GiB physical floor;
its allocator budget is 12 GiB. Set `DS4_TEST_SCRATCH` to the existing task
scratch directory, plus `DS4_METAL_WIRED_CORE_KV=1`,
`DS4_NATIVE_SESSION_HOST_BYTES=12884901888` and
`DS4_NATIVE_SESSION_HOST_FLOOR_BYTES=17179869184`. Compile it against the same
engine core objects and `ds4_image.o`, with Foundation/Metal, at one stable
test executable path. No actual model is loaded by the component test.

The eight small-port full-model proofs (all four arms at 8K and 131K), plus
four full-FFN/reader 8K proofs, match all 128 × 129280 baseline float32 decision
logits. Logit-capture requests are excluded from throughput means; the
verify/full-FFN sweeps explicitly record their enabled command-buffer and
Engram diagnostics. Forty Python compatibility, pairing,
memory, build-identity and environment checks plus the native routing/disk-seed
fixture pass in isolated scratch data directories. An initial environment
check correctly rejected stale Mac source references; SHA synchronization fixed
the mismatch without weakening the test.

## Measurement scope

The inherited standalone harness omitted the explicit owned-wired flag. Its
earlier Round 4 sweeps are matched comparative diagnostics, not live Studio
preset qualification. The corrected harness always enables owned-wired core/KV
and records that policy. No production preset or service rollout has occurred
in this round yet.

At 393216 allocated context and 128 greedy continuation tokens, the comparative
baseline measured 34.273 serial prose 8K, 33.308 serial prose 131K and 43.688
DSpark code 8K. Copies/view policy plus static low projection measured
34.601 / 33.794 / 45.812. Persistent readers reached 35.117 serial prose 8K;
full FFN and down-only overlap were slower than the existing fused producer
on this fixture. These are not a 40–60 tok/s release claim.

The command-buffer diagnostic attributes about 26.44 ms/token to GPU spans
and 2.43 ms/token between spans on 8K prose. The dispatch Engram reader exposes
about 2.53 ms at layer 1. Shader work therefore remains relevant even if every
host wait is hidden. All measured jobs exited gracefully within their explicit
budgets/floors and observed no new swap I/O. Final owned-wired, Studio GUI,
FP4-to-FP4 release and fleet qualification remain required.

The later quiet owned/wired shared-down/HC plus complete-RoPE sweep reaches
36.475 serial prose 8K, 35.769 at 131K, 36.715 serial code and 47.503 DSpark
code. All eight full-vocabulary decision-logit proofs match the new-tokenizer
reference. The segmented Engram gate adds about one percent on a separate
8K sweep (36.249 to 36.655 serial prose); exact Q8 row reuse does not beat the
fork's six-row schedules. DSpark Q8 matrices alone regress code from 46.025 to
40.090, with both serial proofs unchanged, so they remain optional.

The fresh upstream `f48c5e2e` control, on the same existing requant and fixtures
with vision loaded, reaches 38.559/37.753 serial prose 8K/131K and 53.113
DSpark code. Published native-weight references are 41.4/40.5 and 58.8.
The target remains unmet; these comparative results do not qualify a rollout.
See [measurements.json](macdecode-r4/measurements.json) for binary hashes,
applied levers, individual runs, guards and proof hashes. Later batched-attention
ports and final release qualification are still in progress.

Indexed verification (`verify_batch_core=2`) adds the upstream 12-split
small-row attention schedule and BF16 derived index-key views, with explicit
RAM admission. Native payloads still store the original F32 keys. Import
rebuilds the derived views and checks every lower half; a nonrepresentable
cache keeps its complete F32 scoring path. Historical environment presence
(including the string `0`) still selects the old generic batch mode; absence
selects serial rows. Explicit `2` selects the new mode. Synthetic checks cover
2–17 score rows, masks and visibility bounds, signed zero, nonrepresentable
restored keys, and complete attention equality against chronological rows.
The quiet full-model code fixture improves from 46.172 to 48.987 tok/s.

Verification output batching (`verify_output=1`) shares the attention low and
output weight passes while retaining the intermediate low and block tensors.
With indexed verification and optional DSpark Q8 matrices, code reaches
52.227 tok/s (two timed runs after warmup). Matrices alone remain slower; the
surrounding batch schedule matters. All four full serial decision-logit proofs
match, and every component low-row word matches independent scalar rows;
matrix rows meet the upstream numerical bound. These remain comparative
engine measurements, not a paired or live Studio acceptance result.

Mode `verify_output=2` also joins projection/HC and shared-down/routed-add/HC
tails. It writes every original projection, shared, block and HC word, retaining
the fork's ordered FMA expansion. Component checks compare every word with
the retained projection plus epilogue chain at rows 2/6 and K 2304/8192, scalar
and matrix reductions, with and without routed addition, plus invalid ranges.
The quiet full-model matrix comparison reaches 51.833 without the tail and
52.375 tok/s with it; continuation token IDs match. The scalar verification
comparison regresses from 50.556 to 48.449, so mode 2 is selected only with
matrix rows. All four full serial decision-logit proofs match, and the isolated
40 compatibility/environment/build/guard regressions plus native routing and
disk-seed fixture pass. Peak RSS is 356194844672 bytes, physical impact
357554061312, minimum available 182228434944; no paging or budget violation.
Stable executable SHA256:
`4fd81af6cd509f15e003c2bd812b9ec94fe2c01897e0d9190c7eb7699847a620`.

## Capability reductions

The owner-selected target/drafter requants are lossy speed proxies. No other
capability reduction is introduced by these ports. No unsupported feature is
hidden, disabled or replaced with a no-op to make the model load or tests pass.
