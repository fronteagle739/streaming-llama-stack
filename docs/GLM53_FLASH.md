# GLM-5.3 Flash: experimental graph port

The retained streaming-llama sources now recognize `glm5-next`, including the
ds4 GGUF layout. **The single-machine ULTRA-prefill/ds4-decode engine is not
implemented or qualified for this architecture.** The stack refuses to pass a
GLM file through its DSV4.1 single-engine runtime.

The expected original model is `GLM-5.3-Flash-Q4_K.gguf`, 190,875,526,464 bytes,
SHA-256 `c7a0d950363238dd7804782c88340d737775aba53a15f8d4fdcc34e984f25221`.
Round two independently verified an existing LAN copy against this SHA and byte
count, checking file holders and unchanged physical identity. That copy is still
on the peer; there is no verified RTX copy or trained-model qualification.
No weights, SDK libraries, generated GGUFs or binaries are committed.

## Reused source

The paused GLM port contributes the hybrid attention/recurrent/indexer memory,
KDA/MLA model graph, loader aliases, mHC operations and gated delta-net kernels.
The CUDA launch compatibility shim is retained. Existing DSV4.1 operation IDs
remain unchanged; the three new operations are appended.

The new integration synchronizes indexer and attention cache dimensions when
reserving a layer-major graph. The CPU build also exposes two existing assert
bugs: an unavailable power-of-two helper and reading the sequence count of an
empty KV cell. These are fixed without disabling assertions.

The graph tools cover full/shared indexers, the upstream and ds4 tensor naming
layouts, 64/320-token prompts, 64/16-token microbatches, and eight continuation
tokens. State restore clears the original caches before restoring, exercising
the indexer and recurrent state as well as attention. Tests use F32 KV.
An independent scalar HC check covers the legacy weighted pre operation, the
new gated pre operation and identity post mixing with both four and twelve
residual streams.

## Reproduce CPU checks

Configure the retained llama CMake project with `LLAMA_BUILD_GLM53_TESTS=ON`,
`LLAMA_BUILD_COMMON=OFF`, `LLAMA_BUILD_TOOLS=OFF`, `LLAMA_BUILD_TESTS=OFF`,
`GGML_CUDA=OFF` and `BUILD_SHARED_LIBS=OFF`; build `glm53-logits` and
`glm53-load-check` and `glm53-hc-check`. On Windows the retained optional DirectStorage headers
require a locally provided SDK import library. Run all builds through the stack
guardian with explicit host and commit limits.

Using an existing local `gguf` Python package and NumPy, generate tiny fixtures:

```sh
python -B tests/streamllama/glm5-next/make_tiny_glm5next.py "$FIXTURES"
```

From the retained llama source directory, run the checked-in matrix:

```sh
python -B tests/streamllama/run_glm53_cpu.py --config "$CONFIG" \
  --models "$FIXTURES" --binary "$CPU_LOGITS_BINARY" --output "$RESULTS"
```

`CONFIG` is a stack config. Keep its runtime, temporary files and results inside
one task scratch root; set equal host/commit caps and a physical-RAM floor.
If a shared engine hold ledger is used, supply it as `hold_file`. The runner
yields to active holds. It performs CPU checks and acquires no GPU lease.

The original `run_acceptance.sh` separately supports comparisons against an
already available upstream CPU build. Same-fork layout/microbatch checks do
not replace that independent comparison or real-model acceptance.

## Remaining single-engine work

1. Generalize the ULTRA producer's model receipt, sweep and weight provider.
   Its carry sizing, Engram handling and state views currently describe V4.1.
   Keep DirectStorage lookup tied to original GGUF offsets after tensor aliases.
2. Extend ds4's external decode adapter with the existing GLM graph kernels.
   Bind KDA convolution/recurrent state, MLA latents, pooled indexer keys and
   raw key/gate tails in the correct layouts, preserving ownership across turns.
3. Use GLM's tokenizer, chat template, tool parser and actual stop IDs. Reuse
   the process/session controller after these architecture-specific contracts
   are supplied. Do not use V4.1's hard-coded EOS, renderer or checkpoint ABI.
4. Derive fold/sweep memory admission from GLM shapes. Preserve the 512 MiB
   desktop reserve, 4 GiB runtime floor, explicit host cap and progress guards.
   Qualify quality/speed calibration for this model. V4.1 YOCO is unavailable
   until an architecture-specific equivalence proof exists.
5. Verify the real file SHA, then qualify a coherent multi-turn WebUI chat,
   roughly 16K/50K prefill and warm decode on fast storage. Only then publish
   262K and measured maximum-context WebUI/Studio presets.

Two-machine serving is outside this change. No presets are installed, no
production services are restarted, and no benchmark numbers are inferred from
the synthetic models.

## Round two: scalar KDA and CUDA bring-up

The CPU planner previously omitted `GATED_DELTA_NET` workspace. The operation
uses a per-thread delta vector and, with multiple snapshot slots, a working
state matrix. Larger graphs could accidentally provide enough workspace for a
different operation. A standalone KDA graph instead crashed with an access
violation. The planner now reserves the delta and optional matrix explicitly;
the existing cache-line allowance covers the operation's prefix padding.

`glm53-kda-check` checks output and every carried-state element against a
separate scalar recurrence. Ten cases per backend cover dimensions 16/128,
1/4/17/65 tokens, multiple sequences, grouped/broadcast queries, nonzero
asymmetric initial state, the actual 64-head by 128-dimension Flash shape,
and three snapshot slots. CPU and CUDA both pass; maximum absolute output or
state error is below 3e-8. The original 18-case CPU graph matrix also passes
after recompiling the planner. CUDA same-layout clear/restore checks pass
exactly at 64 and 65 prompt tokens plus eight continuation tokens.

Build the optional `glm53-kda-check` target with the other graph tools. A CUDA
build accepts `--cuda`; requesting CUDA in a CPU-only build fails explicitly.
Use the existing guarded runner, with a config setting equal host/commit caps
at or below 3 GiB, at least 4 GiB available RAM, a selected GPU UUID and the
shared hold ledger:

```sh
python -B tests/streamllama/run_glm53_kda.py --config "$CONFIG" \
  --cpu-binary "$CPU_KDA_CHECK" --cuda-binary "$CUDA_KDA_CHECK" \
  --output "$RESULTS"
```

The result directory must be new and inside the configured runtime. On Windows,
`--cuda-library-directory` can point at the installed CUDA runtime DLL directory.
The runner checks holds and acquires/releases a GPU lease for the CUDA run;
both children run under the existing RSS/commit/available-RAM guards.

**Round two did not pass full graph numerical acceptance.** With the existing random
ds4-layout fixture, a 65-token prompt, 16-token microbatches, restored state and
eight continuation tokens, only 67/73 CPU/CUDA top tokens agree; maximum logit
difference is 3.6379032. CUDA 64-versus-16 microbatch comparison at a 64-token
prompt agrees on 71/72 top tokens, with maximum difference 1.393062. These are
failure measurements, not tolerances chosen to pass. Scalar KDA correctness
and same-layout restore do not establish full-model correctness or explain
the remaining graph/precision differences.

Sanitized measurements are in `examples/glm53_flash/round2.json`. No trained
model chat, 16K/50K prefill rate, warm decode rate, ULTRA producer, external GLM
ds4 state binding, or context preset is qualified by this round. Fast storage
capacity remains insufficient for the full model under the authorized move
destinations. Model discovery alone does not remove that blocker.

## Round three: CUDA numerical agreement

Two independent causes were isolated with the tiny fixtures and scalar oracles:

* NVIDIA's floating-point MMF kernel consumes TF32 operands for F32 weights.
  Switching microbatch size selected different precision, and an explicit F32
  request did not exclude that kernel. Routed expert slices also lost the
  caller's precision parameter. The first material discrepancy was in MoE;
  it then changed a later sparse pool selection. F32 dense/routed operations now
  reuse the existing F32 vector/cuBLAS paths, and expert slices preserve their
  precision. Odd-width routed fallback graphs are excluded from capture because
  their existing host ID sort synchronizes the stream.
* Correcting precision exposed exact zero plateaus in indexer scores at longer
  prompts. CPU partial sorting and CUDA bitonic sorting could choose different
  tied pools. For example, scores tied at zero for pools 0, 4 and 6; the backends
  selected `{0,4}` and `{0,6}`. CPU argsort/top-k and both CUDA bitonic sorting
  paths now break equal-value ties by the smaller original index, matching the
  existing tiled CUDA merge. No score perturbation or tolerance increase is used.

The fixes change default F32 execution as well as explicit-F32 diagnostics.
F32 matrix throughput may be lower than TF32; no performance qualification is
inferred from these fixtures. Default quantized matmul dispatch is retained.
The tie convention also applies to other users of the changed CPU/CUDA sorts.
Other backend implementations and optional CUB fallbacks are not qualified here.

`glm53-op-check` adds scalar comparisons for dense/routed matmuls, masked MLA,
mHC pre/gated-pre/Sinkhorn/post, and indexer scoring/top-k. It checks the exact
selected set at a zero plateau, including signed zeros and masked future pools.
Four successive graph executions change input values and expert IDs. The KDA
scalar matrix remains separate. Negative controls linked to the pre-fix libraries
fail these new regressions; corrected binaries pass.

`glm53-logits` accepts optional `trace_dir` and `force_f32` arguments after the
existing restore argument. The trace reads synchronized tensors without adding
nodes or pinning buffers, but scheduler callbacks split execution; use it only
for diagnosis. `trace_dir=- force_f32=0` installs no callback. Backend
`GGML_CUDA_GRAPH_TRACE=1` emits bounded direct/launch markers after execution.

The guarded runner verifies the resolved child environment and actual launch
markers. This matters: the producer launch profile intentionally disables CUDA
graphs. Standalone checks use the native engine profile, preserving the same
GPU lease and memory guards, without changing that production default.

Build `glm53-logits`, `glm53-op-check` and `glm53-kda-check` for CPU and CUDA,
then run from the retained llama source directory:

```sh
python -B tests/streamllama/run_glm53_cuda.py --config "$CONFIG" \
  --models "$FIXTURES" --cpu-binary "$CPU_LOGITS" --cuda-binary "$CUDA_LOGITS" \
  --cpu-ops "$CPU_OPS" --cuda-ops "$CUDA_OPS" \
  --cpu-kda "$CPU_KDA" --cuda-kda "$CUDA_KDA" --output "$RESULTS"
```

Use explicit host/commit caps at or below 3 GiB, a 4 GiB physical-RAM floor,
the selected GPU UUID, a shared hold ledger and task-local cache/temp paths.
The output directory must be new. On Windows the optional
`--cuda-library-directory` selects installed CUDA runtime DLLs. A nonzero
`gpu_wait_seconds` lets admission wait for transient utilization to settle.

The matrix covers all three layouts, 64/65/320 prompt tokens, 64/16 microbatches,
eight continuation tokens, cleared-cache restore, default/F32 precision and
capture enabled/disabled. Every top token must agree, and the existing
`compare_logits.py` maximum-logit threshold remains 1e-2. Numerical evidence and
the separate scalar thresholds are in `examples/glm53_flash/round3.json`.
These checks qualify synthetic numerical agreement only. The original model,
ULTRA/ds4 producer integration, trained-model WebUI and speed acceptance remain
outside this round.


## Round four: trained-model streaming bring-up (partial)

The verified Q4_K GGUF is now on local fast storage. The target engine is still
incomplete: the measurements below use the existing layer-major streaming graph
for both prefill and decode. They do not qualify the requested ULTRA/ds4 GLM
producer or its WebUI/Studio presets.

The trained model exposed several integration defects absent from a resident
random-weight graph:

* Target streaming stages now exclude the NextN/MTP block, and include the
  indexer's learned pool gate and positional weights.
* Initial streamed addresses use the fetcher's aligned allocation offsets, so
  views constructed before execution agree with subsequent weight remapping.
  Global weights already resident on a device keep their allocation.
* BF16 scalar normalization and affine tensors are explicitly cast to F32
  inside their consuming layer. The graph no longer passes them to binary
  broadcast operations that expect F32/F16. F32 fixture weights remain unchanged.
* Direct-compute DirectStorage heaps do not use the staging-only range relay.
  Its absent staging pointer had left a CUDA invalid-value error reported by
  the next layer's cast. Missing staging/destinations now fail explicitly.
* mHC input normalization is named at its actual layer boundary. A multi-batch
  restore targets the prior four-stream activation; the first mHC matmul's
  first source is a weight. Pool plans are rebuilt when replay seeks a batch.
* Output-head sizing includes the largest requested output count across batches,
  including last-token-only requests. Final hidden restoration honors pinned
  host staging, rather than reading an empty ordinary CPU vector.

Recurrent streaming callers must submit a uniform bulk and then the exact tail
in a separate decode call. The bring-up and logits tools do this explicitly;
implicit padding would advance KDA recurrence through fake tokens. Unsupported
CUDA convolution geometry (projection not divisible by 128) and multi-batch
shared-indexer models now refuse. The latter needs an additional per-batch
selection boundary owner. The trained ds4-layout model has a full indexer in
every MLA block. Resident graph coverage for shared-indexer layouts remains.

`glm53-stream-smoke MODEL PROMPT_FILE CONTEXT DECODE_TOKENS
[FOLLOWUP_SUFFIX_FILE]` is built with `LLAMA_BUILD_GLM53_TESTS=ON`. Invoke it
through the stack memory/GPU guardian with explicit budgets, a selected GPU and
shared lease ledger. It requires `LLAMA_PREFILL_WF=1`; the numerical runner
below records the other streaming settings. It accepts preformatted GLM text,
uses F16 KV and one sequence, and caps requested decode at 128 tokens. The
optional suffix follows the already committed model end/next-role token,
without a new BOS. A truncated first response refuses the follow-up.

The trained model answered `5` to two plus three, then `7` to ?add two? in the
same process: 21 existing token positions were retained and 17 suffix tokens
appended. This is CLI evidence, not GLM WebUI acceptance. Each streamed decode
step still reads the target layers, taking about 11.7 seconds; no native warm
decode throughput is claimed.

| Trained-model bring-up | Measurement |
|---|---|
| 16,380-token prefill | 55.281544 s; 296.301421 tokens/s |
| 49,520-token prefill | 232.208352 s; 213.256756 tokens/s |
| Result after both long prompts | `5`, then end of turn |

Times include uniform bulk plus exact-tail processing. Repeated `word` filler
and a short arithmetic question are a basic coherence check, not long-range
retrieval or a quality benchmark. The 49.5K loop spends much of its time in
hidden transfers and input setup; this is not the dedicated ULTRA fold/sweep
executor. The body-only rates must not be substituted for the full times above.

`ds4-decode.dll` also has an experimental `ds4glm_*` C ABI. It reuses native
GLM token kernels, external weight registration and expert hooks. Its graph
allocates bounded single-token scratch without native weight residency,
per-layer caches or a prefill workspace. The caller owns all KDA convolution
and recurrent buffers, F16 latent/index caches and raw pool tails. Binding
validates every span, precision, context and frontier before replacing views;
failed decode requires fresh binding. Close detaches borrowed views before
freeing scratch. GLM and V4.1 external sessions are exclusive because native
shape/backend globals represent one model.

The real-model-header ownership test passes eight cases, including a one-byte
scratch refusal, missing/undersized owners, bad frontiers and preservation of
borrowed CUDA memory after close. The successful 128-context bank owns
154,210,304 bytes; native scratch is 48,051,540 bytes. This does not prove weight
registration, prefill state conversion or even one trained-model native decode.
The ABI's context bound is not a qualified serving preset.

For the synthetic streaming checks, generate the existing fixtures with the
optional KDA head dimension `32` (128 channels) and run:

```sh
python -B tests/streamllama/run_glm53_streaming.py --config "$CONFIG" \
  --models "$FIXTURES" --cpu-binary "$CPU_LOGITS" \
  --cuda-binary "$CUDA_LOGITS" --unsupported-model "$ORIGINAL_64_CHANNEL_MODEL" \
  --output "$NEW_OUTPUT"
```

Fourteen uninstrumented streaming comparisons against seven CPU references
pass, including 64/65/320 prompt tokens, 64/16 microbatches, all-row/last-only
logits and eight continuation steps. All 1,081 top tokens agree; maximum logit
difference is 2.3603439331e-05 at the unchanged 1e-2 threshold. Two unsupported
controls refuse cleanly. The original CPU graph suite passes 18 cases. These
checks use a 3 GiB host cap, up to 8 GiB WDDM commit allowance and a 4 GiB free
physical-RAM floor. Long trained-model runs use a 6 GiB host cap and 32 GiB
WDDM commit allowance under the same physical floor. Commit allowance is not a
physical-memory allowance. Final monitored receipts report no owned paging;
kernel ETW is unavailable, so they do not prove system-wide absence of paging.

The native ownership check has a guarded entry point:

```sh
python -B engines/ds4-win/run_glm_external.py --config "$CONFIG" \
  --model "$MODEL" --library "$DECODE_DLL" --cuda-runtime "$CUDA_RUNTIME_DLL" \
  --output "$NEW_RESULT"
```

Remaining work: bridge and round-trip native/llama convolution and pooled-index
layouts; register caller-owned dense and expert weights; compare actual native
decode logits and keep-open appends against streaming references; integrate the
existing producer/router/planner and quality/speed controls; then qualify GLM
WebUI/Studio 262K/max presets and warm decode. No GLM YOCO behavior is enabled.
Sanitized measurements and the separate DSV4.1 gate are in
`examples/glm53_flash/round4.json`.


The required DSV4.1 shared-code WebUI regression was rerun with the current
producer and external-decode DLL. Four real turns pass, including retrieval of
`AMBER OTTER 731` from 49,553 tokens, a cached follow-up, and two 1,000-token
quality/speed responses. Incremental SSE, progress and live stats remain active;
there are no browser page errors. The native process shuts down gracefully.

| DSV4.1 comparison | Current | Round-two reference | Change |
|---|---:|---:|---:|
| Fresh prefill tokens/s | 1,773.6774 | 1,592.1285 | +11.40% |
| Max-quality warm tokens/s | 7.04406 | 6.81180 | +3.41% |
| Max-speed warm tokens/s | 18.79700 | 18.15505 | +3.54% |

Both warm intervals contain all 899 evaluated steps after the first 100.
The final prefill is faster than the requested absolute +/-5% band, so the
strict all-metrics timing-band verdict is **false**. All three metrics satisfy
a 5% no-slowdown floor. An earlier run measured 1,659.57 prefill tokens/s;
both runs are retained instead of selecting the one inside the band. This is
a regression smoke with observed timing variation, not a statistical speedup
claim or permission to merge the unfinished GLM engine. Browser screenshots
are `round4-dsv-recall.png` and `round4-dsv-speed.png` beside the JSON evidence.
