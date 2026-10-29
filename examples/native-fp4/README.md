# Native-format kernel and stored-weight experiments

Timing fields now report prompt-only prefill and launch-to-ready TTFT separately. Older startup-inclusive fields retain that meaning; see the primary 100K tables below.

These are synthetic balanced grouped GEMMs, not engine throughput or observed
route histograms. The `hash32-v1` fixture generates the same mathematical
operands independently in Python and CUDA. The activation values are exactly
representable in both FP8 and FP4, so this comparison does not measure their
different activation-quantization errors. FlashInfer produces BF16 and the
HYPER fixture F32; each checks its matching output rounding against a sampled
double reference. Tiny cases rotate weight banks larger than 128 MiB.

Pinned dependencies used for FlashInfer: source
`flashinfer-ai/flashinfer@1d622fdb1b053fc532069914bc9a6b498bd5f722`,
`nvidia-cutlass-dsl==4.8.1`, Torch `2.11.0+cu128`, on SM120. The direct
unmodified GEMM module is imported without the full serving/EP wrapper, whose
declared Torch dependency is newer. Its activation scale granularity is
explicitly set to 32 to match the native checkpoint. This does not qualify
the package's complete fused MoE path. FlashInfer is Apache-2.0. The standalone
scripts import the pinned source; the optional engine adapter separately vendors
a small dependency-closed subset with its licence and byte hashes.

Set `NATIVEFP4_BENCH_ROOT` to the sole benchmark scratch directory,
`NATIVEFP4_FLASHINFER_SOURCE` to the pinned source archive extraction, and
`NATIVEFP4_GPU` to the admitted SM120/121 GPU. Run
`flashinfer_microbench.py` inside the isolated dependency environment through
the existing `slstack.process.OwnedProcess` GPU lease and memory guard. The
recorded admission was a 3 GiB host RSS ceiling and 4 GiB physical RAM floor;
the CUDA driver's commitment allowance is separate. Cache/temp directories
must be in that scratch directory. Do not launch this script beside another
GPU benchmark.

`hyper_microbench.cu` includes the existing GGML `mmq.cuh` directly. Build
with CUDA 13.2, C++17, `-O3 -gencode=arch=compute_120a,code=sm_120a`,
`--expt-relaxed-constexpr`, `GGML_USE_CUDA`, and the GGML include/src/CUDA
include directories. On MSVC the recorded flags also include
`-Xcompiler /EHsc,/Zc:preprocessor`. Invoke the executable with four positional
integers: experts, M per expert, N, K. For example, `384 768 2304 5120`.
Run each case through the same exclusive GPU/memory guard. This fixture uses
prepacked weights/activations, so it excludes the runtime Q4-to-FP4 packer.
It derives from the existing GGML implementation under its MIT licence.

`flashinfer-microbench.json`, `hyper-microbench.json` and `q4-baselines.json`
are distinct experiment families. Kernel TFLOPS cannot be converted directly
into model tokens/s. The full model still needs routed activation packing,
epilogues in the model's required rounding/weighting order, streaming,
attention/indexer work and end-to-end quality gates.

## Stored-weight speed proxy

`speed_proxy.py` runs one model comparison through the shared benchmark lease,
host-memory budget and paging guard. Supply a private experiment config with
the verified model/receipt, rebuilt producer/decode library, ULTRA profile and
YOCO settings. Keep every runtime/cache/output path on the admitted scratch
volume. Set explicit `host_budget_gib`, `producer_host_budget_gib`,
`min_available_gib` and `runtime_min_free_gib` values before launch. The measured
configuration used 7.25/6.25 GiB host ceilings, a 1 GiB physical floor and a
512 MiB desktop GPU reserve. Windows device-backing commitment is a separate
allowance, not permission to page host data.

Create little-endian uint32 prompt and fixed-continuation token files once and
reuse their exact bytes in both arms. Calibrate each model at bias zero first:

```text
python -B examples/native-fp4/speed_proxy.py --config q4-config.json --tokens calibration.tokens.le32 --arm q4 --decode 64 --calibrate q4-spreads.txt --output q4-calibration.json
python -B examples/native-fp4/speed_proxy.py --config q4-config.json --tokens prompt.tokens.le32 --arm q4 --bias 0 --decode 1000 --continuation forced.tokens.le32 --output q4-quality.json
python -B examples/native-fp4/speed_proxy.py --config proxy-config.json --tokens prompt.tokens.le32 --arm proxy --bias 0 --decode 1000 --continuation forced.tokens.le32 --output proxy-quality.json
python -B examples/native-fp4/compare_logits.py --reference q4-quality.json --candidate proxy-quality.json --out proxy-kl.json
```

For the calibrated speed end, set the config's `cache_affinity_spread_file` to
that model's generated file and use `--bias 1.36125`. Generate the proxy's own
calibration using `--arm proxy`; Q4 and proxy files have different identities.
The first 100 decode steps are excluded from warm throughput. The logit
comparison checks input, continuation, producer/decode hashes and profile,
compares the full vocabulary and excludes the pre-setter frontier row.
The shipping dial waits 100 tokens and ramps for eight. To measure its quality
cost, collect a 140-token fixed continuation in each arm and pass
`--skip-first 109` to the comparator, leaving 31 rows after warmup/ramp. Short
32-row continuations can compare proxy versus Q4 at bias zero, but cannot
establish the dial's quality cost; the comparator refuses that claim. A
bias-zero proxy can also serve as the reference for the proxy's dial comparison.

`--arm proxy` explicitly enables the stored MXFP4 experiment; the default
engine setting remains off. Every result is labelled **Q4→MXFP4 requant
(lossy), speed proxy for native FP4**. This W4A4 path uses the existing MMA
dispatch, with no runtime weight conversion; tiny batches retain the generic
MXFP4 path. See [the format and conversion notes](../../docs/NATIVEFP4.md).
Use `--profile-stages` in separate diagnostic runs to collect per-node GPU
timings and tile walls; profiling synchronizes operations and must not be
mixed into the headline throughput comparison.

The accepted diagnostic records are in `proxy-stage-profiles.json`. At 49,555
tokens, Q4 took 29.875 s: stream wait 7.364 s, experts 4.590 s, sparse attention
2.250 s, indexer 0.726 s, other 14.945 s. The **Q4→MXFP4 requant (lossy), speed
proxy for native FP4** took 29.547 s: 8.201 / 3.336 / 2.245 / 0.726 / 15.038 s
in the same order. Other includes projections, compressor/HC work, scheduling,
startup and carry work; it is a wall residual, not a measured compressor timer.
The records also provide disjoint executor phase wall sums and subsets of
other GPU time identified by HC/compressor node names. Anonymous intermediates
remain unattributed. Registration is nested inside binding/preparation, and
launch/GPU spans overlap compute; these diagnostic clocks must not be added
to the stage totals.

Dispatch witnesses confirm the stored-weight GGML native FP4 MMA path, with
no runtime weight conversion. These engine numbers use the original GGML
kernel. An optional [FlashInfer prefill adapter](../../engines/streaming-llama/flashinfer-sm120/README.md)
has passed narrow adapter and matched synthetic 100K checks; its ABI 2
integration regresses speed, as reported below. The 50K proxy expert work achieved
132.80 effective TFLOPS versus about 490 TFLOPS on the balanced standalone
FlashInfer fixture; these different routing workloads are not a matched
kernel comparison. The complete prompt fetches 298.84 GB of model tensors,
not the entire 501.61 GB file. Engram rows are fetched on demand.

New comparisons use 100K prompts as the primary point. The longer campaigns
and decode MMA qualification are deferred while attention and single-pass
experiments have higher GPU priority. No default changes follow from these
lossy or diagnostic results.

The separate `DS4_CUDA_MXFP4_DECODE_MMA` experiment is also off by default.
Its bounded K=2304 tail loader and corrected full-output GPU fixture pass. A
short full-model functional smoke is recorded below; broad accuracy and warm
decode speed remain unqualified. The accepted warm-decode results use the
existing MMVQ path.

## Primary 100K comparison: both timing boundaries

Every table below separates **prompt-only prefill** from **launch-to-ready TTFT**. All tokens/s columns use the stated clock. Use prompt-only time for comparisons with the reference repository. Historical `prefill_seconds` and `prefill_tok_s` fields remain aliases for TTFT and its throughput; explicit fields remove that ambiguity.

The native prompt clock covers the embedding, every prompt sweep, between-sweep ownership admission and token preparation through synchronized frontier logits. It excludes initial model open/allocation and final bulk export/certification. Early Engram prefetch can overlap initial open. TTFT includes process startup, model open, prompt processing, final certification and READY IPC; external harness prechecks, recall and decode are excluded. Each recovered clock is bound to the model, producer and exact input hashes, and its chunks reconcile to the native total. The original raw measurement files are unchanged.

Both FP4 rows in every table are **Q4→MXFP4 requant (lossy), speed proxy for native FP4**. Each matrix uses one common producer and stable decode DLL, a frozen 100,000-token synthetic input, ULTRA, bias zero and approximate YOCO140. These are single measurements.

### Historical ABI2 integration

Full timing, dispatch, quality and diagnostic evidence: [`primary-100k-comparison.json`](primary-100k-comparison.json).

| Path | Prompt-only s | Prompt-only tokens/s | TTFT s | Tokens/TTFT s |
|---|---:|---:|---:|---:|
| Q4 control | 43.686 | 2289.08 | 46.453 | 2152.71 |
| Stored MXFP4 W4A4 | 41.631 | 2402.08 | 44.406 | 2251.95 |
| Stored MXFP4 FlashInfer W4A8 | 66.411 | 1505.77 | 69.547 | 1437.88 |

Separate synchronized stage diagnostics; the stage columns sum to the **prompt-only** wall. TTFT is also shown, and is not part of that sum.

| Path | Profile prompt s | Profile TTFT s | Stream wait s | Expert s | Attention s | Indexer s | Other prompt wall s | Expert TFLOPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4 control | 46.663 | 49.500 | 6.840 | 9.144 | 4.456 | 2.500 | 23.724 | 97.66 |
| Stored MXFP4 W4A4 | 44.608 | 47.422 | 7.404 | 6.628 | 4.454 | 2.497 | 23.625 | 134.72 |
| Stored MXFP4 FlashInfer W4A8 | 122.350 | 125.156 | 4.503 | 17.656 | 4.430 | 2.522 | 93.239 | 50.58 |

### ABI3 workspace integration before the MMQ tail bounds fix

Full timing, dispatch, quality and diagnostic evidence: [`primary-100k-workspace-comparison.json`](primary-100k-workspace-comparison.json).

| Path | Prompt-only s | Prompt-only tokens/s | TTFT s | Tokens/TTFT s |
|---|---:|---:|---:|---:|
| Q4 control | 43.991 | 2273.22 | 46.922 | 2131.20 |
| Stored MXFP4 W4A4 | 42.029 | 2379.33 | 44.859 | 2229.21 |
| Stored MXFP4 FlashInfer W4A8 | 40.734 | 2454.96 | 43.532 | 2297.16 |

Separate synchronized stage diagnostics; the stage columns sum to the **prompt-only** wall. TTFT is also shown, and is not part of that sum.

| Path | Profile prompt s | Profile TTFT s | Stream wait s | Expert s | Attention s | Indexer s | Other prompt wall s | Expert TFLOPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4 control | 48.692 | 51.531 | 8.632 | 9.123 | 4.451 | 2.493 | 23.993 | 97.88 |
| Stored MXFP4 W4A4 | 44.816 | 47.672 | 7.542 | 6.621 | 4.452 | 2.496 | 23.703 | 134.86 |
| Stored MXFP4 FlashInfer W4A8 | 46.431 | 49.328 | 9.537 | 5.370 | 4.452 | 2.515 | 24.556 | 166.30 |

Each profile accounts for 892,943,518,924,800 useful routed-expert FLOPs: `2*3*5120*2304*6*(21*100000+19*140)`. There are 61 layer passes; upper-layer YOCO work runs only in the final sweep. This approximate 140-row tail differs from the reference’s exact 2541-row cone. Expert timings include packing/layout work. The roughly 490-TFLOPS balanced standalone GEMM excludes those adapters and uses different routing, so the integrated figures do not measure pure GEMM-core utilization.

ABI3 uses caller-owned bounded workspace, skips inactive experts during layout conversion and coalesces packed weight stores. The combined change reduced FlashInfer prompt-only prefill from 66.411 to 40.734 seconds, and TTFT from 69.547 to 43.532 seconds. Individual contributions were not isolated. Diagnostic tile compute fell from 108.579 to 27.106 seconds. Profiling walls are diagnostic, separate from clean throughput.

[`flashinfer-workspace-regression.json`](flashinfer-workspace-regression.json) checks every one of the 8 x 129280 logit bit patterns between old and new builds: all three paths match exactly. This is a narrow regression check on one fixture. Seven forced rows give unchanged mean KL(Q4 to W4A4)=0.036049, KL(Q4 to FlashInfer)=0.046110 and KL(W4A4 to FlashInfer)=0.043305; top-1 agrees on all seven. The untouched frontier is separate. This does not establish broad accuracy or recall, or remove the proxy’s measured quality differences from Q4.

[`flashinfer-workspace-validation.json`](flashinfer-workspace-validation.json) records the current ABI3 identity, source hashes, four routing/layout oracle checks and three checks using actual stored gate/up/down weights. Tail checks and all stored-weight checks pass compute-sanitizer with zero errors. It also preserves the rejected initial 128-byte-alignment smoke test and cleanup result.

ABI3 is the implementation behind the existing FlashInfer toggle. The toggle stays off by default because the proxy is lossy. GGML VMM workspace mappings remain bounded and retained until backend destruction; decode-cache admission observes the remaining physical free memory. Decode performance with this retained workspace is not qualified.

FlashInfer has about 3.2% higher prompt-only throughput than current MXFP4 in this fixture. That small end-to-end difference needs repeated workload testing. Integrated expert throughput is 166.3 TFLOPS. No source-native or broad quality-neutral speed claim follows.

## MMQ tail bounds fix and repeated 100K matrix

The streaming MXFP4 loader previously read beyond the final K=2304 row while loading the padded MMA tile. An exact-size expert allocation exposed 11 compute-sanitizer errors. It now zero-fills the missing blocks and scale pairs. This correctness fix applies to the existing stored-MXFP4 path; the lossy proxy and FlashInfer settings remain off by default.

[`mmq-tail-validation.json`](mmq-tail-validation.json) preserves the rejected read and fixture failure, source/build identities and fresh guards. Both streaming and DS4 fixtures pass all 44,544 gate/down output values each, with zero sanitizer errors. The streaming 384-expert and observed sparse tail checks also pass. The DS4 numerical failure was in the fixture: its fixup launch omitted grid.y=Y/32. Full-model decode MMA qualification remains pending.

[`primary-100k-tail-comparison.json`](primary-100k-tail-comparison.json) repeats the matched primary matrix with one new common producer and the same input, decode DLL, bias and approximate YOCO. Both FP4 rows are **Q4→MXFP4 requant (lossy), speed proxy for native FP4**.

| Path | Prompt-only s | Prompt-only tokens/s | TTFT s | Tokens/TTFT s |
|---|---:|---:|---:|---:|
| Q4 control | 44.147 | 2265.14 | 46.984 | 2128.38 |
| Stored MXFP4 W4A4 | 41.910 | 2386.06 | 44.781 | 2233.09 |
| Stored MXFP4 FlashInfer W4A8 | 41.492 | 2410.11 | 44.360 | 2254.28 |

Separate synchronized diagnostics, with stages summing to prompt-only wall:

| Path | Profile prompt s | Profile TTFT s | Wait s | Expert s | Attention s | Indexer s | Other prompt s | Expert TFLOPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4 control | 48.085 | 51.016 | 7.938 | 9.122 | 4.451 | 2.506 | 24.068 | 97.89 |
| Stored MXFP4 W4A4 | 45.313 | 48.375 | 7.720 | 6.623 | 4.452 | 2.497 | 24.021 | 134.82 |
| Stored MXFP4 FlashInfer W4A8 | 44.538 | 47.422 | 7.910 | 5.393 | 4.454 | 2.510 | 24.272 | 165.58 |

[`mmq-tail-regression.json`](mmq-tail-regression.json) verifies the recorded artifact hashes and all 8 x 129280 logit bit patterns for each path against the previous ABI3 build: zero differences. The two 100K repetitions show a small, variable end-to-end FlashInfer gain over current MXFP4; these are not enough to establish a general speed improvement. No additional quality claim follows. The first batch completed the clean matrix and Q4 profile, then the next launcher exited before starting an engine. The two missing profiles passed under a fresh lease; the rejected batch receipt is retained.

## Observed routing kernel comparison

[`observed-kernel-comparison.json`](observed-kernel-comparison.json) uses three histograms selected from the captured 49,555-token prompt: the most and least concentrated full 4096-row graphs, plus the final 140-row graph. All 384 experts remain in the weight banks, including inactive experts. These are synthetic representable operands using observed counts, not actual stored-weight tensors. Both implementations check finite full outputs and sampled exact double references, with their respective F32/BF16 output rounding.

| Graph | N x K | Active experts | Maximum M | HYPER eager TFLOPS | FlashInfer eager TFLOPS | FlashInfer graph TFLOPS |
|---|---|---:|---:|---:|---:|
| q4-layer20-start8192-rows4096 | 2304 x 5120 | 97 | 4069 | 99.55 | 481.00 | 476.90 |
| q4-layer20-start8192-rows4096 | 5120 x 2304 | 97 | 4069 | 83.35 | 465.11 | 459.64 |
| q4-layer19-start0-rows4096 | 2304 x 5120 | 276 | 1192 | 125.22 | 338.57 | 329.06 |
| q4-layer19-start0-rows4096 | 5120 x 2304 | 276 | 1192 | 132.50 | 326.86 | 316.29 |
| q4-layer39-start49415-rows140 | 2304 x 5120 | 120 | 58 | 17.11 | 41.65 | 41.82 |
| q4-layer39-start49415-rows140 | 5120 x 2304 | 120 | 58 | 19.26 | 42.30 | 42.43 |

The raw normalized results are [`observed-hyper-microbench.json`](observed-hyper-microbench.json) and [`observed-flashinfer-microbench.json`](observed-flashinfer-microbench.json). Times exclude packing and activation quantization. HYPER uses 100 eager launches; FlashInfer records 20 eager launches and separate CUDA graph replay timing. Logical effective bandwidth counts the operands touched by active experts and outputs, not PCIe traffic. These are distinct output precisions and timing protocols, so the columns do not isolate instruction throughput.

To reproduce a histogram, pass `--route-cases observed-routing.json --output <new-result.json>` to the FlashInfer fixture. For HYPER, pack one case’s 384 counts as little-endian signed int32 values and pass `384 <maximum-count> N K --counts <counts-file>`. In counts mode the fixture’s legacy `M_per_expert` output means the launch maximum; normalized public records use `maximum_M`. Build with `NATIVEFP4_DS4_MMQ=1` and DS4 MMQ headers to check that implementation; add `--full-reference` for the tiny six-expert M=1 full-output oracle. Always use an exclusive GPU lease, explicit host-memory ceiling, physical-RAM and device floors, and paging stop. The recorded accepted host ceilings were 2 or 3 GiB with a 1 GiB physical floor. The first FlashInfer histogram launcher used Python3.12 with a Python3.13 dependency environment and failed before benchmarking; the accepted retry uses the pinned environment’s interpreter.

## CUTLASS Windows dense probe

[`cutlass-windows-diagnostics.json`](cutlass-windows-diagnostics.json) records a rejected MXFP8 dense probe built from CUTLASS `91843b71d90737e0a9b86f7d9df213cff5a3526f`. The derived [`cutlass_microbench.cu`](cutlass_microbench.cu) retains NVIDIA's BSD-3-Clause notice and adds a sampled double/BF16 oracle, rotating weight banks and diagnostic checks. No dense performance result is accepted.

The CUDA13.2/MSVC18 build without `/Zc:__cplusplus` runs initialization, then fails at a TMA descriptor prefetch with `cudaErrorMisalignedAddress`; compute-sanitizer confirms the fault. CUDA's header enables `alignas(128)` through its `__cplusplus` check. Adding `/Zc:__cplusplus` satisfies `alignof(CUtensorMap)==128`, but MSVC rejects the generated by-value CUDA parameter stubs with C2719. This is a specific Windows build/ABI blocker, not an SM120 support verdict. The FP4 FlashInfer AOT adapter uses a different interface and has separate accepted evidence. Native conversion still uses Q8 dense because the required FP8 GGUF/runtime path is absent.

## Decode MMA functional smoke

[`decode-mma-smoke.json`](decode-mma-smoke.json) records a matched 2K smoke test of the stable vector DLL, candidate DLL with MMA disabled, and candidate DLL with `DS4_CUDA_MXFP4_DECODE_MMA=1`. All use the same producer/model/input, bias zero, ULTRA/approximate YOCO140 and disabled DS4 decode graphs. This is **Q4→MXFP4 requant (lossy), speed proxy for native FP4**.

Both vector DLLs match all eight full-vocabulary logit rows bit for bit. All three paths have identical untouched prefill logits. The MMA arm completes eight forced rows and decode64, with launch witnesses for all40 layers. The report separates seven forced-row quality comparisons from the untouched frontier. These are narrow functional checks; the 63 returned decode step intervals do not discard100warmup steps and are not a qualified warm-decode benchmark. The candidate remains off; no100K or broad quality/speed adoption claim follows.

On the seven forced rows, vector-to-MMA mean KL is 0.165582 (maximum 0.933429); top-1 agrees on all seven. The two candidate-DLL arms take 11.811 and 11.851 seconds for their 63 returned decode intervals. This short sample demonstrates no speed gain and has a numerical cost, so it does not justify adopting the candidate. Native-source and broader quality/warm-decode checks remain outstanding.


## 100K exact-cone expert batching

[`sweep-batch-comparison.json`](sweep-batch-comparison.json) records the ABI4
experiment. Both rows are **Q4→MXFP4 requant (lossy), speed proxy for native FP4**.
Matched 100000 tokens, ULTRA, exact cone keep2541, context1000001, bias0,
49152/50848-token sweeps; common producer, model, bridge and decode library.

| Expert chunk | Prompt-only s | Prompt-only tokens/s | TTFT s | Expert s (profile) | Effective expert TFLOPS |
|---|---:|---:|---:|---:|---:|
| 4096 | 38.142 | 2621.76 | 41.187 | 5.646 | 166.24 |
| 8192 | 41.396 | 2415.72 | 44.469 | 4.994 | 187.95 |

The 8K encoder chunks reduce completed expert products from1689 to969.
Attention and router products retain their original4K shapes; layer20 and
the short replay suffix remain4K to bound scratch. Gather already writes
MXFP8 values/scales directly, and that fusion is reused. The larger graph and
its RAM spill cost outweigh the expert saving. Keep4096 as the default.

All8×129280logit bit patterns match exactly after fixing the paired causal
indexer's query origin. The original 8K experiment diverged at layer2's
second attention child because the correct offset was applied only to
native-half pairs. The checked visibility-owner offset now applies to every
paired graph. The initial rejected comparison remains in
[`sweep-batch-before-causal-fix.json`](sweep-batch-before-causal-fix.json).

Four synthetic8K/16K adapter checks pass sampled independent double/BF16
references and full-output finiteness. Both16K projections and actual stored
gate/up/down matrices pass compute-sanitizer with zero errors. These kernel
oracles do not establish broad model accuracy. The16K model attempt refuses
allocation:7.483GB graph plus4.622GB temporary exceeds its7.709GB graph cap.
No16K throughput or full49K expert-batch claim follows.

Reproduce the clean pair with `sweep_batch.py --config <config> --tokens
<100k.le32> --output-prefix <scratch>/batch --continuation <fixed.le32>`;
repeat with a new prefix and `--profile-stages` for separate synchronized
diagnostics. Use the ABI4 build, verified proxy receipt and explicit host
headroom described in the bridge README. No production model path changes.
