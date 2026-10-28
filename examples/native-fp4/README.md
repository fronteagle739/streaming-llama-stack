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
Its CPU build includes a bounded K=2304 tail loader and launch counters, but
the corrected GPU fixture, full-model accuracy and speed qualification are
deferred. The accepted decode results use the existing MMVQ path.

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

### Current ABI3 workspace integration

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
