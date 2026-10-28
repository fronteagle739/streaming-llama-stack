# Native-format kernel and stored-weight experiments

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

## Primary 100K comparison

[`primary-100k-comparison.json`](primary-100k-comparison.json) records the matched
100,000-token fixture, one producer/decode DLL, bias zero, ULTRA and approximate
YOCO140. Both FP4 rows are **Q4→MXFP4 requant (lossy), speed proxy for native FP4**.
Startup is included; these are single measurements on a synthetic prompt.

| Path | Unprofiled prefill seconds | tokens/s |
|---|---:|---:|
| Q4 control | 46.453 | 2152.71 |
| Stored MXFP4, current W4A4 | 44.406 | 2251.95 |
| Stored MXFP4, FlashInfer W4A8 ABI 2 | 69.547 | 1437.88 |

FlashInfer completed all 544 graphs and 1632 gate/up/down products. Its adapter
is correct within the scalar-check scope but this integration is slower.
The separate synchronized profiles give:

| Path | Stream wait s | Expert operators s | Sparse attention s | Indexer s | Other wall s | Effective expert TFLOPS |
|---|---:|---:|---:|---:|---:|---:|
| Q4 | 6.840 | 9.144 | 4.456 | 2.500 | 26.561 | 97.66 |
| MXFP4 W4A4 | 7.404 | 6.628 | 4.454 | 2.497 | 26.439 | 134.72 |
| FlashInfer W4A8 ABI 2 | 4.503 | 17.656 | 4.430 | 2.522 | 96.045 | 50.58 |

Each profile accounts for 892,943,518,924,800 useful routed-expert FLOPs:
`2*3*5120*2304*6*(21*100000+19*140)`. The two sweeps execute 61 layer passes;
upper-layer YOCO work runs only in the final sweep. Expert operator timings
include activation packing and FlashInfer's runtime weight-layout rearrangement.
The roughly 490-TFLOPS standalone balanced GEMM excludes those adapters and uses
different routing. These figures do not measure pure FlashInfer core utilization.
The FlashInfer profile's tile compute spans total 108.579 s, versus 34.713 s
summed node timings; that gap includes execution/instrumentation overhead and
cannot be assigned to compressor arithmetic. Profiling walls are diagnostic.

Seven matched forced rows across the full 129280-token vocabulary give mean
KL(Q4‖W4A4)=0.03605 and KL(Q4‖FlashInfer)=0.04611; top-1 matches all seven.
The untouched frontier is reported separately in the JSON. One fixture is not
broad accuracy or recall evidence. Both lossy experiments remain off by default.
