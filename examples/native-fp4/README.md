# Standalone native-format kernel experiment

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
the package's complete fused MoE path. FlashInfer is Apache-2.0; no kernel
source is copied into the engine by this experiment.

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
