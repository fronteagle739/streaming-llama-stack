# Opt-in FP8 dense experiment

The separate dense bridge calls the pinned FlashInfer/CUTLASS SM120 warp-MMA
kernel, using MXFP8 E4M3 inputs and E8M0 block scales. It does not use SM100-only
tcgen05 kernels. `dense-source/manifest.json` binds the original source bytes
and BSD-3-Clause/Apache-2.0 licences. The local adapter is MIT.

`export_dense_fp8.py --output <scratch>/fp8-aot` exports seven shape-specialized
objects without creating a CUDA context. Its pure CuTe utility closure is
extracted unchanged from the pinned `.source` files; Torch-dependent helpers
and environment setup are not imported. Build `build_dense_fp8.ps1` against
these objects and the same pinned CUTLASS DSL runtime used by the expert bridge.
Use the explicit host-memory guard for export, compilation and GPU tests.

Input format0 reads ordinary Q8_0 weights. It requantizes them on the GPU to
E4M3 with a shared E8M0 exponent for each32x32 weight tile. This is **Q8_0 -> FP8
dense (lossy), a speed proxy for native FP8**. Weight conversion is included
in timings. The original GGUF is unchanged; no second large model is needed.
Input format1 accepts native E4M3 bytes `[N,K]`, then E8M0 bytes
`[N/32,K/32]`. It leaves all weight codes unchanged and expands/swizzles scales
into the kernel's128x4 SF layout. This native operand API is qualified by
component tests. The source converter keeps Q8_0 dense by default; opt in with
`--dense-format native-fp8` alongside the kernel-native expert options.

The native GGUF flag is
`deepseek41.dense_layout=fp8-e4m3-e8m0-32x32-v1`. It applies to each layer's
q_a, q_b, kv, output_b and three shared-expert matrices. Each retains a Q8_0
logical tensor/byte extent, with E4M3 `[N,K]` values, E8M0 `[N/32,K/32]` scales,
then zero padding to that extent. The flag makes this a distinct encoding.
Generic GGML and both canonical ds4 loaders refuse it. The streaming provider
marks those exact roots and requires the native FP8 kernel; sliced/canonical
uses and unsupported shapes fail before compute. The adapter zero-pads the
private activation/result arena to16 rows and copies only valid outputs, so
the2541-row cone never falls through to canonical Q8 arithmetic. The remaining tensors keep the reference
recipe, including Q8_0 output_a/head; this is not a fully native checkpoint.
Native-source full-model timing and quality remain pending the source download.

The adapter gathers/quantizes F32 activations with E8M0 per32 values, accumulates
in F32 and stores BF16 results exposed as F32. Only the explicit original-Q8
projection nodes whose sole consumer is their BF16 boundary are eligible.
Unsupported shapes retain the original arithmetic. Router, head and HC
projections are excluded. No accuracy-neutral or production-default claim is
made for the lossy proxy.

Set `LLAMA_DSV41_DENSE_FP8=1`, `LLAMA_DSV41_DENSE_FP8_DLL=<absolute DLL>` and
`LLAMA_DSV41_DENSE_BF16_STORE=1` in a prefill-only experiment. Keep the toggle
off otherwise. Rebuild MMQ and the producer together. Admission returns576MiB
from the existing graph allowance to backend temporaries, keeping total device
and host ceilings unchanged. The bridge allocates no CUDA buffers; the backend
reuses one fixed512MiB arena, including alignment, legacy-pool lookahead and
module headroom. The graph's completed-product counters must match the actual
FP8 launches. Model timings and quality must be checked separately.

`dense_fp8_check.cu` tests both Q8 and native operand formats against independent
double/BF16 sampled references, using representable values and varying32x32
scales. It initializes outputs to NaN and checks every output for finiteness.
Both128- and4096-row cases passed for all seven exported shapes. The128-row
checks passed compute-sanitizer with zero errors; their instrumented times are
not performance results. The4096-row checks record separate warm event timings
including activation/weight packing and the output conversion. They are
synthetic component results, not native-source or full-model quality evidence.
