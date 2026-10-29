# Optional stored MXFP4 prefill adapter

The new opt-in fused FFN and its direct-packed GGUF layout are documented in
[`NATIVE_LAYOUT.md`](NATIVE_LAYOUT.md). Its loader and ABI are separate from
the historical canonical ABI4 adapter below. Full-model qualification remains
in progress; no production default changes follow from its component checks.

The separate opt-in [FP8 dense experiment](DENSE_FP8.md) keeps the model file
unchanged and explicitly labels Q8_0 requantization as a lossy speed proxy.

This adapter calls FlashInfer's pinned SM120a W4A8 grouped GEMM from the existing
GGML routed-expert operator. It is an experiment, off by default. Its source
manifest and Apache-2.0 licence are in `source/`. The adapter is MIT.

Use a separate pinned build environment with NVIDIA CUTLASS DSL 4.8.1 and Torch
2.11.0+cu128. Run `export_kernel.py --output <scratch>/aot` under the existing
host-memory guard (the measured export fit a 3 GiB cap). The export uses fake
descriptors and creates no CUDA context. From a Visual Studio x64 developer
shell, run `build.ps1` with the AOT directory, an output directory and the
installed CUTLASS `cuda_dialect_runtime_static.lib`. CUDA 13.2 is required.
Keep build, compiler caches and temporary files under the task's scratch root.

Build the normal producer with the modified GGML CUDA backend. In an experiment
config, enable `LLAMA_DSV41_STORED_MXFP4=1`,
`LLAMA_DSV41_FLASHINFER_MXFP4=1`, `GGML_CUDA_DISABLE_GRAPHS=1` and set
`LLAMA_DSV41_FLASHINFER_MXFP4_DLL` to the absolute bridge DLL path. Use
`examples/native-fp4/speed_proxy.py` through its shared GPU lease and explicit
physical-RAM/paging guard. It records the DLL hash and requires completed
FlashInfer dispatch witnesses for every expert graph. Missing/incorrect ABI,
unsupported shape and insufficient VRAM fail loudly.

For the batching experiment, set `LLAMA_DSV41_FLASHINFER_BATCH_TOKENS=8192`
(default `4096`). Rebuild both `ggml-cuda.cu` and `mmq.cu`, and the producer's
C++ sources, against ABI4. The encoder uses 8K expert chunks; attention and
router GEMMs retain their original 4K children. Layer20 and the short exact
cone suffix retain 4K execution to bound indexer scratch and preserve the
established cone graph. Wider mode reassigns 512 MiB of the declared provider
allowance to graph scratch; the exact provider heaps and combined device bill
still require admission. The overall device ceiling remains unchanged.
Use explicit host headroom: the recorded experiment used a 7.75 GiB process
tree ceiling, 1 GiB physical reserve, and producer base/max 7.25/7.75 GiB.
The adapter supports 16K calls for qualification; no 16K full-model performance
result is accepted. Oversized graphs fail admission before allocation.

Gather already performs MXFP8/E8M0 activation quantization directly from the
original F32 rows. This experiment reuses that fused kernel; it does not add
a separate gathered F32 activation matrix or claim a newly introduced fusion.
The corrected 8K experiment matches all eight full-vocabulary logit rows on
the recorded 100K fixture, but is slower overall and is not adopted. The
initial mismatch exposed a shared-Q8 paired causal-indexer offset bug; that
correctness fix applies to all paired graphs. See
[`sweep-batch-comparison.json`](../../../examples/native-fp4/sweep-batch-comparison.json).

The opt-in reserves 3328 MiB for the adapter and existing backend temporaries
inside the producer's existing device cap. The fixed graph arena shrinks by
the same amount; admission rejects a graph that no longer fits. ABI 4 takes
one caller-owned workspace, with offsets shared by the bridge and GGML. GGML
reuses its fixed maximum size across gate/up/down and aligns the suballocation
to 256 bytes. Admission includes padding, legacy-pool lookahead and module
headroom. The bridge performs no CUDA allocations and changes no default CUDA
memory-pool policy. At the decode transition, legacy pool storage is released;
VMM pools retain bounded mappings until backend destruction, and decode-cache
admission uses the remaining physical free memory.

Imported DirectStorage/D3D resources can make
CUDA's free-memory query count the same allocation twice, so admission uses
the existing physical device plan and still treats allocation errors as fatal.

The bridge supports 384 experts, six routes and the exact gate/up/down dimensions
5120↔2304, with at most 16384 source rows per adapter call. It rearranges canonical
17-byte MXFP4 weight blocks into FlashInfer's adjacent-nibble values and packed
E8M0 scales. Inactive experts are skipped and packed values use vector stores.
Weight values/scales are unchanged; this runtime layout work is
included in engine timings. Activations are quantized to E4M3 with an E8M0
scale per 32 values. FlashInfer accumulates in F32 and stores BF16; the adapter
scatters those values into the original F32 graph output. Activation rounding
and the BF16 epilogue require model quality checks. No precision-neutral speed
or default recommendation follows from a standalone GEMM result.

`bridge_check.cu` checks this complete integration chain against an independent
double/BF16 oracle on representable `hash32-v1` inputs and canonical stored
weights. Supply the bridge DLL, N, K and 384 little-endian int32 routing counts.
It permutes both gather and scatter, checks every output for finiteness and
compares samples across active and highly loaded experts. Run both projections
under compute-sanitizer and the exclusive GPU guard before model measurements.
Both projections passed the tail fixture under compute-sanitizer (zero errors)
and the most skewed observed 4K routing tile with exact sampled reference
results. The historical ABI 2 matched synthetic 100K run completed all 1632 expert products
but regressed throughput; the full timing and narrow quality comparison are in
[`primary-100k-comparison.json`](../../../examples/native-fp4/primary-100k-comparison.json).
Source-native conversion and broad model qualification remain in progress.
The source hashes, DLL identity and guarded checks are recorded in
[`flashinfer-adapter-validation.json`](../../../examples/native-fp4/flashinfer-adapter-validation.json).

`bridge_native_check.cu` additionally accepts the hash-verified proxy GGUF and
an absolute tensor data offset after the four ordinary checker arguments.
It reads the first/middle/last/hottest active expert matrices from that actual
stored tensor and compares their outputs with a separate scalar E2M1/E8M0
decoder and double/BF16 dot-product oracle. Gate, up and down each passed 36
sampled comparisons exactly, with full-output finiteness and zero sanitizer
errors. See [`flashinfer-native-weight-validation.json`](../../../examples/native-fp4/flashinfer-native-weight-validation.json).
These checks use representable synthetic activations; the full-model comparison
still determines the effect of activation rounding.

The historical ABI3 checks and 100K results are in [`flashinfer-workspace-validation.json`](../../../examples/native-fp4/flashinfer-workspace-validation.json) and [`primary-100k-workspace-comparison.json`](../../../examples/native-fp4/primary-100k-workspace-comparison.json). The earlier ABI2 checks apply to the implementation at commit `0c3361c7b978c58c62c06d7ad32ade72a7445832`; its DLL and receipts are preserved. ABI3 reduced this fixture’s prompt-only prefill to 40.734 seconds (43.532-second launch-to-ready TTFT) with all eight logit rows bit-identical to ABI2, but the lossy proxy toggle remains off.

`speed_proxy.py` records both `prompt_prefill_seconds`/`prompt_prefill_tok_s` from the native sequential clock and `ttft_seconds`/`ttft_tok_s` from launch through READY. Use prompt-only throughput against the reference repository. Historical `prefill_seconds`/`prefill_tok_s` remain explicit TTFT aliases.
