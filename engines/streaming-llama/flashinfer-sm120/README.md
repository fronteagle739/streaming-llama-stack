# Optional stored MXFP4 prefill adapter

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

The opt-in reserves 3328 MiB for the adapter and existing backend temporaries
inside the producer's existing device cap. The fixed graph arena shrinks by
the same amount; admission rejects a graph that no longer fits. ABI 2 takes
this explicit workspace bound. Imported DirectStorage/D3D resources can make
CUDA's free-memory query count the same allocation twice, so admission uses
the existing physical device plan and still treats allocation errors as fatal.

The bridge supports 384 experts, six routes and the exact gate/up/down dimensions
5120↔2304, with at most 4096 source rows per graph. It rearranges canonical
17-byte MXFP4 weight blocks into FlashInfer's adjacent-nibble values and packed
E8M0 scales. Weight values/scales are unchanged; this runtime layout work is
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
results. Full model qualification is still in progress.
The source hashes, DLL identity and guarded checks are recorded in
[`flashinfer-adapter-validation.json`](../../../examples/native-fp4/flashinfer-adapter-validation.json).
