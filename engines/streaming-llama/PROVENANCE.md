# Streaming llama provenance

Derived from [ggml-org/llama.cpp at `b83111815e9a79949257e9d4b087206b320a3063`](https://github.com/ggml-org/llama.cpp/commit/b83111815e9a79949257e9d4b087206b320a3063), MIT. This reconstructed import base uniquely matches 2,202 source blobs with eight modifications, three omissions and four additions among 58 upstream revisions in the import interval; the import did not retain an upstream parent. Later selected upstream kernels and architectures are also included.

Changes: DirectStorage weight streaming, Flash V4.1 prefill and native session transport, single-process ds4 decode integration, locked carry banks, portable builds, memory admission and runtime guards. Retained source includes the server UI and LoRA exporter. See the original LICENSE and AUTHORS in `prefill_llama.cpp`.

The optional `flashinfer-sm120/source` directory contains five unmodified files
from [FlashInfer at `1d622fdb1b053fc532069914bc9a6b498bd5f722`](https://github.com/flashinfer-ai/flashinfer/tree/1d622fdb1b053fc532069914bc9a6b498bd5f722/flashinfer/fused_moe/cute_dsl/blackwell_sm12x),
Apache-2.0. Its manifest records the original paths and SHA-256 hashes; its
LICENSE is retained. The engine adapter and AOT export script are local MIT
changes. The external NVIDIA CUTLASS DSL compiler/runtime remains a separately
installed dependency under its own licence and is not vendored here.

ABI4 extends the same pinned FlashInfer adapter to bounded wider expert chunks.
The gather still quantizes directly from original F32 rows into MXFP8/E8M0;
no new quantization formula or separately materialized gather is introduced.
The graph preserves 4K attention/router products and limits layer20/replay
execution to4K. All paired causal indexer tags now recover the child query
offset, including the shared Q8_1/ULTRA path. See the native-fp4 sweep report
for kernel oracles, full-model numerical checks and measured limitations.
