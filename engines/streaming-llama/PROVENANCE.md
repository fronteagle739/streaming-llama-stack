# Streaming llama provenance

Derived from [ggml-org/llama.cpp at `b83111815e9a79949257e9d4b087206b320a3063`](https://github.com/ggml-org/llama.cpp/commit/b83111815e9a79949257e9d4b087206b320a3063), MIT. This reconstructed import base uniquely matches 2,202 source blobs with eight modifications, three omissions and four additions among 58 upstream revisions in the import interval; the import did not retain an upstream parent. Later selected upstream kernels and architectures are also included.

Changes: DirectStorage weight streaming, Flash V4.1 prefill and native session transport, single-process ds4 decode integration, locked carry banks, portable builds, memory admission and runtime guards. Retained source includes the server UI and LoRA exporter. See the original LICENSE and AUTHORS in `prefill_llama.cpp`.
