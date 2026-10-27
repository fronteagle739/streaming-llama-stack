# Model routes and existing measurements

Streaming-llama is a general llama.cpp fork with DirectStorage weight reads,
layer-major prefill and model-specific graph/cache integrations. RTX-to-Mac
handoff and Mac decode are also general mechanisms. The combined V4.1 producer
adds a specialized ds4-kernel decode path. A graph registration, launch mapping,
saved preset and measured end-to-end run provide different levels of evidence.

## Retained RTX streaming families

The streaming launch mappings are in
[`general_launch.py`](../vendor/studio/studio/general_launch.py); the retained
graph registry is in
[`llama-arch.cpp`](../engines/streaming-llama/prefill_llama.cpp/src/llama-arch.cpp).
Each checkpoint still needs matching tensor, cache and receiver implementations.
This table lists the identified streaming integrations, rather than treating
every llama.cpp graph registration as a tested streaming model.

| Family | Integration evidence | Existing measurement |
| --- | --- | --- |
| DeepSeek V2/V3 (`deepseek2`) | Streaming MLA launch mapping and graph | No comparable measurement established in this review |
| DeepSeek V4 Flash (`deepseek4`) | Streaming mapping, graph and Studio paired/local presets | The V4.1 figures are not V4 measurements |
| DeepSeek V4.1 Flash (`deepseek41`) | Streaming mapping and graph; combined producer and native state handoff | [Performance](PERFORMANCE.md), including measured plain ULTRA two-machine operation |
| Kimi K2.6/K2.7 (`deepseek2`, `kimi` launch mapping) | Studio RTX/MLX paired presets; retained MLA handoff | K2.7 about 8,190-token A/B: 50.27 vs 49.55 prefill tok/s, limited by approximately 3.7 GB/s SSD reads; not a fast-NVMe ceiling |
| Full GLM-5.1/5.2/5.3 (`glm-dsa`) | Streaming DSA mapping, graph, MLX receiver and Studio presets | GLM-5.1: 414.8 prefill tok/s at 15,616 tokens. GLM-5.2: 389.9/418.6 prefill at 11,780/23,742 tokens; paired Mac decode 12.58/11.8 tok/s, but needle recall failed. GLM-5.3: separate Mac decode 14.0738 short / 4.89483 at 11,550 prompt tokens; see [backend measurements](BACKEND_SOURCE_AUDIT.md) |
| MiniMax M2/M3 (`minimax-m2`, `minimax-m3`) | Streaming GQA launch mappings and retained graphs; M3 paired Studio preset | No comparable end-to-end measurement established in this review |
| Step-3.5/3.7 (`step35`) | Retained graph; Studio Step-3.7 local streaming preset | Source/preset evidence; no comparable measurement established in this review |

GLM-5.3 **Flash** uses `glm5-next`, not `glm-dsa`. The general launch aliases
exist, but this retained graph registry lacks `glm5-next` and `qwen4exp` for
Qwen3.8 Flash Next. Separate experimental ports have CPU tiny-model acceptance
receipts; those do not establish retained RTX CUDA streaming support.
Qwen3.8 Flash Next can use the independent Mac mlx-serve HTTP route. Other
Qwen, Gemma and MiMo MLX/llama graph or preset entries alone are not proof of
an RTX streaming handoff in this tree.

## Launcher and Mac interfaces

[`general_engine.py`](../slstack/general_engine.py) constructs general streaming
commands, reads GGUF architecture/layer metadata and rejects a missing graph.
Use `general.mode=streaming` with a compatible `general.rtx` configuration and
`general.mac.backend=mlx-receiver`, `llama-decode` or `llama-decode-upstream`.
The current general producer supervisor still applies its GPU reservation
lifetime to the process. Request-scoped reservation handling is needed before
continuous general serving can be advertised. These are retained integrations,
not a claim that every row has passed a fresh installation/model-load test.

Mac MLX receivers cover multiple model families; the native ds4 session broker
is a separate route. MLX-serve, MLX-LM, MLX-VLM and llama-server also provide
HTTP decoding for their own supported models. HTTP backends do their own prefill
unless they implement a compatible state import. See
[backend interfaces and setup](BACKEND_SOURCE_AUDIT.md) for launch configuration,
optional source dependencies and historical measurements.

## Combined V4.1 producer and quantization

The combined single-engine producer checks `general.architecture=deepseek41`,
the 40-layer Flash dimensions, KV/index sharing, Engram metadata and tensor
layouts in
[`llama-dsv41-weights.cpp`](../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-weights.cpp).
Its loader requires Q4_K routed gate/up/down tensors, Q8_0 attention/shared/head
matrices, F16 embeddings and selected projections, F32 normalization/router
tensors and the native Engram byte layout. Its
[`main.cpp`](../engines/streaming-llama/prefill_llama.cpp/tools/dsv41-prefill/main.cpp)
also requires the measured 1017 Q4 receipt and SHA-256. These are current code
checks, rather than a general restriction on streaming-llama or all ds4 builds.

The broader ds4 CUDA decoder has IQ2_XXS gate/up with Q2_K down, Q4_K routed
expert dispatch, and MXFP4 expert dispatch through its MMQ backend in
[`ds4_cuda.cu`](../engines/ds4-win/ds4_cuda.cu). Its model loader validates
architecture-specific geometry in [`ds4.c`](../engines/ds4-win/ds4.c).
The public [antirez V4.1 Flash files](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/tree/main)
include IQ2_XXS/Q2_K, MXFP4-expert shards and Q4 parts. Kernel coverage does not
establish that these files satisfy the combined producer's current loader,
receipt, metadata and state-pairing checks. No new artifact was downloaded or
qualified for this documentation change. The exact measured Q4 artifact still
needs a verified public source or conversion recipe.
