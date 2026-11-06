# Streaming Llama Stack

Stream large mixture-of-experts models from NVMe into a desktop GPU. The stack
contains a llama.cpp streaming prefill engine, packed expert CUDA kernels,
Metal decode, and an OpenAI-compatible API. Use one RTX, one Mac, or an RTX for
long prefill with a Mac for decode. Qwen3.8 Flash Next (QFN) and DeepSeek V4.1
Flash have dedicated recipes. This software is experimental.

## Requirements

- Windows with an RTX 5090/SM120 GPU, CUDA 13+, compatible MSVC, CMake/Ninja,
  DirectStorage SDK, and fast local NVMe for streamed models.
- Python 3.12 with [runtime requirements](requirements.txt). Conversion also
  uses [pack requirements](requirements-pack.txt).
- For paired decode, an Apple Silicon Mac with enough unified memory for the
  chosen model, Xcode command-line tools, CMake, Python, and configured SSH.
  The QFN measurements below use a 512 GB M3 Ultra Mac Studio.
- Explicit physical-memory and commit budgets. Every heavy job runs through
  the included RAM guard and refuses work when the available-memory floor is
  exceeded. Disk spill is not a substitute for physical RAM.

## Install, build and run

Run these commands from the checkout root. Put environments, models, build
outputs and caches on a fast volume with sufficient free space.

```text
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-pack.txt
python -B -m slstack --help
python -B scripts/qfn_pinned_build.py --help
```

QFN's CMake target is `qfn-engine`, in
`engines/streaming-llama/prefill_llama.cpp`, with `LLAMA_QFN_ENGINE=ON` and
`GGML_CUDA=ON`. Supply the JSON, DirectStorage and packed-kernel dependencies
described in [QFN setup](docs/QFN.md). Use the official
[Microsoft.Direct3D.DirectStorage 1.3.0 NuGet package](https://www.nuget.org/packages/Microsoft.Direct3D.DirectStorage/1.3.0),
with the version pinned explicitly (using the NuGet CLI):

```powershell
nuget install Microsoft.Direct3D.DirectStorage -Version 1.3.0 -Source https://api.nuget.org/v3/index.json -OutputDirectory dependencies
```

From an x64 Native Tools console, pass the extracted package root through the
CMake flag `-DDS_SDK_DIR`; it must contain `native/include/dstorage.h`,
`native/lib/x64/dstorage.lib` and `native/bin/x64` runtime DLLs. Headers alone
cannot complete the link. Run the following commands under the RAM guard with
explicit physical-memory, commit and available-memory budgets:

```powershell
cmake -S engines/streaming-llama/prefill_llama.cpp -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_QFN_ENGINE=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DDS_SDK_DIR="$((Resolve-Path dependencies/Microsoft.Direct3D.DirectStorage.1.3.0).Path)" -DOPENSSL_ROOT_DIR=OPENSSL_ROOT
cmake --build build-native --target qfn-engine --parallel 2
Copy-Item build-native/bin/*.dll bin/
Copy-Item dependencies/Microsoft.Direct3D.DirectStorage.1.3.0/native/bin/x64/*.dll bin/
bin/qfn-engine.exe --model MODEL.slpack --host-budget-gib 6 --ctx 100000 --validate-options
```

Keep the CUDA runtime and cuBLAS DLLs available through the CUDA installation's
`bin` directory on `PATH`. Replace `OPENSSL_ROOT` with an OpenSSL 3 development
installation, and keep its runtime DLLs available on `PATH` as well. The native
option check parses the settings without loading a model or starting a listener.
Build the Mac `ds4-server` target with
`DS4_NATIVE_FRONTIER_ENABLE=1` for paired serving. Keep one stable executable
path and its companion libraries for each engine. The guarded build wrapper
records all source and link inputs and detects drift before serving.

Copy [the QFN config](examples/single-machine/qfn.example.json), fill the model,
engine, vision companion, GPU and local paths, then validate the selected mode:

```powershell
powershell -File scripts/start-qfn-single.ps1 -Config LOCAL.json -Mode max_quality -DryRun
powershell -File scripts/start-qfn-single.ps1 -Config LOCAL.json -Mode max_speed -DryRun
```

Remove `-DryRun` to serve on loopback. `-MaxSpeedBias 0.35` with a matching local
model-settings file selects the intermediate routing bias. Every sampler
remains available, including temperature, top-k, top-p, min-p, typical,
repetition/presence/frequency penalties, seed, stop strings, logit bias and
token limits. Reasoning, tools, streaming, prefix reuse, cancellation and
guarded image encoding are supported. Advanced samplers can use serial decode.
MTP and the host-RAM expert tier remain selectable.

DeepSeek uses [single-machine recipes](examples/single-machine/README.md),
or the [paired setup](docs/TWO_MACHINE.md). The generic GGUF route,
streamed media, MCP tools and web UI remain documented in
[backend setup](docs/BACKEND_SETUP.md), [media](docs/STACK_MEDIA_AND_STATE.md)
and [supported models](docs/SUPPORTED_MODELS.md).

## Model preparation

Models are supplied separately. Recipes pin source revisions and expected
input hashes. No launcher downloads weights implicitly.

| Model | Recipe | Format and commands |
| --- | --- | --- |
| QFN | [NVIDIA NVFP4](recipes/qfn-nvfp4.json) plus [compatible auxiliary sections](recipes/qfn-auxiliary.json) | [Streaming Llama Pack](docs/SLPACK_RECIPES.md) |
| DeepSeek V4.1 Flash | [Pinned checkpoint](recipes/dsv41-flash.json) | [GGUF conversion](docs/DSV41_GGUF.md) |

QFN expert relayout preserves raw NVFP4 expert values/scales. The complete pack
also contains quantized dense/PLE tensors and auxiliary MTP, vision, tokenizer
and sampler metadata; it is not a lossless full-checkpoint conversion. Keep
the complete companion set and validate its receipts before serving.
Both compact Mac Q4_0 and Q8_0 conversion recipes ship with
[QFN paired setup](examples/dual-machine/qfn/README.md).

## Performance and quality

Single-machine hardware: RTX 5090 (32 GB VRAM), 32 GB host RAM and local
DirectStorage NVMe. Temperature 0, MTP off, eight read parts. Rates are tokens/s.
Decode discards 100 evaluations and measures 128. KL is in nats over the full
vocabulary at the same forced continuation; top-1 is next-token agreement.

### QFN, one RTX

KL compares each routing bias with unbiased RTX QFN, using the same quantized
pack and 16-bit KV. Quality uses 256 computed rows. It does not compare with
the original full-precision weights. The maximum speed preset uses bias 0.68;
0.35 remains an explicit numeric choice.

| Prompt tokens | Bias | Prefill | Decode | KL | Top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0 | 4,440 | 25.25 | 0 | 100.0% |
| 100,000 | 0.35 | 4,440 | 46.25 | 0.0372 | 92.6% |
| 100,000 | 0.68 | 4,150 | 50.12 | 0.137 | 88.3% |

### DeepSeek V4.1, packed FP4 speed proxy

KL compares each bias with bias 0 on the same packed FP4 proxy, over 128
predictive rows. This is not equivalence to the original DeepSeek weights.
The 100K and 199,952-token prompts use different real-text fixtures. Prefill
counts embedding, all 40 layers and the head; model loading is excluded.

| Prompt tokens | Bias | Prefill | Decode | KL | Top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0 | 3,742 | 7.73 | 0 | 100.0% |
| 100,000 | 0.35 | 3,635 | 12.38 | 0.0526 | 94.5% |
| 100,000 | 1.36125 | 3,587 | 27.06 | 0.475 | 80.5% |
| 199,952 | 0 | 4,670 | 7.57 | 0 | 100.0% |
| 199,952 | 0.35 | 4,538 | 12.67 | 0.0710 | 93.8% |
| 199,952 | 1.36125 | 4,597 | 21.23 | 0.401 | 79.7% |

### QFN, RTX prefill and Mac decode

Hardware: RTX 5090 (32 GB VRAM), 32 GB host RAM, local DirectStorage NVMe,
and an M3 Ultra Mac Studio with 512 GB unified memory on a 10 Gb/s private link.
Temperature 1, seed 1, top-k 20, top-p 0.95, min-p 0, 100,000 prompt tokens,
1,024 output tokens, 16-bit KV, direct TCP and automatic MTP (actual depth 2).
Rows are medians of two fresh RTX processes after a short, different warm-up.
First-token time is measured on the resident pair, excluding cold model loading.

| Mac experts | RTX prefill tok/s | Handover s | First token s | Decode tok/s | MTP accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_0 | 4,532 | 4.4 | 27.3 | 41.0 | 67.5% |
| Q8_0 | 4,498 | 5.0 | 27.8 | 39.7 | 68.9% |

Quality uses 256 computed full-vocabulary rows, excluding the imported frontier,
with KL(unbiased RTX QFN || Mac). The RTX reference uses NVFP4/native-Q8
arithmetic and 16-bit KV. Q4: KL 0.0073, top-1 97.7%; Q8: KL 0.0020,
top-1 98.8%. These are not near-exact results and are not a comparison with
original full-precision weights. Q4 and Q8 can be mixed independently with
the RTX NVFP4 prefill pack; the measurements do not establish a format ranking.

Reproduction commands: [QFN single-machine](docs/QFN_100K_REPRO.md),
[routing quality](examples/single-machine/CALIBRATION.md),
[DeepSeek tables](examples/single-machine/DSV41_BENCHMARK.md), and
[QFN pair](examples/dual-machine/qfn/README.md#reproduce-the-tables).

## Limits and security

The pair supports session reuse, bidirectional deltas, rewind, ordered catch-up,
checkpoint acknowledgement and visible peer-failure recovery. Images use lazy
RTX encoding and prefill. Context, transfer size and host memory are bounded;
overflow refuses clearly without truncation. Q6_K full-model Mac qualification
is not established. DFlash needs a compatible assistant model and CUDA support.
GLM is not supported yet.

APIs and Windows test listeners bind to 127.0.0.1. Direct state TCP is intended
for trusted private links: state and pairing tokens are not encrypted, and each
payload is not authenticated against an active attacker. Select SSH transport
for encryption and transit integrity. No firewall or security-setting changes
are made. Keep generated configuration, tokens and run receipts private.

The software and model components retain their respective upstream licences.
See [LICENSE](LICENSE), vendored notices and source provenance manifests.
