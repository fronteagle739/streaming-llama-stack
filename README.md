# Streaming Llama Stack

Run frontier-size mixture-of-experts models on one desktop GPU. The stack streams
the weights from NVMe straight into an RTX 5090 with DirectStorage, so a 483 GiB
model runs on a machine with 32 GB of VRAM and 32 GB of system RAM. Add a
512 GB Mac Studio and the RTX does the long prefill while the Mac decodes.

It ships a llama.cpp fork with streaming prefill, the ds4 decode kernels for
DeepSeek V4.1 Flash, an OpenAI-compatible API, MCP tools and the llama.cpp web UI.

## Results

DeepSeek V4.1 Flash (Q4_K experts, Q8_0 everything else, 483 GiB) on one
RTX 5090 desktop with 32 GB of RAM:

| What | Result |
|---|---|
| Prefill, 48,520-token prompt | **1,163 tok/s** |
| Prefill with YOCO, 103,189-token prompt | **1,635 tok/s** |
| Prefill time, 1,000,000 tokens in one session | **30 min 8 s** |
| Prefill rate, same session | **553 tok/s** |
| Decode at Max speed, warm, 14,859-token prompt | **19–21 tok/s** |
| Decode at Max speed, warm, 43,327-token prompt | **18–20 tok/s** |
| Decode at Max quality (full model), warm | **6.0 tok/s** |
| Peak engine RAM during the one-million-token run | **5.7 GiB** |

"Warm" means after the first 100 generated tokens, once the expert cache has
filled. Averaged over the whole reply, including that warm-up, Max speed gives
12.7–12.9 tok/s for a 500-token answer. The 48K and 103K prefill rates include
engine startup.

Add a Mac Studio M3 Ultra (512 GB) for decode:

| What | Result |
|---|---|
| Prefill time, 198,287-token image prompt, YOCO | **118.8 s** |
| Prefill rate, same prompt | **1,670 tok/s** |
| Mac decode after that prompt | **31.0 tok/s** |

Other models on the same streaming engine and RTX 5090:

| Model | Best prefill | Prompt length |
|---|---:|---:|
| Step-3.5 Flash (Q8_0, 194 GB) | 2,294 tok/s | about 25K tokens |
| MiniMax M2.5 (Q3_K_XL, 100 GB) | 1,576 tok/s | 9K tokens |
| GLM-5 (Q4_K_XL, 400 GB) | 764 tok/s | 16K tokens |
| Kimi K2.5 (Q3_K_XL, 500 GB) | 652 tok/s | 16K tokens |

[Full numbers, with the conditions for each](docs/PERFORMANCE.md).

### Host-RAM expert tier

An experimental host-RAM expert tier was evaluated and is not in this release:
it gave no gain on this machine. Warm decode after the first 100 tokens, about
50K context, RTX 5090 and 32 GB RAM:

| Run | Max quality | Max speed |
|---|---:|---:|
| Tier off | 7.16 tok/s | 18.54 tok/s |
| 3.5 GiB, upload to GPU | 7.16 tok/s | 18.18 tok/s |
| 3.5 GiB, CPU compute | 6.86 tok/s | 15.87 tok/s |
| 10 GiB, upload to GPU | 6.49 tok/s | 15.23 tok/s |
| 10 GiB, CPU compute | 6.93 tok/s | 13.22 tok/s |
| 10 GiB, mix of both | 6.94 tok/s | 12.08 tok/s |

The speed settings already favour VRAM-resident experts, misses are spread
thinly, and at Max speed the GPU waits for the CPU. It is expected to matter
more on machines with much more RAM; that is untested and listed in the
[roadmap](docs/ROADMAP.md).

## How it works

**Streaming prefill.** Prefill reads each layer's weights once per pass and runs
the whole prompt chunk through that layer before moving on. The cost of reading
483 GiB is shared by every token in the pass, so longer prompts run faster per
token. DirectStorage reads the next layer while the GPU computes the current one.
DeepSeek V4.1 uses 49,152-token passes.

**Expert-cache decode.** Dense weights stay in VRAM. Routed experts live in a VRAM
cache, and misses are read from NVMe. After the first hundred or so tokens the
cache holds the experts the conversation uses, and decode speeds up.

**Quality/speed dial.** The faster settings prefer experts that are already in the
cache instead of fetching from disk. Max quality never does this: it is the full
model. You set the dial from the web UI, the API or MCP, separately for reasoning,
tool calls, answers and code. The model can set it itself.

**YOCO.** Optional approximate prefill from DeepSeek's
[DeepInfer](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md)
release, keeping the last 140 positions. Our streaming variant skips the reads and
compute of the decoder layers (21–39) on every pass except the last. It speeds up
long prefills by 1.26–1.45× and changes outputs slightly.
[Speed and quality](docs/ULTRAYOCO_QUALIFICATION.md).

**Two machines.** The RTX prefills the prompt and hands the finished state to the
Mac once. The Mac decodes. Short prompts and follow-ups (under 11,776 new tokens)
prefill on the Mac with no round trip. [Details](docs/TWO_MACHINE.md).

Streaming prefill, the RTX-to-Mac handoff and Mac decode are general and work
with several model families. The single-engine ds4-kernel decode on the RTX is the
one DeepSeek V4.1-specific part.

## Who can run this

| Mode | Hardware needed |
|---|---|
| Single RTX | Windows RTX 50-series desktop, 32 GB VRAM, 32 GB system RAM and fast DirectStorage NVMe; measured on an RTX 5090. |
| RTX + Mac | The same RTX machine plus an Apple Silicon Mac with enough unified memory for the model; measured on an M3 Ultra with 512 GB. |
| RTX + large-RAM box | The same RTX machine plus a Windows/Linux RAM-rich decode host sized for the whole GGUF and session; **untested example**, receiver gap documented [here](examples/dual-machine/dsv41-large-ram-untested/README.md). |

## Requirements

- **GPU:** RTX 50-series (compute capability 12.0). Measured on an RTX 5090, 32 GB.
- **Storage:** fast local NVMe with DirectStorage. Measured on a NVMe RAID-0 at
  about 25 GB/s. Storage speed sets prefill and cold decode speed.
- **System RAM:** 32 GB. The engine needs its 9.25 GiB budget plus a 512 MiB
  desktop reserve free, and stops cleanly if RAM drops below 0.5 GiB free.
- **Build tools:** Windows, Python 3.12, Visual Studio C++ Build Tools, CMake, Ninja,
  CUDA 13.2 and Node.js 24.
- **Optional Mac:** M3 Ultra with 512 GB, on the same network, with the same model file.

## Any-GGUF single engine

A native DeepSeek V4.1 Flash GGUF using supported tensor types:
Q4_K, IQ2_XXS/Q2_K or MXFP4 routed experts, with Q8_0, Q4_K or Q4_0 dense
matrices. The measured file uses Q4_K experts and Q8_0 dense matrices (483 GiB).
Our Q4_K/Q8_0 build is GPU-tested. IQ2_XXS/Q2_K and MXFP4 builds are
CPU-tested only. Public MXFP4 shards must first be merged with antirez's
`merge_deepseek41_shards.py` tool. “Any-GGUF” covers these supported V4.1 Flash
layouts and geometry, rather than arbitrary GGUF architectures.
[Layouts and assembly instructions](docs/DSV41_GGUF.md) and the
[conversion script](scripts/convert_dsv41_gguf.py) cover existing safetensors,
with a header-only dry run. The installer never downloads weights.

## Quickstart

```powershell
git clone --recurse-submodules https://github.com/fronteagle739/streaming-llama-stack.git
cd streaming-llama-stack
./slstack.ps1 install --all
./slstack.ps1 install --all --check
```

Create `my.local.json` from [config.example.json](config.example.json) with your
model path and GPU UUID. `*.local.*` files are ignored by Git.

```json
{
  "model": "<model_dir>/<deepseek-v4.1-flash>.gguf",
  "gpu_uuid": "<GPU-UUID>",
  "host_budget_gib": 9.25,
  "min_available_gib": 4,
  "runtime_min_free_gib": 0.5
}
```

```powershell
./slstack.ps1 --config my.local.json check
powershell -File scripts/start-dsv41-single-262k.ps1 -WebUI
```

Open the printed URL (default `http://127.0.0.1:8088`). The same port serves
`/v1/chat/completions`, `/v1/models`, `/health` and `/mcp`. The first check hashes
the model once. The first chat loads the engine.

Other presets (run one at a time; stop with the same script and `-Stop`):

```powershell
powershell -File scripts/start-dsv41-single-1m.ps1 -WebUI             # 1M context
powershell -File scripts/start-dsv41-single-262k-ultra-yoco.ps1 -WebUI # YOCO
powershell -File scripts/start-dsv41-single-1m-ultra-yoco.ps1 -WebUI   # 1M + YOCO
```

Each script accepts `-Config`, `-Model`, `-DryRun` and `-Stop`.

## Supported models

| Model | RTX streaming prefill | Decode |
|---|---|---|
| DeepSeek V4.1 Flash | Yes | RTX (single engine, ds4 kernels) or Mac (ds4) |
| DeepSeek V4 Flash, V2, V3 | Yes | Mac |
| Kimi K2.5 / K2.6 / K2.7 | Yes | Mac (MLX) |
| GLM-5 / 5.1 / 5.2 / 5.3 | Yes | Mac (MLX) |
| MiniMax M2.5 / M3 | Yes | Mac |
| Step-3.5 / 3.7 Flash | Yes | Mac |
| MiMo V2.6 Pro | — | Mac (ds4 with expert streaming and the quality/speed dial) |

The Mac can also serve other models through MLX-LM, MLX-VLM, mlx-serve and
llama-server. [Model notes and measured speeds](docs/SUPPORTED_MODELS.md).

## Two machines

Install on both. On the Mac:

```sh
./bin/slstack install --all
./bin/slstack mac --model '<model_dir>/<deepseek-v4.1-flash>.gguf' --host-budget-gib 400 --min-available-gib 0.5
```

It prints an endpoint, a pairing token and the Windows command:

```powershell
./slstack.ps1 rtx --config my.local.json --mac <mac-endpoint> --pair-token <printed-token> --webui
```

With YOCO, the 198,287-token image prompt prefills at **1,670 tok/s**
and Mac decode reaches **31.0 tok/s**. The matched 49,787-token image prompt
prefills at **1,477 tok/s** with YOCO and **1,197 tok/s** without it.
[Pairing, routing and measured speeds](docs/TWO_MACHINE.md).
Use the [ULTRA + YOCO paired preset](examples/dual-machine/dsv41-ultra-yoco/preset.json)
with the [example launchers](examples/dual-machine/README.md) to enable YOCO.

For a Windows/Linux second machine with enough RAM to hold the model, see the
[large-RAM ds4-win untested example](examples/dual-machine/dsv41-large-ram-untested/README.md).
It documents the missing mode3/YOCO CPU receiver; its launcher only previews the
configuration.

## Features

- **Web UI** with live prefill progress, decode speed, the quality/speed slider and
  per-phase settings. [Live controls](docs/LIVE_GENERATION.md)
- **Persistent sessions.** The conversation stays loaded between turns, and
  follow-ups only process new tokens. [Prefix reuse](docs/PREFIX_REUSE.md)
- **Images** in prompts, on one machine or two.
- **MCP** endpoint with the quality/speed controls and media tools.
- **Media generation (experimental):** Anima, Qwen-Image, Krea and H3 image/video
  from the chat. [Media setup](docs/MEDIA_SETUP_VERIFICATION.md)
- **LoRA (experimental)** for the native ds4 path, with frozen-prefix training.
  [Example](examples/train_jeff_lora/README.md)
- **Configuration reference:** [Variables](docs/VARIABLES.md),
  [walkthrough](examples/WALKTHROUGH.md), [roadmap](docs/ROADMAP.md).

## Limitations

- Windows kernels target compute capability 12.0 (RTX 50-series).
- The RTX path is Windows-only today because it uses DirectStorage.
- The single-engine DeepSeek V4.1 path accepts the [supported GGUF layouts](docs/DSV41_GGUF.md); split files require assembly first.
- The faster dial settings cost quality. Max speed introduced syntax errors in C++
  tests; use Max quality or Balanced for code.
- YOCO is approximate and opt-in.
- The first prompt includes engine startup, and decode is slower until the expert
  cache warms up (about 100 tokens).
- Media generation and LoRA are experimental.

## Credits and licence

MIT licence for the stack code ([LICENSE](LICENSE)). Embedded projects keep their
own licences; see [NOTICE](NOTICE).

- [DeepSeek](https://github.com/deepseek-ai) for the model, and
  [DeepInfer](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md)
  for YOCO. TensorSharp was a secondary reference.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) for the inference base and web UI.
- [antirez/ds4](https://github.com/antirez/ds4),
  [ds4-win-NovaStar](https://github.com/mishal-harbi/ds4-win-NovaStar) and
  [IngeniousIdiocy's V4.1 backend](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra)
  for the native kernels.
- [mlx-serve](https://github.com/ddalcu/mlx-serve) for the optional MLX service.
- [Microsoft DirectStorage](https://github.com/microsoft/DirectStorage) for
  storage-to-GPU transfers.
