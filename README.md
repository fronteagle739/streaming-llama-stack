# Streaming Llama Stack

Run frontier-size mixture-of-experts models on one desktop GPU. The stack streams
the weights from NVMe straight into an RTX 5090 with DirectStorage, so a 483 GiB
model runs on a machine with 32 GB of VRAM and 32 GB of system RAM. Add a
512 GB Mac Studio and the RTX does the long prefill while the Mac decodes.

It ships a llama.cpp fork with streaming prefill, the ds4 decode kernels for
DeepSeek V4.1 Flash, an OpenAI-compatible API, MCP tools and the llama.cpp web UI.

## Results

DeepSeek V4.1 Flash on one RTX 5090 desktop with 32 GB of RAM.
**FP4 (tensor-core-native layout)** measures **26.3 s prompt-only** at 100K,
about **3,800 tok/s** (16.9 GB/s storage; 29.6 s before the round-6 host and
HC-fusion steps), now in one weight pass by default (OPUS22 round 6: the
compact host carry record and an all-rows VRAM hidden bank; logits bit-identical
to the two-sweep plan). The earlier two-sweep FP4 run measured 32.4 s. The matched eager control for the Q4 shipping-default
dense path measures **32.6 s**. The Q4_K/Q8_0 default model is 483 GiB.

| Prompt tokens | Configuration | Timing | Time | Rate |
|---:|---|---|---:|---:|
| 100,000 | FP4 (tensor-core-native layout), single weight pass (default), eager | Prompt-only | **26.307 s** | **3,801.33 tok/s** |
| 100,000 | FP4 (tensor-core-native layout), single weight pass (default), eager | TTFT | **29.406 s** | — |
| 100,000 | FP4 (tensor-core-native layout), CUDA graphs, two sweeps | Prompt-only | 32.374 s | 3,088.92 tok/s |
| 100,000 | FP4 (tensor-core-native layout), CUDA graphs, two sweeps | TTFT | 35.406 s | — |
| 100,000 | Q4 shipping-default dense path, matched eager control | Prompt-only | 32.616 s | 3,066.01 tok/s |
| 100,000 | Q4 shipping-default dense path, matched eager control | TTFT | 35.797 s | — |
| 100,000 | Earlier Q4_K/Q8_0 default, KERNELS2 build | Prompt-only | 38.368 s | 2,606.32 tok/s |
| 100,000 | Earlier Q4_K/Q8_0 default, KERNELS2 build | TTFT | 42.000 s | — |
| 50,000 | Earlier exact-cone defaults | Startup + prefill | 24.203 s | 2,065.86 tok/s |
| 200,000 | Earlier exact-cone defaults | Startup + prefill | 92.407 s | 2,164.34 tok/s |
| 1,000,000 | Earlier approximate YOCO140 defaults | Startup + prefill | 956.734 s | 1,045.22 tok/s |

The current kernel stack combines the exact dependency cone, certified FP4
index scoring, fused Sinkhorn, TileLang attention, E18 transfer staging and
allocation-plan reuse. Packed FP4 weights are an available prefill option;
CUDA graphs remain opt-in. [Timing, builds and measurements](docs/PERFORMANCE.md#prefill).

Latest calibrated Q4 decode controls at 49,555 prompt tokens, with the first
100 evaluated steps discarded and the next 899 timed:

| Setting | Warm decode |
|---|---:|
| Max quality | **6.14 tok/s** |
| Max speed | **20.66 tok/s** |

[Decode controls, E8 residency and E10 graph islands](docs/PERFORMANCE.md#decode).

Add a Mac Studio M3 Ultra (512 GB) for decode:

| Prompt tokens | Prefill mode | WebUI prompt interval | Mac decode |
|---:|---|---:|---:|
| 105,054 | Exact cone, state v2 | **59.005 s** | **27.50 tok/s** |
| 198,287 | Earlier approximate YOCO140, image | 118.8 s | 31.0 tok/s |

[Paired measurements and timing scope](docs/PERFORMANCE.md#rtx-prefill-mac-decode).

Other models on the same streaming engine and RTX 5090:

| Model | Best prefill | Prompt length |
|---|---:|---:|
| Step-3.5 Flash (Q8_0, 194 GB) | 2,294 tok/s | about 25K tokens |
| MiniMax M2.5 (Q3_K_XL, 100 GB) | 1,576 tok/s | 9K tokens |
| GLM-5 (Q4_K_XL, 400 GB) | 764 tok/s | 16K tokens |
| Kimi K2.5 (Q3_K_XL, 500 GB) | 652 tok/s | 16K tokens |

[Full numbers, with the conditions for each](docs/PERFORMANCE.md).

### Host-RAM expert tier

The separate host-RAM tier experiment remains off by default and is not in
this release. Latest accepted Q4 measurements use a 49,553-token prompt and
reused-prefix follow-up; warm decode times 899 evaluated steps after discarding
the first 100:

| Run | RAM setting | Max quality | Max speed |
|---|---:|---:|---:|
| Tier off | 0 GiB | 6.87 tok/s | 18.26 tok/s |
| Upload to GPU | 3.49 GiB | 6.79 tok/s | 17.34 tok/s |
| CPU compute | 3.49 GiB | 6.81 tok/s | 14.58 tok/s |
| Mix of both | 3.49 GiB | 6.84 tok/s | 15.80 tok/s |
| Upload to GPU | 10 GiB | 6.49 tok/s | 15.23 tok/s |
| CPU compute | 10 GiB | 6.93 tok/s | 13.22 tok/s |
| Mix of both | 10 GiB | 6.94 tok/s | 12.08 tok/s |

[Round 3 results](https://github.com/fronteagle739/streaming-llama-stack/blob/a2371b6e84a09c8ef5f64ab67fc5230fc24aa95e/docs/RAMTIER_R3.md).

The speed settings already favour VRAM-resident experts, misses are spread
thinly, and at Max speed the GPU waits for the CPU. It is expected to matter
more on machines with much more RAM; that is untested and listed in the
[roadmap](docs/ROADMAP.md).

## How it works

**Streaming prefill.** Prefill reads each layer's weights once per pass and runs
the whole prompt chunk through that layer before moving on. The cost of reading
the weights is shared by every token in the pass. A complete DeepSeek V4.1 Q4
pass reads 315.8 GB; Engram rows are gathered separately. DirectStorage reads the
next layer while the GPU computes the current one. DeepSeek V4.1 uses
49,152-token passes, with optional RAM-admitted chunk grouping.
[Routed reads, the exact replay cone and byte accounting](docs/MULTICHUNK_PREFILL.md).

**Expert-cache decode.** Dense weights stay in VRAM. Routed experts live in a VRAM
cache, and misses are read from NVMe. After the first hundred or so tokens the
cache holds the experts the conversation uses, and decode speeds up.

**Quality/speed dial.** The faster settings prefer experts that are already in the
cache instead of fetching from disk. Max quality never does this: it is the full
model. You set the dial from the web UI, the API or MCP, separately for reasoning,
tool calls, answers and code. The model can set it itself.

**Exact cone.** Single-machine and paired ULTRA defaults compute every encoder row
and the decoder's final 2,541-row dependency cone. Aligned replay preserves
full-prefill arithmetic; the matched 8K and 49,555-token controls are exact.
Paired mode negotiates state v2 and retains the mode3 export/import payload,
owner hashes and compatibility checks. [Design](docs/MULTICHUNK_PREFILL.md) ·
[Paired state and live proof](docs/CONE_STATE.md).

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
Our Q4_K/Q8_0 build is GPU-tested; IQ2_XXS/Q2_K remains CPU-tested.
MXFP4 has measured GPU prefill results with a separate tensor-core-native
packed layout. An experimental dense FP8 path is also available.
[FP4/FP8 layouts and measurements](docs/PERFORMANCE.md#fp4-and-experimental-fp8). Public MXFP4 shards must first be merged with antirez's
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
powershell -File scripts/start-dsv41-single-1m.ps1 -WebUI # 1M context, exact cone
powershell -File scripts/start-dsv41-single-262k-ultra-yoco.ps1 -ApproximateYoco -WebUI
powershell -File scripts/start-dsv41-single-1m-ultra-yoco.ps1 -ApproximateYoco -WebUI
```

Each script accepts `-Config`, `-Model`, `-DryRun` and `-Stop`.
The legacy `ultra-yoco` filenames also default to exact cone; `-ApproximateYoco`
selects approximate replay. Paired ULTRA presets also default to exact cone
with negotiated state v2; approximate YOCO140/v1 remains explicit opt-in. [Defaults and optional experiments](docs/MERGEPERF.md).

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

The current exact-cone WebUI proof processes **105,054 tokens** in a
**59.005 s** prompt interval; initial Mac decode measures **27.50 tok/s**.
The same prompt on one RTX takes **46.282 s**. Historical image-prompt results
remain in the [paired measurements](docs/TWO_MACHINE.md).
Use the [ULTRA exact-cone paired preset](examples/dual-machine/dsv41-ultra-cone/preset.json)
with the [example launchers](examples/dual-machine/README.md).
[Cone state protocol](docs/CONE_STATE.md).

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
- Approximate YOCO140 is opt-in; the default dependency cone keeps 2,541 rows.
- The measured FP4 file was requantised from Q4; native-precision FP4 conversion from the official weights is coming, with the same layout and expected speed.
- The matched eager control uses the Q4 shipping-default Q8 dense path with
  stored FP4 experts; the earlier KERNELS2 row measures the full Q4_K/Q8_0 file.
- The latest 100K FP4 run uses experimental prefill graphs and does not establish
  a repeatable graph speedup. No 50K or 200K run on that same final build is recorded.
- The 1M result and the 49,555-token warm-decode controls use earlier YOCO140
  prefill defaults. Warm decode has not been remeasured on the final OTHER22 build.
- FP4 prefill timing does not qualify decode or broad recall quality. Dense FP8
  remains experimental and has no demonstrated prompt gain.
- TileLang attention is the selected default but is numerically different from
  native attention. The tested retrieval answer matches; later forced continuations
  can diverge. Set `LLAMA_DSV41_TILELANG_ATTENTION=0` for native attention.
  [Quality measurements](examples/kernels2/PERFORMANCE.md).
- The first prompt includes engine startup, and decode is slower until the expert
  cache warms up (about 100 tokens).
- The separate host-RAM tier has no consistent gain across dial settings.
  CPU/mix continuations can differ and fail strict sequence-equivalence checks.
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

<sub>Useful to you? [Ko-fi](https://ko-fi.com/FrontEagle).</sub>
