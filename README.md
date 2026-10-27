# Streaming Llama Stack

Stream model weights from NVMe through DirectStorage for RTX prefill, then decode locally or hand the session to a Mac. The streaming-llama engine (a llama.cpp fork), session handoff and Mac decoder integrations serve multiple model families. The combined streaming-prefill/ds4-kernel decode producer is specific to DeepSeek-V4.1 Flash. An OpenAI-compatible API and the llama.cpp web UI sit above these engines.

The measurements below use a roughly 483 GiB DeepSeek-V4.1 Flash GGUF on an RTX 5090 in a desktop with 32 GB of system RAM, with an optional M3 Ultra for decode. This path admits approximately 1M tokens within a 9.25 GiB host-RAM budget; the two-machine session handoff has been used through Studio for months.

## Supported models

Streaming support belongs to the selected engine and model architecture, rather than to one model filename. The retained general launcher can construct non-DeepSeek streaming commands; its producer reservation lifetime still needs request-scoped handling before continuous general serving can be advertised. See [model routes, existing measurements and launcher boundaries](docs/SUPPORTED_MODELS.md).

| Model family | Retained route | Existing evidence |
|---|---|---|
| DeepSeek V4 / V4.1 Flash | Streaming-llama prefill; paired native/MLX decode; V4.1 combined producer | V4.1 measurements below; combined producer requirements below |
| Kimi K2.6 / K2.7 | Streaming-llama prefill and MLX receiver handoff | K2.7 about 8,190 prompt tokens: 50.27 / 49.55 prefill tok/s in an SSD-bound A/B |
| Full GLM-5.1 / 5.2 / 5.3 (`glm-dsa`) | Streaming-llama prefill and Mac MLX receiver | Historical GLM-5.1 prefill 414.8 tok/s at 15,616 tokens; separate GLM-5.3 Mac decode 14.074 tok/s on a short prompt |
| DeepSeek V2/V3; MiniMax M2/M3 | Retained streaming launch mappings and graph implementations | Source integration; no comparable end-to-end measurement established here |
| Step-3.5 / 3.7 (`step35`) | Retained graph; Studio local streaming preset | Preset/source evidence; no comparable measurement established here |

Mac decode also has MLX-VLM, MLX-LM, mlx-serve and llama-server routes for their supported models. An HTTP decoder performs its own prefill unless it implements a compatible state import; an HTTP connection alone is not an RTX KV handoff. GLM-5.3 Flash and Qwen3.8 Flash Next are distinct architectures from full GLM DSA and earlier Qwen families; see the linked model notes for their current source/port status.

## Measured results

Existing measurements on the Flash 1017 Q4 model; no new benchmark runs. Prefill, decode and request wall times have different denominators. Context below means actual prompt tokens unless a configured window is specified. Details and measurement limits are in [Performance](docs/PERFORMANCE.md).

| Configuration | Context | Measured result | Hardware |
|---|---:|---|---|
| ULTRA, fold planner | 28,241 / 32,234 / 48,520 | 985 / 1,036 / 1,163 prefill tok/s | RTX 5090, NVMe, Windows |
| ULTRA, exact → approximate YOCO140 | 43,327 | 1,063.7 → 1,373.4 prefill tok/s; **1.29×** | RTX 5090, NVMe |
| ULTRA, exact → approximate YOCO140 | 103,189 | 1,127.2 → 1,634.5 prefill tok/s (≈**1,635**); **1.45×** | RTX 5090, NVMe |
| Single engine, bias off / Max speed | 14,859 | 6.03 / 12.93 decode tok/s overall | RTX 5090, NVMe |
| Single engine, bias off / Max speed | 43,327 | 5.97 / 12.65 decode tok/s overall | RTX 5090, NVMe |
| Single engine, bias off, 999,424-token window | 237,703 | Correct long-document recall; 254.326 s native prefill; 5.184 decode tok/s over 52 tokens | RTX 5090, NVMe |
| RTX prefill → Mac decode, measured before YOCO (plain ULTRA prefill) | 184,432 | **173.252 s** RTX compute + **7.662 s** transfer/import; **31.494 decode tok/s** | RTX 5090 → M3 Ultra, 512 GB |
| Mac-only ds4, first 100 decoded tokens | ≈9K / ≈24K | 30.7–31.2 / 31.26–31.47 decode tok/s | M3 Ultra, 512 GB |
| Native quality/speed dial, 3K-token story / C++ runs | Five routing settings | 5.8→14.2 / 5.0→13.3 decode tok/s | RTX 5090, NVMe |
| Single-engine host budget | 262,144 / 999,424 configured | Producer maximum **7.75 GiB**; controller + producer **9.25 GiB** | RTX 5090, bounded host RAM |

The 1M preset admits **999,424 tokens** with a growable bf16 context cache. The long recall above used 237,703 prompt tokens; it is not a full million-token prompt measurement. The paired long request took 229.128 s through prefill and 232.208 s through decode; RTX compute and transfer omit other startup and request work. Faster routing changes expert selection and can reduce answer quality.

## How it works

**Prefill amortizes each weight pass across many tokens.** Each full prefill pass streams every required weight once, so throughput depends on how many tokens share that pass. The combined V4.1 producer's default planner uses 49,152-token sweeps and folds a remainder into the last sweep when it fits the 61,440-token cap. Overlapping reads and compute hide much of the storage cost at large sweep sizes.

**The combined V4.1 decoder fetches selected experts.** Dense weights stay resident and a VRAM expert tier retains frequently used experts; DirectStorage supplies misses. Streaming prefill and the ds4 decode kernels share one producer process. The conversation stays resident between turns, and append-only turns process their new tokens.

**YOCO is approximate and opt-in.** It follows DeepSeek’s [DeepInfer YOCO prefill truncation](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md), with keep=140; TensorSharp is a secondary reference. Our streaming variant skips the weight reads and evaluation of layers 21–39 on non-final sweeps. Layer 20 runs in full; the final sweep evaluates the decoder suffix’s last 140 positions. It changes outputs. The candidate image replay scheduler has CPU coverage, but image quality and paired WebUI qualification remain incomplete. See [candidate release blockers](docs/YOCO_PAIRED_QUALIFICATION.md), [YOCO measurements and quality](docs/ULTRAYOCO_QUALIFICATION.md) and [state transfer](docs/STACK_MEDIA_AND_STATE.md).

**Two machines share a live session.** The RTX sends initial or incremental KV/native state to the Mac, which decodes and sends state updates back. Short follow-ups can stay on the Mac; longer additions use RTX prefill. Pairing validates model, renderer and engine identities. The measured paired result predates YOCO and uses plain ULTRA prefill. Negotiated YOCO state v1 export exists; Mac decode with YOCO has not been benchmarked. See [paired operation](docs/TWO_MACHINE.md).

**The model can choose its quality/speed setting.** Cache-affinity routing favors resident experts at faster settings. The UI slider, API and MCP tools expose Max quality, Balanced, Fast, Faster and Max speed, independently for reasoning, tool calls, answers and code. Use the lower settings for precise code; the fastest measured settings introduced code errors. [Live controls](docs/LIVE_GENERATION.md)

**Memory admission and runtime guards bound host use.** The launch presets cap producer RAM at 7.75 GiB, retain an 2 GiB desktop admission reserve, and stop gracefully below 4 GiB available physical RAM. GPU backing commitment has a separate allowance. Available-RAM and paging checks run during inference; these budgets do not require loading the GGUF into host RAM.

## Requirements

- **Windows RTX:** RTX 50-series, compute capability 12.0; the measured configuration uses a 32 GB RTX 5090. Native builds target CUDA `120a` by default.
- **Fast local NVMe and DirectStorage:** room for the roughly 483 GiB GGUF plus build/runtime files. Storage throughput affects inference speed.
- **Host RAM:** all measurements ran on a desktop with 32 GB of system RAM. The stack needs its 9.25 GiB budget plus the 2 GiB desktop reserve available; admission refuses to start otherwise.
- **Build tools:** Python 3.12, Visual Studio C++ Build Tools, CMake, Ninja, CUDA Toolkit (tested with 13.2), and Node.js/npm (tested with Node 24). Installation needs package-network access; inference uses locally supplied weights.
- **Optional Mac:** M3 Ultra with 512 GB unified memory, the same language-model GGUF, and network connectivity to the RTX.

The combined producer requires native `deepseek41` architecture metadata and the checked 40-layer Flash geometry, `Q4_K` routed experts, `Q8_0` dense/shared/head matrices, and the specified F16/F32 tensors and Engram layout. Its receipt currently pins `DeepSeek-V4.1-Flash-1017-Q4-v41.gguf`, SHA-256 `2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f`. **TODO: publish a verified download URL or conversion recipe for this exact measured file.**

The broader ds4 CUDA decoder contains `IQ2_XXS`/`Q2_K`, `Q4_K` and `MXFP4` expert kernels. Public [antirez DeepSeek-V4.1 Flash GGUF files](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/tree/main) include these quantization families. That kernel coverage does not bypass the combined producer’s current Q4 loader and receipt checks. [Code requirements and model notes](docs/SUPPORTED_MODELS.md) distinguish these paths. Model weights are supplied locally and are never downloaded by the installer.

## Quickstart

```powershell
git clone --recurse-submodules https://github.com/fronteagle739/streaming-llama-stack-clean.git
cd streaming-llama-stack-clean
./slstack.ps1 install --all
./slstack.ps1 install --all --check
```

Create `my.local.json` using the keys in [config.example.json](config.example.json). Local `*.local.*` files are ignored by Git. Select your GPU UUID and model path, for example:

```json
{
  "model": "<model_dir>/DeepSeek-V4.1-Flash-1017-Q4-v41.gguf",
  "gpu_uuid": "<GPU-UUID>",
  "host_budget_gib": 9.25,
  "min_available_gib": 4,
  "runtime_min_free_gib": 4
}
```

```powershell
./slstack.ps1 --config my.local.json check
powershell -File scripts/start-dsv41-single-262k.ps1 -WebUI
```

Open the printed URL (default `http://127.0.0.1:8088`). The router also serves `/v1/chat/completions`, `/v1/models`, `/health` and `/mcp`. Initial setup hashes the entire model once and shows progress; subsequent checks reuse the verified identity while file metadata matches. The first chat loads the producer. Appended turns show prefill progress and completed replies show decode throughput.

Start one preset at a time. These alternatives use the same router port:

```powershell
powershell -File scripts/start-dsv41-single-262k.ps1 -Stop
powershell -File scripts/start-dsv41-single-1m.ps1 -WebUI
# Or opt into approximate YOCO:
powershell -File scripts/start-dsv41-single-1m.ps1 -Stop
powershell -File scripts/start-dsv41-single-262k-ultra-yoco.ps1 -WebUI
# Also available: scripts/start-dsv41-single-1m-ultra-yoco.ps1
```

Each script accepts `-Config <file>`, `-Model <gguf>`, `-DryRun` and `-Stop`. Stop with the same script/config used to start; shutdown drains DirectStorage and releases memory. For a foreground `./slstack.ps1 start --config my.local.json --webui`, use Ctrl-C.

For RTX → Mac operation, install on both computers. On the Mac:

```sh
./bin/slstack install --all
./bin/slstack mac --model '<model_dir>/DeepSeek-V4.1-Flash-1017-Q4-v41.gguf' --host-budget-gib 400 --min-available-gib 8
```

The receiver prints the endpoint, pairing token and Windows command. Use those values on Windows:

```powershell
./slstack.ps1 rtx --config my.local.json --mac <mac-endpoint> --pair-token <printed-token> --webui
```

Keep the pairing token in local runtime configuration. Stop foreground launchers with Ctrl-C. [Pairing details](docs/TWO_MACHINE.md)

## Features

- **Web UI and media tools (experimental):** chat can generate images/video using Anima, Qwen-Image, Krea and H3, with explicit per-generation adapters, progress cards, uploads and inline results. Select existing model/component folders in Settings → Stack. Streamed/Resident preferences are saved per model. [Media setup](docs/MEDIA_SETUP_VERIFICATION.md) · [Media and state](docs/STACK_MEDIA_AND_STATE.md)
- **LoRA (experimental):** native ds4 supports dense-output adapters and frozen-prefix training. Use `start --serving native` for language-model adapters. Permanent export writes a new model; unmerge selects the verified original. [Adapter example](examples/train_jeff_lora/README.md)
- **MCP:** the local `/mcp` endpoint exposes quality/speed controls and enabled media tools. The chat model can use the same tools itself.
- **Additional decoders:** retained MLX, llama and HTTP backend integrations cover other supported models; an HTTP backend performs its own prefill unless it explicitly supports RTX state injection. [Backend interfaces and measurements](docs/BACKEND_SOURCE_AUDIT.md)
- **Configuration:** [Variables](docs/VARIABLES.md) lists environment variables, launcher flags, defaults and source consumers. [Walkthrough](examples/WALKTHROUGH.md) covers setup and use; [Roadmap](docs/ROADMAP.md) lists limitations and planned work.

## Credits and licence

Stack code and original Studio integration modules use the [MIT licence](LICENSE). Embedded projects keep their own licences and attribution; see [NOTICE](NOTICE).

- [DeepSeek](https://github.com/deepseek-ai) for the model and [DeepInfer](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md) for YOCO prefill truncation (keep=140); TensorSharp is a secondary implementation reference. Our contribution is skipping decoder-layer reads on non-final streaming sweeps.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) for inference infrastructure and the web UI.
- [antirez/ds4](https://github.com/antirez/ds4), [ds4-win-NovaStar](https://github.com/mishal-harbi/ds4-win-NovaStar), and [IngeniousIdiocy's V4.1 backend](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra) for native inference and kernels.
- [mlx-serve](https://github.com/ddalcu/mlx-serve) for the optional MLX service integration.
- [Microsoft DirectStorage](https://github.com/microsoft/DirectStorage) for storage-to-GPU transport; redistributables retain Microsoft's terms.
