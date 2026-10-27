# Streaming Llama Stack

Run DeepSeek-V4.1 Flash, a roughly 483 GiB GGUF, on one RTX 5090 by streaming weights from NVMe through DirectStorage. The stack combines streaming prefill and ds4 decode kernels in one process, an OpenAI-compatible API, and the llama.cpp web UI. It supports context windows up to approximately 1M tokens with a small host-RAM budget. An optional Apple M3 Ultra can take over decode after the RTX prefills a prompt and hands over its KV session; this two-machine path has been used daily through Studio for months.

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
| RTX prefill → Mac decode | 184,432 | **173.252 s** RTX compute + **7.662 s** transfer/import; **31.494 decode tok/s** | RTX 5090 → M3 Ultra, 512 GB |
| Mac-only ds4, first 100 decoded tokens | ≈9K / ≈24K | 30.7–31.2 / 31.26–31.47 decode tok/s | M3 Ultra, 512 GB |
| Native quality/speed dial, 3K-token story / C++ runs | Five routing settings | 5.8→14.2 / 5.0→13.3 decode tok/s | RTX 5090, NVMe |
| Single-engine host budget | 262,144 / 999,424 configured | Producer maximum **7.75 GiB**; controller + producer **9.25 GiB** | RTX 5090, ≈32 GB host RAM |

The 1M preset admits **999,424 tokens** with a growable bf16 context cache. The long recall above used 237,703 prompt tokens; it is not a full million-token prompt measurement. The paired long request took 229.128 s through prefill and 232.208 s through decode; RTX compute and transfer omit other startup and request work. Faster routing changes expert selection and can reduce answer quality.

## How it works

**Prefill amortizes each weight pass across many tokens.** Each full prefill pass streams every required weight once, so throughput depends on how many tokens share that pass. The default planner uses 49,152-token sweeps and folds a remainder into the last sweep when it fits the 61,440-token cap. Overlapping reads and compute hide much of the storage cost at large sweep sizes.

**Decode fetches selected experts.** Dense weights stay resident and a VRAM expert tier retains frequently used experts; DirectStorage supplies misses. Streaming prefill and the ds4 decode kernels share one producer process. The conversation stays resident between turns, and append-only turns process their new tokens.

**YOCO is approximate and opt-in.** Layer 20 runs in full. Earlier sweeps skip decoder layers 21–39; the final sweep evaluates their last 140 positions. It changes outputs. Image-containing prompts use full evaluation. See [YOCO measurements and quality](docs/ULTRAYOCO_QUALIFICATION.md) and [state transfer](docs/STACK_MEDIA_AND_STATE.md).

**Two machines share a live session.** The RTX sends initial or incremental KV/native state to the Mac, which decodes and sends state updates back. Short follow-ups can stay on the Mac; longer additions use RTX prefill. Pairing validates model, renderer and engine identities. See [paired operation](docs/TWO_MACHINE.md).

**The model can choose its quality/speed setting.** Cache-affinity routing favors resident experts at faster settings. The UI slider, API and MCP tools expose Max quality, Balanced, Fast, Faster and Max speed, independently for reasoning, tool calls, answers and code. Use the lower settings for precise code; the fastest measured settings introduced code errors. [Live controls](docs/LIVE_GENERATION.md)

**Memory admission and runtime guards bound host use.** The launch presets cap producer RAM at 7.75 GiB, retain an 8 GiB desktop admission reserve, and stop gracefully below 4 GiB available physical RAM. GPU backing commitment has a separate allowance. Available-RAM and paging checks run during inference; these budgets do not require loading the GGUF into host RAM.

## Requirements

- **Windows RTX:** RTX 50-series, compute capability 12.0; the measured configuration uses a 32 GB RTX 5090. Native builds target CUDA `120a` by default.
- **Fast local NVMe and DirectStorage:** room for the roughly 483 GiB GGUF plus build/runtime files. Storage throughput affects inference speed.
- **Host RAM:** enough for the 9.25 GiB stack budget, desktop reserve and other applications. The measurements used approximately 32 GB. Admission refuses insufficient available memory.
- **Build tools:** Python 3.12, Visual Studio C++ Build Tools, CMake, Ninja, CUDA Toolkit (tested with 13.2), and Node.js/npm (tested with Node 24). Installation needs package-network access; inference uses locally supplied weights.
- **Optional Mac:** M3 Ultra with 512 GB unified memory, the same language-model GGUF, and network connectivity to the RTX.

The measured file is `DeepSeek-V4.1-Flash-1017-Q4-v41.gguf`, SHA-256 `2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f`. Other V4/V4.1 GGUFs are not interchangeable. **TODO: publish a verified download URL or conversion recipe for this exact 1017 Q4 file.** The public [ds4 model repository](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf) contains other builds; an exact matching public artifact has not been established. Model weights are never downloaded by the installer.

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

- [DeepSeek](https://github.com/deepseek-ai) for the model.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) for inference infrastructure and the web UI.
- [antirez/ds4](https://github.com/antirez/ds4), [ds4-win-NovaStar](https://github.com/mishal-harbi/ds4-win-NovaStar), and [IngeniousIdiocy's V4.1 backend](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra) for native inference and kernels.
- [mlx-serve](https://github.com/ddalcu/mlx-serve) for the optional MLX service integration.
- [Microsoft DirectStorage](https://github.com/microsoft/DirectStorage) for storage-to-GPU transport; redistributables retain Microsoft's terms.
