# Streaming Llama Stack

Run frontier-size mixture-of-experts models on one desktop GPU. The stack streams
the weights from NVMe straight into an RTX 5090 with DirectStorage, so a 483 GiB
model runs on a machine with 32 GB of VRAM and 32 GB of system RAM. Add a
512 GB Mac Studio and the RTX does the long prefill while the Mac decodes.

It ships a llama.cpp fork with streaming prefill, the ds4 decode kernels for
DeepSeek V4.1 Flash, an OpenAI-compatible API, MCP tools and the llama.cpp web UI.

## QFN and DSV4.1: recipes, modes and measured scope

The recipes pin upstream Hugging Face revisions and every source-file identity.
QFN uses Streaming Llama Pack (`.slpack`). DSV4.1 keeps its existing GGUF main,
DSpark/MTP and vision route. Source/model licenses remain the upstream terms.

| Model | Pinned source | Recipe and format |
| --- | --- | --- |
| QFN | [NVIDIA Qwen3.8 Flash Next NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47) | [QFN recipe](recipes/qfn-nvfp4.json), [auxiliary recipe](recipes/qfn-auxiliary.json), [format and pack commands](docs/SLPACK_RECIPES.md) |
| DSV4.1 | [DeepSeek V4.1 Flash 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) | [DSV4.1 recipe](recipes/dsv41-flash.json), [GGUF converter](docs/DSV41_GGUF.md) |

QFN preserves raw NVFP4 routed-expert bytes through its native relayout. Its
dense BF16 projections were quantized to Q6_K/Q8_0 and its FP8 PLE table to Q8_0;
MTP, vision, profiles and explicit sampler defaults come from the pinned
compatible auxiliary container. The complete pack is not claimed lossless
against the raw NVIDIA checkpoint. No supported fields, samplers, arithmetic
aliases, stop behavior, numeric bias range or MTP capability were removed.

### Performance and quality

Hardware: RTX 5090 with 32 GB VRAM, 32 GB system RAM and fast local
DirectStorage NVMe storage. Rates are tokens/s. Bias applies to decode;
prefill uses unbiased routing. KL is relative to unbiased routing and top-1
is agreement. A dash means no matching measurement is available.

#### QFN

| Prompt length | Bias | Prefill | Decode | KL | Top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0 | 4,440 | 25.25 | 0 | 100.0% |
| 100,000 | 0.35 | 4,440 | 46.25 | 0.0372 | 92.6% |
| 100,000 | 0.68 | 4,150 | 50.12 | 0.137 | 88.3% |

#### DeepSeek V4.1

| Prompt length | Bias | Prefill | Decode | KL | Top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0 | — | — | — | — |
| 100,000 | 0.35 | — | — | — | — |
| 100,000 | 1.36125 | — | — | — | — |

QFN timing uses 100 discarded warmup tokens and 128 measured tokens per mode
on one continuing sequence, MTP off and eight read parts. Full-vocabulary
quality comparisons use the same forced continuation against bias zero.
QFN named modes use 0 / 0 / 0.68; numeric intermediate values, including 0.35,
remain available. DeepSeek modes use 0 / 0.35 / 1.36125.

Reproduce QFN's exact fixture and timing protocol with the
[100K guide](docs/QFN_100K_REPRO.md), and use the
[calibration workflow](examples/single-machine/CALIBRATION.md) for KL/top-1.
The portable timing wrapper has exact-fixture and CPU protocol checks;
native wrapper qualification remains pending. DeepSeek's matching current-stack
100K measurements are unavailable; no short-prompt result substitutes for them.

### Download, pack and run

Replace uppercase path placeholders with caller-selected locations. Downloads,
outputs, caches and temporary files belong on the intended data/scratch volume.
Do not launch conversion or hashing in a reserved timing window. Set an explicit
host budget that fits available physical RAM above the configured floor.

QFN packing was verified with Python 3.12.10, NumPy 2.5.0, `regex` 2026.5.9 and
`psutil` 7.2.2. Install [requirements-pack.txt](requirements-pack.txt) in an
isolated environment before packing, and use that interpreter for the QFN
Python commands below (POSIX uses `.venv-pack/bin/python`):

```text
python -m venv .venv-pack
.venv-pack/Scripts/python -m pip install -r requirements-pack.txt
```

```text
python scripts/download_slpack_source.py --recipe recipes/qfn-nvfp4.json --out QFN_SOURCE
python scripts/download_slpack_source.py --recipe recipes/qfn-auxiliary.json --out QFN_AUXILIARY
python scripts/pack_slpack.py --source QFN_SOURCE --sections-from QFN_AUXILIARY_FILE --allow-legacy-base --out QFN.slpack --scratch PACK_SCRATCH --host-budget-gib 6 --runtime-floor-gib 4
python scripts/slpack.py QFN.slpack
qfn-engine --pack-check --model QFN.slpack
```

Both required QFN inputs are pinned. The compatible auxiliary container is
69,685,970,297 bytes, SHA256
`dee4ae542817d73e4f7a56c448ef97cb7bb47ef92b8b2c5b7721ec0e8c25f76c`.
The verified migrated reference is 131,730,218,667 bytes, SHA256
`0f6291a714839961007f0d130aa715b53d82f8d1e73cef9785700e83bce74fbe`.
The complete raw import regenerated all 26 sections, inverse/HC8 checks and
canonical metadata after freshly authenticating all 25 source files, then read
back the entire reference. The verified canonical output is 131,730,218,740 bytes,
SHA256 `195df4842abf2dd335ad706be6518611b2a433e806d98c4edcc8fe5e2b309a92`.
Its metadata is 73 bytes longer and changes exactly seven declared provenance
fields while retaining every field. All 25 nonmetadata payloads remain identical
to the reference artifact. The proof calculated the exact output identity
without writing a second full model. Both memory guards closed successfully;
bounded paging observation is not a claim of zero physical paging I/O.

The [exact QFN 100K reproduction guide](docs/QFN_100K_REPRO.md) includes the
pinned public novel/tokenizer, deterministic fixture builder, all commands
and the guarded same-sequence benchmark. The fixture reproduces the measured
400,000 bytes exactly (SHA256
`87fb9af861237f91724d56ca9c1a29f77c0e4aadce673ff939abe8500fa6c936`).
Its CPU protocol checks pass, including sampler forwarding and early-EOG
failure without extension; **native qualification of the portable wrapper is
pending**. The guide retains the 0.35 reproduction command and explains the
selected 0.68 mode without inventing a new measurement. Continue using the
[calibration workflow](examples/single-machine/CALIBRATION.md) for independent
full-vocabulary KL/top-1 and sampled natural-stop qualification.

```text
python scripts/download_slpack_source.py --recipe recipes/dsv41-flash.json --out DSV_SOURCE
python -B scripts/convert_dsv41_gguf.py --source DSV_SOURCE --ds4-tools DS4_TOOLS --component main --execute --expert-format native-fp4 --expert-layout flashinfer-sm120-mxfp8-mxfp4-v1 --dense-format q8 --out DSV_MAIN.gguf --quants-library DS4_QUANTS_LIBRARY --host-budget-gib 8 --runtime-floor-gib 4 --threads 1
python -B scripts/convert_dsv41_gguf.py --source DSV_SOURCE --ds4-tools DS4_TOOLS --component dspark --execute --out DSV_DSPARK.gguf --quants-library DS4_QUANTS_LIBRARY --host-budget-gib 8 --runtime-floor-gib 4 --threads 1
python -B scripts/convert_dsv41_gguf.py --source DSV_SOURCE --ds4-tools DS4_TOOLS --component vision --execute --out DSV_VISION.gguf --host-budget-gib 8 --runtime-floor-gib 4 --threads 1
powershell -File examples/single-machine/dsv41-modes.ps1 -Config LOCAL.json -Mode max_quality -Context 262k
powershell -File examples/single-machine/dsv41-modes.ps1 -Config LOCAL.json -Mode max_speed -Context 262k
```

The DSV commands describe the supported conversion and launch route; they do
not reproduce the 100K decode receipt automatically. Its exact
fixture/benchmark command and a current matched 100K prefill,
quality and speed qualification remain required. Preserve all main,
DSpark/MTP and vision outputs. The recipe's 518,596,067,328-byte Q4 main-only
reference, SHA256
`2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f`,
is not the native-FP4 result or a complete auxiliary identity. Native main,
DSpark/MTP and vision output identities remain null pending full conversion
and qualification receipts.

GLM's shared-format source port and 33 CPU fixtures are a separate branch
handoff. Its `scripts/glm53f_pack.py` route is not present in this QFN review
tree. Native adoption/build, full packing, inverse verification, runtime
qualification and output size/hash remain pending. This package is not claimed
to provide complete measured three-model download/pack/verify/run reproduction.

### Presets and scripts (fastest defaults)

- RTX single machine: `examples/single-machine/dsv41-ultra-cone.ps1 -Context 1m`
  (packed FP4 file in your config; Max quality or Max speed is the live quality-speed
  control).
- Mac single machine: `slstack mac` with the Mac decode FP4 file and its DSpark support
  model. Metal layer prefill is on for Mac-only requests.
- Two machines: `examples/dual-machine/dsv41-ultra-cone-fp4/preset.json`
  (`python examples/run_preset.py --preset dsv41-ultra-cone-fp4 --machine mac|rtx`). The
  Mac receiver relays the producer stream over direct TCP; `host_budget_gib` 9.25 admits
  the whole-prompt Engram rows.

The host-RAM expert tier and the CPU expert lane stay opt-in toggles.

### Host-RAM expert tier

The host-RAM expert tier is supported and stays off by default
(`LLAMADECODE_HOST_CACHE_MB=0`). VRAM-to-RAM copy-in runs on its own stream;
RAM hits upload on a separate H2D stream while DirectStorage reads. The route
cost is configurable (`LLAMADECODE_RAM_ROUTE_COST=0.5`), as are asynchronous
and synchronous transfer policies. CPU expert lanes also remain optional.
On Windows the tier uses pinned staging when CUDA registration is unavailable;
size it within the physical-RAM guardian's budget. Decode headroom reserves
128 MiB plus a 64 MiB L2 growth margin, with guarded allocation step-down.
See [runtime variables](docs/VARIABLES.md) for all controls and
[performance details](docs/PERFORMANCE.md) for the measured configuration scope.

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
Paired mode negotiates state v2 and retains the mode 3 export/import payload,
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

## Making the model files

DeepSeek V4.1 Flash uses four GGUF files, all produced from the original
[DeepSeek-V4.1-Flash-1017 safetensors checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)
(revision `1bae0e10…`, about 475 GiB) with `scripts/convert_dsv41_gguf.py`. The two
FP4 files keep DeepSeek's released E2M1/E8M0 expert values in two different byte
layouts and are not interchangeable: the RTX file carries
`deepseek41.expert_layout = flashinfer-sm120-mxfp8-mxfp4-v1`, which ds4 on the Mac
refuses to load, and the Mac file has no layout flag (canonical `ggml-mxfp4`).

Status: this native safetensors path has not yet been run end to end and qualified on this stack.

### Which file is which

| File | Purpose | Runs on | Preset or script | Command (after the plan step below) |
|---|---|---|---|---|
| **RTX packed FP4** | RTX prefill (single machine and the prefill side of two machines) and RTX single-machine decode | Windows RTX 5090 | `examples/single-machine/dsv41-ultra-cone.ps1`; `rtx.model` of `examples/dual-machine/dsv41-ultra-cone-fp4/preset.json` | `--expert-format native-fp4 --expert-layout flashinfer-sm120-mxfp8-mxfp4-v1 --dense-format q8` |
| **Mac decode FP4** | Mac decode with ds4: Mac single machine and the decode side of two machines | Mac Studio M3 Ultra | `slstack mac --model …`; `mac.model` of `dsv41-ultra-cone-fp4/preset.json` and `dsv41-ultra-cone/preset.json` | `--expert-format native-fp4 --expert-layout ggml-mxfp4 --dense-format q8` |
| **DSpark drafter** | Speculative drafts for Mac decode | Mac Studio M3 Ultra | `mac.mtp_model` in the paired presets; the DSpark support model for `slstack mac` | `--component dspark` (Q4_K experts, Q8_0 attention/shared/projections; no `--expert-format`/`--expert-layout`) |
| **Q4** | The Q4_K/Q8_0 alternative on the RTX | Windows RTX 5090 (also loads on the Mac) | `scripts/start-dsv41-single-262k.ps1` / `-1m.ps1` with the Q4 path in your config; `rtx.model` of `dsv41-ultra-cone/preset.json` | `--expert-format q4` (the default) |

Rule of thumb: **MXFP4 experts with the layout flag = RTX; MXFP4 experts with no flag
= Mac; Q4_K experts = Q4.** You can check a file without reading its payload:

```powershell
python -B -c "import sys; sys.path.insert(0,'.'); from slstack.gguf_header import read_header, TYPE_NAMES; h=read_header(sys.argv[1]); m=h['metadata']; t=[x for x in h['tensors'] if x['name'].endswith('_exps.weight')][0]; print('experts', TYPE_NAMES.get(t['type']), '| layout', m.get('deepseek41.expert_layout', 'none = ggml-mxfp4 if MXFP4'))" <file.gguf>
```

### Commands

Inputs: the downloaded checkpoint directory (`--source`; nothing downloads weights)
and a directory for the pinned antirez/ds4 `gguf-tools` sources (`--ds4-tools`).
`--fetch-tools` fills that directory with the hash-pinned source files only, never
weights. `--execute` also needs `--out`, `--host-budget-gib` and, for main and
DSpark, `--quants-library`: a built `libds4quants` (`make -C <ds4-tools>
quants-shared`, C compiler and pthreads required).

**1. Plan** (opens only `config.json` and the index, never the shards; `--plan` must be a new file):

```sh
python -B scripts/convert_dsv41_gguf.py --source <HF checkpoint dir> --ds4-tools <ds4-tools> --fetch-tools \
  --expert-format native-fp4 --expert-layout flashinfer-sm120-mxfp8-mxfp4-v1 --dense-format q8 \
  --dry-run --plan <new-plan.json>
```

**2. Convert**, with the same file flags as the plan:

```sh
# RTX packed FP4
python -B scripts/convert_dsv41_gguf.py --source <HF checkpoint dir> --ds4-tools <ds4-tools> \
  --expert-format native-fp4 --expert-layout flashinfer-sm120-mxfp8-mxfp4-v1 --dense-format q8 \
  --execute --out <rtx-fp4.gguf> --quants-library <libds4quants> --threads 8 --host-budget-gib 8

# Mac decode FP4
python -B scripts/convert_dsv41_gguf.py --source <HF checkpoint dir> --ds4-tools <ds4-tools> \
  --expert-format native-fp4 --expert-layout ggml-mxfp4 --dense-format q8 \
  --execute --out <mac-fp4.gguf> --quants-library <libds4quants> --threads 8 --host-budget-gib 8

# DSpark drafter
python -B scripts/convert_dsv41_gguf.py --source <HF checkpoint dir> --ds4-tools <ds4-tools> \
  --component dspark --execute --out <dspark.gguf> --quants-library <libds4quants> --threads 8 --host-budget-gib 8

# Q4
python -B scripts/convert_dsv41_gguf.py --source <HF checkpoint dir> --ds4-tools <ds4-tools> \
  --expert-format q4 --execute --out <q4.gguf> --quants-library <libds4quants> --threads 8 --host-budget-gib 8
```

- **Both machines' files must use the same `--dense-format`.** The code accepts
  `--dense-format native-fp8` only with the RTX layout, so a matched RTX/Mac pair
  uses `q8` on both.
- `--threads` is 1..32. `--host-budget-gib` is required with `--execute` and must
  fit in physical RAM above `--runtime-floor-gib` (default and minimum 4).
- `--resume` continues an interrupted run from its journal. Existing outputs are
  never overwritten. Use a distinct `--out` for each file; logs and the memory
  receipt go beside it (or under `--runtime-root`).
- The header is checked against the plan before success is reported.

### Disk and memory

| File | Output size | Free disk needed | Host RAM |
|---|---:|---|---|
| RTX packed FP4 (`q8` dense) | ~502 GB | output + 32 GiB reserve, checked by the writer | `--host-budget-gib` |
| Mac decode FP4 (`q8` dense) | ~502 GB | output + 32 GiB reserve | `--host-budget-gib` |
| DSpark drafter | ~7.8 GiB | output + 32 GiB reserve | `--host-budget-gib` |
| Q4 | ~483 GiB | output + 32 GiB reserve | `--host-budget-gib` |

The ~475 GiB checkpoint stays read-only and is needed in addition to the outputs.
Conversion is CPU-only.

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
  "min_available_gib": 0.5,
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

Use the [ULTRA exact-cone paired preset](examples/dual-machine/dsv41-ultra-cone/preset.json)
with the [example launchers](examples/dual-machine/README.md).
[Cone state protocol](docs/CONE_STATE.md).

For a Windows/Linux second machine with enough RAM to hold the model, see the
[large-RAM ds4-win untested example](examples/dual-machine/dsv41-large-ram-untested/README.md).
It documents the missing mode 3/YOCO CPU receiver; its launcher only previews the
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
- Experimental prefill graphs do not have a demonstrated repeatable gain across
  all prompt lengths. Configured long-context capacity does not establish
  warm-decode performance at every length.
- FP4 prefill timing does not qualify decode or broad recall quality. Dense FP8
  remains experimental and has no demonstrated prompt gain.
- TileLang attention is the selected default but is numerically different from
  native attention. The tested retrieval answer matches; later forced continuations
  can diverge. Set `LLAMA_DSV41_TILELANG_ATTENTION=0` for native attention.
  Mode 2 now runs a hand-written SM120 kernel with the same arithmetic in a
  different summation order (`LLAMA_DSV41_SPARSE_HAND=0` restores the TileLang image).
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

## Qwen3.8 Flash Next

The standalone Windows CUDA streaming engine, named modes and calibration workflow
are described in [QFN setup](docs/QFN.md). The shared model pack uses `.slpack`.
