# ds4-stack

Private glue repository for pinned ds4-win, RTX streaming llama and Studio engines.
**Integration is incomplete.** The image worker and video endpoint have real RTX
smoke proofs. Live bias control, Windows ULTRA handoff and ds4 LoRA attachment
are unavailable in the pinned dependencies. `check`/`start` report missing optional
capabilities; attempts to enable an unqualified handoff or attachment are refused.

```powershell
git clone --branch agent/ds4stack-20261022 --recurse-submodules https://github.com/fronteagle739/streaming-llama-stack-clean.git
cd ds4-stack
python -m pip install -r requirements.txt
Copy-Item config.example.json config.local.json
# Set model paths; all runtime/build/cache writes on this RTX stay under runtime/ds4stack.
./ds4stack.ps1 --config config.local.json check
./build.ps1 -Config config.local.json
```

Source pins and provenance are in `.gitmodules`, gitlinks and `ds4stack/check.py`.
Studio is a pinned submodule, not a vendored source copy. No GitHub workflows.
The llama build uses its own CMake targets; ds4 uses its own Windows build script.
`check` fails loudly for missing required models/builds/pins or unsafe journal state.
It reports the missing optional producer/bias service separately.
Every engine process requires an explicit RSS budget, a physical-memory floor, a
Windows Job private-commit ceiling and an independent process-tree guardian.

You need read access to all three private submodule repos when sharing this repo.
No model weights or user stores are included. Exact pins are committed as Git
submodule links, rather than source copies. No main or colleague branch is used.

Use Windows, Python 3.12+, Visual Studio C++ Build Tools and CUDA 13.2 with an
RTX 5090 (32 GB). The native streaming model needs its original files on fast
local NVMe. Host RAM must have room for the chosen budget **plus** the guard
floor before launch; the example admits 8 GiB only when at least 10 GiB is free.
Change `gpu_uuid` to the UUID reported by `nvidia-smi` on your machine. Studio
also needs its existing Cosmos/diffusers/DirectStorage Python environment; set
`studio_python` to it. Run CMake builds from the VS Developer PowerShell. The
pinned webui already includes `public/index.html.gz`; rebuilding it is optional.
Build targets reuse native Windows/CMake scripts. The producer target additionally
uses the pinned JSON header and DirectStorage SDK, without downloads.

The 48 GiB Windows Job commit ceiling allows WDDM device mappings; it is **not**
permission to allocate 48 GiB of physical RAM. An independent 50 ms guardian
checks the complete process tree against the 8 GiB physical RSS budget and the
2 GiB available-physical floor. It kills the Job on breach, parent death or timeout.
The floor is headroom for polling/OS activity; this is a fail-fast guard, not an
OS guarantee that Windows will never page unrelated processes. Timed state stays
in VRAM/RAM, weights on D:, all task writes on L:, with no M: staging or KV spill.

Headless launch uses one loopback OpenAI-compatible endpoint:

```powershell
./ds4stack.ps1 --config config.local.json start
./ds4stack.ps1 --config config.local.json start --webui --image-model models/anima-base-v1.0.safetensors --image-arch anima-1.0 --image-mode streaming
curl.exe http://127.0.0.1:8088/v1/models
curl.exe http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' --data-binary '@request.json'
```

CLI options override the config. `--mode` sets both media modes; `--image-mode`
and `--video-mode` override each individually. Paths identify the actual model,
not a Studio production registration. `--comfyui-models` supplies the support
components through Studio's own `comfy_support_defaults`; select the generation
model with `--image-model`/`--video-model`. Architecture can be inferred for known
single-file safetensors using Studio's checkpoint-header detector; explicit
`image_arch`/`video_arch` is preferable. Extra component files go in the respective
`image_support_files`/`video_support_files` objects. Unsupported modes fail rather
than silently falling back: Wan is streaming only in this adapter; Bernini folders
are resident and single files use its streamed backend. Anima honors both modes.
Workers exit after each request, so resident means resident **during the job**;
there is currently no warm resident worker reuse across requests.

`/v1/images/generations` and `/v1/videos/generations` accept the same JSON settings
as `generate_image`/`generate_video`. The media HTTP adapter also accepts `model`
(alias `image`/`video`, the selected architecture or filename stem), `n=1`,
`size=WIDTHxHEIGHT` and `response_format=url`. Other batch counts/formats fail
explicitly. Each tool's schema/description includes all
accepted settings, defaults and ranges. Model paths are startup settings, outside
model-controlled arguments. All outputs go to `<runtime_root>/outputs` and are
served under `/outputs/`. The existing llama.cpp webui is served at `/` with its
API pointed at this router. A small media-rendering adapter turns local video
Markdown references into native video controls; the UI source is unchanged.

The pinned webui lacks MCP. Current upstream supports MCP (see the official
[server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)).
For this older pin the router injects OpenAI tools, executes stack calls server-side,
then asks native decode to finish the reply. Only startup-configured capabilities
are advertised to the chat model. Tool loops are bounded to eight
rounds and serialized with direct endpoint/MCP jobs. Client-owned tools remain
client-owned. SSE currently buffers the tool loop and final answer; it does not
stream native tokens live. Upstream usage/timings are preserved when supplied.

`GET /v1/models` describes the configured model; `/health` distinguishes router
readiness from model loading. `POST /mcp` offers sessionless JSON-RPC MCP; stdio
MCP is `python -B -m ds4stack --config config.local.json mcp`. Adapt
[`examples/mcp.json`](examples/mcp.json) for LM Studio or Claude Desktop. It points
to this stack, whose bias calls delegate to the existing service at `bias_mcp_url`.
Until DS4WINBIAS2 supplies that live service, a change returns an explicit tool
error. The router never reports a stored preference as an applied engine setting.

| Quality/speed setting | Value |
|---|---:|
| Max quality (default) | 0 |
| Balanced | 0.35 |
| Max speed (experimental quality loss) | 0.9 |

These labels are the owner's interface, not a promise of equal answer quality.
Live speed-change and judged-quality proofs remain outstanding.

LoRA commands reuse existing trainers; no new trainer or weight-fusing code lives
here. `hf_streamed` uses Studio's HF causal-LM streamed trainer; `gguf_streamed`
supports only its step35/deepseek2/glm_dsa architectures, and `cosmos3` expects a
media folder with captions. **DSV4.1 1017 is unsupported** by those training paths.
Separate `train_lora.ps1`, `merge_lora.ps1`, `unmerge_lora.ps1` and `run_prompts.ps1`
wrappers accept `-Config` and forward the command-specific arguments.

The default native smoke preset uses 4K context, the existing `28GB` cache-target
syntax and the qualified `DS4_METAL_DISABLE_V41_LAYER_PREFILL=1` flag, which also
controls CUDA. Omitting that flag can send even a tool preamble down bulk prefill
and exhaust GPU headroom. The larger-context attempts in this task were rejected
by native admission or the physical-memory guardian; they are not qualified presets.

```powershell
./ds4stack.ps1 --config config.local.json train_lora --data-folder runtime/TEST-prompts --settings examples/train-hf.json --output runtime/TEST-lora
./ds4stack.ps1 --config config.local.json run_prompts --folder runtime/TEST-prompts --output runtime/TEST-evaluation
./ds4stack.ps1 --config config.local.json merge_lora --test --base runtime/TEST-base.gguf --lora runtime/TEST-adapter.gguf:0.5 --exporter build/llama/bin/Release/llama-export-lora.exe --output runtime/TEST-merged.gguf
./ds4stack.ps1 --config config.local.json unmerge_lora --entry JOURNAL_ENTRY_ID
```

D1: temporary attachment means per-layer VRAM fusion/attachment by the engine,
with no base-file writes. This remains unavailable for ds4-win; `--lora path[:scale]`
is parsed but refused before launching an unsupported model. D2: permanent merge
writes a **new** output and never edits the base in place. Quantized in-place
merge/unmerge is not reversible. The pinned `llama-export-lora` actually writes
modified tensors as F16/F32, **without requantizing**; this is a flagged refinement
to the brief's requantization assumption, and no requantizer is run. Execution is
limited to explicit small TEST bases (<=1 GiB), with exclusive atomic publication.
`unmerge_lora` selects the verified base in `state/model_selection.json`; future
launches read that choice. The merged copy is preserved, never deleted.

Every command checks `state/lora_journal.json`. Base and adapter SHA-256, scales,
output/hash, status, PID plus creation time and timestamps are atomically recorded
before/after changes. A dead `attaching`/`attached` owner is detached; a dead merge
is failed and partial bytes are renamed to a recoverable quarantine. Base-hash
mismatch refuses all operations. Live owners are never commandeered. GPU holds
have equivalent dead-guardian recovery; each lease lasts <=19 minutes and yields
to any other ACTIVE RTX hold, including DS4WINBIAS2. No production service,
conversation, preset or store is stopped or modified.

Existing offline Studio components may still require HF configuration metadata.
The real Wan proof staged a cached 724-byte VAE config into L:'s isolated HF cache;
weights remained in their D: locations. A clean install must supply those configs
in the isolated cache as well. No network model download is performed automatically.

The long-prompt adapter/parity requirements are in [`docs/HANDOFF.md`](docs/HANDOFF.md).
Current source and proof limits are in [`docs/STATUS.md`](docs/STATUS.md). Run CPU
safety/protocol checks with an explicit budget:

```powershell
./ds4stack.ps1 --host-budget-gib 1 guarded -- python -B -m unittest discover -s tests -v
```
