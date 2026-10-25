# Single-RTX walkthrough

What a new user did on 2026-10-24, on the Windows RTX 5090 machine, from a fresh clone of this repository. It covers the single-RTX native path only. The two-machine steps (README steps 4 and 5) have not been walked through yet.

**Which path these receipts are from.** Every timing, log line and screenshot below was recorded on the native ds4-win path, which was then the default. `start` now serves the single engine by default (README, single-RTX section); the native path is `start --serving native`. The single-engine rerun of this walkthrough on the GPU is pending; until it is done, nothing below is a single-engine receipt. Expected differences: the first prompt also loads the producer, and the image swap re-prefills the conversation instead of restoring saved state.

## 1. Clone, install and check

```powershell
git clone --recurse-submodules -b agent/slstack-clean-20261022 <repository URL> slstack
cd slstack
./slstack.ps1 install --all          # isolated media venv, native engines, chat UI
./slstack.ps1 install --all --check
```

- The first install from a fresh clone took 1611 s. Most of that was the CUDA build of the streaming llama targets (1061 s).
- Updating an existing clone with `git pull` and rerunning `install --all` took 258 s.
- The install leaves the checkout clean.
- Toolchain on that machine: Python 3.12.10, CUDA 13.2 (nvcc 13.2.51), Node 24.14.0 with npm 11.9.0, Visual Studio 2026 Build Tools with its CMake/Ninja.

## 2. Point the stack at local models

Create `my.local.json` in the clone. Git ignores files named `*.local.*`.

```json
{
  "model": "<path>/DeepSeek-V4.1-Flash-1017-Q4-v41.gguf",
  "comfyui_models": "<ComfyUI>/models",
  "image_model": "<ComfyUI>/models/diffusion_models/anima-base-v1.0.safetensors",
  "image_arch": "anima-1.0",
  "image_support_files": {
    "anima_dir": "<path>/Anima-Base-v1.0",
    "text_encoder": "<ComfyUI>/models/text_encoders/qwen_3_06b_base.safetensors"
  }
}
```

`./slstack.ps1 --config my.local.json check` should then report every required check as OK.

## 3. Start

```powershell
./slstack.ps1 start --config my.local.json --webui --serving native   # the path this run used
```

- The launcher prints `Router starting on http://127.0.0.1:8088`. Open that address.
- The model loads on the first request; it was ready 5 s after loading started.

## 4. Use it

| Step | What the user typed | Result |
|---|---|---|
| Chat | "In two sentences, what is a lighthouse for?" | Answer in 37 s, including the first prefill. Decode 6.1 tok/s. |
| Quality/speed | "Please switch yourself to max speed using your quality/speed tool, then tell me which setting is now active." | The model called the tool. The endpoint then read `max_speed` (1.36125), applied. |
| Image | "Make me a picture of a small red lighthouse on a rock, simple flat illustration. Use 512x512, 4 steps and seed 7." | The model called `generate_image` and the image appeared in the chat after 115 s. |

During the image request, the stack swapped the model out and back (native path):

1. It saved the chat state into locked RAM.
2. It shut the engine down gracefully (2.8 s).
3. It generated the image (33 s).
4. It relaunched the engine with the chat state restored (4.5 s). The engine resumed at position 2016 instead of prefilling again.

The only setting changed in the web UI was the maximum response length (200 tokens), to keep the run short.

Screenshots: [chat](walkthrough/chat-desktop.png) ([phone](walkthrough/chat-phone.png)), [quality](walkthrough/quality-desktop.png) ([phone](walkthrough/quality-phone.png)), [image](walkthrough/image-desktop.png) ([phone](walkthrough/image-phone.png)).

## 5. Stop

Press Ctrl-C in the `start` window. The launcher asks the router to stop. The router closes the native engine through its graceful shutdown, which drains DirectStorage and frees GPU and host memory. Both engine processes in this run logged a graceful shutdown, and GPU memory returned to its idle level.

## 6. Single engine with the DSV4.1 launch scripts (steps; GPU receipt pending)

These are the steps for the default single engine (streaming-llama ULTRA prefill + `ds4-decode.dll` decode in one producer). They have not been run on the GPU from a fresh clone yet, so nothing in this section is a receipt.

1. Build as in section 1 (`./slstack.ps1 install --all`, then `--check`). The build puts `llama-dsv41-prefill.exe` and `ds4-decode.dll` in `build/llama/bin/`.
2. Write `my.local.json` as in section 2; only `model` is required for chat.
3. Check what will run, without starting anything:
   ```powershell
   powershell -File scripts/start-dsv41-single-262k.ps1 -DryRun
   ```
   It prints the start config and the resolved producer settings: `--context 262144`, `--ctx-cache-precision f32 --ctx-cache-growable 1`, single sweep off (capacity 49152), and the producer environment (`DS4_DS_READ_PARTS=8`, `LLAMADECODE_VRAM_RESERVE_MIB=512`, `LLAMADECODE_ENGRAM_EARLY=auto`, `LLAMADECODE_ENGRAM_CACHE_MB=0`, `LLAMADECODE_DS4_DECODE=1`). The 1M script prints `--context 999424` with `bf16`.
4. Start the 262K configuration with the web UI:
   ```powershell
   powershell -File scripts/start-dsv41-single-262k.ps1 -WebUI
   ```
   The script returns once the router answers and prints `Router ready: http://127.0.0.1:8088`, the log path, the health command and the shutdown command. Without `-WebUI` only the API is served. For the 1M context use `scripts/start-dsv41-single-1m.ps1` instead (one preset at a time).
5. Open `http://127.0.0.1:8088`. Send a short prompt. The first prompt of a conversation also loads the producer and hashes the model once for its receipt, so it takes longer and shows only the waiting indicator. Send a second, longer turn: the web UI shows the prompt-processing bar while the producer prefills the appended tokens, then streams the answer; the chat statistics show tokens per second.
6. Quality/speed: ask the model to switch itself to max speed with its quality/speed tool and to say which setting is active. The tool reply reports the requested and applied values.
7. Health: `Invoke-RestMethod http://127.0.0.1:8088/health`.
8. Stop gracefully:
   ```powershell
   powershell -File scripts/start-dsv41-single-262k.ps1 -Stop
   ```
   The router asks the single engine to shut down; the producer leaves through its own graceful shutdown and frees VRAM and the GPU hold. The log (`runtime/logs/dsv41-single-262k-*.log`) and the producer's `stderr.log` under `runtime/single-engine/ultra-*` record it.

## Notes found during the walkthrough

- The walkthrough above ran with the earlier 4096-token default context, and one short chat plus the tool definitions used 47% of it. The default is now 262144 tokens. A follow-up short chat from the same clone (2e536c1) logged `context=262144`, context VRAM growing in 2048-token steps (1972 MiB mapped of 8859 MiB), an expert tier of 666 experts against the fixed full-context plan of 301, and a graceful shutdown.
- The web UI does not yet show the "saving / reloading" status during an image swap. The router log records it.
- Not walked through yet: video, two-machine mode, LoRA.
