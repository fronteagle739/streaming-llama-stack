# Single-RTX walkthrough

What a new user did on 2026-10-24, on the Windows RTX 5090 machine, from a fresh clone of this repository. It covers the single-RTX native path only. The two-machine steps (README steps 4 and 5) have not been walked through yet.

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
./slstack.ps1 start --config my.local.json --webui
```

- The launcher prints `Router starting on http://127.0.0.1:8088`. Open that address.
- The model loads on the first request; it was ready 5 s after loading started.

## 4. Use it

| Step | What the user typed | Result |
|---|---|---|
| Chat | "In two sentences, what is a lighthouse for?" | Answer in 37 s, including the first prefill. Decode 6.1 tok/s. |
| Quality/speed | "Please switch yourself to max speed using your quality/speed tool, then tell me which setting is now active." | The model called the tool. The endpoint then read `max_speed` (1.36125), applied. |
| Image | "Make me a picture of a small red lighthouse on a rock, simple flat illustration. Use 512x512, 4 steps and seed 7." | The model called `generate_image` and the image appeared in the chat after 115 s. |

During the image request, the stack swapped the model out and back:

1. It saved the chat state into locked RAM.
2. It shut the engine down gracefully (2.8 s).
3. It generated the image (33 s).
4. It relaunched the engine with the chat state restored (4.5 s). The engine resumed at position 2016 instead of prefilling again.

The only setting changed in the web UI was the maximum response length (200 tokens), to keep the run short.

Screenshots: [chat](walkthrough/chat-desktop.png) ([phone](walkthrough/chat-phone.png)), [quality](walkthrough/quality-desktop.png) ([phone](walkthrough/quality-phone.png)), [image](walkthrough/image-desktop.png) ([phone](walkthrough/image-phone.png)).

## 5. Stop

Press Ctrl-C in the `start` window. The launcher asks the router to stop. The router closes the native engine through its graceful shutdown, which drains DirectStorage and frees GPU and host memory. Both engine processes in this run logged a graceful shutdown, and GPU memory returned to its idle level.

## Notes found during the walkthrough

- The walkthrough above ran with the earlier 4096-token default context, and one short chat plus the tool definitions used 47% of it. The default is now 262144 tokens. A follow-up short chat from the same clone (2e536c1) logged `context=262144`, context VRAM growing in 2048-token steps (1972 MiB mapped of 8859 MiB), an expert tier of 666 experts against the fixed full-context plan of 301, and a graceful shutdown.
- The web UI does not yet show the "saving / reloading" status during an image swap. The router log records it.
- Not walked through yet: video, two-machine mode, LoRA.
