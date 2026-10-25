# Streaming Llama Stack

A private, portable launcher and OpenAI-compatible chat endpoint with the llama.cpp chat UI, quality/speed control, and Studio image/video engines.

Development status: the expanded two-machine and release acceptance tests are in progress. The earlier checkpoint's media proofs do not qualify the new clean vendored tree. See docs/STATUS.md for actual verification.

See [Status and plan](docs/STATUS_AND_PLAN.md) for the current deployments, measurements, open decisions and remaining work.

The stack's own code and owner-authored Studio exports use MIT. Engine and embedded library licenses remain in their source directories; see NOTICE and each PROVENANCE file.

The intended six-step walkthrough is:

1. Clone this repository with `git clone --recurse-submodules` on both computers. The repository is private, so Git needs credentials with access to it.
2. On Windows, run `./slstack.ps1 install --all` to create isolated media runtimes and build native engines and the chat UI, then `./slstack.ps1 install --all --check` to verify the build. Prerequisites: Python 3.12 on `PATH`, Visual Studio C++ Build Tools, CMake and Ninja (the Visual Studio CMake tools provide both), the CUDA toolkit (verified with 13.2), and Node.js with npm (verified with Node 24) for the chat UI build. The CUDA targets are built for `sm_120a`, so the native engines need an RTX 50-series GPU with at least 30 GB VRAM. The installer needs network access for PyPI, the npm registry and the pinned DirectStorage NuGet package; it never downloads model weights. A full install took about 27 minutes on the qualification machine.
3. Set your model and ComfyUI paths, either with flags (`--model`, `--comfyui-models`) or in a local JSON file that overrides keys from `config.example.json` (for example `my.local.json`, which Git ignores), passed with `--config my.local.json`; then run `./slstack.ps1 --config my.local.json check`, which needs the model path. Model weights are read in place and are never downloaded. The default Anima image model also needs its component files in `image_support_files`: `anima_dir` (the Anima diffusers folder with transformer config, VAE and tokenizers) and `text_encoder` (the Qwen3 0.6B base text encoder).
4. On the Mac, run `./bin/slstack mac --model <model.gguf>`. The receiver prints its endpoint and pairing instructions.
5. On Windows, run `./slstack.ps1 rtx --model <model.gguf> --comfyui-models <models> --mac <mac-endpoint>` with the printed pairing information.
6. Open the printed chat URL. Exercise chat, quality/speed, image/video, and a long prompt with a second turn.

Without a Mac, `./slstack.ps1 start --config my.local.json --webui` (or `--model <model.gguf>`) serves the RTX alone; open the router URL it prints (default `http://127.0.0.1:8088`).

- **Default: single engine.** The router fronts the same single-engine server Studio uses (`slstack/combo_proxy.py` with `serving_mode=single_engine`): one streaming-llama producer process does the ULTRA prefill and decodes with ds4-win's own kernels (`ds4-decode.dll`, built by `install --all` beside `build/llama/bin/llama-dsv41-prefill.exe`). The producer starts with the first prompt, keeps the conversation resident, and appends only the new tokens on the next turn. It takes the GPU hold while it is resident and gives it back after `single_engine_idle_seconds` without use (default: the lease length less one minute). The quality/speed tool and the web UI statistics work as on the native path. `producer_profile` (default `ultra`) and `producer_host_budget_gib` (default 6.25) set the producer; `host_budget_gib` must leave 0.75 GiB above that for the controller. LoRA adapters need the native path.
- **Native path: `start --serving native`** (or `"serving": "native"` in your config) runs the ds4-win server instead, as in earlier revisions. Its default context is 262144 tokens; context VRAM grows in 2048-token steps and the expert cache tier shrinks to match. To change its arguments, copy the whole `ds4_command` list from `config.example.json` into your local config; a local key replaces the example's key rather than merging.
- **Images and video with the single engine:** before a media job the producer exits gracefully and frees the GPU. Afterwards it is relaunched on the same conversation tokens, which means **a full prefill of the conversation again** (the producer has no state export yet). The next turn then appends only its new tokens. The native path instead saves and restores its exact chat state without a prefill.
- **Stop:** press Ctrl-C in the `start` window. The router asks the engine to shut down gracefully: the producer (or the native server) exits, drains DirectStorage and frees GPU and host memory.

### DSV4.1 single engine: 262K and 1M launch scripts

Two scripts start the single engine through `slstack start` with the gated configurations. Each one writes `runtime/dsv41-single-<preset>.local.json` (your local config plus the settings below), starts the stack detached, waits until the router answers and prints the URL, log, health check and shutdown command.

| | `scripts/start-dsv41-single-262k.ps1` | `scripts/start-dsv41-single-1m.ps1` |
|---|---|---|
| Context | 262,144 tokens | 999,424 tokens (the largest growable bf16 context the producer admits) |
| Context cache | f32, growable (full L2 expert tier) | bf16, growable |

Both use DirectStorage read parts 8, a 512 MiB VRAM reserve beside the L2 expert tier, the default quality/speed setting, single-sweep prefill off (49,152-token sweeps; `producer_host_max_gib` is removed if your config sets it), `producer_host_reserve_gib` 1, and Engram early load `auto` with no Engram row cache (`LLAMADECODE_ENGRAM_CACHE_MB=0`). The settings live in `scripts/dsv41_single.py`.

- **Turn after an append (keep-open, default on):** after the prefill sweep of an appended turn the producer keeps its ds4 decoder open and re-registers only the resident weights that moved, instead of closing and reopening the decoder. With the 1M script the decode of the turn after a long append starts in about 4 s instead of about 14 s; at 262K the gain is small. The output is byte-identical to the close/reopen path. `LLAMADECODE_DS4X_KEEP_OPEN=0` restores close/reopen. The producer also verifies the conversation's owner state incrementally (only the appended rows are re-hashed); the digests on the wire are unchanged.

```powershell
./slstack.ps1 install --all                 # build once (producer, ds4-decode.dll, chat UI)
./slstack.ps1 install --all --check
# my.local.json holds at least {"model": "<path>/DeepSeek-V4.1-Flash-1017-Q4-v41.gguf"} (see step 3)
powershell -File scripts/start-dsv41-single-262k.ps1 -WebUI          # endpoint and web UI
powershell -File scripts/start-dsv41-single-1m.ps1                   # endpoint only, 1M context
powershell -File scripts/start-dsv41-single-262k.ps1 -DryRun         # print the resolved settings; start nothing
powershell -File scripts/start-dsv41-single-262k.ps1 -Stop           # graceful shutdown
```

- `-Config <file>` picks a local config (default: `my.local.json` in the clone, if present); `-Model <gguf>` overrides its `model`. Start one preset at a time: both use the same router port.
- **Web UI** (`-WebUI`): open the printed router URL (default `http://127.0.0.1:8088`). The stack adds its own tools to every chat, so the model can call `set_quality_speed` / `get_quality_speed` itself (ask it, for example, to switch to max speed). MCP clients reach the same tools at `http://127.0.0.1:8088/mcp`. Without `-WebUI` the same router serves the OpenAI-compatible endpoint `/v1/chat/completions`.
- **Prefill progress:** streamed chats show the web UI's prompt-processing bar while the producer prefills an appended turn (`prompt_progress`, from the producer's per-layer sweep lines; OpenAI clients also get the `prefill_progress` chunk). The first prompt of a conversation also loads the producer; that prefill shows the waiting indicator only. The producer refuses to start without a full SHA-256 receipt for the model (`<runtime>/single-engine/state/original-model.json`, reused while the file's size, volume, file id and mtime are unchanged). `install` and `check` (and therefore `start`) compute it once, with percent, GB and rate on the console (about 6 minutes for the 483 GB GGUF). If it is still missing when the first prompt arrives, that prompt waits for the hash; the progress appears in the router's model status (`/health` `model_status`, `/slstack/status`, and the `Model status:` log lines). Tokens per second appear in the chat statistics when the reply finishes.
- **Health:** `Invoke-RestMethod http://127.0.0.1:8088/health` (router) and `/v1/models`.
- **Shutdown:** `-Stop` posts `{}` to `/v1/stack/shutdown`. The router closes the single engine, whose producer exits through its own graceful shutdown (DirectStorage drained, VRAM and the GPU hold released); nothing is killed. The detached `start` process then exits.
- The log is `runtime/logs/dsv41-single-<preset>-<time>.log`.

The [single-RTX walkthrough](examples/WALKTHROUGH.md) records a fresh-clone install, chat, model-made quality/speed change and image swap with screenshots; those receipts are from the native path. Its single-engine section lists the launch-script steps; the GPU run of those steps is pending.

The two-machine steps 4-5 are not yet qualified for this revision; step 6 is qualified on the single-RTX native path for chat, the model-made quality/speed change and an image with state swap (see the walkthrough), and the same walkthrough on the single-engine default is pending; video and the long prompt with a second turn are not yet walked through.

Commands currently implemented include `install`, `check`, `start` (local `rtx` alias), `mcp`, `train_lora`, `merge_lora`, `unmerge_lora`, and `run_prompts`. Native DSV4.1 dense-output LoRA attachment passed the real 1017 Jeff test, including held-out, rainbow, full-base-hash, clean-detach and deliberate-crash recovery checks; see [the recipe and card](examples/train_jeff_lora/README.md). The LoRA-off comparison was skipped per owner instruction. The one-GPU Windows state handoff remains refused until its parity gate passes.

Media uses the existing Studio implementations in an isolated venv. The native weight-fetcher shared library must also be built. H3's optional `flash_attn2` backend requires a separately compatible flash-attn installation; explicit SDPA selection is supported by the reused engine. No silent backend substitution is performed.

The [user guide](USER_GUIDE.md) documents the environment-to-flag table and literal engine defaults. `--mode accurate|fast|ultra|hyper` selects the delivered prefill arithmetic profile; media uses `--image-mode streaming|resident` and `--video-mode streaming|resident`. The launcher supplies offline caches and memory budgets automatically, and refuses overrides that conflict with its GPU identity or residency requirements. Profile definitions are implemented; the two-machine runtime qualification remains pending.

See the measured [native/producer parity](examples/engine_parity/README.md) and [environment behaviour samples](examples/engine_environment/README.md) for the checks completed so far.

The requested quality/speed points are Max quality 0, Balanced 0.35, Fast 0.65, Faster 0.9 and Max speed 1.36125. Use Max quality or Balanced for code and precise technical output; Faster and Max speed are intended for prose, drafts and casual chat. The vendored provider accepts these five names and independent `reasoning`, `tool_call`, `answer`, and `code` settings. Partial updates preserve other phases; see the [tool examples](USER_GUIDE.md#quality-and-speed). The native rebuild and compiled request transport checks pass. [Live endpoint evidence](examples/quality_speed/RESULTS.json) proves all five settings, changed expert routing and an independent queued Code setting. [Actual webui chats](examples/quality_speed/UI-RESULTS.json) also pass: the model selected Max quality and Max speed through the tool, with 5.90 and 9.99 tokens/s displayed for 256-token answers. Desktop and phone screenshots are saved beside that receipt; these are sequential observations, not a controlled benchmark.

Owner-supplied 3K-token story/C++ measurements were respectively 5.8/5.0, 8.1/7.7, 10.3/9.7, 12.3/10.7 and 14.2/13.3 tok/s. Faster showed a code typo; Max speed had C++ syntax errors. These measurements describe the qualified model/hardware, not a general speed guarantee.

Permanent LoRA export always writes a new model. Unmerge selects the original verified base. Incomplete outputs and prior copies are preserved, and journal records use atomic writes, process creation times, and base SHA-256 checks. Only supported existing trainers are wrapped; unsupported families are refused.

Enable the local pre-commit gate with `git config core.hooksPath .githooks`, and run `python -B scripts/release_check.py` before committing or pushing. Audit-specific forbidden literals belong only in the ignored `release-secrets.local.json`; the scanner never prints their values. There are no stack GitHub workflows. The only submodule is the public upstream mlx-serve tag; private engine histories are excluded from the vendored source tree.

The current development branch starts at a new root commit with neutral authorship. An earlier published development branch remains private and untouched. Public release requires owner approval and will put the clean tree into a new public repository; this private repository is not certified for publication with all of its refs.


Register native models with `--models first.gguf,second.gguf`, or set `models` to objects with `id`, `path` and optional `lora` in your local config. `/v1/models` exposes those ids; the existing web UI picker and each request's `model` field select them per chat. An unloaded selection uses the normal owned unload/load path, with status in the UI. `--resident first,second` records a requested pair; it currently performs normal switching and does not claim simultaneous residency or fast switching. The pinned llama-server has its own `--models-preset`/`--models-dir` router, but it cannot directly host the native ds4 adapter, so this bridge reuses its UI API shape.
