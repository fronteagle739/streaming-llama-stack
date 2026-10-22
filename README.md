# Streaming Llama Stack

A private, portable launcher and OpenAI-compatible chat endpoint with the llama.cpp chat UI, quality/speed control, and Studio image/video engines.

Development status: the expanded two-machine and release acceptance tests are in progress. The earlier checkpoint's media proofs do not qualify the new clean vendored tree. See docs/STATUS.md for actual verification.

The stack's own code and owner-authored Studio exports use MIT. Engine and embedded library licenses remain in their source directories; see NOTICE and each PROVENANCE file.

The intended six-step walkthrough is:

1. Clone this repository with `git clone --recurse-submodules` on both computers.
2. On Windows, run `./slstack.ps1 install --all` to create isolated media runtimes. Python 3.12, Visual Studio C++ tools, CMake/Ninja, and CUDA are prerequisites; model weights are supplied locally.
3. Edit example model/ComfyUI paths. Model weights are read in place and are never downloaded by the installer.
4. On the Mac, run `./bin/slstack mac --model <model.gguf>`. The receiver prints its endpoint and pairing instructions.
5. On Windows, run `./slstack.ps1 rtx --model <model.gguf> --comfyui-models <models> --mac <mac-endpoint>` with the printed pairing information.
6. Open the printed chat URL. Exercise chat, quality/speed, image/video, and a long prompt with a second turn.

Steps 4-6 are not yet qualified for this revision. Do not treat this draft as a completed acceptance transcript.

Commands currently implemented include `install`, `check`, `start` (local `rtx` alias), `mcp`, `train_lora`, `merge_lora`, `unmerge_lora`, and `run_prompts`. Native DSV4.1 dense-output LoRA attachment passed the real 1017 Jeff test, including held-out, rainbow, on/off, full-base-hash, clean-detach and deliberate-crash recovery checks; see [the recipe and card](examples/train_jeff_lora/README.md). The one-GPU Windows state handoff remains refused until its parity gate passes.

Media uses the existing Studio implementations in an isolated venv. The native weight-fetcher shared library must also be built. H3's optional `flash_attn2` backend requires a separately compatible flash-attn installation; explicit SDPA selection is supported by the reused engine. No silent backend substitution is performed.

The requested quality/speed points are Max quality 0, Balanced 0.35, Fast 0.65, Faster 0.9 and Max speed 1.36125. Use Max quality or Balanced for code and precise technical output; Faster and Max speed are intended for prose, drafts and casual chat. The current vendored setter still reflects the earlier range; the five-point/per-phase interface update is pending and must be reused from its source implementation before acceptance.

Owner-supplied 3K-token story/C++ measurements were respectively 5.8/5.0, 8.1/7.7, 10.3/9.7, 12.3/10.7 and 14.2/13.3 tok/s. Faster showed a code typo; Max speed had C++ syntax errors. These measurements describe the qualified model/hardware, not a general speed guarantee.

Permanent LoRA export always writes a new model. Unmerge selects the original verified base. Incomplete outputs and prior copies are preserved, and journal records use atomic writes, process creation times, and base SHA-256 checks. Only supported existing trainers are wrapped; unsupported families are refused.

Enable the local pre-commit gate with `git config core.hooksPath .githooks`, and run `python -B scripts/release_check.py` before committing or pushing. Audit-specific forbidden literals belong only in the ignored `release-secrets.local.json`; the scanner never prints their values. There are no stack GitHub workflows. The only submodule is the public upstream mlx-serve tag; private engine histories are excluded from the vendored source tree.

The current development branch starts at a new root commit with neutral authorship. An earlier published development branch remains private and untouched. Public release requires owner approval and will put the clean tree into a new public repository; this private repository is not certified for publication with all of its refs.


Register native models with `--models first.gguf,second.gguf`, or set `models` to objects with `id`, `path` and optional `lora` in your local config. `/v1/models` exposes those ids; the existing web UI picker and each request's `model` field select them per chat. An unloaded selection uses the normal owned unload/load path, with status in the UI. `--resident first,second` records a requested pair; it currently performs normal switching and does not claim simultaneous residency or fast switching. The pinned llama-server has its own `--models-preset`/`--models-dir` router, but it cannot directly host the native ds4 adapter, so this bridge reuses its UI API shape.
