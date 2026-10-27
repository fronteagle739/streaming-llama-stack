# Media setup verification

`./slstack.ps1 install --all` installs both pinned media environments, verifies
pipeline imports, installs hash-verified configuration/tokenizer assets, and
builds the native engines and WebUI. Follow the prerequisites in the README.
Use `install --all --check` to verify readiness. Setup does not download model
weights: select their folders in Stack settings, including any required text
encoder and VAE weights. Complete pipeline folders are recognized as models;
their individual shards are not separate model choices.

The general media worker and Qwen-Image 2.1 worker have separate virtual
environments and dependency locks. No Studio checkout, modified site-packages
or inherited model cache is needed. Setup supplies Krea's tokenizer, model
configuration and VAE configuration; discovered local files supply the weights.
Windows setup also installs hash-verified FFmpeg and FFprobe. H3 defaults to
built-in SDPA and uses Studio's labelled audiovisual caption format, with 121
frames at 24 fps when the frame count is omitted. It never starts an extra
Studio LLM for prompt enhancement.

Verification on 2026-10-26:

- Both environments installed into an empty runtime and passed import checks.
- Production environments passed the same setup checks.
- Actual image generation passed for Kroma with an explicitly selected turbo
  LoRA, Krea turbo, Qwen-Image 2.1, a Z-Image checkpoint, and a complete Flux2
  Klein pipeline.
- Krea resident tensors and VAE use DirectStorage on Windows CUDA. The loader
  checks tensor coverage and makes owned copies before releasing stream buffers.
- Browser checks passed for collapsed tool cards, expanding results/errors,
  streaming updates without duplicate cards, and persistence after reload.
- Model loading preferences are saved per model. One Streamed / Resident toggle
  controls both image and video uses; unrelated models keep their choices.
  Anima with an explicit turbo LoRA also passed a real resident-mode generation.
- Tool loops passed more than eight rounds in streaming and nonstreaming tests;
  there is no fixed tool-round limit. Malformed tool syntax is contained in an
  error card and is never executed as a guessed call.

These checks qualify the tested Windows setup and model files. They do not
qualify every checkpoint, GPU, video family, or native Mac build. A standalone
checkpoint still needs matching model components; a similarly named pipeline
with a different parameter size cannot substitute for them.

Additional H3 verification on 2026-10-27:

- Native streaming H3 generation produced H.264/AAC MP4 output. Correct QKV
  layout profiles are applied to known fused Comfy exports; a model config can
  explicitly declare `qkv_layout` as `sectioned` or `interleaved`.
- H3 reads only immediate transformer shards, excluding nested composition
  caches. Complete single-file transformers need no synthetic shard manifest.
- A real model-driven WebUI chat called the video tool once, showed a completed
  collapsible tool card, played and sought the inline video, and continued with
  its final response. Browser console checks were clean.
- A 121-frame request produced a 5.175-second clip depicting the requested cup
  scene, both at eight steps and with an explicitly attached three-step TaoMate
  LoRA. MP4 inspection confirmed H.264 video and AAC audio.
- Output video routes support byte ranges and stream bounded chunks. When a
  conversation checkpoint is unavailable, completed media is published before
  replaying its conversation on the next model request.
