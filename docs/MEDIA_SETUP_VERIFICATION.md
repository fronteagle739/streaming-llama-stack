# Experimental media setup

`./slstack.ps1 install --all` installs pinned media environments, checks pipeline imports, supplies hash-verified configuration/tokenizer assets and FFmpeg/FFprobe, and builds the native engines and WebUI. `install --all --check` checks readiness. Model weights and any required text encoders/VAEs are supplied locally.

## Model families

The general media worker and Qwen-Image 2.1 worker use separate environments and dependency locks. Krea's bundled pipeline uses supplied tokenizer/config assets and discovered local weights. Complete pipeline folders are model choices; individual shards are not.

Existing Windows examples include Anima, Krea turbo, Qwen-Image 2.1, Z-Image, Flux2 Klein and Kroma with explicit compatible adapters. A matching family name does not guarantee matching tensor shapes or components. [Model discovery and adapters](STACK_MEDIA_AND_STATE.md)

Krea resident tensors and VAE use DirectStorage on Windows CUDA, with tensor-coverage checks and owned copies before stream-buffer release. Select Streamed or Resident per model in Settings → Stack. An adapter is applied only when explicitly selected for that generation.

## H3 video

H3 defaults to PyTorch SDPA and a labelled audiovisual caption. If omitted, frame count is 121 at 24 fps. The tool uses the chat model's generation prompt without starting a separate prompt-enhancement model. Optional attention packages can be selected through `h3_attention_backend` when installed.

Known fused Comfy exports receive their corresponding QKV layout; model configuration can declare `qkv_layout` as `sectioned` or `interleaved`. The loader indexes immediate transformer shards and excludes nested composition caches. Complete single-file transformers need no synthetic manifest.

Recorded 121-frame examples produced 5.175-second H.264/AAC MP4 clips, including explicit three-step adapters and an eight-step request. This is model-specific behavior, rather than coverage of every checkpoint or platform.

## Chat interaction

Tool cards show arguments, progress, errors and results. Completed images enter vision context, and video plays inline. There is no fixed tool-round limit; malformed tool syntax is reported instead of executed as a guessed call. Chat preserves the conversation across media GPU handoffs using its configured state/checkpoint behavior.

See the [walkthrough](../examples/WALKTHROUGH.md) and [media/state reference](STACK_MEDIA_AND_STATE.md).
