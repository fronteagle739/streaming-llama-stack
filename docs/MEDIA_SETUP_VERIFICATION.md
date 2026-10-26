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
- Tool loops passed more than eight rounds in streaming and nonstreaming tests;
  there is no fixed tool-round limit. Malformed tool syntax is contained in an
  error card and is never executed as a guessed call.

These checks qualify the tested Windows setup and model files. They do not
qualify every checkpoint, GPU, video family, or native Mac build. A standalone
checkpoint still needs matching model components; a similarly named pipeline
with a different parameter size cannot substitute for them.
