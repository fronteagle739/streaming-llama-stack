# Limitations and roadmap

## Known limitations

- The Windows kernels target compute capability 12.0. Hardware beyond the measured RTX 5090 and M3 Ultra configurations needs separate evaluation.
- The 1M launcher admits 999,424 tokens. Published long-document recall measurements cover up to 237,703 prompt tokens, not a complete million-token prompt.
- YOCO is approximate. Image-containing prompts use full evaluation, and faster expert-routing settings can introduce code errors.
- Media families and LoRA are experimental. Language-model adapters require the native ds4 path; a media checkpoint needs matching encoders, VAEs and adapters.
- The first prompt includes producer startup. Initial model verification hashes the complete file. Neither cost should be reported as sustained decode speed.
- Paired operation requires matching model/renderer identities, compatible native state formats and sufficient memory on both hosts. A layer-by-layer distributed pipeline is separate from the supported RTX-prefill/Mac-decode session handoff.
- There is no verified public download URL for the exact measured Flash 1017 Q4 GGUF; see the README model TODO.

## Planned work

- Publish a reproducible source and conversion recipe for the measured GGUF.
- Extend quality and long-context measurements across more prompt types, context depths and devices.
- Reduce cold-start and DirectStorage setup overhead; improve VRAM use without increasing the host-RAM budget.
- Broaden media/adapter coverage and improve visible save/reload progress during GPU handoffs.
- Explore distributed layer execution independently of the existing paired session transport.

See [Performance](PERFORMANCE.md), [paired operation](TWO_MACHINE.md), and [Variables](VARIABLES.md) for current behavior.
