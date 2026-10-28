# Limitations and roadmap

## Limitations

- The RTX path is Windows-only today (DirectStorage), targeting compute capability 12.0 (RTX 50-series). Other GPUs are untested.
- The single-engine DeepSeek V4.1 path supports the [listed tensor layouts](DSV41_GGUF.md); split GGUFs require lossless assembly first.
- The faster quality/speed settings cost quality. Max speed introduced syntax
  errors in C++ tests; use Max quality or Balanced for code.
- YOCO is approximate and opt-in.
- The first prompt includes engine startup, and decode is slower until the expert
  cache warms up (about 100 tokens). The first check hashes the model once.
- GLM-5.3 Flash and Qwen3.8 Flash Next are not yet in the streaming engine.
- Media generation and LoRA are experimental.

## Planned

- Make resident follow-ups exact (the original resident decoder's observed
  layer-boundary divergence starts after attention, before routing). The current
  sweep-resident default is faster but still numerically different: measured
  follow-up KL0.0000235/0.000224/0.00250 at2K/8K/32K. The owner accepted this
  tradeoff; set `LLAMADECODE_APPEND_RESIDENT_SWEEP=0` for full-sweep follow-ups.
  Finer traces localize the original decoder difference to query/KV projection
  and RoPE; the sweep-resident alternative first differs at layer8's FFN output
  in the2K first-token diagnostic. See [E2c evidence](EXPERIMENT_E2C.md).

- Revisit the experimental host-RAM expert tier on machines with much more RAM
  (untested). It gave no gain on the 32 GB machine and is not in this release;
  [measured decode rates](PERFORMANCE.md#host-ram-expert-tier).
- Complete the CPU mode3/YOCO receiver for the delivered
  [large-RAM ds4-win untested example](../examples/dual-machine/dsv41-large-ram-untested/README.md).
- Faster cold start and DirectStorage setup; better VRAM use within the same RAM budget.
- Wider quality and long-context testing across prompt types and hardware.
- More media models and adapters.
