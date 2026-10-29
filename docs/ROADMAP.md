# Limitations and roadmap

## Limitations

- The RTX path is Windows-only today (DirectStorage), targeting compute capability 12.0 (RTX 50-series). Other GPUs are untested.
- The single-engine DeepSeek V4.1 path supports the [listed tensor layouts](DSV41_GGUF.md); split GGUFs require lossless assembly first.
- The faster quality/speed settings cost quality. Max speed introduced syntax
  errors in C++ tests; use Max quality or Balanced for code.
- Approximate YOCO140 is opt-in. Exact cone keep2541 is the single-machine
  and paired ULTRA default, with negotiated state v2 for paired use.
- The measured FP4 file was requantised from Q4; native-precision FP4 conversion from the official weights is coming, with the same layout and expected speed.
- Dense FP8 remains experimental. The shipping model remains Q4_K/Q8_0.
- TileLang attention is numerically different from native attention. Retrieval
  matches on the tested fixture, but later continuations can diverge; the native
  off toggle remains available. [Evidence](../examples/kernels2/PERFORMANCE.md).
- The first prompt includes engine startup, and decode is slower until the expert
  cache warms up (about 100 tokens). The first check hashes the model once.
- GLM-5.3 Flash and Qwen3.8 Flash Next are not yet in the streaming engine.
- Media generation and LoRA are experimental.

## Planned

- Reach **10,000 prompt-only tok/s at 100K tokens**. FP4 (tensor-core-native
  layout) measures 3,088.92 tok/s in 32.373785 s, with 35.406 s TTFT;
  the target remains unmet. [Measured configurations](PERFORMANCE.md#prefill).
- Remove host/device synchronization from the tile loop while preserving GPU
  checks, transaction boundaries, streamed-weight and carry lifetimes.
- Complete the fused layer path: separate routing/attention/FFN lifetimes,
  retain larger expert batches and fuse remaining projection and small-op work.
  Exact HC collapse, QAT batching and CONCAT improvements are already present.
- Build binding-independent layer/sweep graph templates. Actual streamed CUDA
  graph launches are implemented and opt-in; changing layer/tile bindings still
  require CPU dispatch and capture/update work. [Current support](../examples/other22/ROUND2_RESULTS.md).
- Finish verified official-safetensors download and native FP4/FP8 conversion,
  then qualify source-native full-model logits, recall, decode and long context.
  Keep the lossy requant proxies separate from those results.
- Make TileLang attention exact against native arithmetic; retain the existing
  native path and the current default's published numerical caveats.

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

### Packed KV integration after E17

The E17 sampled codec is lossless on the tested existing QAT KV/index values.
Native packed owner allocation, append/publication and checkpoint/wire handling,
and fused sparse attention/index loads remain future work. Qualify full-model
long-context recall and KL/top-1 against an actual8-bit cache; then measure
VRAM peaks, larger sweep/context admission and end-to-end speed. The890B/token
layout formula and component unpack timings alone do not establish those gains.
