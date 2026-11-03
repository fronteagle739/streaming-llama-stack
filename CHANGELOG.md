# Changelog

## 2026-11-03 — portable QFN pack and exact 100K benchmark

- Add Streaming Llama Pack (`.slpack`) reader/writer support, the portable QFN
  importer, pinned NVIDIA and auxiliary recipes, full source/section identities,
  canonical provenance, inverse checks and explicit memory supervision. Preserve
  dense quantization provenance, MTP, vision, profiles, sampler defaults and all
  supported runtime controls. The complete pack is not described as raw-source
  lossless; a fresh canonical output size/hash is still pending full raw proof.
- Add the audited exact 100K public-text fixture reconstruction and portable
  guarded stdio timing wrapper. The 400,000-byte token fixture matches exactly.
  Three CPU-only protocol cases pass; native wrapper performance qualification
  is pending. Keep full-vocabulary calibration and sampled natural-stop tests.
- Select QFN Max speed bias 0.68 from historical 100K evidence: 50.12 tok/s,
  mean KL 0.13732 and top-1 88.28125%. Those receipts predate the installed
  build. No new 0.68 inference was run. Retain numeric 0.35 and its prior data.
- Accept the final quiet installed QFN control A2 at4,439.86 /25.2482 /
  46.2537 tok/s (prefill /quality /bias-0.35), exact100K with100 warmup and128
  measured tokens per mode, same sequence, MTP off/eight read parts, recall
  and guard PASS. Keep the historical4,573.37 /26.7441 /46.7450 checkpoint
  separate; supersede the earlier installed4,326.40 /25.6518 /42.8092 row.
- Close the regression investigation by owner decision: engine/DLL changes
  cleared, most earlier difference accepted as concurrent load. Control
  range/mean spreads were1.3311% /4.4259% /1.3186%. The residual roughly3%
  prefill read-path difference remains unexplained and roadmap-only, below
  the owner's action threshold. Workspace32 and fresh-placement performance
  remain unproven; no further workspace/storage tests for this delivery.
  Current installed-build KL/top-1 remain queued rather than borrowed from
  historical metrics.
- Document both models' immutable Hugging Face links, conversion/launch commands,
  settings and evidence boundaries. DSV4.1's historical 100K decode is
  7.39 /24.11 tok/s with a 100-discarded +899-timed protocol. Its separate
  short-prompt KL 0.479/top-1 84.1% result is not presented as 100K quality.
  Current native DSV4.1 prefill/KL/top-1 and exact historical benchmark command
  remain queued; native main/auxiliary output identities are not yet measured.
- Keep GLM's separate source-format handoff distinct from executable support in
  this review tree; full native build/pack/inverse/runtime proof remains pending.
  No public push or complete three-model reproducibility claim is made.
