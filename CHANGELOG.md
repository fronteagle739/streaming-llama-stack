# Changelog

## 2026-11-03 — portable QFN pack and exact 100K benchmark

- Add Streaming Llama Pack (`.slpack`) reader/writer support, a portable QFN
  importer, pinned NVIDIA and auxiliary recipes, source/section identities,
  canonical provenance, inverse checks and explicit memory supervision. Preserve
  dense quantization provenance, MTP, vision, profiles, sampler defaults and all
  runtime controls. The complete raw-source-plus-pinned-auxiliary proof passes;
  canonical output is 131,730,218,740 bytes with SHA256
  `195df4842abf2dd335ad706be6518611b2a433e806d98c4edcc8fe5e2b309a92`.
  The pack includes quantized dense/PLE tensors and requires the auxiliary input;
  it is not a lossless conversion of the NVIDIA download alone.
- Pin Python 3.12.10, NumPy 2.5.0, regex 2026.5.9 and psutil 7.2.2 in
  [requirements-pack.txt](requirements-pack.txt).
- Add the exact 100K public-text fixture builder and guarded stdio benchmark.
  Preserve full-vocabulary calibration, sampled natural stops and all samplers.
- Set QFN Max speed to bias 0.68 while retaining numeric intermediate values.
- Document QFN and DeepSeek V4.1 source links, pinned revisions, pack/run commands
  and verification in [the recipes guide](docs/SLPACK_RECIPES.md).
- GLM is not supported yet.

### Performance and quality

Hardware: RTX 5090 with 32 GB VRAM, 32 GB system RAM and fast local
DirectStorage NVMe storage. Rates are tokens/s. Bias applies to decode;
prefill uses unbiased routing. KL is relative to unbiased routing and top-1
is agreement. A dash means no matching measurement is available.

#### QFN

| Prompt length | Bias | Prefill | Decode | KL | Top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0 | 4,440 | 25.25 | 0 | 100.0% |
| 100,000 | 0.35 | 4,440 | 46.25 | 0.0372 | 92.6% |
| 100,000 | 0.68 | 4,150 | 50.12 | 0.137 | 88.3% |

#### DeepSeek V4.1

| Prompt length | Bias | Prefill | Decode | KL | Top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0 | 2,614 | — | — | — |
| 100,000 | 0.35 | — | — | — | — |
| 100,000 | 1.36125 | — | — | — | — |

QFN timing uses 100 discarded warmup tokens and 128 measured tokens per mode
on one continuing sequence, MTP off and eight read parts. Full-vocabulary
quality comparisons use the same forced continuation against bias zero.
QFN named modes use 0 / 0 / 0.68; numeric intermediate values, including 0.35,
remain available. DeepSeek modes use 0 / 0.35 / 1.36125.

Reproduce QFN's exact fixture and timing protocol with the
[100K guide](docs/QFN_100K_REPRO.md), and use the
[calibration workflow](examples/single-machine/CALIBRATION.md) for KL/top-1.
The portable timing wrapper has exact-fixture and CPU protocol checks;
native wrapper qualification remains pending. The short DeepSeek 100K attempt did not complete every requested metric;
unavailable decode or full-vocabulary quality measurements remain dashes.

